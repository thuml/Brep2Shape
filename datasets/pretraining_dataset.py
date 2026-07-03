import json
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import dgl
import numpy as np
import torch
from dgl.data.utils import load_graphs
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from datasets.common import (
    convert_feature_tensors_to_float32,
    get_dataset_logger,
    pad_or_sample,
    sample_identifier,
)


VALID_SPLITS = {"train", "val", "test"}
REQUIRED_ITEM_KEYS = {"face", "topo", "graph", "line_graph"}
MAX_FACE_PRIMITIVES = 100
MAX_EDGE_PRIMITIVES = 50
MAX_ADJACENT_FACES = 10
MAX_WIRES_PER_FACE = 30
MAX_EDGES_PER_WIRE = 10
NORMALIZATION_EPS = 1e-7


def _to_float_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.float32)
    return torch.from_numpy(np.asarray(value, dtype=np.float32))


class PretrainingDataset(Dataset):
    """Load processed B-rep geometry for Brep2Shape self-supervised training."""

    def __init__(
        self,
        dataset_dir: str,
        split: str = "train",
        center_and_scale: bool = False,
        random_rotate: bool = False,
        lazy_load: bool = True,
        num_workers_loading: int = 8,
        num_uv_samples: int = 3,
        log_dir: str | pathlib.Path | None = "logs",
    ):
        if split not in VALID_SPLITS:
            raise ValueError(f"Unknown split {split!r}; expected one of {sorted(VALID_SPLITS)}")
        if num_uv_samples <= 0:
            raise ValueError("num_uv_samples must be positive")
        if num_workers_loading < 0:
            raise ValueError("num_workers_loading must be non-negative")

        self.logger = get_dataset_logger(log_dir)
        self.log_dir = pathlib.Path(log_dir) if log_dir is not None else None
        self.path = pathlib.Path(dataset_dir)
        self.lazy_load = lazy_load
        self.num_uv_samples = num_uv_samples
        self.random_rotate = random_rotate

        if center_and_scale:
            self.logger.warning(
                "center_and_scale is deprecated: pretraining samples are always normalized"
            )
        if random_rotate:
            self.logger.warning(
                "random_rotate is not implemented for PretrainingDataset and will be ignored"
            )

        self.file_list = self._load_split(split)

        if lazy_load:
            self.logger.info("Using lazy loading mode for %s data", split)
            self.data = None
            self.logger.info("Registered %d files", len(self.file_list))
        else:
            self.logger.info("Loading %s data with %d workers", split, num_workers_loading)
            self.load_samples(self.file_list, num_workers=num_workers_loading)
            self.logger.info("Done loading %d files", len(self.data))

    def _load_split(self, split: str) -> list[dict[str, Any]]:
        split_path = self.path / "datasplit.json"
        if not split_path.is_file():
            raise FileNotFoundError(f"Dataset split file does not exist: {split_path}")

        with split_path.open("r", encoding="utf-8") as f:
            split_data = json.load(f)

        if split not in split_data:
            raise KeyError(f"Split {split!r} is missing from {split_path}")
        if not isinstance(split_data[split], list):
            raise TypeError(f"Split {split!r} in {split_path} must be a list")

        items = []
        for index, raw_item in enumerate(split_data[split]):
            if not isinstance(raw_item, dict):
                raise TypeError(f"Item {index} in split {split!r} must be a JSON object")
            missing = REQUIRED_ITEM_KEYS.difference(raw_item)
            if missing:
                raise KeyError(
                    f"Item {index} in split {split!r} is missing keys: {sorted(missing)}"
                )
            item = dict(raw_item)
            for key in REQUIRED_ITEM_KEYS:
                path = pathlib.Path(item[key])
                item[key] = str(path if path.is_absolute() else self.path / path)
            items.append(item)
        return items

    def _process_single_item(self, item):
        try:
            sample, error = self.load_one_sample(item)
            if sample is None or error != "success":
                return None, error
            sample = self.normalize(sample)
            sample = self.padding(sample)
            sample = self.convert_to_float32(sample)
            return sample, "success"
        except Exception as exc:
            identifier = sample_identifier(item)
            self.logger.exception("Failed to process sample %s", identifier)
            return None, f"{type(exc).__name__}: {exc}"

    def load_samples(self, items, num_workers: int = 8):
        """Eagerly load and validate a sequence of split items."""
        self.data = []
        invalid_samples = []

        if num_workers > 0:
            self.logger.info("Using %d workers for parallel loading", num_workers)
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                results = executor.map(self._process_single_item, items)
                for idx, (item, result) in enumerate(
                    zip(items, tqdm(results, total=len(items), desc="Loading samples"))
                ):
                    sample, error = result
                    if sample is not None and error == "success":
                        self.data.append(sample)
                    else:
                        invalid_samples.append((idx, item, error))

        else:
            for idx, item in enumerate(tqdm(items, desc="Loading samples")):
                sample, error = self._process_single_item(item)
                if sample is not None and error == "success":
                    self.data.append(sample)
                else:
                    invalid_samples.append((idx, item, error))

        current_time = time.strftime("%Y%m%d%H%M%S")
        if self.data:
            self.logger.debug("Loaded sample keys: %s", sorted(self.data[0]))
            self.logger.info(
                "Successfully loaded %d samples at %s", len(self.data), current_time
            )

        if invalid_samples:
            self.logger.warning(
                "%d samples failed to load at %s", len(invalid_samples), current_time
            )
            if self.log_dir is not None:
                error_file = self.log_dir / f"invalid_samples_{current_time}.json"
                self.log_dir.mkdir(parents=True, exist_ok=True)
                with error_file.open("w", encoding="utf-8") as f:
                    json.dump(
                        [
                            {
                                "index": idx,
                                "sample": sample_identifier(item),
                                "error": error,
                            }
                            for idx, item, error in invalid_samples
                        ],
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )
                self.logger.warning("Invalid samples saved to %s", error_file)

        if not self.data:
            raise RuntimeError("No valid samples could be loaded from the dataset split")

    def convert_to_float32(self, data):
        """Convert floating-point model features and targets to float32."""
        return convert_feature_tensors_to_float32(
            data,
            {
                "face",
                "edge",
                "tri_normal",
                "uv_face_points",
                "uv_edge_points",
            },
        )

    def padding(
        self,
        data,
        max_facet_len: int = MAX_FACE_PRIMITIVES,
        max_arc_len: int = MAX_EDGE_PRIMITIVES,
        padding_mode: str = "zero",
    ):
        """Pad or subsample the primitive sequences of every face and edge."""
        if padding_mode not in {"zero", "circular"}:
            raise ValueError("padding_mode must be either 'zero' or 'circular'")
        if max_facet_len <= 0 or max_arc_len <= 0:
            raise ValueError("Padding lengths must be positive")

        faces = data["face"]
        in_masks = data["face_vis_mask"]
        tri_normals = data["tri_normal"]
        edges = data["edge"]
        if not faces or not edges:
            raise ValueError("A B-rep sample must contain at least one face and one edge")
        if not (len(faces) == len(in_masks) == len(tri_normals)):
            raise ValueError("Face features, visibility masks, and normals must align")

        faces_padded = []
        tri_normals_padded = []
        in_masks_padded = []
        face_padding_masks = []
        generator = torch.Generator()

        for nodes, tri_normal, in_mask in zip(faces, tri_normals, in_masks):
            nodes = torch.as_tensor(nodes)
            tri_normal = torch.as_tensor(tri_normal)
            in_mask = torch.as_tensor(in_mask)
            if nodes.shape[0] == 0:
                raise ValueError("Faces with zero Bézier primitives are not supported")
            if not (nodes.shape[0] == tri_normal.shape[0] == in_mask.shape[0]):
                raise ValueError("Per-face primitive features have inconsistent lengths")

            if nodes.shape[0] > max_facet_len:
                indices = torch.randperm(nodes.shape[0], generator=generator)[:max_facet_len]
                nodes = nodes[indices]
                in_mask = in_mask[indices]
                tri_normal = tri_normal[indices]

            num_nodes = nodes.shape[0]
            if num_nodes == max_facet_len:
                nodes_padded = nodes
                in_mask_padded = in_mask
                tri_normal_padded = tri_normal
            else:
                if padding_mode == "zero":
                    nodes_padded = nodes.new_zeros((max_facet_len, *nodes.shape[1:]))
                    tri_normal_padded = tri_normal.new_zeros(
                        (max_facet_len, *tri_normal.shape[1:])
                    )
                    in_mask_padded = in_mask.new_zeros(max_facet_len)
                    nodes_padded[:num_nodes] = nodes
                    tri_normal_padded[:num_nodes] = tri_normal
                    in_mask_padded[:num_nodes] = in_mask
                else:
                    repeats = (max_facet_len + num_nodes - 1) // num_nodes
                    nodes_padded = torch.cat([nodes] * repeats, dim=0)[:max_facet_len]
                    tri_normal_padded = torch.cat([tri_normal] * repeats, dim=0)[
                        :max_facet_len
                    ]
                    in_mask_padded = torch.cat([in_mask] * repeats, dim=0)[
                        :max_facet_len
                    ]

            padding_mask = torch.zeros(max_facet_len, dtype=torch.bool)
            padding_mask[:num_nodes] = True

            faces_padded.append(nodes_padded)
            in_masks_padded.append(in_mask_padded)
            face_padding_masks.append(padding_mask)
            tri_normals_padded.append(tri_normal_padded)

        edges_padded = []
        edge_padding_masks = []
        for edge in edges:
            edge = torch.as_tensor(edge)
            if edge.shape[0] == 0:
                raise ValueError("Edges with zero Bézier primitives are not supported")
            if edge.shape[0] > max_arc_len:
                indices = torch.randperm(edge.shape[0], generator=generator)[:max_arc_len]
                edge = edge[indices]

            num_primitives = edge.shape[0]
            edge_padded = edge.new_zeros((max_arc_len, *edge.shape[1:]))
            padding_mask = torch.zeros(max_arc_len, dtype=torch.bool)
            edge_padded[:num_primitives] = edge
            padding_mask[:num_primitives] = True
            edges_padded.append(edge_padded)
            edge_padding_masks.append(padding_mask)

        data["face"] = torch.stack(faces_padded)
        data["face_vis_mask"] = torch.stack(in_masks_padded)
        data["face_padding_mask"] = torch.stack(face_padding_masks)
        data["tri_normal"] = torch.stack(tri_normals_padded)
        data["edge"] = torch.stack(edges_padded)
        data["edge_padding_mask"] = torch.stack(edge_padding_masks)

        return data

    def normalize(self, data):
        """Normalize model inputs and targets, then discard temporary points."""
        points = torch.as_tensor(data["points"])
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError(
                "points must have shape [num_faces, num_points, 3], "
                f"got {points.shape}"
            )
        if not torch.isfinite(points).all():
            raise ValueError("points contains NaN or infinite values")

        center = points.mean(dim=(0, 1))
        centered_points = points - center
        scale = centered_points.abs().amax().clamp_min(NORMALIZATION_EPS)

        uv_face_points = torch.as_tensor(data["uv_face_points"])
        uv_edge_points = torch.as_tensor(data["uv_edge_points"])
        data["uv_face_points"] = (uv_face_points - center) / scale
        data["uv_edge_points"] = (uv_edge_points - center) / scale

        edges = [torch.as_tensor(edge).clone() for edge in data["edge"]]
        for edge in edges:
            edge[..., :3] = (edge[..., :3] - center) / scale
        data["edge"] = edges
        data.pop("points", None)

        return data

    def __len__(self):
        return len(self.file_list) if self.lazy_load else len(self.data)

    def __getitem__(self, index):
        if self.lazy_load:
            for offset in range(len(self.file_list)):
                item = self.file_list[(index + offset) % len(self.file_list)]
                sample, error = self._process_single_item(item)
                if sample is not None and error == "success":
                    return sample
                self.logger.warning(
                    "Failed to load sample %s: %s", sample_identifier(item), error
                )
            raise RuntimeError("No valid samples could be loaded from the dataset split")
        return self.data[index]

    @staticmethod
    def _offset_padded_indices(
        indices: torch.Tensor,
        lengths: torch.Tensor,
        offset: int,
    ) -> torch.Tensor:
        """Offset only valid entries of a padded index tensor."""
        shifted = indices.clone()
        valid = torch.arange(indices.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
        shifted[valid] += offset
        return shifted

    def _collate(self, batch):
        if not batch:
            raise ValueError("Cannot collate an empty batch")

        graphs = [sample["graph"] for sample in batch]
        line_graphs = [sample["line_graph"] for sample in batch]
        graph_file_paths = [sample["graph_file_path"] for sample in batch]

        face_counts = torch.tensor(
            [sample["face"].shape[0] for sample in batch], dtype=torch.long
        )
        edge_counts = torch.tensor(
            [sample["edge"].shape[0] for sample in batch], dtype=torch.long
        )
        wire_counts = torch.tensor(
            [sample["edge_index"].shape[0] for sample in batch], dtype=torch.long
        )
        face_offsets = torch.cumsum(face_counts, dim=0) - face_counts
        edge_offsets = torch.cumsum(edge_counts, dim=0) - edge_counts
        wire_offsets = torch.cumsum(wire_counts, dim=0) - wire_counts

        adj_face_indices = []
        edge_indices = []
        wire_indices = []
        for sample, face_offset, edge_offset, wire_offset in zip(
            batch, face_offsets, edge_offsets, wire_offsets
        ):
            adj_face_indices.append(
                self._offset_padded_indices(
                    sample["adj_face_index"],
                    sample["adj_face_index_length"],
                    int(face_offset),
                )
            )
            edge_indices.append(
                self._offset_padded_indices(
                    sample["edge_index"],
                    sample["edge_index_length"],
                    int(edge_offset),
                )
            )
            wire_indices.append(
                self._offset_padded_indices(
                    sample["wire_index"],
                    sample["wire_index_length"],
                    int(wire_offset),
                )
            )

        batched_graph = dgl.batch(graphs)
        batched_line_graph = dgl.batch(line_graphs)

        node_feature_keys = (
            "face",
            "tri_normal",
            "face_vis_mask",
            "face_padding_mask",
            "uv_face_points",
        )
        edge_feature_keys = ("edge", "edge_padding_mask", "uv_edge_points")
        for key in node_feature_keys:
            batched_graph.ndata[key] = torch.cat(
                [sample[key] for sample in batch], dim=0
            )
        for key in edge_feature_keys:
            batched_graph.edata[key] = torch.cat(
                [sample[key] for sample in batch], dim=0
            )

        return {
            "graph": batched_graph,
            "line_graph": batched_line_graph,
            "num_faces_per_solid": face_counts,
            "graph_file_path": graph_file_paths,
            "adj_face_index": torch.cat(adj_face_indices, dim=0),
            "adj_face_index_length": torch.cat(
                [sample["adj_face_index_length"] for sample in batch], dim=0
            ),
            "edge_index": torch.cat(edge_indices, dim=0),
            "edge_index_length": torch.cat(
                [sample["edge_index_length"] for sample in batch], dim=0
            ),
            "wire_index": torch.cat(wire_indices, dim=0),
            "wire_index_length": torch.cat(
                [sample["wire_index_length"] for sample in batch], dim=0
            ),
        }

    def get_dataloader(
        self,
        batch_size: int = 128,
        shuffle: bool = True,
        num_workers: int = 0,
        drop_last: bool = True,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    @staticmethod
    def _load_dgl_graph(file_path: str) -> dgl.DGLGraph:
        graphs, _ = load_graphs(file_path)
        if not graphs:
            raise ValueError(f"No DGL graph found in {file_path}")
        graph = graphs[0]
        graph.ndata.clear()
        graph.edata.clear()
        return graph

    @staticmethod
    def _more_uv_grid_path(file_path: str, suffix: str) -> pathlib.Path:
        source = pathlib.Path(file_path)
        return (
            source.parent.parent
            / "more_uvgrid"
            / source.name
            / f"{source.stem}{suffix}"
        )

    def load_one_sample(self, item):
        """Load and validate one preprocessed B-rep sample."""
        face = self.load_face(item["face"])
        topo = self.load_topo(item["topo"])
        graph = self._load_dgl_graph(item["graph"])
        line_graph = self._load_dgl_graph(item["line_graph"])

        num_faces = len(face["face"])
        num_edges = len(topo["edge"])
        if num_faces != graph.num_nodes():
            return (
                None,
                f"Face count mismatch: features={num_faces}, graph={graph.num_nodes()}",
            )
        if num_edges != graph.num_edges():
            return (
                None,
                f"Edge count mismatch: features={num_edges}, graph={graph.num_edges()}",
            )
        if line_graph.num_nodes() != num_edges:
            return (
                None,
                "Line-graph node count must equal the number of B-rep edges: "
                f"line_graph={line_graph.num_nodes()}, edges={num_edges}",
            )
        if face["uv_face_points"].shape != (
            num_faces,
            self.num_uv_samples,
            self.num_uv_samples,
            3,
        ):
            return (
                None,
                f"Unexpected face target shape: {tuple(face['uv_face_points'].shape)}",
            )
        if topo["uv_edge_points"].shape != (num_edges, self.num_uv_samples, 3):
            return (
                None,
                f"Unexpected edge target shape: {tuple(topo['uv_edge_points'].shape)}",
            )

        data = {**face, **topo}
        data["graph_file_path"] = item["graph"]
        data["graph"] = graph
        data["line_graph"] = line_graph
        return data, "success"

    def load_face(self, file_path):
        """Load face primitives and the face-level shape prediction target."""
        # Legacy preprocessed files contain Python and NumPy objects, so they
        # cannot all be read with weights_only=True. Only load trusted files.
        labels = torch.load(file_path, map_location="cpu", weights_only=False)
        required_keys = {"nodes", "in_mask", "points", "tri_normals"}
        missing = required_keys.difference(labels)
        if missing:
            raise KeyError(f"Face file {file_path} is missing keys: {sorted(missing)}")

        if self.num_uv_samples == 3:
            if "uv_face_points" not in labels:
                raise KeyError(f"Face file {file_path} is missing uv_face_points")
            uv_face_points = _to_float_tensor(labels["uv_face_points"])
        else:
            uv_file = self._more_uv_grid_path(file_path, "_uvgrid.bin")
            uv_data = torch.load(uv_file, map_location="cpu", weights_only=False)
            key = (
                f"uv_face_points_{self.num_uv_samples + 2}_"
                f"{self.num_uv_samples + 2}"
            )
            if key not in uv_data:
                raise KeyError(f"UV file {uv_file} is missing {key}")
            uv_face_points = _to_float_tensor(uv_data[key])

        return {
            "face": labels["nodes"],
            "face_vis_mask": labels["in_mask"],
            "points": torch.as_tensor(labels["points"]),
            "tri_normal": labels["tri_normals"],
            "uv_face_points": uv_face_points,
        }

    def load_topo(self, file_path):
        """Load edge primitives, topology indices, and edge prediction targets."""
        solid = torch.load(file_path, map_location="cpu", weights_only=False)
        required_keys = {"edge", "edge_index", "wire_index", "adj_face_index"}
        missing = required_keys.difference(solid)
        if missing:
            raise KeyError(
                f"Topology file {file_path} is missing keys: {sorted(missing)}"
            )

        adj_face_index_tensor, adj_face_index_length = pad_or_sample(
            solid["adj_face_index"], MAX_ADJACENT_FACES, dtype=torch.long
        )
        wire_index_tensor, wire_index_length = pad_or_sample(
            solid["wire_index"], MAX_WIRES_PER_FACE, dtype=torch.long
        )
        edge_index_tensor, edge_index_length = pad_or_sample(
            solid["edge_index"], MAX_EDGES_PER_WIRE, dtype=torch.long
        )

        if self.num_uv_samples == 3:
            if "uv_edge_points" not in solid:
                raise KeyError(f"Topology file {file_path} is missing uv_edge_points")
            uv_edge_points = _to_float_tensor(solid["uv_edge_points"])
        else:
            uv_file = self._more_uv_grid_path(file_path, "_uvgrid_edge.bin")
            uv_data = torch.load(uv_file, map_location="cpu", weights_only=False)
            key = f"uv_edge_points_{self.num_uv_samples + 2}"
            if key not in uv_data:
                raise KeyError(f"UV file {uv_file} is missing {key}")
            uv_edge_points = _to_float_tensor(uv_data[key])

        return {
            "edge": solid["edge"],
            "uv_edge_points": uv_edge_points,
            "edge_index_length": edge_index_length,
            "wire_index_length": wire_index_length,
            "adj_face_index_length": adj_face_index_length,
            "edge_index": edge_index_tensor,
            "wire_index": wire_index_tensor,
            "adj_face_index": adj_face_index_tensor,
        }
