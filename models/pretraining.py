from typing import Any

import pytorch_lightning as pl
import torch
from torch import nn

from utils.training import build_optimizer_and_scheduler

from .backbone import encode_brep
from .dual_encoder import DualCurveEncoder, DualGraphEncoder, DualSurfaceEncoder
from .reporting import report_parameters


class UVPointPrediction(nn.Module):
    """Predict sampled 3D points for every B-rep face and edge."""

    def __init__(self, args):
        super().__init__()
        if args.u_samples != args.v_samples:
            raise ValueError(
                "u_samples and v_samples must be equal because processed data "
                "uses square UV grids"
            )
        self.curve_layer = DualCurveEncoder(
            input_dim=4 * 11,
            curve_emb_dim=args.curve_emb_dim,
            dropout=args.dropout,
            n_heads=args.curve_num_heads,
            hidden_dim=args.curve_hidden_dim,
            n_layers=args.edge_num_layers,
            use_layer_norm=getattr(args, "use_layer_norm", False),
            norm_first=getattr(args, "norm_first", False),
            act=getattr(args, "act", "gelu"),
        )
        self.surface_layer = DualSurfaceEncoder(
            input_dim=28 * 4 + 1 + 7,
            surface_emb_dim=args.surface_emb_dim,
            dropout=args.dropout,
            n_heads=args.surface_num_heads,
            hidden_dim=args.surface_hidden_dim,
            n_layers=args.surface_num_layers,
            use_class_token=getattr(args, "use_class_token", False),
            use_layer_norm=getattr(args, "use_layer_norm", False),
            norm_first=getattr(args, "norm_first", False),
            act=getattr(args, "act", "gelu"),
        )
        self.graph_layer = DualGraphEncoder(
            input_edge_dim=args.curve_emb_dim,
            input_surface_dim=args.surface_emb_dim,
            output_dim=args.graph_emb_dim,
            hidden_dim=args.graph_hidden_dim,
            num_layers=args.graph_num_layers,
            num_heads=args.graph_num_heads,
            dropout=args.dropout,
            attention_dropout=getattr(args, "attention_dropout", args.dropout),
            dim_feedforward=args.dim_feedforward,
            add_positional_encoding=args.add_positional_encoding,
            use_edge_bias=args.use_edge_bias,
            use_node_bias=args.use_node_bias,
            act=args.act,
            return_edge_feat=True,
            add_edge_to_graph=args.add_edge_to_graph,
        )
        self.edge_uv_predictor = nn.Sequential(
            nn.Linear(args.curve_emb_dim, args.mlp_hidden_dim),
            nn.Dropout(args.mlp_dropout),
            nn.GELU(),
            nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim),
            nn.Dropout(args.mlp_dropout),
            nn.GELU(),
            nn.Linear(args.mlp_hidden_dim, 3 * args.u_samples),
        )
        self.face_uv_predictor = nn.Sequential(
            nn.Linear(
                args.surface_emb_dim + args.curve_emb_dim,
                args.mlp_hidden_dim,
            ),
            nn.Dropout(args.mlp_dropout),
            nn.GELU(),
            nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim),
            nn.Dropout(args.mlp_dropout),
            nn.GELU(),
            nn.Linear(args.mlp_hidden_dim, 3 * args.u_samples * args.v_samples),
        )
        self.use_checkpoint = args.use_checkpoint
        self.u_samples = args.u_samples
        self.v_samples = args.v_samples

    def print_parameters(self, save_path=None):
        """Print parameter counts and optionally save the report."""
        return report_parameters(
            {
                "Curve Encoder": self.curve_layer,
                "Surface Encoder": self.surface_layer,
                "Graph Encoder": self.graph_layer,
                "Edge Prediction Head": self.edge_uv_predictor,
                "Face Prediction Head": self.face_uv_predictor,
            },
            save_path,
        )

    def forward(self, batch) -> dict[str, torch.Tensor]:
        encoding = encode_brep(
            batch,
            curve_layer=self.curve_layer,
            surface_layer=self.surface_layer,
            graph_layer=self.graph_layer,
            use_checkpoint=self.use_checkpoint,
        )
        if encoding.edge is None:
            raise RuntimeError("Pretraining requires graph-level edge embeddings")

        graph = batch["graph"]
        _, dst = graph.edges()
        num_faces = encoding.face.shape[0]
        curve_dim = encoding.edge.shape[-1]

        face_edge_counts = torch.zeros(
            num_faces,
            device=encoding.edge.device,
            dtype=encoding.edge.dtype,
        )
        face_edge_counts.index_add_(
            0,
            dst,
            torch.ones_like(dst, dtype=encoding.edge.dtype),
        )
        face_edge_sum = torch.zeros(
            num_faces,
            curve_dim,
            device=encoding.edge.device,
            dtype=encoding.edge.dtype,
        )
        face_edge_sum.index_add_(0, dst, encoding.edge)
        face_edge_mean = face_edge_sum / face_edge_counts.clamp_min(1).unsqueeze(-1)

        face_features = torch.cat([encoding.face, face_edge_mean], dim=-1)
        return {
            "edge_uv_points": self.edge_uv_predictor(encoding.edge),
            "face_uv_points": self.face_uv_predictor(face_features),
        }


class PretrainingPL(pl.LightningModule):
    """Lightning module for self-supervised face and edge point prediction."""

    def __init__(self, args=None):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.u_samples = args.u_samples
        self.v_samples = args.v_samples
        self.learning_rate = args.learning_rate
        self.model = UVPointPrediction(args=args)
        self.mse_loss = nn.MSELoss()

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._uv_prediction_step(batch, "train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._uv_prediction_step(batch, "val")

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._uv_prediction_step(batch, "test")

    def _uv_prediction_step(
        self,
        batch: dict[str, Any],
        stage: str,
    ) -> torch.Tensor:
        outputs = self.model(batch)
        edge_predictions = outputs["edge_uv_points"].view(-1, self.u_samples, 3)
        face_predictions = outputs["face_uv_points"].view(
            -1,
            self.u_samples,
            self.v_samples,
            3,
        )
        graph = batch["graph"]
        edge_loss = self.mse_loss(
            edge_predictions,
            graph.edata["uv_edge_points"],
        )
        face_loss = self.mse_loss(
            face_predictions,
            graph.ndata["uv_face_points"],
        )
        loss = edge_loss + face_loss
        batch_size = graph.batch_size

        self.log(
            f"loss/{stage}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            f"loss/{stage}_edge_loss",
            edge_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            f"loss/{stage}_face_loss",
            face_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        return loss

    def configure_optimizers(self):
        return build_optimizer_and_scheduler(
            self,
            self.args,
            learning_rate=self.learning_rate,
        )

    def on_train_epoch_end(self):
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log(
            "current_lr",
            current_lr,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
