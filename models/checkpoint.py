import argparse
import logging
import pathlib
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModuleLoadReport:
    """Result of loading one encoder module from a pretraining checkpoint."""

    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


ENCODER_PREFIXES = {
    "curve": ("model.curve_layer.", "model.curve_encoder."),
    "surface": (
        "model.surface_layer.",
        "model.face_layer.",
        "model.face_encoder.",
    ),
    "graph": ("model.graph_layer.", "model.graph_encoder."),
}


def load_checkpoint_state_dict(
    checkpoint_path: str | pathlib.Path,
) -> Mapping[str, torch.Tensor]:
    """Safely load a state dict from a PyTorch or Lightning checkpoint."""
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals(
            [argparse.Namespace, pathlib.PosixPath, pathlib.WindowsPath]
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must contain a mapping")

    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, Mapping):
        raise TypeError("Checkpoint state_dict must be a mapping")
    if not all(isinstance(key, str) for key in state_dict):
        raise TypeError("Checkpoint state_dict keys must be strings")
    if not all(isinstance(value, torch.Tensor) for value in state_dict.values()):
        raise TypeError("Checkpoint state_dict values must be tensors")
    return state_dict


def _extract_module_state(
    state_dict: Mapping[str, torch.Tensor],
    prefixes: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    extracted = {}
    for key, value in state_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                extracted[key.removeprefix(prefix)] = value
                break
    return extracted


def load_pretrained_encoders(
    checkpoint_path: str | pathlib.Path,
    *,
    curve_layer: nn.Module,
    surface_layer: nn.Module,
    graph_layer: nn.Module,
) -> dict[str, ModuleLoadReport]:
    """Load all three Brep2Shape encoders, including legacy key aliases."""
    state_dict = load_checkpoint_state_dict(checkpoint_path)
    modules = {
        "curve": curve_layer,
        "surface": surface_layer,
        "graph": graph_layer,
    }
    reports = {}

    for name, module in modules.items():
        module_state = _extract_module_state(state_dict, ENCODER_PREFIXES[name])
        if not module_state:
            raise KeyError(f"Checkpoint does not contain the {name} encoder")
        if name == "graph" and not hasattr(module, "edge_output_proj"):
            module_state = {
                key: value
                for key, value in module_state.items()
                if not key.startswith("edge_output_proj.")
            }

        incompatible = module.load_state_dict(module_state, strict=False)
        report = ModuleLoadReport(
            missing_keys=tuple(incompatible.missing_keys),
            unexpected_keys=tuple(incompatible.unexpected_keys),
        )
        reports[name] = report
        if report.missing_keys:
            LOGGER.warning(
                "%s encoder is missing checkpoint keys: %s",
                name,
                report.missing_keys,
            )
        if report.unexpected_keys:
            LOGGER.warning(
                "%s encoder has unexpected checkpoint keys: %s",
                name,
                report.unexpected_keys,
            )

    return reports
