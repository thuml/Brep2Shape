import json
import math
import os
import pathlib
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Iterable

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy, FSDPStrategy


@dataclass(frozen=True)
class RunPaths:
    results_path: pathlib.Path
    month_day: str
    run_id: str
    run_dir: pathlib.Path


def create_run_paths(project_root: pathlib.Path, experiment_name: str, desc: str | None = None) -> RunPaths:
    results_path = project_root.joinpath("results", experiment_name)
    timestamp = os.environ.get("BREP2SHAPE_RUN_TIMESTAMP")
    if timestamp is None:
        timestamp = time.strftime("%m%d-%H%M%S")
        os.environ["BREP2SHAPE_RUN_TIMESTAMP"] = timestamp
    month_day, hour_min_second = timestamp.split("-", maxsplit=1)
    desc_str = desc.strip().replace(" ", "_") if desc else None
    run_id = f"{hour_min_second}_{desc_str}" if desc_str else hour_min_second
    run_dir = results_path.joinpath(month_day, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        results_path=results_path,
        month_day=month_day,
        run_id=run_id,
        run_dir=run_dir,
    )


def save_run_config(args: Any, run_dir: pathlib.Path) -> None:
    with open(run_dir.joinpath("args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False, default=str)


def save_model_architecture(model: torch.nn.Module, run_dir: pathlib.Path) -> None:
    with open(run_dir.joinpath("model_architecture.txt"), "w", encoding="utf-8") as f:
        f.write(str(model))


def print_run_banner(title: str, experiment_name: str, paths: RunPaths, checkpoint_name: str) -> None:
    rel_run_dir = pathlib.Path("results").joinpath(experiment_name, paths.month_day, paths.run_id)
    print(
        f"""
-----------------------------------------------------------------------------------
{title}
-----------------------------------------------------------------------------------
Logs written to {rel_run_dir}

To monitor the logs, run:
tensorboard --logdir {rel_run_dir}

The main checkpoint will be written to:
{rel_run_dir / checkpoint_name}
-----------------------------------------------------------------------------------
"""
    )


def _build_callbacks(run_dir: pathlib.Path, checkpoint_specs: Iterable[dict[str, Any]]) -> list[ModelCheckpoint]:
    callbacks = []
    for spec in checkpoint_specs:
        callbacks.append(ModelCheckpoint(dirpath=str(run_dir), **spec))
    return callbacks


def _resolve_accelerator_and_strategy(args: Any, timeout_hours: int):
    accelerator_arg = getattr(args, "accelerator", "gpu")
    is_train = getattr(args, "traintest", "train") == "train"
    if accelerator_arg in (None, "None", "cpu"):
        return "cpu", "auto"
    if not is_train:
        return "gpu", "auto"
    if accelerator_arg == "gpu":
        return "gpu", "auto"
    if accelerator_arg == "ddp":
        return "gpu", DDPStrategy(
            timeout=timedelta(hours=timeout_hours),
            find_unused_parameters=True,
        )
    if accelerator_arg == "fsdp":
        return "gpu", FSDPStrategy(
            timeout=timedelta(hours=timeout_hours),
            sharding_strategy="FULL_SHARD",
        )
    raise ValueError(f"Unknown accelerator: {accelerator_arg}")


def build_trainer(
    args: Any,
    paths: RunPaths,
    checkpoint_specs: Iterable[dict[str, Any]],
    *,
    timeout_hours: int = 2,
) -> Trainer:
    accelerator, strategy = _resolve_accelerator_and_strategy(args, timeout_hours)
    devices = 1 if accelerator == "cpu" else getattr(args, "gpus", "auto")
    return Trainer(
        callbacks=_build_callbacks(paths.run_dir, checkpoint_specs),
        logger=TensorBoardLogger(
            str(paths.results_path),
            name=paths.month_day,
            version=paths.run_id,
        ),
        devices=devices,
        accelerator=accelerator,
        strategy=strategy,
        max_epochs=args.max_epochs,
        num_sanity_val_steps=0,
        gradient_clip_val=getattr(args, "max_grad_norm", 0.0),
        gradient_clip_algorithm="norm",
    )


def build_optimizer(module: torch.nn.Module, args: Any, learning_rate: float | None = None):
    lr = learning_rate if learning_rate is not None else args.learning_rate
    optimizer_name = args.optimizer.lower()
    if optimizer_name == "adam":
        return torch.optim.Adam(
            module.parameters(),
            lr=lr,
            betas=getattr(args, "betas", (0.9, 0.999)),
            weight_decay=getattr(args, "weight_decay", 0.0),
        )
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            module.parameters(),
            lr=lr,
            betas=getattr(args, "betas", (0.9, 0.999)),
            weight_decay=getattr(args, "weight_decay", 0.0),
        )
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            module.parameters(),
            lr=lr,
            momentum=getattr(args, "momentum", 0.9),
            weight_decay=getattr(args, "weight_decay", 0.0),
        )
    raise ValueError(f"Unknown optimizer: {args.optimizer}")


def build_scheduler(optimizer: torch.optim.Optimizer, args: Any, learning_rate: float | None = None):
    scheduler_name = args.scheduler.lower()
    max_epochs = max(int(getattr(args, "max_epochs", 1)), 1)
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epochs,
            eta_min=getattr(args, "min_lr", 0.0),
        )
    if scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max_epochs,
            gamma=getattr(args, "gamma", 0.1),
        )
    if scheduler_name == "fix":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    if scheduler_name == "cosine_warmup":
        warmup_epochs = min(max(int(getattr(args, "warmup_epochs", 10)), 0), max_epochs)
        base_lr = learning_rate if learning_rate is not None else getattr(args, "learning_rate", 1.0)
        min_lr = getattr(args, "min_lr", 0.0)
        min_factor = 0.0 if base_lr == 0 else min_lr / base_lr

        def lr_lambda(epoch: int) -> float:
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return float(epoch + 1) / float(warmup_epochs)
            progress = (epoch - warmup_epochs) / max(max_epochs - warmup_epochs, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
            return min_factor + (1.0 - min_factor) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    raise ValueError(f"Unknown scheduler: {args.scheduler}")


def build_optimizer_and_scheduler(module: torch.nn.Module, args: Any, learning_rate: float | None = None):
    optimizer = build_optimizer(module, args, learning_rate=learning_rate)
    scheduler = build_scheduler(optimizer, args, learning_rate=learning_rate)
    return [optimizer], [scheduler]


def require_checkpoint(path: str | None) -> str:
    if not path:
        raise ValueError("Expected the --checkpoint argument to be provided")
    return path
