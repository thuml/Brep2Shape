import argparse
import pathlib


def build_parser():
    parser = argparse.ArgumentParser(description="Brep2Shape Self-Supervised Pretraining")
    parser.add_argument("traintest", choices=("train", "test"), help="Whether to train or test")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Dataset directory containing datasplit.json")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--num_workers_loading", type=int, default=8, help="Workers for eager dataset loading")
    parser.add_argument("--lazy_load", action="store_true", help="Load samples lazily")
    parser.add_argument("--u_samples", type=int, default=3, help="UV samples per dimension")
    parser.add_argument("--v_samples", type=int, default=3,help="Face UV samples per dimension; must match --u_samples",)

    parser.add_argument("--graph_num_heads", type=int, default=8)
    parser.add_argument("--curve_num_heads", type=int, default=8)
    parser.add_argument("--surface_num_heads", type=int, default=8)
    parser.add_argument("--edge_num_layers", type=int, default=3)
    parser.add_argument("--surface_num_layers", type=int, default=3)
    parser.add_argument("--graph_num_layers", type=int, default=3)
    parser.add_argument("--mlp_hidden_dim", type=int, default=128)
    parser.add_argument("--curve_hidden_dim", type=int, default=128)
    parser.add_argument("--surface_hidden_dim", type=int, default=128)
    parser.add_argument("--graph_hidden_dim", type=int, default=128)
    parser.add_argument("--dim_feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--mlp_dropout", type=float, default=0.1)
    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument("--act", type=str, default="gelu")
    parser.add_argument("--curve_emb_dim", type=int, default=64)
    parser.add_argument("--surface_emb_dim", type=int, default=64)
    parser.add_argument("--graph_emb_dim", type=int, default=128)
    parser.add_argument("--add_positional_encoding", action="store_true")
    parser.add_argument("--use_node_bias", action="store_true")
    parser.add_argument("--use_edge_bias", action="store_true")
    parser.add_argument("--add_edge_to_graph", action="store_true")
    parser.add_argument("--use_layer_norm", action="store_true")
    parser.add_argument("--norm_first", action="store_true")
    parser.add_argument("--use_class_token", action="store_true")

    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gpus", type=str, default="-1", help="GPU devices for Lightning, use -1 for all GPUs")
    parser.add_argument("--accelerator", type=str, default="ddp", choices=("ddp", "gpu", "None", "fsdp"), help="Training accelerator")
    parser.add_argument("--precision", type=str, default="medium", choices=("medium", "high", "highest"), help="PyTorch matmul precision")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=("cosine", "step", "fix", "cosine_warmup"), help="Scheduler")
    parser.add_argument("--optimizer", type=str, default="adam", choices=("adam", "adamw", "sgd"), help="Optimizer")
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95), help="Adam/AdamW betas")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--experiment_name", type=str, default="pretraining", help="Experiment name")
    parser.add_argument("--desc", type=str, default="", help="Optional run description")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Checkpoint to resume trainer state")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file for testing")
    parser.add_argument("--use_checkpoint", action="store_true", help="Use activation checkpointing")
    parser.add_argument("--max_grad_norm", type=float, default=0.0, help="Max gradient norm")
    return parser


def _checkpoint_specs():
    return [
        {"monitor": "loss/val_loss", "filename": "best", "save_last": True, "mode": "min"},
        {"filename": "epoch_{epoch:04d}", "every_n_epochs": 50, "save_top_k": -1},
    ]


def build_model(args):
    from models.pretraining import PretrainingPL

    _validate_sampling_args(args)
    return PretrainingPL(args=args)


def _validate_sampling_args(args):
    if args.u_samples != args.v_samples:
        raise ValueError(
            "The processed dataset uses square UV grids, so "
            "--u_samples and --v_samples must be equal"
        )


def _build_dataset(args, split):
    from datasets.pretraining_dataset import PretrainingDataset

    _validate_sampling_args(args)
    return PretrainingDataset(
        dataset_dir=args.dataset_dir,
        split=split,
        lazy_load=args.lazy_load,
        num_workers_loading=args.num_workers_loading,
        num_uv_samples=args.u_samples,
    )


def main():
    args = build_parser().parse_args()
    _validate_sampling_args(args)
    import torch
    from lightning.pytorch import seed_everything
    from utils.training import (
        build_trainer,
        create_run_paths,
        print_run_banner,
        require_checkpoint,
        save_model_architecture,
        save_run_config,
    )

    seed_everything(seed=args.seed, workers=True)
    torch.set_float32_matmul_precision(args.precision)

    paths = create_run_paths(pathlib.Path(__file__).parent, args.experiment_name, args.desc)
    trainer = build_trainer(args, paths, _checkpoint_specs(), timeout_hours=10)

    if args.traintest == "train":
        print_run_banner("Brep2Shape Self-Supervised Pretraining", args.experiment_name, paths, "best.ckpt")
        save_run_config(args, paths.run_dir)
        args.param_save_path = paths.run_dir.joinpath("parameters.txt")
        train_loader = _build_dataset(args, "train").get_dataloader(
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True,
        )
        val_loader = _build_dataset(args, "val").get_dataloader(
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
        )
        model = build_model(args)
        model.model.print_parameters(args.param_save_path)
        save_model_architecture(model, paths.run_dir)
        trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume_from_checkpoint)
        return

    checkpoint = require_checkpoint(args.checkpoint)
    from models.pretraining import PretrainingPL

    test_loader = _build_dataset(args, "test").get_dataloader(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    model = PretrainingPL.load_from_checkpoint(checkpoint)
    results = trainer.test(model=model, dataloaders=[test_loader], verbose=True)
    print(f"Pretraining Loss on test set: {results[0]['loss/test_loss']}")


if __name__ == "__main__":
    main()
