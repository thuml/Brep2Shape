import argparse
import pathlib


def build_parser():
    parser = argparse.ArgumentParser("Brep2Shape Classification")
    parser.add_argument("traintest", choices=("train", "test"), help="Whether to train or test")
    parser.add_argument("--method", choices=("dual",), default="dual", help="Model method")
    parser.add_argument("--experiment_name", type=str, default="classification", help="Experiment name")
    parser.add_argument("--desc", type=str, default=None, help="Optional run description")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing datasplit_new.json")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of classes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of dataloader workers")
    parser.add_argument("--max_epochs", type=int, default=350, help="Number of epochs")
    parser.add_argument("--precision", choices=("medium", "high", "highest"), default="medium", help="PyTorch matmul precision")
    parser.add_argument("--gpus", type=str, default="-1", help="GPU devices for Lightning, use -1 for all GPUs")
    parser.add_argument("--accelerator", type=str, default="ddp", choices=("ddp", "gpu", "None", "fsdp"), help="Training accelerator")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file for testing")
    parser.add_argument("--pretrain_checkpoint", type=str, default=None, help="Pretrained checkpoint for finetuning")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=("cosine", "step", "fix", "cosine_warmup"), help="Scheduler")
    parser.add_argument("--optimizer", type=str, default="adam", choices=("adam", "adamw", "sgd"), help="Optimizer")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95), help="Adam/AdamW betas")
    parser.add_argument("--warmup_epochs", type=int, default=10, help="Warmup epochs for cosine_warmup")
    parser.add_argument("--min_lr", type=float, default=0.0, help="Minimum LR for cosine schedulers")
    parser.add_argument("--gamma", type=float, default=0.1, help="Step scheduler gamma")
    parser.add_argument("--max_grad_norm", type=float, default=0.0, help="Max gradient norm")

    parser.add_argument("--curve_num_heads", type=int, default=8)
    parser.add_argument("--surface_num_heads", type=int, default=8)
    parser.add_argument("--graph_num_heads", type=int, default=8)
    parser.add_argument("--edge_num_layers",type=int,default=3)
    parser.add_argument("--surface_num_layers", type=int, default=3)
    parser.add_argument("--graph_num_layers", type=int, default=3)
    parser.add_argument("--curve_hidden_dim", type=int, default=128)
    parser.add_argument("--surface_hidden_dim", type=int, default=128)
    parser.add_argument("--graph_hidden_dim", type=int, default=128)
    parser.add_argument("--dim_feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--head_dropout", type=float, default=0.1)
    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument("--act", type=str, default="gelu")
    parser.add_argument("--curve_emb_dim", type=int, default=64)
    parser.add_argument("--surface_emb_dim", type=int, default=64)
    parser.add_argument("--graph_emb_dim", type=int, default=128)
    parser.add_argument("--add_positional_encoding", action="store_true")
    parser.add_argument("--use_node_bias", action="store_true")
    parser.add_argument("--use_edge_bias", action="store_true")
    parser.add_argument("--use_checkpoint", action="store_true")
    parser.add_argument("--add_edge_to_graph", action="store_true")
    parser.add_argument("--use_class_token", action="store_true")
    parser.add_argument("--use_layer_norm", action="store_true")
    parser.add_argument("--norm_first", action="store_true")
    parser.add_argument("--lazy_load", action="store_true")
    return parser


def _checkpoint_specs():
    return [
        {"monitor": "val/val_loss", "filename": "best_loss", "save_last": True, "mode": "min"},
        {"monitor": "val/val_acc", "filename": "best_acc", "save_last": True, "mode": "max"},
        {"filename": "epoch_{epoch:04d}", "every_n_epochs": 25, "save_top_k": -1},
    ]


def _build_model(args):
    from models.classification import ClassificationPL

    return ClassificationPL(
        num_classes=args.num_classes,
        args=args,
        pretrain_checkpoint=args.pretrain_checkpoint,
        use_checkpoint=args.use_checkpoint,
    )


def _build_dataset(args, split):
    from datasets.finetuning_dataset import FinetuningDataset

    return FinetuningDataset(
        root_dir=args.dataset_dir,
        split=split,
        use_for_classification=True,
        lazy_load=args.lazy_load,
    )


def _format_results(checkpoint, results):
    acc = results[0]["test/test_acc"] * 100.0
    return f"| Checkpoint | Acc |\n| --- | --- |\n| {checkpoint} | {acc:.2f} |"


def main():
    args = build_parser().parse_args()
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
    trainer = build_trainer(args, paths, _checkpoint_specs(), timeout_hours=2)

    if args.traintest == "train":
        print_run_banner("Brep2Shape Classification", args.experiment_name, paths, "best_loss.ckpt")
        save_run_config(args, paths.run_dir)
        args.param_save_path = paths.run_dir.joinpath("parameters.txt")
        model = _build_model(args)
        model.model.print_parameters(args.param_save_path)
        save_model_architecture(model, paths.run_dir)
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
        trainer.fit(model, train_loader, val_loader)
        best_acc = model.best_val_acc.item() if torch.is_tensor(model.best_val_acc) else model.best_val_acc
        best_acc_epoch = model.best_acc_epoch.item() if torch.is_tensor(model.best_acc_epoch) else model.best_acc_epoch
        print(f"Best accuracy: {best_acc * 100:.2f}% at epoch {best_acc_epoch}")
        return

    checkpoint = require_checkpoint(args.checkpoint)
    from models.classification import ClassificationPL

    test_loader = _build_dataset(args, "test").get_dataloader(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    model = ClassificationPL.load_from_checkpoint(checkpoint)
    results = trainer.test(model=model, dataloaders=[test_loader], verbose=True)
    print(_format_results(checkpoint, results))


if __name__ == "__main__":
    main()
