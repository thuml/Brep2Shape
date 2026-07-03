import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics

from utils.training import build_optimizer_and_scheduler

from .dual_classification import DualClassification


class ClassificationPL(pl.LightningModule):
    """Train, validate, and evaluate the solid classification model."""

    def __init__(
        self,
        num_classes,
        args=None,
        pretrain_checkpoint=None,
        use_checkpoint=False,
        masking_rate=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["masking_rate"])
        if args.method != "dual":
            raise ValueError(f"Unsupported classification method: {args.method}")

        self.args = args
        self.model = DualClassification(
            args=args,
            pretrain_checkpoint=pretrain_checkpoint,
            use_checkpoint=use_checkpoint,
        )
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=num_classes,
        )
        self.val_acc = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=num_classes,
        )
        self.test_acc = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=num_classes,
        )
        self.best_val_acc = 0.0
        self.best_acc_epoch = 0

    def forward(self, batch):
        return self.model(batch)

    def _model_inputs(self, batch):
        return {
            "graph": batch["graph"].to(self.device),
            "line_graph": batch["line_graph"].to(self.device),
        }

    def _shared_step(self, batch, stage: str, metric) -> torch.Tensor:
        labels = batch["label"].to(self.device)
        logits = self(self._model_inputs(batch))
        loss = F.cross_entropy(logits, labels)
        batch_size = labels.shape[0]

        self.log(
            f"{stage}/{stage}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        metric.update(logits, labels)
        self.log(
            f"{stage}/{stage}_acc",
            metric,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train", self.train_acc)

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val", self.val_acc)

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test", self.test_acc)

    def getBatchSize(self, batch):
        """Return the number of solids; retained for API compatibility."""
        return batch["label"].shape[0]

    def configure_optimizers(self):
        return build_optimizer_and_scheduler(self, self.args)

    def on_train_epoch_end(self):
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log(
            "current_lr",
            current_lr,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def on_validation_epoch_end(self):
        current_val_acc = self.val_acc.compute()
        if current_val_acc > self.best_val_acc:
            self.best_val_acc = current_val_acc.item()
            self.best_acc_epoch = self.current_epoch
        self.log(
            "val/best_val_acc",
            self.best_val_acc,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val/best_acc_epoch",
            float(self.best_acc_epoch),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
