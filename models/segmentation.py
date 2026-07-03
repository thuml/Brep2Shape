import logging

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics

from utils.training import build_optimizer_and_scheduler

from .dual_segmentation import DualSegmentation


LOGGER = logging.getLogger(__name__)


class SegmentationPL(pl.LightningModule):
    """Train, validate, and evaluate the face segmentation model."""

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
            raise ValueError(f"Unsupported segmentation method: {args.method}")

        self.args = args
        self.num_classes = num_classes
        load_pretrain = getattr(args, "traintest", "train") == "train"
        if not load_pretrain:
            LOGGER.info("Test mode: skipping pretraining-weight initialization")
        self.model = DualSegmentation(
            args=args,
            pretrain_checkpoint=pretrain_checkpoint,
            use_checkpoint=use_checkpoint,
            load_pretrain=load_pretrain,
        )

        metric_options = {"task": "multiclass", "num_classes": num_classes}
        self.train_acc = torchmetrics.Accuracy(**metric_options)
        self.val_acc = torchmetrics.Accuracy(**metric_options)
        self.test_acc = torchmetrics.Accuracy(**metric_options)
        self.train_iou = torchmetrics.JaccardIndex(**metric_options)
        self.val_iou = torchmetrics.JaccardIndex(**metric_options)
        self.test_iou = torchmetrics.JaccardIndex(**metric_options)

        self.best_val_iou = 0.0
        self.best_val_acc = 0.0
        self.best_iou_epoch = 0
        self.best_acc_epoch = 0
        self.best_iou_acc = 0.0
        self.best_acc_iou = 0.0

    def forward(self, batch):
        return self.model(batch)

    def _model_inputs(self, batch):
        return {
            "graph": batch["graph"].to(self.device),
            "line_graph": batch["line_graph"].to(self.device),
        }

    def _shared_step(
        self,
        batch,
        stage: str,
        accuracy,
        iou,
    ) -> torch.Tensor:
        inputs = self._model_inputs(batch)
        labels = inputs["graph"].ndata["label"]
        logits = self(inputs)
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
        accuracy.update(logits, labels)
        iou.update(logits, labels)
        self.log(
            f"{stage}/{stage}_acc",
            accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}/{stage}_iou",
            iou,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(
            batch,
            "train",
            self.train_acc,
            self.train_iou,
        )

    def validation_step(self, batch, batch_idx):
        return self._shared_step(
            batch,
            "val",
            self.val_acc,
            self.val_iou,
        )

    def test_step(self, batch, batch_idx):
        return self._shared_step(
            batch,
            "test",
            self.test_acc,
            self.test_iou,
        )

    def on_validation_epoch_end(self):
        """Record the best validation IoU and accuracy reached so far."""
        current_iou = self.val_iou.compute()
        current_acc = self.val_acc.compute()
        if current_iou > self.best_val_iou:
            self.best_val_iou = current_iou.item()
            self.best_iou_epoch = self.current_epoch
            self.best_iou_acc = current_acc.item()
        if current_acc > self.best_val_acc:
            self.best_val_acc = current_acc.item()
            self.best_acc_epoch = self.current_epoch
            self.best_acc_iou = current_iou.item()

        best_metrics = {
            "val/best_val_iou": self.best_val_iou,
            "val/best_val_acc": self.best_val_acc,
            "val/best_iou_epoch": float(self.best_iou_epoch),
            "val/best_acc_epoch": float(self.best_acc_epoch),
            "val/best_iou_acc": self.best_iou_acc,
            "val/best_acc_iou": self.best_acc_iou,
        }
        for name, value in best_metrics.items():
            self.log(
                name,
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

    def getBatchSize(self, batch):
        """Return the number of faces; retained for API compatibility."""
        return batch["graph"].ndata["label"].shape[0]

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
