from torch import nn

from .backbone import encode_brep
from .checkpoint import load_pretrained_encoders
from .dual_encoder import DualCurveEncoder, DualGraphEncoder, DualSurfaceEncoder
from .encoders import PredictionHead
from .reporting import report_parameters


class DualClassification(nn.Module):
    """Classify a solid from its curve, surface, and graph representations."""

    def __init__(
        self,
        args,
        pretrain_checkpoint=None,
        use_checkpoint=False,
        load_pretrain=True,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint or getattr(args, "use_checkpoint", False)
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
            return_edge_feat=False,
            add_edge_to_graph=args.add_edge_to_graph,
        )
        self.head = PredictionHead(
            input_dim=args.graph_emb_dim,
            num_classes=args.num_classes,
            dropout=getattr(args, "head_dropout", args.dropout),
            act=args.act,
            use_layer_norm=getattr(args, "use_layer_norm", True),
        )

        if load_pretrain and pretrain_checkpoint is not None:
            self.load_pretrained_encoders(pretrain_checkpoint)

    def print_parameters(self, save_path=None):
        """Print parameter counts and optionally save the report."""
        return report_parameters(
            {
                "Curve Encoder": self.curve_layer,
                "Surface Encoder": self.surface_layer,
                "Graph Encoder": self.graph_layer,
                "Classification Head": self.head,
            },
            save_path,
        )

    def load_pretrained_encoders(self, checkpoint_path):
        """Initialize all encoder modules from a pretraining checkpoint."""
        return load_pretrained_encoders(
            checkpoint_path,
            curve_layer=self.curve_layer,
            surface_layer=self.surface_layer,
            graph_layer=self.graph_layer,
        )

    def forward(self, batch):
        encoding = encode_brep(
            batch,
            curve_layer=self.curve_layer,
            surface_layer=self.surface_layer,
            graph_layer=self.graph_layer,
            use_checkpoint=self.use_checkpoint,
        )
        return self.head(encoding.solid)
