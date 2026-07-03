from dataclasses import dataclass


@dataclass(frozen=True)
class Brep2ShapeConfig:
    """Stable model-only configuration independent of the command-line parser."""

    num_classes: int = 10
    curve_emb_dim: int = 64
    surface_emb_dim: int = 64
    graph_emb_dim: int = 128
    curve_hidden_dim: int = 128
    surface_hidden_dim: int = 128
    graph_hidden_dim: int = 128
    mlp_hidden_dim: int = 128
    edge_num_layers: int = 3
    surface_num_layers: int = 3
    graph_num_layers: int = 3
    curve_num_heads: int = 8
    surface_num_heads: int = 8
    graph_num_heads: int = 8
    dim_feedforward: int = 512
    dropout: float = 0.25
    attention_dropout: float = 0.1
    head_dropout: float = 0.1
    mlp_dropout: float = 0.1
    act: str = "gelu"
    add_positional_encoding: bool = False
    use_node_bias: bool = False
    use_edge_bias: bool = False
    add_edge_to_graph: bool = False
    use_class_token: bool = False
    use_layer_norm: bool = False
    norm_first: bool = False
    use_checkpoint: bool = False
    u_samples: int = 3
    v_samples: int = 3

    def __post_init__(self):
        positive_values = {
            "num_classes": self.num_classes,
            "curve_emb_dim": self.curve_emb_dim,
            "surface_emb_dim": self.surface_emb_dim,
            "graph_emb_dim": self.graph_emb_dim,
            "curve_hidden_dim": self.curve_hidden_dim,
            "surface_hidden_dim": self.surface_hidden_dim,
            "graph_hidden_dim": self.graph_hidden_dim,
            "edge_num_layers": self.edge_num_layers,
            "surface_num_layers": self.surface_num_layers,
            "graph_num_layers": self.graph_num_layers,
            "u_samples": self.u_samples,
            "v_samples": self.v_samples,
        }
        invalid = [name for name, value in positive_values.items() if value <= 0]
        if invalid:
            raise ValueError(f"Model configuration values must be positive: {invalid}")
        if self.u_samples != self.v_samples:
            raise ValueError(
                "u_samples and v_samples must be equal because processed data "
                "uses square UV grids"
            )

        head_pairs = (
            ("curve_hidden_dim", self.curve_hidden_dim, self.curve_num_heads),
            ("surface_hidden_dim", self.surface_hidden_dim, self.surface_num_heads),
            ("graph_hidden_dim", self.graph_hidden_dim, self.graph_num_heads),
        )
        for name, dimension, heads in head_pairs:
            if heads <= 0 or dimension % heads != 0:
                raise ValueError(f"{name} must be divisible by its attention head count")

        for name, value in {
            "dropout": self.dropout,
            "attention_dropout": self.attention_dropout,
            "head_dropout": self.head_dropout,
            "mlp_dropout": self.mlp_dropout,
        }.items():
            if not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1)")
