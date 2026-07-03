"""Primitive encoders shared by the Brep2Shape task models."""
import math
import torch
from torch import nn
import torch.nn.functional as F


class PredictionHead(nn.Module):
    """Three-layer prediction head for classification or segmentation logits."""

    def __init__(self, input_dim, num_classes, dropout=0.3, act='relu', use_layer_norm=True):
        """Initialize the prediction head.

        Args:
            input_dim: Input feature dimension.
            num_classes: Number of output classes.
            dropout: Dropout probability after each hidden layer.
        """
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 512, bias=False)
        self.bn1 = nn.LayerNorm(512) if use_layer_norm else nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn2 = nn.LayerNorm(256) if use_layer_norm else nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, num_classes)
        self.act = act

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, inp):
        """Map input features to unnormalized class logits.

        Args:
            inp: Feature tensor with shape ``[batch_size, input_dim]``.

        Returns:
            Logits with shape ``[batch_size, num_classes]``.
        """
        if self.act == 'relu':
            x = F.relu(self.bn1(self.linear1(inp)))
        elif self.act == 'gelu':
            x = F.gelu(self.bn1(self.linear1(inp)))
        else:
            raise NotImplementedError(f"Activation function {self.act} not implemented")
        x = self.dp1(x)
        if self.act == 'relu':
            x = F.relu(self.bn2(self.linear2(x)))
        elif self.act == 'gelu':
            x = F.gelu(self.bn2(self.linear2(x)))
        else:
            raise NotImplementedError(f"Activation function {self.act} not implemented")
        x = self.dp2(x)
        x = self.linear3(x)
        return x  

class _MLP(nn.Module):
    """Configurable multilayer perceptron with a linear output layer."""

    def __init__(
        self,
        num_layers,
        input_dim,
        hidden_dim,
        output_dim,
        use_layer_norm=False,
        act="relu",
    ):
        """Initialize the multilayer perceptron.

        Args:
            num_layers: Number of linear layers.
            input_dim: Input feature dimension.
            hidden_dim: Shared hidden feature dimension.
            output_dim: Output feature dimension.

        Raises:
            ValueError: If ``num_layers`` is less than one.
        """
        super(_MLP, self).__init__()
        self.linear_or_not = True
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.use_layer_norm = use_layer_norm
        self.act = act
        if num_layers < 1:
            raise ValueError("Number of layers should be positive!")
        elif num_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                if self.use_layer_norm:
                    self.batch_norms.append(nn.LayerNorm(hidden_dim))
                else:
                    self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            h = x
            for i in range(self.num_layers - 1):
                h = self.linears[i](h)
                h = self.batch_norms[i](h)
                    
                if self.act == 'relu':
                    h = F.relu(h)
                elif self.act == 'gelu':
                    h = F.gelu(h)
                else:
                    raise NotImplementedError(f"Activation function {self.act} not implemented")
            return self.linears[-1](h)

class BezierEncoderMLP(nn.Module):
    """Encode flattened Bezier control points with a residual MLP."""

    def __init__(
        self,
        out_dim=64,
        input_dim=28 * 4,
        hidden_dim=256,
        use_layer_norm=False,
        act="relu",
    ):
        super().__init__()
        options = {
            "num_layers": 3,
            "hidden_dim": hidden_dim,
            "output_dim": out_dim,
            "use_layer_norm": use_layer_norm,
            "act": act,
        }
        self.mlp = _MLP(input_dim=input_dim, **options)
        self.mlp2 = _MLP(input_dim=out_dim, **options)

    def forward(self, x: torch.Tensor):
        x = self.mlp(x)
        x = x+self.mlp2(x)
        return x

    def weights_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

class PositionalEncoding(nn.Module):
    """Add fixed sinusoidal position features to primitive sequences.

    Args:
        d_model: Embedding dimension.
        max_len: Maximum supported sequence length.
        dropout: Dropout probability after adding position features.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 8000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        odd_dimensions = pe[:, 1::2].shape[1]
        pe[:, 1::2] = torch.cos(position * div_term[:odd_dimensions])

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional features to a sequence or batch of sequences.

        Args:
            x: Tensor with shape ``[batch, sequence, d_model]`` or
                ``[sequence, d_model]``.

        Returns:
            A tensor with the same shape and dtype as ``x``.
        """
        if x.dim() == 3:
            x = x + self.pe[:, :x.size(1), :]
        elif x.dim() == 2:
            x = x + self.pe[:, :x.size(0), :]
        else:
            raise ValueError("Input tensor must have 2 or 3 dimensions")

        return self.dropout(x)

class TransformerEncoderBlock(nn.Module):
    """Thin wrapper around a stack of PyTorch transformer encoder layers."""

    def __init__(
        self,
        input_dim,
        c_hidden,
        n_layers,
        n_heads,
        dropout=0.01,
        batch_first=True,
        act="relu",
        norm_first=False,
    ):
        """Initialize a batch-first transformer encoder stack."""
        super().__init__()
        if input_dim % n_heads != 0:
            raise ValueError("input_dim must be divisible by n_heads")
        encoder_layers = nn.TransformerEncoderLayer(
            input_dim,
            n_heads,
            c_hidden,
            dropout,
            batch_first=batch_first,
            activation=act,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(encoder_layers, n_layers)

    def forward(self, x, src_key_padding_mask=None, src_mask=None):
        return self.encoder(
            x,
            src_key_padding_mask=src_key_padding_mask,
            mask=src_mask,
        )


# Backward-compatible aliases for older imports.
_NonLinearClassifier = PredictionHead
TransformerEncoderBLock = TransformerEncoderBlock
