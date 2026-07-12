import math

import dgl
import torch
import torch.nn.functional as F
from torch import nn

from .encoders import BezierEncoderMLP, PositionalEncoding, TransformerEncoderBlock


ACTIVATION_FACTORIES = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(0.1),
    "softplus": nn.Softplus,
    "elu": nn.ELU,
    "silu": nn.SiLU,
}


def _mean_multiedge_bias(
    edge_bias: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Aggregate parallel directed edges without depending on edge order."""
    if edge_bias.ndim != 2:
        raise ValueError("edge_bias must have shape [num_edges, num_heads]")
    if edge_bias.shape[0] != src.numel() or src.shape != dst.shape:
        raise ValueError("edge_bias, src, and dst must describe the same edges")

    flat_indices = src * num_nodes + dst
    aggregated = edge_bias.new_zeros((num_nodes * num_nodes, edge_bias.shape[1]))
    counts = edge_bias.new_zeros((num_nodes * num_nodes, 1))
    aggregated.index_add_(0, flat_indices, edge_bias)
    counts.index_add_(
        0,
        flat_indices,
        edge_bias.new_ones((edge_bias.shape[0], 1)),
    )
    aggregated = aggregated / counts.clamp_min(1)
    return aggregated.view(num_nodes, num_nodes, edge_bias.shape[1])

def _bd_attn(q, k, v, edge_bias, src, dst, num_nodes_list, num_edges_list, scale, nhead, attn_dropout=None):
    """Batched block-diagonal attention == the per-part loop, in one shot.
    q,k,v: [N_total, H, d]. edge_bias: [E_total, H] or None. Returns [N_total, H, d]."""
    dev = q.device
    d = q.shape[-1]
    P = len(num_nodes_list)
    nn_t = torch.tensor(num_nodes_list, device=dev)
    Nmax = int(nn_t.max())
    N = int(nn_t.sum())
    node_part = torch.repeat_interleave(torch.arange(P, device=dev), nn_t)
    node_off = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), nn_t.cumsum(0)[:-1]])
    node_local = torch.arange(N, device=dev) - node_off[node_part]

    def pad(x):
        o = x.new_zeros((P, Nmax, nhead, d)); o[node_part, node_local] = x; return o
    qp, kp, vp = pad(q), pad(k), pad(v)

    key_valid = torch.arange(Nmax, device=dev)[None, :] < nn_t[:, None]
    add = torch.where(key_valid[:, None, None, :], 0.0, float("-inf"))  # (P,1,1,Nmax)

    if edge_bias is not None:
        ne_t = torch.tensor(num_edges_list, device=dev)
        edge_part = torch.repeat_interleave(torch.arange(P, device=dev), ne_t)
        lsrc = src - node_off[edge_part]
        ldst = dst - node_off[edge_part]
        flat = edge_part * (Nmax * Nmax) + lsrc * Nmax + ldst
        agg = edge_bias.new_zeros((P * Nmax * Nmax, nhead))
        cnt = edge_bias.new_zeros((P * Nmax * Nmax, 1))
        agg.index_add_(0, flat, edge_bias)
        cnt.index_add_(0, flat, edge_bias.new_ones((edge_bias.shape[0], 1)))
        bias = (agg / cnt.clamp_min(1)).view(P, Nmax, Nmax, nhead)
        bias = bias + bias.transpose(1, 2)
        add = add + bias.permute(0, 3, 1, 2)

    scores = torch.einsum("pihd,pjhd->phij", qp, kp) / scale + add
    w = torch.nn.functional.softmax(scores, dim=-1)
    if attn_dropout is not None:
        w = attn_dropout(w)
    outp = torch.einsum("phij,pjhd->pihd", w, vp)
    return outp[node_part, node_local]


class MLP(nn.Module):
    """Feed-forward network with optional residual hidden layers."""

    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()

        try:
            activation_factory = ACTIVATION_FACTORIES[act.lower()]
        except KeyError as exc:
            raise ValueError(f"Unsupported activation: {act}") from exc
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), activation_factory())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(n_hidden, n_hidden), activation_factory())
                for _ in range(n_layers)
            ]
        )

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x

class SubgraphPE(nn.Module):
    """Apply sinusoidal positions independently within each graph in a batch."""

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        if d_model <= 0 or max_len <= 0:
            raise ValueError("d_model and max_len must be positive")
        self.d_model = d_model
        self.max_len = max_len
        pe = self._build_pe(
            max_len,
            d_model,
            device=device or torch.device("cpu"),
            dtype=torch.float32,
        )
        self.register_buffer("pe", pe, persistent=False)

    @staticmethod
    def _build_pe(max_len: int, d_model: int, device, dtype):
        pe = torch.zeros(max_len, d_model, device=device, dtype=dtype)
        position = torch.arange(0, max_len, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device, dtype=dtype)
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        odd_dimensions = pe[:, 1::2].shape[1]
        pe[:, 1::2] = torch.cos(position * div_term[:odd_dimensions])
        return pe

    def _maybe_extend(self, need_len: int, device, dtype):
        if need_len <= self.pe.size(0):
            return
        # Extend the cached encoding only when a larger graph is encountered.
        new_pe = self._build_pe(need_len, self.d_model, device=device, dtype=dtype)
        new_pe[:self.pe.size(0)] = self.pe.to(device=device, dtype=dtype)
        self.pe = new_pe

    @torch.no_grad()
    def build_pos_ids(self, g: dgl.DGLGraph, device) -> torch.Tensor:
        sizes = g.batch_num_nodes().to(device)
        pos_list = [torch.arange(int(n), device=device) for n in sizes]
        pos_ids = (
            torch.cat(pos_list, dim=0)
            if pos_list
            else torch.empty(0, device=device, dtype=torch.long)
        )
        return pos_ids

    def forward(self, g: dgl.DGLGraph, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        dtype = x.dtype
        pos_ids = self.build_pos_ids(g, device)
        need_len = int(pos_ids.max().item()) + 1 if pos_ids.numel() > 0 else 1
        self._maybe_extend(need_len, device=device, dtype=torch.float32)

        pe_slice = self.pe.index_select(dim=0, index=pos_ids).to(dtype=dtype)
        return x + pe_slice

class FaceTransformerLayer(nn.Module):
    """Dense multi-head self-attention over the faces of each solid."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        curve_dim: int = 64,
        last_layer: bool = False,
        act: str = "gelu",
        use_curve_bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.curve_dim = curve_dim
        self.last_layer = last_layer
        
        self.ln_1 = nn.LayerNorm(d_model)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.scale = math.sqrt(d_model // nhead)
        self.use_curve_bias = use_curve_bias
        if use_curve_bias:
            self.curve_proj = nn.Linear(curve_dim, nhead)

        self.attn_dropout = nn.Dropout(attention_dropout)

        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout)
        )
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dim_feedforward, d_model, n_layers=0, act=act, res=False)

        if self.last_layer:
            self.ln_3 = nn.LayerNorm(d_model)
            self.mlp_2 = nn.Linear(d_model, d_model)
        
        
    def forward(
        self,
        g: dgl.DGLGraph,
        node_feat: torch.Tensor,
        edge_feat: torch.Tensor,
    ):
        """
        Args:
            g: Batched DGL graph whose components represent individual solids.
            node_feat: Face features with shape ``[num_faces, d_model]``.
            edge_feat: Edge features with shape ``[num_edges, edge_dim]``.
        """
        num_nodes = node_feat.shape[0]
        if g.num_nodes() != num_nodes:
            raise ValueError(
                f"Graph has {g.num_nodes()} faces but received {num_nodes} features"
            )
        if g.num_edges() != edge_feat.shape[0]:
            raise ValueError(
                f"Graph has {g.num_edges()} edges but received "
                f"{edge_feat.shape[0]} edge features"
            )

        node_feat_res = node_feat
        node_feat = self.ln_1(node_feat)
        src, dst = g.edges()

        q = self.q_proj(node_feat).view(num_nodes, self.nhead, -1)
        k = self.k_proj(node_feat).view(num_nodes, self.nhead, -1)
        v = self.v_proj(node_feat).view(num_nodes, self.nhead, -1)
        curve_bias = self.curve_proj(edge_feat) if self.use_curve_bias else None

        if __import__("os").environ.get("VERIFY_FACE"):
            import torch as _t
            _acc=[]; _ns=0; _es=0
            for _cn,_ce in zip(g.batch_num_nodes().tolist(), g.batch_num_edges().tolist()):
                _ne=_ns+_cn; _ee=_es+_ce
                _sc=_t.einsum("ihd,jhd->hij", q[_ns:_ne], k[_ns:_ne])/self.scale
                if curve_bias is not None:
                    _ls=src[_es:_ee]-_ns; _ld=dst[_es:_ee]-_ns
                    _lb=_mean_multiedge_bias(curve_bias[_es:_ee],_ls,_ld,_cn); _lb=_lb+_lb.transpose(0,1)
                    _sc=_sc+_lb.permute(2,0,1)
                _w=_t.softmax(_sc,dim=-1)
                _acc.append(_t.einsum("hij,jhd->ihd",_w,v[_ns:_ne])); _ns=_ne; _es=_ee
            _loop=_t.cat(_acc,dim=0)
            _bat=_bd_attn(q,k,v,curve_bias,src,dst,g.batch_num_nodes().tolist(),g.batch_num_edges().tolist(),self.scale,self.nhead)
            _d=(_loop-_bat).abs()
            print("[VERIFY_FACE] p=%s train=%s max=%.3e mean=%.3e"%(self.attn_dropout.p,self.training,_d.max().item(),_d.mean().item()), flush=True)
        attended_components = []
        node_start = 0
        edge_start = 0
        _BF = bool(__import__("os").environ.get("BATCHED_FACE"))
        for component_nodes, component_edges in ([] if _BF else zip(
            g.batch_num_nodes().tolist(),
            g.batch_num_edges().tolist(),
        )):
            node_end = node_start + component_nodes
            edge_end = edge_start + component_edges
            component_q = q[node_start:node_end]
            component_k = k[node_start:node_end]
            component_v = v[node_start:node_end]
            scores = (
                torch.einsum("ihd,jhd->hij", component_q, component_k)
                / self.scale
            )
            if curve_bias is not None:
                local_src = src[edge_start:edge_end] - node_start
                local_dst = dst[edge_start:edge_end] - node_start
                local_bias = _mean_multiedge_bias(
                    curve_bias[edge_start:edge_end],
                    local_src,
                    local_dst,
                    component_nodes,
                )
                local_bias = local_bias + local_bias.transpose(0, 1)
                scores = scores + local_bias.permute(2, 0, 1)

            weights = self.attn_dropout(F.softmax(scores, dim=-1))
            attended_components.append(
                torch.einsum("hij,jhd->ihd", weights, component_v)
            )
            node_start = node_end
            edge_start = edge_end

        if _BF:
            node_feat = _bd_attn(q, k, v, curve_bias, src, dst,
                                 g.batch_num_nodes().tolist(), g.batch_num_edges().tolist(),
                                 self.scale, self.nhead, self.attn_dropout).reshape(num_nodes, -1)
        else:
            node_feat = torch.cat(attended_components, dim=0).reshape(num_nodes, -1)
        node_feat = self.out_proj(node_feat)
        node_feat = node_feat_res + node_feat
        node_feat = self.mlp(self.ln_2(node_feat)) + node_feat

        if self.last_layer:
            node_feat = self.mlp_2(self.ln_3(node_feat))
        
        return node_feat

class EdgeTransformerLayer(nn.Module):
    """Dense multi-head self-attention over B-rep edges within each solid."""

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        attention_dropout=0.1,
        use_node_bias=False,
        act="gelu",
        last_layer=False,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.nhead = nhead
        self.use_node_bias = use_node_bias
        self.last_layer = last_layer

        self.ln_1 = nn.LayerNorm(d_model)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.d_k = d_model // nhead
        self.scale = math.sqrt(self.d_k)

        if use_node_bias:
            self.node2head = nn.Linear(d_model, nhead, bias=False)

        self.attn_dropout = nn.Dropout(attention_dropout)

        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout)
        )
        
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dim_feedforward, d_model, n_layers=0, act=act, res=False)

        if self.last_layer:
            self.ln_3 = nn.LayerNorm(d_model)
            self.mlp_2 = nn.Linear(d_model, d_model)

    @torch.no_grad()
    def _lnode_to_eid(self, L: dgl.DGLGraph, g: dgl.DGLGraph, device):
        if dgl.NID in L.ndata:
            mapping = L.ndata[dgl.NID].to(device)
        else:
            mapping = torch.arange(g.number_of_edges(), device=device)
        if mapping.numel() != L.num_nodes():
            raise ValueError("Line-graph node mapping has an unexpected length")
        if mapping.numel() and (mapping.min() < 0 or mapping.max() >= g.num_edges()):
            raise ValueError("Line-graph node mapping contains an invalid edge ID")
        return mapping

    def forward(
        self,
        L: dgl.DGLGraph,
        g: dgl.DGLGraph,
        edge_feat: torch.Tensor,
        node_feat: torch.Tensor,
    ):
        device = edge_feat.device
        if L.num_nodes() != edge_feat.shape[0]:
            raise ValueError(
                f"Line graph has {L.num_nodes()} nodes but received "
                f"{edge_feat.shape[0]} edge features"
            )
        if g.num_edges() != edge_feat.shape[0]:
            raise ValueError(
                f"Graph has {g.num_edges()} edges but received "
                f"{edge_feat.shape[0]} edge features"
            )
        if g.num_nodes() != node_feat.shape[0]:
            raise ValueError(
                f"Graph has {g.num_nodes()} faces but received "
                f"{node_feat.shape[0]} face features"
            )

        graph_edge_counts = g.batch_num_edges().tolist()
        line_node_counts = L.batch_num_nodes().tolist()
        if graph_edge_counts != line_node_counts:
            raise ValueError(
                "Each line-graph component must contain one node per B-rep edge"
            )

        edge_feat_res = edge_feat
        edge_feat = self.ln_1(edge_feat)
        num_edges = edge_feat.shape[0]

        q = self.q_proj(edge_feat).view(num_edges, self.nhead, self.d_k)
        k = self.k_proj(edge_feat).view(num_edges, self.nhead, self.d_k)
        v = self.v_proj(edge_feat).view(num_edges, self.nhead, self.d_k)

        line_bias = None
        if self.use_node_bias:
            l_src, l_dst = L.edges()
            lnode2eid = self._lnode_to_eid(L, g, device)
            src_eid = lnode2eid[l_src]
            dst_eid = lnode2eid[l_dst]
            g_src, g_dst = g.edges()
            u_1 = g_src[src_eid]
            v_1 = g_dst[src_eid]
            x_1 = g_src[dst_eid]
            w_1 = g_dst[dst_eid]
            shares_face = (u_1 == x_1) | (u_1 == w_1) | (v_1 == x_1) | (v_1 == w_1)
            if not bool(shares_face.all()):
                raise ValueError(
                    "Line graph contains an edge between B-rep edges that do not "
                    "share a face"
                )
            shared_nid = torch.where(
                u_1 == x_1,
                u_1,
                torch.where(u_1 == w_1, u_1, torch.where(v_1 == x_1, v_1, w_1)),
            )
            line_bias = self.node2head(node_feat)[shared_nid]

        l_src, l_dst = L.edges()
        attended_components = []
        edge_start = 0
        line_edge_start = 0
        _BE = bool(__import__("os").environ.get("BATCHED_EDGE")) and bool(line_node_counts) and (max(line_node_counts) <= int(__import__("os").environ.get("EDGE_CAP", "2000")))
        for component_edges, component_line_edges in ([] if _BE else zip(
            line_node_counts,
            L.batch_num_edges().tolist(),
        )):
            edge_end = edge_start + component_edges
            line_edge_end = line_edge_start + component_line_edges
            component_q = q[edge_start:edge_end]
            component_k = k[edge_start:edge_end]
            component_v = v[edge_start:edge_end]
            scores = (
                torch.einsum("ihd,jhd->hij", component_q, component_k)
                / self.scale
            )
            if line_bias is not None:
                local_src = l_src[line_edge_start:line_edge_end] - edge_start
                local_dst = l_dst[line_edge_start:line_edge_end] - edge_start
                local_bias = _mean_multiedge_bias(
                    line_bias[line_edge_start:line_edge_end],
                    local_src,
                    local_dst,
                    component_edges,
                )
                local_bias = local_bias + local_bias.transpose(0, 1)
                scores = scores + local_bias.permute(2, 0, 1)

            weights = self.attn_dropout(F.softmax(scores, dim=-1))
            attended_components.append(
                torch.einsum("hij,jhd->ihd", weights, component_v)
            )
            edge_start = edge_end
            line_edge_start = line_edge_end

        if _BE:
            edge_feat = _bd_attn(q, k, v, line_bias, l_src, l_dst,
                                 line_node_counts, L.batch_num_edges().tolist(),
                                 self.scale, self.nhead, self.attn_dropout).reshape(num_edges, -1)
        else:
            edge_feat = torch.cat(attended_components, dim=0).reshape(num_edges, -1)
        edge_feat = self.out_proj(edge_feat)

        edge_feat = edge_feat + edge_feat_res
        edge_feat = self.mlp(self.ln_2(edge_feat)) + edge_feat

        if self.last_layer:
            edge_feat = self.mlp_2(self.ln_3(edge_feat))
        return edge_feat

class DualAwareBlock(nn.Module):
    """Update face and edge streams with coupled transformer layers."""

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=512,
        dropout=0.3,
        attention_dropout=0.1,
        use_node_bias=False,
        use_edge_bias=False,
        act="gelu",
        last_layer=False,
    ):
        super().__init__()
        self.edge_layer = EdgeTransformerLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
            dropout=dropout,
            attention_dropout=attention_dropout,
            use_node_bias=use_node_bias,
            act=act,
            last_layer=last_layer,
        )
        self.node_layer = FaceTransformerLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout,
            attention_dropout=attention_dropout,
            curve_dim=d_model,
            use_curve_bias=use_edge_bias,
            act=act,
            last_layer=last_layer,
        )

    def forward(self, g, L, node_feat, edge_feat):
        n_feat_new = self.node_layer(g, node_feat, edge_feat)
        e_feat_new = self.edge_layer(
            L=L,
            g=g,
            edge_feat=edge_feat,
            node_feat=node_feat,
        )
        return n_feat_new, e_feat_new

class DualCurveEncoder(nn.Module):
    """Encode the Bezier primitives of each B-rep edge."""

    def __init__(
        self,
        input_dim,
        curve_emb_dim,
        dropout,
        hidden_dim,
        n_layers,
        n_heads,
        act='gelu',
        norm_first=False,
        use_layer_norm=True,
    ):
        super().__init__()
        self.bezier_encoder = BezierEncoderMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            act=act,
            use_layer_norm=use_layer_norm,
        )
        self.transformer_encoder = TransformerEncoderBlock(
            input_dim=hidden_dim,
            c_hidden=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            batch_first=True,
            act=act,
            norm_first=norm_first,
        )

        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos = PositionalEncoding(hidden_dim, dropout)
        self.out_proj = nn.Linear(hidden_dim, curve_emb_dim)
    def forward(self, control_pts, mask):
        if control_pts.ndim != 4:
            raise ValueError(
                "Curve control points must have shape "
                "[num_edges, num_primitives, num_points, channels]"
            )
        if mask.shape != control_pts.shape[:2]:
            raise ValueError("Curve padding mask must match the first two input axes")
        mask = mask.to(dtype=torch.bool)
        x = torch.flatten(control_pts, start_dim=2)
        curve_emb = self.bezier_encoder(x.view(-1, x.shape[-1])).view(x.shape[0], x.shape[1], -1)

        curve_emb = torch.cat(
            [self.class_token.repeat(curve_emb.shape[0], 1, 1), curve_emb],
            dim=1,
        )
        curve_emb = self.pos(curve_emb)
        mask = torch.cat(
            [torch.ones(mask.shape[0], 1, dtype=torch.bool, device=mask.device), mask],
            dim=1,
        )

        src_mask = torch.logical_not(mask)
        curve_emb = self.transformer_encoder(curve_emb, src_mask)
        curve_emb = curve_emb[:, 0]
        curve_emb = self.out_proj(curve_emb)
        return curve_emb

class DualSurfaceEncoder(nn.Module):
    """Encode the Bezier primitives and visibility data of each face."""

    def __init__(
        self,
        input_dim,
        surface_emb_dim,
        dropout,
        hidden_dim,
        n_layers,
        n_heads,
        use_class_token=False,
        act='gelu',
        norm_first=False,
        use_layer_norm=False,
    ):
        super().__init__()
        self.bezier_encoder = BezierEncoderMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            act=act,
            use_layer_norm=use_layer_norm,
        )

        self.transformer_encoder = TransformerEncoderBlock(
            input_dim=hidden_dim,
            c_hidden=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            batch_first=True,
            act=act,
            norm_first=norm_first,
        )

        self.use_class_token = use_class_token
        if self.use_class_token:
            self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos = PositionalEncoding(hidden_dim, dropout)
        self.out_proj = nn.Linear(hidden_dim, surface_emb_dim)
    def forward(self, control_pts, tri_normal, in_mask, padding_mask):
        if control_pts.ndim != 4:
            raise ValueError(
                "Surface control points must have shape "
                "[num_faces, num_primitives, num_points, channels]"
            )
        expected_shape = control_pts.shape[:2]
        if tri_normal.shape[:2] != expected_shape:
            raise ValueError("Triangle normals must align with surface primitives")
        if in_mask.shape != expected_shape or padding_mask.shape != expected_shape:
            raise ValueError("Surface masks must match the first two input axes")
        mask = padding_mask.to(dtype=torch.bool)
        if not bool(mask.any(dim=1).all()):
            raise ValueError("Every face must contain at least one valid primitive")

        B, L = expected_shape
        x = torch.cat(
            [torch.flatten(control_pts, start_dim=2), tri_normal, in_mask.unsqueeze(-1)],
            dim=-1,
        )
        surface_emb = self.bezier_encoder(x.view(-1, x.shape[-1])).view(B, L, -1)
        
        if self.use_class_token:
            surface_emb = torch.cat(
                [self.class_token.repeat(surface_emb.shape[0], 1, 1), surface_emb],
                dim=1,
            )
            surface_emb = self.pos(surface_emb)
            mask = torch.cat(
                [
                    torch.ones(mask.shape[0], 1, dtype=torch.bool, device=mask.device),
                    mask,
                ],
                dim=1,
            )
        else:
            surface_emb = self.pos(surface_emb)
        
        src_mask = torch.logical_not(mask)
        surface_emb = self.transformer_encoder(surface_emb, src_mask)

        if self.use_class_token:
            feature = surface_emb[:, 0]
        else:
            surface_emb = surface_emb.masked_fill(torch.logical_not(mask.unsqueeze(-1)), 0)
            count = mask.sum(dim=1).clamp_min(1)
            feature = surface_emb.sum(dim = 1) / count.unsqueeze(-1)
        feature = self.out_proj(feature)
        return feature

class DualGraphEncoder(nn.Module):
    """Jointly encode face and edge features and pool a solid embedding."""

    def __init__(
        self,
        input_surface_dim: int,
        input_edge_dim: int, 
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.3,
        attention_dropout: float = 0.1,
        add_positional_encoding: bool = False,
        use_node_bias: bool = False,
        use_edge_bias: bool = False,
        act='gelu',
        return_edge_feat: bool = False,
        add_edge_to_graph: bool = False,
    ):
        super().__init__()
        
        self.input_surface_dim = input_surface_dim
        self.input_edge_dim = input_edge_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.add_positional_encoding = add_positional_encoding
        self.return_edge_feat = return_edge_feat
        self.add_edge_to_graph = add_edge_to_graph
        self.node_input_proj = nn.Linear(input_surface_dim, hidden_dim)
        self.edge_input_proj = nn.Linear(input_edge_dim, hidden_dim)

        if add_positional_encoding:
            self.node_pos_encoder = SubgraphPE(hidden_dim)
            self.edge_pos_encoder = SubgraphPE(hidden_dim)
        
        # The final block applies the output transformation used by the paper.
        self.transformer_layers = nn.ModuleList([
            DualAwareBlock(
                d_model=hidden_dim,
                nhead=num_heads, 
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                attention_dropout=attention_dropout,
                use_node_bias=use_node_bias,
                use_edge_bias=use_edge_bias,
                act=act,
                last_layer=(i == num_layers - 1)
            ) for i in range(num_layers)
        ])
        
        self.node_output_proj = nn.Linear(hidden_dim, input_surface_dim)
        if return_edge_feat:
            self.edge_output_proj = nn.Linear(hidden_dim, input_edge_dim)
        
        graph_input_dim = hidden_dim + (hidden_dim if add_edge_to_graph else 0)
        self.graph_proj = nn.Linear(graph_input_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        batched_graph: dgl.DGLGraph,
        node_feat: torch.Tensor,
        edge_feat: torch.Tensor,
        line_graph: dgl.DGLGraph,
    ):
        if node_feat.ndim != 2 or edge_feat.ndim != 2:
            raise ValueError("Graph encoder inputs must be rank-2 feature tensors")
        if batched_graph.num_nodes() != node_feat.shape[0]:
            raise ValueError("Face feature count does not match graph node count")
        if batched_graph.num_edges() != edge_feat.shape[0]:
            raise ValueError("Edge feature count does not match graph edge count")
        node_counts = batched_graph.batch_num_nodes()
        edge_counts = batched_graph.batch_num_edges()
        if node_counts.numel() == 0 or bool((node_counts <= 0).any()):
            raise ValueError("Every solid must contain at least one face")
        if edge_counts.numel() == 0 or bool((edge_counts <= 0).any()):
            raise ValueError("Every solid must contain at least one edge")
        node_feat_proj = self.node_input_proj(node_feat)
        edge_feat_proj = self.edge_input_proj(edge_feat)
        g = batched_graph
        L = line_graph

        if L.num_nodes() != edge_feat_proj.shape[0]:
            raise ValueError(
                "Line-graph node count must match the number of B-rep edge features: "
                f"line_graph={L.num_nodes()}, edge_features={edge_feat_proj.shape[0]}"
            )

        if self.add_positional_encoding:
            node_feat_proj = self.node_pos_encoder(g, node_feat_proj)
            edge_feat_proj = self.edge_pos_encoder(L, edge_feat_proj)
        
        node_feat_proj = self.dropout(node_feat_proj)
        edge_feat_proj = self.dropout(edge_feat_proj)
        
        x, e = node_feat_proj, edge_feat_proj
        for block in self.transformer_layers:
            x, e = block(g, L, x, e)
        
        node_embeddings = self.node_output_proj(x)
        if self.return_edge_feat:
            edge_embeddings = self.edge_output_proj(e)
        node_counts = batched_graph.batch_num_nodes().to(x.device)
        graph_features = dgl.ops.segment_reduce(
            node_counts,
            x,
            reducer="mean",
        )
        if self.add_edge_to_graph:
            edge_counts = batched_graph.batch_num_edges().to(e.device)
            edge_features = dgl.ops.segment_reduce(
                edge_counts,
                e,
                reducer="mean",
            )
            graph_features = torch.cat([graph_features, edge_features], dim=-1)
        final_graph_embeddings = self.graph_proj(graph_features)
        if self.return_edge_feat:
            return node_embeddings, final_graph_embeddings, edge_embeddings
        return node_embeddings, final_graph_embeddings
