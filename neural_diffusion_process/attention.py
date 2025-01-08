import torch 
import torch.nn as nn
import numpy as np
from einops import rearrange, reduce
import math

#compared to NDP, we will just use Torch Implementation of MultiheadAttention
from torch.nn.functional import scaled_dot_product_attention
from torch.nn import MultiheadAttention
from transformer_utils import TimestepEmbedder

def scaled_dot_product_Attention(q: torch.tensor, k: torch.tensor, v:torch.tensor, 
                                mask: torch.tensor = None, dropout: float = 0.0):
    """
    Calculates the scaled dot product attention,

    q, k, v must have matching leading dimension
    k, v must have matching penultimate dimension

    mask has different shape depedning on its type (padding or look ahead). 

    Returns:
    output, attention_weights
    """

    matmul_qk = torch.einsum('...qd,...kd->...qk', q, k)

    #scale matmul_qk
    depth = k.size(-1) * 1.0
    scaled_attention_logits = matmul_qk / depth**0.5

    #add the mask to the scaled tensor
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    
    #softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1
    attention_weights = torch.nn.functional.softmax(scaled_attention_logits, dim=-1) # (..., seq_len_q, seq_len_k)

    output = torch.einsum('...qk,...kv->...qv', attention_weights, v) # (..., seq_len_q, depth_v)
    return output


class MHA(torch.nn.Module):

    def __init__(self, in_dim, out_dim, num_heads, sparse:bool = False):
        super(MHA, self).__init__()
        in_dim = in_dim
        d_model = out_dim
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % num_heads == 0

        self.depth = d_model // num_heads

        if not sparse:
            self.attention = scaled_dot_product_Attention
        self.q_linear = torch.nn.Linear(in_dim, d_model)
        self.k_linear = torch.nn.Linear(in_dim, d_model)
        self.v_linear = torch.nn.Linear(in_dim, d_model)#

        self.output_lin = torch.nn.Linear(d_model, d_model)
    def forward(self, v, k, q, mask=None):
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        rearrange_arg = "... seq_len (num_heads depth) -> ... num_heads seq_len depth"

        q = rearrange(q, rearrange_arg, num_heads = self.num_heads, depth = self.depth)
        k = rearrange(k, rearrange_arg, num_heads = self.num_heads, depth = self.depth)
        v = rearrange(v, rearrange_arg, num_heads = self.num_heads, depth = self.depth)

        if mask is not None:
            mask_seq_v = mask[..., :, None]
            mask_seq_k = mask[..., None, :]
            mask = mask_seq_v * mask_seq_k
            mask = torch.where(mask == 0, mask, torch.ones_like(mask))
            mask = mask[..., None, :, :]

        scaled_attention = self.attention(q, k, v, mask)
        scaled_attention = rearrange(scaled_attention, "... num_heads seq_len depth -> ... seq_len (num_heads depth)")
        output = self.output_lin(scaled_attention)

        return output 








class AttentionBlock(nn.Module):

    def __init__(self, dim, heads = 8, dim_head = 64, sparse: bool = False):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim = dim
        self.sparse = sparse

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.linear_t = nn.Linear(dim, dim)

        self.MHA = MHA(in_dim=dim, out_dim=dim*2, num_heads=heads, sparse=sparse)
        #doesnt support sparse
        # self.MHA = MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)
    def forward(self, s, t):
        """
        s: [B, N, D]
        t: [b, D]
        """
        t = t.unsqueeze(1)
        t = self.linear_t(t)

        y = s + t
        y_att_d = self.MHA(y, y, y)
        y = y_att_d
        residual, skip = torch.chunk(y, 2, dim=-1)
        resiudal = torch.nn.functional.gelu(residual)
        skip = torch.nn.functional.gelu(skip)
        return ((s + residual)/2**0.5), skip


class AttentionModel(nn.Module):
    
    def __init__(self, cfg, out_dim: int=1, input_dim: int=2, 
                init_zero=True):
        
        super().__init__()
        self.cfg = cfg
        hidden_dim = self.cfg.hidden_dim
        num_heads = self.cfg.num_heads
        n_layers = cfg.n_layers   
        sparse = cfg.sparse_attention

        self.layers = nn.ModuleList([AttentionBlock(hidden_dim, num_heads, sparse=sparse
            ) for _ in range(n_layers)])
        self.eps_layer = nn.Linear(hidden_dim, hidden_dim)
        self.fin_layer = nn.Linear(hidden_dim, out_dim)
        if init_zero:
            self.fin_layer.weight.data.zero_()
        self.t_embedding = TimestepEmbedder(hidden_dim)
        self.n_layers = n_layers

        # if init_zero:
        #     nn.init.constant_(self.output_layer.weight, 0)
        #     nn.init.constant_(self.output_layer.bias, 0)

        self.encoder = nn.Linear(input_dim + out_dim, hidden_dim)
    def forward(self, x, y, t, mask=None):
        """
        Computes the additive noise that was added to `y_0` to obtian `y_t`
        based on x_t, y_t and t

        "x: [batch_size, num_points, input_dim]",
        "y: [batch_size, num_points, output_dim]",
        "t: [batch_size]",
        "mask: [batch_size, num_points] if mask is not None",
        "return: [batch_size, num_points, output_dim]",
        """
        assert x.shape[0] == y.shape[0] == t.shape[0]
        assert x.shape[2] == y.shape[2]
        if mask is not None:
            assert mask.shape[0] == x.shape[0] and mask.shape[1] == x.shape[1]

        del mask
        x = torch.concat([x, y], dim=-1).to(torch.float32)
        x = self.encoder(x)
        t = self.t_embedding(t)
        skip = None

        for layer in self.layers:
            x, skip_connection = layer(x, t)
            skip = skip_connection if skip is None else skip + skip_connection

        assert x.shape[-1] == self.cfg.hidden_dim
        assert x.shape[0] == y.shape[0] and x.shape[1] == y.shape[1]

        eps = skip/math.sqrt(self.n_layers)
        eps = nn.functional.gelu(self.eps_layer(eps))


        eps = self.fin_layer(eps)
        return eps