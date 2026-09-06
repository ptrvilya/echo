import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def get_timestep_embedding(embed_dim, timesteps, device):
    """
    Timestep embedding function. Note that this should work just as well for
    continuous values as for discrete values.
    """
    assert len(timesteps.shape) == 1, f"Timesteps shape: {timesteps.shape}"
    half_dim = embed_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.from_numpy(np.exp(np.arange(0, half_dim) * -emb)).float().to(device)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embed_dim % 2 == 1:  # zero pad
        emb = nn.functional.pad(emb, (0, 1), "constant", 0)
    assert emb.shape == torch.Size([timesteps.shape[0], embed_dim])
    return emb


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0, use_rope=False):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.dim = dim
        self.use_rope = use_rope

        self.kv = nn.Linear(dim, dim * 2, bias=False)
        self.q = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # QK-norm: parameter-free so checkpoints trained without it still load
        self.q_norm = nn.RMSNorm(self.dim_head, elementwise_affine=False)
        self.k_norm = nn.RMSNorm(self.dim_head, elementwise_affine=False)

        if self.use_rope:
            self.rotary_emb = RotaryEmbedding(self.dim_head)

    def forward(self, x, cond):
        # x is used as query, cond is used as key and value
        B, T, C = x.shape
        q = rearrange(self.q(x), "b t (h d) -> b h t d", h=self.num_heads)
        k, v = rearrange(
            self.kv(cond), "b t (kv h d) -> kv b h t d", kv=2, h=self.num_heads
        )
        q, k = self.q_norm(q), self.k_norm(k)

        if self.use_rope:
            q = self.rotary_emb.rotate_queries_or_keys(q, seq_dim=-2)
            k = self.rotary_emb.rotate_queries_or_keys(k, seq_dim=-2)

        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        x = self.dropout(x)
        x = rearrange(x, "b h t d -> b t (h d)", h=self.num_heads)
        x = self.proj(x)
        return x


class RoPEMultiheadAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0, use_rope=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_rope = use_rope

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # QK-norm: parameter-free so checkpoints trained without it still load
        self.q_norm = nn.RMSNorm(self.head_dim, elementwise_affine=False)
        self.k_norm = nn.RMSNorm(self.head_dim, elementwise_affine=False)

        if self.use_rope:
            self.rotary_emb = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = rearrange(
            self.qkv(x), "b t (qkv h d) -> qkv b h t d", qkv=3, h=self.num_heads
        )
        q, k = self.q_norm(q), self.k_norm(k)

        if self.use_rope:
            q = self.rotary_emb.rotate_queries_or_keys(q, seq_dim=-2)
            k = self.rotary_emb.rotate_queries_or_keys(k, seq_dim=-2)

        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        x = self.dropout(x)
        x = rearrange(x, "b h t d -> b t (h d)", h=self.num_heads)
        x = self.proj(x)
        return x


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self,
        dim_input,
        num_heads,
        dim_hidden,
        dropout_attn=0.0,
        dropout_mlp=0.0,
        use_cross_attn=False,
        use_rope=False,
    ):
        super().__init__()
        self.use_cross_attn = use_cross_attn
        self.use_rope = use_rope

        self.norm1 = nn.LayerNorm(dim_input, elementwise_affine=False, eps=1e-6)
        self.attn = RoPEMultiheadAttention(
            dim_input,
            num_heads,
            dropout=dropout_attn,
            use_rope=use_rope,
        )
        if self.use_cross_attn:
            self.norm_cross = nn.LayerNorm(
                dim_input, elementwise_affine=False, eps=1e-6
            )
            self.cross_attn = MultiHeadCrossAttention(
                dim_input, num_heads, dropout=dropout_attn, use_rope=use_rope
            )

        self.norm2 = nn.LayerNorm(dim_input, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout_mlp),
            nn.Linear(dim_hidden, dim_input),
            nn.Dropout(dropout_mlp),
        )

        adaLN_dim = 9 * dim_input if self.use_cross_attn else 6 * dim_input
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim_input, adaLN_dim, bias=True)
        )

    def forward(self, x, c, cross_attn_cond=None):
        if self.use_cross_attn:
            (
                shift_msa,
                scale_msa,
                gate_msa,
                shift_ca,
                scale_ca,
                gate_ca,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            ) = self.adaLN_modulation(c).chunk(9, dim=-1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(c).chunk(6, dim=-1)
            )

        # Self-attention block
        qkv = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(qkv)

        # Cross-attention block
        if self.use_cross_attn and cross_attn_cond is not None:
            x = x + gate_ca.unsqueeze(1) * self.cross_attn(
                modulate(self.norm_cross(x), shift_ca, scale_ca), cross_attn_cond
            )

        # MLP block
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x
