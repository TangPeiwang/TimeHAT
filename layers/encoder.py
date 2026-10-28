from typing import Callable, Optional, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from math import sqrt

def get_activation_fn(activation):
    if callable(activation): return activation()
    elif activation.lower() == "relu": return nn.ReLU()
    elif activation.lower() == "gelu": return nn.GELU()
    raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or a callable')


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model_per_head: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = d_model_per_head
        
        # Generate frequency angle: 1 / (base ^ (2i / d))
        inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        
        # Precompute rotation matrix for maximum possible length (cached)
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len
        # t corresponds to the absolute index of time steps: [0, 1, 2, ..., seq_len-1]
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        
        # freqs shape: [seq_len, dim / 2]
        freqs = torch.outer(t, self.inv_freq)
        
        # Duplicate to match feature dimension: [seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_dim=2):
        """
        x: [B, H, L, D]
        """
        seq_len = x.shape[seq_dim]
        
        # Dynamically extend cache if current sequence length exceeds the cached length
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
            
        # Slice to current required length: [1, 1, L, D]
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """
    Apply rotary positional embeddings to Q and K
    q, k: [B, H, L, D]
    cos, sin: [1, 1, L, D]
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class FullAttention(nn.Module):
    def __init__(self, configs, attn_dropout=0.1):
        super().__init__()
        self.output_attention = configs.output_attention
        self.training = configs.is_training
        self.dropout_p = attn_dropout
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.rope = RotaryPositionalEmbedding(configs.d_model // configs.n_heads)

    def forward(self, queries, keys, values, attn_mask=None):
        """
        attn_mask: [B, 1, L, L] boolean tensor. True means visible, False means masked.
        """
        cos, sin = self.rope(queries)
        queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin)

        if self.output_attention:
            B, H, L, E = queries.shape
            scale = 1.0 / sqrt(E)
            scores = torch.einsum("bhle,bhde->bhld", queries, keys) * scale
            
            if attn_mask is not None:
                # Mask out False positions with -inf
                scores = scores.masked_fill(~attn_mask, -float('inf'))
                
            A = self.attn_dropout(torch.softmax(scores, dim=-1))
            V = torch.einsum("bhls,bhsd->bhld", A, values)
            return V.contiguous(), A
            
        else:
            # Use PyTorch native SDPA for optimization
            V = F.scaled_dot_product_attention(
                queries, keys, values, 
                attn_mask=attn_mask,
                dropout_p=self.dropout_p if self.training else 0.0
            )
            return V, None


class Mix_ScaledDotProductAttention(nn.Module):
    def __init__(self, configs, n_heads: int, attn_dropout: float = 0., 
                 forward_ratio: float = 0.5, causal: bool = True, max_len: int = 4096):
        super().__init__()
        self.n_heads = n_heads
        self.n_forward = int(n_heads * forward_ratio)
        self.n_backward = n_heads - self.n_forward
        self.causal = causal
        self.output_attention = configs.output_attention
        self.training = configs.is_training
        self.dropout_p = attn_dropout
        self.attn_dropout = nn.Dropout(attn_dropout)
        
        self.rope = RotaryPositionalEmbedding(configs.d_model // configs.n_heads)
        
        # Cache basic triangular masks to save memory allocation time during forward pass
        if self.causal:
            mask_template = torch.ones(max_len, max_len, dtype=torch.bool)
            self.register_buffer('L_mask_template', torch.tril(mask_template))
            self.register_buffer('U_mask_template', torch.triu(mask_template))

    def _compute_attention(self, q, k, v, mask=None, is_causal=False):
        """Custom computation used only when outputting attention weights"""
        B, H, L, E = q.shape
        scale = 1.0 / (E ** 0.5)
        attn_scores = torch.einsum("bhle,bhde->bhld", q, k) * scale
        
        if is_causal and mask is None:
            mask = self.L_mask_template[:L, :L]
            
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, -float('inf'))
            
        attn_weights = self.attn_dropout(torch.softmax(attn_scores, dim=-1))
        output = torch.einsum("bhls,bhsd->bhld", attn_weights, v)
        return output, attn_weights

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                forecast_mask: Optional[torch.Tensor] = None):
        """
        forecast_mask: [B, L] boolean tensor. True indicates the position is a [FORECAST] token.
        """
        B, H, L, E = q.shape
        
        cos, sin = self.rope(q)
        q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin)
        
        q_forward, q_backward = q_rope.split([self.n_forward, self.n_backward], dim=1)
        k_forward, k_backward = k_rope.split([self.n_forward, self.n_backward], dim=1)
        v_forward, v_backward = v.split([self.n_forward, self.n_backward], dim=1)

        b_mask = None
        
        # Dynamically construct mixed mask for the backward head
        if self.causal:
            L_mask = self.L_mask_template[:L, :L].view(1, 1, L, L)
            U_mask = self.U_mask_template[:L, :L].view(1, 1, L, L)
            
            if forecast_mask is not None:
                # Reshape for broadcasting: Query [B, 1, L, 1] | Key [B, 1, 1, L]
                F_i = forecast_mask.view(B, 1, L, 1) 
                F_j = forecast_mask.view(B, 1, 1, L) 
                
                # Case A: If Query is historical (~F_i), use U_mask but mask all forecast tokens (F_j).
                # (Cannot look at Forecast tokens)
                H_mask = U_mask & (~F_j)
                
                # Case B: If Query is Forecast (F_i), use L_mask.
                # (Forecast tokens look at history for extrapolation)
                b_mask = torch.where(F_i, L_mask, H_mask) # Shape: [B, 1, L, L]
            else:
                b_mask = U_mask

        if self.output_attention:
            output_forward, weights_forward = self._compute_attention(
                q_forward, k_forward, v_forward, is_causal=self.causal
            )
            output_backward, weights_backward = self._compute_attention(
                q_backward, k_backward, v_backward, mask=b_mask
            )
            output = torch.cat([output_forward, output_backward], dim=1)
            attn_weights = torch.cat([weights_forward, weights_backward], dim=1)
            return output, attn_weights
            
        else:
            # Forward head uses causal=True directly for FlashAttention optimization
            out_forward = F.scaled_dot_product_attention(
                q_forward, k_forward, v_forward, 
                is_causal=self.causal,
                dropout_p=self.dropout_p if self.training else 0.0
            )
            
            out_backward = F.scaled_dot_product_attention(
                q_backward, k_backward, v_backward, 
                attn_mask=b_mask,
                dropout_p=self.dropout_p if self.training else 0.0
            )
            
            output = torch.cat([out_forward, out_backward], dim=1)
            return output, None
        
    
class Mix_MultiheadAttention(nn.Module):
    def __init__(self, configs, layer_type: str, d_model: int, n_heads: int,
                 d_k: Optional[int] = None, d_v: Optional[int] = None,
                 attn_dropout: float = 0., proj_dropout: float = 0., qkv_bias: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads if d_k is None else d_k
        self.d_v = d_model // n_heads if d_v is None else d_v
        self.layer_type = layer_type

        self.W_Q = nn.Linear(d_model, self.d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, self.d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, self.d_v * n_heads, bias=qkv_bias)

        if layer_type == 'mix':
            self.sdp_attn = Mix_ScaledDotProductAttention(
                configs, n_heads=n_heads, attn_dropout=attn_dropout, forward_ratio=0.5, causal=True
            )
        elif layer_type == 'bi':
            self.sdp_attn = FullAttention(configs, attn_dropout=attn_dropout)
        else:
            raise ValueError(f"Unknown layer_type: {layer_type}")

        self.to_out = nn.Sequential(
            nn.Linear(n_heads * self.d_v, d_model),
            nn.Dropout(proj_dropout)
        )

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                forecast_mask: Optional[torch.Tensor] = None):
        """
        forecast_mask: [B, L] boolean tensor. True indicates the position is a [FORECAST] token.
        """
        bs = Q.size(0)

        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2) 
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)

        L = q_s.size(2)

        if self.layer_type == 'mix':
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, forecast_mask=forecast_mask)
            
        else: # layer_type == 'bi' (FullAttention)
            attn_mask = None
            if forecast_mask is not None:
                # Dynamically construct unified Prefix-LM mask
                F_i = forecast_mask.view(bs, 1, L, 1)  # Query identity [B, 1, L, 1]
                F_j = forecast_mask.view(bs, 1, 1, L)  # Key identity [B, 1, 1, L]

                # Base causal lower triangular mask (for smooth extrapolation among future tokens)
                causal_mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=Q.device)).view(1, 1, L, L)

                # Mask logic:
                # 1. Everyone can attend to history (~F_j)
                # 2. History cannot attend to forecast (naturally blocked by ~F_j)
                # 3. Forecast attends to forecast adhering to causality (F_i & F_j & causal_mask)
                attn_mask = (~F_j) | (F_i & F_j & causal_mask)

            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, attn_mask=attn_mask)

        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        output = self.to_out(output)
        
        return output, attn_weights
    

class EncoderLayer(nn.Module):
    def __init__(self, configs, layer_type, bias=True, activation="gelu"):
        super().__init__()
        assert not configs.d_model % configs.n_heads, "d_model must be divisible by n_heads"
        d_k = configs.d_model // configs.n_heads
        d_v = configs.d_model // configs.n_heads
        
        self.self_attn = Mix_MultiheadAttention(
            configs, layer_type, configs.d_model, configs.n_heads, 
            d_k=d_k, d_v=d_v, attn_dropout=configs.dropout, proj_dropout=configs.dropout
        )
        
        self.dropout_attn = nn.Dropout(configs.dropout)
        self.norm_attn = nn.LayerNorm(configs.d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_ff, bias=bias),
            get_activation_fn(activation),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_ff, configs.d_model, bias=bias)
        )

        self.dropout_ffn = nn.Dropout(configs.dropout)
        self.norm_ffn = nn.LayerNorm(configs.d_model)

    def forward(self, src: Tensor, forecast_mask: Optional[Tensor] = None):
        """
        Pass through forecast_mask
        """
        # Multi-Head attention sublayer
        src2, attn = self.self_attn(src, src, src, forecast_mask=forecast_mask)
        src = src + self.dropout_attn(src2)
        src = self.norm_attn(src) 
        
        # Feed-forward sublayer
        src2 = self.ff(src)
        src = src + self.dropout_ffn(src2)
        src = self.norm_ffn(src)

        return src, attn