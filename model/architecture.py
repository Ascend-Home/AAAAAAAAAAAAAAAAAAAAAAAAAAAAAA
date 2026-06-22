"""
MAX-AI Architecture
1.5T parameter MoE transformer with MLA, MoE, MTP, YaRN RoPE.
DeepSeek-V3 / Claude-class architecture.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Tuple


# =====================================================
# CONFIG
# =====================================================
@dataclass
class MaxConfig:
    # Vocab / embedding
    vocab_size: int = 256000

    # Model dimensions
    dim: int = 16384
    n_layers: int = 128
    n_heads: int = 128
    n_kv_heads: int = 16
    head_dim: int = 128

    # MoE
    n_routed_experts: int = 256
    n_shared_experts: int = 2
    n_activated_experts: int = 8
    moe_intermediate_size: int = 2048
    dense_intermediate_size: int = 53248
    first_k_dense_layers: int = 3
    n_expert_groups: int = 8
    n_limited_groups: int = 4
    score_func: str = "sigmoid"   # "sigmoid" or "softmax"
    route_scale: float = 2.5
    aux_loss_alpha: float = 0.001

    # Multi-Latent Attention (MLA)
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128

    # Context / RoPE / YaRN
    max_seq_len: int = 1_048_576
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40.0
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0

    # Training
    norm_eps: float = 1e-6
    dropout: float = 0.0
    initializer_range: float = 0.02

    # Multi-Token Prediction
    n_mtp_layers: int = 3
    mtp_loss_weight: float = 0.3


# =====================================================
# RMSNORM
# =====================================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * norm).to(dtype)


# =====================================================
# YARN ROPE — million-token context extension
# =====================================================
def _yarn_find_correction_dim(num_rot, dim, base, max_pos):
    return (dim * math.log(max_pos / (num_rot * 2 * math.pi))) / (2 * math.log(base))


def _yarn_find_correction_range(low_rot, high_rot, dim, base, max_pos):
    low = math.floor(_yarn_find_correction_dim(low_rot, dim, base, max_pos))
    high = math.ceil(_yarn_find_correction_dim(high_rot, dim, base, max_pos))
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_mask(min_, max_, dim):
    if min_ == max_:
        max_ += 0.001
    linear = (torch.arange(dim, dtype=torch.float32) - min_) / (max_ - min_)
    return torch.clamp(linear, 0, 1)


def _yarn_get_mscale(scale=1.0, mscale=1.0):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def precompute_freqs_cis(args: MaxConfig) -> torch.Tensor:
    dim = args.qk_rope_head_dim
    base = args.rope_theta
    factor = args.rope_factor
    max_pos = args.max_seq_len
    orig_max = args.original_seq_len

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

    if max_pos > orig_max:
        low, high = _yarn_find_correction_range(
            args.beta_fast, args.beta_slow, dim, base, orig_max
        )
        smooth = 1.0 - _yarn_linear_ramp_mask(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(max_pos)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # freqs_cis shape: (seq_len, dim//2) -> broadcast to (1, seq_len, 1, dim//2)
    freqs_cis = freqs_cis.view(1, x_c.size(1), 1, x_c.size(-1))
    y = torch.view_as_real(x_c * freqs_cis).flatten(3)
    return y.to(dtype)


# =====================================================
# MULTI-LATENT ATTENTION (MLA) — DeepSeek-V3 style
# =====================================================
class MLA(nn.Module):
    def __init__(self, args: MaxConfig):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        # Query: down-project then up-project
        self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(self.q_lora_rank, args.norm_eps)
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False)

        # Key/Value: shared low-rank latent + RoPE part
        self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_norm = RMSNorm(self.kv_lora_rank, args.norm_eps)
        self.wkv_b = nn.Linear(
            self.kv_lora_rank,
            self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim, bias=False)

        self.softmax_scale = self.qk_head_dim ** -0.5
        if args.max_seq_len > args.original_seq_len:
            mscale = _yarn_get_mscale(args.rope_factor, args.mscale)
            self.softmax_scale = self.softmax_scale * mscale * mscale

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        # Query path
        q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(bsz, seqlen, self.n_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)

        # KV latent path
        kv = self.wkv_a(x)
        kv_latent, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        # k_pe is shared across heads → shape (bsz, seqlen, 1, qk_rope_head_dim)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)

        kv_up = self.wkv_b(self.kv_norm(kv_latent))
        kv_up = kv_up.view(bsz, seqlen, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv_up, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # Assemble full K
        k_pe_expanded = k_pe.expand(-1, -1, self.n_heads, -1)
        k = torch.cat([k_nope, k_pe_expanded], dim=-1)
        q_full = torch.cat([q_nope, q_pe], dim=-1)

        # (B, H, T, D)
        q_full = q_full.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q_full, k, v, is_causal=True, scale=self.softmax_scale
        )
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(out)


# =====================================================
# MoE: 256 experts, top-8 routing, group-limited
# =====================================================
class Expert(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Gate(nn.Module):
    """Group-limited top-k router with sigmoid scores (DeepSeek-V3 style)."""
    def __init__(self, args: MaxConfig):
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_routed = args.n_routed_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale

        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(args.n_routed_experts))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (N, dim)
        scores = F.linear(x, self.weight)
        if self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = scores.softmax(dim=-1)
        original_scores = scores
        scores = scores + self.bias

        # Group-limited routing: only allow top-K groups
        if self.n_groups > 1:
            scores_g = scores.view(x.size(0), self.n_groups, -1)
            group_scores = scores_g.topk(2, dim=-1)[0].sum(dim=-1)
            group_idx = group_scores.topk(self.topk_groups, dim=-1)[1]
            group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1.0)
            scores = (scores_g * group_mask.unsqueeze(-1)).flatten(1)

        indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)
        weights = weights * self.route_scale
        return weights, indices


class MoE(nn.Module):
    def __init__(self, args: MaxConfig):
        super().__init__()
        self.dim = args.dim
        self.n_routed = args.n_routed_experts
        self.n_activated = args.n_activated_experts
        self.gate = Gate(args)
        self.experts = nn.ModuleList(
            [Expert(args.dim, args.moe_intermediate_size) for _ in range(args.n_routed_experts)]
        )
        # Shared expert is always-on, larger hidden dim
        self.shared = Expert(args.dim, args.moe_intermediate_size * args.n_shared_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x_flat = x.view(-1, self.dim)
        weights, indices = self.gate(x_flat)  # (N, topk)
        y = torch.zeros_like(x_flat)

        # Dispatch each token to its top-k experts
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed).tolist()
        for expert_id, expert in enumerate(self.experts):
            if counts[expert_id] == 0:
                continue
            tok_idx, top_idx = torch.where(indices == expert_id)
            if tok_idx.numel() == 0:
                continue
            expert_out = expert(x_flat[tok_idx])
            y[tok_idx] += expert_out * weights[tok_idx, top_idx, None]

        z = self.shared(x_flat)
        return (y + z).view(shape)


# =====================================================
# TRANSFORMER BLOCK
# =====================================================
class Block(nn.Module):
    def __init__(self, layer_id: int, args: MaxConfig):
        super().__init__()
        self.layer_id = layer_id
        self.attn = MLA(args)
        if layer_id < args.first_k_dense_layers:
            self.ffn = Expert(args.dim, args.dense_intermediate_size)
        else:
            self.ffn = MoE(args)
        self.attn_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), freqs_cis)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# =====================================================
# FULL MODEL + Multi-Token Prediction
# =====================================================
class MaxTransformer(nn.Module):
    def __init__(self, args: MaxConfig):
        super().__init__()
        self.args = args

        self.embed = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([Block(i, args) for i in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.head = nn.Linear(args.dim, args.vocab_size, bias=False)

        # Multi-Token Prediction heads — predict next K tokens beyond t+1
        self.n_mtp = args.n_mtp_layers
        self.mtp_weight = args.mtp_loss_weight
        if self.n_mtp > 0:
            self.mtp_layers = nn.ModuleList([Block(0, args) for _ in range(self.n_mtp)])
            self.mtp_norms = nn.ModuleList([RMSNorm(args.dim, args.norm_eps) for _ in range(self.n_mtp)])
            self.mtp_proj = nn.ModuleList(
                [nn.Linear(args.dim * 2, args.dim, bias=False) for _ in range(self.n_mtp)]
            )

        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = self.args.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, seqlen = tokens.shape
        h = self.embed(tokens)
        freqs = self.freqs_cis[:seqlen].to(h.device)

        for layer in self.layers:
            h = layer(h, freqs)
        h_main = self.norm(h)
        logits = self.head(h_main)

        loss = None
        if targets is not None:
            main_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )

            mtp_loss_total = 0.0
            if self.n_mtp > 0:
                h_prev = h_main
                for k in range(self.n_mtp):
                    # Shift target tokens by (k+1) more
                    shifted_tokens = torch.roll(tokens, shifts=-(k + 1), dims=1)
                    shifted_embed = self.embed(shifted_tokens)
                    combined = self.mtp_proj[k](
                        torch.cat([self.mtp_norms[k](h_prev), shifted_embed], dim=-1)
                    )
                    h_prev = self.mtp_layers[k](combined, freqs)
                    mtp_logits = self.head(self.norm(h_prev))
                    shifted_targets = torch.roll(targets, shifts=-(k + 1), dims=1)
                    # Mask last (k+1) positions where roll wraps around
                    shifted_targets[:, -(k + 1):] = -100
                    mtp_loss_total = mtp_loss_total + F.cross_entropy(
                        mtp_logits.view(-1, mtp_logits.size(-1)),
                        shifted_targets.view(-1),
                        ignore_index=-100,
                    )
                mtp_loss_total = mtp_loss_total / self.n_mtp

            loss = main_loss + self.mtp_weight * mtp_loss_total

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        ids = prompt
        for _ in range(max_new_tokens):
            idx_cond = ids if ids.size(1) <= self.args.max_seq_len else ids[:, -self.args.max_seq_len:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)
            if eos_token_id is not None and (next_id == eos_token_id).all():
                break
        return ids
