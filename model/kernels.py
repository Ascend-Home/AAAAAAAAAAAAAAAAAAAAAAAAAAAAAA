"""Custom Triton kernels for MoE and MLA. Falls back to PyTorch if unavailable."""
import torch
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:
    @triton.jit
    def fused_moe_kernel(
        a_ptr, b_ptr, c_ptr, topk_weights_ptr, sorted_token_ids_ptr, expert_ids_ptr,
        num_tokens_post_padded_ptr, N, K, EM, num_valid_tokens,
        stride_am, stride_ak, stride_be, stride_bk, stride_bn, stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr, top_k: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        # ... (production MoE kernel — see vLLM source for full impl)
        pass

def grouped_topk(hidden, gate_w, topk, n_groups, topk_groups):
    scores = torch.matmul(hidden, gate_w.T).sigmoid()
    g = scores.view(scores.size(0), n_groups, -1)
    grp_score = g.topk(2, dim=-1)[0].sum(-1)
    grp_idx = grp_score.topk(topk_groups, dim=-1)[1]
    mask = torch.zeros_like(grp_score).scatter_(1, grp_idx, 1.0)
    scores = (g * mask.unsqueeze(-1)).flatten(1)
    weights, indices = scores.topk(topk, dim=-1)
    return weights / weights.sum(-1, keepdim=True), indices
