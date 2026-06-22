"""
MAX-AI Pretraining
FSDP + bf16 mixed precision + cosine LR + gradient accumulation.
Streams from tokenized binary shards.
"""
import os
import sys
import time
import glob
import math
import yaml
import functools
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import StateDictType, FullStateDictConfig

# Make sure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.architecture import MaxTransformer, MaxConfig, Block


# =====================================================
# DATASET — memory-mapped tokenized shards
# =====================================================
class TokenShardDataset(Dataset):
    """
    Reads tokenized uint32 .bin files produced by data/pipeline.py.
    Each __getitem__ returns a contiguous (seq_len+1) chunk → (x, y).
    """
    def __init__(self, glob_pattern: str, seq_len: int):
        self.files = sorted(glob.glob(glob_pattern))
        if not self.files:
            raise FileNotFoundError(f"No shards matched: {glob_pattern}")
        self.seq_len = seq_len
        self.arrays = [np.memmap(f, dtype=np.uint32, mode="r") for f in self.files]
        # Number of full (seq_len+1) windows in each shard
        self.lengths = [max(0, (len(a) - 1) // seq_len) for a in self.arrays]
        self.cum = np.cumsum(self.lengths)
        self.total = int(self.cum[-1])
        if self.total == 0:
            raise ValueError("Shards contain no full windows. Check seq_len / shard sizes.")

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int):
        file_idx = int(np.searchsorted(self.cum, idx, side="right"))
        prev_cum = int(self.cum[file_idx - 1]) if file_idx > 0 else 0
        local = idx - prev_cum
        start = local * self.seq_len
        end = start + self.seq_len + 1
        chunk = np.asarray(self.arrays[file_idx][start:end], dtype=np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


# =====================================================
# LR SCHEDULE — linear warmup + cosine decay to min_lr
# =====================================================
def get_lr(step: int, warmup: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup:
        return max_lr * (step + 1) / max(1, warmup)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup) / max(1, max_steps - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


# =====================================================
# DISTRIBUTED SETUP
# =====================================================
def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return local_rank, global_rank, world_size


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# =====================================================
# CHECKPOINTING (full state dict, rank 0 saves)
# =====================================================
def save_checkpoint(model: FSDP, optimizer, scheduler_state, step: int, out_dir: str, rank: int):
    os.makedirs(out_dir, exist_ok=True)
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        model_state = model.state_dict()
    if rank == 0:
        ckpt = {
            "step": step,
            "model": model_state,
            "scheduler": scheduler_state,
        }
        path = os.path.join(out_dir, f"step_{step}.pt")
        torch.save(ckpt, path)
        # Also write a "latest" symlink
        latest = os.path.join(out_dir, "latest.pt")
        if os.path.lexists(latest):
            os.remove(latest)
        try:
            os.symlink(os.path.basename(path), latest)
        except OSError:
            pass
        print(f"[ckpt] Saved {path}", flush=True)
    dist.barrier()


def load_checkpoint_if_exists(model: FSDP, ckpt_dir: str, rank: int):
    latest = os.path.join(ckpt_dir, "latest.pt")
    if not os.path.exists(latest):
        return 0
    if rank == 0:
        print(f"[ckpt] Loading {latest}", flush=True)
    ckpt = torch.load(latest, map_location="cpu")
    load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, load_policy):
        model.load_state_dict(ckpt["model"])
    return int(ckpt.get("step", 0))


# =====================================================
# PARAM COUNT (for logging)
# =====================================================
def estimate_params(cfg: MaxConfig) -> dict:
    emb = cfg.vocab_size * cfg.dim * 2  # embed + lm_head
    # Per layer attn (MLA approx)
    attn = (
        cfg.dim * cfg.q_lora_rank
        + cfg.q_lora_rank * cfg.n_heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim)
        + cfg.dim * (cfg.kv_lora_rank + cfg.qk_rope_head_dim)
        + cfg.kv_lora_rank * cfg.n_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim)
        + cfg.n_heads * cfg.v_head_dim * cfg.dim
    )
    dense_ffn = 3 * cfg.dim * cfg.dense_intermediate_size
    moe_ffn = cfg.n_routed_experts * 3 * cfg.dim * cfg.moe_intermediate_size
    shared_ffn = 3 * cfg.dim * (cfg.moe_intermediate_size * cfg.n_shared_experts)

    n_dense = cfg.first_k_dense_layers
    n_moe = cfg.n_layers - n_dense
    total = emb + n_dense * (attn + dense_ffn) + n_moe * (attn + moe_ffn + shared_ffn)

    active_ffn_per_moe = 3 * cfg.dim * cfg.moe_intermediate_size * cfg.n_activated_experts
    active = (
        emb
        + n_dense * (attn + dense_ffn)
        + n_moe * (attn + active_ffn_per_moe + shared_ffn)
    )
    return {"total_B": total / 1e9, "active_B": active / 1e9}


# =====================================================
# MAIN
# =====================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/max_config.yaml")
    parser.add_argument("--data-glob", default="/data/tokenized/*.bin")
    parser.add_argument("--ckpt-dir", default="/checkpoints")
    parser.add_argument("--log-dir", default="/logs")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cpu-offload", action="store_true",
                        help="Enable FSDP CPU offload (slow, only for OOM survival).")
    args = parser.parse_args()

    local_rank, global_rank, world_size = setup_distributed()
    is_master = global_rank == 0

    # Load config
    cfg_yaml = yaml.safe_load(open(args.config))
    mcfg = MaxConfig(**cfg_yaml["model"])
    tcfg = cfg_yaml["training"]

    if is_master:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        sizes = estimate_params(mcfg)
        print(f"[init] world_size={world_size}", flush=True)
        print(f"[init] total params: ~{sizes['total_B']:.1f}B | "
              f"active per token: ~{sizes['active_B']:.1f}B", flush=True)
        print(f"[init] seq_len={tcfg['seq_len']} micro_bs={tcfg['micro_batch_size']} "
              f"grad_accum={tcfg['grad_accum']}", flush=True)

    # Build model on GPU
    torch.cuda.set_device(local_rank)
    with torch.device("cuda"):
        model = MaxTransformer(mcfg)

    # FSDP wrap on every Block
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy, transformer_layer_cls={Block}
    )
    mp = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.bfloat16,
    )
    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mp,
        sharding_strategy=ShardingStrategy.HYBRID_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=local_rank,
        use_orig_params=True,
        limit_all_gathers=True,
        cpu_offload=CPUOffload(offload_params=True) if args.cpu_offload else None,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg["lr"],
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=tcfg["weight_decay"],
        fused=True,
    )

    # Resume
    start_step = 0
    if args.resume:
        start_step = load_checkpoint_if_exists(model, args.ckpt_dir, global_rank)

    # Data
    dataset = TokenShardDataset(args.data_glob, tcfg["seq_len"])
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=tcfg["micro_batch_size"],
        sampler=sampler,
        num_workers=tcfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    # Training loop
    model.train()
    grad_accum = tcfg["grad_accum"]
    max_steps = tcfg["max_steps"]
    warmup = tcfg["warmup_steps"]
    max_lr = tcfg["lr"]
    min_lr = tcfg["min_lr"]
    clip = tcfg["grad_clip"]
    log_interval = tcfg["log_interval"]
    save_interval = tcfg["save_interval"]

    step = start_step
    micro_step = 0
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()
    running_loss = 0.0

    epoch = 0
    while step < max_steps:
        sampler.set_epoch(epoch)
        for x, y in loader:
            x = x.to(local_rank, non_blocking=True)
            y = y.to(local_rank, non_blocking=True)

            # Disable grad sync until the final micro-step of accumulation
            is_accum_boundary = ((micro_step + 1) % grad_accum == 0)
            sync_ctx = model.no_sync() if not is_accum_boundary else _nullcontext()
            with sync_ctx:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, loss = model(x, y)
                    loss = loss / grad_accum
                loss.backward()

            running_loss += loss.item()
            micro_step += 1

            if is_accum_boundary:
                # LR schedule
                lr = get_lr(step, warmup, max_steps, max_lr, min_lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                # Grad clip (FSDP handles sharded clipping)
                model.clip_grad_norm_(clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if is_master and step % log_interval == 0:
                    dt = time.time() - t0
                    tokens_seen = (
                        step * grad_accum * tcfg["micro_batch_size"]
                        * tcfg["seq_len"] * world_size
                    )
                    tok_per_sec = tokens_seen / max(dt, 1e-6)
                    print(
                        f"[step {step:>7}] loss={running_loss * grad_accum / log_interval:.4f} "
                        f"lr={lr:.3e} tok/s={tok_per_sec:,.0f} "
                        f"elapsed={dt/3600:.2f}h",
                        flush=True,
                    )
                    running_loss = 0.0

                if step % save_interval == 0:
                    save_checkpoint(model, optimizer, {"step": step}, step, args.ckpt_dir, global_rank)

                if step >= max_steps:
                    break
        epoch += 1

    # Final save
    save_checkpoint(model, optimizer, {"step": step}, step, args.ckpt_dir, global_rank)
    if is_master:
        print("[done] Training complete.", flush=True)
    cleanup_distributed()


# Tiny no-op context manager fallback (avoid importing contextlib at top)
class _nullcontext:
    def __enter__(self): return None
    def __exit__(self, *exc): return False


if __name__ == "__main__":
    main()
