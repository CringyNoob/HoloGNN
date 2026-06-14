"""
pretrain_uniref.py
==================
Holo-GNN — Unsupervised GATv2 Pre-Training via Masked Language Modeling.

Hardened for an interruption-prone environment (load shedding / power cuts) and
tuned for a single RTX 5070 Ti behind a 16 GB WSL2 RAM ceiling.

Key engineering upgrades over the previous version
--------------------------------------------------
1. Anti-load-shedding checkpointing
   * Full training state (model / optimizer / scheduler / AMP scaler / epoch /
     global_step / step-within-epoch) is written to ``latest_checkpoint.pth``
     every ``--save_interval`` steps (default 500) and at the end of each epoch.
   * Writes are ATOMIC (write to ``*.tmp`` then ``os.replace``).  A power cut
     mid-write can never corrupt the checkpoint — the previous good file stays
     intact until the new one is fully flushed and renamed.
   * On launch the script auto-detects ``latest_checkpoint.pth`` in the output
     directory and resumes from the exact epoch / step / LR state.  No flags
     required; ``--resume_from`` is available only as a manual override.

2. Hardware-optimized, RAM-safe pipeline
   * AMP (fp16 autocast + GradScaler) to light up the 5070 Ti Tensor Cores.
   * TF32 / cuDNN autotuning enabled (fixed padded shapes => benchmark wins).
   * ``ShardAwareSampler`` shuffles WITHIN each parquet shard while keeping each
     shard contiguous in iteration order.  This is the fix for the OOM seen at
     num_workers=4: UniRefDataset keeps only ONE shard in RAM at a time, so
     global shuffling thrashed that cache and duplicated shards across workers.
     Shard-aware ordering reads each shard from disk exactly once per epoch and
     bounds RAM to ``num_workers × one_shard`` — making num_workers=2 safe.
   * DataLoader defaults tuned for the 16 GB ceiling: num_workers=2,
     pin_memory=True, persistent_workers=True, prefetch_factor=2.
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler
from torch.utils.tensorboard import SummaryWriter

# ── Local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from src.backbone import HoloGNNBackbone          # noqa: E402
from src.dataset  import UniRefDataset            # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# UTF-8 safe stdout (defensive: emoji/✓ prints crash on cp1252 consoles)
# ─────────────────────────────────────────────────────────────────────────────
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("pretrain_mlm")

ESM2_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
ESM2_VOCAB_SIZE = 33
ESM2_MASK_TOKEN = 32
ESM2_SPECIAL_TOKENS = {0, 1, 2, 3}
MLM_MASK_PROB = 0.15

CHECKPOINT_NAME = "latest_checkpoint.pth"


# ─────────────────────────────────────────────────────────────────────────────
# MLM head + masking (unchanged logic)
# ─────────────────────────────────────────────────────────────────────────────
class MLM_Head(nn.Module):
    def __init__(self, input_dim: int = 320, vocab_size: int = ESM2_VOCAB_SIZE):
        super().__init__()
        self.dense      = nn.Linear(input_dim, input_dim)
        self.act        = nn.GELU()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.decoder    = nn.Linear(input_dim, vocab_size, bias=False)
        self.bias       = nn.Parameter(torch.zeros(vocab_size))
        self.decoder.bias = self.bias

    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        x = self.dense(node_embeddings)
        x = self.act(x)
        x = self.layer_norm(x)
        return self.decoder(x)


def apply_mlm_mask(
    input_ids:      torch.Tensor,
    attention_mask: torch.Tensor,
    mask_prob:      float = MLM_MASK_PROB,
    mask_token_id:  int   = ESM2_MASK_TOKEN,
    special_tokens: set   = ESM2_SPECIAL_TOKENS,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = input_ids.clone()
    masked = input_ids.clone()

    is_real     = attention_mask.bool()
    is_eligible = is_real.clone()
    for st in special_tokens:
        is_eligible &= (input_ids != st)

    prob_matrix = torch.full(input_ids.shape, mask_prob, device=input_ids.device)
    prob_matrix[~is_eligible] = 0.0
    should_mask = torch.bernoulli(prob_matrix).bool()

    labels[~should_mask] = -100

    replace_mask = torch.bernoulli(torch.full(input_ids.shape, 0.80, device=input_ids.device)).bool()
    replace_mask &= should_mask
    masked[replace_mask] = mask_token_id

    replace_random = torch.bernoulli(torch.full(input_ids.shape, 0.50, device=input_ids.device)).bool()
    replace_random &= should_mask & ~replace_mask
    random_tokens = torch.randint(4, 24, input_ids.shape, device=input_ids.device)
    masked[replace_random] = random_tokens[replace_random]

    return masked, labels


class HoloGNN_MLM(nn.Module):
    def __init__(self, output_dim: int = 320, vocab_size: int = ESM2_VOCAB_SIZE):
        super().__init__()
        self.backbone = HoloGNNBackbone(output_dim=output_dim)
        self.mlm_head = MLM_Head(input_dim=output_dim, vocab_size=vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                mechanistic_features: torch.Tensor) -> torch.Tensor:
        node_embeddings, _ = self.backbone(
            input_ids            = input_ids,
            attention_mask       = attention_mask,
            mechanistic_features = mechanistic_features,
        )
        return self.mlm_head(node_embeddings)


# ─────────────────────────────────────────────────────────────────────────────
# Shard-aware sampler — the RAM-safety cornerstone for UniRefDataset
# ─────────────────────────────────────────────────────────────────────────────
class ShardAwareSampler(Sampler[int]):
    """Yield indices shuffled *within* each parquet shard, keeping each shard
    contiguous in iteration order.

    Why this exists
    ---------------
    ``UniRefDataset`` caches exactly ONE parquet shard at a time
    (``self._cache_df``).  A standard ``shuffle=True`` produces a globally
    random index order, so consecutive ``__getitem__`` calls hop between shards
    and the single-slot cache is re-read (300–500 MB parquet) on nearly every
    sample.  With DataLoader workers, each worker independently caches whatever
    random shard it last touched, so many full shards live in RAM at once —
    exactly the duplication that triggered the WSL OOM killer at num_workers=4.

    This sampler instead:
      * shuffles the ORDER in which shards are visited (epoch-to-epoch variety),
      * shuffles rows WITHIN each shard (true SGD shuffling for MLM quality),
      * but never interleaves shards.

    Result: each shard is read from disk exactly once per epoch, and peak RAM
    is bounded to ``num_workers × one_shard`` regardless of dataset size.

    Resume support
    --------------
    ``skip`` drops the first ``skip`` *global indices* of an epoch without
    loading them, so resuming mid-epoch costs no wasted data I/O.
    """

    def __init__(self, shard_offsets: Sequence[int], shuffle: bool = True, seed: int = 42):
        # shard_offsets is cumulative row counts: len == n_shards + 1
        self.shard_offsets: List[int] = list(shard_offsets)
        self.shuffle = shuffle
        self.seed    = seed
        self.epoch   = 0
        self.skip    = 0                      # number of leading indices to drop
        self._total  = self.shard_offsets[-1]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        rng      = random.Random(self.seed + self.epoch)
        n_shards = len(self.shard_offsets) - 1

        shard_order = list(range(n_shards))
        if self.shuffle:
            rng.shuffle(shard_order)

        order: List[int] = []
        for s in shard_order:
            block = list(range(self.shard_offsets[s], self.shard_offsets[s + 1]))
            if self.shuffle:
                rng.shuffle(block)
            order.extend(block)

        if self.skip:
            order = order[self.skip:]
        return iter(order)

    def __len__(self) -> int:
        return max(0, self._total - self.skip)


# ─────────────────────────────────────────────────────────────────────────────
# Atomic checkpoint helpers (power-cut safe)
# ─────────────────────────────────────────────────────────────────────────────
def _atomic_save(state: dict, path: Path) -> None:
    """Write ``state`` to ``path`` atomically.

    torch.save to a temp file on the same filesystem, then os.replace (atomic
    rename).  If power is lost mid-write, ``path`` still points at the previous
    complete checkpoint — never a half-written, unloadable file.
    """
    path = Path(path)
    tmp  = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    # Best-effort durability before the rename.
    try:
        with open(tmp, "rb") as fh:
            os.fsync(fh.fileno())
    except Exception:
        pass
    os.replace(tmp, path)


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    """Load a checkpoint, tolerating both the new and legacy key spellings."""
    ckpt = torch.load(path, map_location=device)

    # Normalise to the canonical *_state_dict keys used by this script.
    aliases = {
        "model_state":     "model_state_dict",
        "optimizer_state": "optimizer_state_dict",
        "scheduler_state": "scheduler_state_dict",
        "scaler_state":    "scaler_state_dict",
    }
    for old, new in aliases.items():
        if new not in ckpt and old in ckpt:
            ckpt[new] = ckpt[old]
    return ckpt


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def pretrain(
    parquet_dir:    str,
    output_path:    str,
    epochs:         int   = 3,
    batch_size:     int   = 16,
    max_length:     int   = 512,
    lr:             float = 1e-4,
    weight_decay:   float = 0.01,
    num_workers:    int   = 2,
    grad_clip:      float = 1.0,
    log_interval:   int   = 100,
    save_interval:  int   = 500,
    checkpoint_dir: Optional[str] = None,
    resume_from:    Optional[str] = None,
    seed:           int   = 42,
) -> None:
    from src.device import get_device, describe_device

    t_global = time.time()
    device   = get_device()
    describe_device(device)
    use_cuda    = device.type == "cuda"
    use_amp     = use_cuda                       # AMP only meaningful on CUDA
    amp_device  = "cuda" if use_cuda else "cpu"
    amp_dtype   = torch.float16

    # Reproducibility + Tensor-Core friendly fast paths.
    random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark        = True   # fixed padded shapes -> autotune wins
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # ── Output / checkpoint locations ────────────────────────────────────────
    output_path = Path(output_path)
    ckpt_dir    = Path(checkpoint_dir) if checkpoint_dir else output_path.parent
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = ckpt_dir / CHECKPOINT_NAME

    log.info("=" * 68)
    log.info("  Holo-GNN Unsupervised Pre-Training — MLM on UniRef50")
    log.info(f"  Device        : {device}")
    if use_cuda:
        log.info(f"  GPU           : {torch.cuda.get_device_name(0)}")
        log.info(f"  VRAM          : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    log.info(f"  AMP (fp16)    : {use_amp}")
    log.info(f"  Workers       : {num_workers}  | batch_size={batch_size} | max_length={max_length}")
    log.info(f"  Checkpoint    : {latest_ckpt}")
    log.info(f"  Save every    : {save_interval} steps  (atomic, power-cut safe)")
    log.info("=" * 68)

    # ── Dataset + shard-aware loader ─────────────────────────────────────────
    dataset = UniRefDataset(parquet_dir=parquet_dir, max_length=max_length, shuffle_shards=True)
    sampler = ShardAwareSampler(dataset._shard_offsets, shuffle=True, seed=seed)

    loader = DataLoader(
        dataset,
        batch_size         = batch_size,
        sampler            = sampler,                       # NB: do NOT also pass shuffle=True
        num_workers        = num_workers,
        pin_memory         = use_cuda,                      # faster pinned H2D copies
        persistent_workers = (num_workers > 0),             # keep shard caches between epochs
        prefetch_factor    = (2 if num_workers > 0 else None),
        drop_last          = True,
    )

    # Schedule length is computed from the FULL dataset so the LR schedule is
    # stable regardless of any mid-epoch resume skip.
    steps_per_epoch = max(1, len(dataset) // batch_size)
    total_steps     = steps_per_epoch * epochs
    log.info(f"  Steps/epoch   : {steps_per_epoch:,}   | total steps: {total_steps:,}")

    # ── Model / optim / sched / scaler ───────────────────────────────────────
    model     = HoloGNN_MLM(output_dim=320, vocab_size=ESM2_VOCAB_SIZE).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    scaler    = torch.amp.GradScaler(amp_device, enabled=use_amp)

    # ── Auto-resume (anti-load-shedding) ─────────────────────────────────────
    start_epoch   = 1
    global_step   = 0
    skip_in_epoch = 0     # optimizer steps already completed in the resumed epoch

    resume_target: Optional[Path] = None
    if resume_from:
        resume_target = Path(resume_from)
        if not resume_target.exists():
            log.warning(f"--resume_from '{resume_from}' not found; ignoring.")
            resume_target = None
    elif latest_ckpt.exists():
        resume_target = latest_ckpt

    if resume_target is not None:
        log.info(f"⟳ Resuming from checkpoint: {resume_target}")
        ckpt = _load_checkpoint(resume_target, device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"] is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])

        global_step       = int(ckpt.get("global_step", 0))
        saved_epoch       = int(ckpt.get("epoch", 1))
        done_in_epoch     = int(ckpt.get("step_in_epoch", 0))

        if done_in_epoch >= steps_per_epoch:
            # That epoch was fully completed — start the next one cleanly.
            start_epoch   = saved_epoch + 1
            skip_in_epoch = 0
        else:
            start_epoch   = saved_epoch
            skip_in_epoch = done_in_epoch

        log.info(
            f"   → resumed at epoch {start_epoch}, "
            f"global_step {global_step:,}, skipping {skip_in_epoch:,} steps of this epoch | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        if start_epoch > epochs:
            log.info("Checkpoint already covers all requested epochs. Saving backbone and exiting.")
            _save_backbone(model, output_path)
            return
    else:
        log.info("No checkpoint found — starting a fresh run from scratch.")

    # ── TensorBoard ──────────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=str(ckpt_dir / "runs" / "uniref_pretrain"))

    def _full_state(epoch: int, step_in_epoch: int) -> dict:
        return {
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict":    scaler.state_dict(),
            "epoch":                epoch,
            "step_in_epoch":        step_in_epoch,
            "global_step":          global_step,
        }

    # ── Train ────────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_skip = skip_in_epoch if epoch == start_epoch else 0

        # Configure the sampler for this epoch BEFORE the loader is iterated.
        sampler.set_epoch(epoch)
        sampler.skip = epoch_skip * batch_size

        epoch_loss = epoch_acc = 0.0
        n_batches  = 0
        t_epoch    = time.time()

        for step, batch in enumerate(loader, 1):
            completed_in_epoch = epoch_skip + step          # 1-indexed batch # within epoch

            if step == 1:
                msg = "🔥 BATCH 1 SUCCESSFULLY LOADED! The engine is firing."
                if epoch_skip:
                    msg += f"  (fast-forwarded past {epoch_skip:,} steps)"
                log.info(msg)

            input_ids      = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            mech_features  = batch["mechanistic_features"].to(device, non_blocking=True)

            masked_ids, labels = apply_mlm_mask(input_ids, attention_mask)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=amp_device, dtype=amp_dtype, enabled=use_amp):
                logits = model(input_ids=masked_ids, attention_mask=attention_mask,
                               mechanistic_features=mech_features)
                B, L, V = logits.shape
                loss = criterion(logits.view(B * L, V), labels.view(B * L))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            with torch.no_grad():
                n_mask_pos = (labels != -100).sum().item()
                correct    = ((logits.argmax(-1) == labels) & (labels != -100)).sum().item()
                acc        = correct / max(n_mask_pos, 1)

            epoch_loss  += loss.item()
            epoch_acc   += acc
            n_batches   += 1
            global_step += 1

            # ── Telemetry ────────────────────────────────────────────────────
            writer.add_scalar("Training/Loss",        loss.item(), global_step)
            writer.add_scalar("Training/Accuracy",    acc,         global_step)
            writer.add_scalar("System/Learning_Rate", optimizer.param_groups[0]["lr"], global_step)

            if completed_in_epoch % log_interval == 0 or step == len(loader):
                log.info(
                    f"Epoch {epoch} step {completed_in_epoch}/{steps_per_epoch} | "
                    f"loss={epoch_loss/n_batches:.4f} acc={epoch_acc/n_batches:.3f} | "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )

            # ── Anti-load-shedding checkpoint (atomic) ───────────────────────
            if global_step % save_interval == 0:
                _atomic_save(_full_state(epoch, completed_in_epoch), latest_ckpt)
                log.info(f"💾 checkpoint @ step {global_step:,} → {latest_ckpt.name}")

        # ── End-of-epoch checkpoints ─────────────────────────────────────────
        epoch_avg_loss = epoch_loss / max(n_batches, 1)
        log.info(
            f"Epoch {epoch} complete in {time.time()-t_epoch:.1f}s | "
            f"avg loss={epoch_avg_loss:.4f}"
        )

        # mark this epoch as fully done so a resume jumps to the next one
        _atomic_save(_full_state(epoch, steps_per_epoch), latest_ckpt)
        epoch_ckpt = ckpt_dir / f"mlm_epoch_{epoch:02d}.pth"
        _atomic_save(_full_state(epoch, steps_per_epoch), epoch_ckpt)
        log.info(f"💾 epoch checkpoint → {epoch_ckpt.name}")

    writer.close()
    _save_backbone(model, output_path)
    log.info(f"PRE-TRAINING COMPLETE in {time.time()-t_global:.1f}s. Backbone saved to {output_path}")


def _save_backbone(model: HoloGNN_MLM, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_save(model.backbone.state_dict(), output_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Holo-GNN UniRef50 MLM pre-training (resilient).")
    parser.add_argument("--parquet_dir", "-d", default="CLEANED_DATA/")
    parser.add_argument("--output", "-o", dest="output_path", default="uniref_pretrained_weights.pth")
    parser.add_argument("--epochs",        type=int,   default=3)
    parser.add_argument("--batch_size",    type=int,   default=16)
    parser.add_argument("--max_length",    type=int,   default=512)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--weight_decay",  type=float, default=0.01)
    parser.add_argument("--num_workers",   type=int,   default=2,
                        help="2 is the sweet spot under a 16GB WSL ceiling. 0 = single process.")
    parser.add_argument("--grad_clip",     type=float, default=1.0)
    parser.add_argument("--log_interval",  type=int,   default=100)
    parser.add_argument("--save_interval", type=int,   default=500,
                        help="Atomic full-state checkpoint cadence (steps).")
    parser.add_argument("--checkpoint_dir", type=str,  default=None,
                        help="Where latest_checkpoint.pth lives. Defaults to the output dir.")
    parser.add_argument("--resume_from",   type=str,   default=None,
                        help="Manual override. Normally auto-resume finds latest_checkpoint.pth.")
    parser.add_argument("--seed",          type=int,   default=42)
    args = parser.parse_args()

    pretrain(**vars(args))
