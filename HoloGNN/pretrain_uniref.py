"""
pretrain_uniref.py
==================
Holo-GNN V5.0 — Unsupervised GATv2 Pre-Training via Masked Language Modeling

Objective
---------
Teach the GATv2 backbone the rules of protein folding *before* fine-tuning on
labelled stability / pathogenicity data.  We use a BERT-style Masked Language
Modeling (MLM) objective over UniRef50:

    1. For each sequence in a batch, randomly mask 15 % of the amino-acid
       token positions in-place.
    2. Feed the masked sequence through the full ESM-2 + GATv2 backbone to get
       per-residue node embeddings (B, L, output_dim).
    3. A lightweight MLM_Head projects those embeddings back to the 20-AA +
       special-token vocabulary.
    4. Cross-entropy loss is computed ONLY on the masked positions.

Why MLM on GATv2?
-----------------
ESM-2 is already pre-trained as a protein language model, so the ESM-2 weights
start with strong amino-acid context.  What they do NOT know is how to route
information through a *graph* topology (the GATv2 message-passing layers were
randomly initialised).  MLM on UniRef50 forces the GATv2 layers to learn which
neighbouring residues are most informative for predicting a masked token — this
is structurally equivalent to learning co-evolutionary contact patterns, the
same signal that a contact-map pre-training objective provides.

After this pre-training, the GATv2 weights are saved to
  uniref_pretrained_weights.pth
and loaded by the main V5.0 training loop via:
  model.backbone.load_state_dict(torch.load("uniref_pretrained_weights.pth"))

Architecture during pre-training
---------------------------------
  HoloGNNBackbone (ESM-2 + GATv2 + residual)   ← shared weights saved
  MLM_Head        (Linear: output_dim → vocab)   ← discarded after pre-training

  The task-specific heads (SiameseStabilityHead, ClinVar classifier) are NOT
  instantiated during pre-training to keep GPU memory overhead minimal.

Usage
-----
  python pretrain_uniref.py \\
      --parquet_dir "CLEANED_DATA/" \\
      --output      "uniref_pretrained_weights.pth" \\
      --epochs      3 \\
      --batch_size  128 \\
      --max_length  512
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

# ── Local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from src.backbone import HoloGNNBackbone, ESM2_HIDDEN_DIM, _PYGEO_AVAILABLE
from src.dataset  import UniRefDataset

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


# =============================================================================
# ESM-2 Tokeniser constants
# =============================================================================
# ESM-2 uses a 33-token alphabet.  The amino acid tokens occupy indices 4–23.
# Special tokens: <cls>=0, <pad>=1, <eos>=2, <unk>=3, mask=32.
# We need the MASK token id and the vocab size for the MLM head output.

ESM2_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
ESM2_VOCAB_SIZE = 33        # full ESM-2 vocabulary (all special + 20 AA)
ESM2_MASK_TOKEN = 32        # <mask> token id in ESM-2 tokeniser

# Tokens to NEVER mask (special tokens: cls, pad, eos, unk)
ESM2_SPECIAL_TOKENS = {0, 1, 2, 3}

# Standard BERT MLM probability
MLM_MASK_PROB = 0.15


# =============================================================================
# MLM_Head — temporary, discarded after pre-training
# =============================================================================
class MLM_Head(nn.Module):
    """
    Lightweight per-residue classifier projecting GATv2 node embeddings back
    to the full ESM-2 vocabulary.

    Architecture:  Linear(output_dim → output_dim) → GELU → LayerNorm
                   → Linear(output_dim → vocab_size)

    This mirrors the BERT MLM head design: a two-layer transform with a hidden
    layer of the same width, then a normed projection to the vocabulary.  The
    additional depth (vs a single linear layer) gives the head enough capacity
    to decode structural context from the GATv2 embeddings without overwhelming
    the backbone gradient signal.

    This module is DISCARDED after pre-training — only backbone weights are saved.

    Args:
        input_dim  : Must match HoloGNNBackbone.output_dim (default 320).
        vocab_size  : ESM-2 vocabulary size (default 33).
    """

    def __init__(self, input_dim: int = 320, vocab_size: int = ESM2_VOCAB_SIZE):
        super().__init__()
        self.dense      = nn.Linear(input_dim, input_dim)
        self.act        = nn.GELU()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.decoder    = nn.Linear(input_dim, vocab_size, bias=False)
        self.bias       = nn.Parameter(torch.zeros(vocab_size))
        self.decoder.bias = self.bias

    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_embeddings : (B, L, input_dim) — per-residue GATv2 output.

        Returns:
            logits : (B, L, vocab_size) — unnormalised scores over AA vocabulary.
        """
        x = self.dense(node_embeddings)
        x = self.act(x)
        x = self.layer_norm(x)
        return self.decoder(x)


# =============================================================================
# MLM Masking Utility
# =============================================================================
def apply_mlm_mask(
    input_ids:     torch.Tensor,
    attention_mask: torch.Tensor,
    mask_prob:     float = MLM_MASK_PROB,
    mask_token_id: int   = ESM2_MASK_TOKEN,
    special_tokens: set  = ESM2_SPECIAL_TOKENS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply BERT-style masking in-place on a batch of token sequences.

    Masking strategy (identical to original BERT paper):
      80 % of masked positions → replace with <mask> token
      10 % of masked positions → replace with a random amino-acid token
      10 % of masked positions → keep original token unchanged

    Only non-special (amino acid) positions are eligible for masking.
    Padding positions (attention_mask == 0) are always ignored.

    Args:
        input_ids      : (B, L) — original token ids, will be MODIFIED in-place.
        attention_mask : (B, L) — 1 for real tokens, 0 for padding.
        mask_prob      : probability of masking each eligible position.
        mask_token_id  : ESM-2 <mask> token id.
        special_tokens : set of token ids that must never be masked.

    Returns:
        masked_input_ids : (B, L) — input_ids with masking applied (clone).
        labels           : (B, L) — original token ids at masked positions,
                                     -100 everywhere else (ignored by CrossEntropyLoss).
    """
    labels    = input_ids.clone()
    masked    = input_ids.clone()

    # Boolean mask: True where the position is real and non-special
    is_real   = attention_mask.bool()
    is_eligible = is_real.clone()
    for st in special_tokens:
        is_eligible &= (input_ids != st)

    # Sample mask_prob fraction of eligible positions
    prob_matrix  = torch.full(input_ids.shape, mask_prob, device=input_ids.device)
    prob_matrix[~is_eligible] = 0.0
    should_mask  = torch.bernoulli(prob_matrix).bool()

    # Only compute loss on masked positions (all others → -100)
    labels[~should_mask] = -100

    # 80 % → replace with <mask>
    replace_mask = torch.bernoulli(torch.full(input_ids.shape, 0.80,
                                              device=input_ids.device)).bool()
    replace_mask &= should_mask
    masked[replace_mask] = mask_token_id

    # 10 % → random token from the amino-acid range (4–23 inclusive in ESM-2).
    # This is applied to the ~20 % of masked positions NOT turned into <mask>,
    # so the conditional probability must be 0.50 (= 10 % of all masked), not 0.10.
    replace_random = torch.bernoulli(
        torch.full(input_ids.shape, 0.50, device=input_ids.device)
    ).bool()
    replace_random &= should_mask & ~replace_mask
    random_tokens   = torch.randint(4, 24, input_ids.shape, device=input_ids.device)
    masked[replace_random] = random_tokens[replace_random]

    # Remaining 10 % → keep original (do nothing; masked already equals input_ids)

    return masked, labels


# =============================================================================
# Pre-Training Model: Backbone + MLM_Head
# =============================================================================
class HoloGNN_MLM(nn.Module):
    """
    Wraps HoloGNNBackbone + MLM_Head for pre-training.

    Only the backbone weights are saved after training — MLM_Head is ephemeral.

    Args:
        output_dim  : GATv2 output dimension (must match backbone, default 320).
        vocab_size  : ESM-2 vocabulary size (default 33).
    """

    def __init__(self, output_dim: int = 320, vocab_size: int = ESM2_VOCAB_SIZE):
        super().__init__()
        self.backbone = HoloGNNBackbone(output_dim=output_dim)
        self.mlm_head = MLM_Head(input_dim=output_dim, vocab_size=vocab_size)

    def forward(
        self,
        input_ids:            torch.Tensor,
        attention_mask:       torch.Tensor,
        mechanistic_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids            : (B, L) — masked token ids.
            attention_mask       : (B, L) — 1 for real tokens.
            mechanistic_features : (B, L, 3) — biophysical features.

        Returns:
            logits : (B, L, vocab_size) — MLM predictions over vocabulary.
        """
        # Backbone: (B, L, output_dim),  (B, output_dim)
        node_embeddings, _ = self.backbone(
            input_ids            = input_ids,
            attention_mask       = attention_mask,
            mechanistic_features = mechanistic_features,
        )
        # MLM head operates on per-residue embeddings
        return self.mlm_head(node_embeddings)   # (B, L, vocab_size)


# =============================================================================
# Training Loop
# =============================================================================
def pretrain(
    parquet_dir:  str,
    output_path:  str,
    epochs:       int   = 3,
    batch_size:   int   = 128,
    max_length:   int   = 512,
    lr:           float = 1e-4,
    weight_decay: float = 0.01,
    num_workers:  int   = 8,
    grad_clip:    float = 1.0,
    log_interval: int   = 100,
    resume_from:  Optional[str] = None,
) -> None:
    """
    Main pre-training entry point.

    Architecture decisions
    ----------------------
    • Mixed precision (torch.cuda.amp):
        All forward passes run in float16 on GPU.  GradScaler handles the
        loss scaling to prevent underflow in float16 gradients.
        This roughly halves VRAM usage vs float32, allowing batch_size=128
        on a 24 GB L4 GPU with max_length=512.

    • Gradient clipping (max_norm=1.0):
        Essential for MLM training stability — the cross-entropy loss over a
        33-class vocabulary can produce large gradient spikes early in training
        when the GATv2 weights are random.

    • AdamW + cosine LR schedule:
        AdamW decouples weight decay from the adaptive moment estimates, which
        is critical for large language-model style objectives.  Cosine annealing
        ensures the learning rate decays smoothly over the full pre-training run.

    • Epoch checkpointing:
        Each epoch saves the full model state dict (backbone + MLM head) to
        protect against Vertex AI preemption during the multi-hour run.
        Final output saves only the backbone for use in fine-tuning.
    """
    from src.device import get_device, describe_device
    t_global = time.time()
    device   = get_device()
    describe_device(device)          # Blackwell / sm_120 runtime guard
    log.info("=" * 68)
    log.info("  Holo-GNN Unsupervised Pre-Training — MLM on UniRef50")
    log.info(f"  Device     : {device}")
    if torch.cuda.is_available():
        log.info(f"  GPU        : {torch.cuda.get_device_name(0)}")
        log.info(f"  VRAM       : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    log.info(f"  Parquet dir: {parquet_dir}")
    log.info(f"  Output     : {output_path}")
    log.info(f"  Epochs     : {epochs}  |  Batch: {batch_size}  |  MaxLen: {max_length}")
    log.info(f"  GATv2      : {'ENABLED' if _PYGEO_AVAILABLE else 'DISABLED (PyG missing)'}")
    log.info("=" * 68)

    # ── Dataset & DataLoader ─────────────────────────────────────────────────
    log.info("Building UniRef50 streaming dataset …")
    dataset = UniRefDataset(
        parquet_dir   = parquet_dir,
        max_length    = max_length,
        shuffle_shards = True,
    )
    loader = DataLoader(
        dataset,
        batch_size       = batch_size,
        shuffle          = True,
        num_workers      = num_workers,
        pin_memory       = True,
        persistent_workers = True,
        prefetch_factor  = 2,
        drop_last        = True,
    )
    total_steps = len(loader) * epochs
    log.info(f"  {len(dataset):,} sequences  |  {len(loader):,} steps/epoch  |"
             f"  {total_steps:,} total steps")

    # ── Model ───────────────────────────────────────────────────────────────
    log.info("Initialising HoloGNN_MLM …")
    model = HoloGNN_MLM(output_dim=320, vocab_size=ESM2_VOCAB_SIZE).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  Total parameters    : {n_params:,}")
    log.info(f"  Trainable params    : {n_train:,}")

    # ── Optim & Scheduler ───────────────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr           = lr,
        weight_decay = weight_decay,
        betas        = (0.9, 0.999),
    )
    # Cosine annealing over the full training run
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=lr * 0.01
    )

    # ── Loss ────────────────────────────────────────────────────────────────
    # ignore_index=-100 matches the labels tensor from apply_mlm_mask
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # ── Mixed precision ──────────────────────────────────────────────────────
    use_amp   = torch.cuda.is_available()
    scaler    = GradScaler(enabled=use_amp)

    # ── Resume from checkpoint ───────────────────────────────────────────────
    start_epoch = 1
    if resume_from and Path(resume_from).exists():
        log.info(f"Resuming from checkpoint: {resume_from}")
        ckpt        = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt.get("epoch", 1) + 1
        log.info(f"  Resumed from epoch {start_epoch - 1}")

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    global_step   = 0

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss  = 0.0
        epoch_acc   = 0.0
        n_batches   = 0
        n_masked    = 0
        t_epoch     = time.time()

        log.info(f"\n{'─' * 68}")
        log.info(f"  Epoch {epoch}/{epochs}  —  {len(loader):,} batches")
        log.info(f"{'─' * 68}")

        for step, batch in enumerate(loader, 1):
            input_ids      = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            mech_features  = batch["mechanistic_features"].to(device, non_blocking=True)

            # ── Apply MLM masking ─────────────────────────────────────────
            masked_ids, labels = apply_mlm_mask(
                input_ids, attention_mask,
                mask_prob     = MLM_MASK_PROB,
                mask_token_id = ESM2_MASK_TOKEN,
            )

            # ── Forward pass (mixed precision) ────────────────────────────
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                logits = model(
                    input_ids            = masked_ids,
                    attention_mask       = attention_mask,
                    mechanistic_features = mech_features,
                )
                # logits : (B, L, vocab_size)
                # labels : (B, L) — -100 at non-masked positions
                B, L, V = logits.shape
                loss = criterion(
                    logits.view(B * L, V),
                    labels.view(B * L),
                )

            # ── Backward + gradient clip ──────────────────────────────────
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # ── Metrics ───────────────────────────────────────────────────
            with torch.no_grad():
                n_mask_pos = (labels != -100).sum().item()
                correct    = ((logits.argmax(-1) == labels) & (labels != -100)).sum().item()
                acc        = correct / max(n_mask_pos, 1)

            epoch_loss += loss.item()
            epoch_acc  += acc
            n_masked   += n_mask_pos
            n_batches  += 1
            global_step += 1

            # ── Periodic log ──────────────────────────────────────────────
            if step % log_interval == 0 or step == len(loader):
                avg_loss = epoch_loss / n_batches
                avg_acc  = epoch_acc  / n_batches
                lr_now   = optimizer.param_groups[0]["lr"]
                elapsed  = time.time() - t_epoch
                eta      = elapsed / step * (len(loader) - step)
                log.info(
                    f"  Epoch {epoch}  step {step:>6}/{len(loader)}  "
                    f"loss={avg_loss:.4f}  acc={avg_acc:.3f}  "
                    f"lr={lr_now:.2e}  "
                    f"elapsed={elapsed/60:.1f}min  eta={eta/60:.1f}min  "
                    f"masked_tokens={n_masked:,}"
                )

        # ── End of epoch ──────────────────────────────────────────────────
        epoch_avg_loss = epoch_loss / max(n_batches, 1)
        epoch_avg_acc  = epoch_acc  / max(n_batches, 1)
        epoch_time     = time.time() - t_epoch

        log.info(
            f"\n  ✅ Epoch {epoch} complete  "
            f"avg_loss={epoch_avg_loss:.4f}  avg_acc={epoch_avg_acc:.4f}  "
            f"time={epoch_time/60:.1f}min"
        )

        # ── Epoch checkpoint (preemption-safe) ────────────────────────────
        epoch_ckpt = Path(output_path).parent / f"mlm_epoch_{epoch:02d}.pth"
        torch.save({
            "epoch":           epoch,
            "global_step":     global_step,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch_loss":      epoch_avg_loss,
            "epoch_acc":       epoch_avg_acc,
        }, epoch_ckpt)
        log.info(f"  💾 Epoch checkpoint saved → {epoch_ckpt}")

    # ── Save final backbone weights only ──────────────────────────────────────
    # The MLM_Head is intentionally discarded.  Only the backbone weights are
    # needed for the fine-tuning stage (V5.0 training loop).
    backbone_state = model.backbone.state_dict()
    torch.save(backbone_state, output_path)

    total_time = time.time() - t_global
    log.info("\n" + "=" * 68)
    log.info("  PRE-TRAINING COMPLETE")
    log.info(f"  Total wall time : {total_time/3600:.2f} h")
    log.info(f"  Final avg loss  : {epoch_avg_loss:.4f}")
    log.info(f"  Final avg acc   : {epoch_avg_acc:.4f}")
    log.info(f"  Backbone saved  : {output_path}")
    log.info("")
    log.info("  Load in V5.0 training loop with:")
    log.info("    model.backbone.load_state_dict(")
    log.info(f"        torch.load('{output_path}', map_location=device)")
    log.info("    )")
    log.info("=" * 68)


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Holo-GNN MLM Pre-training on UniRef50",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--parquet_dir", "-d",
        default="CLEANED_DATA/",
        help="Directory containing uniref50_clean_part_*.parquet shards.",
    )
    parser.add_argument(
        "--output", "-o",
        default="uniref_pretrained_weights.pth",
        help="Path to save final pre-trained backbone weights.",
    )
    parser.add_argument("--epochs",      type=int,   default=3)
    parser.add_argument("--batch_size",  type=int,   default=128,
                        help="Sequences per batch. 128 fits 24 GB L4 at max_length=512.")
    parser.add_argument("--max_length",  type=int,   default=512,
                        help="Token sequence length. Must be ≤ 1022 (ESM-2 limit).")
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--weight_decay",type=float, default=0.01)
    parser.add_argument("--num_workers", type=int,   default=8)
    parser.add_argument("--grad_clip",   type=float, default=1.0)
    parser.add_argument("--log_interval",type=int,   default=100,
                        help="Print metrics every N steps.")
    parser.add_argument("--resume_from", type=str,   default=None,
                        help="Path to an mlm_epoch_XX.pth checkpoint to resume from.")
    args = parser.parse_args()

    pretrain(
        parquet_dir  = args.parquet_dir,
        output_path  = args.output,
        epochs       = args.epochs,
        batch_size   = args.batch_size,
        max_length   = args.max_length,
        lr           = args.lr,
        weight_decay = args.weight_decay,
        num_workers  = args.num_workers,
        grad_clip    = args.grad_clip,
        log_interval = args.log_interval,
        resume_from  = args.resume_from,
    )
