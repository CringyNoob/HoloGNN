"""
pretrain_uniref.py
==================
Holo-GNN — Unsupervised GATv2 Pre-Training via Masked Language Modeling
"""
from __future__ import annotations

import argparse
import logging
import time
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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

ESM2_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
ESM2_VOCAB_SIZE = 33        
ESM2_MASK_TOKEN = 32        
ESM2_SPECIAL_TOKENS = {0, 1, 2, 3}
MLM_MASK_PROB = 0.15

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
    input_ids:     torch.Tensor,
    attention_mask: torch.Tensor,
    mask_prob:     float = MLM_MASK_PROB,
    mask_token_id: int   = ESM2_MASK_TOKEN,
    special_tokens: set  = ESM2_SPECIAL_TOKENS,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels    = input_ids.clone()
    masked    = input_ids.clone()

    is_real   = attention_mask.bool()
    is_eligible = is_real.clone()
    for st in special_tokens:
        is_eligible &= (input_ids != st)

    prob_matrix  = torch.full(input_ids.shape, mask_prob, device=input_ids.device)
    prob_matrix[~is_eligible] = 0.0
    should_mask  = torch.bernoulli(prob_matrix).bool()

    labels[~should_mask] = -100

    replace_mask = torch.bernoulli(torch.full(input_ids.shape, 0.80, device=input_ids.device)).bool()
    replace_mask &= should_mask
    masked[replace_mask] = mask_token_id

    replace_random = torch.bernoulli(torch.full(input_ids.shape, 0.50, device=input_ids.device)).bool()
    replace_random &= should_mask & ~replace_mask
    random_tokens   = torch.randint(4, 24, input_ids.shape, device=input_ids.device)
    masked[replace_random] = random_tokens[replace_random]

    return masked, labels

class HoloGNN_MLM(nn.Module):
    def __init__(self, output_dim: int = 320, vocab_size: int = ESM2_VOCAB_SIZE):
        super().__init__()
        self.backbone = HoloGNNBackbone(output_dim=output_dim)
        self.mlm_head = MLM_Head(input_dim=output_dim, vocab_size=vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, mechanistic_features: torch.Tensor) -> torch.Tensor:
        node_embeddings, _ = self.backbone(
            input_ids            = input_ids,
            attention_mask       = attention_mask,
            mechanistic_features = mechanistic_features,
        )
        return self.mlm_head(node_embeddings)

def pretrain(
    parquet_dir:  str,
    output_path:  str,
    epochs:       int   = 3,
    batch_size:   int   = 16,
    max_length:   int   = 512,
    lr:           float = 1e-4,
    weight_decay: float = 0.01,
    num_workers:  int   = 0,
    grad_clip:    float = 1.0,
    log_interval: int   = 100,
    resume_from:  Optional[str] = None,
) -> None:
    from src.device import get_device, describe_device
    t_global = time.time()
    device   = get_device()
    describe_device(device)          
    log.info("=" * 68)
    log.info("  Holo-GNN Unsupervised Pre-Training — MLM on UniRef50")
    log.info(f"  Device     : {device}")
    if torch.cuda.is_available():
        log.info(f"  GPU        : {torch.cuda.get_device_name(0)}")
        log.info(f"  VRAM       : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    log.info("=" * 68)

    dataset = UniRefDataset(parquet_dir=parquet_dir, max_length=max_length, shuffle_shards=True)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=False, 
        persistent_workers=(num_workers > 0), 
        prefetch_factor=2 if num_workers > 0 else None, 
        drop_last=True,
    )
    total_steps = len(loader) * epochs

    model = HoloGNN_MLM(output_dim=320, vocab_size=ESM2_VOCAB_SIZE).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    scaler    = GradScaler(enabled=torch.cuda.is_available())

    start_epoch = 1
    if resume_from and Path(resume_from).exists():
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt.get("epoch", 1) + 1

    # --- TENSORBOARD INIT ---
    writer = SummaryWriter(log_dir="runs/uniref_pretrain")
    global_step = 0
    
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = epoch_acc = 0.0
        n_batches = n_masked = 0
        t_epoch = time.time()

        for step, batch in enumerate(loader, 1):
            if step == 1:
                log.info("🔥 BATCH 1 SUCCESSFULLY LOADED! The engine is firing.")

            input_ids      = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            mech_features  = batch["mechanistic_features"].to(device, non_blocking=True)

            masked_ids, labels = apply_mlm_mask(input_ids, attention_mask)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=torch.cuda.is_available()):
                logits = model(input_ids=masked_ids, attention_mask=attention_mask, mechanistic_features=mech_features)
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

            epoch_loss += loss.item()
            epoch_acc  += acc
            n_masked   += n_mask_pos
            n_batches  += 1
            global_step += 1

            # --- TENSORBOARD TELEMETRY ---
            writer.add_scalar("Training/Loss", loss.item(), global_step)
            writer.add_scalar("Training/Accuracy", acc, global_step)
            writer.add_scalar("System/Learning_Rate", optimizer.param_groups[0]["lr"], global_step)

            if step % log_interval == 0 or step == len(loader):
                log.info(
                    f"Epoch {epoch} step {step}/{len(loader)} | "
                    f"loss={epoch_loss/n_batches:.4f} acc={epoch_acc/n_batches:.3f} | "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )

        epoch_avg_loss = epoch_loss / max(n_batches, 1)
        epoch_ckpt = Path(output_path).parent / f"mlm_epoch_{epoch:02d}.pth"
        torch.save({
            "epoch": epoch, "global_step": global_step,
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        }, epoch_ckpt)

    writer.close()
    torch.save(model.backbone.state_dict(), output_path)
    log.info(f"PRE-TRAINING COMPLETE. Backbone saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_dir", "-d", default="CLEANED_DATA/")
    parser.add_argument("--output", "-o", dest="output_path", default="uniref_pretrained_weights.pth")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--resume_from", type=str, default=None)
    args = parser.parse_args()

    pretrain(**vars(args))