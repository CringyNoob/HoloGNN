"""
predict.py
==========
Command-line ΔΔG (stability change) predictor for Holo-GNN.

Given a wild-type protein sequence and a point mutation (e.g. ``M1A``), this
script reports the predicted change in folding free energy ΔΔG.

Two modes:
  • FULL   — when the trained weights (``holognn_stability_final.pth``) and the
             torch / transformers / torch_geometric stack are available, the real
             Holo-GNN Siamese pass is used:  model((data_wt, data_mt), task="idr").
  • DEMO   — otherwise, a deterministic biophysical heuristic (see
             ``src/heuristics.py``) is used so the script is always runnable.
             The public repo ships no weights, so demo mode is the default.

Sign convention (ΔG_mut − ΔG_wt):  ΔΔG > 0 → stabilising;  ΔΔG < 0 → destabilising.

Examples
--------
    python predict.py                       # built-in ubiquitin M1A demo
    python predict.py --seq MQIFVK... --mut M1A
    python predict.py --wt MQIFVK... --mt AQIFVK...
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse

from src.heuristics import apply_mutation, heuristic_ddg, heuristic_confidence

MODEL_PATH  = os.environ.get("HOLOGNN_WEIGHTS", "holognn_stability_final.pth")
MAX_LENGTH  = 512

# Ubiquitin (wild type) + the classic M1A mutant used as the built-in example.
_DEMO_WT  = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
_DEMO_MUT = "M1A"


# ---------------------------------------------------------------------------
# FULL mode — real Holo-GNN Siamese inference
# ---------------------------------------------------------------------------
def _try_load_model():
    """Return (model, tokenizer, device, torch) or None if FULL mode is unavailable."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        import torch
        from transformers import EsmTokenizer
        from src.full_model import HoloGNN
        from src.checkpoint import load_checkpoint
    except Exception as exc:  # noqa: BLE001
        print(f"[demo] inference stack unavailable ({exc.__class__.__name__}); using heuristic.")
        return None

    from src.device import describe_device
    device = describe_device()
    model = HoloGNN()
    try:
        _, meta = load_checkpoint(MODEL_PATH, model, device, strict=False)
        # predict.py runs the Siamese ΔΔG path; warn if this checkpoint's head
        # for that path was never trained (would yield meaningless predictions).
        trained = meta.get("trained_heads")
        if trained is not None and "siamese_head" not in trained:
            print(f"[warn] {MODEL_PATH} was trained for {meta.get('trained_task')!r} "
                  f"(heads={trained}); the Siamese ΔΔG head may be untrained.")
    except Exception as exc:  # noqa: BLE001
        print(f"[demo] could not load weights from {MODEL_PATH} ({exc}); using heuristic.")
        return None
    model.to(device).eval()
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    return model, tokenizer, device, torch


def _make_batch(seq, tokenizer, device, torch):
    """Build a DataBatch with input_ids, mask, mechanistic_features and edge_index=None."""
    from src.dataset import mechanistic_features_for_protein

    enc = tokenizer(seq, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    input_ids = enc["input_ids"].to(device)
    mask      = enc["attention_mask"].to(device)
    L         = input_ids.size(1)
    mech      = mechanistic_features_for_protein(seq, L).unsqueeze(0).to(device)

    class DataBatch:  # lightweight attribute carrier
        pass

    data = DataBatch()
    data.input_ids            = input_ids
    data.mask                 = mask
    data.mechanistic_features = mech
    data.edge_index           = None
    return data


def predict_full(wt_seq, mt_seq, bundle):
    model, tokenizer, device, torch = bundle
    data_wt = _make_batch(wt_seq, tokenizer, device, torch)
    data_mt = _make_batch(mt_seq, tokenizer, device, torch)
    with torch.no_grad():
        dG_wt_to_mt, _dG_mt_to_wt = model((data_wt, data_mt), task="idr")
    return float(dG_wt_to_mt.item())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Holo-GNN ΔΔG stability-change predictor")
    ap.add_argument("--seq", help="wild-type sequence (use with --mut)")
    ap.add_argument("--mut", help="point mutation, e.g. M1A (use with --seq)")
    ap.add_argument("--wt",  help="wild-type sequence (use with --mt)")
    ap.add_argument("--mt",  help="full mutant sequence (use with --wt)")
    args = ap.parse_args()

    # Resolve WT / MT sequences + a human-readable label for the mutation.
    if args.wt and args.mt:
        wt_seq, mt_seq, label = args.wt.upper(), args.mt.upper(), "WT→MT"
        mut_pos = None
    else:
        wt_seq = (args.seq or _DEMO_WT).upper()
        mut    = (args.mut or _DEMO_MUT)
        mt_seq, idx, wt_aa, mut_aa = apply_mutation(wt_seq, mut)
        label   = f"{wt_aa}{idx + 1}{mut_aa}"
        mut_pos = (wt_aa, mut_aa)

    bundle = _try_load_model()
    if bundle is not None:
        ddg = predict_full(wt_seq, mt_seq, bundle)
        ci_low, ci_high = heuristic_confidence(ddg)
        mode = "FULL (trained Holo-GNN)"
    else:
        if mut_pos is None:
            # Two arbitrary sequences in demo mode: sum per-position heuristics.
            ddg = sum(
                heuristic_ddg(a, b)
                for a, b in zip(wt_seq, mt_seq) if a != b
            )
        else:
            ddg = heuristic_ddg(*mut_pos)
        ci_low, ci_high = heuristic_confidence(ddg)
        mode = "DEMO (biophysical heuristic — no trained weights)"

    verdict = "STABILISING" if ddg > 0 else "DESTABILISING"
    print(f"\n=== Holo-GNN stability prediction [{mode}] ===")
    print(f"Mutation        : {label}")
    print(f"WT length       : {len(wt_seq)}")
    print(f"Predicted ΔΔG   : {ddg:+.4f} kcal/mol   (95% CI {ci_low:+.2f} … {ci_high:+.2f})")
    print(f"Conclusion      : {verdict}")
    print("Convention      : ΔΔG > 0 stabilising, ΔΔG < 0 destabilising (ΔG_mut − ΔG_wt)\n")


if __name__ == "__main__":
    main()
