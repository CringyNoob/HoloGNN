"""
build_megascale_pairs.py
========================
Build real WT/MT ΔΔG **sequence pairs** for the Siamese stability head from the
cleaned MegaScale parquet.

Why this exists
---------------
``fireprotdb_clean.parquet`` ships with ``SOURCE_SEQUENCE_ID`` /
``TARGET_SEQUENCE_ID`` **empty** (0 / 586k populated), so ``train_siamese.py``
would tokenize empty strings — no biological signal.  The web app and
``predict.py`` use that Siamese ``idr`` head, so it must train on real
sequences.

MegaScale (``mega_scale_clean.parquet``) has, for ~1,400 base proteins, a
wild-type row (``name == base``) plus many variant rows (``name == base_…``),
each with a folding ΔG.  We pair every variant with its base WT:

    WT  protein = translate(dna_seq of  name==base)
    MT  protein = translate(dna_seq of  variant)
    ΔΔG         = deltaG(variant) − deltaG(WT)        # >0 stabilising

Output columns match what ``FireProtDataset`` consumes
(``SOURCE_SEQUENCE_ID``, ``TARGET_SEQUENCE_ID``, ``DDG_mean`` + metadata), so
``train_siamese.py`` can read the result with no code changes.

Run:
    python build_megascale_pairs.py
    python build_megascale_pairs.py --max-pairs 150000 --max-per-base 200
"""
from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Seq import Seq

warnings.filterwarnings("ignore")   # Bio.Seq partial-codon warnings

# Tsuboyama single-point-mutant naming, e.g. "1A32.pdb_A45D" (WT->A, pos 45, ->D).
_POINT_MUT_RE = re.compile(r"_([A-Z]\d+[A-Z])$")


def _translate(dna: str) -> str:
    """DNA → protein (stop-truncated), matching MegaScaleDataset's convention."""
    try:
        return str(Seq(str(dna)).translate(to_stop=True))
    except Exception:
        return ""


def build(args):
    rng = np.random.default_rng(args.seed)
    src = Path(args.input)
    print(f"[pairs] reading {src} …")
    df = pd.read_parquet(src, columns=["name", "dna_seq", "deltaG"])
    df["name"] = df["name"].astype(str)

    # 1. Drop sentinel / invalid ΔG (e.g. -15.0 = unmeasurable).
    bad = (df["deltaG"] <= -14) | (df["deltaG"].abs() > 50) | df["deltaG"].isna()
    df = df[~bad].copy()
    df["base"] = df["name"].str.split("_").str[0]
    print(f"[pairs] {len(df):,} valid rows after dG filter.")

    # 2. WT lookup: the bare-name row per base.
    wt_rows = df[df["name"] == df["base"]].drop_duplicates("base", keep="first")
    wt_dg = dict(zip(wt_rows["base"], wt_rows["deltaG"]))
    wt_dna = dict(zip(wt_rows["base"], wt_rows["dna_seq"]))
    print(f"[pairs] {len(wt_rows):,} base proteins have a wild-type row.")

    # 3. Candidate variants: those whose base has a WT.
    variants = df[(df["name"] != df["base"]) & (df["base"].isin(wt_dg))].copy()
    print(f"[pairs] {len(variants):,} pairable variant rows.")

    # 3b. Optionally keep only single-point mutants (Tsuboyama '<base>.pdb_X##Y').
    #     This matches the app/predict use case (one point mutation) and gives a
    #     cleaner, benchmark-grade ΔΔG signal than multi-residue designed variants.
    variants["substitution"] = variants["name"].str.extract(_POINT_MUT_RE, expand=False)
    if args.point_mutants_only:
        variants = variants[variants["substitution"].notna()].copy()
        print(f"[pairs] {len(variants):,} single-point-mutant rows kept (point-mutants-only).")

    # 4. Cap per base so a single deep-scan protein can't dominate, then sample.
    #    Shuffle first, then keep the first N per base via cumcount (random subset).
    variants = variants.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    if args.max_per_base > 0:
        keep = variants.groupby("base").cumcount() < args.max_per_base
        variants = variants[keep]
    if args.max_pairs and len(variants) > args.max_pairs:
        variants = variants.sample(args.max_pairs, random_state=args.seed)
    variants = variants.reset_index(drop=True)
    print(f"[pairs] {len(variants):,} variants selected (cap/base={args.max_per_base}, "
          f"max={args.max_pairs}).")

    # 5. Translate (only the WT set + selected variants → cheap).
    print("[pairs] translating WT sequences …")
    wt_prot = {b: _translate(d) for b, d in wt_dna.items()}

    print("[pairs] translating variant sequences + assembling pairs …")
    rows = []
    for r in variants.to_dict("records"):
        base = r["base"]
        wt_seq = wt_prot.get(base, "")
        mt_seq = _translate(r["dna_seq"])
        if len(wt_seq) < 10 or len(mt_seq) < 10 or wt_seq == mt_seq:
            continue                      # skip empty / no-op pairs
        if args.point_mutants_only:
            # Enforce a TRUE single substitution at the sequence level (names can
            # hide double mutants like _V54S_I6F or stop-codon truncations).
            if len(wt_seq) != len(mt_seq):
                continue
            if sum(a != b for a, b in zip(wt_seq, mt_seq)) != 1:
                continue
        rows.append({
            "SEQUENCE_ID":        base,
            "MUTANT_ID":          r["name"],
            "SOURCE_SEQUENCE_ID": wt_seq,         # WT protein (FireProtDataset reads these)
            "TARGET_SEQUENCE_ID": mt_seq,         # MT protein
            "DDG_mean":           float(r["deltaG"] - wt_dg[base]),
            "SUBSTITUTION":       r.get("substitution") or "",
            "SEQUENCE_LENGTH":    len(wt_seq),
        })

    out_df = pd.DataFrame(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out, index=False)
    print(f"\n[pairs] wrote {len(out_df):,} WT/MT pairs -> {out}")
    print(f"[pairs] ddG: mean={out_df['DDG_mean'].mean():.3f} "
          f"std={out_df['DDG_mean'].std():.3f} "
          f"min={out_df['DDG_mean'].min():.3f} max={out_df['DDG_mean'].max():.3f}")
    print(f"[pairs] WT protein length: median={int(out_df['SEQUENCE_LENGTH'].median())} "
          f"max={int(out_df['SEQUENCE_LENGTH'].max())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build MegaScale WT/MT ΔΔG pairs for the Siamese head.")
    ap.add_argument("--input",  default="CLEANED_DATA/mega_scale_clean.parquet")
    ap.add_argument("--output", default="data/megascale_siamese/megascale_ddg_pairs.parquet")
    ap.add_argument("--max-pairs",    type=int, default=150000,
                    help="Total pairs to keep (0 = all).")
    ap.add_argument("--max-per-base", type=int, default=200,
                    help="Cap variants per base protein before sampling (0 = no cap).")
    ap.add_argument("--point-mutants-only", action="store_true",
                    help="Keep only single-point mutants (Tsuboyama '<base>.pdb_X##Y'); "
                         "matches the app's single-mutation inference and is benchmark-grade.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build(args)
