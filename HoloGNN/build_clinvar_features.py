"""
build_clinvar_features.py
=========================
Re-clean ClinVar from the **raw VCF**, keeping the predictive annotations the
original ETL discarded.

The old `clinvar_clean.parquet` kept only `chrom, pos, ref, alt, label` — bare
alleles (79% single-nucleotide) with no context, which capped the pathogenicity
classifier at AUROC ~0.61. The raw VCF INFO field actually carries the strong
signal:

    GENEINFO   -> gene symbol            (pathogenicity is gene-dependent)
    MC         -> molecular consequence  (missense / nonsense / synonymous / ...)
    AF_ESP/EXAC/TGP -> population allele frequency (common variant => benign)
    CLNVC      -> variant type
    CLNSIG     -> clinical significance (the label)

This script streams the VCF and writes a richer parquet with those columns plus
the original ref/alt/label, so we can train a pathogenicity model that actually
has something to learn from.

Labels (more inclusive than the old ETL, which dropped "Likely_*"):
    pathogenic / likely_pathogenic -> 1
    benign     / likely_benign     -> 0
    uncertain / conflicting / other -> skipped

Run:
    python build_clinvar_features.py
    python build_clinvar_features.py --input ../data/clinvar/clinvar.vcf
"""
from __future__ import annotations

import argparse
import gzip
import time
from pathlib import Path

import pandas as pd


def _open(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", errors="replace") if path.suffix == ".gz" \
        else open(path, "r", encoding="utf-8", errors="replace")


def _parse_info(info: str) -> dict:
    d = {}
    for kv in info.split(";"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            d[k] = v
        else:
            d[kv] = True
    return d


def _label_from_clnsig(clnsig: str):
    s = clnsig.lower()
    if "conflict" in s or "uncertain" in s:
        return None
    has_path = "pathogenic" in s
    has_benign = "benign" in s
    if has_path and not has_benign:
        return 1
    if has_benign and not has_path:
        return 0
    return None


def _consequence(mc: str) -> str:
    # MC=SO:0001583|missense_variant,SO:0001627|intron_variant -> "missense_variant"
    if not mc or mc is True:
        return "unknown"
    first = mc.split(",")[0]
    return first.split("|")[-1] if "|" in first else first


def _allele_freq(info: dict):
    vals = []
    for k in ("AF_ESP", "AF_EXAC", "AF_TGP"):
        v = info.get(k)
        if v and v is not True:
            try:
                vals.append(float(v))
            except ValueError:
                pass
    if not vals:
        return 0.0, 0           # af, af_known
    return max(vals), 1


def build(args):
    src = Path(args.input)
    if not src.exists():
        raise FileNotFoundError(f"VCF not found: {src}")
    t0 = time.time()
    print(f"[clinvar] streaming {src} ({src.stat().st_size/1e6:.0f} MB) ...")

    rows = []
    n_total = n_kept = n_path = n_benign = 0
    with _open(src) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue
            n_total += 1
            chrom, pos, _id, ref, alt = parts[0], parts[1], parts[2], parts[3], parts[4]
            info = _parse_info(parts[7])

            clnsig = info.get("CLNSIG")
            if not clnsig or clnsig is True:
                continue
            label = _label_from_clnsig(clnsig)
            if label is None:
                continue

            gene = info.get("GENEINFO", "")
            gene = gene.split(":")[0] if gene and gene is not True else ""
            consequence = _consequence(info.get("MC", ""))
            af, af_known = _allele_freq(info)
            cln_vc = info.get("CLNVC", "")

            rows.append({
                "chrom": chrom, "pos": int(pos) if pos.isdigit() else 0,
                "ref": ref, "alt": alt, "label": label,
                "gene": gene, "consequence": consequence,
                "af": af, "af_known": af_known, "variant_type": cln_vc,
                "clnsig": clnsig,
            })
            n_kept += 1
            n_path += (label == 1)
            n_benign += (label == 0)
            if n_total % 500_000 == 0:
                print(f"[clinvar]   {n_total:,} scanned, {n_kept:,} kept "
                      f"(path={n_path:,} / benign={n_benign:,})  [{time.time()-t0:.0f}s]")

    df = pd.DataFrame(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    print(f"\n[clinvar] wrote {len(df):,} labeled variants -> {out}")
    print(f"[clinvar] label balance: pathogenic={n_path:,} ({100*n_path/max(n_kept,1):.1f}%) "
          f"/ benign={n_benign:,}")
    print(f"[clinvar] top consequences:")
    print(df["consequence"].value_counts().head(8).to_string())
    print(f"[clinvar] AF known for {100*df['af_known'].mean():.1f}% of rows; "
          f"distinct genes: {df['gene'].nunique():,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Re-clean ClinVar VCF with predictive features.")
    ap.add_argument("--input", default="../data/clinvar/clinvar.vcf",
                    help="Raw ClinVar VCF (.vcf or .vcf.gz).")
    ap.add_argument("--output", default="CLEANED_DATA/clinvar_features.parquet")
    args = ap.parse_args()
    build(args)
