"""
master_etl_pipeline.py
======================
Holo-GNN — Master ETL Pipeline
-------------------------------------
Orchestrates parallel preprocessing of all raw biological datasets into
optimised .parquet files for the PyTorch training loop.

Designed for: Local Machine · 32 GB RAM
Parallelism:  concurrent.futures.ProcessPoolExecutor (max 4 workers)
Dataframes:   polars (lazy evaluation, arrow memory, multithreaded)
Streaming:    manual line-by-line generators for VCF and FASTA
"""

from __future__ import annotations

import argparse
import gzip
import logging
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Generator

import polars as pl

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
MAX_WORKERS       = 4           # Optimised for 32GB RAM
FASTA_CHUNK_SIZE  = 1_000_000   # flush to parquet every N sequences
ESM2_MAX_LEN      = 1022        # ESM-2 tokeniser hard limit
AMBIGUOUS_AA      = set("XBZJOU")  # filter out sequences containing these
POLARS_THREADS    = 8           # polars internal thread pool per worker

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("holo_etl")

def _set_polars_threads(n: int = POLARS_THREADS) -> None:
    os.environ["POLARS_MAX_THREADS"] = str(n)

def _elapsed(t0: float) -> str:
    secs = time.time() - t0
    return f"{secs // 60:.0f}m {secs % 60:.1f}s"

def _mb(path: Path) -> float:
    try:
        return path.stat().st_size / (1024 ** 2)
    except OSError:
        return 0.0

# ─────────────────────────────────────────────────────────────────────────────
# 1. FireProtDB Cleaner
# ─────────────────────────────────────────────────────────────────────────────
FIREPROTDB_KEEP_COLS = [
    "SEQUENCE_ID", "MUTANT_ID", "SOURCE_SEQUENCE_ID", "TARGET_SEQUENCE_ID",
    "SUBSTITUTION", "PROTEIN", "ORGANISM",
    "DDG", "DG", "DTM", "DH", "SEQUENCE_LENGTH",
]

def clean_fireprotdb(csv_path: Path, out_dir: Path) -> str:
    _set_polars_threads()
    t0   = time.time()
    name = csv_path.name
    log.info(f"[FireProtDB] START  {name}  ({_mb(csv_path):.1f} MB)")

    try:
        lf = pl.scan_csv(csv_path, infer_schema_length=10_000, ignore_errors=True)
        existing = lf.collect_schema().names()
        keep     = [c for c in FIREPROTDB_KEEP_COLS if c in existing]
        lf       = lf.select(keep)
        lf = lf.filter(pl.col("DDG").is_not_null())

        group_keys = [c for c in ["SEQUENCE_ID", "MUTANT_ID"] if c in keep]
        agg_exprs  = [pl.col("DDG").mean().alias("DDG_mean")]
        scalar_cols = [c for c in keep if c not in group_keys + ["DDG"]]
        agg_exprs  += [pl.col(c).first() for c in scalar_cols]

        lf = lf.group_by(group_keys).agg(agg_exprs)
        df = lf.collect()

        n_rows  = len(df)
        out_path = out_dir / "fireprotdb_clean.parquet"
        df.write_parquet(out_path, compression="zstd", statistics=True)

        msg = f"[FireProtDB] DONE   {n_rows:,} rows → {out_path.name}  ({_mb(out_path):.1f} MB)  [{_elapsed(t0)}]"
        log.info(msg)
        return msg
    except Exception as exc:
        log.error(f"[FireProtDB] FAILED  {exc}", exc_info=True)
        raise

# ─────────────────────────────────────────────────────────────────────────────
# 2. ClinVar VCF Cleaner
# ─────────────────────────────────────────────────────────────────────────────
_CLNSIG_RE  = re.compile(r"CLNSIG=([^;]+)")

def _stream_vcf(vcf_path: Path) -> Generator[dict, None, None]:
    opener = gzip.open if vcf_path.suffix == ".gz" else open
    with opener(vcf_path, "rt", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n\r")
            if line.startswith("#"): continue
            parts = line.split("\t")
            if len(parts) < 8: continue

            chrom, pos, _, ref, alt, _, _, info_raw = parts[:8]
            m = _CLNSIG_RE.search(info_raw)
            if not m: continue
            clnsig = m.group(1)

            clnsig_lower = clnsig.lower()
            if "pathogenic" in clnsig_lower and "likely" not in clnsig_lower and "uncertain" not in clnsig_lower and "conflicting" not in clnsig_lower:
                label = 1
            elif "benign" in clnsig_lower and "likely" not in clnsig_lower and "conflicting" not in clnsig_lower:
                label = 0
            else:
                continue

            yield {"chrom": chrom, "pos": pos, "ref": ref, "alt": alt, "label": label, "clnsig": clnsig}

def clean_clinvar_vcf(vcf_path: Path, out_dir: Path) -> str:
    _set_polars_threads()
    t0   = time.time()
    log.info(f"[ClinVar]    START  {vcf_path.name}  ({_mb(vcf_path):.1f} MB)")

    try:
        records = []
        n_total = n_path = n_benign = 0

        for rec in _stream_vcf(vcf_path):
            n_total += 1
            if rec["label"] == 1: n_path += 1
            else: n_benign += 1
            records.append(rec)
            if n_total % 500_000 == 0:
                log.info(f"[ClinVar]    {n_total:,} rows streamed …  path={n_path:,}  benign={n_benign:,}")

        df = pl.DataFrame(records)
        out_path = out_dir / "clinvar_clean.parquet"
        df.write_parquet(out_path, compression="zstd", statistics=True)

        msg = f"[ClinVar]    DONE   {n_total:,} rows (path={n_path:,} / benign={n_benign:,}) → {out_path.name}  ({_mb(out_path):.1f} MB)  [{_elapsed(t0)}]"
        log.info(msg)
        return msg
    except Exception as exc:
        log.error(f"[ClinVar]    FAILED  {exc}", exc_info=True)
        raise

# ─────────────────────────────────────────────────────────────────────────────
# 3. UniRef50 FASTA Cleaner
# ─────────────────────────────────────────────────────────────────────────────
def _is_valid_sequence(seq: str) -> bool:
    if len(seq) > ESM2_MAX_LEN: return False
    if any(c in AMBIGUOUS_AA for c in seq): return False
    return True

def _stream_fasta(fasta_path: Path) -> Generator[tuple[str, str, str], None, None]:
    opener = gzip.open if fasta_path.suffix == ".gz" else open
    seq_id, description, seq_parts = "", "", []

    with opener(fasta_path, "rt", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n\r")
            if line.startswith(">"):
                if seq_id: yield seq_id, description, "".join(seq_parts)
                header = line[1:].strip()
                parts  = header.split(None, 1)
                seq_id      = parts[0] if parts else ""
                description = parts[1] if len(parts) > 1 else ""
                seq_parts   = []
            else:
                seq_parts.append(line.strip().upper())
    if seq_id and seq_parts:
        yield seq_id, description, "".join(seq_parts)

def clean_uniref_fasta(fasta_path: Path, out_dir: Path) -> str:
    _set_polars_threads()
    t0   = time.time()
    log.info(f"[UniRef50]   START  {fasta_path.name}  ({_mb(fasta_path):.1f} MB)")

    try:
        chunk = []
        part_idx, n_total, n_passed, n_toolong, n_ambig = 0, 0, 0, 0, 0
        part_paths = []

        def _flush_chunk(chunk_list: list[dict], idx: int) -> Path:
            df = pl.DataFrame(chunk_list, schema={"seq_id": pl.Utf8, "description": pl.Utf8, "sequence": pl.Utf8, "seq_len": pl.UInt16})
            p = out_dir / f"uniref50_clean_part_{idx:03d}.parquet"
            df.write_parquet(p, compression="zstd", statistics=True)
            log.info(f"[UniRef50]   Flushed shard {idx:03d}  {len(chunk_list):,} seqs  ({_mb(p):.1f} MB)  [{_elapsed(t0)}]")
            return p

        for seq_id, desc, seq in _stream_fasta(fasta_path):
            n_total += 1
            if len(seq) > ESM2_MAX_LEN:
                n_toolong += 1
                continue
            if any(c in AMBIGUOUS_AA for c in seq):
                n_ambig += 1
                continue

            n_passed += 1
            chunk.append({"seq_id": seq_id, "description": desc, "sequence": seq, "seq_len": len(seq)})

            if n_total % 500_000 == 0:
                log.info(f"[UniRef50]   {n_total:,} read  passed={n_passed:,}  too_long={n_toolong:,}  ambig={n_ambig:,}")

            if len(chunk) >= FASTA_CHUNK_SIZE:
                part_paths.append(_flush_chunk(chunk, part_idx))
                chunk = []
                part_idx += 1

        if chunk:
            part_paths.append(_flush_chunk(chunk, part_idx))

        reject_pct = (1 - n_passed / max(n_total, 1)) * 100
        msg = f"[UniRef50]   DONE   {n_total:,} total → {n_passed:,} passed ({reject_pct:.1f}% rejected)  {len(part_paths)} shards  [{_elapsed(t0)}]"
        log.info(msg)
        return msg
    except Exception as exc:
        log.error(f"[UniRef50]   FAILED  {exc}", exc_info=True)
        raise

# ─────────────────────────────────────────────────────────────────────────────
# 4. MegaScale CSV Cleaner
# ─────────────────────────────────────────────────────────────────────────────
MEGA_SCALE_LABEL_ALIAS = "deltaG"
MEGA_SCALE_KEEP = ["name", "dna_seq", "deltaG_t", "deltaG_t_95CI", "deltaG_c", "deltaG_c_95CI", "log10_K50_t", "log10_K50_c", "fitting_error_t", "fitting_error_c"]

def clean_mega_scale_csvs(csv_paths: list[Path], out_dir: Path) -> str:
    _set_polars_threads()
    t0   = time.time()
    log.info(f"[MegaScale]  START  {len(csv_paths)} file(s)")

    try:
        frames = []
        for csv_path in csv_paths:
            try:
                lf = pl.scan_csv(csv_path, infer_schema_length=10_000, ignore_errors=True)
                schema  = lf.collect_schema().names()
                keep    = [c for c in MEGA_SCALE_KEEP if c in schema]
                lf      = lf.select(keep)

                if "deltaG_t" in keep: lf = lf.with_columns(pl.col("deltaG_t").alias(MEGA_SCALE_LABEL_ALIAS))
                elif "deltaG_c" in keep: lf = lf.with_columns(pl.col("deltaG_c").alias(MEGA_SCALE_LABEL_ALIAS))

                lf = lf.filter(pl.col(MEGA_SCALE_LABEL_ALIAS).is_not_null() & pl.col("dna_seq").is_not_null() & (pl.col("dna_seq").str.len_chars() > 0))
                lf = lf.with_columns(pl.lit(csv_path.name).alias("source_file"))

                df = lf.collect()
                frames.append(df)
            except Exception as inner_exc:
                log.error(f"[MegaScale]  SKIP {csv_path.name}: {inner_exc}")
                continue

        if not frames: raise ValueError("No MegaScale data survived cleaning.")
        combined  = pl.concat(frames, how="diagonal_relaxed")
        out_path  = out_dir / "mega_scale_clean.parquet"
        combined.write_parquet(out_path, compression="zstd", statistics=True)

        msg = f"[MegaScale]  DONE   {len(combined):,} rows → {out_path.name}  ({_mb(out_path):.1f} MB)  [{_elapsed(t0)}]"
        log.info(msg)
        return msg
    except Exception as exc:
        log.error(f"[MegaScale]  FAILED  {exc}", exc_info=True)
        raise

# ─────────────────────────────────────────────────────────────────────────────
# PROCESS POOL ENTRY POINTS
# ─────────────────────────────────────────────────────────────────────────────
def _run_fireprotdb(args): return clean_fireprotdb(Path(args[0]), Path(args[1]))
def _run_clinvar(args): return clean_clinvar_vcf(Path(args[0]), Path(args[1]))
def _run_uniref(args): return clean_uniref_fasta(Path(args[0]), Path(args[1]))
def _run_mega_scale(args): return clean_mega_scale_csvs([Path(p) for p in args[0]], Path(args[1]))

# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────
def discover_inputs(input_dir: Path) -> dict[str, list[Path]]:
    tasks = {"fireprotdb": [], "clinvar": [], "uniref": [], "mega_scale": []}
    for dirpath_str, dirnames, filenames in os.walk(input_dir):
        dirpath  = Path(dirpath_str)
        for fname in filenames:
            if fname.startswith("."): continue
            p, stem, ext = dirpath / fname, fname.lower(), Path(fname).suffix.lower()
            if "fireprotdb" in stem and ext == ".csv": tasks["fireprotdb"].append(p)
            elif "clinvar" in stem and ext == ".vcf": tasks["clinvar"].append(p)
            elif "uniref" in stem and ext in {".fasta", ".fa"}: tasks["uniref"].append(p)
            elif ext == ".csv" and any(kw in stem for kw in ["tsuboyama", "lib1_k50", "lib2_k50", "lib3_k50", "lib4_k50"]):
                tasks["mega_scale"].append(p)
    return tasks

def main(input_dir: str, output_dir: str) -> None:
    t_global = time.time()
    inp, out = Path(input_dir), Path(output_dir)

    log.info("=" * 70)
    log.info("  Holo-GNN  Master ETL Pipeline")
    log.info(f"  Input  : {inp}")
    log.info(f"  Output : {out}")
    log.info(f"  Workers: {MAX_WORKERS}")
    log.info("=" * 70)

    if not inp.exists():
        log.error(f"Input directory not found: {inp}")
        sys.exit(1)

    out.mkdir(parents=True, exist_ok=True)
    tasks = discover_inputs(inp)
    jobs = []

    if tasks["fireprotdb"]: jobs.extend([(_run_fireprotdb, (str(fp), str(out))) for fp in tasks["fireprotdb"]])
    if tasks["clinvar"]: jobs.extend([(_run_clinvar, (str(fp), str(out))) for fp in tasks["clinvar"]])
    if tasks["uniref"]: jobs.extend([(_run_uniref, (str(fp), str(out))) for fp in tasks["uniref"]])
    if tasks["mega_scale"]: jobs.append((_run_mega_scale, ([str(p) for p in tasks["mega_scale"]], str(out))))

    if not jobs:
        log.error("No recognised input files found. Exiting.")
        sys.exit(1)

    results, failures = [], []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_map = {pool.submit(fn, args): fn.__name__ for fn, args in jobs}
        for future in as_completed(future_map):
            try: results.append(future.result())
            except Exception as exc: failures.append(f"{future_map[future]} FAILED: {exc}")

    log.info("\n" + "=" * 70)
    log.info("  ETL PIPELINE COMPLETE")
    log.info(f"  Wall time  : {_elapsed(t_global)}")
    log.info("=" * 70)
    
    if failures:
        for f in failures: log.error(f"  ❌  {f}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", default="data", help="Input directory")
    parser.add_argument("--output", "-o", default="CLEANED_DATA", help="Output directory")
    args = parser.parse_args()
    main(args.input, args.output)