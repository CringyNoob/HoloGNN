"""
master_etl_pipeline.py
======================
Holo-GNN V5.0 — Master ETL Pipeline
-------------------------------------
Orchestrates parallel preprocessing of all raw biological datasets into
optimised .parquet files for the PyTorch training loop.

Designed for: Google Vertex AI · 16 vCPUs · 64 GB RAM
Parallelism:  concurrent.futures.ProcessPoolExecutor (max 14 workers)
Dataframes:   polars (lazy evaluation, arrow memory, multithreaded)
Streaming:    manual line-by-line generators for VCF (1.7 GB) and FASTA (23 GB)

Datasets processed
------------------
  1. FireProtDB CSV  → fireprotdb_clean.parquet
  2. ClinVar VCF     → clinvar_clean.parquet
  3. UniRef50 FASTA  → uniref50_clean_part_NNN.parquet  (1M-seq chunks)
  4. MegaScale CSVs  → mega_scale_clean.parquet
       • Tsuboyama2023_Dataset1_20230416.csv
       • Tsuboyama2023_Dataset2_Dataset3_20230416.csv
       • Lib1_K50dG.csv … Lib4_K50dG.csv

Usage
-----
  python master_etl_pipeline.py \\
      --input  "D:/ML MODELS/HOLO_GNN_PROJECT/DATA" \\
      --output "D:/ML MODELS/HOLO_GNN_PROJECT/CLEANED_DATA"
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
MAX_WORKERS       = 14          # leave 2 cores for the OS
FASTA_CHUNK_SIZE  = 1_000_000  # flush to parquet every N sequences  (RAM guard)
ESM2_MAX_LEN      = 1022        # ESM-2 tokeniser hard limit
AMBIGUOUS_AA      = set("XBZJOU")  # filter out sequences containing these
POLARS_THREADS    = 8           # polars internal thread pool per worker

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING — rich timestamped output, mirrors Vertex AI log format
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("holo_etl")


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _set_polars_threads(n: int = POLARS_THREADS) -> None:
    """Cap polars internal thread pool — important inside child processes."""
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
#    Input  : fireprotdb_*.csv  (confirmed columns: EXPERIMENT_ID, SEQUENCE_ID,
#              MUTANT_ID, DDG, DG, DH, DOMAINOME_DDG, … +33 more)
#    Output : fireprotdb_clean.parquet
#    Logic  :
#      • Drop rows where DDG is null (primary regression target).
#      • Group by (SEQUENCE_ID, MUTANT_ID) and take mean(DDG) to resolve
#        duplicate measurements from different labs / experimental conditions.
#      • Retain SUBSTITUTION, PROTEIN, ORGANISM, DTM, DG as auxiliary columns.
# ─────────────────────────────────────────────────────────────────────────────

FIREPROTDB_KEEP_COLS = [
    "SEQUENCE_ID", "MUTANT_ID", "SOURCE_SEQUENCE_ID", "TARGET_SEQUENCE_ID",
    "SUBSTITUTION", "PROTEIN", "ORGANISM",
    "DDG", "DG", "DTM", "DH", "SEQUENCE_LENGTH",
]

def clean_fireprotdb(csv_path: Path, out_dir: Path) -> str:
    """
    ETL for FireProtDB CSV.

    Returns a completion summary string for the orchestrator log.
    """
    _set_polars_threads()
    t0   = time.time()
    name = csv_path.name
    log.info(f"[FireProtDB] START  {name}  ({_mb(csv_path):.1f} MB)")

    try:
        # ── Read with polars lazy API ─────────────────────────────────────────
        # infer_schema_length=10000 avoids type mismatches in large heterogeneous CSV.
        lf = pl.scan_csv(
            csv_path,
            infer_schema_length=10_000,
            ignore_errors=True,           # skip malformed rows silently
        )

        # ── Select only columns that exist in this file ───────────────────────
        existing = lf.collect_schema().names()
        keep     = [c for c in FIREPROTDB_KEEP_COLS if c in existing]
        lf       = lf.select(keep)

        # ── Drop nulls in DDG (primary label) ────────────────────────────────
        lf = lf.filter(pl.col("DDG").is_not_null())

        # ── Group by identity pair, aggregate DDG to mean ─────────────────────
        # This resolves conflicts where different labs measured the same mutant.
        group_keys = [c for c in ["SEQUENCE_ID", "MUTANT_ID"] if c in keep]
        agg_exprs  = [pl.col("DDG").mean().alias("DDG_mean")]

        # Carry through auxiliary scalar columns that are constant per group
        scalar_cols = [c for c in keep if c not in group_keys + ["DDG"]]
        agg_exprs  += [pl.col(c).first() for c in scalar_cols]

        lf = lf.group_by(group_keys).agg(agg_exprs)

        df = lf.collect()

        n_rows  = len(df)
        out_path = out_dir / "fireprotdb_clean.parquet"
        df.write_parquet(out_path, compression="zstd", statistics=True)

        msg = (f"[FireProtDB] DONE   {n_rows:,} rows → {out_path.name}"
               f"  ({_mb(out_path):.1f} MB)  [{_elapsed(t0)}]")
        log.info(msg)
        return msg

    except Exception as exc:
        log.error(f"[FireProtDB] FAILED  {exc}", exc_info=True)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# 2. ClinVar VCF Cleaner
#    Input  : clinvar.vcf  (1.77 GB)  — columns: CHROM, POS, ID, REF, ALT,
#             QUAL, FILTER, INFO (44 meta-lines before #CHROM header)
#    Output : clinvar_clean.parquet
#    Logic  :
#      • Stream line by line; never load the full 1.7 GB into RAM.
#      • Skip all ## meta-lines and the #CHROM header line.
#      • Parse the INFO field for CLNSIG (ClinVar significance).
#      • Strict Pathogenic  → label = 1  (excludes "Likely_pathogenic")
#      • Benign             → label = 0  (excludes "Likely_benign")
#      • All other variants → skip       (uncertain, conflicting, etc.)
#      • Extract: CHROM, POS, REF, ALT, label, CLNSIG string.
# ─────────────────────────────────────────────────────────────────────────────

# Regex compiled once, reused for every row
_CLNSIG_RE  = re.compile(r"CLNSIG=([^;]+)")
_GENEINFO_RE = re.compile(r"GENEINFO=([^;]+)")

def _stream_vcf(vcf_path: Path) -> Generator[dict, None, None]:
    """
    Line-by-line VCF generator.  Handles plain and gzip-compressed files.
    Yields dicts with keys: chrom, pos, ref, alt, info_raw, clnsig, label.
    """
    opener = gzip.open if vcf_path.suffix == ".gz" else open
    mode   = "rt"

    with opener(vcf_path, mode, encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n\r")

            # Skip meta-lines and column header
            if line.startswith("#"):
                continue

            # VCF columns: CHROM POS ID REF ALT QUAL FILTER INFO [FORMAT …]
            parts = line.split("\t")
            if len(parts) < 8:
                continue

            chrom, pos, _, ref, alt, _, _, info_raw = parts[:8]

            # ── Extract CLNSIG from INFO string ───────────────────────────
            m = _CLNSIG_RE.search(info_raw)
            if not m:
                continue
            clnsig = m.group(1)

            # ── Assign binary label (strict) ──────────────────────────────
            # Pathogenic: INFO must contain "Pathogenic" but NOT "Likely",
            # "Uncertain", or "Conflicting" (those are soft / disputed calls,
            # not ground-truth labels — e.g. "Conflicting_interpretations_of_
            # pathogenicity" must be skipped, not counted as pathogenic).
            clnsig_lower = clnsig.lower()
            if "pathogenic" in clnsig_lower and \
               "likely" not in clnsig_lower and \
               "uncertain" not in clnsig_lower and \
               "conflicting" not in clnsig_lower:
                label = 1
            elif "benign" in clnsig_lower and \
                 "likely" not in clnsig_lower and \
                 "conflicting" not in clnsig_lower:
                label = 0
            else:
                continue   # skip uncertain / conflicting / other

            yield {
                "chrom":   chrom,
                "pos":     pos,
                "ref":     ref,
                "alt":     alt,
                "label":   label,
                "clnsig":  clnsig,
            }


def clean_clinvar_vcf(vcf_path: Path, out_dir: Path) -> str:
    _set_polars_threads()
    t0   = time.time()
    log.info(f"[ClinVar]    START  {vcf_path.name}  ({_mb(vcf_path):.1f} MB)")

    try:
        records: list[dict] = []
        n_total = n_path = n_benign = 0

        for rec in _stream_vcf(vcf_path):
            n_total += 1
            if rec["label"] == 1:
                n_path += 1
            else:
                n_benign += 1
            records.append(rec)

            # ── Progress heartbeat every 500k rows ───────────────────────
            if n_total % 500_000 == 0:
                log.info(f"[ClinVar]    {n_total:,} rows streamed …"
                         f"  path={n_path:,}  benign={n_benign:,}")

        df = pl.DataFrame(records)

        out_path = out_dir / "clinvar_clean.parquet"
        df.write_parquet(out_path, compression="zstd", statistics=True)

        msg = (f"[ClinVar]    DONE   {n_total:,} rows "
               f"(path={n_path:,} / benign={n_benign:,}) → {out_path.name}"
               f"  ({_mb(out_path):.1f} MB)  [{_elapsed(t0)}]")
        log.info(msg)
        return msg

    except Exception as exc:
        log.error(f"[ClinVar]    FAILED  {exc}", exc_info=True)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# 3. UniRef50 FASTA Cleaner
#    Input  : uniref50.fasta  (23 GB)
#    Output : uniref50_clean_part_000.parquet, _001, … (1M-seq chunks)
#    Logic  :
#      • Stream line by line; accumulate current sequence in a buffer.
#      • On ">" header line: finalise previous sequence and emit if valid.
#      • Validity filters:
#          a) len(seq) <= 1022   (ESM-2 OOM guard)
#          b) no character in AMBIGUOUS_AA
#      • Every FASTA_CHUNK_SIZE valid sequences → write parquet, clear buffer.
#      • Parquet columns: seq_id (header up to first space), description, sequence.
# ─────────────────────────────────────────────────────────────────────────────

def _is_valid_sequence(seq: str) -> bool:
    """Return True if the sequence passes all ESM-2 pre-filtering criteria."""
    if len(seq) > ESM2_MAX_LEN:
        return False
    if any(c in AMBIGUOUS_AA for c in seq):
        return False
    return True


def _stream_fasta(fasta_path: Path) -> Generator[tuple[str, str, str], None, None]:
    """
    Streaming FASTA parser.
    Yields (seq_id, description, sequence) tuples.
    Handles both plain .fasta and .fasta.gz files transparently.
    """
    opener = gzip.open if fasta_path.suffix == ".gz" else open
    mode   = "rt"

    seq_id      = ""
    description = ""
    seq_parts: list[str] = []

    with opener(fasta_path, mode, encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n\r")

            if line.startswith(">"):
                # Emit previous record (if any)
                if seq_id:
                    yield seq_id, description, "".join(seq_parts)

                # Parse new header: ">UniRef50_XXXX some description"
                header = line[1:].strip()
                parts  = header.split(None, 1)
                seq_id      = parts[0] if parts else ""
                description = parts[1] if len(parts) > 1 else ""
                seq_parts   = []
            else:
                seq_parts.append(line.strip().upper())

    # Emit final record
    if seq_id and seq_parts:
        yield seq_id, description, "".join(seq_parts)


def clean_uniref_fasta(fasta_path: Path, out_dir: Path) -> str:
    _set_polars_threads()
    t0   = time.time()
    log.info(f"[UniRef50]   START  {fasta_path.name}  ({_mb(fasta_path):.1f} MB)")
    log.info(f"[UniRef50]   Chunk size: {FASTA_CHUNK_SIZE:,} sequences per parquet shard")

    try:
        chunk: list[dict]  = []
        part_idx           = 0
        n_total = n_passed = n_toolong = n_ambig = 0
        part_paths: list[Path] = []

        def _flush_chunk(chunk: list[dict], idx: int) -> Path:
            df   = pl.DataFrame(chunk, schema={
                "seq_id":      pl.Utf8,
                "description": pl.Utf8,
                "sequence":    pl.Utf8,
                "seq_len":     pl.UInt16,
            })
            p = out_dir / f"uniref50_clean_part_{idx:03d}.parquet"
            df.write_parquet(p, compression="zstd", statistics=True)
            log.info(f"[UniRef50]   Flushed shard {idx:03d}  "
                     f"{len(chunk):,} seqs  ({_mb(p):.1f} MB)  "
                     f"[{_elapsed(t0)}]")
            return p

        for seq_id, desc, seq in _stream_fasta(fasta_path):
            n_total += 1

            # ── Filter: length ────────────────────────────────────────────
            if len(seq) > ESM2_MAX_LEN:
                n_toolong += 1
                continue

            # ── Filter: ambiguous characters ──────────────────────────────
            if any(c in AMBIGUOUS_AA for c in seq):
                n_ambig += 1
                continue

            n_passed += 1
            chunk.append({
                "seq_id":      seq_id,
                "description": desc,
                "sequence":    seq,
                "seq_len":     len(seq),
            })

            # ── Heartbeat every 500k reads ────────────────────────────────
            if n_total % 500_000 == 0:
                log.info(
                    f"[UniRef50]   {n_total:,} read  "
                    f"passed={n_passed:,}  "
                    f"too_long={n_toolong:,}  "
                    f"ambig={n_ambig:,}"
                )

            # ── Chunk flush guard (64 GB RAM protection) ──────────────────
            if len(chunk) >= FASTA_CHUNK_SIZE:
                part_paths.append(_flush_chunk(chunk, part_idx))
                chunk    = []
                part_idx += 1

        # Flush any remaining sequences
        if chunk:
            part_paths.append(_flush_chunk(chunk, part_idx))

        reject_pct = (1 - n_passed / max(n_total, 1)) * 100
        msg = (
            f"[UniRef50]   DONE   "
            f"{n_total:,} total → {n_passed:,} passed "
            f"({reject_pct:.1f}% rejected: "
            f"{n_toolong:,} too long, {n_ambig:,} ambiguous)  "
            f"{len(part_paths)} shards  [{_elapsed(t0)}]"
        )
        log.info(msg)
        return msg

    except Exception as exc:
        log.error(f"[UniRef50]   FAILED  {exc}", exc_info=True)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# 4. MegaScale CSV Cleaner
#    Input  : Multiple MegaScale CSV files (confirmed schema from inventory):
#               • Tsuboyama2023_Dataset1_20230416.csv   (1.2 GB)
#               • Tsuboyama2023_Dataset2_Dataset3…csv   (665 MB)
#               • Lib1_K50dG.csv … Lib4_K50dG.csv       (149–505 MB each)
#    Output : mega_scale_clean.parquet  (all sources merged)
#    Logic  :
#      • Use polars lazy scan_csv for each file.
#      • Keep canonical columns: name, dna_seq, deltaG (from deltaG_t or
#        deltaG_c as fallback), plus confidence intervals.
#      • Drop rows where dna_seq is null or empty.
#      • Drop rows where deltaG is null (primary training label).
#      • Add a 'source_file' column for traceability.
#      • Concatenate all frames and write one unified parquet.
# ─────────────────────────────────────────────────────────────────────────────

# The main deltaG column used by MegaScaleDataset (thermal denaturation)
MEGA_SCALE_LABEL_COL   = "deltaG_t"
MEGA_SCALE_LABEL_ALIAS = "deltaG"

# Columns to keep; filtered at runtime against what each file actually has
MEGA_SCALE_KEEP = [
    "name", "dna_seq",
    "deltaG_t", "deltaG_t_95CI",
    "deltaG_c", "deltaG_c_95CI",
    "log10_K50_t", "log10_K50_c",
    "fitting_error_t", "fitting_error_c",
]


def clean_mega_scale_csvs(csv_paths: list[Path], out_dir: Path) -> str:
    """
    ETL for all MegaScale CSV files.  Merges into one parquet.
    Called from within a ProcessPoolExecutor worker.
    """
    _set_polars_threads()
    t0   = time.time()
    log.info(f"[MegaScale]  START  {len(csv_paths)} file(s)")
    for p in csv_paths:
        log.info(f"[MegaScale]    {p.name}  ({_mb(p):.1f} MB)")

    try:
        frames: list[pl.DataFrame] = []

        for csv_path in csv_paths:
            log.info(f"[MegaScale]  Reading {csv_path.name} …")
            try:
                lf = pl.scan_csv(
                    csv_path,
                    infer_schema_length=10_000,
                    ignore_errors=True,
                )
                schema  = lf.collect_schema().names()
                keep    = [c for c in MEGA_SCALE_KEEP if c in schema]
                lf      = lf.select(keep)

                # ── Primary label: prefer deltaG_t (thermal), fall back to deltaG_c
                if "deltaG_t" in keep:
                    lf = lf.with_columns(
                        pl.col("deltaG_t").alias(MEGA_SCALE_LABEL_ALIAS)
                    )
                elif "deltaG_c" in keep:
                    lf = lf.with_columns(
                        pl.col("deltaG_c").alias(MEGA_SCALE_LABEL_ALIAS)
                    )

                # ── Drop null labels and null/empty dna_seq ───────────────
                lf = lf.filter(
                    pl.col(MEGA_SCALE_LABEL_ALIAS).is_not_null()
                    & pl.col("dna_seq").is_not_null()
                    & (pl.col("dna_seq").str.len_chars() > 0)
                )

                # ── Add provenance column ─────────────────────────────────
                lf = lf.with_columns(
                    pl.lit(csv_path.name).alias("source_file")
                )

                df = lf.collect()
                log.info(f"[MegaScale]    {csv_path.name}  → {len(df):,} rows")
                frames.append(df)

            except Exception as inner_exc:
                log.error(f"[MegaScale]  SKIP {csv_path.name}: {inner_exc}")
                continue

        if not frames:
            raise ValueError("No MegaScale data survived cleaning — check input files.")

        combined  = pl.concat(frames, how="diagonal_relaxed")
        n_rows    = len(combined)
        out_path  = out_dir / "mega_scale_clean.parquet"
        combined.write_parquet(out_path, compression="zstd", statistics=True)

        msg = (f"[MegaScale]  DONE   {n_rows:,} rows across "
               f"{len(frames)} source(s) → {out_path.name}"
               f"  ({_mb(out_path):.1f} MB)  [{_elapsed(t0)}]")
        log.info(msg)
        return msg

    except Exception as exc:
        log.error(f"[MegaScale]  FAILED  {exc}", exc_info=True)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# PROCESS POOL ENTRY POINTS
# These thin wrappers are needed because lambda / closures are not picklable
# across ProcessPoolExecutor workers.
# ─────────────────────────────────────────────────────────────────────────────

def _run_fireprotdb(args: tuple[str, str]) -> str:
    return clean_fireprotdb(Path(args[0]), Path(args[1]))

def _run_clinvar(args: tuple[str, str]) -> str:
    return clean_clinvar_vcf(Path(args[0]), Path(args[1]))

def _run_uniref(args: tuple[str, str]) -> str:
    return clean_uniref_fasta(Path(args[0]), Path(args[1]))

def _run_mega_scale(args: tuple[list[str], str]) -> str:
    return clean_mega_scale_csvs([Path(p) for p in args[0]], Path(args[1]))


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def discover_inputs(input_dir: Path) -> dict[str, list[Path]]:
    """
    Walk input_dir and return a dict mapping task name → list of file paths.
    Uses the same ghost-file rules as dataset_scanner.py.
    """
    tasks: dict[str, list[Path]] = {
        "fireprotdb":  [],
        "clinvar":     [],
        "uniref":      [],
        "mega_scale":  [],
    }

    skip_dirs = {"__macosx"}

    for dirpath_str, dirnames, filenames in os.walk(input_dir):
        dirpath  = Path(dirpath_str)
        dirnames[:] = [d for d in dirnames if d.lower() not in skip_dirs]

        for fname in filenames:
            if fname.startswith("."):
                continue   # dot-files
            p    = dirpath / fname
            stem = fname.lower()
            ext  = p.suffix.lower()

            if "fireprotdb" in stem and ext == ".csv":
                tasks["fireprotdb"].append(p)

            elif "clinvar" in stem and ext == ".vcf":
                # Prefer the uncompressed .vcf (larger but faster to stream)
                tasks["clinvar"].append(p)

            elif "uniref" in stem and ext in {".fasta", ".fa"}:
                tasks["uniref"].append(p)

            elif ext == ".csv" and any(kw in stem for kw in
                                       ["tsuboyama", "lib1_k50", "lib2_k50",
                                        "lib3_k50", "lib4_k50"]):
                tasks["mega_scale"].append(p)

    # Prefer uncompressed ClinVar VCF if both exist
    vcf_plain = [p for p in tasks["clinvar"] if p.suffix == ".vcf"]
    tasks["clinvar"] = vcf_plain if vcf_plain else tasks["clinvar"]

    return tasks


def main(input_dir: str, output_dir: str) -> None:
    t_global = time.time()
    inp      = Path(input_dir)
    out      = Path(output_dir)

    log.info("=" * 70)
    log.info("  Holo-GNN  Master ETL Pipeline  — V5.0")
    log.info(f"  Input  : {inp}")
    log.info(f"  Output : {out}")
    log.info(f"  Workers: {MAX_WORKERS}  (of 16 vCPUs)")
    log.info("=" * 70)

    # ── Validate input directory ──────────────────────────────────────────────
    if not inp.exists():
        log.error(f"Input directory not found: {inp}")
        sys.exit(1)

    # ── Create output directory ───────────────────────────────────────────────
    out.mkdir(parents=True, exist_ok=True)

    # ── Discover input files ──────────────────────────────────────────────────
    log.info("Discovering input files …")
    tasks = discover_inputs(inp)

    for task_name, paths in tasks.items():
        if paths:
            log.info(f"  {task_name:15s}  {len(paths)} file(s):")
            for p in paths:
                log.info(f"                   {p.relative_to(inp)}  "
                         f"({_mb(p):.1f} MB)")
        else:
            log.warning(f"  {task_name:15s}  NO FILES FOUND — will skip.")

    # ── Build job list for ProcessPoolExecutor ────────────────────────────────
    # Each job is (worker_fn, args_tuple) — args must be serialisable strings.
    jobs: list[tuple] = []

    if tasks["fireprotdb"]:
        for fp in tasks["fireprotdb"]:
            jobs.append((_run_fireprotdb, (str(fp), str(out))))

    if tasks["clinvar"]:
        for fp in tasks["clinvar"]:
            jobs.append((_run_clinvar, (str(fp), str(out))))

    if tasks["uniref"]:
        for fp in tasks["uniref"]:
            jobs.append((_run_uniref, (str(fp), str(out))))

    if tasks["mega_scale"]:
        # All MegaScale CSVs run together in one worker (they share a schema)
        jobs.append((_run_mega_scale,
                     ([str(p) for p in tasks["mega_scale"]], str(out))))

    if not jobs:
        log.error("No recognised input files found. Exiting.")
        sys.exit(1)

    log.info(f"\nSubmitting {len(jobs)} job(s) to ProcessPoolExecutor "
             f"(max_workers={MAX_WORKERS}) …\n")

    # ── Execute in parallel ───────────────────────────────────────────────────
    results:  list[str] = []
    failures: list[str] = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_map = {
            pool.submit(fn, args): fn.__name__
            for fn, args in jobs
        }

        for future in as_completed(future_map):
            job_name = future_map[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                msg = f"{job_name} FAILED: {exc}"
                log.error(msg)
                failures.append(msg)

    # ── Final report ──────────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("  ETL PIPELINE COMPLETE")
    log.info(f"  Wall time  : {_elapsed(t_global)}")
    log.info(f"  Succeeded  : {len(results)}")
    log.info(f"  Failed     : {len(failures)}")
    log.info("=" * 70)

    for r in results:
        log.info(f"  ✅  {r}")
    for f in failures:
        log.error(f"  ❌  {f}")

    # List output parquet files and their sizes
    parquet_files = sorted(out.glob("*.parquet"))
    if parquet_files:
        log.info(f"\n  Output parquet files ({len(parquet_files)}):")
        total_bytes = 0
        for pf in parquet_files:
            sz = pf.stat().st_size
            total_bytes += sz
            log.info(f"    {pf.name:<55s} {sz / (1024**2):>8.1f} MB")
        log.info(f"  {'Total':55s} {total_bytes / (1024**2):>8.1f} MB")

    if failures:
        log.warning(f"\n{len(failures)} job(s) failed. "
                    "Check logs above for details.")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Holo-GNN V5.0 Master ETL Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",  "-i",
        default=r"D:\ML MODELS\HOLO_GNN_PROJECT\DATA",
        help="Root directory containing raw biological datasets.",
    )
    parser.add_argument(
        "--output", "-o",
        default=r"D:\ML MODELS\HOLO_GNN_PROJECT\CLEANED_DATA",
        help="Output directory for cleaned .parquet files.",
    )
    args = parser.parse_args()
    main(args.input, args.output)
