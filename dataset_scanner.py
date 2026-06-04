"""
dataset_scanner.py
==================
Senior Data Engineer utility for Holo-GNN.

Walks the target data directory, inspects every biological file format,
runs a multi-tag categorisation engine, and writes DATASET_INVENTORY.md.

Engineering guarantees
----------------------
  • Ghost-file blacklist  : __MACOSX directories and dot-files are skipped.
  • Per-file try/except   : one corrupt file cannot crash the whole scan.
  • Format-aware parsing  : CSV/TSV/TXT → pandas, VCF → raw header search,
                            FASTA/SPTXT → first-10-lines preview,
                            GZ/ZIP → stat only (no decompression).
  • Multi-tag support     : a file can carry more than one Holo-GNN tag.
  • Reproducible output   : files grouped by tag in DATASET_INVENTORY.md.
"""

import os
import re
import gzip
import zipfile
import datetime
from pathlib import Path
from collections import defaultdict
from typing import Optional

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
TARGET_DIR   = Path(r"D:\ML MODELS\HOLO_GNN_PROJECT\DATA")
OUTPUT_FILE  = Path(r"D:\ML models\Holo_GNN_Project\DATASET_INVENTORY.md")

# ─────────────────────────────────────────────────────────────────────────────
# GHOST FILE BLACKLIST  (Requirement 1)
# ─────────────────────────────────────────────────────────────────────────────
SKIP_DIRS  = {"__macosx"}          # matched case-insensitively
DOT_PREFIX = "."                   # any file whose name starts with a dot

# ─────────────────────────────────────────────────────────────────────────────
# HOLO-GNN CATEGORISATION ENGINE  (Requirement 3)
# Keywords matched against: filename stem, parent directory names, AND
# column/header strings extracted during parsing.
# ─────────────────────────────────────────────────────────────────────────────
TAG_RULES: dict[str, list[str]] = {
    # Matched against: rel_path (dir names + filename) + column names + notes
    "[STABILITY_HEAD]": [
        r"ddg",           # DDG column (FireProtDB: DDG, DOMAINOME_DDG)
        r"dtm",           # DTM / melting temp delta
        r"stability",     # stability in path or column
        r"fireprot",      # FireProtDB directory/filename
        r"k50",           # log10_K50_t, K50_dG — no word boundary (alphanumeric)
        r"mutant",        # MUTANT_ID column
        r"dms",           # DMS (Deep Mutational Scanning) in path or column
        r"deltag",        # deltaG_t, deltaG_c columns (case-insensitive match)
        r"delta_g",       # delta_G variants
        r"mega.?scale",   # mega_scale_cdna directory
        r"tsuboyama",     # Tsuboyama2023 filename
        r"dna.?seq",      # dna_seq column — canonical MegaScale signal
        r"lib\d+_k50",    # Lib1_K50dG, Lib2_K50dG filenames
    ],
    "[IDR_HEAD]": [
        r"clinvar",       # ClinVar directory/filename
        r"pathogenic",    # pathogenic classification column
        r"disease",       # disease annotation
        r"\bvcf\b",       # .vcf extension in path
        r"\brg\b",        # radius of gyration
    ],
    "[PROTEOMICS]": [
        r"massive.?kb",   # massive_kb directory
        r"massive_kb",
        r"\.sptxt",       # .sptxt extension in rel_path
        r"sptxt",         # sptxt anywhere
        r"\brt\b",        # retention time column
        r"abundance",
        r"expression",
    ],
    "[PRE-TRAINING]": [
        r"uniref",        # UniRef50 directory
        r"\.fasta",       # .fasta extension in rel_path
        r"\bfasta\b",     # fasta keyword
    ],
}

# Compile all patterns once
_COMPILED_RULES: dict[str, list[re.Pattern]] = {
    tag: [re.compile(p, re.IGNORECASE) for p in patterns]
    for tag, patterns in TAG_RULES.items()
}

UNTAGGED = "[UNTAGGED]"


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: assign tags
# ─────────────────────────────────────────────────────────────────────────────
def assign_tags(search_corpus: str) -> list[str]:
    """
    Run the categorisation engine against a combined search string built from
    the file path, directory names, and extracted headers/columns.

    Returns a sorted list of matching tag strings.  If nothing matches,
    returns [UNTAGGED].
    """
    found = []
    for tag, patterns in _COMPILED_RULES.items():
        for pat in patterns:
            if pat.search(search_corpus):
                found.append(tag)
                break          # one match per tag is sufficient
    return sorted(found) if found else [UNTAGGED]


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: human-readable file size
# ─────────────────────────────────────────────────────────────────────────────
def fmt_mb(byte_count: int) -> str:
    mb = byte_count / (1024 ** 2)
    if mb < 0.01:
        return f"{byte_count / 1024:.2f} KB"
    return f"{mb:.2f} MB"


# ─────────────────────────────────────────────────────────────────────────────
# FORMAT-AWARE PARSERS  (Requirement 2)
# Each returns (columns_or_headers: list[str], preview_note: str)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_csv(path: Path) -> tuple[list[str], str]:
    """CSV / TSV / TXT — pandas, first 5 rows only."""
    sep = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    # Some txt files are space-delimited; try comma first, fall back
    try:
        df = pd.read_csv(path, nrows=5, sep=sep, engine="python",
                         on_bad_lines="skip")
        cols = list(df.columns)
        return cols, f"{len(df)} preview rows"
    except Exception:
        # Retry with sep=None (auto-detect)
        df = pd.read_csv(path, nrows=5, sep=None, engine="python",
                         on_bad_lines="skip")
        cols = list(df.columns)
        return cols, f"{len(df)} preview rows (auto-sep)"


def _parse_vcf(path: Path) -> tuple[list[str], str]:
    """VCF — scan raw lines for the #CHROM header."""
    columns: list[str] = []
    meta_lines = 0
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.rstrip()
                if line.startswith("##"):
                    meta_lines += 1
                    continue
                if line.startswith("#CHROM"):
                    columns = line.lstrip("#").split("\t")
                    break
    except Exception as exc:
        return [], f"VCF parse error: {exc}"
    note = f"#CHROM header found; {meta_lines} meta-lines"
    return columns, note


def _parse_fasta_or_sptxt(path: Path) -> tuple[list[str], str]:
    """FASTA / SPTXT — first 10 raw lines as preview headers."""
    lines: list[str] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                if i >= 10:
                    break
                stripped = line.rstrip()
                if stripped:
                    lines.append(stripped)
    except Exception as exc:
        return [], f"Read error: {exc}"
    return [], "\n".join(f"    {ln}" for ln in lines)


def _parse_compressed(path: Path) -> tuple[list[str], str]:
    """GZ / ZIP — stat only, no decompression."""
    try:
        size = path.stat().st_size
        if path.suffix.lower() == ".gz":
            # Peek at compressed filename without extracting
            with gzip.open(path, "rb") as gz:
                gz.peek(1)
            note = f"gzip archive, {fmt_mb(size)} compressed"
        elif path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path, "r") as zf:
                members = zf.namelist()
            note = f"zip archive, {fmt_mb(size)} compressed, {len(members)} member(s)"
        else:
            note = f"compressed archive, {fmt_mb(size)}"
    except Exception as exc:
        note = f"Compressed file (could not inspect): {exc}"
    return [], note


# ─────────────────────────────────────────────────────────────────────────────
# CORE: inspect a single file
# ─────────────────────────────────────────────────────────────────────────────
def inspect_file(path: Path, root: Path) -> dict:
    """
    Inspect one file and return a structured record dict.
    All exceptions are caught; a corrupt/unreadable file returns an error note.
    """
    record: dict = {
        "rel_path":  str(path.relative_to(root)),
        "abs_path":  str(path),
        "size_bytes": 0,
        "size_fmt":  "?",
        "ext":       path.suffix.lower(),
        "columns":   [],
        "note":      "",
        "tags":      [],
    }

    # File size
    try:
        record["size_bytes"] = path.stat().st_size
        record["size_fmt"]   = fmt_mb(record["size_bytes"])
    except OSError as exc:
        record["note"] = f"Cannot stat: {exc}"
        record["tags"] = assign_tags(str(path).lower())
        return record

    ext = record["ext"]
    columns: list[str] = []
    note:    str       = ""

    # ── Dispatch by extension ──────────────────────────────────────────────
    try:
        if ext in {".csv", ".tsv", ".tab", ".txt"}:
            columns, note = _parse_csv(path)

        elif ext == ".vcf":
            columns, note = _parse_vcf(path)

        elif ext in {".fasta", ".fa", ".fna", ".faa", ".sptxt"}:
            columns, note = _parse_fasta_or_sptxt(path)

        elif ext in {".gz", ".zip", ".bz2", ".tar"}:
            columns, note = _parse_compressed(path)

        else:
            note = f"Unsupported extension `{ext}` — no content parsing."

    except Exception as exc:
        note = f"Parse error ({type(exc).__name__}): {exc}"

    record["columns"] = columns
    record["note"]    = note

    # ── Build search corpus for categorisation ────────────────────────────
    # Include: relative path (dir names + filename), all extracted columns/headers
    col_text  = " ".join(columns).lower()
    path_text = record["rel_path"].lower()
    corpus    = f"{path_text} {col_text} {note.lower()}"
    record["tags"] = assign_tags(corpus)

    return record


# ─────────────────────────────────────────────────────────────────────────────
# CORE: walk the target directory
# ─────────────────────────────────────────────────────────────────────────────
def walk_directory(root: Path) -> list[dict]:
    """
    Recursively walk root, applying the ghost-file blacklist, and return
    a list of inspection records for every accepted file.
    """
    records: list[dict] = []
    skipped_ghost = 0
    skipped_dot   = 0

    for dirpath, dirnames, filenames in os.walk(root):
        # ── Prune blacklisted directories in-place (modifies os.walk) ──────
        original_dirs = list(dirnames)
        dirnames[:] = [
            d for d in dirnames
            if d.lower() not in SKIP_DIRS
        ]
        skipped_ghost += len(original_dirs) - len(dirnames)

        for filename in filenames:
            # ── Skip dot-files ───────────────────────────────────────────
            if filename.startswith(DOT_PREFIX):
                skipped_dot += 1
                continue

            file_path = Path(dirpath) / filename
            print(f"  Inspecting: {file_path.relative_to(root)}")
            record = inspect_file(file_path, root)
            records.append(record)

    print(f"\n  Blacklist summary: {skipped_ghost} ghost dir(s) pruned, "
          f"{skipped_dot} dot-file(s) skipped.")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# REPORT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
def build_markdown(records: list[dict], root: Path) -> str:
    """Render the DATASET_INVENTORY.md Markdown string."""

    now        = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_size = sum(r["size_bytes"] for r in records)
    all_tags   = sorted({tag for r in records for tag in r["tags"]})

    # Group records by tag (a file with multiple tags appears in each group)
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        for tag in r["tags"]:
            groups[tag].append(r)

    # ── Document header ───────────────────────────────────────────────────
    lines: list[str] = [
        "# Holo-GNN — Dataset Inventory",
        "",
        f"> **Generated:** {now}  ",
        f"> **Scanned directory:** `{root}`  ",
        f"> **Total files:** {len(records)}  ",
        f"> **Total size:** {fmt_mb(total_size)}  ",
        "",
        "---",
        "",
        "## Table of Contents",
        "",
    ]
    for tag in all_tags:
        anchor = tag.lower().replace("[", "").replace("]", "").replace("-", "").replace(" ", "-").replace("_", "-")
        lines.append(f"- [{tag}](#{anchor}) — {len(groups[tag])} file(s)")

    lines += ["", "---", ""]

    # ── Per-tag sections ──────────────────────────────────────────────────
    for tag in all_tags:
        tag_records = sorted(groups[tag], key=lambda r: r["rel_path"])
        anchor_id   = tag.lower().replace("[", "").replace("]", "").replace("-", "").replace(" ", "-").replace("_", "-")

        lines += [
            f"## {tag}",
            "",
            f"**{len(tag_records)} file(s)**",
            "",
        ]

        for r in tag_records:
            lines += [
                f"### `{r['rel_path']}`",
                "",
                f"| Field | Value |",
                f"|---|---|",
                f"| **Size** | {r['size_fmt']} |",
                f"| **Extension** | `{r['ext']}` |",
                f"| **Tags** | {' · '.join(r['tags'])} |",
            ]

            # Columns / headers row
            if r["columns"]:
                col_str = ", ".join(f"`{c}`" for c in r["columns"][:20])
                if len(r["columns"]) > 20:
                    col_str += f" … (+{len(r['columns']) - 20} more)"
                lines.append(f"| **Columns / Headers** | {col_str} |")
            else:
                lines.append("| **Columns / Headers** | *(none extracted)* |")

            lines.append("")

            # Note / preview block
            if r["note"]:
                # Multi-line notes (e.g. FASTA previews) go in a code block
                if "\n" in r["note"]:
                    lines += [
                        "**Preview (first 10 lines):**",
                        "```",
                        r["note"].strip(),
                        "```",
                    ]
                else:
                    lines.append(f"> {r['note']}")

            lines.append("")
            lines.append("---")
            lines.append("")

    # ── Unrecognised extensions summary ───────────────────────────────────
    ext_set: dict[str, int] = defaultdict(int)
    for r in records:
        ext_set[r["ext"]] += 1

    lines += [
        "## Extension Summary",
        "",
        "| Extension | Count |",
        "|---|---|",
    ]
    for ext, cnt in sorted(ext_set.items(), key=lambda x: -x[1]):
        lines.append(f"| `{ext if ext else '(no ext)'}` | {cnt} |")

    lines += ["", "---", "", "*Report generated by `dataset_scanner.py` — Holo-GNN V5.0*", ""]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 64)
    print("  Holo-GNN Dataset Scanner")
    print(f"  Target : {TARGET_DIR}")
    print(f"  Output : {OUTPUT_FILE}")
    print("=" * 64)

    if not TARGET_DIR.exists():
        raise FileNotFoundError(
            f"Target directory not found: {TARGET_DIR}\n"
            "Check that the path is correct and the drive is mounted."
        )

    print("\n[1/3] Walking directory tree...")
    records = walk_directory(TARGET_DIR)
    print(f"\n  ✅ {len(records)} file(s) inspected.")

    print("\n[2/3] Building Markdown report...")
    md_text = build_markdown(records, TARGET_DIR)

    print("\n[3/3] Writing DATASET_INVENTORY.md...")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        fh.write(md_text)

    print(f"\n  ✅ Report written to: {OUTPUT_FILE}")

    # ── CLI summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  TAG SUMMARY")
    print("=" * 64)
    from collections import Counter
    tag_counts: Counter = Counter()
    for r in records:
        for tag in r["tags"]:
            tag_counts[tag] += 1
    for tag, count in sorted(tag_counts.items()):
        print(f"  {tag:<22} {count:>4} file(s)")
    print("=" * 64)


if __name__ == "__main__":
    main()
