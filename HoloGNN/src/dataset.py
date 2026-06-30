"""
src/dataset.py
==============
HoloGNN Dataset Classes.

Dataset classes:
    MassIVEKBDataset   — Proteomics pre-training (MassIVE-KB .sptxt files)
    MegaScaleDataset   — Single-sequence ΔG stability (MegaScale cDNA)
    ClinVarDataset     — Pathogenicity classification (ClinVar VCF)
    FireProtDataset    — Siamese ΔΔG stability regression (FireProtDB)
    UniRefDataset      — Pre-training / MLM (streaming parquet shards)

Each class tags every sample with  batch['task']  so the HoloGNN
forward pass can dispatch to the correct head.
"""

from __future__ import annotations

import glob
import os
import math
import random
import torch
import pandas as pd
from pathlib import Path
from typing import Iterator, Optional
from torch.utils.data import Dataset, Sampler
from transformers import EsmTokenizer
from Bio.Seq import Seq

# Reuse the biophysical tables from the heuristics module (stdlib-only).
from src.heuristics import KD_HYDROPATHY, RESIDUE_VOLUME


# =============================================================================
# E. coli K-12 Codon Adaptation Index
# =============================================================================
_CODON_CAI_WEIGHT: dict[str, float] = {
    # Phe (F)
    "TTT": 0.296, "TTC": 1.000,
    # Leu (L)
    "TTA": 0.049, "TTG": 0.100, "CTT": 0.100, "CTC": 0.073, "CTA": 0.039, "CTG": 1.000,
    # Ile (I)
    "ATT": 0.731, "ATC": 1.000, "ATA": 0.107,
    # Met (M)
    "ATG": 1.000,
    # Val (V)
    "GTT": 0.726, "GTC": 0.354, "GTA": 0.378, "GTG": 1.000,
    # Ser (S)
    "TCT": 1.000, "TCC": 0.744, "TCA": 0.298, "TCG": 0.260, "AGT": 0.209, "AGC": 0.828,
    # Pro (P)
    "CCT": 0.516, "CCC": 0.195, "CCA": 0.271, "CCG": 1.000,
    # Thr (T)
    "ACT": 0.965, "ACC": 1.000, "ACA": 0.308, "ACG": 0.424,
    # Ala (A)
    "GCT": 1.000, "GCC": 0.556, "GCA": 0.469, "GCG": 0.636,
    # Tyr (Y)
    "TAT": 0.326, "TAC": 1.000,
    # His (H)
    "CAT": 0.424, "CAC": 1.000,
    # Gln (Q)
    "CAA": 0.124, "CAG": 1.000,
    # Asn (N)
    "AAT": 0.366, "AAC": 1.000,
    # Lys (K)
    "AAA": 1.000, "AAG": 0.248,
    # Asp (D)
    "GAT": 0.776, "GAC": 1.000,
    # Glu (E)
    "GAA": 1.000, "GAG": 0.356,
    # Cys (C)
    "TGT": 0.500, "TGC": 1.000,
    # Trp (W)
    "TGG": 1.000,
    # Arg (R)
    "CGT": 1.000, "CGC": 0.758, "CGA": 0.111, "CGG": 0.111, "AGA": 0.070, "AGG": 0.070,
    # Gly (G)
    "GGT": 1.000, "GGC": 0.724, "GGA": 0.145, "GGG": 0.181,
    # Stop codons
    "TAA": 1.000, "TAG": 0.100, "TGA": 0.100,
}


def _cai_per_residue(dna_seq: str, max_length: int) -> list[float]:
    n_codons = len(dna_seq) // 3
    result   = [0.0] * max_length
    for i in range(min(n_codons, max_length)):
        codon     = dna_seq[i * 3: i * 3 + 3].upper()
        result[i] = _CODON_CAI_WEIGHT.get(codon, 0.5)
    return result


# =============================================================================
# Henderson-Hasselbalch charge at pH 7.4
# =============================================================================
_PKA = {
    'D': (3.67,  -1.0), 'E': (4.25,  -1.0),
    'H': (6.54,  +1.0), 'C': (8.18,  -1.0),
    'Y': (10.00, -1.0), 'K': (10.53, +1.0), 'R': (12.00, +1.0),
}
_PH = 7.4


def _hh_charge(aa: str) -> float:
    aa = aa.upper()
    if aa not in _PKA:
        return 0.0
    pka, q_sign = _PKA[aa]
    if q_sign < 0:
        return -1.0 / (1.0 + 10.0 ** (pka - _PH))
    return +1.0 / (1.0 + 10.0 ** (_PH - pka))


def _charge_per_residue(protein_seq: str, max_length: int, window: int = 7) -> list[float]:
    L      = min(len(protein_seq), max_length)
    half   = window // 2
    result = [0.0] * max_length
    for i in range(L):
        raw       = sum(_hh_charge(protein_seq[j])
                        for j in range(max(0, i - half),
                                       min(len(protein_seq), i + half + 1)))
        result[i] = max(0.0, min(1.0, (raw + window) / (2 * window)))
    return result


# =============================================================================
# Codon-level GC-skew × stacking energy proxy
# =============================================================================
_CODON_COMPOSITION: dict[str, tuple[int, int, int, int]] = {}


def _nt_counts(codon: str) -> tuple[int, int, int, int]:
    if codon not in _CODON_COMPOSITION:
        _CODON_COMPOSITION[codon] = (
            codon.count('A'), codon.count('T'),
            codon.count('G'), codon.count('C'),
        )
    return _CODON_COMPOSITION[codon]


def _mrna_fold_per_residue(dna_seq: str, max_length: int,
                            codon_window: int = 3) -> list[float]:
    n_codons = len(dna_seq) // 3
    L        = min(n_codons, max_length)
    half     = codon_window // 2
    result   = [0.0] * max_length
    for i in range(L):
        A = T = G = C = 0
        for ci in range(max(0, i - half), min(n_codons, i + half + 1)):
            codon = dna_seq[ci * 3: ci * 3 + 3].upper().replace('U', 'T')
            if len(codon) == 3:
                a, t, g, c = _nt_counts(codon)
                A += a; T += t; G += g; C += c
        total     = A + T + G + C
        gc_frac   = (G + C) / total if total else 0.0
        gc_skew   = (G - C) / (G + C) if (G + C) > 0 else 0.0
        result[i] = 0.5 * gc_frac + 0.5 * (1.0 - abs(gc_skew))
    return result


# =============================================================================
# Master mechanistic feature generator
# =============================================================================
def _mechanistic_features(
    protein_seq: str,
    dna_seq:     str,
    max_length:  int,
) -> torch.Tensor:
    """
    Returns (max_length, 3) float32 — [mRNA_fold, CAI, Charge].
    Requires both protein_seq AND dna_seq for codon-level accuracy.
    """
    mrna_fold = _mrna_fold_per_residue(dna_seq, max_length, codon_window=3)
    cai_vals  = _cai_per_residue(dna_seq, max_length)
    charge    = _charge_per_residue(protein_seq, max_length, window=7)
    feat      = torch.zeros(max_length, 3, dtype=torch.float32)
    L         = min(len(protein_seq), max_length)
    for i in range(L):
        feat[i, 0] = mrna_fold[i]
        feat[i, 1] = cai_vals[i]
        feat[i, 2] = charge[i]
    return feat


def _protein_only_mech(protein_seq: str, max_length: int) -> torch.Tensor:
    """
    Mechanistic features when no DNA sequence is available (charge only;
    mRNA_fold and CAI are zeroed).  Used by ClinVarDataset and UniRefDataset.
    """
    feat  = torch.zeros(max_length, 3, dtype=torch.float32)
    L     = min(len(protein_seq), max_length)
    ch    = _charge_per_residue(protein_seq, max_length, window=7)
    for i in range(L):
        feat[i, 2] = ch[i]
    return feat


# =============================================================================
# Expanded protein-only mechanistic channels
#   Three extra descriptors computable from the amino-acid sequence alone
#   (no DNA required), enriching the protein-only path (ClinVar, the web UI):
#     3 = Kyte-Doolittle hydropathy  (normalised to [0, 1])
#     4 = side-chain volume          (normalised to [0, 1])
#     5 = helix propensity           (Chou-Fasman P_alpha, normalised to [0, 1])
# =============================================================================
# Chou-Fasman alpha-helix propensities P_alpha.
_HELIX_PROPENSITY = {
    "A": 1.42, "R": 0.98, "N": 0.67, "D": 1.01, "C": 0.70,
    "Q": 1.11, "E": 1.51, "G": 0.57, "H": 1.00, "I": 1.08,
    "L": 1.21, "K": 1.16, "M": 1.45, "F": 1.13, "P": 0.57,
    "S": 0.77, "T": 0.83, "W": 1.08, "Y": 0.69, "V": 1.06,
}
_KD_MIN, _KD_MAX   = -4.5, 4.5      # Kyte-Doolittle range
_VOL_MAX           = 240.0          # > Trp (227.8 A^3)
_PA_MIN, _PA_MAX   = 0.57, 1.51     # helix propensity range


def _expanded_protein_channels(protein_seq: str, max_length: int) -> torch.Tensor:
    """Return (max_length, 3) of [hydropathy, volume, helix_propensity] in [0,1]."""
    feat = torch.zeros(max_length, 3, dtype=torch.float32)
    L    = min(len(protein_seq), max_length)
    for i in range(L):
        aa = protein_seq[i].upper()
        kd = KD_HYDROPATHY.get(aa, 0.0)
        vol = RESIDUE_VOLUME.get(aa, 0.0)
        pa = _HELIX_PROPENSITY.get(aa, 1.0)
        feat[i, 0] = (kd - _KD_MIN) / (_KD_MAX - _KD_MIN)
        feat[i, 1] = min(1.0, vol / _VOL_MAX)
        feat[i, 2] = (pa - _PA_MIN) / (_PA_MAX - _PA_MIN)
    return feat


def mechanistic_features_for_protein(protein_seq: str,
                                     max_length: int,
                                     expanded: bool = False) -> torch.Tensor:
    """
    Public helper for inference: the (max_length, C) mechanistic-feature tensor
    for a protein-only input (no DNA).  C = 3 (default) or 6 (expanded).

    Used by both ``predict.py`` and the HOLOGNN_APP inference wrapper so the
    features fed to the backbone match those seen during training.
    """
    base = _protein_only_mech(protein_seq, max_length)             # (L, 3)
    if not expanded:
        return base
    extra = _expanded_protein_channels(protein_seq, max_length)    # (L, 3)
    return torch.cat([base, extra], dim=-1)                        # (L, 6)


# =============================================================================
# Shared tokeniser helper
# =============================================================================
def _get_tokenizer(esm_model_name: str = "facebook/esm2_t6_8M_UR50D") -> EsmTokenizer:
    """Cached-on-first-call ESM tokeniser."""
    return EsmTokenizer.from_pretrained(esm_model_name)


def _tokenize(tokenizer: EsmTokenizer, seq: str,
              max_length: int) -> dict[str, torch.Tensor]:
    enc = tokenizer(
        seq,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {
        "input_ids":      enc["input_ids"].squeeze(0),
        "attention_mask": enc["attention_mask"].squeeze(0),
    }


# =============================================================================
# Dataset classes
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# MassIVEKBDataset
# ─────────────────────────────────────────────────────────────────────────────
class MassIVEKBDataset(Dataset):
    """
    Loader for MassIVE-KB spectral library (.sptxt files).
    Extracts peptide sequences for proteomics pre-training.

    task tag : 'proteomics'
    """

    def __init__(self, data_dir: str, max_length: int = 100):
        self.tokenizer  = _get_tokenizer()
        self.data: list[dict] = []
        self.max_length = max_length

        print(f"Scanning {data_dir} for .sptxt files...")
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".sptxt"):
                    self._parse_sptxt(os.path.join(root, file))
        print(f"Loaded {len(self.data):,} sequences from MassIVE-KB.")

    def _parse_sptxt(self, filepath: str) -> None:
        with open(filepath, "r") as f:
            for line in f:
                if line.startswith("Name:"):
                    try:
                        raw  = line.strip().split(" ")[1]
                        seq  = "".join(c for c in raw.split("/")[0] if c.isalpha())
                        if seq:
                            self.data.append({"seq": seq, "label": 0.0})
                    except IndexError:
                        continue

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        tok  = _tokenize(self.tokenizer, item["seq"], self.max_length)
        mech = mechanistic_features_for_protein(item["seq"], self.max_length)
        return {**tok,
                "mechanistic_features": mech,
                "label": torch.tensor(item["label"], dtype=torch.float),
                "task":  "proteomics"}


# ─────────────────────────────────────────────────────────────────────────────
# MegaScaleDataset
# ─────────────────────────────────────────────────────────────────────────────
class MegaScaleDataset(Dataset):
    """
    Full MegaScale cDNA stability dataset.
    Regression target : deltaG (kcal/mol).
    task tag          : 'stability'

    Accepts both raw .csv and pre-cleaned .parquet files.
    """

    def __init__(self, data_path: str, max_length: int = 100,
                 expanded_mech: bool = False):
        self.tokenizer     = _get_tokenizer()
        self.max_length    = max_length
        self.expanded_mech = expanded_mech
        path = Path(data_path)

        print(f"Loading MegaScale data from {path.name} …")
        if path.suffix == ".parquet":
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path)

        # Support both 'deltaG' (ETL output) and 'deltaG_t' (raw K50 tables)
        label_col = "deltaG" if "deltaG" in df.columns else "deltaG_t"
        self.df   = df.dropna(subset=["dna_seq", label_col]).reset_index(drop=True)
        self._label_col = label_col

        # Pre-compute protein translations to avoid repeated Bio.Seq calls
        print("  Pre-computing protein sequences from DNA...")
        self._proteins = []
        for i in range(len(self.df)):
            dna = str(self.df.iloc[i]["dna_seq"])
            self._proteins.append(str(Seq(dna).translate(to_stop=True)))
        print(
            f"  {len(self.df):,} valid samples loaded. "
            f"Label column: '{label_col}'"
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row         = self.df.iloc[idx]
        dna_seq     = str(row["dna_seq"])
        protein_seq = self._proteins[idx]
        label       = float(row[self._label_col])
        tok         = _tokenize(self.tokenizer, protein_seq, self.max_length)
        mech        = _mechanistic_features(protein_seq, dna_seq, self.max_length)
        if self.expanded_mech:
            mech = torch.cat(
                [mech, _expanded_protein_channels(protein_seq, self.max_length)], dim=-1
            )
        return {
            **tok,
            "label":               torch.tensor(label, dtype=torch.float),
            "mechanistic_features": mech,
            "task":                "stability",
        }


# ─────────────────────────────────────────────────────────────────────────────
# ClinVarDataset
# ─────────────────────────────────────────────────────────────────────────────
class ClinVarDataset(Dataset):
    """
    ClinVar pathogenicity classification dataset.
    Loads the ETL-cleaned clinvar_clean.parquet produced by master_etl_pipeline.py.

    Each sample provides a pair of tokenised sequences (WT = REF allele context,
    MT = ALT allele context) and a binary label (1=Pathogenic, 0=Benign) for
    the IDR Classification Head.

    Parquet columns used
    --------------------
      ref    : reference allele string (1–50 nt typically)
      alt    : alternate allele string
      label  : int  (1 = Pathogenic, 0 = Benign)
      chrom  : chromosome (logged; not used in model)
      pos    : position    (logged; not used in model)

    Design note — REF/ALT as protein sequences
    -------------------------------------------
    ClinVar alleles are nucleotide sequences, not proteins.  We interpret the
    REF and ALT strings directly as single-letter amino acid token sequences
    for the ESM-2 tokeniser, which accepts any string of valid amino acid
    letters.  Short indels are padded by the tokeniser automatically.
    For SNVs the two strings will differ by exactly one token, giving the
    model a precise signal about which residue changed — this is the correct
    information for the IDR head.

    task tag : 'idr'
    """

    def __init__(
        self,
        parquet_path: str,
        max_length:   int  = 64,
        expanded_mech: bool = False,
    ):
        self.tokenizer     = _get_tokenizer()
        self.max_length    = max_length
        self.expanded_mech = expanded_mech

        print(f"Loading ClinVar data from {Path(parquet_path).name} …")
        df = pd.read_parquet(parquet_path)

        required = {"ref", "alt", "label"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(
                f"clinvar_clean.parquet is missing columns: {missing}. "
                "Re-run master_etl_pipeline.py to regenerate."
            )

        # Keep only rows with valid allele strings
        df = df.dropna(subset=["ref", "alt", "label"])
        df = df[df["ref"].str.len() > 0]
        df = df[df["alt"].str.len() > 0]
        df["label"] = df["label"].astype(int)
        self.df = df.reset_index(drop=True)

        n_path   = int((self.df["label"] == 1).sum())
        n_benign = int((self.df["label"] == 0).sum())
        print(
            f"  {len(self.df):,} variants loaded  "
            f"(Pathogenic={n_path:,} / Benign={n_benign:,})"
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row   = self.df.iloc[idx]
        ref   = str(row["ref"]).upper()
        alt   = str(row["alt"]).upper()
        label = int(row["label"])

        tok_ref = _tokenize(self.tokenizer, ref, self.max_length)
        tok_alt = _tokenize(self.tokenizer, alt, self.max_length)

        mech_ref = mechanistic_features_for_protein(ref, self.max_length, self.expanded_mech)
        mech_alt = mechanistic_features_for_protein(alt, self.max_length, self.expanded_mech)

        return {
            # Wild-type (reference) allele
            "input_ids_wt":             tok_ref["input_ids"],
            "attention_mask_wt":        tok_ref["attention_mask"],
            "mechanistic_features_wt":  mech_ref,
            # Mutant (alternate) allele
            "input_ids_mt":             tok_alt["input_ids"],
            "attention_mask_mt":        tok_alt["attention_mask"],
            "mechanistic_features_mt":  mech_alt,
            # Classification label
            "label":                    torch.tensor(label, dtype=torch.long),
            "task":                     "idr",
        }


# ─────────────────────────────────────────────────────────────────────────────
# FireProtDataset
# ─────────────────────────────────────────────────────────────────────────────
class FireProtDataset(Dataset):
    """
    FireProtDB thermodynamic stability dataset.
    Loads the ETL-cleaned fireprotdb_clean.parquet produced by master_etl_pipeline.py.

    Formats data identically to MegaScaleDataset so the same SiameseStabilityHead
    and AntisymmetricLoss can be used without modification.

    Parquet columns used
    --------------------
      SEQUENCE_ID   : UniProt-style identifier for the wild-type sequence
      MUTANT_ID     : mutation identifier (e.g. "A23V")
      SOURCE_SEQUENCE_ID : WT protein sequence (1-letter AA, when present)
      TARGET_SEQUENCE_ID : mutant protein sequence (1-letter AA, when present)
      DDG_mean      : mean ΔΔG across replicate experiments (kcal/mol)
      SUBSTITUTION  : substitution string (used to parse WT/MT if sequences absent)
      PROTEIN       : protein name (metadata only)

    WT / MT sequence strategy
    -------------------------
    If SOURCE_SEQUENCE_ID and TARGET_SEQUENCE_ID are full-length sequences
    (common in FireProtDB), we use them directly.
    If they are UniProt accession identifiers (short strings), we fall back
    to using the SEQUENCE_ID + MUTANT_ID pair as string tokens — the Siamese
    head only needs the embedding difference, so relative token positions
    still carry the mutation signal.

    task tag : 'idr'
    """
    # Minimum length to treat a string as a full AA sequence vs an accession ID
    _MIN_SEQ_LEN = 10

    def __init__(
        self,
        parquet_path: str,
        max_length:   int = 100,
        expanded_mech: bool = False,
    ):
        self.tokenizer     = _get_tokenizer()
        self.max_length    = max_length
        self.expanded_mech = expanded_mech

        print(f"Loading FireProtDB data from {Path(parquet_path).name} …")
        df = pd.read_parquet(parquet_path)

        required = {"DDG_mean"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(
                f"fireprotdb_clean.parquet is missing columns: {missing}. "
                "Re-run master_etl_pipeline.py to regenerate."
            )

        df = df.dropna(subset=["DDG_mean"]).reset_index(drop=True)
        self.df = df

        # Identify which sequence representation is available
        has_src = ("SOURCE_SEQUENCE_ID" in df.columns and
                   df["SOURCE_SEQUENCE_ID"].notna().any() and
                   int(df["SOURCE_SEQUENCE_ID"].dropna().str.len().median()) >= self._MIN_SEQ_LEN)
        has_tgt = ("TARGET_SEQUENCE_ID" in df.columns and
                   df["TARGET_SEQUENCE_ID"].notna().any() and
                   int(df["TARGET_SEQUENCE_ID"].dropna().str.len().median()) >= self._MIN_SEQ_LEN)
        self._use_full_seqs = has_src and has_tgt

        print(
            f"  {len(self.df):,} mutations loaded  "
            f"(full sequences: {'yes' if self._use_full_seqs else 'no — using ID tokens'})"
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        ddg = float(row["DDG_mean"])

        if self._use_full_seqs:
            seq_wt = str(row.get("SOURCE_SEQUENCE_ID", ""))
            seq_mt = str(row.get("TARGET_SEQUENCE_ID", ""))
        else:
            # Fall back: use SEQUENCE_ID as WT, MUTANT_ID appended as MT
            # This gives the model the identity string + mutation label as tokens
            seq_wt = str(row.get("SEQUENCE_ID", "UNK"))
            seq_mt = (str(row.get("SEQUENCE_ID", "UNK")) + "_" +
                      str(row.get("MUTANT_ID", "UNK")))

        tok_wt   = _tokenize(self.tokenizer, seq_wt, self.max_length)
        tok_mt   = _tokenize(self.tokenizer, seq_mt, self.max_length)
        mech_wt  = mechanistic_features_for_protein(seq_wt, self.max_length, self.expanded_mech)
        mech_mt  = mechanistic_features_for_protein(seq_mt, self.max_length, self.expanded_mech)

        return {
            # Wild-type
            "input_ids_wt":             tok_wt["input_ids"],
            "attention_mask_wt":        tok_wt["attention_mask"],
            "mechanistic_features_wt":  mech_wt,
            # Mutant
            "input_ids_mt":             tok_mt["input_ids"],
            "attention_mask_mt":        tok_mt["attention_mask"],
            "mechanistic_features_mt":  mech_mt,
            # Regression label
            "label":                    torch.tensor(ddg, dtype=torch.float),
            "task":                     "idr",
        }


# ─────────────────────────────────────────────────────────────────────────────
# UniRefDataset  (streaming parquet shards)
# ─────────────────────────────────────────────────────────────────────────────
class UniRefDataset(Dataset):
    """
    Streaming UniRef50 dataset for pre-training / Masked Language Modeling.

    Loads from the ETL-generated parquet shards:
        uniref50_clean_part_000.parquet
        uniref50_clean_part_001.parquet  …

    Memory strategy
    ---------------
    The full UniRef50 dataset is ~100M sequences.  Loading all shards at once
    would exhaust RAM.  This class implements a two-level lazy scheme:

      1. On __init__: read only shard-level metadata (shard paths + row counts).
         No sequence data is read.  Negligible RAM.

      2. On __getitem__ / _load_shard(): load one shard at a time into a
         self._cache DataFrame.  Automatically evicts the previous shard.
         Each 1M-sequence shard is ~300–500 MB on disk and ~1–1.5 GB in RAM.
         Only one shard lives in memory at any time → safe for 64 GB machines.

    Parquet columns expected
    ------------------------
      sequence  : str  — amino acid sequence (pre-filtered: ≤ 1022 AA, no ambiguous)
      seq_id    : str  — UniRef50 cluster ID (e.g. UniRef50_A0A000XXX)
      seq_len   : int  — pre-computed length

    task tag : 'pretrain'
    """

    def __init__(
        self,
        parquet_dir:    str,
        max_length:     int  = 512,
        shard_pattern:  str  = "uniref50_clean_part_*.parquet",
        shuffle_shards: bool = True,
    ):
        self.tokenizer     = _get_tokenizer()
        self.max_length    = max_length
        self.parquet_dir   = Path(parquet_dir)

        # ── Discover shards ────────────────────────────────────────────────
        shard_paths = sorted(self.parquet_dir.glob(shard_pattern))
        if not shard_paths:
            raise FileNotFoundError(
                f"No parquet shards matching '{shard_pattern}' in {parquet_dir}.\n"
                "Run master_etl_pipeline.py first to generate UniRef50 shards."
            )
        if shuffle_shards:
            random.shuffle(shard_paths)

        # ── Read shard metadata only (fast — just parquet footer) ─────────
        self._shard_paths: list[Path] = []
        self._shard_offsets: list[int] = [0]   # cumulative row offset per shard

        print(f"Indexing UniRef50 shards in {parquet_dir} …")
        for sp in shard_paths:
            import pyarrow.parquet as pq
            meta      = pq.read_metadata(sp)
            n_rows    = meta.num_rows
            self._shard_paths.append(sp)
            self._shard_offsets.append(self._shard_offsets[-1] + n_rows)

        self._total_rows = self._shard_offsets[-1]
        print(
            f"  {len(self._shard_paths)} shard(s)  "
            f"→ {self._total_rows:,} total sequences"
        )

        # ── Shard cache — only one shard loaded at a time ─────────────────
        self._cache_shard_idx: int            = -1
        self._cache_df:        Optional[pd.DataFrame] = None

    def _load_shard(self, shard_idx: int) -> None:
        """Evict current shard and load the requested one."""
        if shard_idx == self._cache_shard_idx:
            return
        self._cache_df        = pd.read_parquet(self._shard_paths[shard_idx])
        self._cache_shard_idx = shard_idx

    def _global_to_local(self, global_idx: int) -> tuple[int, int]:
        """Convert a global row index to (shard_idx, local_row_idx)."""
        # Binary search over cumulative offsets
        lo, hi = 0, len(self._shard_offsets) - 2
        while lo < hi:
            mid = (lo + hi) // 2
            if self._shard_offsets[mid + 1] <= global_idx:
                lo = mid + 1
            else:
                hi = mid
        local_idx = global_idx - self._shard_offsets[lo]
        return lo, local_idx

    def __len__(self) -> int:
        return self._total_rows

    def __getitem__(self, idx: int) -> dict:
        shard_idx, local_idx = self._global_to_local(idx)
        self._load_shard(shard_idx)

        row = self._cache_df.iloc[local_idx]
        seq = str(row["sequence"])

        tok  = _tokenize(self.tokenizer, seq, self.max_length)
        mech = _protein_only_mech(seq, self.max_length)

        return {
            **tok,
            "mechanistic_features": mech,
            "task":                 "pretrain",
        }


# =============================================================================
# MultiTaskBatchSampler
# =============================================================================
class MultiTaskBatchSampler(Sampler):
    """
    Routes batches to specific Holo-GNN task heads by interleaving samples
    from multiple datasets in a controlled ratio.

    Design
    ------
    Rather than concatenating datasets (which would give an unbalanced mix
    dominated by the largest dataset), MultiTaskBatchSampler draws complete
    batches exclusively from one dataset at a time and cycles through tasks
    according to a configurable schedule.

    Usage example
    -------------
    ::

        from torch.utils.data import DataLoader, ConcatDataset
        from src.dataset import (
            MegaScaleDataset, FireProtDataset, ClinVarDataset, MultiTaskBatchSampler
        )

        ds_stability = MegaScaleDataset("cleaned/mega_scale_clean.parquet")
        ds_fireprot  = FireProtDataset ("cleaned/fireprotdb_clean.parquet")
        ds_clinvar   = ClinVarDataset  ("cleaned/clinvar_clean.parquet")

        sampler = MultiTaskBatchSampler(
            datasets  = [ds_stability, ds_fireprot, ds_clinvar],
            tasks     = ["stability",  "stability",  "idr"],
            batch_size = 64,
            # Draw 4 stability batches for every 1 ClinVar batch
            task_ratios = [4, 1, 1],
            shuffle    = True,
        )

        # In the training loop:
        for batch in loader:
            task = batch["task"][0]          # all items in a batch share the same task
            if task == "stability":
                dG_fwd, dG_rev = model((wt_data, mt_data), task="idr")
                loss, _        = criterion(dG_fwd, dG_rev, batch["label"])
            elif task == "idr":
                logits = model(data, task="idr_classify")
                loss   = F.binary_cross_entropy_with_logits(logits, batch["label"].float())

    Batch tag
    ---------
    Every item in a batch carries  batch['task'] = '<task_name>'  so the
    HoloGNN forward() dispatcher never needs to inspect the dataset object —
    it only looks at the string tag.

    Parameters
    ----------
    datasets    : list of Dataset — one per task.
    tasks       : list of str     — task tag per dataset (must align with datasets).
    batch_size  : int
    task_ratios : list of int     — how many consecutive batches to draw from
                                    each dataset before moving to the next.
                                    Defaults to [1, 1, …] (round-robin).
    shuffle     : bool            — shuffle within each dataset at epoch start.
    drop_last   : bool            — drop the final incomplete batch per dataset.
    """

    def __init__(
        self,
        datasets:    list[Dataset],
        tasks:       list[str],
        batch_size:  int,
        task_ratios: Optional[list[int]] = None,
        shuffle:     bool = True,
        drop_last:   bool = True,
    ):
        if len(datasets) != len(tasks):
            raise ValueError("datasets and tasks must have equal length.")
        self.datasets    = datasets
        self.tasks       = tasks
        self.batch_size  = batch_size
        self.task_ratios = task_ratios or [1] * len(datasets)
        self.shuffle     = shuffle
        self.drop_last   = drop_last

        # Pre-compute per-dataset index pools
        self._sizes = [len(ds) for ds in datasets]

    def _make_index_pools(self) -> list[list[int]]:
        pools = []
        for size in self._sizes:
            idx = list(range(size))
            if self.shuffle:
                random.shuffle(idx)
            pools.append(idx)
        return pools

    def __iter__(self) -> Iterator[list[int]]:
        pools    = self._make_index_pools()
        pointers = [0] * len(self.datasets)

        # Build task schedule: [0,0,0,0, 1,1, 2,2, 0,0,0,0, …]
        schedule: list[int] = []
        for ds_idx, ratio in enumerate(self.task_ratios):
            schedule.extend([ds_idx] * ratio)

        exhausted = set()
        while len(exhausted) < len(self.datasets):
            for ds_idx in schedule:
                if ds_idx in exhausted:
                    continue
                pool = pools[ds_idx]
                ptr  = pointers[ds_idx]

                # Gather one batch worth of global indices
                # (global index = offset_of_dataset + local_idx)
                offset  = sum(self._sizes[:ds_idx])
                end_ptr = ptr + self.batch_size
                chunk   = pool[ptr:end_ptr]

                if len(chunk) < self.batch_size:
                    if not self.drop_last and chunk:
                        yield [offset + i for i in chunk]
                    exhausted.add(ds_idx)
                    continue

                yield [offset + i for i in chunk]
                pointers[ds_idx] = end_ptr

    def __len__(self) -> int:
        total = 0
        for size, ratio in zip(self._sizes, self.task_ratios):
            n_batches = size // self.batch_size
            total    += n_batches * ratio
        return total