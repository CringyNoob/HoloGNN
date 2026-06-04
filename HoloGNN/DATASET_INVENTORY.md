# Holo-GNN — Dataset Inventory

> **Generated:** 2026-04-04 20:38:55  
> **Scanned directory:** `D:\ML MODELS\HOLO_GNN_PROJECT\DATA`  
> **Total files:** 18  
> **Total size:** 69250.16 MB  

---

## Table of Contents

- [[IDR_HEAD]](#idr-head) — 2 file(s)
- [[PRE-TRAINING]](#pretraining) — 2 file(s)
- [[PROTEOMICS]](#proteomics) — 2 file(s)
- [[STABILITY_HEAD]](#stability-head) — 13 file(s)

---

## [IDR_HEAD]

**2 file(s)**

### `clinvar\clinvar.vcf`

| Field | Value |
|---|---|
| **Size** | 1773.49 MB |
| **Extension** | `.vcf` |
| **Tags** | [IDR_HEAD] |
| **Columns / Headers** | `CHROM`, `POS`, `ID`, `REF`, `ALT`, `QUAL`, `FILTER`, `INFO` |

> #CHROM header found; 44 meta-lines

---

### `clinvar\clinvar.vcf.gz`

| Field | Value |
|---|---|
| **Size** | 177.55 MB |
| **Extension** | `.gz` |
| **Tags** | [IDR_HEAD] |
| **Columns / Headers** | *(none extracted)* |

> gzip archive, 177.55 MB compressed

---

## [PRE-TRAINING]

**2 file(s)**

### `uniref50\uniref50.fasta`

| Field | Value |
|---|---|
| **Size** | 22946.45 MB |
| **Extension** | `.fasta` |
| **Tags** | [PRE-TRAINING] · [STABILITY_HEAD] |
| **Columns / Headers** | *(none extracted)* |

**Preview (first 10 lines):**
```
>UniRef50_UPI002E2621C6 uncharacterized protein LOC134193701 n=1 Tax=Corticium candelabrum TaxID=121492 RepID=UPI002E2621C6
    MGRIRVWVGTSIPNPVNAHQLVYLKGMAKTKKLILLLFVAAQPNFKEWSLDVDASTLVLT
    FEANSVLSVKPDCSKVTIHSTANGVKNVTLTNSGNGTLDAANDQASCTIDAKDLDNIKLE
    TTLGTNTTNTFLEVKAGFGTKNGTTEFTQGSPYTAAALVTPDVTAPEISATVGFSEFDLN
    SGRVTIAFTEAVDVSTLKFTKLAFRDAKLTGKTSTTGYCNVTKDGKCDAAFCKNGATVVL
    EVDNVDLNCIKSKRGLCTKDSDCIITLEEDDFIQDMAGNKLGKYESGTTANAAETLLHKF
    VPDITSPTLDNFDLDLNANTLTLEFSETVDAKTLKADGLTIQGNGNTADVSLQVKLTSES
    TTESSDSATIIVDIAPADGAKLKMSTNIATKTGDSYIAVATSAMNDMSGNAVKPISSTAA
    KQVRRFTNDTSAAVLSKFSLDLNTNQLTLTFDEPVKVDSLNFTLFTLQSTAAGGTEVKLT
    GSTTMTTGTVREVVVDLSEAALVSIKSNVTVATSLTDTYLTHESSAFKNFIDLLSADLAT
```

---

### `uniref50\uniref50.fasta.gz`

| Field | Value |
|---|---|
| **Size** | 11811.53 MB |
| **Extension** | `.gz` |
| **Tags** | [PRE-TRAINING] |
| **Columns / Headers** | *(none extracted)* |

> gzip archive, 11811.53 MB compressed

---

## [PROTEOMICS]

**2 file(s)**

### `massive_kb\LIBRARY_TO_SPTXT-3440aba4-download_sptxt_library-main.sptxt`

| Field | Value |
|---|---|
| **Size** | 27425.36 MB |
| **Extension** | `.sptxt` |
| **Tags** | [PROTEOMICS] |
| **Columns / Headers** | *(none extracted)* |

**Preview (first 10 lines):**
```
Name: AAPSPSGGGGSGGGSGSGTPGPVGSPAPGHPAVSSMQGK/3
    Comment: Parent=1248.622192382813 Mods=2/0,A,Acetyl/38,K,TMT
    Num peaks: 23
    230.17042541503906	4254.706615621117	"?"
    248.18093872070312	7064.508902427105	"?"
    298.11468505859375	1067.836646805394	"?"
    369.17681884765625	1176.8354014799786	"?"
    376.27606201171875	2337.4440043845175	"?"
    433.2968444824219	3345.7398035088295	"?"
    561.3554077148438	1737.1343610991503	"?"
```

---

### `massive_kb\params.xml`

| Field | Value |
|---|---|
| **Size** | 0.62 KB |
| **Extension** | `.xml` |
| **Tags** | [PROTEOMICS] |
| **Columns / Headers** | *(none extracted)* |

> Unsupported extension `.xml` — no content parsing.

---

## [STABILITY_HEAD]

**13 file(s)**

### `fireprotdb\fireprotdb_20251015-164116.csv`

| Field | Value |
|---|---|
| **Size** | 1693.25 MB |
| **Extension** | `.csv` |
| **Tags** | [STABILITY_HEAD] |
| **Columns / Headers** | `EXPERIMENT_ID`, `SEQUENCE_ID`, `MUTANT_ID`, `SOURCE_SEQUENCE_ID`, `TARGET_SEQUENCE_ID`, `SEQUENCE_LENGTH`, `SUBSTITUTION`, `INSERTION`, `DELETION`, `PROTEIN`, `ORGANISM`, `CHYMOTRYPSIN_ML`, `CM`, `DCP`, `DDG`, `DG`, `DH`, `DHVH`, `DOMAINOME_DDG`, `DOMAINOME_DDG_STD` … (+33 more) |

> 5 preview rows

---

### `mega_scale_cdna\K50_dG_tables\K50_dG_tables\Lib1_K50dG.csv`

| Field | Value |
|---|---|
| **Size** | 149.19 MB |
| **Extension** | `.csv` |
| **Tags** | [STABILITY_HEAD] |
| **Columns / Headers** | `name`, `dna_seq`, `log10_K50_t`, `log10_K50_t_95CI_high`, `log10_K50_t_95CI_low`, `log10_K50_t_95CI`, `fitting_error_t`, `log10_K50unfolded_t`, `deltaG_t`, `deltaG_t_95CI_high`, `deltaG_t_95CI_low`, `deltaG_t_95CI`, `log10_K50_c`, `log10_K50_c_95CI_high`, `log10_K50_c_95CI_low`, `log10_K50_c_95CI`, `fitting_error_c`, `log10_K50unfolded_c`, `deltaG_c`, `deltaG_c_95CI_high` … (+6 more) |

> 5 preview rows

---

### `mega_scale_cdna\K50_dG_tables\K50_dG_tables\Lib2_K50dG.csv`

| Field | Value |
|---|---|
| **Size** | 350.35 MB |
| **Extension** | `.csv` |
| **Tags** | [STABILITY_HEAD] |
| **Columns / Headers** | `name`, `dna_seq`, `log10_K50_t`, `log10_K50_t_95CI_high`, `log10_K50_t_95CI_low`, `log10_K50_t_95CI`, `fitting_error_t`, `log10_K50unfolded_t`, `deltaG_t`, `deltaG_t_95CI_high`, `deltaG_t_95CI_low`, `deltaG_t_95CI`, `log10_K50_c`, `log10_K50_c_95CI_high`, `log10_K50_c_95CI_low`, `log10_K50_c_95CI`, `fitting_error_c`, `log10_K50unfolded_c`, `deltaG_c`, `deltaG_c_95CI_high` … (+6 more) |

> 5 preview rows

---

### `mega_scale_cdna\K50_dG_tables\K50_dG_tables\Lib3_K50dG.csv`

| Field | Value |
|---|---|
| **Size** | 505.20 MB |
| **Extension** | `.csv` |
| **Tags** | [STABILITY_HEAD] |
| **Columns / Headers** | `name`, `dna_seq`, `log10_K50_t`, `log10_K50_t_95CI_high`, `log10_K50_t_95CI_low`, `log10_K50_t_95CI`, `fitting_error_t`, `log10_K50unfolded_t`, `deltaG_t`, `deltaG_t_95CI_high`, `deltaG_t_95CI_low`, `deltaG_t_95CI`, `log10_K50_c`, `log10_K50_c_95CI_high`, `log10_K50_c_95CI_low`, `log10_K50_c_95CI`, `fitting_error_c`, `log10_K50unfolded_c`, `deltaG_c`, `deltaG_c_95CI_high` … (+6 more) |

> 5 preview rows

---

### `mega_scale_cdna\K50_dG_tables\K50_dG_tables\Lib4_K50dG.csv`

| Field | Value |
|---|---|
| **Size** | 186.38 MB |
| **Extension** | `.csv` |
| **Tags** | [STABILITY_HEAD] |
| **Columns / Headers** | `name`, `dna_seq`, `log10_K50_t`, `log10_K50_t_95CI_high`, `log10_K50_t_95CI_low`, `log10_K50_t_95CI`, `fitting_error_t`, `log10_K50unfolded_t`, `deltaG_t`, `deltaG_t_95CI_high`, `deltaG_t_95CI_low`, `deltaG_t_95CI`, `log10_K50_c`, `log10_K50_c_95CI_high`, `log10_K50_c_95CI_low`, `log10_K50_c_95CI`, `fitting_error_c`, `log10_K50unfolded_c`, `deltaG_c`, `deltaG_c_95CI_high` … (+6 more) |

> 5 preview rows

---

### `mega_scale_cdna\Processed_K50_dG_datasets\Processed_K50_dG_datasets\Double_DMS_list.csv`

| Field | Value |
|---|---|
| **Size** | 0.08 MB |
| **Extension** | `.csv` |
| **Tags** | [STABILITY_HEAD] |
| **Columns / Headers** | `K50_corr`, `dg_corr`, `recon_corr`, `NA_num`, `max_ep_dG`, `wt_ep_dG`, `double_mut_name` |

> 5 preview rows

---

### `mega_scale_cdna\Processed_K50_dG_datasets\Processed_K50_dG_datasets\Heat_maps_double_DMS.zip`

| Field | Value |
|---|---|
| **Size** | 84.93 MB |
| **Extension** | `.zip` |
| **Tags** | [STABILITY_HEAD] |
| **Columns / Headers** | *(none extracted)* |

> zip archive, 84.93 MB compressed, 1452 member(s)

---

### `mega_scale_cdna\Processed_K50_dG_datasets\Processed_K50_dG_datasets\Heat_maps_single_DMS.zip`

| Field | Value |
|---|---|
| **Size** | 255.53 MB |
| **Extension** | `.zip` |
| **Tags** | [STABILITY_HEAD] |
| **Columns / Headers** | *(none extracted)* |

> zip archive, 255.53 MB compressed, 1968 member(s)

---

### `mega_scale_cdna\Processed_K50_dG_datasets\Processed_K50_dG_datasets\Single_DMS_list.csv`

| Field | Value |
|---|---|
| **Size** | 0.41 MB |
| **Extension** | `.csv` |
| **Tags** | [STABILITY_HEAD] |
| **Columns / Headers** | `name`, `aa_seq`, `frac_NA`, `raw_corr`, `dg_corr`, `slope`, `y_intercept`, `width_KT`, `width_KC`, `wt_dg_std_max`, `wt_k50_std_max`, `wt_dg_max`, `wt_dgt_med`, `wt_dgc_med`, `wt_dg_med`, `wt_kt`, `wt_kc`, `wt_k50_diff_max`, `wt_d_from_line`, `frac_pos_with_hydrophobic_stabilzing_muts` … (+5 more) |

> 5 preview rows

---

### `mega_scale_cdna\Processed_K50_dG_datasets\Processed_K50_dG_datasets\Triple_DMS_list.csv`

| Field | Value |
|---|---|
| **Size** | 1.53 KB |
| **Extension** | `.csv` |
| **Tags** | [STABILITY_HEAD] |
| **Columns / Headers** | `triple_mut_name`, `pdb_name`, `pos1`, `pos2`, `pos3`, `aa1`, `aa2`, `aa3` |

> 5 preview rows

---

### `mega_scale_cdna\Processed_K50_dG_datasets\Processed_K50_dG_datasets\Tsuboyama2023_Dataset1_20230416.csv`

| Field | Value |
|---|---|
| **Size** | 1225.12 MB |
| **Extension** | `.csv` |
| **Tags** | [STABILITY_HEAD] |
| **Columns / Headers** | `name`, `dna_seq`, `log10_K50_t`, `log10_K50_t_95CI_high`, `log10_K50_t_95CI_low`, `log10_K50_t_95CI`, `fitting_error_t`, `log10_K50unfolded_t`, `deltaG_t`, `deltaG_t_95CI_high`, `deltaG_t_95CI_low`, `deltaG_t_95CI`, `log10_K50_c`, `log10_K50_c_95CI_high`, `log10_K50_c_95CI_low`, `log10_K50_c_95CI`, `fitting_error_c`, `log10_K50unfolded_c`, `deltaG_c`, `deltaG_c_95CI_high` … (+8 more) |

> 5 preview rows

---

### `mega_scale_cdna\Processed_K50_dG_datasets\Processed_K50_dG_datasets\Tsuboyama2023_Dataset2_Dataset3_20230416.csv`

| Field | Value |
|---|---|
| **Size** | 665.34 MB |
| **Extension** | `.csv` |
| **Tags** | [STABILITY_HEAD] |
| **Columns / Headers** | `name`, `dna_seq`, `log10_K50_t`, `log10_K50_t_95CI_high`, `log10_K50_t_95CI_low`, `log10_K50_t_95CI`, `fitting_error_t`, `log10_K50unfolded_t`, `deltaG_t`, `deltaG_t_95CI_high`, `deltaG_t_95CI_low`, `deltaG_t_95CI`, `log10_K50_c`, `log10_K50_c_95CI_high`, `log10_K50_c_95CI_low`, `log10_K50_c_95CI`, `fitting_error_c`, `log10_K50unfolded_c`, `deltaG_c`, `deltaG_c_95CI_high` … (+17 more) |

> 5 preview rows

---

### `uniref50\uniref50.fasta`

| Field | Value |
|---|---|
| **Size** | 22946.45 MB |
| **Extension** | `.fasta` |
| **Tags** | [PRE-TRAINING] · [STABILITY_HEAD] |
| **Columns / Headers** | *(none extracted)* |

**Preview (first 10 lines):**
```
>UniRef50_UPI002E2621C6 uncharacterized protein LOC134193701 n=1 Tax=Corticium candelabrum TaxID=121492 RepID=UPI002E2621C6
    MGRIRVWVGTSIPNPVNAHQLVYLKGMAKTKKLILLLFVAAQPNFKEWSLDVDASTLVLT
    FEANSVLSVKPDCSKVTIHSTANGVKNVTLTNSGNGTLDAANDQASCTIDAKDLDNIKLE
    TTLGTNTTNTFLEVKAGFGTKNGTTEFTQGSPYTAAALVTPDVTAPEISATVGFSEFDLN
    SGRVTIAFTEAVDVSTLKFTKLAFRDAKLTGKTSTTGYCNVTKDGKCDAAFCKNGATVVL
    EVDNVDLNCIKSKRGLCTKDSDCIITLEEDDFIQDMAGNKLGKYESGTTANAAETLLHKF
    VPDITSPTLDNFDLDLNANTLTLEFSETVDAKTLKADGLTIQGNGNTADVSLQVKLTSES
    TTESSDSATIIVDIAPADGAKLKMSTNIATKTGDSYIAVATSAMNDMSGNAVKPISSTAA
    KQVRRFTNDTSAAVLSKFSLDLNTNQLTLTFDEPVKVDSLNFTLFTLQSTAAGGTEVKLT
    GSTTMTTGTVREVVVDLSEAALVSIKSNVTVATSLTDTYLTHESSAFKNFIDLLSADLAT
```

---

## Extension Summary

| Extension | Count |
|---|---|
| `.csv` | 10 |
| `.gz` | 2 |
| `.zip` | 2 |
| `.vcf` | 1 |
| `.sptxt` | 1 |
| `.xml` | 1 |
| `.fasta` | 1 |

---

*Report generated by `dataset_scanner.py` — Holo-GNN V5.0*
