# Data

This directory should contain the NMR bucket tables used in the study.
These files are **not included** in the repository due to patient privacy.

## Required Files

| File | Description | Shape |
|------|-------------|-------|
| `bucket_table_orina_COVID+PRECANCER_noscaling.xlsx` | Urine NMR bucket table (raw, no normalization) | 289 × 236 (2 ID cols + 234 spectral bins) |
| `bucket_table_suero_COVID+PRECANCER_scaling.xlsx` | Serum NMR bucket table (TSP-normalized) + clinical variables | 289 × 257 (2 ID cols + 231 spectral bins + 24 clinical) |

## Cohort

- 73 COVID-19 patients
- 31 healthy controls
- 21 colorectal cancer patients
- 164 cancer-free controls

Total: 289 participants (ages 31–89, both sexes).

## Data Availability

Requests for access to the data should be directed to the corresponding authors:
- Francisco Manuel Arrabal-Campos (fmarrabal@ual.es)
- Ignacio Fernández (ifernan@ual.es)

## Reference

NMR acquisition and sample collection were performed as described in:

> Tristán AI et al. "Metabolomic profiling of COVID-19 using serum and urine samples
> in intensive care and medical ward cohorts" *Sci. Rep.* 2024, 14, 23713.
