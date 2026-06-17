# Data

This directory does **not** ship the dataset. The Law School data used by the
experiments is **third-party data** with its own usage terms, so it is fetched
on demand rather than redistributed here.

## How to obtain the data

```bash
# from the repository root
LAWSCHOOL_DATA_URL="https://<source-you-are-entitled-to-use>/law_school.csv" \
    python data/get_lawschool_data.py
```

This writes `data/LawSchool.csv` (pipe-separated, `sep='|'`) with the columns:

```
race | sex | LSAT | UGPA | ZFYA | race_nonwhite | race_simpler
```

Then regenerate the counterfactual datasets (they are **derived**, not
downloaded):

```bash
Rscript src/get_cf_data_law_school.R
# produces: cf_LawSchool_lev3_doMale.csv, cf_LawSchool_lev3_doWhite.csv
```

## Data statement

The Law School dataset records, for ~21,790 US law-school students, academic
metrics (`LSAT`, `UGPA`, `ZFYA`) together with coarse demographic attributes
(`race`, `sex`). It contains **no names or direct identifiers**; it is an
anonymized, aggregated research dataset originally collected in the United
States in the 1990s. The demographic columns are *special-category* attributes
under GDPR Art. 9, but the dataset is anonymized and out of the scope of GDPR
for these records (Recital 26). Confirm anonymization and your redistribution
rights before publishing any copy.

## Citation

Original data:

> Wightman, L. F. (1998). *LSAC National Longitudinal Bar Passage Study.*
> LSAC Research Report Series. Law School Admission Council.
> Distributed by SEAPHE: http://www.seaphe.org/databases.php

Pre-processed variant (adds `ZFYA` and the simplified race columns), as used in
the algorithmic-fairness literature:

> Kusner, M. J., Loftus, J. R., Russell, C., & Silva, R. (2017).
> *Counterfactual Fairness.* Advances in Neural Information Processing Systems
> (NeurIPS) 30.

If you use this repository, please also cite the accompanying paper — see
`CITATION.cff` at the repository root.
