# *Mutatis Mutandis*

> Repository for [*Mutatis Mutandis: Revisiting the Comparator in Discrimination Testing*](https://arxiv.org/abs/2405.13693), forthcoming in [*Computational Intelligence*](https://onlinelibrary.wiley.com/journal/14678640), by [Jose M. Alvarez](https://cc-jalvarez.github.io/) (Santander AI Lab) and [Salvatore Ruggieri](https://pages.di.unipi.it/ruggieri/) (University of Pisa). Also visit [the main repository](https://github.com/cc-jalvarez/revisiting-the-comparator) maintained by the authors.

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/SantanderAI/mutatis-mutandis/actions/workflows/ci.yml/badge.svg)](https://github.com/SantanderAI/mutatis-mutandis/actions/workflows/ci.yml)
[![CodeQL](https://github.com/SantanderAI/mutatis-mutandis/actions/workflows/codeql.yml/badge.svg)](https://github.com/SantanderAI/mutatis-mutandis/actions/workflows/codeql.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/badge/linter-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg)](https://www.conventionalcommits.org/)

Part of [**Santander AI Open Source**](https://github.com/SantanderAI): open source AI projects from Banco Santander ([santander.com](https://santander.com)).

The paper re-examines the role of comparators in individual discrimination testing, arguing that comparator construction is a causal modeling problem. It introduces two comparator types: *Ceteris Paribus* (CP), which changes only the protected attribute, and *Mutatis Mutandis* (MM), which also adjusts non-protected attributes affected by the protected attribute. Through a real-world case study and the counterfactual situation testing method, the authors show how these comparator choices can lead to different discrimination assessments and highlight opportunities for machine learning in implementing MM comparators.

## References

If you make use of the code, please cite the following papers:

*Counterfactual Situation Testing: From Single to Multidimensional Discrimination*. Jose M. Alvarez, and Salvatore Ruggieri. Journal of Artificial Intelligence Research (JAIR), 82, 2279-2323, 2025.

<pre><code>
@article{DBLP:journals/jair/AlvarezR25,
  author       = {Jos{\'{e}} M. {\'{A}}lvarez and
                  Salvatore Ruggieri},
  title        = {Counterfactual Situation Testing: From Single to Multidimensional
                  Discrimination},
  journal      = {J. Artif. Intell. Res.},
  volume       = {82},
  pages        = {2279--2323},
  year         = {2025}
}
</code></pre>

*Mutatis Mutandis: Revisiting the Comparator in Discrimination Testing*. Jose M. Alvarez, and Salvatore Ruggieri. Forthcoming in Computational Intelligence, 2026.

<pre><code>
@article{DBLP:journals/corr/abs-2405-13693,
  author       = {Jos{\'{e}} M. {\'{A}}lvarez and
                  Salvatore Ruggieri},
  title        = {Mutatis Mutandis: Revisiting the Comparator in Discrimination Testing},
  journal      = {CoRR},
  volume       = {abs/2405.13693},
  year         = {2024}
}
</code></pre>

The Law School dataset must be cited separately; see [data/README.md](data/README.md) for the data statement and citation.

## Overview

### Key Features

- `situation_testing` Python package with a configurable `SituationTesting` estimator.
- Distance functions for mixed continuous / ordinal / nominal attributes (`kdd2011`).
- Wald confidence intervals and a per-individual discrimination report.
- Counterfactual situation testing (`cfST`) and counterfactual-unfairness coding.
- Reproducible Law School experiment pipeline (Python + R).

### Use Cases

- Auditing a decision rule for discrimination against a protected group.
- Comparing factual vs counterfactual comparators in fairness analysis.
- Reproducing the experiments in Section 4 of the paper.

## Quick Start

### Installation

```bash
git clone https://github.com/SantanderAI/mutatis-mutandis.git
cd mutatis-mutandis
pip install -e ".[dev]"
```

### 1. Obtain the data

The Law School dataset is **third-party data and is not redistributed in this
repository**. Fetch it from a source you are entitled to use:

```bash
# see data/README.md for sources, data statement + citation
LAWSCHOOL_DATA_URL="https://<source-you-are-entitled-to-use>/law_school.csv" \
    python data/get_lawschool_data.py
```

This writes `data/LawSchool.csv` (pipe-separated).

### 2. Generate the counterfactual datasets

```bash
# derived, not downloaded
Rscript src/get_cf_data_law_school.R
```

### 3. Run the experiments and analysis

```bash
python src/run_exp_law_school.py        # writes results/ CSVs
jupyter notebook src/analysis_law_school.ipynb   # figures
```

Under the current setup, the RStan models are not required.

### Python API

```python
import pandas as pd
from situation_testing import SituationTesting

df = pd.read_csv("data/LawSchool.csv", sep="|")

st = SituationTesting()
st.setup_baseline(df, nominal_atts=["Gender"], continuous_atts=["LSAT", "UGPA"])
diffs = st.run(
    target_att="Y",
    target_val={"negative": 0},
    sensitive_att="Gender",
    sensitive_val={"protected": "F"},
    k=50,
)
report = st.get_test_discrimination()
```

## Project Structure

```
mutatis-mutandis/
├── src/
│   ├── situation_testing/        # importable Python package
│   │   ├── __init__.py
│   │   ├── situation_testing.py  # SituationTesting estimator
│   │   ├── _distance_functions.py
│   │   └── _utils.py
│   ├── run_exp_law_school.py     # experiment driver
│   ├── get_cf_data_law_school.R  # counterfactual data generation (R)
│   └── analysis_law_school.ipynb # figures
├── data/                         # data fetch script + data statement (NO dataset)
├── results/                      # experiment outputs (CSV)
├── figures/                      # paper figures
├── tests/                        # pytest suite
├── pyproject.toml
├── LICENSE / NOTICE
├── CONTRIBUTING.md / CODE_OF_CONDUCT.md / CODEOWNERS
├── .github/SECURITY.md
└── CHANGELOG.md
```

## Requirements

Python (installed via `pip install -e .`):

- Python >= 3.10
- numpy >= 1.24
- pandas >= 2.0
- scipy >= 1.10
- scikit-learn >= 1.2

R (for `src/get_cf_data_law_school.R`):

- R >= 4.0 with the packages used by the script (see the script header).

## Contributing

We welcome contributions from the community. Please read
[CONTRIBUTING.md](CONTRIBUTING.md) before submitting a pull request. By
contributing, you agree to the terms of our Contributor License Agreement (CLA).

## Security

To report a security vulnerability, follow the process described in
[SECURITY.md](.github/SECURITY.md). Do not open a public issue for security
vulnerabilities.

## License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE)
file for details.

```
Copyright (c) 2026 José M. Álvarez
SPDX-License-Identifier: Apache-2.0
```

The copyright of the research code is held by the paper author(s). The original
research repository is
[cc-jalvarez/revisiting-the-comparator](https://github.com/cc-jalvarez/revisiting-the-comparator).
This open-source release is distributed and maintained by **Santander AI Lab**
with the author's consent. See [NOTICE](NOTICE) for details.
