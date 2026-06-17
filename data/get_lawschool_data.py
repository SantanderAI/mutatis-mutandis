#!/usr/bin/env python3
# Copyright (c) 2026 José M. Álvarez
# SPDX-License-Identifier: Apache-2.0
"""Fetch the Law School dataset used by the experiments in this repository.

The dataset is NOT redistributed in this repository. It is third-party data
(the LSAC National Longitudinal Bar Passage Study) and must be obtained from
its original source under that source's terms. This script downloads it into
``data/LawSchool.csv`` (pipe-separated, ``sep='|'``), the format the R and
Python scripts expect.

Provenance / citation
---------------------
Original data:
    Wightman, L. F. (1998). LSAC National Longitudinal Bar Passage Study.
    LSAC Research Report Series. Law School Admission Council.
    Distributed by SEAPHE: http://www.seaphe.org/databases.php

Pre-processed variant (adds ZFYA = z-scored first-year average, and the
simplified race columns) as commonly used in the algorithmic-fairness
literature:
    Kusner, M. J., Loftus, J. R., Russell, C., & Silva, R. (2017).
    Counterfactual Fairness. Advances in Neural Information Processing
    Systems (NeurIPS) 30.

See ``data/README.md`` for the full data statement and redistribution notes.

Usage
-----
    # point it at a source you are entitled to use:
    LAWSCHOOL_DATA_URL="https://<source>/law_school.csv" python data/get_lawschool_data.py
    # or
    python data/get_lawschool_data.py --url "https://<source>/law_school.csv"

After downloading the factual data, regenerate the counterfactual datasets
(``cf_LawSchool_lev3_doMale.csv`` / ``cf_LawSchool_lev3_doWhite.csv``) with
``src/get_cf_data_law_school.R`` — they are derived, not downloaded.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

# Columns the downstream scripts rely on (pipe-separated).
EXPECTED_COLUMNS = [
    "race",
    "sex",
    "LSAT",
    "UGPA",
    "ZFYA",
    "race_nonwhite",
    "race_simpler",
]

DATA_DIR = Path(__file__).resolve().parent
TARGET = DATA_DIR / "LawSchool.csv"

SOURCES_HELP = """
No data source URL was provided.

This repository does NOT ship the Law School dataset, because it is third-party
data with its own usage terms. Provide a source you are entitled to use:

  - Original LSAC data via SEAPHE:        http://www.seaphe.org/databases.php
  - Pre-processed (ZFYA) variant as used in Kusner et al. (2017),
    redistributed by several public algorithmic-fairness toolkits.

Then set the URL and re-run:

  LAWSCHOOL_DATA_URL="https://<source>/law_school.csv" python data/get_lawschool_data.py

The downloaded file must be convertible to the pipe-separated schema:
  {cols}
""".format(cols=" | ".join(EXPECTED_COLUMNS))


def _validate_schema(path: Path) -> None:
    """Confirm the downloaded file exposes the expected columns."""
    try:
        import pandas as pd
    except ImportError:
        print("[warn] pandas not installed — skipping schema validation.", file=sys.stderr)
        return
    # Try pipe first (target format), then comma (common upstream format).
    for sep in ("|", ","):
        try:
            df = pd.read_csv(path, sep=sep, nrows=5)
        except Exception:
            continue
        cols = set(df.columns)
        if set(EXPECTED_COLUMNS).issubset(cols):
            if sep != "|":
                # Re-write in the pipe-separated format the scripts expect.
                full = pd.read_csv(path, sep=sep)
                full.to_csv(path, sep="|", index=False)
            return
    raise SystemExit(
        f"[error] {path.name} does not expose the expected columns "
        f"{EXPECTED_COLUMNS}. Check the source or adapt the preprocessing."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("LAWSCHOOL_DATA_URL"),
        help="URL of a Law School dataset you are entitled to use.",
    )
    parser.add_argument(
        "--sha256",
        default=os.environ.get("LAWSCHOOL_DATA_SHA256"),
        help="Optional expected SHA-256 of the downloaded file (integrity pin).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite an existing data/LawSchool.csv."
    )
    args = parser.parse_args()

    if not args.url:
        print(SOURCES_HELP, file=sys.stderr)
        return 2

    if TARGET.exists() and not args.force:
        print(f"[skip] {TARGET} already exists. Use --force to overwrite.")
        return 0

    print(f"[info] downloading dataset from {args.url}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(args.url, TARGET)
    except Exception as exc:  # noqa: BLE001 - surface any download failure to the user
        raise SystemExit(f"[error] download failed: {exc}")

    digest = hashlib.sha256(TARGET.read_bytes()).hexdigest()
    print(f"[info] sha256={digest}")
    if args.sha256 and digest.lower() != args.sha256.lower():
        TARGET.unlink(missing_ok=True)
        raise SystemExit(f"[error] checksum mismatch: expected {args.sha256}, got {digest}")

    _validate_schema(TARGET)
    print(f"[ok] wrote {TARGET} (pipe-separated).")
    print("[next] regenerate counterfactuals with src/get_cf_data_law_school.R")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
