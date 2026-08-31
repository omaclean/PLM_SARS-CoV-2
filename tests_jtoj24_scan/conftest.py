#!/usr/bin/env python3
"""Shared fixtures for the ``JtoJ24_scan`` escape test suite.

DESIGN RULE FOR EVERYTHING IN THIS FILE
=======================================
A fixture whose expected values have to be computed by the code under test is
worthless: it can only ever assert that the code agrees with itself. Every
fixture here therefore ships **independently-derived ground truth**, and that
ground truth is either a literal, or arithmetic a reader can redo in one line.

The synthetic geometry
======================
Coordinates are chosen so that every quantity the escape code computes has an
exact closed form. Root at the origin, three mutations::

    root                    (0, 0, 0)
    N122D                   (1, 0, 0)
    T135K                   (0, 2, 0)
    K189R                   (0, 0, 2)
    N122D+T135K             (1.5, 2, 0)     <- 0.5 of planted epistasis on X
    N122D+K189R             (1, 0, 2)       <- exactly additive
    T135K+K189R             (0, 2, 2)       <- exactly additive
    N122D+T135K+K189R       (1, 2, 2)       <- exactly additive: the endpoint

The endpoint axis is therefore ``(1, 2, 2)``, whose length is exactly **3**, so
the unit vector is ``(1/3, 2/3, 2/3)`` and every projection is a third of an
integer. By hand:

===============  ========  ================  ===========================  ==========
genotype         \\|Δ\\|     along axis        off axis                     frac
===============  ========  ================  ===========================  ==========
N122D            1         1/3               sqrt(1 - 1/9)  = 2*sqrt2/3   1/9
T135K            2         4/3               sqrt(4 - 16/9) = 2*sqrt5/3   4/9
K189R            2         4/3               2*sqrt5/3                    4/9
N122D+T135K      2.5       5.5/3             --                           --
N122D+K189R      sqrt5     5/3               --                           --
T135K+K189R      2*sqrt2   8/3               --                           --
endpoint         3         3                 0                            1
===============  ========  ================  ===========================  ==========

Epistasis along the axis is then exactly:

* ``N122D+T135K``: 5.5/3 - (1/3 + 4/3) = **1/6**, from a planted ``(0.5, 0, 0)``
* ``N122D+K189R``: 5/3 - (1/3 + 4/3) = **0**
* ``T135K+K189R``: 8/3 - (4/3 + 4/3) = **0**

Note the deliberate trap: the *triple* is exactly additive while one *pair* is
not. Anything that infers pairwise epistasis from the endpoint instead of
measuring the double mutant gets this case wrong.

The synthetic immune landscape
==============================
Dates and counts are chosen so the weights are exact decimals. With
``--as-of 2023.0`` and a 1-year half-life, a 2020 sequence has raw weight
``0.5**3 = 0.125`` and a 2022 sequence ``0.5**1 = 0.5``. The landscape holds
**four** 2020 sequences and **one** 2022 sequence, so:

* ``--normalise-by none``  -> 2020 mass 4*0.125 = 0.5, 2022 mass 0.5  -> **50/50**
* ``--normalise-by year``  -> 2020 mass 0.125,        2022 mass 0.5   -> **20/80**

which is exactly the surveillance-effort correction, in numbers you can check
without running anything. Two of the four 2020 sequences sit on *identical*
coordinates, so ``--within-period unique`` redistributes within 2020 as well.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

# --- Import bootstrap ------------------------------------------------------
# The scan scripts are not a package and are not installed; they live in
# scripts/JtoJ24_scan and import each other by bare module name.
REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_DIR = REPO_ROOT / "scripts" / "JtoJ24_scan"
for _path in (SCAN_DIR, REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


###############################################################################
# Ground truth, as literals and one-line arithmetic
###############################################################################
#: Endpoint axis length. |(1, 2, 2)| = sqrt(1 + 4 + 4) = 3.
SPAN = 3.0

#: mutation -> (total displacement, component along the axis, off-axis component)
EXPECTED_SINGLES = {
    "N122D": (1.0, 1.0 / 3.0, 2.0 * np.sqrt(2.0) / 3.0),
    "T135K": (2.0, 4.0 / 3.0, 2.0 * np.sqrt(5.0) / 3.0),
    "K189R": (2.0, 4.0 / 3.0, 2.0 * np.sqrt(5.0) / 3.0),
}

#: pair -> epistasis along the axis
EXPECTED_PAIR_EPISTASIS = {
    "N122D+T135K": 1.0 / 6.0,
    "N122D+K189R": 0.0,
    "T135K+K189R": 0.0,
}

#: The planted non-additive displacement of the one non-additive pair.
PLANTED_EPISTASIS_VECTOR = np.array([0.5, 0.0, 0.0])

#: Genotype label -> coordinates. The single source of truth for the fixtures.
GENOTYPE_COORDINATES = {
    "root": (0.0, 0.0, 0.0),
    "N122D": (1.0, 0.0, 0.0),
    "T135K": (0.0, 2.0, 0.0),
    "K189R": (0.0, 0.0, 2.0),
    "N122D+T135K": (1.5, 2.0, 0.0),
    "N122D+K189R": (1.0, 0.0, 2.0),
    "T135K+K189R": (0.0, 2.0, 2.0),
    "N122D+T135K+K189R": (1.0, 2.0, 2.0),
}

#: Background rows: (name, collection date, subclade, X, Y, Z).
#: Rows 2 and 3 are an exact coordinate tie -- that is what --within-period
#: unique collapses. Row 5 is the only 2022 sequence.
BACKGROUND_ROWS = [
    ("A/Test/1/2020", "2020-01-01", "X.1", 0.0, 0.0, 0.0),
    ("A/Test/2/2020", "2020-01-01", "X.1", 6.0, 0.0, 0.0),
    ("A/Test/3/2020", "2020-01-01", "X.1", 6.0, 0.0, 0.0),
    ("A/Test/4/2020", "2020-01-01", "X.2", 0.0, 6.0, 0.0),
    ("A/Test/5/2022", "2022-01-01", "Y.1", 0.0, 0.0, 6.0),
]

#: Raw recency weights at as-of 2023.0 with a 1-year half-life.
AS_OF_2023 = 2023.0
RECENCY_2020 = 0.125  # 0.5 ** 3
RECENCY_2022 = 0.5    # 0.5 ** 1


###############################################################################
# Genotype fixtures
###############################################################################
def _genotype_frame(labels) -> pd.DataFrame:
    """Minimal genotype table: exactly the columns the escape code reads."""
    rows = []
    for label in labels:
        x, y, z = GENOTYPE_COORDINATES[label]
        rows.append(
            {
                "genotype_id": label,
                "genotype_h3": label,
                "n_fixed": 0 if label == "root" else label.count("+") + 1,
                "X": x,
                "Y": y,
                "Z": z,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_genotypes() -> pd.DataFrame:
    """The full 8-genotype hypercube of the documented geometry."""
    return _genotype_frame(GENOTYPE_COORDINATES)


@pytest.fixture
def pairless_genotypes() -> pd.DataFrame:
    """Root, singles and the endpoint only -- a --max-background-size 1 run."""
    return _genotype_frame(
        ["root", "N122D", "T135K", "K189R", "N122D+T135K+K189R"]
    )


@pytest.fixture
def observed_frame() -> pd.DataFrame:
    """Stand-in for observed_sequence_embeddings.csv."""
    return pd.DataFrame(
        {
            "sequence_id": ["EPI1|HA|A/Test/1/2022|X|J", "EPI2|HA|A/Test/2/2024|X|J.2.4"],
            "X": [0.0, 1.0],
            "Y": [0.0, 2.0],
            "Z": [0.0, 2.0],
            "lineage": ["J", "J.2.4"],
        }
    )


def _write_run_dir(directory: Path, genotypes: pd.DataFrame,
                   observed: pd.DataFrame | None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    genotypes.to_csv(directory / "genotype_embeddings.csv", index=False)
    if observed is not None:
        observed.to_csv(directory / "observed_sequence_embeddings.csv", index=False)
    (directory / "run_metadata.json").write_text(
        json.dumps(
            {
                "start_header": "EPI1|HA|A/Test/1/2022|EPI_ISL_1|J",
                "end_header": "EPI2|HA|A/Test/2/2024|EPI_ISL_2|J.2.4",
                "mutations_h3": ["N122D", "T135K", "K189R"],
            },
            indent=2,
        )
    )
    return directory


@pytest.fixture
def run_dir(tmp_path, synthetic_genotypes, observed_frame) -> Path:
    """A completed-looking PLANT run directory with the full hypercube."""
    return _write_run_dir(tmp_path / "plant", synthetic_genotypes, observed_frame)


@pytest.fixture
def pairless_run_dir(tmp_path, pairless_genotypes, observed_frame) -> Path:
    """A run directory with no two-mutation backgrounds."""
    return _write_run_dir(tmp_path / "plant_nopairs", pairless_genotypes, observed_frame)


@pytest.fixture
def bare_run_dir(tmp_path, synthetic_genotypes) -> Path:
    """A run directory with no observed CSV and no metadata."""
    directory = tmp_path / "plant_bare"
    directory.mkdir(parents=True, exist_ok=True)
    synthetic_genotypes.to_csv(directory / "genotype_embeddings.csv", index=False)
    return directory


###############################################################################
# Background / immune-landscape fixtures
###############################################################################
@pytest.fixture
def background_frame() -> pd.DataFrame:
    """Raw background table in the on-disk column layout."""
    return pd.DataFrame(
        BACKGROUND_ROWS, columns=["name", "collection date", "subclade", "X", "Y", "Z"]
    )


@pytest.fixture
def background_csv(tmp_path, background_frame) -> Path:
    path = tmp_path / "backgrounds.csv"
    background_frame.to_csv(path, index=False)
    return path


@pytest.fixture
def loaded_backgrounds(background_csv):
    """``load_backgrounds`` output, for the weighting tests."""
    import plant_population_escape as pop

    return pop.load_backgrounds(background_csv)


###############################################################################
# Real-data gating
###############################################################################
REAL_RUN_DIR = REPO_ROOT / "Results" / "JtoJ.2.4_scan" / "plant"
REAL_BACKGROUND_CSV = Path(
    "/home3/oml4h/hugging_face_downloads/PLANT_model/code/examples/backgrounds.csv"
)


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow", action="store_true", default=False,
        help="Also run the opt-in slow tests against the real run directory and "
             "the full 150k-sequence background CSV.",
    )


def pytest_configure(config):
    for marker in (
        "unit: pure, fast, fully synthetic",
        "integration: several functions together, still synthetic",
        "cli: exercises a module's argparse surface / main()",
        "figure: writes a PNG and inspects it",
        "slow: skipped unless --run-slow",
        "requires_real_run: needs Results/JtoJ.2.4_scan/plant",
        "requires_real_backgrounds: needs the PLANT backgrounds.csv",
    ):
        config.addinivalue_line("markers", marker)


def pytest_collection_modifyitems(config, items):
    run_slow = config.getoption("--run-slow")
    skip_slow = pytest.mark.skip(reason="opt-in; pass --run-slow")
    skip_run = pytest.mark.skip(reason=f"missing {REAL_RUN_DIR}")
    skip_bg = pytest.mark.skip(reason=f"missing {REAL_BACKGROUND_CSV}")
    have_run = (REAL_RUN_DIR / "genotype_embeddings.csv").exists()
    have_bg = REAL_BACKGROUND_CSV.exists()

    for item in items:
        if "slow" in item.keywords and not run_slow:
            item.add_marker(skip_slow)
        if "requires_real_run" in item.keywords and not have_run:
            item.add_marker(skip_run)
        if "requires_real_backgrounds" in item.keywords and not have_bg:
            item.add_marker(skip_bg)


@pytest.fixture
def real_run_dir() -> Path:
    return REAL_RUN_DIR


@pytest.fixture
def real_background_csv() -> Path:
    return REAL_BACKGROUND_CSV
