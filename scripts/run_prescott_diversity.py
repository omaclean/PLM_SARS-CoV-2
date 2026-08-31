#!/usr/bin/env python3
"""Run codon mutational accessibility against ESCOTT/PRESCOTT diversity scores.

This is the ESCOTT-flavoured sibling of ``scripts/run_mutational_accessibility.py``.
The two scripts answer the same question -- how well does a per-site amino-acid
preference model, traded off against codon-level mutational accessibility, predict
the diversity actually observed in a lineage panel -- but they differ in where the
per-site preference comes from:

    run_mutational_accessibility.py : plm_prob  <- a protein language model
    run_prescott_diversity.py       : plm_prob  <- ESCOTT (optionally frequency-
                                                  modified into PRESCOTT)

Everything downstream of that one column is deliberately *shared code*: the codon
mutation model, the observed-diversity profiles, the alpha sweep, the metric tables
and every figure come from ``run_mutational_accessibility`` and
``Functions_HuggingFace`` by direct import. Reimplementing any of it would let the
two pipelines drift apart, and the whole point of this script is that its output
tables concatenate cleanly with the PLM run's.

The run is split into two *stages*, not two environments. One conda env (PRESCOTT)
carries both halves -- prody, biotite+mkdssp, freesasa, R and the ``escott`` console
script for stage 1, and torch/esm for ``Functions_HuggingFace`` in stage 2 -- so the
old two-interpreter split is gone. Stage 1 is still invoked as a set of subprocesses
against an absolute interpreter path (``--prescott-python``, default the PRESCOTT
env's own python) because each stage-1 step is independently cacheable, resumable and
runnable by hand, not because the imports conflict.

The contract between the stages is one file format:
``scores/<lineage_key>_<variant>_score_matrix.csv``, byte-compatible with the PLM
run's ``plm_cache/*_plm_probability_profile.csv``.

Evaluation design (differs from the PLM run, on purpose): this is NOT leave-one-out.
For each target lineage the population-frequency input handed to PRESCOTT comes from
the BASAL (parent) lineage panel underneath it in the H3N2 clade topology, and the
evaluation target is the descendant's own observed diversity. G.1 has no basal panel
and is therefore an input-only lineage, never an evaluation target.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Cap CPU thread usage before numpy/torch land, exactly as the PLM driver does --
# these have to be set before the first BLAS import to have any effect.
os.environ.setdefault("OMP_NUM_THREADS", "12")
os.environ.setdefault("MKL_NUM_THREADS", "12")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "12")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "12")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "12")
# prescott.py imports pyplot at module load; headless compute nodes need Agg.
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for _path in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# run_mutational_accessibility guards its CLI behind `if __name__ == "__main__"`,
# so importing it is side-effect free and gives us the whole analysis half for free.
import run_mutational_accessibility as rma  # noqa: E402


# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

# Bumped whenever the *semantics* of a cached table change (not merely its contents).
# Deliberately separate from rma.PANEL_CACHE_VERSION: the two pipelines cache different
# things and must be able to invalidate independently.
PRESCOTT_CACHE_VERSION = 1

# FALLBACK ONLY. scripts/prescott_iav/constants.py is the authority and wins whenever it
# is importable (see parent_map_presets); these literals exist so the driver still parses
# its own CLI if the stage-1 package is absent. Keep them in sync -- they are checked
# against the shared module on every use, not merely hoped to agree.
#
# Clade topology for the H3N2 2024-25 nextstrain nomenclature:
#     G.1 -> ... -> G.1.3.1 -> J -> J.2 -> J.2.4 -> J.2.4.1 (= K)
# K is J.2.4.1 (see Sequences/IAV_lineage_files/K.nt.fa header, which carries the
# clade call `J.2.4.1`, and Sequences/nextclade_id_clade.tsv), so its basal panel is
# J.2.4 rather than J.2_int. The project brief specified J.2_int; --parent-map-preset
# brief_as_stated restores that, and --parent-sensitivity scores both in one table.
DEFAULT_PARENT_MAPS: Dict[str, Dict[str, str]] = {
    "clade_evidence": {
        "J_int": "G.1",
        "J.2_int": "J_int",
        "J.2.4": "J.2_int",
        "K": "J.2.4",
    },
    "brief_as_stated": {
        "J_int": "G.1",
        "J.2_int": "J_int",
        "J.2.4": "J.2_int",
        "K": "J.2_int",
    },
}

# Lineages with no basal panel underneath them. They are still prepared (they are
# somebody's parent) but never scored, because there is nothing to condition on.
INPUT_ONLY_LINEAGES = frozenset({"G.1"})

# parent_map_presets() is called once per CLI parse and several times per run; the drift
# warning belongs on stderr once, not on every call.
_PRESET_DRIFT_REPORTED: List[bool] = []

# The deep, pre-cutoff evolutionary MSA source. Its April-2024 NCBI cutoff precedes
# the 2025/26 GISAID panels by ~2 years, which is what keeps the test data out of the
# ESCOTT alignment. Do NOT fold the GISAID panels in here.
DEFAULT_DEEP_FASTA = Path(
    "/home3/oml4h/my_SC2_finetunes/myflu/full_job4243186/data/"
    "ncbiflu_HA_all_110424_noX_clu99_filt_HA-80.clean_input.fasta"
)

DEFAULT_GUIDE_PATH = REPO_ROOT / "Sequences" / "IAV_lineage_guide.csv"
DEFAULT_STRUCTURE = REPO_ROOT / "Sequences" / "6WXB-assembly1.cif"
DEFAULT_PRESCOTT_PYTHON = Path("/home3/oml4h/miniconda3/envs/PRESCOTT/bin/python3.10")

# Only the real JET2 output PRESCOTT ships, and the inputs it came from. The driver
# runs jet_surrogate.py --validate-only against it once per stage-1 pass so CAVEATS.md
# caveat 1 points at a table that actually exists.
JET_VALIDATION_BASENAME = "jet_surrogate_vs_blat_reference.tsv"

# Stage-1 modules. They are written to run under the PRESCOTT interpreter, so we
# invoke them by path rather than importing them -- but we still try to import the
# pure-python constants module so the two halves cannot disagree about the parent map.
PRESCOTT_IAV_DIR = SCRIPT_DIR / "prescott_iav"
STAGE1_SCRIPTS = {
    "prepare": PRESCOTT_IAV_DIR / "prepare_inputs.py",
    "jet": PRESCOTT_IAV_DIR / "jet_surrogate.py",
    "escott": PRESCOTT_IAV_DIR / "run_escott.py",
}

# Written onto every alpha-sweep row so a concatenated PLM+ESCOTT table stays readable.
INPUT_SCORE_FORMULA = "escott_prob * mut_prob^alpha"


# --------------------------------------------------------------------------------------
# Defensive access to the sibling stage-1 package
# --------------------------------------------------------------------------------------

def load_prescott_iav_constants() -> Optional[object]:
    """Return ``prescott_iav.constants`` if it is importable, else None.

    Soft-fails on purpose. The parent map and lineage tags are duplicated here so the
    driver stays usable while stage 1 is still being written; if the shared constants
    module *is* present it wins, which is what stops the two halves drifting. A module
    that exists but is broken still raises, because that is a real bug, not an absence.
    """
    module_path = PRESCOTT_IAV_DIR / "constants.py"
    if not module_path.exists():
        return None
    try:
        return importlib.import_module("prescott_iav.constants")
    except ImportError as exc:
        raise RuntimeError(
            f"{module_path} exists but could not be imported ({exc}). "
            "Fix the module or delete it; a half-written stage-1 package is worse than none."
        ) from exc


def parent_map_presets() -> Dict[str, Dict[str, str]]:
    """The preset table, taking ``prescott_iav.constants`` as the authority.

    Called from ``build_parser`` (for --parent-map-preset's choices), from
    ``resolve_parent_map`` and from ``sensitivity_edges``, so there is exactly one
    place where the driver could disagree with the half of the pipeline that actually
    writes the frequency files -- and it does not, because the shared module wins
    whenever it is importable.
    """
    shared = load_prescott_iav_constants()
    presets = getattr(shared, "DEFAULT_PARENT_MAPS", None) if shared is not None else None
    if not isinstance(presets, dict) or not presets:
        return DEFAULT_PARENT_MAPS
    if presets != DEFAULT_PARENT_MAPS and not _PRESET_DRIFT_REPORTED:
        # Not fatal -- the shared module still wins, and it is the one prepare_inputs
        # actually builds the frequency files from -- but silence here is how the two
        # halves drift until panel_metadata.parent_lineage names a parent PRESCOTT
        # never conditioned on.
        _PRESET_DRIFT_REPORTED.append(True)
        print(
            "WARNING: scripts/prescott_iav/constants.DEFAULT_PARENT_MAPS differs from this "
            "driver's fallback copy. The shared module wins (it is what stage 1 uses); "
            f"update run_prescott_diversity.DEFAULT_PARENT_MAPS to match.\n"
            f"  shared  : {json.dumps(presets, sort_keys=True)}\n"
            f"  fallback: {json.dumps(DEFAULT_PARENT_MAPS, sort_keys=True)}"
        )
    return presets


def default_trace_top_fraction() -> float:
    """jet_surrogate.py's measured default, read from the shared constants module.

    Never hard-coded here: the driver used to pass 0.30 unconditionally, which
    overrode stage B's tuned 0.90 and left ~14% of HA positions at trace == 0 (an
    identically-zero ESCOTT column, hence a uniform softmax, hence pure noise).
    Reported in the manifest and CAVEATS so a changed value is visible.
    """
    shared = load_prescott_iav_constants()
    value = getattr(shared, "DEFAULT_TRACE_TOP_FRACTION", None) if shared is not None else None
    return float(value) if value is not None else 0.90


# Fallback copies of leakage_check.py's defaults, used ONLY for --help text when the
# stage-1 package is not importable. Every one of these flags defaults to None in the
# driver and is forwarded only when the user sets it, exactly like --trace-top-fraction,
# so leakage_check.py stays the single authority on the numbers themselves.
LEAKAGE_DEFAULT_FALLBACKS: Dict[str, object] = {
    "min_identity": 99.0,
    "max_hamming": 10,
    "min_coverage": 95.0,
    "coverage_basis": "both",
    "max_removed_fraction": 0.25,
    "min_depth_after": 500,
}


def leakage_default(name: str) -> object:
    """leakage_check.py's default for one threshold, for --help text only.

    Deliberately never used to *set* a value: the driver forwards these flags only
    when the user passed them, so stage 1's own default always wins at runtime. This
    exists so that ``--help`` quotes the number that will actually be used rather
    than a copy that can rot.
    """
    try:
        sys.path.insert(0, str(PRESCOTT_IAV_DIR.parent))
        from prescott_iav import leakage_check as _lc  # noqa: PLC0415
        mapping = {
            "min_identity": _lc.DEFAULT_MIN_IDENTITY,
            "max_hamming": _lc.DEFAULT_MAX_HAMMING,
            "min_coverage": _lc.DEFAULT_MIN_COVERAGE,
            "coverage_basis": "both",
            "max_removed_fraction": _lc.DEFAULT_MAX_REMOVED_FRACTION,
            "min_depth_after": _lc.DEFAULT_MIN_DEPTH_AFTER,
        }
        return mapping.get(name, LEAKAGE_DEFAULT_FALLBACKS.get(name))
    except Exception:  # pragma: no cover - help text must never crash the parser
        return LEAKAGE_DEFAULT_FALLBACKS.get(name)


def require_stage1_script(kind: str) -> Path:
    """Resolve one stage-1 script, with an actionable error instead of a traceback."""
    path = STAGE1_SCRIPTS[kind]
    if not path.exists():
        raise RuntimeError(
            f"Stage-1 script missing: {path}\n"
            f"  This driver only consumes scores/*_score_matrix.csv; the ESCOTT/PRESCOTT side "
            f"lives in {PRESCOTT_IAV_DIR}/ (prepare_inputs.py, jet_surrogate.py, run_escott.py).\n"
            f"  Either create it, or run with --no-auto-prepare once the score matrices already "
            f"exist under --scores-dir."
        )
    return path


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------

def variant_token(label: str) -> str:
    """Alphanumeric-only lineage token used inside variant names.

    Variant names become the ``model`` column, which becomes a filename, and
    ``os.path.splitext`` inside prescott.py truncates at the last dot -- so dots and
    underscores are stripped entirely: G.1 -> G1, J_int -> Jint, J.2_int -> J2int,
    J.2.4 -> J24, K -> K.
    """
    return re.sub(r"[^A-Za-z0-9]", "", str(label))


def safe_key(label: str) -> str:
    """Filesystem key for a lineage, identical to what stage 1 uses (rma._safe_label)."""
    from Functions_HuggingFace import _safe_label

    return _safe_label(str(label))


def file_md5(path: Path) -> Optional[str]:
    if not Path(path).exists():
        return None
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_float_grid(text: str) -> List[float]:
    return [float(chunk.strip()) for chunk in str(text).split(",") if chunk.strip()]


def parse_int_grid(text: str) -> List[int]:
    return [int(float(chunk.strip())) for chunk in str(text).split(",") if chunk.strip()]


def _iqr(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return float("nan")
    return float(np.percentile(finite, 75) - np.percentile(finite, 25))


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_prescott_diversity.py",
        description=(
            "Run codon mutational accessibility against ESCOTT/PRESCOTT diversity scores "
            "for influenza A H3N2 HA. Mirrors run_mutational_accessibility.py, with ESCOTT "
            "in place of a protein language model."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- analysis targets: identical semantics to the PLM driver -----------------------
    targets = parser.add_argument_group("analysis targets")
    targets.add_argument(
        "--analysis-mode",
        choices=("SINGLE_FASTA", "MONTHLY_GUIDE"),
        help="MONTHLY_GUIDE only. The choice list keeps the PLM driver's spelling, but "
             "SINGLE_FASTA is rejected at parse time: this pipeline's unit of work is a "
             "(child, basal parent) lineage PAIR, and a lone diversity FASTA carries no "
             "parent panel to build the PRESCOTT frequency prior from. "
             "Required unless --regen-figures-only.",
    )
    targets.add_argument("--guide-path", type=Path, default=DEFAULT_GUIDE_PATH,
                         help="CSV guide with columns month|label, fasta|path, reference.")
    targets.add_argument("--reference-fasta", type=Path, default=None,
                         help="Focal nucleotide CDS FASTA (SINGLE_FASTA mode, or guide fallback).")
    targets.add_argument("--diversity-fasta", type=Path, default=None,
                         help="Diversity FASTA for SINGLE_FASTA mode.")
    targets.add_argument("--label", default="population",
                         help="Label used for SINGLE_FASTA mode.")
    targets.add_argument("--mutation-model", choices=("SC2", "H1N1", "H3N2"), default="H3N2",
                         help="Nucleotide mutation model behind the codon accessibility matrix.")
    targets.add_argument("--output-dir", type=Path, required=True,
                         help="Directory for every table, plot, score matrix and the manifest.")
    targets.add_argument(
        "--expect-protein-diversity", action="store_true",
        help="Treat diversity FASTAs as protein alignments. The GISAID H3N2 panels ARE "
             "protein, so pass this.",
    )
    targets.add_argument("--filter-fixed-mutations", action=argparse.BooleanOptionalAction, default=True,
                         help="Exclude mutations already at frequency 1.0 in the target panel.")
    targets.add_argument("--filter-singleton-mutations", action=argparse.BooleanOptionalAction, default=False,
                         help="Zero out or skip mutations seen fewer than --min-obs-count times.")
    targets.add_argument("--skip-low-count-sites", action=argparse.BooleanOptionalAction, default=False,
                         help="With singleton filtering, drop those rows instead of zeroing obs_freq.")
    targets.add_argument("--min-obs-count", type=int, default=2,
                         help="Minimum retained count when --filter-singleton-mutations is on.")

    # --- score source: replaces the PLM group ------------------------------------------
    scores = parser.add_argument_group("ESCOTT/PRESCOTT score source")
    scores.add_argument("--scores-dir", type=Path, default=None,
                        help="Where <lineage>_<variant>_score_matrix.csv live. Default <output-dir>/scores.")
    scores.add_argument("--inputs-dir", type=Path, default=None,
                        help="Stage-1 input tree (structure/msa/query/frequency/jet). Default <output-dir>/inputs.")
    scores.add_argument("--escott-workdir", type=Path, default=None,
                        help="Per-lineage escott working directories. Default <output-dir>/escott.")
    scores.add_argument("--prescott-ref-dir", type=Path, default=None,
                        help="Reference prescott.py runs used only for the 2-dp parity check. "
                             "Default <output-dir>/prescott_ref.")
    scores.add_argument("--score-variant", action="append", default=None, dest="score_variants",
                        help="Restrict the run to these variants. Repeatable. Accepts EITHER the "
                             "model tag printed as 'Models:' and used in the output tables "
                             "(PRESCOTT_eq2_c0p50_k1) OR the stage-1 variant name in the `variant` "
                             "column of scores/score_variants.tsv, which carries the parent suffix "
                             "(PRESCOTT_eq2_c0p50_k1_parentG1). Default: all.")
    scores.add_argument("--auto-prepare", action=argparse.BooleanOptionalAction, default=True,
                        help="Run stage 1 (prepare_inputs / jet_surrogate / run_escott) automatically "
                             "when score matrices are missing.")
    scores.add_argument("--prescott-python", type=Path, default=DEFAULT_PRESCOTT_PYTHON,
                        help="Absolute interpreter for stage 1. Never 'conda activate'.")
    scores.add_argument("--deep-fasta", type=Path, default=DEFAULT_DEEP_FASTA,
                        help="Deep pre-cutoff NCBI HA protein set used to build the ESCOTT MSA.")
    scores.add_argument("--escott-temperature", type=float, default=1.0,
                        help="Softmax temperature applied to the raw ESCOTT matrix.")
    scores.add_argument("--escott-temperature-mode", choices=("fixed", "match-plm"), default="fixed",
                        help="'match-plm' rescales T so sd(log score) matches a PLM run's sd(log plm_prob).")
    scores.add_argument("--plm-reference-table", type=Path, default=None,
                        help="A completed PLM run's tables/combined_long_table.csv. Required for "
                             "--escott-temperature-mode match-plm; otherwise used only for the scale report.")
    scores.add_argument("--force-recompute-scores", action="store_true",
                        help="Regenerate score matrices even when cached files exist.")

    # --- parent (basal) lineage design --------------------------------------------------
    parents = parser.add_argument_group("basal/parent lineage design")
    parents.add_argument("--parent-map", default=None,
                         help="Override individual edges: 'child=parent,child=parent'.")
    parents.add_argument("--parent-map-preset", choices=tuple(parent_map_presets()), default="clade_evidence",
                         help="clade_evidence: K<-J.2.4 (on-disk evidence). brief_as_stated: K<-J.2_int.")
    parents.add_argument("--parent-sensitivity", action=argparse.BooleanOptionalAction, default=True,
                         help="Also score the lineages where the two presets disagree under the OTHER "
                              "preset's parent. Forwarded as --sensitivity-parent-map to "
                              "prepare_inputs.py (which writes a second frequency file per edge) and "
                              "to run_escott.py (which emits a second _parent<TOK> score matrix per "
                              "grid point), so both parents appear as separate model rows in one "
                              "table instead of needing two runs.")
    parents.add_argument("--parent-min-count", type=int, default=1,
                         help="Minimum parent-panel count for a mutant to get a frequency at all.")
    parents.add_argument("--parent-min-depth", type=int, default=50,
                         help="Minimum parent-panel column depth before a position is used.")
    parents.add_argument("--drop-parent-reversions", action=argparse.BooleanOptionalAction, default=True,
                         help="Drop, regardless of frequency, any mutant whose mutant residue is the "
                              "PARENT reference's residue at that site. This -- not --parent-freq-max "
                              "-- is what removes the exact ancestral reversions; K's N160S sits at "
                              "0.932 and K176I at 0.897, both under the 0.95 threshold. Forwarded to "
                              "prepare_inputs.py and recorded in the manifest and CAVEATS.")
    parents.add_argument("--parent-freq-max", type=float, default=0.95,
                         help="Drop mutants at/above this parent frequency. This is the reverse-"
                              "substitution guard: at lineage-defining sites the ancestral residue is "
                              "near-fixed in the parent and would otherwise read as 'highly tolerated'.")
    parents.add_argument("--frequency-cutoff-mode", choices=("depth_scaled", "fixed"), default="depth_scaled",
                         help="depth_scaled: Fc = log10(k / median parent depth), which makes the v2 "
                              "penalty c*log_N(count) and therefore comparable across 229- and "
                              "27452-sequence panels.")
    parents.add_argument("--frequency-cutoff-k", default="1",
                         help="Comma-separated k values for the depth-scaled cutoff.")
    parents.add_argument("--frequency-cutoff", type=float, default=-4.0,
                         help="Fixed log10 frequency cutoff (only with --frequency-cutoff-mode fixed).")
    parents.add_argument("--coefficient-grid", default="0.25,0.5,1.0",
                         help="PRESCOTT penalty coefficients c.")
    parents.add_argument("--equation-grid", default="2",
                         help="PRESCOTT equations. 1, 2, 3, 5 are usable; 4 is broken upstream "
                              "(prescott.py has an unconditional sys.exit there).")

    # --- jet surrogate pass-through -----------------------------------------------------
    jet = parser.add_argument_group("JET2 surrogate (stage 1 pass-through)")
    jet.add_argument("--structure", type=Path, default=DEFAULT_STRUCTURE,
                     help="Structure used for cv/pc/DSSP. 6WXB is the 1968 HA trimer (485/566 coverage).")
    jet.add_argument("--structure-role", choices=("primary", "extra"), default="primary",
                     help="Which structure prepare_inputs built to score against: 'primary' is "
                          "--structure (6WXB, 485/566 coverage), 'extra' is the contemporary "
                          "full-coverage J.2.4.1 model. The caveats call for reporting both.")
    jet.add_argument("--weight-mode", choices=("structural", "tjet"), default="structural")
    jet.add_argument("--pc-mode", choices=("interface_propensity", "constant", "zero"),
                     default="interface_propensity")
    jet.add_argument("--sasa-context", choices=("trimer", "monomer"), default="trimer",
                     help="SASA/circular-variance environment. jet_surrogate.py expresses this as "
                          "--context-pdb, so this flag selects the trimer or the lone chain.")
    jet.add_argument("--cv-radius", type=float, default=7.0, help="Circular-variance cutoff radius, Angstrom.")
    jet.add_argument("--max-coil-length", type=int, default=5)
    jet.add_argument("--trace-definition", choices=("bootstrap", "direct"), default="bootstrap")
    jet.add_argument("--trace-bootstraps", type=int, default=50)
    # default=None on purpose. jet_surrogate.py is the AUTHORITY on this number
    # (constants.DEFAULT_TRACE_TOP_FRACTION = 0.90, measured); the driver used to pass
    # 0.30 unconditionally and silently override it. pred.R:487 multiplies each ESCOTT
    # column by trace[i], so a zero-trace site is an identically-zero column, a uniform
    # softmax and pure noise -- 0.30 leaves 77/566 HA positions there, 0.90 leaves 18.
    # Left unset the flag is not forwarded at all, so stage B's tested default wins.
    jet.add_argument("--trace-top-fraction", type=float, default=None,
                     help=f"Only forwarded to jet_surrogate.py when you set it. Unset means "
                          f"jet_surrogate's own measured default ({default_trace_top_fraction()}).")
    jet.add_argument("--max-zero-trace-fraction", type=float, default=None,
                     help="Forwarded to jet_surrogate.py --max-zero-trace-fraction, which REFUSES "
                          "to write a jet table with more than this share of trace == 0 columns. "
                          "Unset means jet_surrogate's own default (0.10); 1.0 disables the guard.")
    jet.add_argument("--jet-validation", action=argparse.BooleanOptionalAction, default=True,
                     help="Run jet_surrogate.py --validate-only once per stage-1 pass and write "
                          f"tables/diagnostics/{JET_VALIDATION_BASENAME} -- the surrogate-vs-real-JET2 "
                          "comparison on the shipped BLAT inputs that CAVEATS.md caveat 1 cites.")
    jet.add_argument("--seed", type=int, default=20260805)

    # --- leakage detection / purge (stage 1 pass-through) --------------------------------
    # The deep evolutionary set has an 11 Apr 2024 cutoff and the evaluation panels are
    # 2025/26 GISAID; the two use DIFFERENT accession namespaces (protein vs nucleotide),
    # so identifier matching cannot detect an overlap and hashing cannot either (the deep
    # set keeps the HA signal peptide, the panels do not). Alignment is the only
    # instrument, and a report is not enough: near neighbours of the evaluation target
    # are REMOVED from the deep set before ESCOTT sees it. See
    # scripts/prescott_iav/leakage_check.py.
    leak = parser.add_argument_group("data-leakage detection and purge (stage 1 pass-through)")
    leak.add_argument("--leakage-check", action=argparse.BooleanOptionalAction, default=True,
                      help="Run the BLAST leakage audit in stage 1 (checks A/B/C). Default on.")
    leak.add_argument("--purge-leakage", action=argparse.BooleanOptionalAction, default=True,
                      help="REMOVE deep-set sequences that are near neighbours of each evaluation "
                           "target's own panel, per target, before jet_surrogate.py or "
                           "run_escott.py see the alignment. Default on. Turning it off leaves any "
                           "leak in place and every reported correlation unaudited.")
    leak.add_argument("--fail-on-leakage", action="store_true",
                      help="Make stage 1 exit non-zero on residual leakage, so a slurm job stops "
                           "instead of producing an inflated number.")
    leak.add_argument("--leakage-min-identity", default=None,
                      help=f"%%AA identity at/above which a deep sequence is dropped (stage-1 "
                           f"default {leakage_default('min_identity')}; 'none' disables THIS rule "
                           f"only). COMBINED WITH --leakage-max-hamming BY *OR*: a sequence goes if "
                           f"EITHER fires. They are not equivalent -- on a ~550 aa HA "
                           f"{leakage_default('max_hamming')} mismatches is ~98.2%% identity, so at "
                           f"the defaults the HAMMING rule is the stricter one and is what actually "
                           f"governs the purge. Only forwarded when you set it.")
    leak.add_argument("--leakage-max-hamming", default=None,
                      help=f"AA mismatches at/below which a deep sequence is dropped (stage-1 "
                           f"default {leakage_default('max_hamming')}; 'none' disables THIS rule "
                           f"only). Hamming is derived from the BLAST alignment as "
                           f"min(qlen,slen)-nident, so gaps and unaligned overhang count. Only "
                           f"forwarded when you set it.")
    leak.add_argument("--leakage-min-coverage", type=float, default=None,
                      help=f"Coverage gate both rules must clear (stage-1 default "
                           f"{leakage_default('min_coverage')}%%), so a short high-identity local "
                           f"hit is never mistaken for a full-length duplicate.")
    leak.add_argument("--leakage-coverage-basis",
                      choices=("both", "shorter", "query", "subject"), default=None,
                      help="Coverage denominator; stage-1 default 'both'.")
    leak.add_argument("--leakage-max-removed-fraction", type=float, default=None,
                      help="Warn loudly above this removed fraction. MSA depth drives GEMME "
                           "quality, so over-removal is its own failure mode.")
    leak.add_argument("--leakage-min-depth-after", type=int, default=None,
                      help="Hard floor on post-purge MSA depth; stage 1 refuses below it.")
    leak.add_argument("--leakage-threads", type=int, default=None,
                      help="blastp -num_threads for the leakage stage.")
    leak.add_argument("--blast-task", choices=("blastp", "blastp-fast"), default=None,
                      help="blastp-fast (stage-1 default) is ~3x faster and cannot miss a >=95%% "
                           "identity hit.")

    # --- sweep / metrics: identical to the PLM driver -----------------------------------
    sweep = parser.add_argument_group("alpha sweep and metrics")
    sweep.add_argument("--alpha-start", type=float, default=-1.0)
    sweep.add_argument("--alpha-stop", type=float, default=1.0)
    sweep.add_argument("--alpha-step", type=float, default=0.1)
    sweep.add_argument("--alpha-parallel", action=argparse.BooleanOptionalAction, default=True)
    sweep.add_argument("--alpha-sweep-min-grid", type=int, default=8)
    sweep.add_argument("--alpha-sweep-max-workers", type=int, default=None)
    sweep.add_argument("--scatter-alphas", default="-1,0,1")
    sweep.add_argument("--scatter-max-points", type=int, default=200000)
    sweep.add_argument("--mutation-baseline-x", type=float, default=-2.0)
    sweep.add_argument("--diagnostic-exports", action=argparse.BooleanOptionalAction, default=False)
    sweep.add_argument("--alignment-verify-max-cols", type=int, default=100)
    sweep.add_argument("--rolling-identity-window", type=int, default=30)
    sweep.add_argument("--observed-mutation-fasta", type=Path, default=None)
    sweep.add_argument("--observed-mutation-sequence-id", default=None)
    sweep.add_argument("--observed-mutation-selection", default="last")

    # --- run control ---------------------------------------------------------------------
    control = parser.add_argument_group("run control")
    control.add_argument("--test-mode", action=argparse.BooleanOptionalAction, default=False,
                         help="Smoke-test pass. Exactly like the PLM driver's: it limits how much "
                              "data is READ (--test-max-targets guide rows, --test-max-records "
                              "diversity records each) and NOTHING else. It does not touch a single "
                              "modelling parameter -- --trace-definition, --trace-bootstraps, "
                              "--trace-top-fraction and the alpha grid are exactly what you asked "
                              "for, so a test run exercises the production scoring path. For a fast "
                              "pass add --trace-definition direct --alpha-step 0.5 yourself.")
    control.add_argument("--test-max-targets", type=int, default=1,
                         help="Guide targets in test mode. Raised automatically if the leading rows "
                              "are all input-only lineages.")
    control.add_argument("--test-max-records", type=int, default=0,
                         help="Diversity records per target in test mode. 0 means NO truncation, which "
                              "deliberately overrides the PLM driver's default of 5 -- with 5 sequences "
                              "the observed-frequency profile is meaningless and every metric is noise.")
    control.add_argument("--dry-run", action="store_true",
                         help="Resolve targets and build the lineage cache (observed diversity plus "
                              "codon accessibility), report it, then stop before any ESCOTT scoring. "
                              "Verifies the whole input half without needing stage 1.")
    control.add_argument("--regen-figures-only", action=argparse.BooleanOptionalAction, default=False,
                         help="Re-plot from existing tables in --output-dir. Because the schemas match "
                              "the PLM run's, run_mutational_accessibility.py --regen-figures-only also "
                              "works against this output directory.")
    control.add_argument("--prepare-args", default="", help="Extra args appended to prepare_inputs.py.")
    control.add_argument("--jet-args", default="", help="Extra args appended to jet_surrogate.py.")
    control.add_argument("--escott-args", default="", help="Extra args appended to run_escott.py.")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.regen_figures_only:
        if args.output_dir is None:
            raise ValueError("--output-dir is required when --regen-figures-only is used")
        return

    if args.analysis_mode is None:
        raise ValueError("--analysis-mode is required unless --regen-figures-only is used")
    if args.mutation_model is None:
        raise ValueError("--mutation-model is required unless --regen-figures-only is used")

    if args.analysis_mode == "SINGLE_FASTA":
        # Rejected here rather than 1000 lines later. resolve_targets would yield one
        # target labelled --label ('population'), which is never a key of the parent
        # map, so the run died with "No basal lineage defined for ['population']";
        # and even with --parent-map population=X, run_stage1 builds prepare_inputs
        # from --guide-path and could not find either panel. The design is not a
        # missing feature, it is a category error: PRESCOTT needs a *parent panel* to
        # build the population-frequency prior from, so the unit of work here is a
        # (child, basal parent) PAIR, not a FASTA.
        raise ValueError(
            "--analysis-mode SINGLE_FASTA is not supported by this pipeline.\n"
            "  ESCOTT scores a sequence, but PRESCOTT conditions on the population "
            "frequency of each mutant in the BASAL (parent) lineage panel, so every "
            "target needs a parent panel as well as its own diversity FASTA.\n"
            "  Use --analysis-mode MONTHLY_GUIDE with a guide CSV that has one row per "
            "lineage (label,fasta,reference) and name the topology with --parent-map "
            "'child=parent,...'. A two-row guide plus --parent-map child=parent is the "
            "single-FASTA equivalent.\n"
            f"  The shipped guide is {DEFAULT_GUIDE_PATH}."
        )
    if args.analysis_mode == "MONTHLY_GUIDE":
        if args.guide_path is None:
            raise ValueError("--guide-path is required for MONTHLY_GUIDE mode")
        if not Path(args.guide_path).exists():
            raise FileNotFoundError(f"Guide file not found: {args.guide_path}")

    # `nan <= 0` and `nan < 0` are both False, so every one of these guards used to be
    # transparent to `--alpha-step nan` / `--escott-temperature nan`, which argparse
    # accepts because `float("nan")` parses. A NaN temperature poisons the entire score
    # matrix and only resurfaces as an AssertionError from inside stage 1; `inf` is worse
    # still because it does not raise at all -- it returns an exactly uniform 1/20 matrix,
    # i.e. every site dead, and the run completes with meaningless metrics.
    if not np.isfinite(args.alpha_step) or args.alpha_step <= 0:
        raise ValueError("--alpha-step must be > 0 and finite")
    if not np.isfinite(args.escott_temperature) or args.escott_temperature <= 0:
        raise ValueError("--escott-temperature must be > 0 and finite")
    if args.escott_temperature_mode == "match-plm" and args.plm_reference_table is None:
        raise ValueError("--escott-temperature-mode match-plm requires --plm-reference-table")

    equations = parse_int_grid(args.equation_grid)
    if 4 in equations:
        # prescott.py's equation-4 branch is an unconditional sys.exit(-1); offering it
        # would only produce a confusing mid-run death inside a subprocess.
        raise ValueError("PRESCOTT equation 4 is not implemented upstream; choose from 1, 2, 3, 5")
    if any(eq not in (1, 2, 3, 5) for eq in equations):
        raise ValueError(f"--equation-grid must be a subset of 1,2,3,5 (got {equations})")
    coefficients = parse_float_grid(args.coefficient_grid)
    if any(not np.isfinite(c) or c < 0 for c in coefficients):
        raise ValueError("--coefficient-grid values must be >= 0 and finite")


def apply_prescott_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Fill in derived paths and the arguments rma's helpers expect to exist."""
    args = rma.apply_arg_defaults(args)

    output_dir = Path(args.output_dir)
    if args.scores_dir is None:
        args.scores_dir = output_dir / "scores"
    if args.inputs_dir is None:
        args.inputs_dir = output_dir / "inputs"
    if args.escott_workdir is None:
        args.escott_workdir = output_dir / "escott"
    if args.prescott_ref_dir is None:
        args.prescott_ref_dir = output_dir / "prescott_ref"

    # rma.build_lineage_cache reads these PLM-only knobs; they are meaningless here
    # (the ESCOTT frame is always the full 566-aa HA0) but must exist.
    args.plm_max_aa_length = None
    args.plm_max_nt_length = None
    args.use_global_plm_reference = False

    # rma.load_diversity_records truncates to args.test_max_records whenever test_mode
    # is on. 0 is our "do not truncate" sentinel; translate it to an unreachable bound
    # so the observed-diversity profile stays real in the smoke test.
    if args.test_mode and int(args.test_max_records) <= 0:
        args.test_max_records = 10 ** 9

    # NOTHING else keys off test_mode. It used to force trace_definition='direct',
    # which silently discarded an explicit --trace-definition bootstrap and, worse,
    # meant the smoke test never executed the production trace path at all -- that is
    # exactly why a wrong --trace-top-fraction survived a full end-to-end test run.
    # The PLM driver's --test-mode limits how much data is read and nothing else; so
    # does this one now. Speed knobs stay in the user's hands (--trace-definition
    # direct, --trace-bootstraps, --alpha-step).

    return args


# --------------------------------------------------------------------------------------
# Parent map resolution
# --------------------------------------------------------------------------------------

def resolve_parent_map(args: argparse.Namespace) -> Dict[str, str]:
    """Resolve preset + explicit overrides into one child -> parent dict."""
    preset_source = parent_map_presets()

    if args.parent_map_preset not in preset_source:
        raise ValueError(f"Unknown --parent-map-preset {args.parent_map_preset!r}")
    parent_map = dict(preset_source[args.parent_map_preset])

    if args.parent_map:
        for chunk in str(args.parent_map).split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "=" not in chunk:
                raise ValueError(f"--parent-map entries must look like child=parent (got {chunk!r})")
            child, parent = (part.strip() for part in chunk.split("=", 1))
            if not child or not parent:
                raise ValueError(f"--parent-map entry {chunk!r} has an empty side")
            parent_map[child] = parent

    # A cycle would make stage 1 loop or produce a nonsensical frequency file.
    for child in parent_map:
        seen = {child}
        cursor = parent_map.get(child)
        while cursor is not None:
            if cursor in seen:
                raise ValueError(f"--parent-map contains a cycle through {cursor!r}")
            seen.add(cursor)
            cursor = parent_map.get(cursor)
    return parent_map


def input_only_lineages() -> frozenset:
    shared = load_prescott_iav_constants()
    value = getattr(shared, "INPUT_ONLY_LINEAGES", None) if shared is not None else None
    return frozenset(value) if value else INPUT_ONLY_LINEAGES


def sensitivity_edges(args: argparse.Namespace, parent_map: Dict[str, str]) -> Dict[str, str]:
    """Alternative parent edges to score alongside the primary ones.

    Only lineages where the two presets disagree are worth a second pass -- everywhere
    else the sensitivity variant would be a byte-identical duplicate.
    """
    if not args.parent_sensitivity:
        return {}
    presets = parent_map_presets()
    others = [name for name in presets if name != args.parent_map_preset]
    if not others:
        return {}
    other = presets[others[0]]
    return {
        child: other[child]
        for child in parent_map
        if child in other and other[child] != parent_map[child]
    }


def effective_sensitivity_edges(
    args: argparse.Namespace,
    parent_map: Dict[str, str],
    evaluable: Sequence[str],
) -> Dict[str, str]:
    """The sensitivity edges this run can actually act on.

    ``sensitivity_edges`` answers "where do the presets disagree"; that set is a
    property of the presets, not of the run. Only the edges whose CHILD is being
    evaluated produce a model row, and reporting the others as 'requested but missing'
    would make every J_int-only smoke run look like a failed sensitivity analysis.
    """
    return {
        child: parent for child, parent in sensitivity_edges(args, parent_map).items()
        if child in set(evaluable)
    }


def sensitivity_edge_spec(edges: Dict[str, str]) -> str:
    """``{"K": "J.2_int"}`` -> ``"K=J.2_int"``, the wire format both stage-1 scripts take.

    ``prescott_iav.constants.parse_edge_spec`` is the reader; keeping the writer here
    (rather than in each caller) is what stops prepare_inputs being told to build
    ``K_parentJ2int_frequency.txt`` while run_escott looks for a different edge.
    """
    return ",".join(f"{child}={parent}" for child, parent in sorted(edges.items()))


# --------------------------------------------------------------------------------------
# Score variants
# --------------------------------------------------------------------------------------

def stage1_variant_name(equation: int, coefficient: float, k_value: int, parent: str) -> str:
    """Reproduce run_escott.build_variant_name: PRESCOTT_eq2_c0p50_k1_parentJ24.

    Only used to predict filenames before stage 1 has ever run; once
    scores/score_variants.tsv exists it is the authority.
    """
    return (
        f"PRESCOTT_eq{int(equation)}_c{float(coefficient):.2f}"
        f"_k{int(k_value)}_parent{variant_token(parent)}"
    ).replace(".", "p")


def canonical_model_tag(
    equation: object,
    coefficient: object,
    k_value: object,
    parent: object,
    resolved_parent: object,
) -> str:
    """Group stage-1 variants into the *models* the pooled tables should compare.

    Stage 1 always folds the parent lineage into its variant name, which is right for it
    -- one matrix per (lineage, parent, grid point) -- but wrong as a model identity
    here. Each lineage has a different parent under the resolved map, so keeping the
    parent in the model name would split one PRESCOTT grid point into four
    single-lineage models and make ``alpha_sweep_fit_metrics.tsv`` average one lineage
    per row while ESCOTT averages four. So a grid point scored under the parent the map
    actually specifies gets a parent-free model tag, and only the deliberate
    *sensitivity* rows (a lineage scored under the other preset's parent) keep the
    suffix -- those must stay separate, since separating them is the entire point.
    """
    if equation is None or (isinstance(equation, float) and np.isnan(equation)):
        return "ESCOTT"
    base = (
        f"PRESCOTT_eq{int(equation)}_c{float(coefficient):.2f}_k{int(k_value)}"
    ).replace(".", "p")
    if resolved_parent is not None and str(parent) != str(resolved_parent):
        return f"{base}_parent{variant_token(parent)}"
    return base


def expected_variant_plan(
    args: argparse.Namespace,
    parent_map: Dict[str, str],
    evaluable: Sequence[str],
) -> List[Dict[str, object]]:
    """The (source variant, lineage) grid this run intends to score.

    This is the REQUESTED design and it is derived from the CLI alone -- the grids, the
    parent map and the sensitivity edges. It is the authority: ``scores/score_variants.tsv``
    supplies the exact filenames stage 1 chose for the combinations it has already built,
    but it can neither add a combination the CLI did not ask for nor suppress one it did.
    (It used to do both: the cached table replaced this plan wholesale, so on any rerun
    into an existing --output-dir --coefficient-grid / --equation-grid /
    --frequency-cutoff-k / --parent-map were silently ignored while run_manifest.json
    still recorded the requested values. The manifest lied about the run.)
    """
    plan: List[Dict[str, object]] = []
    for lineage in evaluable:
        plan.append({
            "source_variant": "ESCOTT",
            "lineage": lineage,
            "lineage_key": safe_key(lineage),
            "parent_lineage": None,
            "equation": None,
            "coefficient": None,
            "frequency_cutoff_k": None,
            "score_matrix_path": None,
        })

    alternates = sensitivity_edges(args, parent_map)
    for equation in parse_int_grid(args.equation_grid):
        for coefficient in parse_float_grid(args.coefficient_grid):
            for k_value in parse_int_grid(args.frequency_cutoff_k):
                edges = [(lineage, parent_map[lineage]) for lineage in evaluable]
                edges += [(lineage, parent) for lineage, parent in alternates.items()
                          if lineage in evaluable]
                for lineage, parent in edges:
                    plan.append({
                        "source_variant": stage1_variant_name(equation, coefficient, k_value, parent),
                        "lineage": lineage,
                        "lineage_key": safe_key(lineage),
                        "parent_lineage": parent,
                        "equation": equation,
                        "coefficient": coefficient,
                        "frequency_cutoff_k": k_value,
                        "score_matrix_path": None,
                    })
    return plan


def load_score_variants_table(scores_dir: Path) -> pd.DataFrame:
    """Read stage 1's scores/score_variants.tsv, or return an empty frame."""
    path = Path(scores_dir) / "score_variants.tsv"
    if not path.exists():
        return pd.DataFrame()
    table = pd.read_csv(path, sep="\t")
    missing = {"variant", "lineage"}.difference(table.columns)
    if missing:
        raise RuntimeError(
            f"{path} exists but lacks the required column(s) {sorted(missing)}; "
            "stage 1 wrote something this driver cannot consume."
        )
    return table


def variant_plan_from_table(
    variants_table: pd.DataFrame,
    evaluable: Sequence[str],
) -> List[Dict[str, object]]:
    """Turn stage 1's score_variants.tsv into this driver's plan rows."""
    keep = set(evaluable)
    plan: List[Dict[str, object]] = []
    for _, row in variants_table.iterrows():
        lineage = str(row["lineage"])
        if lineage not in keep:
            continue
        plan.append({
            "source_variant": str(row["variant"]),
            "lineage": lineage,
            "lineage_key": str(row.get("lineage_key") or safe_key(lineage)),
            "parent_lineage": None if pd.isna(row.get("parent_lineage")) else row.get("parent_lineage"),
            "equation": None if pd.isna(row.get("equation")) else row.get("equation"),
            "coefficient": None if pd.isna(row.get("coefficient")) else row.get("coefficient"),
            "frequency_cutoff_k": None if pd.isna(row.get("frequency_cutoff_k")) else row.get("frequency_cutoff_k"),
            "score_matrix_path": None if pd.isna(row.get("score_matrix_path")) else str(row.get("score_matrix_path")),
        })
    return plan


def _optional_number(value: object) -> Optional[float]:
    """None / NaN / '' -> None; anything numeric -> float. TSV round-trips are lossy."""
    if value is None:
        return None
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return None
    except (TypeError, ValueError):
        pass
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(number) else number


def _normalised_label(value: object) -> Optional[str]:
    """None / NaN / '' -> None, anything else -> its stripped text.

    Used wherever a lineage label makes a round trip through a TSV, where "no parent"
    comes back as NaN rather than None.
    """
    if value is None:
        return None
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or None


def plan_entry_key(entry: Dict[str, object]) -> Tuple:
    """The design identity of one planned combination, comparable across a TSV round-trip.

    Two entries with the same key describe the same model on the same lineage under the
    same parent, whatever stage 1 happened to call the file. This is what lets the
    cached table be *matched against* the requested design rather than replace it.
    """
    lineage = str(entry.get("lineage"))
    equation = _optional_number(entry.get("equation"))
    if equation is None:
        return (lineage, "ESCOTT")
    coefficient = _optional_number(entry.get("coefficient")) or 0.0
    k_value = _optional_number(entry.get("frequency_cutoff_k")) or 0.0
    parent_text = _normalised_label(entry.get("parent_lineage")) or ""
    return (lineage, int(equation), round(float(coefficient), 6), int(k_value), parent_text)


def describe_plan_entry(entry: Dict[str, object]) -> str:
    """Human-readable name for one requested combination, for error messages."""
    equation = _optional_number(entry.get("equation"))
    if equation is None:
        return f"ESCOTT / {entry.get('lineage')}"
    return (
        f"eq{int(equation)} c={_optional_number(entry.get('coefficient'))} "
        f"k={int(_optional_number(entry.get('frequency_cutoff_k')) or 0)} "
        f"parent={entry.get('parent_lineage')} / {entry.get('lineage')}"
    )


def score_matrix_path(
    scores_dir: Path,
    lineage_key: str,
    source_variant: str,
    recorded: Optional[str] = None,
) -> Path:
    """Where one (lineage, source variant) matrix lives.

    Prefers the path stage 1 recorded, so a future change to its layout does not need a
    matching change here -- but only while that path still exists. score_variants.tsv
    stores ABSOLUTE paths, so a copied or moved output tree would otherwise keep reading
    matrices out of the directory it was copied from, and a rerun would appear to be
    fully cached while scoring somebody else's files.
    """
    conventional = Path(scores_dir) / f"{lineage_key}_{source_variant}_score_matrix.csv"
    if conventional.exists():
        return conventional
    if recorded and Path(recorded).exists():
        return Path(recorded)
    return conventional


def reconcile_variant_plan(
    requested: Sequence[Dict[str, object]],
    variants_table: pd.DataFrame,
    scores_dir: Path,
    evaluable: Sequence[str],
) -> Tuple[List[Dict[str, object]], List[str], List[str]]:
    """Match the cached ``score_variants.tsv`` against the design the CLI asked for.

    Returns ``(plan, missing, ignored)``:

    * ``plan`` -- one entry per REQUESTED combination, carrying stage 1's recorded
      variant name and matrix path wherever stage 1 has already built it and the
      predicted name otherwise. Never longer or shorter than ``requested``.
    * ``missing`` -- the requested combinations with no cached row, or with a cached
      row whose matrix is not on disk. Non-empty means stage 1 must run.
    * ``ignored`` -- cached combinations the CLI did *not* ask for. Reported, not
      silently analysed: shrinking --coefficient-grid used to keep analysing the
      dropped coefficients because ``write_variants_table`` merges rather than replaces.
    """
    cached: Dict[Tuple, Dict[str, object]] = {}
    if not variants_table.empty:
        for entry in variant_plan_from_table(variants_table, evaluable):
            cached.setdefault(plan_entry_key(entry), entry)

    plan: List[Dict[str, object]] = []
    missing: List[str] = []
    wanted: set = set()
    for want in requested:
        key = plan_entry_key(want)
        wanted.add(key)
        hit = cached.get(key)
        entry = dict(hit) if hit is not None else dict(want)
        path = score_matrix_path(
            scores_dir, str(entry["lineage_key"]), str(entry["source_variant"]),
            entry.get("score_matrix_path"),
        )
        entry["score_matrix_path"] = str(path)
        if hit is None:
            missing.append(f"{describe_plan_entry(want)} [not in score_variants.tsv]")
        elif not path.exists():
            missing.append(str(path))
        plan.append(entry)

    ignored = [
        describe_plan_entry(entry) for key, entry in cached.items() if key not in wanted
    ]
    return plan, missing, sorted(ignored)


def build_score_specs(
    args: argparse.Namespace,
    variant_plan: Sequence[Dict[str, object]],
    parent_map: Dict[str, str],
) -> List[Dict[str, object]]:
    """Collapse the (source variant, lineage) plan into one model spec per model.

    The returned dicts use exactly the key names rma.build_combined_rows and
    rma.export_plots expect, so nothing downstream needs to know that "model" here means
    an ESCOTT variant rather than a PLM checkpoint. Setting epoch_value to the PRESCOTT
    coefficient makes the existing epoch-trajectory figures plot metric vs penalty
    strength for free.
    """
    ordered: List[str] = []
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for entry in variant_plan:
        lineage = str(entry["lineage"])
        model_tag = canonical_model_tag(
            entry.get("equation"), entry.get("coefficient"), entry.get("frequency_cutoff_k"),
            entry.get("parent_lineage"), parent_map.get(lineage),
        )
        if model_tag not in grouped:
            grouped[model_tag] = []
            ordered.append(model_tag)
        grouped[model_tag].append({**entry, "model_tag": model_tag})

    # --score-variant accepts EITHER the model tag this driver reports (and writes into
    # every output table) OR the `variant` column of scores/score_variants.tsv, which is
    # what a user actually has in front of them and which always carries stage 1's
    # _parent<TOK> suffix. Matching only the former turned an obvious copy-paste into
    # "ValueError: ... not in the resolved model list".
    requested = set(args.score_variants) if args.score_variants else None
    aliases: Dict[str, set] = {
        model_tag: {model_tag, *(str(entry["source_variant"]) for entry in entries)}
        for model_tag, entries in grouped.items()
    }
    matched: set = set()
    specs: List[Dict[str, object]] = []
    for model_tag in ordered:
        if requested is not None:
            hits = requested.intersection(aliases[model_tag])
            if not hits:
                continue
            matched.update(hits)
        entries = grouped[model_tag]
        head = entries[0]

        if model_tag == "ESCOTT":
            epoch_label, epoch_value = "escott", 0.0
            display = "ESCOTT (no frequency term)"
        else:
            coefficient = float(head.get("coefficient") or 0.0)
            epoch_value = coefficient
            if model_tag.rsplit("_", 1)[-1].startswith("parent"):
                suffix = model_tag.rsplit("_", 1)[-1]
                epoch_label = f"prescott_c{coefficient:.2f}_{suffix}"
                display = f"PRESCOTT c={coefficient:.2f} (parent {head.get('parent_lineage')})"
            else:
                epoch_label = f"prescott_c{coefficient:.2f}"
                display = f"PRESCOTT c={coefficient:.2f}"

        specs.append({
            "model_tag": model_tag,
            "model_display_label": display,
            "base_model": "ESCOTT",
            "checkpoint_label": None,
            "epoch_label": epoch_label,
            "epoch_value": float(epoch_value),
            "precomputed_plm_path": None,
            "checkpoint_dir": None,
            # ESCOTT-specific bookkeeping. Ignored by the shared code, but written into
            # panel_metadata.tsv so a changed design invalidates the cache.
            "lineages": [str(entry["lineage"]) for entry in entries],
            "parent_by_lineage": {str(e["lineage"]): e.get("parent_lineage") for e in entries},
            "source_variant_by_lineage": {str(e["lineage"]): str(e["source_variant"]) for e in entries},
            "matrix_path_by_lineage": {str(e["lineage"]): e.get("score_matrix_path") for e in entries},
            "equation": head.get("equation"),
            "coefficient": head.get("coefficient"),
            "frequency_cutoff_k": head.get("frequency_cutoff_k"),
        })

    if requested is not None:
        missing = requested.difference(matched)
        if missing:
            known = sorted({name for names in aliases.values() for name in names})
            raise ValueError(
                f"--score-variant asked for {sorted(missing)}, which match neither a model tag "
                f"nor a stage-1 variant name in this run.\n"
                f"  Model tags     : {ordered}\n"
                f"  Stage-1 variants: {known}"
            )
    if not specs:
        raise RuntimeError("No score variants resolved; check --score-variant / --equation-grid")
    return specs


# --------------------------------------------------------------------------------------
# Stage 1 orchestration (runs under the PRESCOTT interpreter)
# --------------------------------------------------------------------------------------

def stage1_environment(args: argparse.Namespace) -> Dict[str, str]:
    """Environment for stage-1 subprocesses.

    biotite's DsspApp resolves ``mkdssp`` from PATH, so the PRESCOTT env's bin must be
    on it -- a bare interpreter path is not enough and fails with FileNotFoundError.
    """
    env = dict(os.environ)
    prescott_bin = Path(args.prescott_python).resolve().parent
    env["PATH"] = f"{prescott_bin}{os.pathsep}{env.get('PATH', '')}"
    env["MPLBACKEND"] = "Agg"
    env["R_LIBS_USER"] = ""  # do not let a stale user R library shadow the env's seqinr
    return env


def run_stage1_step(command: List[str], env: Dict[str, str], label: str) -> None:
    printable = " ".join(shlex.quote(str(part)) for part in command)
    print(f"[stage1:{label}] {printable}", flush=True)
    started = time.time()
    completed = subprocess.run([str(part) for part in command], env=env, cwd=str(REPO_ROOT))
    if completed.returncode != 0:
        raise RuntimeError(
            f"stage 1 step {label!r} failed with exit code {completed.returncode}.\n  {printable}"
        )
    print(f"[stage1:{label}] ok in {time.time() - started:.1f}s", flush=True)


def resolve_structure_entry(manifest: Dict[str, object], structure_role: str) -> Dict[str, object]:
    """``inputs_manifest['structures'][role]``, or a hard error naming what is available.

    A missing role used to fall through to a hard-coded ``6WXB_chainA_qnum.pdb``: pass
    ``--structure-role extra`` together with ``--prepare-args --no-extra-structure`` and
    the surrogate would quietly score the 1968 trimer while the manifest and CAVEATS
    both claimed the contemporary full-coverage model. Whichever structure is scored,
    the manifest must be able to name it.
    """
    structures = manifest.get("structures") if manifest else None
    if not isinstance(structures, dict) or not structures:
        # prepare_inputs has not run yet; the literal fallbacks in stage1_paths only
        # matter on that first pass, when we are about to create these files anyway.
        return {}
    entry = structures.get(structure_role)
    if not isinstance(entry, dict) or not entry:
        raise RuntimeError(
            f"--structure-role {structure_role!r} is not present in inputs_manifest.json "
            f"(available: {sorted(structures)}).\n"
            f"  The 'extra' role is the contemporary full-coverage model and is skipped by "
            f"prepare_inputs.py --no-extra-structure. Either drop that flag from "
            f"--prepare-args, or run with --structure-role primary."
        )
    return entry


def stage1_paths(
    inputs_dir: Path,
    lineage_key: str,
    manifest: Dict[str, object],
    structure_role: str = "primary",
) -> Dict[str, Path]:
    """Resolve stage-1 file names for one lineage.

    prepare_inputs.py records everything it wrote in inputs_manifest.json
    (``lineage_msas``/``queries``/``structures``), and that is the authority -- the
    structure PDB name in particular depends on which structure was passed. The literal
    fallbacks only matter before prepare_inputs has ever run, i.e. when we are about to
    create these files anyway.
    """
    inputs_dir = Path(inputs_dir)
    manifest = manifest or {}
    msa_entry = (manifest.get("lineage_msas") or {}).get(lineage_key, {})
    query_entry = (manifest.get("queries") or {}).get(lineage_key, {})
    structure_entry = resolve_structure_entry(manifest, structure_role)
    monomer = (structure_entry.get("monomer") or {}).get("path")
    trimer = (structure_entry.get("trimer") or {}).get("path")

    return {
        "msa": Path(msa_entry.get("path") or inputs_dir / "msa" / f"msa_{lineage_key}.fasta"),
        "query": Path(query_entry.get("path") or inputs_dir / "query" / f"{lineage_key}_query.fasta"),
        "jet": inputs_dir / "jet" / f"{lineage_key}_surrogate_jet.res",
        "jet_components": inputs_dir / "jet" / f"{lineage_key}_jet_components.tsv",
        "jet_manifest": inputs_dir / "jet" / f"{lineage_key}_jet_manifest.json",
        "dssp": inputs_dir / "jet" / f"{lineage_key}.dssp.csv",
        "chain_pdb": Path(monomer) if monomer else inputs_dir / "structure" / "6WXB_chainA_qnum.pdb",
        "trimer_pdb": Path(trimer) if trimer else inputs_dir / "structure" / "6WXB_trimer_qnum.pdb",
    }


def read_inputs_manifest(inputs_dir: Path) -> Dict[str, object]:
    path = Path(inputs_dir) / "inputs_manifest.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def run_stage1(
    args: argparse.Namespace,
    parent_map: Dict[str, str],
    evaluable: Sequence[str],
    diagnostics_dir: Path,
) -> None:
    """prepare_inputs -> jet_surrogate (per lineage) -> run_escott.

    Prepared lineages are the union of evaluation targets, their parents *and* the
    alternate parents named by --parent-sensitivity: parents are needed only for their
    population-frequency file, targets need the full query/MSA/jet chain. Every lineage
    that is scored also gets its own escott run -- GEMME uses the query as the epistatic
    reference, so one shared run would bias exactly the antigenic-site positions that
    differ between lineages.
    """
    prepare = require_stage1_script("prepare")
    jet = require_stage1_script("jet")
    escott = require_stage1_script("escott")

    interpreter = Path(args.prescott_python)
    if not interpreter.exists():
        raise RuntimeError(
            f"--prescott-python {interpreter} does not exist. Stage 1 needs an interpreter with "
            "prody, biotite+mkdssp, freesasa, R and the escott console script -- the PRESCOTT "
            "conda env. It is invoked as a subprocess because each stage-1 step is separately "
            "cacheable and runnable by hand, not because its imports conflict with this one."
        )
    env = stage1_environment(args)

    alternates = effective_sensitivity_edges(args, parent_map, evaluable)
    sensitivity_spec = sensitivity_edge_spec(alternates)

    prepared = sorted({
        *evaluable,
        *(parent_map[l] for l in evaluable if l in parent_map),
        *alternates.values(),
    })
    inputs_dir = Path(args.inputs_dir)

    prepare_cmd: List[object] = [
        interpreter, prepare,
        "--guide-path", args.guide_path,
        "--deep-fasta", args.deep_fasta,
        "--inputs-dir", inputs_dir,
        "--structure", args.structure,
        "--parent-map-preset", args.parent_map_preset,
        "--parent-min-count", args.parent_min_count,
        "--parent-min-depth", args.parent_min_depth,
        "--parent-freq-max", args.parent_freq_max,
        "--frequency-cutoff-mode", args.frequency_cutoff_mode,
        "--frequency-cutoff-k", args.frequency_cutoff_k,
        "--frequency-cutoff", args.frequency_cutoff,
        "--seed", args.seed,
    ]
    # Pinned rather than left to prepare_inputs' default: this is the switch that
    # actually removes the exact ancestral reversions (--parent-freq-max 0.95 does not
    # catch K's N160S at 0.932), so flipping the default downstream must not silently
    # change every PRESCOTT score under an unchanged manifest.
    prepare_cmd.append(
        "--drop-parent-reversions" if args.drop_parent_reversions else "--no-drop-parent-reversions"
    )
    # ---- leakage detection / purge ------------------------------------------------
    # Both booleans are forwarded EXPLICITLY in both directions rather than "only when
    # off". prepare_inputs.py defaults them on, and an inputs tree carries a purge cache
    # keyed on the thresholds; if the driver stayed silent, "--no-purge-leakage" on one
    # run and nothing on the next would leave the operator unable to tell from the
    # command line which alignment ESCOTT actually read. The manifest records the
    # answer, but the command that produced it should say so too.
    prepare_cmd.append("--leakage-check" if args.leakage_check else "--no-leakage-check")
    prepare_cmd.append("--purge-leakage" if args.purge_leakage else "--no-purge-leakage")
    if args.fail_on_leakage:
        prepare_cmd.append("--fail-on-leakage")
    # Thresholds default to None here and are forwarded only when set, so
    # leakage_check.py remains the single authority on the numbers -- the same
    # discipline --trace-top-fraction had to be put under after the driver silently
    # overrode a measured default with a worse one.
    for flag, value in (
        ("--leakage-min-identity", args.leakage_min_identity),
        ("--leakage-max-hamming", args.leakage_max_hamming),
        ("--leakage-min-coverage", args.leakage_min_coverage),
        ("--leakage-coverage-basis", args.leakage_coverage_basis),
        ("--leakage-max-removed-fraction", args.leakage_max_removed_fraction),
        ("--leakage-min-depth-after", args.leakage_min_depth_after),
        ("--leakage-threads", args.leakage_threads),
        ("--blast-task", args.blast_task),
    ):
        if value is not None:
            prepare_cmd += [flag, value]

    if args.parent_map:
        prepare_cmd += ["--parent-map", args.parent_map]
    if sensitivity_spec:
        # This is what makes --parent-sensitivity a real analysis: a SECOND frequency
        # file per alternate edge (<key>_parent<TOK>_frequency.txt), which run_escott
        # below turns into an independently named _parent<TOK> score matrix per grid
        # point. Without it the flag was inert -- the driver planned the sensitivity
        # variants, stage 1 could not produce them and they vanished without a word.
        prepare_cmd += ["--sensitivity-parent-map", sensitivity_spec]
    for lineage in prepared:
        prepare_cmd += ["--only-lineage", lineage]
    if args.force_recompute_scores:
        prepare_cmd.append("--force")
    prepare_cmd += shlex.split(args.prepare_args or "")
    run_stage1_step([str(part) for part in prepare_cmd], env, "prepare_inputs")

    manifest = read_inputs_manifest(inputs_dir)
    for lineage in evaluable:
        key = safe_key(lineage)
        paths = stage1_paths(inputs_dir, key, manifest, args.structure_role)
        # jet_surrogate.py has no --sasa-context: the SASA/CV environment IS whatever
        # --context-pdb points at, so 'trimer' vs 'monomer' selects the file here.
        context_pdb = paths["trimer_pdb"] if args.sasa_context == "trimer" else paths["chain_pdb"]
        jet_cmd: List[object] = [
            interpreter, jet,
            "--msa", paths["msa"],
            "--query", paths["query"],
            "--pdb", paths["chain_pdb"],
            "--context-pdb", context_pdb,
            "--out-jet", paths["jet"],
            "--out-components", paths["jet_components"],
            "--out-manifest", paths["jet_manifest"],
            "--out-dssp", paths["dssp"],
            "--weight-mode", args.weight_mode,
            "--pc-mode", args.pc_mode,
            "--cv-radius", args.cv_radius,
            "--max-coil-length", args.max_coil_length,
            "--trace-definition", args.trace_definition,
            "--trace-bootstraps", args.trace_bootstraps,
            "--seed", args.seed,
        ]
        # Forwarded ONLY when the user set them. jet_surrogate.py is the authority on
        # both numbers (constants.DEFAULT_TRACE_TOP_FRACTION / MAX_ZERO_TRACE_FRACTION);
        # passing them unconditionally is how the driver came to override a measured
        # 0.90 with 0.30 and leave 77/566 HA positions scoring as pure noise.
        if args.trace_top_fraction is not None:
            jet_cmd += ["--trace-top-fraction", args.trace_top_fraction]
        if args.max_zero_trace_fraction is not None:
            jet_cmd += ["--max-zero-trace-fraction", args.max_zero_trace_fraction]
        if args.force_recompute_scores:
            jet_cmd.append("--force")
        jet_cmd += shlex.split(args.jet_args or "")
        run_stage1_step([str(part) for part in jet_cmd], env, f"jet_surrogate:{lineage}")

    # The surrogate-vs-real-JET2 comparison CAVEATS.md caveat 1 cites. It runs on BLAT
    # because BLAT is the only protein PRESCOTT ships a real JET2 .res for, so it is the
    # only place a row-for-row comparison is possible at all.
    if args.jet_validation:
        validation_path = rma.ensure_dir(diagnostics_dir) / JET_VALIDATION_BASENAME
        validation_cmd: List[object] = [
            interpreter, jet,
            "--validate-only",
            "--out-validation", validation_path,
            "--weight-mode", args.weight_mode,
            "--pc-mode", args.pc_mode,
            "--cv-radius", args.cv_radius,
            "--trace-definition", args.trace_definition,
            "--trace-bootstraps", args.trace_bootstraps,
            "--seed", args.seed,
        ]
        if args.trace_top_fraction is not None:
            validation_cmd += ["--trace-top-fraction", args.trace_top_fraction]
        run_stage1_step([str(part) for part in validation_cmd], env, "jet_surrogate:validate")

    escott_cmd: List[object] = [
        interpreter, escott,
        "--inputs-dir", inputs_dir,
        "--escott-workdir", args.escott_workdir,
        "--scores-dir", args.scores_dir,
        # Explicit, because run_escott defaults it to <scores-dir>/../tables/diagnostics
        # -- which is this run's diagnostics dir only while --scores-dir is left alone.
        # Share a scores dir between runs and the parity table used to land outside the
        # output directory that CAVEATS.md and the manifest describe.
        "--diagnostics-dir", diagnostics_dir,
        "--max-coil-length", args.max_coil_length,
        # run_escott.py takes a single scalar T; --escott-temperature-mode match-plm is
        # resolved to a number by resolve_escott_temperature() before we get here.
        "--escott-temperature", args.escott_temperature,
        "--coefficient-grid", args.coefficient_grid,
        "--equation-grid", args.equation_grid,
        "--frequency-cutoff-k", args.frequency_cutoff_k,
        "--frequency-cutoff-mode", args.frequency_cutoff_mode,
        "--frequency-cutoff", args.frequency_cutoff,
    ]
    for lineage in evaluable:
        escott_cmd += ["--lineage", lineage]
    if args.parent_map:
        escott_cmd += ["--parent-map", args.parent_map]
    if sensitivity_spec:
        escott_cmd += ["--sensitivity-parent-map", sensitivity_spec]
    else:
        # Stage C otherwise picks alternate edges up from inputs_manifest.json, so an
        # earlier --parent-sensitivity run would keep emitting them after the flag was
        # turned off, and the driver would then report them as "cached but unrequested".
        escott_cmd.append("--no-parent-sensitivity")
    if args.force_recompute_scores:
        escott_cmd.append("--force")
    # The prescott.py parity run is the slow, fragile part of stage C and is a
    # cross-check, not a product, so the smoke test skips it. We deliberately do NOT
    # pass run_escott's own --test-mode: that would drop every PRESCOTT variant and
    # leave the smoke test exercising only half the score path, and the PRESCOTT grid
    # is pure numpy on an already-loaded matrix (well under a second per grid point).
    if not args.test_mode:
        escott_cmd += ["--prescott-ref-dir", args.prescott_ref_dir]
    escott_cmd += shlex.split(args.escott_args or "")
    run_stage1_step([str(part) for part in escott_cmd], env, "run_escott")


# --------------------------------------------------------------------------------------
# Score matrix access -- the ESCOTT analogue of rma.ensure_plm_matrix
# --------------------------------------------------------------------------------------

def ensure_score_matrix(
    args: argparse.Namespace,
    spec: Dict[str, object],
    lineage_label: str,
    lineage_data: Dict[str, object],
    scores_dir: Path,
) -> Tuple[pd.DataFrame, str, str]:
    """Return (matrix, path, source_sequence), matching rma.ensure_plm_matrix's contract.

    No torch, no GPU, no model: the matrix is just read from disk. Stage 1 is fired
    up-front in run_analysis rather than lazily here, because stage 1 is what *names*
    the variants -- triggering it mid-loop would mean the specs were built from guessed
    names that stage 1 may not agree with.
    """
    from Functions_HuggingFace import load_plm_probability_matrix

    lineage_key = str(lineage_data["lineage_key"])
    source_variant = str(spec["source_variant_by_lineage"][lineage_label])
    path = score_matrix_path(
        scores_dir, lineage_key, source_variant, spec["matrix_path_by_lineage"].get(lineage_label)
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Score matrix not found: {path}\n"
            f"  Produce it with:\n"
            f"    {args.prescott_python} {STAGE1_SCRIPTS['escott']} "
            f"--inputs-dir {args.inputs_dir} --escott-workdir {args.escott_workdir} "
            f"--scores-dir {scores_dir} --lineage {lineage_label}\n"
            f"  or rerun this driver with --auto-prepare."
        )

    raw_matrix = load_plm_probability_matrix(str(path))
    matrix = rma.normalise_plm_matrix(raw_matrix)
    source_sequence = rma.infer_plm_source_sequence(raw_matrix) or str(lineage_data["plm_ref_protein"])
    return matrix, str(path), source_sequence


def read_jet_manifests(
    inputs_dir: Path,
    inputs_manifest: Dict[str, object],
    lineages: Sequence[str],
    structure_role: str,
) -> Dict[str, Dict[str, object]]:
    """Per-lineage JET-surrogate manifests, keyed by lineage label.

    The zero-trace counts in there are the single most important quality number stage B
    produces -- pred.R multiplies every ESCOTT column by ``trace[i]``, so a zero-trace
    column is a uniform softmax and a dead site -- and they were previously not surfaced
    anywhere in the output tree, which is why a bad --trace-top-fraction was invisible.
    """
    out: Dict[str, Dict[str, object]] = {}
    for lineage in lineages:
        path = stage1_paths(inputs_dir, safe_key(lineage), inputs_manifest, structure_role)["jet_manifest"]
        if not Path(path).exists():
            continue
        try:
            with Path(path).open(encoding="utf-8") as handle:
                out[str(lineage)] = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
    return out


def write_jet_surrogate_summary(
    jet_manifests: Dict[str, Dict[str, object]],
    diagnostics_dir: Path,
) -> Optional[Path]:
    """One row per lineage: the trace settings actually used and how much of HA died."""
    if not jet_manifests:
        return None
    rows = []
    for lineage, manifest in sorted(jet_manifests.items()):
        structure = manifest.get("structure") or {}
        rows.append({
            "lineage": lineage,
            "msa_n_sequences": manifest.get("msa_n_sequences"),
            "msa_n_columns": manifest.get("msa_n_columns"),
            "weight_mode": manifest.get("weight_mode"),
            "trace_definition": manifest.get("trace_definition"),
            "trace_bootstraps": manifest.get("trace_bootstraps"),
            "trace_top_fraction": manifest.get("trace_top_fraction"),
            "n_zero_trace_columns": manifest.get("n_zero_trace_columns"),
            "frac_zero_trace_columns": manifest.get("frac_zero_trace_columns"),
            "n_positions_without_structure": manifest.get("n_positions_without_structure"),
            "structure_pdb": structure.get("pdb"),
            "structure_context_pdb": structure.get("context_pdb"),
            "structure_covered": structure.get("covered"),
            "structure_query_identity": structure.get("structure_query_identity"),
            "jet_res_path": manifest.get("jet_res_path"),
            "jet_res_md5": manifest.get("jet_res_md5"),
        })
    out_path = rma.ensure_dir(diagnostics_dir) / "jet_surrogate_summary.tsv"
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    worst = max(
        (float(row["frac_zero_trace_columns"] or 0.0) for row in rows if row.get("frac_zero_trace_columns") is not None),
        default=0.0,
    )
    if worst > 0.05:
        print(
            f"WARNING: up to {worst:.1%} of positions have trace == 0 in the JET surrogate. "
            f"pred.R multiplies every ESCOTT column by trace[i], so those sites score as a "
            f"uniform 1/20 and contribute noise. See {out_path}."
        )
    return out_path


def resolve_escott_temperature(args: argparse.Namespace, scores_dir: Path, evaluable: Sequence[str]) -> float:
    """Resolve --escott-temperature-mode match-plm into the scalar run_escott takes.

    T is chosen so sd(log P) matches a reference PLM run's sd(log plm_prob): with
    P = softmax(E/T), sd(log P) ~ sd(E)/T, so T = sd(E) / sd(log plm_prob). That needs
    the raw ESCOTT values, which only exist after stage 1 has run once -- hence the
    two-pass requirement, and hence a single global T (run_escott takes one scalar, not
    one per lineage). The per-lineage spread is reported in score_scale_report.tsv so
    the residual mismatch is visible rather than hidden.
    """
    if args.escott_temperature_mode != "match-plm":
        return float(args.escott_temperature)

    reference_path = Path(args.plm_reference_table)
    if not reference_path.exists():
        raise FileNotFoundError(f"--plm-reference-table not found: {reference_path}")
    reference = pd.read_csv(reference_path, usecols=["plm_prob"])
    reference_sd = float(np.nanstd(
        np.log(pd.to_numeric(reference["plm_prob"], errors="coerce").clip(lower=1e-32).to_numpy())
    ))
    if not np.isfinite(reference_sd) or reference_sd <= 0:
        raise ValueError(f"{reference_path} gave a non-positive sd(log plm_prob); cannot match it")

    temperatures: List[float] = []
    closed_form: List[float] = []
    for lineage in evaluable:
        raw_path = Path(scores_dir) / f"{safe_key(lineage)}_ESCOTT_raw.tsv"
        if not raw_path.exists():
            raise FileNotFoundError(
                f"--escott-temperature-mode match-plm needs the raw ESCOTT values at {raw_path}, "
                "which stage 1 only writes after a first pass. Run once with "
                "--escott-temperature-mode fixed, then rerun with match-plm."
            )
        raw = pd.read_csv(raw_path, sep="\t", index_col=0)
        values = raw.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        closed_form.append(float(np.nanstd(values)) / reference_sd)
        temperatures.append(solve_softmax_temperature(values, reference_sd, context=str(raw_path)))

    temperature = float(np.median(temperatures))
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(
            f"--escott-temperature-mode match-plm resolved to T={temperature}, which is not a "
            f"usable temperature. Per-lineage solutions were {temperatures}."
        )
    print(f"[match-plm] sd(log plm_ref)={reference_sd:.4f}, per-lineage T={[round(t, 4) for t in temperatures]} "
          f"=> T={temperature:.4f} (the discarded sd(E)/sd(log p) estimate would have been "
          f"{float(np.median(closed_form)):.4f})")
    return temperature


def sd_log_softmax(values: np.ndarray, temperature: float) -> float:
    """``sd(log softmax_column(E / T))`` over the whole 20 x L grid.

    This -- not ``sd(E) / T`` -- is the quantity ``--escott-temperature-mode match-plm``
    has to control, because ``plm_prob`` in the combined table IS the per-column softmax
    and the alpha sweep optimises ``log(plm_prob) + alpha * log(mut_prob)``.

    The two differ by more than a rounding error.  ``log P = E/T - logsumexp_column(E/T)``
    subtracts a *per-column* constant, which annihilates all BETWEEN-column variance and
    leaves only the within-column part.  On real ESCOTT output (the PRESCOTT
    distribution's own MLH1 matrix) the total sd is 1.8155 and the within-column sd is
    0.9005, so ``sd(E)`` overstates the spread the sweep actually sees by 2.0x and the
    old closed form chose a T that was 1.36-1.75x too large.

    Evaluated as ``shifted - log(sum(exp(shifted)))`` rather than ``log(softmax(...))``
    so it stays exact at temperatures where the probabilities themselves underflow to
    zero and ``log`` would return ``-inf``.  The wild-type NaN is filled with the column
    maximum first, exactly as :func:`run_escott.escott_to_probability` does, so this
    measures the matrix that is actually written to disk.
    """
    grid = np.array(values, dtype=float, copy=True)
    if grid.ndim == 1:
        grid = grid.reshape(-1, 1)
    rows, cols = np.where(np.isnan(grid))
    if rows.size:
        grid[rows, cols] = np.nanmax(grid, axis=0)[cols]
    scaled = grid / float(temperature)
    shifted = scaled - scaled.max(axis=0, keepdims=True)
    log_norm = np.log(np.exp(shifted).sum(axis=0, keepdims=True))
    return float(np.std(shifted - log_norm))


# Bisection bracket for solve_softmax_temperature. 1e-8 is far below any temperature
# whose softmax is representable and 1e8 is far above the one that flattens any real
# matrix to uniform, so the root is always interior for a non-degenerate matrix.
SOFTMAX_TEMPERATURE_BRACKET = (1e-8, 1e8)


def solve_softmax_temperature(
    values: np.ndarray,
    target_sd: float,
    *,
    context: str = "",
    bracket: Tuple[float, float] = SOFTMAX_TEMPERATURE_BRACKET,
    max_iter: int = 200,
) -> float:
    """The T for which ``sd_log_softmax(values, T) == target_sd``, by bisection in log T.

    ``sd_log_softmax`` is strictly decreasing in T -- it is dominated by the ``E / T``
    term and tends to ``0`` as ``T -> inf`` (every column becomes uniform) and to
    infinity as ``T -> 0`` -- so a bisection on ``log T`` converges to the unique root.
    Solving instead of using the ``T = sd(E) / target`` closed form is what makes
    match-plm actually match: the closed form ignores the per-column ``logsumexp``.
    """
    if not np.isfinite(target_sd) or target_sd <= 0:
        raise ValueError(f"cannot match a non-positive target sd(log plm_prob): {target_sd}")
    lo, hi = float(bracket[0]), float(bracket[1])
    where = f" for {context}" if context else ""
    sd_lo = sd_log_softmax(values, lo)
    sd_hi = sd_log_softmax(values, hi)
    if not np.isfinite(sd_lo) or sd_lo <= target_sd:
        raise ValueError(
            f"cannot reach sd(log plm_prob) = {target_sd:g}{where}: even T={lo:g} only gives "
            f"{sd_lo:g}. The raw ESCOTT matrix has almost no within-position spread (every "
            f"column near-constant), so no temperature can match this PLM reference."
        )
    if sd_hi >= target_sd:  # pragma: no cover - needs a target below 1e-8 sd
        raise ValueError(
            f"cannot reach sd(log plm_prob) = {target_sd:g}{where}: even T={hi:g} still gives "
            f"{sd_hi:g}."
        )
    for _ in range(max_iter):
        mid = math.sqrt(lo * hi)
        if sd_log_softmax(values, mid) > target_sd:
            lo = mid
        else:
            hi = mid
        if hi / lo - 1.0 < 1e-12:
            break
    return math.sqrt(lo * hi)


# --------------------------------------------------------------------------------------
# Manifest, caveats, diagnostics
# --------------------------------------------------------------------------------------

def resolve_structure_record(
    args: argparse.Namespace,
    inputs_manifest: Dict[str, object],
) -> Dict[str, object]:
    """What the JET surrogate ACTUALLY scored against, for the manifest and CAVEATS.

    ``--structure`` is only the source file handed to prepare_inputs for the *primary*
    role. With ``--structure-role extra`` the surrogate reads the contemporary J.2.4.1
    model instead, and the manifest used to record ``--structure`` regardless -- so a
    run whose cv/pc/DSSP terms came from the 566-residue 2025 model was documented as
    having used the 485/566-coverage 1968 trimer, with a matching md5 that 'proved' it.
    """
    entry = resolve_structure_entry(inputs_manifest, args.structure_role)
    if not entry:
        return {
            "structure_role": args.structure_role,
            "structure_source_path": str(args.structure),
            "structure_source_md5": file_md5(args.structure),
            "structure_monomer_path": None,
            "structure_trimer_path": None,
            "structure_coverage_fraction": None,
            "structure_n_covered": None,
            "structure_query_identity": None,
            "structure_resolved_from_inputs_manifest": False,
        }
    monomer = (entry.get("monomer") or {}).get("path")
    trimer = (entry.get("trimer") or {}).get("path")
    return {
        "structure_role": args.structure_role,
        "structure_source_path": entry.get("source_path"),
        "structure_source_md5": entry.get("source_md5"),
        "structure_monomer_path": monomer,
        "structure_monomer_md5": file_md5(monomer) if monomer else None,
        "structure_trimer_path": trimer,
        "structure_trimer_md5": file_md5(trimer) if trimer else None,
        "structure_coverage_fraction": entry.get("coverage_fraction"),
        "structure_n_covered": entry.get("n_covered"),
        "structure_query_identity": entry.get("offset_identity"),
        "structure_resolved_from_inputs_manifest": True,
    }


def leakage_manifest_record(
    args: argparse.Namespace,
    inputs_manifest: Dict[str, object],
) -> Dict[str, object]:
    """The leakage block for run_manifest.json, taken from what stage 1 actually did.

    ``inputs_manifest['leakage']`` is the authority, not ``args``, for the same reason
    ``resolve_structure_record`` reads the manifest: the inputs tree may have been
    prepared by an earlier invocation at different thresholds and reused from cache, in
    which case ``args`` describes a purge that did not happen. When stage 1 has never
    run (``--no-auto-prepare`` on a pre-built tree from before this feature existed) the
    record says so explicitly rather than claiming the requested settings were applied.
    """
    block = (inputs_manifest or {}).get("leakage")
    requested = {
        "leakage_check_requested": bool(args.leakage_check),
        "purge_leakage_requested": bool(args.purge_leakage),
        "fail_on_leakage": bool(args.fail_on_leakage),
    }
    if not isinstance(block, dict) or not block:
        return {
            **requested,
            "leakage_stage_ran": False,
            "leakage_status": None,
            "leakage_thresholds": None,
            "leakage_purge_applied": None,
            "leakage_per_target": None,
            "leakage_report_dir": None,
            "leakage_note": (
                "No 'leakage' block in inputs_manifest.json. Either stage 1 did not run "
                "this time (--no-auto-prepare on an inputs tree built before leakage "
                "screening existed) or it was disabled there. The deep MSA handed to "
                "ESCOTT is UNAUDITED for overlap with the evaluation panels."
            ),
        }
    purges = block.get("purges") or {}
    per_target = {
        str(target): {
            "depth_before": entry.get("depth_before"),
            "n_removed": entry.get("n_removed"),
            "depth_after": entry.get("depth_after"),
            "removed_fraction": entry.get("removed_fraction"),
            "n_removed_exact_full_length": entry.get("n_removed_exact_full_length"),
            "removed_identity_max": (entry.get("removed_identity_distribution") or {}).get("max"),
            "removed_hamming_min": entry.get("removed_hamming_min"),
            "query_would_have_been_purged": bool(entry.get("query_exempted")),
            "drop_manifest_path": entry.get("drop_manifest_path"),
            "prepurge_path": entry.get("prepurge_path"),
        }
        for target, entry in purges.items()
    }
    b_checks = ((block.get("checks") or {}).get("B_parent_vs_target") or {})
    return {
        **requested,
        "leakage_stage_ran": True,
        "leakage_status": block.get("status"),
        "leakage_failures": block.get("failures") or None,
        "leakage_thresholds": block.get("thresholds"),
        "leakage_blast": block.get("blast"),
        "leakage_purge_applied": bool(block.get("purge")),
        "leakage_per_target": per_target or None,
        "leakage_parent_vs_target": {
            str(target): {
                "parent": (entry or {}).get("parent"),
                "n_shared_accessions": ((entry or {}).get("accessions") or {}).get(
                    "n_shared_accessions"),
                "n_shared_exact_sequences": ((entry or {}).get("accessions") or {}).get(
                    "n_shared_exact_sequences"),
                "n_flagged": (entry or {}).get("n_flagged"),
            }
            for target, entry in b_checks.items()
        } or None,
        "leakage_report_dir": block.get("report_dir"),
    }


def shared_design_signature(args: argparse.Namespace) -> Dict[str, object]:
    """The design terms EVERY model in the run shares.

    Deliberately excludes the PRESCOTT grids: which coefficients were requested changes
    which models exist, but it does not change the numbers inside a model that was
    requested by both runs -- least of all the ESCOTT baseline, which has no coefficient
    at all. Folding the grid in here would invalidate ESCOTT's cached alpha sweep every
    time the coefficient grid moved, for no reason.
    """
    return {
        "frequency_cutoff_mode": args.frequency_cutoff_mode,
        "frequency_cutoff": float(args.frequency_cutoff),
        "drop_parent_reversions": bool(args.drop_parent_reversions),
        "parent_min_count": int(args.parent_min_count),
        "parent_min_depth": int(args.parent_min_depth),
        "parent_freq_max": float(args.parent_freq_max),
        "escott_temperature": float(args.escott_temperature),
        "mutation_model": args.mutation_model,
        "alpha_grid": [round(float(a), 6) for a in rma.parse_alpha_grid(args).tolist()],
        "filter_fixed_mutations": bool(args.filter_fixed_mutations),
        "filter_singleton_mutations": bool(args.filter_singleton_mutations),
        "skip_low_count_sites": bool(args.skip_low_count_sites),
        "min_obs_count": int(args.min_obs_count),
        "expect_protein_diversity": bool(getattr(args, "expect_protein_diversity", False)),
        "test_mode": bool(args.test_mode),
        "test_max_records": int(args.test_max_records),
        "cache_version": PRESCOTT_CACHE_VERSION,
    }


def design_signature(
    args: argparse.Namespace,
    parent_map: Dict[str, str],
    evaluable: Sequence[str],
) -> Dict[str, object]:
    """The WHOLE run's design, for run_manifest.json. Superset of the shared part."""
    signature = shared_design_signature(args)
    signature.update({
        "prescott_equations": parse_int_grid(args.equation_grid),
        "prescott_coefficients": parse_float_grid(args.coefficient_grid),
        "frequency_cutoff_k": parse_int_grid(args.frequency_cutoff_k),
        "parent_map": {label: parent_map.get(label) for label in sorted(evaluable)},
        "parent_sensitivity_edges": effective_sensitivity_edges(args, parent_map, evaluable),
    })
    return signature


def model_design_signature(
    args: argparse.Namespace,
    spec: Dict[str, object],
    parent_map: Dict[str, str],
) -> Dict[str, object]:
    """One MODEL's design: the shared terms plus what makes this model itself.

    This, hashed, is the ``design_key`` column of panel_metadata.tsv, and it is what
    ``model_cache_is_valid`` compares. Per-model rather than per-run so that changing
    --coefficient-grid recomputes only the coefficients that changed and leaves ESCOTT
    (and any surviving grid point) cached -- while any change that DOES reach a model's
    numbers still invalidates it.
    """
    signature = shared_design_signature(args)
    signature.update({
        "model_tag": str(spec["model_tag"]),
        "equation": _optional_number(spec.get("equation")),
        "coefficient": _optional_number(spec.get("coefficient")),
        "frequency_cutoff_k": _optional_number(spec.get("frequency_cutoff_k")),
        "lineages": sorted(str(label) for label in spec["lineages"]),
        "parent_by_lineage": {
            str(label): _normalised_label(parent)
            for label, parent in sorted(spec["parent_by_lineage"].items())
        },
        "resolved_parent_map": {
            str(label): parent_map.get(str(label)) for label in sorted(spec["lineages"])
        },
        "source_variant_by_lineage": {
            str(label): str(variant)
            for label, variant in sorted(spec["source_variant_by_lineage"].items())
        },
    })
    return signature


def design_key(signature: Dict[str, object]) -> str:
    payload = json.dumps(signature, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def model_design_key(
    args: argparse.Namespace,
    spec: Dict[str, object],
    parent_map: Dict[str, str],
) -> str:
    return design_key(model_design_signature(args, spec, parent_map))


def save_run_manifest(
    args: argparse.Namespace,
    output_dir: Path,
    target_specs: List[Dict[str, str]],
    parent_map: Dict[str, str],
    specs: List[Dict[str, object]],
    variants_table: pd.DataFrame,
    evaluable: Sequence[str],
    jet_manifests: Optional[Dict[str, Dict[str, object]]] = None,
) -> None:
    """Genuine superset of the PLM run's manifest: every key it has, plus ours.

    'Superset' is checked, not asserted: ``rma.save_run_manifest``'s key list is
    reproduced below in full (alignment_verify_max_cols, rolling_identity_window and
    the three observed_mutation_* keys used to be missing even though this CLI exposes
    all five and all five change the --diagnostic-exports figures).
    """
    inputs_manifest = read_inputs_manifest(args.inputs_dir)
    jet_paths = {}
    for lineage in sorted({*parent_map, *evaluable}):
        key = safe_key(lineage)
        jet_file = stage1_paths(args.inputs_dir, key, inputs_manifest, args.structure_role)["jet"]
        if jet_file.exists():
            jet_paths[lineage] = {"path": str(jet_file), "md5": file_md5(jet_file)}

    signature = design_signature(args, parent_map, evaluable)
    validation_path = Path(output_dir) / "tables" / "diagnostics" / JET_VALIDATION_BASENAME
    manifest = {
        "analysis_mode": args.analysis_mode,
        "mutation_model": args.mutation_model,
        "output_dir": str(output_dir),
        "score_source": "escott" if len(specs) == 1 and specs[0]["model_tag"] == "ESCOTT" else "prescott",
        "escott_transform": "per_position_softmax",
        "escott_temperature": float(args.escott_temperature),
        "escott_temperature_mode": args.escott_temperature_mode,
        "plm_reference_table": str(args.plm_reference_table) if args.plm_reference_table else None,
        "prescott_equations": parse_int_grid(args.equation_grid),
        "prescott_coefficients": parse_float_grid(args.coefficient_grid),
        "frequency_cutoff_mode": args.frequency_cutoff_mode,
        "frequency_cutoff_k": parse_int_grid(args.frequency_cutoff_k),
        "frequency_cutoff": float(args.frequency_cutoff),
        "parent_map": parent_map,
        "parent_map_preset": args.parent_map_preset,
        "parent_sensitivity": bool(args.parent_sensitivity),
        "parent_sensitivity_edges": sensitivity_edges(args, parent_map),
        # The edges above are a property of the presets; these are the ones this run
        # could act on, i.e. those whose child lineage was actually evaluated.
        "parent_sensitivity_edges_applied": effective_sensitivity_edges(args, parent_map, evaluable),
        "input_only_lineages": sorted(input_only_lineages()),
        "evaluable_lineages": list(evaluable),
        "parent_min_count": int(args.parent_min_count),
        "parent_min_depth": int(args.parent_min_depth),
        "parent_freq_max": float(args.parent_freq_max),
        # Read back from inputs_manifest.json rather than from args wherever stage 1
        # is the one that decided: it is what actually built the frequency files.
        "drop_parent_reversions": (
            (inputs_manifest.get("args") or {}).get("drop_parent_reversions",
                                                    bool(args.drop_parent_reversions))
            if inputs_manifest else bool(args.drop_parent_reversions)
        ),
        "n_parent_reversion_mutants_dropped": {
            str(lineage): (entry or {}).get("n_parent_reversion_mutants")
            for lineage, entry in ((inputs_manifest.get("frequency") or {}) if inputs_manifest else {}).items()
        } or None,
        "msa_source_path": str(args.deep_fasta),
        "msa_source_md5": file_md5(args.deep_fasta),
        # Which sequences ESCOTT was allowed to see. Read back from
        # inputs_manifest.json, never from args: stage 1 is what ran the purge, and a
        # cached inputs tree may have been purged at thresholds this invocation did not
        # ask for. Two runs with different `leakage` blocks are not comparable, so this
        # has to be legible at the top level rather than inferable from a file listing.
        **leakage_manifest_record(args, inputs_manifest),
        # The structure the surrogate actually read, not just --structure. See
        # resolve_structure_record: with --structure-role extra these differ.
        **resolve_structure_record(args, inputs_manifest),
        "structure_arg_path": str(args.structure),
        "structure_arg_md5": file_md5(args.structure),
        "weight_mode": args.weight_mode,
        "pc_mode": args.pc_mode,
        "sasa_context": args.sasa_context,
        "cv_radius": float(args.cv_radius),
        "max_coil_length": int(args.max_coil_length),
        "trace_definition": args.trace_definition,
        "trace_bootstraps": int(args.trace_bootstraps),
        # None means "whatever jet_surrogate.py's own default is"; record both so a
        # later change to that default is attributable.
        "trace_top_fraction_requested": (
            None if args.trace_top_fraction is None else float(args.trace_top_fraction)
        ),
        "trace_top_fraction": (
            float(args.trace_top_fraction) if args.trace_top_fraction is not None
            else default_trace_top_fraction()
        ),
        "max_zero_trace_fraction": (
            None if args.max_zero_trace_fraction is None else float(args.max_zero_trace_fraction)
        ),
        "jet_validation": bool(args.jet_validation),
        "jet_validation_table": str(validation_path) if validation_path.exists() else None,
        "jet_surrogate_manifests": jet_manifests or None,
        "seed": int(args.seed),
        "jetfile_paths": jet_paths,
        "inputs_manifest": inputs_manifest or None,
        "score_variants": variants_table.to_dict("records") if not variants_table.empty else None,
        "model_specs": [
            {
                **{key: spec[key] for key in ("model_tag", "epoch_label", "epoch_value", "lineages")},
                "equation": spec.get("equation"),
                "coefficient": spec.get("coefficient"),
                "frequency_cutoff_k": spec.get("frequency_cutoff_k"),
                "parent_by_lineage": spec.get("parent_by_lineage"),
                "source_variant_by_lineage": spec.get("source_variant_by_lineage"),
                # Matches the design_key column of the model's panel_metadata.tsv rows,
                # so "does this manifest describe these outputs?" is a string compare.
                "design_key": model_design_key(args, spec, parent_map),
            }
            for spec in specs
        ],
        "filter_fixed_mutations": bool(args.filter_fixed_mutations),
        "filter_singleton_mutations": bool(args.filter_singleton_mutations),
        "skip_low_count_sites": bool(args.skip_low_count_sites),
        "min_obs_count": int(args.min_obs_count),
        "expect_protein_diversity": bool(getattr(args, "expect_protein_diversity", False)),
        "diagnostic_exports": bool(args.diagnostic_exports),
        # --- the five PLM-manifest keys this one used to drop. All five exist on this
        # CLI and all five change the --diagnostic-exports figures, so two runs that
        # differed only in --rolling-identity-window produced identical manifests.
        "alignment_verify_max_cols": getattr(args, "alignment_verify_max_cols", 100),
        "rolling_identity_window": getattr(args, "rolling_identity_window", 30),
        "observed_mutation_fasta": (
            str(args.observed_mutation_fasta) if getattr(args, "observed_mutation_fasta", None) else None
        ),
        "observed_mutation_sequence_id": getattr(args, "observed_mutation_sequence_id", None),
        "observed_mutation_selection": getattr(args, "observed_mutation_selection", "last"),
        "alpha_start": float(args.alpha_start),
        "alpha_stop": float(args.alpha_stop),
        "alpha_step": float(args.alpha_step),
        "alpha_grid": rma.parse_alpha_grid(args).tolist(),
        "scatter_alphas": rma.parse_scatter_alphas(args.scatter_alphas),
        "test_mode": bool(args.test_mode),
        "test_max_targets": int(args.test_max_targets),
        "test_max_records": int(args.test_max_records),
        "force_recompute_scores": bool(args.force_recompute_scores),
        "auto_prepare": bool(args.auto_prepare),
        "prescott_python": str(args.prescott_python),
        "prepare_args": args.prepare_args,
        "jet_args": args.jet_args,
        "escott_args": args.escott_args,
        "scores_dir": str(args.scores_dir),
        "inputs_dir": str(args.inputs_dir),
        "escott_workdir": str(args.escott_workdir),
        "prescott_ref_dir": str(args.prescott_ref_dir),
        "targets": target_specs,
        "design_signature": signature,
        "design_key": design_key(signature),
        "PRESCOTT_CACHE_VERSION": PRESCOTT_CACHE_VERSION,
    }
    with (Path(output_dir) / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, default=str)


def write_score_scale_report(combined_df: pd.DataFrame, tables_dir: Path, args: argparse.Namespace) -> None:
    """Per (model, lineage) spread of log(score) against log(mut_prob).

    alpha is not scale-free: the sweep optimises log(score) + alpha*log(mut_prob), so an
    alpha from this run and an alpha from a PLM run only mean the same trade-off if the
    two score spreads match. This table is what makes that comparison auditable.
    """
    if combined_df.empty:
        return

    plm_reference_sd = np.nan
    if args.plm_reference_table is not None and Path(args.plm_reference_table).exists():
        reference = pd.read_csv(args.plm_reference_table, usecols=["plm_prob"])
        reference_values = np.log(pd.to_numeric(reference["plm_prob"], errors="coerce").clip(lower=1e-32))
        plm_reference_sd = float(np.nanstd(reference_values.to_numpy()))

    rows: List[Dict[str, object]] = []
    for (model, lineage), frame in combined_df.groupby(["model", "lineage"], sort=False):
        log_score = np.log(pd.to_numeric(frame["plm_prob"], errors="coerce").clip(lower=1e-32).to_numpy())
        log_mut = np.log(pd.to_numeric(frame["mut_prob"], errors="coerce").clip(lower=1e-32).to_numpy())
        sd_score = float(np.nanstd(log_score))
        sd_mut = float(np.nanstd(log_mut))
        # A softmaxed all-zero ESCOTT column is exactly uniform (1/20 everywhere); those
        # are the positions where the JET2 weight was zero and the score carries no rank
        # information. Counting them is the honest way to report surrogate coverage.
        flat_sites = int(
            frame.groupby("position")["plm_prob"].nunique().eq(1).sum()
        ) if "position" in frame.columns else 0
        rows.append({
            "model": model,
            "lineage": lineage,
            "n_rows": int(len(frame)),
            "sd_log_score": sd_score,
            "iqr_log_score": _iqr(log_score),
            "sd_log_mut": sd_mut,
            "iqr_log_mut": _iqr(log_mut),
            "ratio_sd": sd_score / sd_mut if sd_mut else np.nan,
            "n_flat_sites": flat_sites,
            "escott_temperature": float(args.escott_temperature),
            "escott_temperature_mode": args.escott_temperature_mode,
            "sd_log_plm_reference": plm_reference_sd,
            "alpha_rescale": (plm_reference_sd / sd_score) if (np.isfinite(plm_reference_sd) and sd_score) else np.nan,
        })

    out_dir = rma.ensure_dir(Path(tables_dir) / "diagnostics")
    pd.DataFrame(rows).to_csv(out_dir / "score_scale_report.tsv", sep="\t", index=False)


CAVEATS_TEMPLATE = """# Caveats for this ESCOTT/PRESCOTT diversity run

Generated by scripts/run_prescott_diversity.py from the arguments actually used, so it
cannot drift from the run it describes.

Output directory : {output_dir}
Parent map       : {parent_map} (preset: {preset})
Sensitivity edges: {sensitivity_edges}
Input-only       : {input_only}
Score variants   : {variants}
Structure        : {structure} (role: {structure_role}, coverage {structure_coverage})
Deep MSA source  : {deep_fasta}
Temperature      : T = {temperature} (mode: {temperature_mode})
Frequency cutoff : mode={frequency_cutoff_mode}, k={frequency_cutoff_k}, fixed={frequency_cutoff}
Reversion guard  : --parent-freq-max {parent_freq_max}, drop_parent_reversions={drop_parent_reversions}
JET surrogate    : weight={weight_mode}, pc={pc_mode}, sasa={sasa_context}, cv_radius={cv_radius},
                   trace={trace_definition} (B={trace_bootstraps}, top={trace_top_fraction}), seed={seed}
Zero-trace sites : {zero_trace}
Leakage purge    : {leakage_headline}

1.  The jet.res table is a SURROGATE, not JET2. JET2 and naccess are unavailable here.
    `trace` is a Henikoff-weighted, occupancy-scaled Kullback-Leibler divergence to a
    Robinson-Robinson background, bootstrap-decomposed into tr/freq; real T_JET is a
    tree-based Lichtarge evolutionary trace over BLAST-retrieved subsets. Ours is
    phylogeny-free, so positions conserved through shared ancestry rather than
    constraint are over-scored. `pc` is 0.5*(interface propensity + relative SASA) from
    freesasa, not JET2's trained interface-patch propensity. `cv` follows JET2's own
    definition (Rc = {cv_radius} A, 1 = buried) and is the one faithful column.
    Surrogate-vs-real-JET2 check: {jet_validation_status}
    Per-lineage surrogate quality (including the zero-trace count, which is what
    decides how many sites carry any rank information at all):
    tables/diagnostics/jet_surrogate_summary.tsv

2.  escott's --normweightmode is inert: escott.py overwrites it unconditionally in
    main(). We therefore compute the sstjetormaxtwocomponent weight ourselves and write
    it into the `trace` column, running escott without --pdbfile so computePred.R's tjet
    branch reads it directly. Behaviourally identical to escott's structural mode, but a
    non-obvious route through the code that must be disclosed.

3.  Structural coverage. THIS run scored against {structure} in the
    '{structure_role}' role, covering {structure_coverage} HA0 positions.
    6WXB is A/Aichi/2/1968, ~85% identical to the 2024-26
    references, and covers 485 of 566 positions. Uncovered: the signal peptide plus
    HA1 1-8, the HA0 cleavage loop, and HA2 173-221 including the TM helix and
    cytoplasmic tail. Those positions fall back to pure `trace`. Several of the
    positions that vary across these lineages sit in antigenic sites A/B/D whose loop
    conformation and glycosylation differ between 1968 and 2025. Rerun with
    --structure-role extra (the contemporary J.2.4.1 model prepare_inputs.py already
    builds alongside 6WXB) for 100% coverage on a contemporary sequence, and report
    both. Whichever role is selected, structure_source_path/structure_monomer_path in
    run_manifest.json name the file the surrogate actually read.

4.  The frequency cutoff was recalibrated away from PRESCOTT's published Fc = -4.0,
    which is indefensible for GISAID panels spanning 229 to 27452 sequences. With
    Fc = log10(k / median depth) the v2 penalty becomes c*log_N(count): zero for a
    singleton, exactly c for a fixed variant, independent of panel depth. Check
    n_clipped_to_zero in scores/score_variants.tsv; grid points above ~5% have degraded
    rank information.

5.  Parent-lineage assignment. The project brief specified K <- J.2_int. On-disk
    evidence says K is J.2.4.1, a child of J.2.4 (K.nt.fa's own header carries the
    clade call J.2.4.1). The default preset is clade_evidence (K <- J.2.4);
    --parent-map-preset brief_as_stated restores the brief. This run used:
    {parent_map}
    --parent-sensitivity: {sensitivity_status}
    A sensitivity model carries a _parent<TOK> suffix in the `model` column; the
    primary models do not. The two are NOT comparable at equal alpha without care:
    each edge gets its own Fc from its own panel depth (K <- J.2.4 is Fc -2.943 at
    depth 877, K <- J.2_int is Fc -4.439 at depth 27452), which is deliberate -- a
    shared Fc would compare two models under two different penalty scales.

6.  alpha is NOT on the same scale as a PLM run's alpha. The combined score is
    E/T + alpha*log(mut_prob) + const, so alpha_ESCOTT ~ alpha_PLM * sd(log plm_prob) /
    sd(E/T). tables/diagnostics/score_scale_report.tsv gives the ratio per lineage;
    --escott-temperature-mode match-plm equalises it. Never compare a best-alpha value
    across the two pipelines without applying that rescale.

7.  prescott.py's own output is quantised to 2 decimal places, collapsing ~15k mutations
    to ~97 distinct values and crippling Spearman. We recompute PRESCOTT at full
    precision and use prescott.py only as a parity check: {parity_status}

8.  Test-set leakage was ruled out by ALIGNMENT, not by date and not by identifier.
    The deep set has no date column -- the April-2024 cap rests on its filename -- and
    it cannot be checked by ID: the deep set uses protein accessions (QBM69670) and the
    panels nucleotide accessions (OQ233153), so the same physical sequence appears
    under two unrelated IDs. Hashing does not work either: the deep set retains the
    16-residue HA signal peptide and the GISAID panels start at the mature N-terminus,
    so a genuine duplicate hashes differently (measured: zero hash collisions between
    the deep set and any panel, at full length and at every C-terminal suffix tried).
    scripts/prescott_iav/leakage_check.py therefore BLASTs the deep set against every
    evaluation panel and PURGES the near neighbours before ESCOTT sees the alignment.

    Status for this run: {leakage_status}
    Rule            : {leakage_rule}
    Per target      : {leakage_per_target}
    Parent vs target: {leakage_parent_status}
    Audit trail     : {leakage_report}

    Three things about the purge the reader must know:
      * The two thresholds are combined with OR. They are NOT equivalent: on a ~550 aa
        mature HA, 10 mismatches is ~98.2% identity, so at the defaults the Hamming
        rule is the STRICTER one and is what actually governs removal.
      * The purge is PER TARGET, so each lineage's inputs/msa/msa_<key>.fasta is a
        different alignment and ESCOTT must be run once per evaluation target. It
        already was (the query is GEMME's epistatic reference), but the file may now
        never be shared between lineages for a second, stronger reason.
      * The lineage query (row 0) is EXEMPT and asserted to survive. It is ~identical
        to its own target panel by construction -- the manifest records that it would
        otherwise have been purged -- and removing it would have ESCOTT score a
        different protein than the one being evaluated.
    The parent panel is NOT purged: it is the declared frequency input, not a leak.
    Purging the deep set against the target does incidentally remove close parent
    relatives, which is correct.

9.  mkdssp is version 4, not the 3.0.0 named in prescott.yml. v4 emits the polyproline-II
    code `P`, which v3 never did; escott tests ss == 'C', so `P` counts as non-coil and
    fragments coil runs. Small but non-zero. Do not claim bit-parity with published
    PRESCOTT numbers.

10. The evaluation is prospective within a fixed snapshot, not out-of-sample in time.
    Parent and child panels both come from the same Feb-2026 GISAID snapshot and the
    parent panel is not truncated to sequences collected before the child emerged.
    Two guards remove the most flagrant artefact -- the ancestral residue at
    lineage-defining sites reading as 'highly tolerated' in the parent:
      * drop_parent_reversions = {drop_parent_reversions}. This is the one that
        actually works: it drops any mutant whose mutant residue IS the parent
        reference's residue at that site, regardless of frequency. K's N160S sits at
        0.932 and K176I at 0.897, so the frequency threshold alone misses both.
      * --parent-freq-max {parent_freq_max}, a blunt frequency ceiling on top.
    Even so, this measures clade-relative diversity prediction, not forecasting.
"""


def write_caveats(
    args: argparse.Namespace,
    output_dir: Path,
    parent_map: Dict[str, str],
    specs: List[Dict[str, object]],
    evaluable: Sequence[str],
    jet_manifests: Optional[Dict[str, Dict[str, object]]] = None,
) -> str:
    """Render CAVEATS.md, citing only files this run actually produced.

    Every path in the document is checked against the filesystem before it is named.
    The one thing worse than an undocumented limitation is a document that points at a
    file which does not exist, because the reader concludes the check was never done.
    """
    output_dir = Path(output_dir)
    diagnostics_dir = output_dir / "tables" / "diagnostics"
    inputs_manifest = read_inputs_manifest(args.inputs_dir)
    structure_record = resolve_structure_record(args, inputs_manifest)

    coverage = structure_record.get("structure_n_covered")
    coverage_text = "unknown (stage 1 has not run)" if coverage is None else f"{coverage}/566"

    validation_path = diagnostics_dir / JET_VALIDATION_BASENAME
    if validation_path.exists():
        jet_validation_status = f"tables/diagnostics/{JET_VALIDATION_BASENAME}"
    elif not args.jet_validation:
        jet_validation_status = (
            "NOT RUN (--no-jet-validation). Rerun jet_surrogate.py --validate-only "
            "--out-validation <path> to produce it."
        )
    else:
        jet_validation_status = (
            "NOT PRESENT -- stage 1 did not run in this pass (every score matrix was "
            "cached). The table from the pass that built them, if any, is at "
            f"tables/diagnostics/{JET_VALIDATION_BASENAME}."
        )

    parity_path = diagnostics_dir / "prescott_parity_check.tsv"
    if parity_path.exists():
        parity_status = "tables/diagnostics/prescott_parity_check.tsv"
    elif args.test_mode:
        parity_status = "SKIPPED in --test-mode (the reference prescott.py run is the slow part)."
    else:
        parity_status = (
            "not present in this output tree -- stage 1 did not run in this pass, or "
            "--prescott-ref-dir produced nothing."
        )

    edges = effective_sensitivity_edges(args, parent_map, evaluable)
    sensitivity_models = [
        str(spec["model_tag"]) for spec in specs
        if str(spec["model_tag"]).rsplit("_", 1)[-1].startswith("parent")
    ]
    if not args.parent_sensitivity:
        sensitivity_status = "OFF (--no-parent-sensitivity); only the primary parent was scored."
    elif not edges:
        sensitivity_status = (
            "ON, but the presets agree on every lineage evaluated here "
            f"({sorted(evaluable)}), so a sensitivity variant would be a byte-identical "
            "duplicate and none was produced. The contested edge is "
            f"{json.dumps(sensitivity_edges(args, parent_map), sort_keys=True)}; run a "
            "pass that evaluates that lineage to test it."
        )
    elif sensitivity_models:
        sensitivity_status = (
            f"ON. Alternate edges {json.dumps(edges, sort_keys=True)} were scored as "
            f"separate model rows: {', '.join(sensitivity_models)}"
        )
    else:
        sensitivity_status = (
            f"ON and edges {json.dumps(edges, sort_keys=True)} were requested, but no "
            "_parent-suffixed model reached the tables -- treat the parent choice as UNTESTED "
            "in this run."
        )

    zero_trace_bits = []
    for lineage, manifest in sorted((jet_manifests or {}).items()):
        n_zero = manifest.get("n_zero_trace_columns")
        frac = manifest.get("frac_zero_trace_columns")
        if n_zero is None:
            continue
        zero_trace_bits.append(f"{lineage} {n_zero} ({float(frac or 0.0):.1%})")
    zero_trace = "; ".join(zero_trace_bits) if zero_trace_bits else "not recorded (stage 1 not run this pass)"

    # ---- leakage (caveat 8) ------------------------------------------------------- #
    # Rendered from the stage-1 record, and every path is existence-checked before it is
    # named. A caveat that cites a missing audit trail reads as "the check was never
    # done", which is worse than saying so plainly.
    leak = leakage_manifest_record(args, inputs_manifest)
    if not leak.get("leakage_stage_ran"):
        leakage_headline = "NOT AUDITED -- no leakage record in inputs_manifest.json"
        leakage_status = str(leak.get("leakage_note"))
        leakage_rule = "n/a"
        leakage_per_target_text = "n/a"
        leakage_parent_status = "n/a"
        leakage_report = "none"
    else:
        thresholds = leak.get("leakage_thresholds") or {}
        leakage_rule = (
            f"drop when coverage >= {thresholds.get('min_coverage')}% "
            f"({thresholds.get('coverage_basis')} basis) AND "
            f"(identity >= {thresholds.get('min_identity')} OR "
            f"hamming <= {thresholds.get('max_hamming')})"
        )
        per_target = leak.get("leakage_per_target") or {}
        if not per_target and not leak.get("leakage_purge_applied"):
            leakage_per_target_text = (
                "PURGE OFF (--no-purge-leakage): detection only, nothing was removed. "
                "Any hit reported in the audit trail is still in the alignment ESCOTT scored."
            )
        elif not per_target:
            leakage_per_target_text = "no evaluation target was purged in this pass"
        else:
            leakage_per_target_text = "; ".join(
                f"{target}: {entry.get('depth_before')} -> {entry.get('depth_after')} "
                f"({entry.get('n_removed')} removed, "
                f"{float(entry.get('removed_fraction') or 0.0):.2%}"
                + (f", max removed identity {entry.get('removed_identity_max')}%"
                   if entry.get("n_removed") else "")
                + ")"
                for target, entry in sorted(per_target.items())
            )
        parent_checks = leak.get("leakage_parent_vs_target") or {}
        if not parent_checks:
            leakage_parent_status = "not run (--no-leakage-check)"
        else:
            leakage_parent_status = "; ".join(
                f"{entry.get('parent')} -> {target}: "
                f"{entry.get('n_shared_accessions')} shared accessions, "
                f"{entry.get('n_shared_exact_sequences')} shared exact sequences"
                for target, entry in sorted(parent_checks.items())
            )
        report_dir = leak.get("leakage_report_dir")
        leakage_report = (
            str(report_dir) if report_dir and Path(str(report_dir)).exists()
            else "not present in this output tree"
        )
        failures = leak.get("leakage_failures") or []
        leakage_status = (
            f"{leak.get('leakage_status')}"
            + (f" -- {len(failures)} gate(s) failed: " + "; ".join(str(f) for f in failures)
               if failures else " (no residual leakage above the configured gates)")
        )
        n_removed_total = sum(int(entry.get("n_removed") or 0) for entry in per_target.values())
        leakage_headline = (
            f"{leak.get('leakage_status')} -- "
            f"{'purge ON' if leak.get('leakage_purge_applied') else 'DETECTION ONLY (purge off)'}, "
            f"{n_removed_total} deep-set sequences removed across "
            f"{len(per_target)} target(s)"
        )

    text = CAVEATS_TEMPLATE.format(
        output_dir=output_dir,
        parent_map=json.dumps(parent_map, sort_keys=True),
        preset=args.parent_map_preset,
        sensitivity_edges=json.dumps(edges, sort_keys=True) if edges else "none applicable to " + str(sorted(evaluable)),
        input_only=sorted(input_only_lineages()),
        variants=", ".join(str(spec["model_tag"]) for spec in specs),
        structure=structure_record.get("structure_source_path") or args.structure,
        structure_role=args.structure_role,
        structure_coverage=coverage_text,
        deep_fasta=args.deep_fasta,
        temperature=args.escott_temperature,
        temperature_mode=args.escott_temperature_mode,
        frequency_cutoff_mode=args.frequency_cutoff_mode,
        frequency_cutoff_k=args.frequency_cutoff_k,
        frequency_cutoff=args.frequency_cutoff,
        weight_mode=args.weight_mode,
        pc_mode=args.pc_mode,
        sasa_context=args.sasa_context,
        cv_radius=args.cv_radius,
        trace_definition=args.trace_definition,
        trace_bootstraps=args.trace_bootstraps,
        trace_top_fraction=(
            args.trace_top_fraction if args.trace_top_fraction is not None
            else f"{default_trace_top_fraction()} (jet_surrogate.py default)"
        ),
        zero_trace=zero_trace,
        seed=args.seed,
        parent_freq_max=args.parent_freq_max,
        drop_parent_reversions=(
            (inputs_manifest.get("args") or {}).get("drop_parent_reversions",
                                                    bool(args.drop_parent_reversions))
            if inputs_manifest else bool(args.drop_parent_reversions)
        ),
        jet_validation_status=jet_validation_status,
        parity_status=parity_status,
        sensitivity_status=sensitivity_status,
        leakage_headline=leakage_headline,
        leakage_status=leakage_status,
        leakage_rule=leakage_rule,
        leakage_per_target=leakage_per_target_text,
        leakage_parent_status=leakage_parent_status,
        leakage_report=leakage_report,
    )
    (output_dir / "CAVEATS.md").write_text(text, encoding="utf-8")
    return text


# --------------------------------------------------------------------------------------
# Caching
# --------------------------------------------------------------------------------------

def model_cache_is_valid(
    existing_panel_metadata_df: pd.DataFrame,
    args: argparse.Namespace,
    spec: Dict[str, object],
    parent_map: Dict[str, str],
    model_tables_dir: Path,
) -> bool:
    """Per-MODEL cache guard: may this one model's tables be reused as they stand?

    The PLM driver resumes model by model (``_load_cached_model_outputs`` per spec), so
    a run killed after variant 3 of 4 does not repeat three alpha sweeps. This driver
    previously gated every model behind one all-or-nothing whole-run check, which threw
    that away. Restoring per-model resumption is safe only if the design checks that
    used to be whole-run are applied per model too -- hence design_key, plus the
    parent/temperature/cache-version checks, evaluated against this model's own rows.
    """
    if args.force_recompute_scores or args.diagnostic_exports or existing_panel_metadata_df.empty:
        return False

    model_label = str(spec["model_tag"])
    frame = existing_panel_metadata_df
    if "model" not in frame.columns:
        return False
    rows = frame.loc[frame["model"].astype(str) == model_label]
    if rows.empty:
        return False

    if "cache_version" not in rows.columns:
        return False
    if not pd.to_numeric(rows["cache_version"], errors="coerce").eq(PRESCOTT_CACHE_VERSION).all():
        return False
    # design_key covers this model's equation/coefficient/k/parent plus the frequency
    # cutoff, temperature, alpha grid and panel filters in one comparison. Tables
    # written before it existed have no column and are treated as stale, which is the
    # safe direction: recomputing is expensive, reporting the wrong design is worse.
    if "design_key" not in rows.columns:
        return False
    if not rows["design_key"].astype(str).eq(model_design_key(args, spec, parent_map)).all():
        return False
    if "mutation_model" in rows.columns and not rows["mutation_model"].astype(str).eq(args.mutation_model).all():
        return False
    if "escott_temperature" in rows.columns:
        if not np.allclose(
            pd.to_numeric(rows["escott_temperature"], errors="coerce").fillna(-1).to_numpy(),
            float(args.escott_temperature),
        ):
            return False
    if "parent_lineage" not in rows.columns:
        return False
    for _, row in rows.iterrows():
        lineage = str(row.get("lineage"))
        expected = spec["parent_by_lineage"].get(lineage, parent_map.get(lineage))
        # Normalised on BOTH sides. The ESCOTT baseline is conditioned on no parent, so
        # its cell is empty and pandas reads it back as NaN; a naive str() comparison
        # made 'nan' != 'None' and quietly recomputed the ESCOTT alpha sweep on every
        # single rerun -- the exact per-model resumption this guard exists to allow.
        if _normalised_label(row.get("parent_lineage")) != _normalised_label(expected):
            return False

    return rma._load_cached_model_outputs(model_tables_dir, spec) is not None


def prescott_cache_is_valid(
    existing_panel_metadata_df: pd.DataFrame,
    args: argparse.Namespace,
    specs: List[Dict[str, object]],
    parent_map: Dict[str, str],
    model_tables_dir: Path,
) -> bool:
    """Whole-run short circuit: is EVERY model cached and valid?

    Only this answer may skip building the lineage cache, because that is the step that
    parses and aligns the 27452- and 17898-sequence GISAID panels. It is deliberately
    just the conjunction of the per-model checks, so the two can never disagree.
    """
    if not specs:
        return False
    return all(
        model_cache_is_valid(existing_panel_metadata_df, args, spec, parent_map, model_tables_dir)
        for spec in specs
    )


# --------------------------------------------------------------------------------------
# The analysis
# --------------------------------------------------------------------------------------

def resolve_targets(args: argparse.Namespace) -> List[Dict[str, str]]:
    """Guide/single-FASTA targets, without the test-mode truncation applied yet."""
    from Functions_HuggingFace import load_analysis_targets

    return load_analysis_targets(
        analysis_mode=args.analysis_mode,
        guide_path=str(args.guide_path) if args.guide_path else None,
        diversity_fasta=str(args.diversity_fasta) if args.diversity_fasta else None,
        reference_fasta=str(args.reference_fasta) if args.reference_fasta else None,
        default_label=args.label,
        test_mode=False,
        test_max_targets=1,
    )


def resolve_test_target_count(
    all_targets: Sequence[Dict[str, str]],
    requested: int,
    skip: Iterable[str],
) -> int:
    """Smallest leading slice of the guide that contains an evaluable lineage.

    Guide row 1 is G.1, which is input-only, so --test-max-targets 1 would resolve to a
    run with nothing to score. Growing the slice keeps the smoke test honest: it lands on
    J_int (4132 target sequences) against parent G.1 (229), the cheapest real pair.
    """
    skip = set(skip)
    count = max(1, int(requested))
    while count <= len(all_targets):
        if any(str(target["label"]) not in skip for target in all_targets[:count]):
            return count
        count += 1
    return len(all_targets)


def run_analysis(args: argparse.Namespace) -> int:
    from Functions_HuggingFace import build_codon_aa_mutation_tables

    args = apply_prescott_defaults(args)
    output_dir = rma.ensure_dir(args.output_dir)
    group_dir = rma.ensure_dir(output_dir / "groups")
    scores_dir = rma.ensure_dir(args.scores_dir)
    tables_dir = rma.ensure_dir(output_dir / "tables")
    plots_dir = rma.ensure_dir(output_dir / "plots")
    model_tables_dir = rma.ensure_dir(tables_dir / "per_model")

    existing_panel_metadata_path = tables_dir / "panel_metadata.tsv"
    existing_panel_metadata_df = (
        pd.read_csv(existing_panel_metadata_path, sep="\t")
        if existing_panel_metadata_path.exists() else pd.DataFrame()
    )

    # --- figures-only path: the schemas are identical to the PLM run's, so reuse it ------
    if args.regen_figures_only:
        return rma._regenerate_figures_from_existing_tables(
            args,
            tables_dir=tables_dir,
            plots_dir=plots_dir,
            existing_panel_metadata_df=existing_panel_metadata_df,
        )

    parent_map = resolve_parent_map(args)
    skip_labels = input_only_lineages()

    all_targets = resolve_targets(args)
    if not all_targets:
        raise RuntimeError(f"No targets resolved from {args.guide_path}")
    if args.test_mode:
        args.test_max_targets = resolve_test_target_count(all_targets, args.test_max_targets, skip_labels)
        print(f"[test-mode] using the first {args.test_max_targets} guide row(s) so at least one "
              f"evaluable lineage is present (input-only: {sorted(skip_labels)})")

    considered = [str(t["label"]) for t in all_targets[: args.test_max_targets]] if args.test_mode \
        else [str(t["label"]) for t in all_targets]
    evaluable = [label for label in considered if label not in skip_labels]
    missing_parents = [label for label in evaluable if label not in parent_map]
    if missing_parents:
        raise ValueError(
            f"No basal lineage defined for {missing_parents}. Add an edge with --parent-map "
            f"child=parent, or list them as input-only."
        )
    if not evaluable:
        raise RuntimeError(
            f"Every resolved target is input-only ({considered}); there is nothing to score."
        )
    print(f"Evaluable lineages: {evaluable}")
    print(f"Parent map in use : { {k: parent_map[k] for k in evaluable} }")

    # --- stage 1: build the ESCOTT/PRESCOTT score matrices --------------------------------
    # Planned BEFORE the lineage cache is built, not after. Two reasons, both learned the
    # hard way: (a) stage 1 is what *names* the variants, so specs built from guessed
    # names would not survive a mid-loop stage-1 invocation that chose different ones;
    # (b) on a fully cached rerun this lets the whole-run cache decision be taken before
    # build_lineage_cache parses and aligns the 27452- and 17898-sequence GISAID panels,
    # which is tens of minutes of work the PLM driver has always skipped and this one
    # used to redo every time.
    diagnostics_dir = tables_dir / "diagnostics"
    requested_plan = expected_variant_plan(args, parent_map, evaluable)
    variants_table = load_score_variants_table(scores_dir)
    planned, missing, ignored = reconcile_variant_plan(
        requested_plan, variants_table, scores_dir, evaluable
    )
    rerun_reason = f"{len(missing)} requested score matrix/matrices not available" if missing else None
    if args.force_recompute_scores:
        rerun_reason = "--force-recompute-scores"
    if ignored:
        # Loud, because this is the direction the old code got wrong silently: a cached
        # table that is a superset of the requested design must NOT be analysed.
        print(
            f"Ignoring {len(ignored)} cached score variant(s) outside the requested design "
            f"(they stay on disk but are not analysed): {ignored}"
        )

    # match-plm needs the raw ESCOTT values, which only exist after a first pass, so it is
    # inherently two-pass. Resolve T here (not inside the stage-1 branch) so that matrices
    # cached at a different temperature are detected and rebuilt rather than silently
    # reused under a manifest claiming the new T.
    if not args.dry_run and args.escott_temperature_mode == "match-plm":
        args.escott_temperature = resolve_escott_temperature(args, scores_dir, evaluable)
        if not variants_table.empty and "temperature" in variants_table.columns:
            cached_temperatures = pd.to_numeric(variants_table["temperature"], errors="coerce")
            if not np.allclose(cached_temperatures.dropna().to_numpy(), args.escott_temperature):
                rerun_reason = "temperature changed under --escott-temperature-mode match-plm"
                args.force_recompute_scores = True

    if not args.dry_run and rerun_reason:
        if args.auto_prepare:
            print(f"Stage 1 needed ({rerun_reason}); running it.")
            if missing:
                for item in missing[:10]:
                    print(f"  needs: {item}")
                if len(missing) > 10:
                    print(f"  ... and {len(missing) - 10} more")
            run_stage1(args, parent_map, evaluable, diagnostics_dir)
            variants_table = load_score_variants_table(scores_dir)
            planned, missing, ignored = reconcile_variant_plan(
                requested_plan, variants_table, scores_dir, evaluable
            )
            if missing:
                # Never silently analyse a different design than the one requested.
                raise RuntimeError(
                    "Stage 1 ran but the requested design is still incomplete; refusing to "
                    "report a run whose manifest would not describe its outputs.\n  "
                    + "\n  ".join(missing[:20])
                    + (f"\n  ... and {len(missing) - 20} more" if len(missing) > 20 else "")
                    + "\n  Check the run_escott output above: an alternate-parent variant "
                      "needs prepare_inputs.py --sensitivity-parent-map, and equation 4 does "
                      "not exist upstream."
                )
        else:
            raise FileNotFoundError(
                f"Stage 1 is needed ({rerun_reason}) but --no-auto-prepare was set. "
                + (f"First missing: {missing[0]}" if missing else "")
            )

    specs = build_score_specs(args, planned, parent_map)
    print(f"Models            : {[str(spec['model_tag']) for spec in specs]}")

    # --- inputs: identical code path to the PLM run ---------------------------------------
    # ...unless every per-model table is already cached and the design key matches, in
    # which case rma's lightweight metadata-derived cache is enough, exactly as SC2 does.
    use_cached_outputs_only = (
        not args.dry_run
        and prescott_cache_is_valid(
            existing_panel_metadata_df, args, specs, parent_map, model_tables_dir
        )
    )
    if use_cached_outputs_only:
        print("All per-variant tables are cached and the design key matches; reusing them "
              "without re-parsing the diversity panels.")
        mutation_tables = None
        lineage_cache = rma._build_lightweight_lineage_cache_from_metadata(existing_panel_metadata_df)
        evaluation_cache = {
            label: data for label, data in lineage_cache.items() if label in set(evaluable)
        }
    else:
        mutation_tables = build_codon_aa_mutation_tables(args.mutation_model)
        lineage_cache = rma.build_lineage_cache(args, mutation_tables)
        if not lineage_cache:
            raise RuntimeError("No valid targets were resolved for this run")

        # Parents stay in the cache only if they are also evaluation targets; an input-only
        # lineage would otherwise appear in the figures as an empty panel.
        evaluation_cache = {
            label: data for label, data in lineage_cache.items() if label in set(evaluable)
        }
        for label, data in lineage_cache.items():
            role = "target" if label in evaluation_cache else "input-only"
            print(
                f"  {label:<10s} [{role}] n_seq={len(data['records']):>6d} "
                f"ref_len={len(data['full_ref_protein'])} "
                f"mapped_ref_sites={data['alignment_diff_stats']['mapped_sites']} "
                f"differing_sites={data['alignment_diff_stats']['differing_sites']}"
            )

    if not evaluation_cache:
        raise RuntimeError(
            f"None of the evaluable lineages {evaluable} produced a usable panel; "
            "check the guide FASTA paths."
        )

    # Surrogate quality (zero-trace counts above all) into the output tree, so a bad
    # --trace-top-fraction is visible in a table instead of only in a stage-1 log.
    jet_manifests = read_jet_manifests(args.inputs_dir, read_inputs_manifest(args.inputs_dir),
                                       evaluable, args.structure_role)
    write_jet_surrogate_summary(jet_manifests, diagnostics_dir)

    target_specs = [
        {"label": label, "diversity_path": data.get("diversity_path", ""),
         "reference_path": data.get("reference_path", "")}
        for label, data in lineage_cache.items()
    ]
    save_run_manifest(args, output_dir, target_specs, parent_map, specs, variants_table,
                      evaluable, jet_manifests)
    caveats = write_caveats(args, output_dir, parent_map, specs, evaluable, jet_manifests)

    if args.dry_run:
        # The profiles are already in memory and are the expensive part, so write them:
        # that turns --dry-run into a usable "build the inputs" command and makes the
        # check verifiable (e.g. reference positions 1-16 are the signal peptide, absent
        # from the mature-HA panels, and must come out at depth 0).
        for label, data in evaluation_cache.items():
            data["mut_profile"].to_csv(
                group_dir / f"{data['lineage_key']}_mutation_accessibility_profile.csv"
            )
            data["obs_freq"].to_csv(
                group_dir / f"{data['lineage_key']}_observed_diversity_profile.csv"
            )
            pd.Series(data["obs_depth"], name="depth").rename_axis("position").to_csv(
                group_dir / f"{data['lineage_key']}_observed_depth_profile.csv"
            )
        print("\n--dry-run: inputs resolved and observed-diversity profiles built; "
              "stopping before ESCOTT scoring.")
        print(f"  groups/           -> {group_dir}")
        print(f"  run_manifest.json -> {output_dir / 'run_manifest.json'}")
        print(f"  CAVEATS.md        -> {output_dir / 'CAVEATS.md'}")
        return 0

    if args.diagnostic_exports and mutation_tables is not None:
        rma.export_codon_model_diagnostics(
            tables_dir / "diagnostics" / str(args.mutation_model).lower(), mutation_tables
        )

    # --- per-variant scoring and sweep ------------------------------------------------------
    metadata_rows: List[Dict[str, object]] = []
    status_rows: List[Dict[str, object]] = []
    all_combined_frames: List[pd.DataFrame] = []
    all_alpha_frames: List[pd.DataFrame] = []
    all_alpha_lineage_frames: List[pd.DataFrame] = []
    best_rows: List[Dict[str, object]] = []
    per_group_best_rows: List[Dict[str, object]] = []
    # Read once: the jet md5 goes into every metadata row and this file does not change
    # under us mid-run.
    inputs_manifest = read_inputs_manifest(args.inputs_dir)

    alpha_grid = rma.parse_alpha_grid(args)
    use_parallel = args.alpha_parallel and len(alpha_grid) >= args.alpha_sweep_min_grid

    for spec in specs:
        model_label = str(spec["model_tag"])
        spec_lineages = [label for label in spec["lineages"] if label in evaluation_cache]
        model_combined_df = pd.DataFrame()
        alpha_df = pd.DataFrame()
        alpha_by_lineage_df = pd.DataFrame()

        # PER-MODEL resumption, as the PLM driver does. The whole-run gate above only
        # decides whether the diversity panels have to be parsed at all; a model whose
        # own tables are cached and whose own design key matches is reused even when a
        # sibling model has to be recomputed, so a run killed after variant 3 of 4 does
        # not repeat three ~6-minute alpha sweeps. Only read the cached CSVs when we are
        # allowed to use them -- otherwise this loads a multi-megabyte table per variant
        # just to throw it away.
        cached_outputs = (
            rma._load_cached_model_outputs(model_tables_dir, spec)
            if model_cache_is_valid(existing_panel_metadata_df, args, spec, parent_map, model_tables_dir)
            else None
        )
        if cached_outputs is not None:
            print(f"Reusing cached tables for {model_label} "
                  f"(model design key {model_design_key(args, spec, parent_map)}).")
            model_combined_df, alpha_df = cached_outputs
            alpha_by_lineage_path = model_tables_dir / f"{model_label}_alpha_sweep_fit_metrics_BY_LINEAGE.tsv"
            if alpha_by_lineage_path.exists():
                alpha_by_lineage_df = pd.read_csv(alpha_by_lineage_path, sep="\t")
            if (
                alpha_by_lineage_df.empty
                or not rma._alpha_table_has_complete_logistic_metrics(alpha_df)
                or not rma._alpha_table_has_complete_logistic_metrics(alpha_by_lineage_df)
            ):
                alpha_df, alpha_by_lineage_df = rma._build_alpha_tables_from_combined(
                    model_combined_df, alpha_grid,
                    model_label=model_label, model_spec=spec,
                    parallel=use_parallel, max_workers=args.alpha_sweep_max_workers,
                    alpha_sweep_min_grid=args.alpha_sweep_min_grid, pseudocount=1e-16,
                )
                alpha_df = _stamp_score_formula(alpha_df)
                alpha_by_lineage_df = _stamp_score_formula(alpha_by_lineage_df)
                alpha_df.to_csv(model_tables_dir / f"{model_label}_alpha_sweep_fit_metrics.tsv",
                                sep="\t", index=False)
                if not alpha_by_lineage_df.empty:
                    alpha_by_lineage_df.to_csv(alpha_by_lineage_path, sep="\t", index=False)
            rma.warn_on_excess_mutation_rows(
                model_combined_df, context_label=f"cached combined table ({model_label})"
            )
            all_combined_frames.append(model_combined_df)
            all_alpha_frames.append(alpha_df)
            if not alpha_by_lineage_df.empty:
                all_alpha_lineage_frames.append(alpha_by_lineage_df)
            if not existing_panel_metadata_df.empty:
                cached_metadata = existing_panel_metadata_df.loc[
                    existing_panel_metadata_df["model"] == model_label
                ]
                metadata_rows.extend(cached_metadata.to_dict("records"))
            status_rows.append({"model": model_label, "lineage": "all",
                                "status": "completed", "reason": "cached"})
        else:
            model_combined_rows: List[Dict[str, object]] = []
            for lineage_label in spec_lineages:
                lineage_data = evaluation_cache[lineage_label]
                print(
                    f"Processing {model_label} / {lineage_label}: "
                    f"n_seq={len(lineage_data['records'])}, "
                    f"ref_len={len(lineage_data['full_ref_protein'])}, "
                    f"parent={spec['parent_by_lineage'].get(lineage_label, parent_map.get(lineage_label))}"
                )
                try:
                    matrix, score_path, source_sequence = ensure_score_matrix(
                        args, spec, lineage_label, lineage_data, scores_dir,
                    )
                    # A silent frame mismatch (ESCOTT scored a different sequence to the
                    # one the diversity panel was mapped onto) would corrupt every metric
                    # while still producing plausible-looking numbers. Fail loudly.
                    expected_sequence = str(lineage_data["full_ref_protein"])
                    if source_sequence != expected_sequence:
                        raise RuntimeError(
                            f"score matrix {score_path} was built on a different sequence than the "
                            f"lineage reference (matrix {len(source_sequence)} aa vs reference "
                            f"{len(expected_sequence)} aa); refusing to combine them"
                        )

                    rows = rma.build_combined_rows(
                        args, spec, lineage_label, lineage_data, matrix,
                        coord_map=lineage_data["coord_map"],
                    )
                    model_combined_rows.extend(rows)

                    lineage_data["mut_profile"].to_csv(
                        group_dir / f"{lineage_data['lineage_key']}_mutation_accessibility_profile.csv"
                    )
                    lineage_data["obs_freq"].to_csv(
                        group_dir / f"{lineage_data['lineage_key']}_observed_diversity_profile.csv"
                    )
                    if args.diagnostic_exports:
                        rma.export_lineage_diagnostics(
                            args=args,
                            plot_dir=plots_dir / "diagnostics",
                            table_dir=tables_dir / "diagnostics",
                            model_label=model_label,
                            lineage_label=lineage_label,
                            lineage_data=lineage_data,
                            plm_matrix=matrix,
                            coord_map=lineage_data["coord_map"],
                            source_plm_sequence=source_sequence,
                            mutation_tables=mutation_tables,
                            global_to_lineage_trim={i: i for i in lineage_data["coord_map"]},
                            remap_alignment=None,
                        )

                    parent_label = spec["parent_by_lineage"].get(lineage_label, parent_map.get(lineage_label))
                    jet_file = stage1_paths(
                        args.inputs_dir, str(lineage_data["lineage_key"]),
                        inputs_manifest, args.structure_role,
                    )["jet"]
                    metadata_rows.append({
                        "model": model_label,
                        "epoch_label": spec["epoch_label"],
                        "epoch_value": float(spec["epoch_value"]),
                        "mutation_model": args.mutation_model,
                        "lineage": lineage_label,
                        "n_sequences": len(lineage_data["records"]),
                        "reference_length": len(lineage_data["full_ref_protein"]),
                        "mapped_ref_sites": int(lineage_data["alignment_diff_stats"]["mapped_sites"]),
                        "compared_sites_non_gap_non_stop": int(lineage_data["alignment_diff_stats"]["compared_sites"]),
                        "differing_sites_vs_reference_non_gap_non_stop": int(lineage_data["alignment_diff_stats"]["differing_sites"]),
                        "fixed_differing_sites_vs_reference_non_gap_non_stop": int(lineage_data["alignment_diff_stats"]["fixed_differing_sites"]),
                        "diversity_fasta": lineage_data["diversity_path"],
                        "reference_fasta": lineage_data["reference_path"],
                        "plm_profile": score_path,
                        "diversity_sequences_detected_as_nucleotide": bool(lineage_data["any_nucleotide_diversity"]),
                        # ESCOTT-specific keys beyond the PLM run's 17. They are part of the
                        # cache key, which is how a changed design invalidates stale tables.
                        "parent_lineage": parent_label,
                        "frequency_cutoff_mode": args.frequency_cutoff_mode,
                        "escott_temperature": float(args.escott_temperature),
                        "jetfile_md5": file_md5(jet_file),
                        # The share of positions whose ESCOTT column pred.R zeroed out.
                        # Every metric below is computed over the remainder, so this
                        # belongs next to n_sequences, not only in a stage-1 log.
                        "n_zero_trace_columns": (jet_manifests.get(lineage_label) or {}).get(
                            "n_zero_trace_columns"),
                        "frac_zero_trace_columns": (jet_manifests.get(lineage_label) or {}).get(
                            "frac_zero_trace_columns"),
                        "source_variant": str(spec["source_variant_by_lineage"][lineage_label]),
                        "design_key": model_design_key(args, spec, parent_map),
                    })
                except Exception as exc:
                    status_rows.append({"model": model_label, "lineage": lineage_label,
                                        "status": "failed", "reason": str(exc)})
                    print(f"Failed on {model_label} / {lineage_label}: {exc}")

            model_combined_df = pd.DataFrame(model_combined_rows)
            if model_combined_df.empty:
                status_rows.append({"model": model_label, "lineage": "all",
                                    "status": "failed", "reason": "no combined rows produced"})
                continue

            rma.warn_on_excess_mutation_rows(
                model_combined_df, context_label=f"combined table before alpha sweep ({model_label})"
            )
            all_combined_frames.append(model_combined_df)
            model_combined_df.to_csv(model_tables_dir / f"{model_label}_combined_long_table.csv", index=False)

            # One call, not two. The PLM driver runs evaluate_alpha_sweep_by_lineage and then
            # _build_alpha_tables_from_combined, which repeats the same sweep internally --
            # on 21 alpha values over a 27452-sequence panel that doubles the dominant cost
            # for a result it throws away. Test the returned frame for emptiness instead.
            alpha_df, alpha_by_lineage_df = rma._build_alpha_tables_from_combined(
                model_combined_df, alpha_grid,
                model_label=model_label, model_spec=spec,
                parallel=use_parallel, max_workers=args.alpha_sweep_max_workers,
                alpha_sweep_min_grid=args.alpha_sweep_min_grid, pseudocount=1e-16,
            )
            alpha_df = _stamp_score_formula(alpha_df)
            alpha_by_lineage_df = _stamp_score_formula(alpha_by_lineage_df)
            if not alpha_by_lineage_df.empty:
                all_alpha_lineage_frames.append(alpha_by_lineage_df)
                alpha_by_lineage_df.to_csv(
                    model_tables_dir / f"{model_label}_alpha_sweep_fit_metrics_BY_LINEAGE.tsv",
                    sep="\t", index=False,
                )
            alpha_df.to_csv(model_tables_dir / f"{model_label}_alpha_sweep_fit_metrics.tsv",
                            sep="\t", index=False)
            all_alpha_frames.append(alpha_df)

        # --- per-variant epoch summary and best-alpha rows ---------------------------------
        model_epoch_summary_df = rma.summarize_epoch_metrics(
            rma.compute_epoch_lineage_metrics(model_combined_df)
        )
        if not model_epoch_summary_df.empty:
            baseline_cols = [
                "model", "epoch_label", "epoch_value",
                "logistic_site_mutated_vs_mut_corr_baseline",
                "spearman_obs_freq_vs_mut_baseline",
                "pearson_obs_freq_vs_mut_baseline",
                "spearman_mut_vs_mut_baseline",
                "pearson_mut_vs_mut_baseline",
            ]
            present = [col for col in baseline_cols if col in model_epoch_summary_df.columns]
            model_epoch_summary_df.loc[:, present].to_csv(
                model_tables_dir / f"{model_label}_mutation_baseline_summary.tsv", sep="\t", index=False
            )

        if not alpha_df.empty:
            # NB: best_alpha_index excludes the mutation-only baseline row, whose
            # alpha is NaN. Selecting over the whole frame lets that row win and
            # writes best_alpha = NaN. See alpha_sweep_grid_rows.
            for method, criterion, idx, column in (
                ("Method A (Site-level)", "max site_top10pct_mutated_enrichment",
                 best_alpha_index(alpha_df, "site_top10pct_mutated_enrichment"),
                 "site_top10pct_mutated_enrichment"),
                ("Method B (Mutation-level flattened)", "max mut_flat_global_spearman_r",
                 best_alpha_index(alpha_df, "mut_flat_global_spearman_r"),
                 "mut_flat_global_spearman_r"),
            ):
                if idx is None:
                    continue
                best_rows.append({
                    "model": model_label,
                    "epoch_label": spec["epoch_label"],
                    "epoch_value": float(spec["epoch_value"]),
                    "method": method,
                    "criterion": criterion,
                    "best_alpha": float(alpha_df.loc[idx, "alpha"]),
                    "best_value": float(alpha_df.loc[idx, column]),
                })

        if not alpha_by_lineage_df.empty:
            for lineage_name, lineage_alpha in alpha_by_lineage_df.groupby("lineage"):
                if lineage_alpha.empty:
                    continue
                for method, criterion, column in (
                    ("Method A (Site-level)", "max site_top10pct_mutated_enrichment",
                     "site_top10pct_mutated_enrichment"),
                    ("Method B (Mutation-level flattened)", "max mut_flat_global_spearman_r",
                     "mut_flat_global_spearman_r"),
                ):
                    idx = best_alpha_index(lineage_alpha, column)
                    if idx is None:
                        continue
                    per_group_best_rows.append({
                        "model": model_label,
                        "epoch_label": spec["epoch_label"],
                        "epoch_value": float(spec["epoch_value"]),
                        "lineage": lineage_name,
                        "method": method,
                        "criterion": criterion,
                        "best_alpha": float(lineage_alpha.loc[idx, "alpha"]),
                        "best_value": float(lineage_alpha.loc[idx, column]),
                    })

        status_rows.append({"model": model_label, "lineage": "all", "status": "completed", "reason": "ok"})

    if not all_combined_frames:
        raise RuntimeError("No combined rows were produced for any score variant")

    # --- pooled tables --------------------------------------------------------------------
    combined_df = pd.concat(all_combined_frames, ignore_index=True)
    combined_df.to_csv(tables_dir / "combined_long_table.csv", index=False)

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df["cache_version"] = PRESCOTT_CACHE_VERSION
    # NOT blanket-assigned: design_key is per MODEL now, and cached rows already carry
    # the right one (that is precisely why model_cache_is_valid accepted them).
    if "design_key" not in metadata_df.columns:
        # Only reachable if every row came from a cached table written before design_key
        # existed. Leave it blank rather than stamping the current key onto rows we
        # cannot vouch for: a blank key fails model_cache_is_valid next time, which
        # recomputes -- the safe direction.
        metadata_df["design_key"] = pd.NA
    if "mutation_model" not in metadata_df.columns:
        metadata_df["mutation_model"] = args.mutation_model
    metadata_df.to_csv(tables_dir / "panel_metadata.tsv", sep="\t", index=False)
    pd.DataFrame(status_rows).to_csv(tables_dir / "model_run_status.tsv", sep="\t", index=False)

    alpha_df = pd.concat(all_alpha_frames, ignore_index=True) if all_alpha_frames else pd.DataFrame()
    if not alpha_df.empty:
        alpha_df.to_csv(tables_dir / "alpha_sweep_fit_metrics.tsv", sep="\t", index=False)
    alpha_by_lineage_df = (
        pd.concat(all_alpha_lineage_frames, ignore_index=True) if all_alpha_lineage_frames else pd.DataFrame()
    )
    if not alpha_by_lineage_df.empty:
        alpha_by_lineage_df.to_csv(tables_dir / "alpha_sweep_fit_metrics_BY_LINEAGE.tsv", sep="\t", index=False)
    if best_rows:
        pd.DataFrame(best_rows).to_csv(tables_dir / "best_alpha_two_methods.tsv", sep="\t", index=False)
    if per_group_best_rows:
        pd.DataFrame(per_group_best_rows).to_csv(
            tables_dir / "best_alpha_per_group_two_methods.tsv", sep="\t", index=False
        )

    epoch_lineage_metrics_df = rma.compute_epoch_lineage_metrics(combined_df)
    epoch_summary_df = rma.summarize_epoch_metrics(epoch_lineage_metrics_df)
    if not epoch_lineage_metrics_df.empty:
        epoch_lineage_metrics_df.to_csv(tables_dir / "epoch_lineage_metrics.tsv", sep="\t", index=False)
    if not epoch_summary_df.empty:
        epoch_summary_df.to_csv(tables_dir / "epoch_metric_summary.tsv", sep="\t", index=False)

    write_score_scale_report(combined_df, tables_dir, args)

    # Same rule as the PLM driver: the pseudocount tracks panel depth so a zero-frequency
    # mutation plots one decade below the smallest observable frequency.
    dynamic_pseudocount = float(10 ** -round(np.log10(10 * max(1, combined_df["depth"].max()))))
    rma.export_plots(
        output_dir=plots_dir,
        combined_df=combined_df,
        alpha_df=alpha_df,
        epoch_summary_df=epoch_summary_df,
        scatter_alphas=rma.parse_scatter_alphas(args.scatter_alphas),
        scatter_max_points=args.scatter_max_points,
        lineage_cache=evaluation_cache,
        dynamic_pseudocount=dynamic_pseudocount,
        mutation_baseline_x=args.mutation_baseline_x,
        metrics_output_dir=tables_dir,
        mutation_model=args.mutation_model,
    )

    print("\n" + caveats.split("\n\n", 1)[0])
    print(f"\nCaveats written to {output_dir / 'CAVEATS.md'} -- read them before reporting anything.")
    return 0


def alpha_sweep_grid_rows(alpha_frame: pd.DataFrame) -> pd.DataFrame:
    """The sweep rows that carry a real alpha, i.e. everything but the baseline.

    ``rma.compute_mutation_only_alpha_baseline_row`` appends a row with
    ``alpha = np.nan`` / ``alpha_label = "mutation_only"`` /
    ``is_mutation_only_baseline = True``.  It is the ``alpha -> +inf`` limit with
    ``plm_prob`` pinned at 1.0, so it competes on the metric columns but has no
    position on the grid.  Any ``idxmax`` used to pick a *best alpha* must exclude
    it, or a run in which the codon model alone out-ranks every alpha reports
    ``best_alpha = NaN`` -- which reads as a failed fit rather than "the baseline
    won".  Frames without the marker column are returned unchanged so this stays
    safe on hand-made or older tables.
    """
    if alpha_frame.empty or "is_mutation_only_baseline" not in alpha_frame.columns:
        return alpha_frame
    keep = ~alpha_frame["is_mutation_only_baseline"].fillna(False).astype(bool)
    return alpha_frame.loc[keep]


def best_alpha_index(alpha_frame: pd.DataFrame, column: str) -> Optional[object]:
    """Index of the best row of ``column`` among the rows that have a real alpha.

    Returns ``None`` when there is nothing to choose from, so the caller emits no
    best-alpha row at all rather than one whose ``best_alpha`` is NaN.
    """
    grid = alpha_sweep_grid_rows(alpha_frame)
    if grid.empty or column not in grid.columns:
        return None
    values = pd.to_numeric(grid[column], errors="coerce")
    if not values.notna().any():
        return None
    return values.idxmax()


def _stamp_score_formula(alpha_frame: pd.DataFrame) -> pd.DataFrame:
    """Relabel the sweep rows so a concatenated PLM+ESCOTT table stays interpretable.

    evaluate_alpha_sweep_by_lineage hard-codes 'plm_prob * mut_prob^alpha'. The
    mutation-only baseline row keeps its own formula, which is why this only touches
    rows whose model_variant is the sweep itself.
    """
    if alpha_frame.empty or "input_score_formula" not in alpha_frame.columns:
        return alpha_frame
    frame = alpha_frame.copy()
    if "model_variant" in frame.columns:
        mask = frame["model_variant"].astype(str).eq("plm_alpha_sweep")
    else:
        mask = frame["input_score_formula"].astype(str).eq("plm_prob * mut_prob^alpha")
    frame.loc[mask, "input_score_formula"] = INPUT_SCORE_FORMULA
    return frame


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
        return run_analysis(args)
    except Exception as exc:  # noqa: BLE001 - mirror the PLM driver's CLI behaviour
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
