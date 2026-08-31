#!/usr/bin/env python3
"""Generate a JET2-compatible ``<prot>_jet.res`` table without JET2 or naccess.

WHY THIS MODULE EXISTS
----------------------
ESCOTT (``/home3/oml4h/PRESCOTT/prescott/escott.py``) normally shells out to JET2
to produce a per-residue table of evolutionary/structural weights.  JET2 needs a
Java runtime plus ``naccess`` (a licensed binary), neither of which is available
here and neither of which can be installed.  ``escott --jetfile <file>`` skips
the JET2 subprocess entirely and simply copies ``<file>`` to ``<prot>_jet.res``
(escott.py:1112-1113), so if we can synthesise a credible table ourselves the
rest of the ESCOTT/GEMME machinery runs untouched.

This module builds that table from (a) a query-anchored MSA and (b) a PDB
structure, and is deliberately explicit about which columns are faithful
reproductions of JET2 and which are surrogates.

WHAT ESCOTT / GEMME ACTUALLY READ
---------------------------------
Only two consumers exist and both parse by *column name*:

* ``escott.py:1124/1161/1198/1262`` -> ``pd.read_table(prot+"_jet.res", sep="\\s+")``
  and then indexes ``row['trace']``, ``row['pc']``, ``row['cv']``.  These blocks
  only run when ``--pdbfile`` is supplied.
* ``computePred.R:46`` -> ``read.table(..., head=TRUE)`` and then
  ``computePred.R:58-65``: under ``normWeightMode=="tjet"`` it takes column
  ``traceMax`` **if that column exists, else** ``trace``.

Two consequences drive the whole design:

1. ``escott.main()`` (escott.py:1504-1509) **unconditionally overwrites**
   ``--normweightmode``: ``'tjet'`` when ``--pdbfile`` is absent,
   ``'sstjetormaxtwocomponent'`` when it is present.  The user's flag is
   discarded either way.  With ``--pdbfile`` escott re-runs DSSP itself and
   *inner-joins* its DSSP table onto our rows by ``pos`` (escott.py:1166), which
   silently drops every residue missing from the structure and then dies on a
   length mismatch at escott.py:1185.  Our 6WXB structure covers 485 of 566
   query positions, so ``--pdbfile`` is simply not usable.
2. Therefore ``--weight-mode structural`` (the default) computes escott's own
   ``sstjetormaxtwocomponent`` formula *here* and writes the result into the
   ``trace`` column.  Running escott **without** ``--pdbfile`` then makes
   ``computePred.R``'s ``tjet`` branch read exactly that value.  The behaviour is
   identical to escott's structural mode, but with no coverage requirement and
   full determinism.  ``--weight-mode tjet`` writes the pure conservation trace
   instead, for ablation.

   *** NEVER emit a column named ``traceMax``: computePred.R:61 prefers it. ***

OUTPUT FORMAT
-------------
Tab-separated, one header line, one row per MSA/query column (no gaps, no
skips).  Columns, in order::

    AA  pos  chain  pc  tr  freq  trace  cv

The shipped JET2 reference ``/home3/oml4h/PRESCOTT/data/BLAT_jet.res`` has the
first seven of these (space-padded rather than tab-separated); ``cv`` is
appended because escott's structural branches require it and because escott
itself rewrites the file as a tab-separated table at escott.py:1187.  Both
``pd.read_table(sep="\\s+")`` and R's ``read.table(head=TRUE)`` are
whitespace-agnostic, so the two layouts are interchangeable.

Row count is a *hard* constraint, not a nicety: ``computePred.R:140`` evaluates
``binAli[2:N[1],] %*% trace^2``, so ``nrow(jet)`` must equal the number of MSA
columns exactly or R aborts with "non-conformable arguments".  Residues with no
structural counterpart are therefore **never dropped** - see the fallback rule
below.

HONEST DEVIATIONS FROM REAL JET2
--------------------------------
``trace`` (surrogate).  Real JET2 reports ``tr`` = the mean Lichtarge
    evolutionary trace T_JET over the iterations in which a residue was selected,
    where each iteration builds a distance tree over a subset of BLAST-retrieved
    homologues; ``freq`` = the fraction of iterations retaining the residue
    (~50 iterations, hence the 0.02 quantisation visible in BLAT_jet.res); and
    ``trace = tr * freq`` (verified arithmetically against BLAT_jet.res).
    Ours is a **phylogeny-free** substitute: a Henikoff-weighted Kullback-Leibler
    divergence of each column to a Robinson & Robinson background, multiplied by
    the column's weighted occupancy, quantile-scaled to [0, 1], then
    bootstrap-decomposed into ``tr``/``freq`` so the ``trace = tr * freq``
    identity still holds.  Henikoff weighting is only a first-order correction
    for phylogenetic redundancy, so positions conserved through shared ancestry
    rather than through constraint are **over-scored** relative to real JET2.
``freq`` (surrogate, different meaning).  JET2 resamples *sequences retrieved by
    BLAST*; we resample *rows of a fixed pre-built MSA*.  Our ``freq`` therefore
    measures MSA-sampling stability, not retrieval stability.  JET2 also retains
    nearly everything (BLAT_jet.res: mean freq 0.927, median 1.0), whereas our
    "selected" set is a fixed top-``--trace-top-fraction`` cut, so the shape of
    the freq distribution is ours, not JET2's.
``pc`` (surrogate, not a reproduction).  JET2's ``pc`` is a trained
    physico-chemical/interface-patch propensity combining residue type with
    patch clustering and protrusion; the training data are not available to us.
    Ours is ``0.5 * (rescaled Jones-Thornton interface propensity + relative
    SASA)``, with SASA from freesasa (Shrake-Rupley) in the *trimer* context by
    default so that the oligomeric interface reads as buried.  Same direction,
    same [0, 1] scale, same core intuition (exposed + interface-favourable
    chemistry); no patch clustering, no trained weights.
``cv`` (faithful).  Circular variance with Rc = 7.0 A (``default.conf``:
    ``>CV max_dist 7.0``), 0 = protruding/convex, 1 = buried/concave.  This is
    JET2's own definition and the one column we reproduce properly.
Scale.  The [0, 1] mapping of ``trace`` is a quantile of *this* MSA, so trace
    values are not comparable across MSAs.  Every lineage in this project shares
    one alignment frame, so within-run comparisons are safe.
DSSP version.  ``mkdssp`` here is 4.6.1, not the 3.0.0 named in prescott.yml.
    v4 emits the polyproline-II code ``P`` that v3 never did.  escott tests
    ``ss == 'C'``, so ``P`` counts as non-coil *and* fragments coil runs.  We
    reproduce escott's behaviour exactly (we do **not** remap P -> C) so our
    weight equals what escott would have computed; do not claim bit-parity with
    published PRESCOTT numbers.
No structure.  Positions with no coordinates fall back to **pure ``trace``**,
    which is the same fallback escott applies to long disordered coils
    (escott.py:1180-1184).  ``pc``/``cv`` are reported as NaN in the components
    TSV and written as ``0.0`` in the ``.res`` (R's ``read.table`` would turn
    ``nan`` into ``NA`` and poison the arithmetic).  For 6WXB the 81 uncovered
    query positions are the signal peptide + HA1 1-8, the HA0 cleavage loop and
    the TM/cytoplasmic tail - genuinely disordered or absent, so "no structural
    term" is the correct treatment rather than a fudge.

VALIDATION
----------
``--validate-only --out-validation <path>`` re-runs the surrogate on the shipped
BLAT reference inputs (``aliBLAT.fasta`` + ``blat-af2.pdb``, the only protein for
which PRESCOTT ships a real JET2 ``.res``) and writes the per-column agreement
table plus the zero-trace fractions.  That is the file CAVEATS.md points at,
``tables/diagnostics/jet_surrogate_vs_blat_reference.tsv``.  It needs neither
``--msa`` nor ``--out-jet``; it prints a RED FLAG if Spearman(trace) < 0.4.
``--validate-against <ref.res>`` (without ``--validate-only``) instead compares
the table just built, and requires the same row count as the reference.

Measured 2026-08-05 on aliBLAT.fasta (13473 x 286) + blat-af2.pdb,
``--trace-bootstraps 50 --seed 20260805``, ~10 s wall::

    column   r@top=0.90   r@top=0.30   ours_mean@0.90   ref_mean
    pc          0.658        0.658         0.315          0.498
    tr          0.877        0.703         0.373          0.593
    freq        0.582        0.478         0.899          0.927
    trace       0.877        0.704         0.373          0.578

``--trace-top-fraction`` defaults to **0.90, not the 0.30 in the design plan**,
and constants.DEFAULT_TRACE_TOP_FRACTION is the single source of truth for it.
0.90 wins on every column above, and on the number that matters most - the share
of columns left at ``trace = 0``, each of which ``pred.R:487`` turns into an
identically-zero ESCOTT column and hence a uniform 1/20 softmax, i.e. pure noise::

    inputs         weight mode  top    zero-trace columns
    BLAT           tjet         0.30   196/286  (68.5%)
    BLAT           tjet         0.90    28/286  ( 9.8%)
    BLAT           structural   0.30    30/286  (10.5%)
    BLAT           structural   0.90    23/286  ( 8.0%)
    HA J_int       structural   0.30    77/566  (13.6%)
    HA J_int       structural   0.90    18/566  ( 3.2%)
    BLAT real JET2 --           --       3/286  ( 1.0%)

A caller therefore cannot quietly pick a bad value: above 5% we warn on stderr,
and above ``--max-zero-trace-fraction`` (default 10%, the cut that separates
every 0.90 row above from every 0.30 row) we raise ``ZeroTraceError`` and write
nothing.  The cache path re-applies the same ceiling, so a stale bad table cannot
be served either.

Usage (PRESCOTT env; PATH must contain the env bin dir so biotite finds mkdssp)::

    export PATH=/home3/oml4h/miniconda3/envs/PRESCOTT/bin:$PATH
    /home3/oml4h/miniconda3/envs/PRESCOTT/bin/python scripts/prescott_iav/jet_surrogate.py \\
        --msa      inputs/msa/msa_J_int.fasta \\
        --query    inputs/query/J_int_query.fasta \\
        --pdb      inputs/structure/6WXB_chainA_qnum.pdb \\
        --context-pdb inputs/structure/6WXB_trimer_qnum.pdb \\
        --out-jet  inputs/jet/J_int_surrogate_jet.res \\
        --out-components inputs/jet/J_int_jet_components.tsv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import warnings
from itertools import groupby
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# The scoring tables below are deliberately local (they are this module's own
# empirical content), but the *tuned defaults* are not: `--trace-top-fraction`
# and the zero-trace ceiling are read from constants.py, because the stage-2
# driver used to hard-code a different top fraction and pass it down
# unconditionally.  constants.py is dependency-free, so importing it here cannot
# drag anything into this process; it is tried three ways because this file is
# used as a package member, as a bare module on sys.path and as a script.
if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:  # pragma: no cover - exercised implicitly by every import path
    from . import constants as _constants  # type: ignore
except ImportError:
    try:
        from prescott_iav import constants as _constants  # type: ignore
    except ImportError:
        import importlib.util as _importlib_util

        _spec = _importlib_util.spec_from_file_location(
            "prescott_iav_constants", Path(__file__).resolve().parent / "constants.py"
        )
        if _spec is None or _spec.loader is None:  # pragma: no cover - unreachable in-tree
            raise
        _constants = _importlib_util.module_from_spec(_spec)
        _spec.loader.exec_module(_constants)

DEFAULT_TRACE_TOP_FRACTION = _constants.DEFAULT_TRACE_TOP_FRACTION
MAX_ZERO_TRACE_FRACTION = _constants.MAX_ZERO_TRACE_FRACTION
WARN_ZERO_TRACE_FRACTION = _constants.WARN_ZERO_TRACE_FRACTION
BLAT_REFERENCE_JET_RES = Path(_constants.BLAT_REFERENCE_JET_RES)
BLAT_REFERENCE_MSA = Path(_constants.BLAT_REFERENCE_MSA)
BLAT_REFERENCE_PDB = Path(_constants.BLAT_REFERENCE_PDB)


class ZeroTraceError(RuntimeError):
    """Raised when too many emitted `trace` values are exactly zero.

    This is a hard failure rather than a warning because the damage is silent
    and total: ``pred.R:487`` multiplies each ESCOTT prediction column by
    ``trace[i]``, so an affected site produces an identically-zero column, the
    per-column softmax turns that into a uniform 1/20, and the site then
    contributes pure noise to every site-level metric downstream -- with every
    file still present and every checksum still valid.
    """


# --------------------------------------------------------------------------- #
# Constant tables
# --------------------------------------------------------------------------- #

STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: i for i, aa in enumerate(STANDARD_AMINO_ACIDS)}
GAP_CHARS = set("-.~ *Xx")  # characters that never count as an observed residue

# Robinson & Robinson (1991) background amino-acid frequencies.  Used as the
# reference distribution for the KL divergence so that a column of, say, all
# tryptophan scores much higher than a column of all leucine - a plain Shannon
# entropy would rate them identically.
ROBINSON_BACKGROUND = {
    "A": 0.07805, "C": 0.01925, "D": 0.05364, "E": 0.06295, "F": 0.03856,
    "G": 0.07377, "H": 0.02199, "I": 0.05142, "K": 0.05744, "L": 0.09019,
    "M": 0.02243, "N": 0.04487, "P": 0.05203, "Q": 0.04264, "R": 0.05129,
    "S": 0.07120, "T": 0.05841, "V": 0.06441, "W": 0.01330, "Y": 0.03216,
}

# Jones & Thornton (1996) style interface propensities: relative preference of
# each residue type to occur in a protein-protein interface patch.  Values are
# the classic log-odds-ish propensities; only their *ranking and spread* matter
# because we min-max rescale them to [0, 1] before use.
JONES_THORNTON_INTERFACE_PROPENSITY = {
    "A": -0.17, "C": 0.43, "D": -0.38, "E": -0.13, "F": 0.82,
    "G": -0.07, "H": 0.41, "I": 0.44, "K": -0.36, "L": 0.40,
    "M": 0.66, "N": -0.14, "P": -0.25, "Q": -0.11, "R": 0.27,
    "S": -0.33, "T": -0.18, "V": 0.27, "W": 0.83, "Y": 0.66,
}

# Tien et al. (2013) theoretical maximum solvent accessible surface areas
# (Gly-X-Gly, "theoretical" column), used to turn absolute SASA into RSA.
TIEN_MAX_ASA = {
    "A": 129.0, "C": 167.0, "D": 193.0, "E": 223.0, "F": 240.0,
    "G": 104.0, "H": 224.0, "I": 197.0, "K": 236.0, "L": 201.0,
    "M": 224.0, "N": 195.0, "P": 159.0, "Q": 225.0, "R": 274.0,
    "S": 155.0, "T": 172.0, "V": 174.0, "W": 285.0, "Y": 263.0,
}

THREE_LETTER = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY",
    "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU", "M": "MET", "N": "ASN",
    "P": "PRO", "Q": "GLN", "R": "ARG", "S": "SER", "T": "THR", "V": "VAL",
    "W": "TRP", "Y": "TYR",
}
ONE_LETTER = {v: k for k, v in THREE_LETTER.items()}

# The exact column order escott/JET2 use, plus `cv` appended (see docstring).
JET_COLUMNS = ["AA", "pos", "chain", "pc", "tr", "freq", "trace", "cv"]


# --------------------------------------------------------------------------- #
# Small IO helpers
# --------------------------------------------------------------------------- #

def md5_of(path: Path) -> str:
    """Content hash, used for the cache key in ``jet_manifest.json``."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_fasta(path: Path) -> List[Tuple[str, str]]:
    """Minimal FASTA reader.

    Deliberately not Bio.SeqIO: this file is imported by a script that must stay
    importable in a bare interpreter, and we need raw (possibly gapped, possibly
    lowercase) strings preserved verbatim rather than Seq objects.
    """
    records: List[Tuple[str, str]] = []
    header: Optional[str] = None
    chunks: List[str] = []
    with open(path, "r") as handle:
        for line in handle:
            line = line.rstrip("\n\r")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(chunks)))
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line.strip())
    if header is not None:
        records.append((header, "".join(chunks)))
    return records


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------- #
# MSA -> conservation (the `trace` surrogate)
# --------------------------------------------------------------------------- #

def encode_msa(records: Sequence[Tuple[str, str]]) -> np.ndarray:
    """Encode an aligned FASTA as an int8 matrix of amino-acid indices.

    Non-standard residues, gaps and stop codons all collapse to -1 ("absent"),
    which keeps every downstream count a clean masked sum.
    """
    n_rows = len(records)
    n_cols = len(records[0][1])
    encoded = np.full((n_rows, n_cols), -1, dtype=np.int8)
    for row, (_, seq) in enumerate(records):
        if len(seq) != n_cols:
            raise ValueError(
                f"MSA row {row} has length {len(seq)}, expected {n_cols}; "
                "the alignment is ragged."
            )
        upper = seq.upper()
        for col, char in enumerate(upper):
            idx = AA_INDEX.get(char, -1)
            if idx >= 0:
                encoded[row, col] = idx
    return encoded


def henikoff_weights(encoded: np.ndarray) -> np.ndarray:
    """Henikoff & Henikoff (1994) position-based sequence weights.

    Row k gets ``w_k = sum_i 1 / (r_i * n_{i, x_ik})`` over the columns where it
    has a residue, with r_i = number of distinct residue types in column i.

    WHY: the deep NCBI HA set is dominated by near-identical seasonal H3N2
    sequences (only ~500 of 6433 rows are >90% identical to the query, but the
    remainder is still heavily clustered).  Without down-weighting, conservation
    would mostly measure sampling bias.  This is a first-order fix only - it is
    not a phylogenetic correction, which is the main gap versus real JET2.
    """
    n_rows, n_cols = encoded.shape
    weights = np.zeros(n_rows, dtype=np.float64)
    for col in range(n_cols):
        column = encoded[:, col]
        present = column >= 0
        if not present.any():
            continue
        counts = np.bincount(column[present], minlength=20).astype(np.float64)
        n_types = int((counts > 0).sum())
        if n_types == 0:
            continue
        # contribution of this column to each row that has a residue here
        per_type = np.zeros(20, dtype=np.float64)
        nonzero = counts > 0
        per_type[nonzero] = 1.0 / (n_types * counts[nonzero])
        weights[present] += per_type[column[present]]
    total = weights.sum()
    if total <= 0:
        return np.ones(n_rows, dtype=np.float64)
    return weights * (n_rows / total)  # normalise to mean 1


def column_conservation(
    encoded: np.ndarray,
    weights: np.ndarray,
    pseudocount: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-column weighted KL divergence to background, and weighted occupancy.

    Returns ``(kl_bits, occupancy)``.  The raw conservation signal handed to the
    scaler is ``kl_bits * occupancy``: a column that is "conserved" only because
    90% of the alignment has a gap there should not be treated as constrained.
    """
    n_rows, n_cols = encoded.shape
    background = np.array([ROBINSON_BACKGROUND[aa] for aa in STANDARD_AMINO_ACIDS])
    total_weight = weights.sum()

    kl_bits = np.zeros(n_cols, dtype=np.float64)
    occupancy = np.zeros(n_cols, dtype=np.float64)

    for col in range(n_cols):
        column = encoded[:, col]
        present = column >= 0
        w_present = weights[present]
        w_sum = w_present.sum()
        occupancy[col] = w_sum / total_weight if total_weight > 0 else 0.0
        if w_sum <= 0:
            kl_bits[col] = 0.0
            continue
        counts = np.bincount(column[present], weights=w_present, minlength=20)
        probs = (counts + pseudocount * background) / (w_sum + pseudocount)
        # KL in bits; the pseudocount guarantees probs > 0 everywhere.
        kl_bits[col] = float(np.sum(probs * np.log2(probs / background)))

    return kl_bits, occupancy


def scale_to_unit(raw: np.ndarray, quantile: float) -> np.ndarray:
    """Map a raw conservation vector onto [0, 1] like JET2's trace scale.

    We divide by a high quantile rather than the max so one freak column cannot
    compress everything else into the bottom of the range.
    """
    finite = raw[np.isfinite(raw)]
    if finite.size == 0:
        return np.zeros_like(raw)
    denom = float(np.quantile(finite, quantile))
    if denom <= 0:
        denom = float(finite.max()) if finite.max() > 0 else 1.0
    return np.clip(raw / denom, 0.0, 1.0)


def compute_trace(
    encoded: np.ndarray,
    definition: str,
    bootstraps: int,
    top_fraction: float,
    pseudocount: float,
    scale_quantile: float,
    weighting: str,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Build the ``tr`` / ``freq`` / ``trace`` triple.

    ``bootstrap`` (default) mirrors JET2's structure: resample the alignment,
    recompute conservation, record which columns land in the top
    ``top_fraction``.  ``freq`` is the selection rate (quantised to 1/B, exactly
    like JET2's 0.02 steps at B=50), ``tr`` is the mean scaled score over the
    iterations in which the column was selected, and ``trace = tr * freq``, which
    is the identity real JET2 satisfies.

    ``direct`` skips the bootstrap (tr = scaled score, freq = 1.0) and exists for
    ``--test-mode``, where the 50 resamples dominate the runtime.

    On ``top_fraction``: unselected columns get ``freq = 0`` hence ``trace = 0``,
    and ``pred.R:487`` multiplies every column of the prediction by ``trace[i]``,
    so a zero-trace position produces an identically zero ESCOTT column - the
    site contributes nothing at all.  Real JET2 has only 3/286 zero-trace rows in
    BLAT_jet.res, so an aggressive cut is not JET2-like; see the module note on
    the default value.
    """
    n_rows, n_cols = encoded.shape

    weights = (
        henikoff_weights(encoded) if weighting == "henikoff"
        else np.ones(n_rows, dtype=np.float64)
    )
    kl_bits, occupancy = column_conservation(encoded, weights, pseudocount)
    raw = kl_bits * occupancy
    trace_raw = scale_to_unit(raw, scale_quantile)

    if definition == "direct":
        tr = trace_raw.copy()
        freq = np.ones(n_cols, dtype=np.float64)
        trace_jet = np.round(tr * freq, 4)
        return {
            "kl_bits": kl_bits, "occupancy": occupancy, "trace_raw": trace_raw,
            "tr": tr, "freq": freq, "trace_jet": trace_jet, "weights": weights,
        }

    rng = np.random.default_rng(seed)
    n_top = max(1, int(round(top_fraction * n_cols)))
    selected_count = np.zeros(n_cols, dtype=np.int64)
    selected_sum = np.zeros(n_cols, dtype=np.float64)

    for _ in range(bootstraps):
        # Row 0 is the query and is always retained: GEMME treats it as the
        # reference sequence, and dropping it would change the alphabet seen at
        # query-specific positions.
        resampled = rng.integers(1, n_rows, size=n_rows - 1) if n_rows > 1 else np.array([], dtype=int)
        rows = np.concatenate(([0], resampled)).astype(int)
        sub = encoded[rows, :]
        sub_weights = (
            henikoff_weights(sub) if weighting == "henikoff"
            else np.ones(sub.shape[0], dtype=np.float64)
        )
        sub_kl, sub_occ = column_conservation(sub, sub_weights, pseudocount)
        sub_scaled = scale_to_unit(sub_kl * sub_occ, scale_quantile)
        # top-N by score; ties broken arbitrarily but deterministically
        order = np.argsort(-sub_scaled, kind="stable")[:n_top]
        selected_count[order] += 1
        selected_sum[order] += sub_scaled[order]

    freq = selected_count / float(bootstraps)
    with np.errstate(invalid="ignore", divide="ignore"):
        tr = np.where(selected_count > 0, selected_sum / np.maximum(selected_count, 1), 0.0)
    trace_jet = np.round(tr * freq, 4)

    return {
        "kl_bits": kl_bits, "occupancy": occupancy, "trace_raw": trace_raw,
        "tr": tr, "freq": freq, "trace_jet": trace_jet, "weights": weights,
    }


# --------------------------------------------------------------------------- #
# Structure -> cv, pc, secondary structure
# --------------------------------------------------------------------------- #

def load_structure(path: Path):
    """Load a PDB/mmCIF into a prody AtomGroup with prody's chatter suppressed."""
    import prody

    prody.confProDy(verbosity="none")
    suffix = path.suffix.lower()
    if suffix in {".cif", ".mmcif"}:
        struct = prody.parseMMCIF(str(path))
    else:
        struct = prody.parsePDB(str(path))
    if struct is None:
        raise ValueError(f"could not parse structure {path}")
    return struct


def residue_index(struct, chain: Optional[str]) -> Dict[int, Dict]:
    """Map residue number -> {resname, heavy-atom coords, CA coord} for one chain.

    Only protein, non-hetero atoms are kept, so the NAG/BMA glycans on 6WXB and
    any waters drop out automatically.  Hydrogens are excluded because JET2's
    circular variance is defined over heavy atoms.
    """
    selection = "protein and not hetero and not hydrogen"
    if chain:
        selection += f" and chain {chain}"
    sel = struct.select(selection)
    if sel is None:
        raise ValueError(f"selection '{selection}' matched no atoms")

    resnums = sel.getResnums()
    resnames = sel.getResnames()
    coords = sel.getCoords()
    names = sel.getNames()

    out: Dict[int, Dict] = {}
    for i in range(len(resnums)):
        rec = out.setdefault(
            int(resnums[i]),
            {"resname": str(resnames[i]), "coords": [], "ca": None},
        )
        rec["coords"].append(coords[i])
        if names[i] == "CA" and rec["ca"] is None:
            rec["ca"] = coords[i]
    for rec in out.values():
        rec["coords"] = np.asarray(rec["coords"], dtype=np.float64)
        rec["centroid"] = rec["coords"].mean(axis=0)
        if rec["ca"] is None:
            rec["ca"] = rec["centroid"]
    return out


def circular_variance(
    context_struct,
    target_chain: str,
    positions: Sequence[int],
    radius: float = 7.0,
    atom_mode: str = "heavy",
) -> Dict[int, float]:
    """JET2 circular variance: 0 = protruding/convex, 1 = buried/concave.

    ``cv_i = 1 - || mean_j (r_j - r_i)/||r_j - r_i|| ||`` over every heavy atom
    j within ``radius`` of the reference point of residue i, excluding residue
    i's own atoms.

    The neighbourhood is drawn from the WHOLE context structure (all chains), so
    for the HA trimer the subunit interface correctly reads as buried; using the
    monomer alone would report interface residues as exposed.
    """
    from scipy.spatial import cKDTree

    sel = context_struct.select("protein and not hetero and not hydrogen")
    if sel is None:
        raise ValueError("context structure has no protein heavy atoms")
    all_coords = sel.getCoords()
    all_chains = sel.getChids()
    all_resnums = sel.getResnums()
    tree = cKDTree(all_coords)

    target = residue_index(context_struct, target_chain)
    out: Dict[int, float] = {}
    for pos in positions:
        rec = target.get(int(pos))
        if rec is None:
            out[int(pos)] = float("nan")
            continue
        origin = rec["ca"] if atom_mode == "ca" else rec["centroid"]
        neighbour_idx = tree.query_ball_point(origin, radius)
        if not neighbour_idx:
            out[int(pos)] = 0.0
            continue
        idx = np.asarray(neighbour_idx, dtype=int)
        # exclude the residue's own atoms (same chain AND same resnum)
        keep = ~((all_chains[idx] == target_chain) & (all_resnums[idx] == int(pos)))
        idx = idx[keep]
        if idx.size == 0:
            out[int(pos)] = 0.0
            continue
        vectors = all_coords[idx] - origin
        norms = np.linalg.norm(vectors, axis=1)
        good = norms > 1e-6
        if not good.any():
            out[int(pos)] = 0.0
            continue
        unit = vectors[good] / norms[good][:, None]
        out[int(pos)] = float(1.0 - np.linalg.norm(unit.mean(axis=0)))
    return out


def relative_sasa(context_pdb: Path, target_chain: str) -> Dict[int, float]:
    """Per-residue relative solvent accessibility (RSA) of ``target_chain``.

    freesasa runs on the full context (trimer by default) so buried interface
    area is not counted as exposed; we then read off only the target chain.
    Normalised by Tien et al. (2013) theoretical max ASA and clipped to [0, 1].
    """
    import freesasa

    freesasa.setVerbosity(freesasa.silent)
    structure = freesasa.Structure(str(context_pdb))
    result = freesasa.calc(structure)
    areas = result.residueAreas()  # {chain: {resnum_str: ResidueArea}}

    chain_areas = areas.get(target_chain, {})
    out: Dict[int, float] = {}
    for resnum_str, area in chain_areas.items():
        try:
            resnum = int(str(resnum_str).strip())
        except ValueError:
            continue  # insertion codes: we assert upstream that there are none
        one = ONE_LETTER.get(str(area.residueType).strip().upper())
        if one is None:
            continue
        max_asa = TIEN_MAX_ASA[one]
        out[resnum] = float(np.clip(area.total / max_asa, 0.0, 1.0))
    return out


def interface_propensity(
    query_protein: str,
    positions: Sequence[int],
    rsa: Dict[int, float],
    mode: str,
) -> Dict[int, float]:
    """``pc`` surrogate = 0.5 * (rescaled interface propensity + RSA).

    Both terms live in [0, 1] and point the same way as JET2's ``pc`` (higher =
    more likely to be part of a functional surface patch).  ``--pc-mode
    constant`` / ``zero`` exist so a reviewer can ask "does pc matter at all?"
    and get an answer from one rerun.
    """
    values = np.array(list(JONES_THORNTON_INTERFACE_PROPENSITY.values()))
    lo, hi = float(values.min()), float(values.max())
    scaled = {
        aa: (val - lo) / (hi - lo)
        for aa, val in JONES_THORNTON_INTERFACE_PROPENSITY.items()
    }

    out: Dict[int, float] = {}
    for pos in positions:
        aa = query_protein[pos - 1]
        chem = scaled.get(aa)
        exposure = rsa.get(int(pos))
        if chem is None or exposure is None:
            out[int(pos)] = float("nan")
        else:
            out[int(pos)] = float(0.5 * (chem + exposure))

    if mode == "zero":
        return {p: 0.0 for p in out}
    if mode == "constant":
        finite = [v for v in out.values() if np.isfinite(v)]
        fill = float(np.median(finite)) if finite else 0.0
        return {p: fill for p in out}
    return out


def mkdssp_bin() -> str:
    """Absolute path to mkdssp.

    biotite's DsspApp defaults to bin_path='mkdssp', i.e. a bare PATH lookup.  This
    module is designed to be invoked as
    ``/home3/oml4h/miniconda3/envs/PRESCOTT/bin/python jet_surrogate.py ...`` WITHOUT
    ``conda activate``, and in that case the env's bin/ is not on PATH and DsspApp dies
    with ``FileNotFoundError: [Errno 2] ... 'mkdssp'``.  Fall back to the interpreter's
    own bin directory, which is where the env's mkdssp lives.
    """
    found = shutil.which("mkdssp")
    if found:
        return found
    sibling = Path(sys.executable).resolve().parent / "mkdssp"
    if sibling.exists():
        return str(sibling)
    raise FileNotFoundError(
        "mkdssp not found on PATH nor next to the running interpreter "
        f"({Path(sys.executable).resolve().parent}). Install dssp into the PRESCOTT env "
        "or put its bin/ on PATH."
    )


def dssp_runs(pdb_path: Path) -> Dict[int, Tuple[str, int]]:
    """Per-residue (secondary structure letter, length of its contiguous run).

    EXACTLY mirrors escott.py:817-911 (``calculateSecondaryStructure`` +
    ``countCoilSegments``): biotite's DsspApp, paired against the CA res_ids,
    then itertools.groupby over consecutive identical letters.

    WHY biotite and not a direct mkdssp call: mkdssp 4.6.1 rejects prody-written
    PDB files with "Not a valid integer in PDB record"; biotite rewrites the
    structure to CIF before invoking it, which succeeds.  Using escott's own code
    path also guarantees our coil-length test matches escott's.

    Note we do NOT remap the mkdssp-4 polyproline code 'P' to 'C'.  escott tests
    ``ss == 'C'``, so under escott 'P' is non-coil and breaks coil runs; matching
    that is the point.
    """
    import biotite.application.dssp as dssp_app
    import biotite.structure.io as strucio

    array = strucio.load_structure(str(pdb_path))
    app = dssp_app.DsspApp(array, bin_path=mkdssp_bin())
    app.start()
    app.join()
    sse = list(app.get_sse())

    calphas = array[array.atom_name == "CA"]
    res_ids = [int(r) for r in calphas.res_id]
    if len(res_ids) != len(sse):
        raise ValueError(
            f"DSSP returned {len(sse)} states for {len(res_ids)} CA atoms in {pdb_path}"
        )

    runs: List[Tuple[str, int]] = [
        (label, sum(1 for _ in group)) for label, group in groupby(sse)
    ]
    expanded: List[Tuple[str, int]] = []
    for label, length in runs:
        expanded.extend([(label, length)] * length)

    return {res_ids[i]: expanded[i] for i in range(len(res_ids))}


def dssp_version() -> str:
    try:
        proc = subprocess.run(
            [mkdssp_bin(), "--version"], capture_output=True, text=True, timeout=30
        )
        return (proc.stdout or proc.stderr).strip().splitlines()[0]
    except Exception as exc:  # pragma: no cover - diagnostics only
        return f"unknown ({exc})"


# --------------------------------------------------------------------------- #
# Weight assembly
# --------------------------------------------------------------------------- #

def combine_weight(
    trace_jet: np.ndarray,
    pc: np.ndarray,
    cv: np.ndarray,
    ss: Sequence[Optional[str]],
    runlength: Sequence[Optional[int]],
    has_structure: np.ndarray,
    max_coil_length: int,
) -> np.ndarray:
    """escott's ``sstjetormaxtwocomponent`` formula (escott.py:1180-1184).

        weight = trace                                      if long coil
               = max((trace+pc)/2, (trace+cv)/2)            otherwise

    plus our documented extension: positions with no coordinates also fall back
    to pure ``trace``.  escott could not express that case at all - its inner
    join would simply have deleted those rows - so this is an addition, not a
    divergence, and it is what makes an 85%-coverage structure usable.
    """
    n = len(trace_jet)
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        t = float(trace_jet[i])
        if not has_structure[i]:
            out[i] = t
            continue
        if ss[i] == "C" and (runlength[i] or 0) > max_coil_length:
            out[i] = t
            continue
        p = float(pc[i]) if np.isfinite(pc[i]) else t
        c = float(cv[i]) if np.isfinite(cv[i]) else t
        out[i] = max((t + p) / 2.0, (t + c) / 2.0)
    return np.round(out, 4)


# --------------------------------------------------------------------------- #
# Top-level build
# --------------------------------------------------------------------------- #

def build_jet_table(
    msa_path: Path,
    query_path: Optional[Path],
    pdb_path: Optional[Path],
    context_pdb_path: Optional[Path],
    *,
    weight_mode: str = "structural",
    trace_definition: str = "bootstrap",
    trace_bootstraps: int = 50,
    trace_top_fraction: float = DEFAULT_TRACE_TOP_FRACTION,
    max_zero_trace_fraction: Optional[float] = MAX_ZERO_TRACE_FRACTION,
    trace_pseudocount: float = 0.5,
    trace_scale_quantile: float = 0.99,
    sequence_weighting: str = "henikoff",
    pc_mode: str = "interface_propensity",
    structure_chain: str = "A",
    cv_radius: float = 7.0,
    cv_atom: str = "heavy",
    max_coil_length: int = 5,
    seed: int = 20260805,
) -> Tuple[pd.DataFrame, Dict]:
    """Build the full components table plus a metadata dict.

    Returns ``(components_df, meta)``.  The ``.res`` file is a projection of
    ``components_df``; keeping the wide frame around is what makes the surrogate
    auditable.

    ``max_zero_trace_fraction`` is the share of emitted ``trace`` values allowed
    to be exactly zero before this raises :class:`ZeroTraceError`.  Pass ``None``
    to warn only -- the validation path does, because a deliberately bad
    ``trace_top_fraction`` is a legitimate thing to *measure*.
    """
    records = read_fasta(msa_path)
    if not records:
        raise ValueError(f"empty MSA: {msa_path}")

    aligned_query = records[0][1].upper()
    query_protein = aligned_query.replace("-", "").replace(".", "")

    # The MSA must be query-anchored and gap-free in row 1: every downstream
    # index (jet row, GEMME column, escott mutation label) is a query position.
    if len(query_protein) != len(aligned_query):
        raise ValueError(
            f"MSA row 1 contains {len(aligned_query) - len(query_protein)} gaps; "
            "jet_surrogate requires a query-anchored, gap-free first row "
            "(strip query-gap columns upstream)."
        )

    if query_path is not None:
        query_records = read_fasta(query_path)
        if len(query_records) != 1:
            raise ValueError(f"{query_path} must hold exactly one record")
        expected = query_records[0][1].upper().replace("-", "")
        if expected != query_protein:
            raise ValueError(
                "MSA row 1 does not match the query FASTA: "
                f"{len(query_protein)} vs {len(expected)} aa, first mismatch at "
                f"{next((i for i, (a, b) in enumerate(zip(query_protein, expected)) if a != b), 'n/a')}"
            )

    n_cols = len(query_protein)
    positions = np.arange(1, n_cols + 1, dtype=int)

    encoded = encode_msa(records)
    trace_parts = compute_trace(
        encoded,
        definition=trace_definition,
        bootstraps=trace_bootstraps,
        top_fraction=trace_top_fraction,
        pseudocount=trace_pseudocount,
        scale_quantile=trace_scale_quantile,
        weighting=sequence_weighting,
        seed=seed,
    )

    # ---- structural columns ------------------------------------------------
    pc = np.full(n_cols, np.nan)
    cv = np.full(n_cols, np.nan)
    ss: List[Optional[str]] = [None] * n_cols
    runlength: List[Optional[int]] = [None] * n_cols
    has_structure = np.zeros(n_cols, dtype=bool)
    # `pdb_md5`/`context_pdb_md5` are not decoration: main()'s cache short-circuit
    # compares them.  prepare_inputs.py REWRITES inputs/structure/<stem>_chain<C>_qnum.pdb
    # in place on every run, so the path alone is invariant under a changed
    # --structure-offset, a re-prepared source structure, or two source files whose
    # stem token collides -- and a path-only comparison then serves the previous
    # structure's table verbatim, skipping build_jet_table's own frame-identity check.
    # The md5 keys appear only when there IS a structure, so that a sequence-only
    # run keeps the historical three-key shape and the cache predicate's
    # ``.get(...) is None`` comparison is true on both sides for it.
    struct_meta: Dict = {"pdb": None, "context_pdb": None, "covered": 0}

    if pdb_path is not None:
        struct = load_structure(pdb_path)
        target = residue_index(struct, structure_chain)
        covered = sorted(p for p in target if 1 <= p <= n_cols)
        has_structure[[p - 1 for p in covered]] = True
        struct_meta["pdb"] = str(pdb_path)
        struct_meta["pdb_md5"] = md5_of(Path(pdb_path))
        struct_meta["covered"] = len(covered)

        # Sanity: the structure must actually be this protein.  A silent frame
        # offset here would poison every structural column without any error.
        matches = sum(
            1 for p in covered
            if ONE_LETTER.get(target[p]["resname"].upper()) == query_protein[p - 1]
        )
        identity = matches / max(1, len(covered))
        struct_meta["structure_query_identity"] = round(identity, 4)
        if identity < 0.60:
            raise ValueError(
                f"structure {pdb_path} chain {structure_chain} matches the query at "
                f"only {identity:.1%} of covered positions - the residue numbering "
                "is almost certainly not in the query frame."
            )

        context_path = context_pdb_path or pdb_path
        context_struct = (
            struct if context_path == pdb_path else load_structure(context_path)
        )
        struct_meta["context_pdb"] = str(context_path)
        struct_meta["context_pdb_md5"] = md5_of(Path(context_path))

        cv_vals = circular_variance(
            context_struct, structure_chain, covered, radius=cv_radius, atom_mode=cv_atom
        )
        for p, val in cv_vals.items():
            cv[p - 1] = val

        rsa = relative_sasa(context_path, structure_chain)
        pc_vals = interface_propensity(query_protein, covered, rsa, pc_mode)
        for p, val in pc_vals.items():
            pc[p - 1] = val

        run_map = dssp_runs(pdb_path)
        for p, (letter, length) in run_map.items():
            if 1 <= p <= n_cols:
                ss[p - 1] = letter
                runlength[p - 1] = length

    weight = combine_weight(
        trace_parts["trace_jet"], pc, cv, ss, runlength, has_structure, max_coil_length
    )

    # `trace` is the column both computePred.R and escott read.  Under
    # `structural` we put the combined weight there so escott's tjet branch
    # reproduces sstjetormaxtwocomponent (see module docstring).
    trace_out = weight if weight_mode == "structural" else trace_parts["trace_jet"]

    components = pd.DataFrame(
        {
            "pos": positions,
            "AA": [THREE_LETTER.get(a, "UNK") for a in query_protein],
            "aa1": list(query_protein),
            "chain": structure_chain,
            "has_structure": has_structure,
            "occupancy": np.round(trace_parts["occupancy"], 6),
            "kl_bits": np.round(trace_parts["kl_bits"], 6),
            "trace_raw": np.round(trace_parts["trace_raw"], 6),
            "tr": np.round(trace_parts["tr"], 4),
            "freq": np.round(trace_parts["freq"], 4),
            "trace_jet": trace_parts["trace_jet"],
            "pc": pc,
            "cv": cv,
            "ss": ss,
            "runlength": runlength,
            "weight": weight,
            "trace_emitted": trace_out,
        }
    )

    meta = {
        "msa_path": str(msa_path),
        "msa_md5": md5_of(msa_path),
        "msa_n_sequences": len(records),
        "msa_n_columns": n_cols,
        "query_length": n_cols,
        "weight_mode": weight_mode,
        "trace_definition": trace_definition,
        "trace_bootstraps": trace_bootstraps if trace_definition == "bootstrap" else 0,
        "trace_top_fraction": trace_top_fraction,
        "trace_pseudocount": trace_pseudocount,
        "trace_scale_quantile": trace_scale_quantile,
        "sequence_weighting": sequence_weighting,
        "pc_mode": pc_mode,
        # Recorded so main()'s cache short-circuit can compare it: on a trimer the
        # chains have different cv, pc and DSSP states, so serving chain A's table
        # for a --structure-chain B request is silently the wrong subunit.
        "structure_chain": structure_chain,
        "cv_radius": cv_radius,
        "cv_atom": cv_atom,
        "max_coil_length": max_coil_length,
        "seed": seed,
        "structure": struct_meta,
        "n_positions_without_structure": int((~has_structure).sum()),
        "mkdssp_version": dssp_version() if pdb_path is not None else None,
    }

    # A zero in the emitted `trace` is not a cosmetic detail: pred.R:487 does
    # `normPred[,i] = normPred[,i] * trace[i]`, so those sites come back as an
    # all-zero ESCOTT column and carry no signal whatsoever.  Real JET2 leaves
    # 3/286 (1.0%) of BLAT at zero.  We warn above WARN_ZERO_TRACE_FRACTION and
    # REFUSE above max_zero_trace_fraction, because the alternative -- a warning
    # nobody reads on a run that produces every expected file with valid
    # checksums -- is exactly how top_fraction=0.30 survived a full end-to-end
    # smoke test.  See constants.py for the measurements behind the ceiling.
    n_zero = int((np.asarray(trace_out, dtype=float) <= 1e-9).sum())
    frac_zero = n_zero / max(1, n_cols)
    meta["n_zero_trace_columns"] = n_zero
    meta["frac_zero_trace_columns"] = round(frac_zero, 4)
    meta["max_zero_trace_fraction"] = max_zero_trace_fraction
    diagnosis = (
        f"{n_zero}/{n_cols} ({frac_zero:.1%}) of emitted `trace` values are exactly 0 "
        f"(trace_definition={trace_definition}, trace_top_fraction={trace_top_fraction}, "
        f"weight_mode={weight_mode}). pred.R:487 multiplies every ESCOTT column by "
        "trace[i], so each of those sites yields an all-zero column and a uniform 1/20 "
        f"softmax -- pure noise. Real JET2 leaves 1.0% of BLAT at zero. Raise "
        f"--trace-top-fraction (measured best: {DEFAULT_TRACE_TOP_FRACTION})"
    )
    if max_zero_trace_fraction is not None and frac_zero > max_zero_trace_fraction:
        raise ZeroTraceError(
            diagnosis
            + f", or pass --max-zero-trace-fraction > {frac_zero:.4f} to accept it "
            "deliberately (1.0 disables the check entirely)."
        )
    if frac_zero > WARN_ZERO_TRACE_FRACTION:
        warnings.warn(diagnosis + ".", RuntimeWarning)
        print(f"[jet_surrogate] WARNING: {diagnosis}.", file=sys.stderr)

    return components, meta


def write_jet_res(components: pd.DataFrame, out_path: Path) -> None:
    """Write the ``.res`` in escott's exact column layout.

    NaN ``pc``/``cv`` become 0.0: R's ``read.table`` maps ``nan`` to ``NA`` and
    any arithmetic on it silently propagates NA through the whole prediction.
    The true NaN survives in the components TSV, which is where auditing should
    happen.
    """
    table = pd.DataFrame(
        {
            "AA": components["AA"],
            "pos": components["pos"].astype(int),
            "chain": components["chain"],
            "pc": np.round(components["pc"].fillna(0.0).to_numpy(), 4),
            "tr": np.round(components["tr"].to_numpy(), 4),
            "freq": np.round(components["freq"].to_numpy(), 4),
            "trace": np.round(components["trace_emitted"].to_numpy(), 4),
            "cv": np.round(components["cv"].fillna(0.0).to_numpy(), 4),
        }
    )[JET_COLUMNS]

    validate_jet_table(table, expected_rows=len(components))
    ensure_dir(out_path.parent)
    table.to_csv(out_path, sep="\t", index=False, header=True, float_format="%.4f")


def validate_jet_table(table: pd.DataFrame, expected_rows: int) -> None:
    """Fail loudly on anything that would break computePred.R downstream."""
    if list(table.columns) != JET_COLUMNS:
        raise AssertionError(f"column layout {list(table.columns)} != {JET_COLUMNS}")
    if len(table) != expected_rows:
        raise AssertionError(
            f"{len(table)} rows for {expected_rows} query positions; "
            "computePred.R:140 requires exactly one row per MSA column"
        )
    if not np.array_equal(table["pos"].to_numpy(), np.arange(1, expected_rows + 1)):
        raise AssertionError("`pos` must be 1..N ascending with no gaps")
    if "traceMax" in table.columns:
        raise AssertionError("never emit `traceMax`: computePred.R:61 prefers it over `trace`")
    for col in ("pc", "tr", "freq", "trace", "cv"):
        values = table[col].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise AssertionError(f"non-finite values in `{col}`")
        if values.min() < -1e-9 or values.max() > 1.0 + 1e-9:
            raise AssertionError(
                f"`{col}` out of [0,1]: [{values.min():.4f}, {values.max():.4f}]"
            )


# --------------------------------------------------------------------------- #
# Validation against the shipped JET2 reference
# --------------------------------------------------------------------------- #

def read_jet_res(path: Path) -> pd.DataFrame:
    """Parse a .res with EXACTLY escott's parser (escott.py:1124)."""
    return pd.read_table(path, sep=r"\s+")


def compare_to_reference(
    ours: pd.DataFrame,
    reference_path: Path,
    extra_pairs: Sequence[Tuple[str, str]] = (),
) -> pd.DataFrame:
    """Spearman + range + zero-fraction comparison against real JET2 output.

    ``extra_pairs`` are ``(our_column, reference_column)`` pairs appended after
    the four standard ones, which is how the validation run compares the
    *emitted* trace (the combined structural weight under
    ``--weight-mode structural``) against JET2's raw conservation trace as well
    as comparing like with like.

    ``frac_zero`` is reported for both sides on every row.  It is the number the
    whole ``--trace-top-fraction`` argument turns on: real JET2 leaves 1.0% of
    BLAT at ``trace = 0`` and every zero is a site that scores as pure noise, so
    a validation table without it would hide the single most damaging way the
    surrogate can differ from the thing it imitates.
    """
    from scipy.stats import spearmanr

    ref = read_jet_res(reference_path)
    if len(ref) != len(ours):
        raise ValueError(
            f"reference {reference_path} has {len(ref)} rows, ours has {len(ours)}; "
            "not comparable -- the validation run must use the same protein as the "
            "reference (see --validate-only, which uses the shipped BLAT inputs)"
        )

    pairs: List[Tuple[str, str]] = [(col, col) for col in ("pc", "tr", "freq", "trace")]
    pairs.extend(extra_pairs)

    rows = []
    for our_col, ref_col in pairs:
        if ref_col not in ref.columns or our_col not in ours.columns:
            continue
        theirs = ref[ref_col].to_numpy(dtype=float)
        mine = ours[our_col].to_numpy(dtype=float)
        # `pc`/`cv` are NaN at positions with no coordinates; ranking over NaN
        # returns NaN for the whole correlation, so restrict to the shared
        # finite support and say how many rows that was.
        usable = np.isfinite(mine) & np.isfinite(theirs)
        rho, pval = spearmanr(mine[usable], theirs[usable]) if usable.sum() > 2 else (float("nan"),) * 2
        rows.append(
            {
                "column": our_col if our_col == ref_col else f"{our_col}_vs_{ref_col}",
                "spearman_r": round(float(rho), 4),
                "spearman_p": float(pval),
                "n": int(usable.sum()),
                "n_total": len(mine),
                "ours_min": round(float(np.nanmin(mine)), 4),
                "ours_max": round(float(np.nanmax(mine)), 4),
                "ours_mean": round(float(np.nanmean(mine)), 4),
                "ours_n_zero": int(np.sum(np.abs(np.nan_to_num(mine)) <= 1e-9)),
                "ours_frac_zero": round(float(np.mean(np.abs(np.nan_to_num(mine)) <= 1e-9)), 4),
                "ref_min": round(float(np.nanmin(theirs)), 4),
                "ref_max": round(float(np.nanmax(theirs)), 4),
                "ref_mean": round(float(np.nanmean(theirs)), 4),
                "ref_n_zero": int(np.sum(np.abs(theirs) <= 1e-9)),
                "ref_frac_zero": round(float(np.mean(np.abs(theirs) <= 1e-9)), 4),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a JET2-compatible <prot>_jet.res from an MSA + structure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Not `required=True`: --validate-only builds from the shipped BLAT reference
    # inputs instead, and main() enforces the requirement for every other mode.
    parser.add_argument("--msa", type=Path, default=None,
                        help="query-anchored MSA; row 1 must be the query with NO gaps "
                             "(required unless --validate-only)")
    parser.add_argument("--query", type=Path, default=None,
                        help="optional single-record FASTA cross-checked against MSA row 1")
    parser.add_argument("--pdb", type=Path, default=None,
                        help="single-chain structure in QUERY numbering; omit for a "
                             "sequence-only table (pc=cv=0, weight falls back to trace)")
    parser.add_argument("--context-pdb", type=Path, default=None,
                        help="oligomer used as the SASA/CV environment; defaults to --pdb")
    parser.add_argument("--structure-chain", default="A")
    parser.add_argument("--out-jet", type=Path, default=None,
                        help="required unless --validate-only, which needs no .res on disk")
    parser.add_argument("--out-components", type=Path, default=None)
    parser.add_argument("--out-manifest", type=Path, default=None)
    parser.add_argument("--out-dssp", type=Path, default=None,
                        help="optional pos,ss,runlength CSV mirroring escott's .dssp.new")

    parser.add_argument("--weight-mode", choices=("structural", "tjet"), default="structural",
                        help="what goes in the `trace` column: the sstjetormaxtwocomponent "
                             "weight (structural) or the raw conservation trace (tjet)")
    parser.add_argument("--trace-definition", choices=("bootstrap", "direct"), default="bootstrap")
    parser.add_argument("--trace-bootstraps", type=int, default=50)
    # THE single source of truth is constants.DEFAULT_TRACE_TOP_FRACTION (0.90),
    # NOT the 0.30 in the design plan.  Measured on BLAT: 0.30 leaves 196/286
    # columns at trace=0 (vs 3/286 in real JET2) and pred.R zeroes every one of
    # those sites; 0.90 gives 28/286 zeros, mean freq 0.899 (ref 0.927) and the
    # best Spearman against real JET2 trace (0.877 vs 0.704 at 0.30).  A caller
    # that lowers this cannot do so silently: --max-zero-trace-fraction refuses.
    parser.add_argument("--trace-top-fraction", type=float,
                        default=DEFAULT_TRACE_TOP_FRACTION)
    parser.add_argument("--max-zero-trace-fraction", type=float,
                        default=MAX_ZERO_TRACE_FRACTION,
                        help="REFUSE to write the table if more than this share of emitted "
                             "`trace` values are exactly 0 (each such site scores as pure "
                             "noise once pred.R multiplies by trace). 1.0 disables the check")
    parser.add_argument("--trace-pseudocount", type=float, default=0.5)
    parser.add_argument("--trace-scale-quantile", type=float, default=0.99)
    parser.add_argument("--sequence-weighting", choices=("henikoff", "none"), default="henikoff")
    parser.add_argument("--pc-mode",
                        choices=("interface_propensity", "constant", "zero"),
                        default="interface_propensity")
    parser.add_argument("--cv-radius", type=float, default=7.0,
                        help="JET2 default.conf '>CV max_dist 7.0'")
    parser.add_argument("--cv-atom", choices=("heavy", "ca"), default="heavy")
    parser.add_argument("--max-coil-length", type=int, default=5,
                        help="matches escott --maxcoillength")
    parser.add_argument("--seed", type=int, default=20260805)

    parser.add_argument("--validate-against", type=Path, default=None,
                        help="a real JET2 .res with the same row count (e.g. "
                             f"{BLAT_REFERENCE_JET_RES}); compared against the table just built")
    parser.add_argument("--validate-only", action="store_true",
                        help="skip the per-lineage build entirely: run the surrogate on the "
                             "shipped BLAT reference inputs (--validation-msa/--validation-pdb) "
                             "and write only the comparison table. This is the mode that "
                             "produces tables/diagnostics/jet_surrogate_vs_blat_reference.tsv; "
                             "--msa/--out-jet are not needed")
    parser.add_argument("--validation-msa", type=Path, default=BLAT_REFERENCE_MSA,
                        help="MSA the reference .res was computed from")
    parser.add_argument("--validation-pdb", type=Path, default=BLAT_REFERENCE_PDB,
                        help="structure the reference .res was computed from")
    parser.add_argument("--out-validation", type=Path, default=None)
    parser.add_argument("--min-validation-spearman", type=float, default=0.4,
                        help="print a RED FLAG below this Spearman on `trace`")
    parser.add_argument("--force", action="store_true",
                        help="rebuild even if --out-jet exists with a matching manifest")
    return parser


def run_validation(args: argparse.Namespace) -> int:
    """``--validate-only``: surrogate-vs-real-JET2 on the shipped BLAT reference.

    This is the one honest check available for a surrogate whose training data we
    do not have, and the file CAVEATS.md points at.  It deliberately runs on BLAT
    rather than on HA: BLAT is the only protein for which PRESCOTT ships a real
    JET2 ``.res``, so it is the only place a row-for-row comparison is possible.

    The zero-trace ceiling is DISABLED here (``max_zero_trace_fraction=None``):
    measuring what a bad ``--trace-top-fraction`` does to the zero fraction is
    the entire point, so this path must be able to build a table the production
    path would refuse.
    """
    reference = args.validate_against or BLAT_REFERENCE_JET_RES
    for label, path in (("--validation-msa", args.validation_msa),
                        ("--validation-pdb", args.validation_pdb),
                        ("--validate-against", reference)):
        if not Path(path).exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    components, meta = build_jet_table(
        Path(args.validation_msa),
        None,
        Path(args.validation_pdb),
        None,
        weight_mode=args.weight_mode,
        trace_definition=args.trace_definition,
        trace_bootstraps=args.trace_bootstraps,
        trace_top_fraction=args.trace_top_fraction,
        max_zero_trace_fraction=None,
        trace_pseudocount=args.trace_pseudocount,
        trace_scale_quantile=args.trace_scale_quantile,
        sequence_weighting=args.sequence_weighting,
        pc_mode=args.pc_mode,
        structure_chain=args.structure_chain,
        cv_radius=args.cv_radius,
        cv_atom=args.cv_atom,
        max_coil_length=args.max_coil_length,
        seed=args.seed,
    )

    # Compare like with like on `trace`: JET2's `trace` is tr * freq, which is
    # our `trace_jet` regardless of --weight-mode.  `trace_emitted` (the column
    # that actually reaches escott) is reported as a second row so a
    # --weight-mode structural run is not silently credited with, or blamed for,
    # the difference between the two definitions.
    ours = components.rename(columns={"trace_jet": "trace"})
    report = compare_to_reference(ours, Path(reference),
                                  extra_pairs=(("trace_emitted", "trace"),))
    report.insert(0, "weight_mode", args.weight_mode)
    report.insert(0, "trace_top_fraction", args.trace_top_fraction)
    report.insert(0, "reference", str(reference))
    print(report.to_string(index=False))
    print(
        f"[jet_surrogate] zero-trace: ours {meta['n_zero_trace_columns']}/"
        f"{meta['msa_n_columns']} ({meta['frac_zero_trace_columns']:.1%}), "
        f"real JET2 {int((read_jet_res(Path(reference))['trace'] <= 1e-9).sum())}/"
        f"{meta['msa_n_columns']}"
    )

    if args.out_validation:
        ensure_dir(args.out_validation.parent)
        report.to_csv(args.out_validation, sep="\t", index=False)
        meta["validation_path"] = str(args.out_validation)
        print(f"[jet_surrogate] wrote {args.out_validation}")
    if args.out_jet:
        write_jet_res(components, args.out_jet)
        meta["jet_res_path"] = str(args.out_jet)
    if args.out_manifest:
        meta["validation_reference"] = str(reference)
        meta["validation_spearman"] = {
            str(row["column"]): row["spearman_r"] for _, row in report.iterrows()
        }
        ensure_dir(args.out_manifest.parent)
        args.out_manifest.write_text(json.dumps(meta, indent=2, default=str))

    trace_rows = report.loc[report["column"] == "trace", "spearman_r"]
    if not trace_rows.empty and float(trace_rows.iloc[0]) < args.min_validation_spearman:
        print(
            f"RED FLAG: Spearman(trace) = {float(trace_rows.iloc[0]):.3f} < "
            f"{args.min_validation_spearman}; the surrogate is not tracking JET2.",
            file=sys.stderr,
        )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.validate_only:
        return run_validation(args)

    missing_required = [
        name for name, value in (("--msa", args.msa), ("--out-jet", args.out_jet))
        if value is None
    ]
    if missing_required:
        build_parser().error(
            f"{', '.join(missing_required)} are required unless --validate-only is given"
        )

    # Cache short-circuit. EVERY input and knob that changes the output must be compared,
    # not just the MSA: --out-jet has one name per lineage, so a second run with a
    # different structure (run_prescott_diversity.py --structure-role extra, --structure,
    # or --sasa-context monomer) writes to the same path and would otherwise be served the
    # previous structure's table verbatim -- 6WXB covers 485/566 residues and the
    # contemporary model covers 566/566, so the difference is silent and large.
    manifest_path = args.out_manifest or args.out_jet.with_suffix(".manifest.json")
    # A cache hit returns before the --out-components / --out-dssp blocks below, so
    # it may only be taken when every side output the caller asked for is ALREADY on
    # disk at the requested path.  Otherwise a first run without --out-components
    # followed by one with it would exit 0 having produced nothing, and the caller
    # (run_prescott_diversity.py asks for the components TSV on reruns) would see
    # success with no file.  Treat a missing requested output as a cache miss.
    side_outputs_present = all(
        path is None or Path(path).exists()
        for path in (args.out_components, args.out_dssp)
    )
    if (
        args.out_jet.exists()
        and manifest_path.exists()
        and side_outputs_present
        and not args.force
    ):
        try:
            cached = json.loads(manifest_path.read_text())
            cached_structure = cached.get("structure") or {}
            # CONTENT, not just path.  prepare_inputs.py rewrites
            # inputs/structure/<stem>_chain<C>_qnum.pdb in place on every run, so the
            # path is invariant under a changed --structure-offset, a re-prepared or
            # edited source structure, and two --structure files whose stem token
            # collides (Sequences/6WXB.cif and Sequences/6WXB-assembly1.cif both reduce
            # to 6WXB_chainA_qnum.pdb).  Comparing paths alone served the PREVIOUS
            # structure's table verbatim with exit 0 -- and, worse, skipped
            # build_jet_table's own frame check, so a structure whose residue numbering
            # was no longer in the query frame (the check refuses below 60% identity)
            # was accepted silently.  A manifest written before these fields existed has
            # None here and is treated as a miss, which is the safe direction.
            requested_context = args.context_pdb or args.pdb
            structure_matches = (
                str(cached_structure.get("pdb") or "") == str(args.pdb or "")
                and str(cached_structure.get("context_pdb") or "")
                == str(requested_context or "")
                and cached_structure.get("pdb_md5")
                == (md5_of(Path(args.pdb)) if args.pdb and Path(args.pdb).exists() else None)
                and cached_structure.get("context_pdb_md5")
                == (
                    md5_of(Path(requested_context))
                    if requested_context and Path(requested_context).exists()
                    else None
                )
            )
            if (
                cached.get("msa_md5") == md5_of(args.msa)
                and structure_matches
                and cached.get("weight_mode") == args.weight_mode
                and cached.get("trace_definition") == args.trace_definition
                # the manifest records trace_bootstraps as 0 under trace_definition=direct,
                # so only compare it where it actually influences the result
                and (
                    args.trace_definition != "bootstrap"
                    or cached.get("trace_bootstraps") == args.trace_bootstraps
                )
                and cached.get("trace_top_fraction") == args.trace_top_fraction
                and cached.get("trace_pseudocount") == args.trace_pseudocount
                and cached.get("trace_scale_quantile") == args.trace_scale_quantile
                and cached.get("sequence_weighting") == args.sequence_weighting
                and cached.get("seed") == args.seed
                and cached.get("pc_mode") == args.pc_mode
                # A table built from chain A is not a table for chain B: every
                # structural column (pc's RSA half, cv, ss, runlength) differs.
                and cached.get("structure_chain") == args.structure_chain
                and cached.get("cv_radius") == args.cv_radius
                and cached.get("cv_atom") == args.cv_atom
                and cached.get("max_coil_length") == args.max_coil_length
            ):
                # A cache hit must not become a way to smuggle a table past the
                # zero-trace ceiling: re-apply the current threshold to the
                # recorded fraction before serving it.
                cached_frac = cached.get("frac_zero_trace_columns")
                if (
                    args.max_zero_trace_fraction is not None
                    and cached_frac is not None
                    and float(cached_frac) > args.max_zero_trace_fraction
                ):
                    raise ZeroTraceError(
                        f"cached {args.out_jet} has {float(cached_frac):.1%} zero `trace` "
                        f"values, above --max-zero-trace-fraction "
                        f"{args.max_zero_trace_fraction}; rebuild it with "
                        f"--force --trace-top-fraction {DEFAULT_TRACE_TOP_FRACTION}"
                    )
                print(f"[jet_surrogate] cache hit, keeping {args.out_jet}")
                return 0
        except (json.JSONDecodeError, OSError):
            pass  # a corrupt manifest is just a cache miss

    components, meta = build_jet_table(
        args.msa,
        args.query,
        args.pdb,
        args.context_pdb,
        weight_mode=args.weight_mode,
        trace_definition=args.trace_definition,
        trace_bootstraps=args.trace_bootstraps,
        trace_top_fraction=args.trace_top_fraction,
        max_zero_trace_fraction=args.max_zero_trace_fraction,
        trace_pseudocount=args.trace_pseudocount,
        trace_scale_quantile=args.trace_scale_quantile,
        sequence_weighting=args.sequence_weighting,
        pc_mode=args.pc_mode,
        structure_chain=args.structure_chain,
        cv_radius=args.cv_radius,
        cv_atom=args.cv_atom,
        max_coil_length=args.max_coil_length,
        seed=args.seed,
    )

    write_jet_res(components, args.out_jet)
    meta["jet_res_path"] = str(args.out_jet)
    meta["jet_res_md5"] = md5_of(args.out_jet)

    if args.out_components:
        ensure_dir(args.out_components.parent)
        components.to_csv(args.out_components, sep="\t", index=False)
        meta["components_path"] = str(args.out_components)

    if args.out_dssp:
        ensure_dir(args.out_dssp.parent)
        components[["pos", "ss", "runlength"]].to_csv(args.out_dssp, index=False)
        meta["dssp_path"] = str(args.out_dssp)

    # Round-trip through escott's own parser before we declare success.
    reparsed = read_jet_res(args.out_jet)
    validate_jet_table(reparsed, expected_rows=len(components))
    identity_ok = np.allclose(
        reparsed["trace"].to_numpy(dtype=float),
        np.round(components["trace_emitted"].to_numpy(dtype=float), 4),
        atol=1e-6,
    )
    if not identity_ok:
        raise AssertionError("round-trip through escott's parser changed `trace`")
    meta["escott_parser_roundtrip_ok"] = True

    if args.validate_against:
        report = compare_to_reference(reparsed, args.validate_against)
        print(report.to_string(index=False))
        if args.out_validation:
            ensure_dir(args.out_validation.parent)
            report.to_csv(args.out_validation, sep="\t", index=False)
            meta["validation_path"] = str(args.out_validation)
        trace_rows = report.loc[report["column"] == "trace", "spearman_r"]
        if not trace_rows.empty:
            rho = float(trace_rows.iloc[0])
            meta["validation_trace_spearman"] = rho
            if rho < args.min_validation_spearman:
                print(
                    f"RED FLAG: Spearman(trace) = {rho:.3f} < "
                    f"{args.min_validation_spearman}; the surrogate is not tracking JET2.",
                    file=sys.stderr,
                )

    ensure_dir(manifest_path.parent)
    manifest_path.write_text(json.dumps(meta, indent=2, default=str))

    print(
        f"[jet_surrogate] wrote {args.out_jet} "
        f"({len(components)} rows, {meta['n_positions_without_structure']} without structure, "
        f"weight_mode={args.weight_mode})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
