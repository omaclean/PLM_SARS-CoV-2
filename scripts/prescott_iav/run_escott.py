#!/usr/bin/env python3
"""Drive ESCOTT/PRESCOTT and turn their output into PLM-style score matrices.

This is stage C of the influenza A H3N2 HA diversity-prediction pipeline.  Stage A
(``prepare_inputs.py``) writes the per-lineage query FASTA, the 566-column MSA and the
basal-lineage frequency file; stage B (``jet_surrogate.py``) writes the surrogate JET2
residue table.  This module consumes those, runs ``escott`` (and optionally ``prescott``),
and emits ``scores/<lineage_key>_<variant>_score_matrix.csv`` files that are byte-compatible
with the ``plm_cache/*_plm_probability_profile.csv`` format that
``run_mutational_accessibility.py`` already knows how to read.  Stage D therefore needs no
new I/O code at all -- it calls ``rma.normalise_plm_matrix`` and ``rma.infer_plm_source_sequence``
on our output exactly as it does on a real PLM cache file.

Everything here runs under the PRESCOTT conda env
(``/home3/oml4h/miniconda3/envs/PRESCOTT/bin/python3.10``); it deliberately imports no torch,
no esm and nothing from ``Functions_HuggingFace``, because those are unavailable there.

WHY the code is shaped the way it is -- the four upstream landmines it exists to defuse:

1. ESCOTT is a *CWD tool*.  ``escott.py:extractQuerySeq`` derives a token ``prot`` from the
   first FASTA header and then writes ``<prot>.fasta``, ``<prot>.pdb``, ``<prot>_jet.res``,
   ``pred.R``, ``computePred.R``, ``<alphabet>.txt`` and a multi-MB ``.RData`` as *bare
   filenames in the current directory*.  Two concurrent runs in one directory silently
   destroy each other.  Every job therefore gets its own working directory
   (:func:`escott_workdir`) and the inputs are copied in under names that provably cannot
   collide with ``<prot>.fasta`` / ``<prot>_jet.res`` (:func:`_stage_escott_inputs`).

2. ``escott.main()`` (escott.py:1500-1509) *unconditionally overwrites* ``--normweightmode``:
   ``tjet`` when no ``--pdbfile`` is given, ``sstjetormaxtwocomponent`` when one is.  The
   user's flag is discarded in both directions.  We exploit this: stage B already folds the
   structural terms into the ``trace`` column, so running without ``--pdbfile`` makes
   ``computePred.R``'s ``tjet`` branch read our precomputed weight verbatim.  ``--pdbfile``
   is still supported (:func:`run_escott_for_lineage`) for the verification run described in
   the plan, but it makes ESCOTT recompute the weight from its own DSSP pass and inner-join
   on ``pos``, which crashes on partial structural coverage -- hence the loud warning.

3. ESCOTT exits 0 even when the R step produced nothing (``escott.py:360`` throws the
   ``subprocess.call`` return code away).  We never trust the exit code;
   :func:`_assert_escott_succeeded` checks for the product file and greps the captured log
   for the handful of R failure signatures we have actually observed.

4. ``prescott.py`` quantises its output to two decimals (``'{:6.2f}'`` at prescott.py:952),
   collapsing ~15k mutations to ~97 distinct values.  Production numbers therefore come from
   our own full-precision reimplementation (:func:`prescott_v2_scores`); ``prescott.py`` is
   run only as a parity check (:func:`run_prescott_reference` + :func:`prescott_parity_check`).

Standalone use::

    /home3/oml4h/miniconda3/envs/PRESCOTT/bin/python3.10 scripts/prescott_iav/run_escott.py \
        --inputs-dir  Results/iav_prescott_diversity/smoke/inputs \
        --escott-workdir Results/iav_prescott_diversity/smoke/escott \
        --scores-dir  Results/iav_prescott_diversity/smoke/scores \
        --prescott-ref-dir Results/iav_prescott_diversity/smoke/prescott_ref \
        --lineage J_int
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata


# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

# Row order used by the PLM cache files this module has to imitate
# (Functions_HuggingFace.py:821).  run_mutational_accessibility.normalise_plm_matrix
# re-indexes to alphabetical order anyway, but matching the PLM writer byte-for-byte means a
# reviewer can diff an ESCOTT score matrix against a PLM one without any reordering step.
PLM_CACHE_AA_ORDER: Tuple[str, ...] = (
    "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
)

# computePred.R:12 -- ESCOTT's own row order, lowercase and alphabetical.  We reindex out of
# this into PLM_CACHE_AA_ORDER as soon as the matrix is parsed.
ESCOTT_AA_ORDER: Tuple[str, ...] = tuple("acdefghiklmnpqrstvwy")

# prescott.py:735 uses 999.0 as "no frequency information for this mutant".  Anything carrying
# the sentinel is left untouched by every equation, which is exactly what we want for mutants
# that were never observed in the basal panel (writing them as 0.0 would give log10(0) = -inf).
NO_FREQUENCY_SENTINEL = 999.0

# prescott.py:796 has an unconditional sys.exit(-1) inside the equation-4 branch.
SUPPORTED_PRESCOTT_EQUATIONS: Tuple[int, ...] = (1, 2, 3, 5)

# The share of a frequency file's records that must land on the ESCOTT grid before
# process_lineage will score with it.  See :func:`assert_frequency_frame`: a
# self-consistent inputs tree lands 100%, a whole-frame error lands 0% and a
# one-residue shift lands ~5%, so 50% separates them with room on both sides.
MIN_FREQUENCY_MATCH_FRACTION = 0.5

DEFAULT_ESCOTT_BIN = Path("/home3/oml4h/miniconda3/envs/PRESCOTT/bin/escott")
DEFAULT_PRESCOTT_BIN = Path("/home3/oml4h/miniconda3/envs/PRESCOTT/bin/prescott")
DEFAULT_PRESCOTT_ENV_BIN = Path("/home3/oml4h/miniconda3/envs/PRESCOTT/bin")

# Substrings that mean "the R half failed" even though escott returned 0.  Collected from
# actual failures: a jet.res with the wrong number of rows produces 'non-conformable
# arguments' out of computePred.R:140, a malformed FASTA raises 'bad FASTA format' out of
# escott.py:76, and a missing column raises 'No field called' inside seqinr.
ESCOTT_FAILURE_SIGNATURES: Tuple[str, ...] = (
    "non-conformable",
    "No field called",
    "bad FASTA format",
    "Error in ",
    "Execution halted",
    "cannot open file",
)

# Lineage -> escott-safe alphanumeric tag.  Only used to *predict* filenames when the MSA is
# not yet on disk; the authoritative token is always re-derived from the MSA header with
# escott's own rule (:func:`escott_prot_token`), because that is what escott itself will do.
LINEAGE_TAGS: Dict[str, str] = {
    "G.1": "G1",
    "J_int": "J",
    "J.2_int": "J2",
    "J.2.4": "J24",
    "K": "K",
}

# ``common.py`` is stage A/B's shared module and owns the canonical labels/keys/hashing.
# Import it so a later edit there cannot silently diverge from this stage; fall back to the
# local definitions below only if this file is being used standalone before that module lands.
try:  # pragma: no cover - depends on the sibling module being importable
    from . import common as _common  # type: ignore
except ImportError:  # running as a script, not as a package member
    try:
        import common as _common  # type: ignore
    except ImportError:
        _common = None  # type: ignore

if _common is not None:
    LINEAGE_TAGS = dict(_common.LINEAGE_TAGS)

# ``constants.py`` is dependency-free and is the single source of truth for the naming
# helpers stage A and stage C MUST agree on byte-for-byte: stage A writes
# ``K_parentJ2int_frequency.txt`` and stage C has to find that exact name, then emit a
# ``..._parentJ2int`` variant that stage D parses the suffix back out of.  It is imported
# directly (not via ``common``) so this still works if ``common`` is unavailable.
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

parse_edge_spec = _constants.parse_edge_spec
alternate_frequency_basename = _constants.alternate_frequency_basename
variant_parent_token = _constants.variant_parent_token


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def safe_label(label: str) -> str:
    """Mirror of ``Functions_HuggingFace._safe_label`` (line 3505).

    Not imported from ``Functions_HuggingFace`` directly because that module pulls in torch at
    module scope and this file must stay importable inside the PRESCOTT env.  Dots survive, so
    ``J.2_int`` stays ``J.2_int`` -- see :func:`dotfree_key` for why that matters.
    """
    if _common is not None:
        return _common.safe_label(label)
    return label.strip().replace(" ", "_").replace("/", "-")


def dotfree_key(lineage_key: str) -> str:
    """``J.2_int`` -> ``J_2_int``.

    ``prescott.py:902`` does ``os.path.splitext(args.outputfile)[0]``, which truncates at the
    *last* dot.  Passing ``-o J.2_int`` would therefore write ``J-details.csv``.  Every value
    we hand to ``-o`` -- and every filename we build from a lineage key -- goes through here.
    """
    if _common is not None:
        return _common.dot_free_key(lineage_key)
    return lineage_key.replace(".", "_")


def file_md5(path: Path, chunk_size: int = 1 << 20) -> str:
    """Streaming md5 -- the MSAs are ~4 MB and the .RData files far larger."""
    if _common is not None:
        return _common.md5_file(Path(path), chunk_size)
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_hash(paths: Sequence[Path], extra: Optional[Dict[str, object]] = None) -> str:
    """Content hash over a set of input files plus a dict of scalar arguments.

    This is the cache key for every stage below.  Hashing *content* rather than mtimes means a
    rerun after ``touch`` is free, and a rerun after an actual edit is not -- which is the
    behaviour we want on a shared filesystem where files get copied around.
    """
    digest = hashlib.md5()
    for path in paths:
        digest.update(str(Path(path).name).encode("utf-8"))
        digest.update(file_md5(Path(path)).encode("utf-8"))
    if extra:
        digest.update(json.dumps(extra, sort_keys=True, default=str).encode("utf-8"))
    return digest.hexdigest()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def escott_prot_token(fasta_path: Path) -> str:
    """Reproduce ``escott.py:76`` exactly: ``re.compile("[^A-Z0-9a-z]").split(header[1:])[0]``.

    We need this *before* running escott because every output filename is ``<prot>_...`` and
    the cache check has to know what to look for.  Reimplementing rather than guessing from
    the lineage tag means a header change in stage A can never desynchronise the two.
    """
    with open(fasta_path, "r") as handle:
        first_line = handle.readline()
    if not first_line.startswith(">"):
        raise ValueError(f"{fasta_path} does not start with '>' -- escott would raise 'bad FASTA format'")
    header = first_line[1:]
    token = _common.escott_prot_token(header) if _common is not None else re.compile("[^A-Z0-9a-z]").split(header)[0]
    if not token:
        raise ValueError(f"{fasta_path} header {first_line.strip()!r} yields an empty escott token")
    return token


def read_single_fasta(path: Path) -> Tuple[str, str]:
    """Return ``(header_without_gt, sequence)`` for a one-record FASTA, uppercased and ungapped."""
    header = None
    chunks: List[str] = []
    with open(path, "r") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    raise ValueError(f"{path} contains more than one record; expected exactly one")
                header = line[1:]
            elif line:
                chunks.append(line.strip().replace("-", "").replace(".", ""))
    if header is None:
        raise ValueError(f"{path} contains no FASTA record")
    return header, "".join(chunks).upper()


def prescott_env(extra_bin: Optional[Path] = None) -> Dict[str, str]:
    """Environment for escott/prescott subprocesses.

    ``escott`` shells out to ``Rscript`` and (with ``--pdbfile``) to ``mkdssp``, both resolved
    from ``PATH``; biotite's ``DsspApp`` does the same.  A bare interpreter invocation without
    the env's bin directory on ``PATH`` fails with ``FileNotFoundError: 'mkdssp'``.
    ``MPLBACKEND=Agg`` is mandatory because ``prescott.py`` imports pyplot at module load and
    compute nodes are headless.
    """
    env = dict(os.environ)
    bin_dir = str(extra_bin or DEFAULT_PRESCOTT_ENV_BIN)
    if bin_dir not in env.get("PATH", "").split(os.pathsep):
        env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    env["MPLBACKEND"] = "Agg"
    # A stale user R library can shadow the env's seqinr and make computePred.R fail in a way
    # that looks like a data problem.
    env["R_LIBS_USER"] = ""
    return env


# --------------------------------------------------------------------------------------
# Working directories and input staging
# --------------------------------------------------------------------------------------


def escott_workdir(root: Path, lineage_key: str) -> Path:
    """One isolated CWD per (lineage) escott job -- see landmine 1 in the module docstring."""
    return ensure_dir(Path(root) / lineage_key)


def _stage_escott_inputs(
    workdir: Path,
    msa_path: Path,
    jet_path: Path,
    prot: str,
) -> Tuple[Path, Path]:
    """Copy the MSA and the surrogate jet table into ``workdir`` under collision-proof names.

    Two names are forbidden inside the working directory:

    * ``<prot>.fasta`` -- ``escott.py:77-85`` opens it for *writing* to dump the extracted
      query, so an input of that name is destroyed before it is read.
    * ``<prot>_jet.res`` -- ``escott.py:1112-1113`` copies our ``--jetfile`` onto that name,
      and the structural modes then rewrite it in place (``escott.py:1187``).  Keeping the
      original under a different name means stage B's product is never mutated.
    """
    forbidden = {f"{prot}.fasta", f"{prot}_jet.res", f"{prot}.pdb"}

    staged_msa = workdir / f"ha_msa_{msa_path.stem}.fasta"
    staged_jet = workdir / jet_path.name
    for staged in (staged_msa, staged_jet):
        if staged.name in forbidden:
            raise ValueError(
                f"staged input {staged.name!r} collides with an escott-owned filename for prot={prot!r}; "
                "rename the stage A/B output"
            )

    shutil.copy2(msa_path, staged_msa)
    shutil.copy2(jet_path, staged_jet)
    return staged_msa, staged_jet


def _assert_escott_succeeded(workdir: Path, prot: str, log_text: str) -> Path:
    """Escott returns 0 on R failure, so verify the product instead of the exit code."""
    product = workdir / f"{prot}_normPred_evolCombi.txt"
    for signature in ESCOTT_FAILURE_SIGNATURES:
        if signature in log_text:
            offending = [line for line in log_text.splitlines() if signature in line]
            raise RuntimeError(
                f"escott reported an R-side failure in {workdir}: {offending[:3]}"
            )
    if not product.exists():
        tail = "\n".join(log_text.splitlines()[-25:])
        raise RuntimeError(
            f"escott produced no {product.name} in {workdir} despite exiting cleanly.\n"
            f"--- last 25 log lines ---\n{tail}"
        )
    return product


def run_escott_for_lineage(
    lineage_key: str,
    msa_path: Path,
    jet_path: Path,
    workdir: Path,
    *,
    escott_bin: Path = DEFAULT_ESCOTT_BIN,
    pdbfile: Optional[Path] = None,
    alphabet: str = "lw-i.7",
    max_coil_length: int = 5,
    cv_radius: Optional[float] = None,
    force: bool = False,
    env_bin: Optional[Path] = None,
) -> Path:
    """Run ESCOTT for one lineage and return the path to ``<prot>_normPred_evolCombi.txt``.

    Cached on the md5 of (MSA, jetfile) plus the escott arguments; a cache hit costs two
    file hashes and no subprocess.  Measured cost of a miss on the shipped BLAT alignment
    (13473 x 286): 17 s wall, 320 MB RSS.  The influenza job (6434 x 566) has a comparable
    N*L product.
    """
    workdir = ensure_dir(Path(workdir))
    msa_path = Path(msa_path)
    jet_path = Path(jet_path)
    prot = escott_prot_token(msa_path)

    escott_args = {
        "alphabet": alphabet,
        "max_coil_length": int(max_coil_length),
        "cv_radius": cv_radius,
        "pdbfile": Path(pdbfile).name if pdbfile else None,
        # normweightmode is NOT part of the key: escott overwrites it unconditionally
        # (escott.py:1504-1509), so recording the user's value would be a lie.
        "effective_normweightmode": "sstjetormaxtwocomponent" if pdbfile else "tjet",
    }
    hash_inputs = [msa_path, jet_path] + ([Path(pdbfile)] if pdbfile else [])
    input_md5 = content_hash(hash_inputs, escott_args)

    product = workdir / f"{prot}_normPred_evolCombi.txt"
    marker = workdir / "escott_exit.json"
    if not force and product.exists() and marker.exists():
        try:
            cached = json.loads(marker.read_text())
        except json.JSONDecodeError:
            cached = {}
        # 21 lines == one header + 20 amino-acid rows; anything else is a truncated write.
        n_lines = sum(1 for _ in open(product, "r"))
        if cached.get("input_md5") == input_md5 and n_lines == 21:
            print(f"@> [{lineage_key}] escott cache hit -> {product}")
            return product

    staged_msa, staged_jet = _stage_escott_inputs(workdir, msa_path, jet_path, prot)

    cmd: List[str] = [
        str(escott_bin),
        staged_msa.name,
        "--jetfile", staged_jet.name,
        "--alphabet", alphabet,
        "--maxcoillength", str(int(max_coil_length)),
    ]
    if cv_radius is not None:
        cmd += ["--cvrc", str(float(cv_radius))]
    if pdbfile is not None:
        # WHY the warning: with --pdbfile, escott forces 'sstjetormaxtwocomponent', re-runs
        # DSSP itself, and inner-joins its DSSP table onto our jet table on 'pos'
        # (escott.py:1166).  Any position missing from the structure is silently dropped and
        # the length assertion at escott.py:1185 then fails.  Only pass a structure with
        # 100% residue coverage of the query.
        print(
            f"@> [{lineage_key}] WARNING: --pdbfile forces normweightmode="
            "'sstjetormaxtwocomponent'; escott will RECOMPUTE the residue weight from its own "
            "DSSP pass and discard the structural terms already folded into our 'trace' column. "
            "This requires the structure to cover every query position."
        )
        cmd += ["--pdbfile", str(Path(pdbfile).resolve())]

    print(f"@> [{lineage_key}] running: {' '.join(cmd)}  (cwd={workdir})")
    started = time.perf_counter()
    completed = subprocess.run(
        cmd,
        cwd=str(workdir),
        env=prescott_env(env_bin),
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - started

    log_text = (completed.stdout or "") + "\n" + (completed.stderr or "")
    (workdir / "escott.log").write_text(log_text)
    product = _assert_escott_succeeded(workdir, prot, log_text)

    marker.write_text(json.dumps(
        {
            "returncode": completed.returncode,
            "wall_seconds": round(elapsed, 3),
            "input_md5": input_md5,
            "prot": prot,
            "cmd": cmd,
            "escott_args": escott_args,
            "msa_path": str(msa_path),
            "jet_path": str(jet_path),
        },
        indent=2,
    ))
    print(f"@> [{lineage_key}] escott finished in {elapsed:.1f}s -> {product}")
    return product


# --------------------------------------------------------------------------------------
# Parsing the ESCOTT product
# --------------------------------------------------------------------------------------


def read_escott_matrix(path: Path, expect_protein: Optional[str] = None) -> pd.DataFrame:
    """Parse ``<prot>_normPred_evolCombi.txt`` into a tidy 20 x L frame.

    The file is written by R's ``write.table`` with defaults, so the header carries L quoted
    column names while every data row carries L+1 fields (the row name plus L values);
    ``pd.read_table(sep=r"\\s+")`` resolves that asymmetry into the index correctly.  Row
    labels are the 20 amino acids lowercase and alphabetical (computePred.R:12); column labels
    are ``"<WT><pos>"`` with no ``--offset`` applied.  The wild-type cell of each column is the
    literal token ``NA``.

    Values are ``-normPred`` (pred.R:489): **negative is deleterious**, near-zero/positive is
    tolerated, and the magnitude of a column is proportional to that position's ``trace``
    weight (pred.R:487), so a ``trace == 0`` position yields an exactly-zero column.

    Returns a frame indexed by :data:`PLM_CACHE_AA_ORDER` with integer columns ``1..L``.  The
    wild-type sequence recovered from the column labels is attached as
    ``frame.attrs["wt_sequence"]``.
    """
    path = Path(path)
    raw = pd.read_table(path, sep=r"\s+")

    if raw.shape[0] != 20:
        raise ValueError(f"{path}: expected 20 amino-acid rows, found {raw.shape[0]}")
    index_upper = [str(value).strip().strip('"').upper() for value in raw.index]
    if sorted(index_upper) != sorted(aa.upper() for aa in ESCOTT_AA_ORDER):
        raise ValueError(f"{path}: unexpected row labels {index_upper}")

    # Column labels are "<WT><pos>"; recover both halves and check the positions are the
    # contiguous 1..L that computePred.R always emits.
    wt_letters: List[str] = []
    positions: List[int] = []
    for label in raw.columns:
        token = str(label).strip().strip('"')
        match = re.fullmatch(r"([A-Za-z])(\d+)", token)
        if match is None:
            raise ValueError(f"{path}: column label {token!r} is not '<WT><pos>'")
        wt_letters.append(match.group(1).upper())
        positions.append(int(match.group(2)))
    if positions != list(range(1, len(positions) + 1)):
        raise ValueError(f"{path}: column positions are not contiguous 1..{len(positions)}")

    frame = raw.copy()
    frame.index = pd.Index(index_upper)
    frame.columns = pd.Index(positions, dtype=int)
    frame = frame.loc[list(PLM_CACHE_AA_ORDER)].astype(float)

    # Exactly one NA per column, and it must sit on the wild-type row.  Anything else means we
    # have misread the file and every downstream number would be quietly wrong.
    nan_per_column = frame.isna().sum(axis=0)
    if set(nan_per_column.unique()) != {1}:
        bad = nan_per_column[nan_per_column != 1]
        raise ValueError(f"{path}: expected exactly one NA per column; offenders: {bad.to_dict()}")
    for position, wt in zip(positions, wt_letters):
        if not np.isnan(frame.at[wt, position]):
            raise ValueError(f"{path}: the NA in column {wt}{position} is not on row {wt}")

    wt_sequence = "".join(wt_letters)
    if expect_protein is not None and wt_sequence != expect_protein:
        mismatches = [
            f"{i + 1}:{a}!={b}"
            for i, (a, b) in enumerate(zip(wt_sequence, expect_protein))
            if a != b
        ]
        raise ValueError(
            f"{path}: wild-type sequence from column labels does not match the lineage reference "
            f"(len {len(wt_sequence)} vs {len(expect_protein)}); first mismatches: {mismatches[:10]}"
        )

    frame.attrs["wt_sequence"] = wt_sequence
    frame.attrs["source_path"] = str(path)
    return frame


def escott_wt_sequence(matrix: pd.DataFrame) -> str:
    """Wild-type sequence recovered by :func:`read_escott_matrix` (kept out of the data)."""
    sequence = matrix.attrs.get("wt_sequence")
    if not sequence:
        raise ValueError("matrix has no 'wt_sequence' attr -- was it produced by read_escott_matrix?")
    return str(sequence)


def fill_wildtype(matrix: pd.DataFrame, mode: str = "column_max") -> pd.DataFrame:
    """Replace the one NaN per column with a finite value.

    ``column_max`` (default) gives the wild type the best score in its own column, so after
    the softmax it is the most probable residue at that site -- the same semantics a masked-LM
    gives.  ``global_max`` reproduces ``prescott.py``'s own choice
    (``np.nanmax`` over the whole matrix, prescott.py:1030) and exists only so the parity check
    operates on identical numbers.  Because the softmax is per column, the two differ downstream
    only by a per-site constant, which every Method A/B metric is invariant to.
    """
    values = matrix.to_numpy(dtype=float, copy=True)
    rows, cols = np.where(np.isnan(values))
    if mode == "column_max":
        values[rows, cols] = np.nanmax(values, axis=0)[cols]
    elif mode == "global_max":
        values[rows, cols] = np.nanmax(values)
    else:
        raise ValueError(f"unknown wildtype fill mode {mode!r}")
    filled = pd.DataFrame(values, index=matrix.index, columns=matrix.columns)
    filled.attrs.update(matrix.attrs)
    return filled


def escott_to_probability(
    matrix: pd.DataFrame,
    temperature: float = 1.0,
    wildtype_fill: str = "column_max",
) -> pd.DataFrame:
    """Per-position softmax of the raw ESCOTT score -> a PLM-shaped probability matrix.

    The alpha sweep in ``Functions_HuggingFace`` feeds ``plm_prob`` straight into ``np.log``
    and ``np.log10`` (lines 2861-2862), so the score has to be strictly positive; raw ESCOTT
    values are negative and would produce NaN everywhere.

    ``P[a,i] = softmax_a(E[a,i] / T)`` is the transform that makes the sweep's arithmetic come
    out right rather than merely finite: the combined score becomes
    ``E[a,i]/T + alpha*log(mut_prob) + c_i`` with ``c_i`` constant per site, i.e. exactly the
    intended linear trade-off between an evolutionary log-fitness and a codon log-accessibility.
    Nothing is invented and nothing is destroyed.

    Note this uses the **raw** ESCOTT orientation (low = deleterious), not the rank-sorted
    ``1 - rank`` form that ``prescott.py`` emits (which is high = deleterious).  High ``P``
    therefore means tolerated, matching a PLM, so the expected correlation with observed
    diversity is positive and every existing figure reads the same way as the PLM run.
    """
    # `nan <= 0` is False, so a bare `temperature <= 0` lets --escott-temperature nan
    # (and inf) straight through: nan poisons the whole matrix and surfaces three lines
    # later as the misleading "softmax produced a non-positive probability", while inf
    # silently returns an exactly uniform 1/20 matrix, i.e. every site dead, with no
    # warning at all.  Test both ends explicitly.
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"escott temperature must be positive and finite, got {temperature}")

    filled = fill_wildtype(matrix, mode=wildtype_fill)
    scaled = filled.to_numpy(dtype=float) / float(temperature)
    # Per-column max subtraction for numerical stability; mathematically a no-op for softmax.
    scaled = scaled - scaled.max(axis=0, keepdims=True)
    exponentiated = np.exp(scaled)
    probabilities = exponentiated / exponentiated.sum(axis=0, keepdims=True)

    if not np.all(probabilities > 0.0):
        # Reached when T is small enough that exp() underflows to exactly zero. On real
        # ESCOTT output the widest column spans ~6 log-units, so anything below ~0.008
        # lands here; the bare message gave no hint that the temperature was the cause.
        spread = float(np.nanmax(np.nanmax(filled.to_numpy(dtype=float), axis=0)
                                 - np.nanmin(filled.to_numpy(dtype=float), axis=0)))
        raise AssertionError(
            "softmax produced a non-positive probability; log() downstream would fail. "
            f"temperature={temperature!r} is too small for this matrix: its widest column "
            f"spans {spread:.4g} score units, and exp() underflows to exactly 0 below "
            f"T = {spread / 700.0:.3g}."
        )
    column_sums = probabilities.sum(axis=0)
    if not np.allclose(column_sums, 1.0, atol=1e-9):
        raise AssertionError(f"softmax columns do not sum to 1 (max deviation {np.abs(column_sums - 1).max():.3e})")

    result = pd.DataFrame(probabilities, index=matrix.index, columns=matrix.columns)
    result.attrs.update(matrix.attrs)
    return result


def count_flat_columns(matrix: pd.DataFrame, tolerance: float = 1e-12) -> int:
    """Number of positions whose ESCOTT column is identically zero.

    ``pred.R:487`` multiplies each column by that position's ``trace`` before negating, so
    ``trace == 0`` produces an all-zero column that softmaxes to a flat 1/20.  We count them
    for the manifest but never drop them -- dropping would change ``n_sites_used`` relative to
    the PLM run and break comparability of every pooled metric.
    """
    values = np.nan_to_num(matrix.to_numpy(dtype=float), nan=0.0)
    return int((np.abs(values) <= tolerance).all(axis=0).sum())


def count_uniform_columns(matrix: pd.DataFrame, tolerance: float = 1e-12) -> int:
    """Number of positions whose softmax is uniform, i.e. whose column is CONSTANT.

    A strict superset of :func:`count_flat_columns`, and the number that actually
    answers "how many sites carry no rank information".  ``trace == 0`` is only one way
    to get there; the other is full conservation, where GEMME/ESCOTT gives every
    substitution at a position the same non-zero score.  Both softmax to exactly 1/20
    and are equally dead in every site-level metric, but ``count_flat_columns`` tests
    ``|value| <= tol`` and therefore sees none of the second kind.

    This is not hypothetical: in the PRESCOTT distribution's own real ESCOTT output
    (``data/MLH1_normPred_evolCombi.txt``) exactly 0 of 756 columns are all-zero and 8
    are constant at values between -5.18 and -6.12.  Those 8 dead sites were reported as
    zero dead sites in ``scores/score_variants.tsv``.

    The wild-type NaN is excluded rather than zero-filled -- unlike the trace-zero test,
    where a zero fill is the right answer -- because ``fill_wildtype`` puts the column
    maximum there, which for a constant column is that same constant.
    """
    values = matrix.to_numpy(dtype=float)
    spread = np.nanmax(values, axis=0) - np.nanmin(values, axis=0)
    return int((np.abs(spread) <= tolerance).sum())


def report_dead_columns(key: str, matrix: pd.DataFrame) -> Tuple[int, int]:
    """Print, and return, the two dead-site counts for one lineage.

    A "dead" position is one whose softmax is uniform, so its 20 scores carry no rank
    information and every site-level metric computed there is noise.  There are two ways
    to get one and only the first was ever reported:

    ``n_flat``
        ``trace == 0`` at that position, so ``pred.R:487`` zeroed the whole column.
    ``n_uniform``
        the column is CONSTANT -- which includes every ``n_flat`` column, plus the
        fully-conserved positions where ESCOTT gives every substitution the same non-zero
        score.  In the PRESCOTT distribution's own real ESCOTT output 8 of 756 columns
        are of the second kind and 0 are of the first, i.e. the old report said "no dead
        sites" about a protein with 1.06% of its positions dead.

    Split out of :func:`process_lineage` so the diagnostic can be tested without a live
    escott run; the counts are also written to ``scores/score_variants.tsv``.
    """
    n_flat = count_flat_columns(matrix)
    n_uniform = count_uniform_columns(matrix)
    width = matrix.shape[1]
    if n_flat:
        print(f"@> [{key}] {n_flat}/{width} positions have an all-zero ESCOTT column (trace == 0)")
    if n_uniform > n_flat:
        print(f"@> [{key}] {n_uniform}/{width} positions softmax to a uniform 1/20 "
              f"({n_flat} from trace == 0, {n_uniform - n_flat} from a constant non-zero "
              f"column); those sites are noise in every site-level metric")
    return n_flat, n_uniform


# --------------------------------------------------------------------------------------
# Frequency files and the PRESCOTT equations
# --------------------------------------------------------------------------------------


def load_frequency_file(path: Path) -> Dict[str, float]:
    """Read a PRESCOTT custom frequency file: two whitespace-separated columns, no header.

    This is the format ``prescott.py:1078`` parses with
    ``pd.read_csv(path, header=None, sep='\\s+')`` and then column-names ``['mutant','frequency']``.
    Note the *file extension must not be ``.csv``* -- prescott switches to the gnomAD parser on
    that suffix (prescott.py:982).
    """
    path = Path(path)
    frame = pd.read_csv(path, header=None, sep=r"\s+", names=["mutant", "frequency"])
    frame["mutant"] = frame["mutant"].astype(str).str.upper()
    if frame["mutant"].duplicated().any():
        duplicates = frame.loc[frame["mutant"].duplicated(), "mutant"].unique().tolist()
        raise ValueError(f"{path}: duplicate mutant keys would make prescott's lookup ambiguous: {duplicates[:10]}")
    # Coerce BEFORE comparing. A non-numeric token leaves the column as object dtype and
    # `frame["frequency"] <= 0` then dies with a bare TypeError naming neither the file
    # nor the row; and `nan <= 0` is False, so NaN and inf sailed through the guard.
    # A NaN frequency is the worst of the three because it fails silently: log10(nan) is
    # nan, `nan > Fc` and `nan <= Fc` are both False, so the mutant is counted in
    # n_mutants_with_frequency and then receives no penalty from any equation.
    numeric = pd.to_numeric(frame["frequency"], errors="coerce")
    unparsed = frame.loc[numeric.isna() & frame["frequency"].notna(), "mutant"].tolist()
    if unparsed:
        raise ValueError(
            f"{path}: non-numeric frequencies for {unparsed[:10]}; expected two "
            f"whitespace-separated columns 'MUTANT FREQUENCY'"
        )
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        offenders = frame.loc[~np.isfinite(numeric.to_numpy(dtype=float)), "mutant"].tolist()
        raise ValueError(
            f"{path}: non-finite frequencies present for {offenders[:10]}; a NaN would be "
            f"silently skipped by every PRESCOTT equation while still counting towards "
            f"n_mutants_with_frequency"
        )
    if (numeric <= 0).any():
        raise ValueError(f"{path}: non-positive frequencies present; log10 would be -inf/NaN")
    return dict(zip(frame["mutant"], numeric.astype(float)))


def build_log10_frequency_matrix(
    frequencies: Dict[str, float],
    matrix: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Broadcast a ``{mutant: frequency}`` map onto the 20 x L ESCOTT grid.

    Cells with no entry keep :data:`NO_FREQUENCY_SENTINEL`, which every equation skips -- the
    same convention ``prescott.py`` uses for mutants absent from the gnomAD file.  Returns the
    matrix plus a small report so the caller can see how many frequency records actually landed
    (a mutant whose wild-type letter disagrees with the ESCOTT column silently matches nothing,
    and that is the failure mode most worth surfacing).
    """
    wt_sequence = escott_wt_sequence(matrix)
    log10_frequency = pd.DataFrame(
        NO_FREQUENCY_SENTINEL,
        index=matrix.index,
        columns=matrix.columns,
        dtype=float,
    )

    matched = 0
    unmatched: List[str] = []
    for mutant, frequency in frequencies.items():
        match = re.fullmatch(r"([A-Z])(\d+)([A-Z])", mutant)
        if match is None:
            unmatched.append(mutant)
            continue
        wild, position, mutated = match.group(1), int(match.group(2)), match.group(3)
        if position < 1 or position > len(wt_sequence) or wt_sequence[position - 1] != wild:
            unmatched.append(mutant)
            continue
        if mutated not in log10_frequency.index or mutated == wild:
            unmatched.append(mutant)
            continue
        log10_frequency.at[mutated, position] = float(np.log10(frequency))
        matched += 1

    report = {
        "n_frequency_records": len(frequencies),
        "n_matched": matched,
        "n_unmatched": len(unmatched),
        "unmatched_examples": unmatched[:10],
    }
    if unmatched:
        print(
            f"@> WARNING: {len(unmatched)} frequency records did not match the ESCOTT frame "
            f"(examples: {unmatched[:5]}) -- check the reference frame used to build them"
        )
    return log10_frequency, report


def assert_frequency_frame(
    report: Dict[str, object],
    lineage_key: str,
    parent_lineage: Optional[str],
    frequency_path: Path,
    matrix: pd.DataFrame,
    min_match_fraction: float = MIN_FREQUENCY_MATCH_FRACTION,
) -> None:
    """Refuse a frequency file that is not in the ESCOTT column frame.

    ``build_log10_frequency_matrix`` is a pure broadcaster: a record whose position
    or wild-type letter disagrees with the ESCOTT frame lands nowhere, keeps the
    :data:`NO_FREQUENCY_SENTINEL`, and every PRESCOTT equation then skips that cell.
    If *no* record lands, ``prescott_v2_scores`` is the identity to within float
    interpolation error -- measured 5.6e-17 on the 20 x 72 fixture -- so every
    PRESCOTT variant becomes a numerical duplicate of the ESCOTT baseline while
    stage C still writes it under its own name, stage D still reports it as a
    separate model, and the only trace is a warning on stdout.

    That is the shape of the single most likely coordinate accident in this
    pipeline: HA is universally published in MATURE (H3) numbering while every
    frame here is the 566-aa HA0 translation, so a frequency file built one
    signal peptide out (16 residues) matches nothing at all.

    ``prepare_inputs.build_parent_frequency_file`` writes ``{wt}{pos}{aa}`` with
    ``wt = child_protein[pos - 1]`` and ESCOTT's wild-type sequence IS that same
    child protein, so a self-consistent tree always matches 100%.  A whole-frame
    error matches 0%; a one-residue shift matches only where two adjacent residues
    happen to be equal (~5% on HA).  The 50% floor sits between the two with an
    order of magnitude of clearance in both directions, so it cannot fire on a
    correct run and cannot miss a shifted one.
    """
    n_records = int(report.get("n_frequency_records") or 0)
    if n_records == 0:
        return
    matched = int(report.get("n_matched") or 0)
    fraction = matched / n_records
    if fraction >= min_match_fraction:
        return
    wt_sequence = escott_wt_sequence(matrix)
    raise ValueError(
        f"[{lineage_key}] frequency file {frequency_path} (parent {parent_lineage!r}) is not in "
        f"the ESCOTT column frame: only {matched}/{n_records} records "
        f"({fraction:.1%}) match, below the {min_match_fraction:.0%} floor.\n"
        f"  ESCOTT scored a {len(wt_sequence)}-residue protein starting "
        f"{wt_sequence[:12]!r}; positions in the frequency file must be 1-based indices "
        f"into THAT sequence.\n"
        f"  The usual cause is mature (H3) HA numbering, which is offset from this "
        f"pipeline's 566-aa HA0 frame by the signal peptide.\n"
        f"  Examples that did not land: {report.get('unmatched_examples')}\n"
        f"  Rebuild it with prepare_inputs.py, which derives every mutant label from the "
        f"same translated CDS ESCOTT is given."
    )


def escott_rank_scores(matrix: pd.DataFrame, wildtype_fill: str = "column_max") -> pd.DataFrame:
    """``1 - rankdata(E)/N`` over the whole matrix: PRESCOTT's own score space.

    This is ``prescott.py:670-686`` composed with ``prescott.py:948``, and it flips the sign
    convention: the result is **high = deleterious**, the opposite of raw ESCOTT.  Unlike the
    upstream implementation we do not push the result through ``'{:6.2f}'`` (prescott.py:952),
    which would collapse ~11k distinct values to ~97 and cripple every rank correlation
    downstream.
    """
    filled = fill_wildtype(matrix, mode=wildtype_fill)
    flat = filled.to_numpy(dtype=float).ravel()
    ranks = rankdata(flat, method="average") / float(flat.size)
    result = pd.DataFrame(
        (1.0 - ranks).reshape(filled.shape),
        index=filled.index,
        columns=filled.columns,
    )
    result.attrs.update(matrix.attrs)
    return result


def apply_prescott_equation(
    ranked: pd.DataFrame,
    log10_frequency: pd.DataFrame,
    coefficient: float,
    frequency_cutoff: float,
    equation: int = 2,
) -> pd.DataFrame:
    """Elementwise reimplementation of ``prescott.py:runPrescottModel`` at full precision.

    Only equations 1, 2, 3 and 5 are offered: equation 4's branch contains an unconditional
    ``sys.exit(-1)`` upstream (prescott.py:796) and has never been runnable.

    All four leave the :data:`NO_FREQUENCY_SENTINEL` cells untouched, and equations 1, 2 and 5
    additionally leave cells at or below ``Fc`` untouched -- matching the upstream control flow
    exactly, including the fact that equation 3 is the only one that also acts below the cutoff.
    """
    if equation not in SUPPORTED_PRESCOTT_EQUATIONS:
        raise ValueError(
            f"equation {equation} is not supported (upstream prescott.py:796 exits inside equation 4); "
            f"choose one of {SUPPORTED_PRESCOTT_EQUATIONS}"
        )
    if frequency_cutoff >= 0:
        raise ValueError(f"frequency cutoff must be negative (it is a log10 frequency), got {frequency_cutoff}")

    scores = ranked.to_numpy(dtype=float, copy=True)
    freq = log10_frequency.to_numpy(dtype=float)

    has_frequency = freq != NO_FREQUENCY_SENTINEL
    above_cutoff = has_frequency & (freq > frequency_cutoff)
    below_cutoff = has_frequency & (freq <= frequency_cutoff)

    if equation == 1:
        # prescott.py:747-757 -- note the penalty is freq*c/Fc, NOT c*(Fc-freq)/Fc.
        candidate = scores - freq * coefficient / frequency_cutoff
        scores[above_cutoff] = np.maximum(candidate[above_cutoff], 0.0)
    elif equation == 2:
        # prescott.py:759-768 -- the published default: penalise only over-frequent variants.
        candidate = scores - coefficient * (frequency_cutoff - freq) / frequency_cutoff
        scores[above_cutoff] = np.maximum(candidate[above_cutoff], 0.0)
    elif equation == 3:
        # prescott.py:770-786 -- symmetric: rare variants are pushed *up*, capped at 1.
        candidate = scores - coefficient * (frequency_cutoff - freq) / frequency_cutoff
        scores[above_cutoff] = np.maximum(candidate[above_cutoff], 0.0)
        scores[below_cutoff] = np.minimum(candidate[below_cutoff], 1.0)
    elif equation == 5:
        # prescott.py:812-819 -- hard reset: any over-frequent variant is declared benign.
        scores[above_cutoff] = 0.0

    result = pd.DataFrame(scores, index=ranked.index, columns=ranked.columns)
    result.attrs.update(ranked.attrs)
    return result


def prescott_v2_scores(
    matrix: pd.DataFrame,
    log10_frequency: pd.DataFrame,
    coefficient: float,
    frequency_cutoff: float,
    equation: int = 2,
    wildtype_fill: str = "column_max",
) -> pd.DataFrame:
    """Apply the PRESCOTT frequency penalty and map the result back onto the ESCOTT scale.

    PRESCOTT modifies the *rank-sorted* score, not the raw one.  Feeding rank-space numbers
    into the softmax would put ESCOTT and PRESCOTT on incomparable scales and make a single
    alpha grid meaningless, so we round-trip through the empirical quantile function of the raw
    matrix::

        R  = 1 - rank(E)/N                    # high = deleterious, full precision
        R' = equation(R, log10 f, c, Fc)      # the frequency penalty
        A  = 1 - R'                           # accessibility quantile in [0,1]
        E' = quantile(E_flat, A)              # back onto the raw ESCOTT scale

    For ``c = 0`` this is the identity to within float interpolation error, which is the
    property that makes the ESCOTT-vs-PRESCOTT comparison an honest ablation: the only thing
    that changes is the frequency penalty.
    """
    filled = fill_wildtype(matrix, mode=wildtype_fill)
    flat = np.sort(filled.to_numpy(dtype=float).ravel())

    ranked = escott_rank_scores(matrix, wildtype_fill=wildtype_fill)
    penalised = apply_prescott_equation(
        ranked, log10_frequency, coefficient, frequency_cutoff, equation=equation
    )

    accessibility = np.clip(1.0 - penalised.to_numpy(dtype=float), 0.0, 1.0)
    # PRESCOTT's rank fraction is rank/N (1/N .. 1), but numpy's quantile argument indexes
    # sorted positions as q*(N-1) (0 .. N-1).  Converting with (A*N - 1)/(N - 1) makes the
    # remap the exact inverse of escott_rank_scores, which is what gives the c = 0 identity
    # below; using A directly leaves a systematic ~1/N shift (0.055 on a 5720-cell matrix).
    n_cells = flat.size
    quantile_positions = np.clip((accessibility * n_cells - 1.0) / (n_cells - 1.0), 0.0, 1.0)
    remapped = np.quantile(flat, quantile_positions, method="linear")

    result = pd.DataFrame(remapped, index=matrix.index, columns=matrix.columns)
    result.attrs.update(matrix.attrs)
    return result


def count_clipped_to_zero(
    ranked: pd.DataFrame,
    penalised: pd.DataFrame,
    log10_frequency: pd.DataFrame,
) -> Dict[str, int]:
    """How many frequency-carrying cells the penalty drove to exactly 0.

    A grid point that clips a large fraction of its mutants has destroyed the rank information
    the sweep depends on; the plan flags anything above ~5%.
    """
    has_frequency = (log10_frequency.to_numpy(dtype=float) != NO_FREQUENCY_SENTINEL)
    clipped = has_frequency & (penalised.to_numpy(dtype=float) <= 0.0) & (ranked.to_numpy(dtype=float) > 0.0)
    return {
        "n_mutants_with_frequency": int(has_frequency.sum()),
        "n_clipped_to_zero": int(clipped.sum()),
    }


# --------------------------------------------------------------------------------------
# Writing the PLM-compatible score matrix
# --------------------------------------------------------------------------------------


def write_score_matrix(probabilities: pd.DataFrame, protein: str, out_csv: Path) -> Path:
    """Write a ``plm_cache``-format CSV that stage D can read with no new code.

    Layout produced by ``run_mutational_accessibility.py:1033-1043``: a first row labelled
    ``sequence`` holding the L residue letters, then the 20 probability rows, written with
    ``header=False`` so the column labels 1..L live only in the ``sequence`` row's own columns.
    ``Functions_HuggingFace.load_plm_probability_matrix`` reads it back with
    ``index_col=0, header=None``; ``rma.infer_plm_source_sequence`` recovers ``protein`` from
    row 0 and ``rma.normalise_plm_matrix`` drops that row and reindexes the rest.
    """
    if len(protein) != probabilities.shape[1]:
        raise ValueError(
            f"protein length {len(protein)} does not match matrix width {probabilities.shape[1]}"
        )
    if list(probabilities.index) != list(PLM_CACHE_AA_ORDER):
        probabilities = probabilities.loc[list(PLM_CACHE_AA_ORDER)]

    sequence_row = pd.DataFrame(
        [list(protein)],
        index=["sequence"],
        columns=probabilities.columns,
    )
    out_csv = Path(out_csv)
    ensure_dir(out_csv.parent)
    pd.concat([sequence_row, probabilities], axis=0).to_csv(out_csv, header=False)
    return out_csv


def write_raw_matrix(matrix: pd.DataFrame, out_tsv: Path) -> Path:
    """Full-precision dump of the raw ESCOTT values, for auditing and for the scale report."""
    out_tsv = Path(out_tsv)
    ensure_dir(out_tsv.parent)
    matrix.to_csv(out_tsv, sep="\t", float_format="%.10g")
    return out_tsv


# --------------------------------------------------------------------------------------
# prescott.py reference run and parity check
# --------------------------------------------------------------------------------------


def run_prescott_reference(
    lineage_key: str,
    escott_txt: Path,
    query_fasta: Path,
    frequency_txt: Path,
    out_dir: Path,
    *,
    coefficient: float,
    frequency_cutoff: float,
    equation: int = 2,
    prescott_bin: Path = DEFAULT_PRESCOTT_BIN,
    force: bool = False,
    env_bin: Optional[Path] = None,
) -> Path:
    """Run the published ``prescott`` tool once, purely as a cross-check on our arithmetic.

    Three upstream constraints are enforced here rather than discovered at runtime:

    * ``-o`` must be dot-free.  ``prescott.py:902`` does ``os.path.splitext(outputfile)[0]``,
      so ``-o J.2_int`` would write ``J-details.csv``.
    * The run needs its own CWD.  ``prescott.py:930/949`` write ``<protein>_singleline.txt``
      and ``<protein>_singleline_1-ranksort.txt`` as bare CWD filenames and delete them at
      :1160-1163, so concurrent runs clobber each other.
    * The frequency file must not end in ``.csv`` -- that suffix routes into the gnomAD parser.

    Returns the path to ``<out>-details.csv``, the per-mutant table.  (There is no
    ``prescott-scores.txt``; the tool writes ``<out>.csv``, a 2-dp matrix, and
    ``<out>-details.csv``.)
    """
    out_dir = ensure_dir(Path(out_dir))
    stem = f"prescott_{dotfree_key(lineage_key)}"
    if "." in stem:
        raise ValueError(f"prescott -o value {stem!r} still contains a dot; splitext would truncate it")
    if Path(frequency_txt).suffix.lower() == ".csv":
        raise ValueError(
            f"{frequency_txt} ends in .csv, which routes prescott into the gnomAD parser; "
            "rename it to .txt"
        )

    details = out_dir / f"{stem}-details.csv"
    marker = out_dir / f"{stem}_exit.json"
    input_md5 = content_hash(
        [Path(escott_txt), Path(query_fasta), Path(frequency_txt)],
        {"coefficient": coefficient, "frequency_cutoff": frequency_cutoff, "equation": equation},
    )
    if not force and details.exists() and marker.exists():
        try:
            if json.loads(marker.read_text()).get("input_md5") == input_md5:
                print(f"@> [{lineage_key}] prescott reference cache hit -> {details}")
                return details
        except json.JSONDecodeError:
            pass

    # prescott reads the ESCOTT matrix by name and derives its internal `protein` label from
    # that basename, so stage the inputs beside the outputs to keep the CWD self-contained.
    staged_escott = out_dir / Path(escott_txt).name
    staged_query = out_dir / Path(query_fasta).name
    staged_freq = out_dir / Path(frequency_txt).name
    for source, target in ((escott_txt, staged_escott), (query_fasta, staged_query), (frequency_txt, staged_freq)):
        if Path(source).resolve() != target.resolve():
            shutil.copy2(source, target)

    # prescott.py:1144-1158 indexes the query sequence positionally against the matrix width;
    # a mismatch is an IndexError deep inside the CSV writer, so check it up front.
    _, query_sequence = read_single_fasta(staged_query)
    matrix_width = pd.read_table(staged_escott, sep=r"\s+").shape[1]
    if len(query_sequence) != matrix_width:
        raise ValueError(
            f"query {staged_query.name} has {len(query_sequence)} residues but the ESCOTT matrix has "
            f"{matrix_width} columns; prescott would IndexError"
        )

    cmd = [
        str(prescott_bin),
        "-e", staged_escott.name,
        "-s", staged_query.name,
        "-g", staged_freq.name,
        "-o", stem,
        "--equation", str(int(equation)),
        "-c", str(float(coefficient)),
        "-f", str(float(frequency_cutoff)),
        "--escottformat", "gemme",
    ]
    # NEVER pass --usefrequencies false: prescott.py:1103-1107 then leaves the `position`
    # column empty and the matrix writer raises IndexError.

    print(f"@> [{lineage_key}] running: {' '.join(cmd)}  (cwd={out_dir})")
    completed = subprocess.run(
        cmd,
        cwd=str(out_dir),
        env=prescott_env(env_bin),
        capture_output=True,
        text=True,
    )
    log_text = (completed.stdout or "") + "\n" + (completed.stderr or "")
    (out_dir / f"{stem}.log").write_text(log_text)
    if completed.returncode != 0 or not details.exists():
        tail = "\n".join(log_text.splitlines()[-25:])
        raise RuntimeError(f"prescott failed (rc={completed.returncode}) in {out_dir}\n{tail}")

    marker.write_text(json.dumps({"input_md5": input_md5, "cmd": cmd, "returncode": completed.returncode}, indent=2))
    return details


def read_prescott_details(path: Path) -> pd.DataFrame:
    """Parse ``<out>-details.csv``.

    Columns written by ``prescott.py:1131``: ``mutant, ESCOTT, protein, log10frequency, labels,
    position, Selected Population, PRESCOTT``.  Both score columns are **high = deleterious**
    (the ``1 - rankSortData`` orientation) and quantised to two decimals -- verified
    empirically: a 5720-mutant BLAT run yields 97 distinct ESCOTT values and
    ``spearman(raw normPred, ESCOTT column) = -0.99995``.
    """
    frame = pd.read_csv(Path(path))
    expected = {"mutant", "ESCOTT", "PRESCOTT", "log10frequency"}
    missing = expected - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing expected columns {sorted(missing)}")
    frame["mutant"] = frame["mutant"].astype(str).str.upper()
    return frame


def prescott_parity_check(
    matrix: pd.DataFrame,
    log10_frequency: pd.DataFrame,
    details: pd.DataFrame,
    *,
    coefficient: float,
    frequency_cutoff: float,
    equation: int = 2,
    tolerance: float = 0.011,
) -> pd.DataFrame:
    """Compare our full-precision PRESCOTT against the tool's own 2-dp output.

    Both sides are computed with the ``global_max`` wild-type fill because that is what
    ``prescott.py:1030`` does (``gemmeDF.to_numpy(na_value=np.nanmax(...))``); using our
    production ``column_max`` fill here would produce a spurious mismatch.  Ours is rounded to
    two decimals before comparison, so the tolerance only has to absorb rounding ties.
    """
    ranked = escott_rank_scores(matrix, wildtype_fill="global_max")
    penalised = apply_prescott_equation(
        ranked, log10_frequency, coefficient, frequency_cutoff, equation=equation
    )
    wt_sequence = escott_wt_sequence(matrix)

    rows: List[Dict[str, object]] = []
    for _, record in details.iterrows():
        mutant = str(record["mutant"])
        match = re.fullmatch(r"([A-Z])(\d+)([A-Z])", mutant)
        if match is None:
            continue
        wild, position, mutated = match.group(1), int(match.group(2)), match.group(3)
        if position > len(wt_sequence) or wt_sequence[position - 1] != wild or mutated == wild:
            continue  # wild-type self-mutations carry the NA-filled cell; not comparable
        rows.append(
            {
                "mutant": mutant,
                "ours_escott": round(float(ranked.at[mutated, position]), 2),
                "theirs_escott": float(record["ESCOTT"]),
                "ours_prescott": round(float(penalised.at[mutated, position]), 2),
                "theirs_prescott": float(record["PRESCOTT"]),
            }
        )

    report = pd.DataFrame(rows)
    if report.empty:
        raise ValueError("parity check produced no comparable mutants; check the reference frame")
    report["abs_delta_escott"] = (report["ours_escott"] - report["theirs_escott"]).abs()
    report["abs_delta_prescott"] = (report["ours_prescott"] - report["theirs_prescott"]).abs()
    report.attrs["max_abs_delta_escott"] = float(report["abs_delta_escott"].max())
    report.attrs["max_abs_delta_prescott"] = float(report["abs_delta_prescott"].max())
    report.attrs["passed"] = bool(
        report["abs_delta_escott"].max() <= tolerance and report["abs_delta_prescott"].max() <= tolerance
    )
    return report


# --------------------------------------------------------------------------------------
# Variant grid
# --------------------------------------------------------------------------------------


def build_variant_name(
    equation: int,
    coefficient: float,
    frequency_cutoff_k: int,
    parent_lineage: str,
) -> str:
    """``PRESCOTT_eq2_c0p50_k1_parentJ24``.

    The variant name becomes the ``model`` column of every downstream table and part of every
    filename, so it must be dot-free and filesystem-safe.  Dots in the coefficient become 'p'
    and the parent key is flattened through :func:`dotfree_key` first.
    """
    # constants.variant_parent_token, not a local re-derivation: prepare_inputs names the
    # alternate frequency file with the same function, and stage D reads the suffix back off
    # the variant name to tell a primary model from a parent-sensitivity row.
    parent_token = variant_parent_token(parent_lineage)
    return f"PRESCOTT_eq{int(equation)}_c{float(coefficient):.2f}_k{int(frequency_cutoff_k)}_parent{parent_token}".replace(".", "p")


def parse_grid(spec: str, cast) -> List:
    """Parse a comma-separated CLI grid like ``"0.25,0.5,1.0"``."""
    return [cast(token.strip()) for token in str(spec).split(",") if token.strip()]


def resolve_frequency_cutoff(
    lineage_key: str,
    k: int,
    inputs_dir: Path,
    fallback: float,
    mode: str = "depth_scaled",
    report_key: Optional[str] = None,
    meta_path_override: Optional[Path] = None,
) -> Tuple[float, Optional[float]]:
    """Resolve ``Fc = log10(k / median_parent_depth)`` from stage A's frequency report.

    WHY not the published ``Fc = -4.0``: our basal panels span 229 to 27452 sequences, a 2.1
    log-unit spread in ``log10(1/N)``.  With a fixed cutoff, a singleton in the 229-sequence
    G.1 panel already incurs 0.41 of penalty while a doubleton in the 27452-sequence J.2_int
    panel incurs none -- the two panels are simply not commensurable.  With
    ``Fc = log10(1/N)`` the equation-2 penalty collapses to ``c * log_N(count)``: exactly 0 for
    a singleton and exactly ``c`` for a fixed variant, independent of depth.

    Returns ``(frequency_cutoff, median_depth_or_None)``.
    """
    report_path = Path(inputs_dir) / "frequency" / "frequency_report.json"
    median_depth: Optional[float] = None
    # `report_key` selects WHICH parent edge's depth to use. It matters: the whole
    # point of Fc = log10(k / median_depth) is that it is calibrated to the panel
    # the frequencies actually came from, and an alternate parent is a different
    # panel with a different depth (J.2_int is 27452 sequences, J.2.4 is 5561), so
    # reusing the primary's Fc for a sensitivity variant would silently compare
    # two models under two different penalty scales.
    lookup = report_key or lineage_key
    if report_path.exists():
        report = json.loads(report_path.read_text())
        # prepare_inputs keys the report by lineage *label* for the primary edge and by
        # the frequency-file stem (K_parentJ2int_frequency) for an alternate one;
        # safe_label is the identity for every label in the guide, but accept the key
        # form too so a future rename cannot silently fall through to the -4.0 default.
        entry = report.get(lookup) or report.get(safe_label(lookup)) or {}
        if isinstance(entry, dict):
            for depth_key in ("median_mapped_depth", "median_depth", "parent_median_depth"):
                if entry.get(depth_key) is not None:
                    median_depth = float(entry[depth_key])
                    break
            # prepare_inputs.resolve_frequency_cutoffs already resolved Fc per k under the same
            # mode; reuse its number verbatim rather than recomputing, so the value that went
            # into the frequency file and the value used here can never disagree.
            cutoffs = entry.get("frequency_cutoffs")
            if isinstance(cutoffs, dict) and str(k) in cutoffs and entry.get("frequency_cutoff_mode", mode) == mode:
                return float(cutoffs[str(k)]), median_depth

    if mode == "fixed":
        return float(fallback), median_depth

    if median_depth is None:
        # Fall back to the per-mutant metadata table, which carries the raw depths.
        meta_path = (
            Path(meta_path_override) if meta_path_override
            else Path(inputs_dir) / "frequency" / f"{lineage_key}_parent_frequency_meta.tsv"
        )
        if meta_path.exists():
            meta = pd.read_csv(meta_path, sep="\t")
            if "depth" in meta.columns and len(meta):
                median_depth = float(meta["depth"].median())

    if median_depth is None or median_depth <= 0:
        print(
            f"@> WARNING: no parent depth available for {lookup}; falling back to "
            f"--frequency-cutoff {fallback}"
        )
        return float(fallback), None

    return float(np.log10(float(k) / median_depth)), median_depth


# --------------------------------------------------------------------------------------
# Input resolution and the per-lineage pipeline
# --------------------------------------------------------------------------------------


def resolve_lineage_inputs(inputs_dir: Path, lineage_label: str) -> Dict[str, object]:
    """Locate stage A/B products for one lineage and fail loudly with a fix-it message."""
    inputs_dir = Path(inputs_dir)
    key = safe_label(lineage_label)
    paths = {
        "query": inputs_dir / "query" / f"{key}_query.fasta",
        "msa": inputs_dir / "msa" / f"msa_{key}.fasta",
        "jet": inputs_dir / "jet" / f"{key}_surrogate_jet.res",
        "frequency": inputs_dir / "frequency" / f"{key}_parent_frequency.txt",
    }
    missing = [str(path) for name, path in paths.items() if name != "frequency" and not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"stage A/B outputs missing for lineage {lineage_label!r}: {missing}\n"
            f"run prepare_inputs.py and jet_surrogate.py first (inputs-dir={inputs_dir})"
        )

    _, protein = read_single_fasta(paths["query"])
    return {
        "lineage_label": lineage_label,
        "lineage_key": key,
        "lineage_key_safe": dotfree_key(key),
        "protein": protein,
        "prot_token": escott_prot_token(paths["msa"]),
        "query_path": paths["query"],
        "msa_path": paths["msa"],
        "jet_path": paths["jet"],
        "frequency_path": paths["frequency"] if paths["frequency"].exists() else None,
        "alternate_frequency_paths": resolve_alternate_frequency_paths(inputs_dir, key),
    }


def resolve_alternate_frequency_paths(inputs_dir: Path, lineage_key: str) -> Dict[str, Dict[str, object]]:
    """Alternate-parent frequency files stage A wrote for one lineage.

    Returns ``{parent_lineage: {"path": ..., "report_key": ..., "meta_path": ...}}``.

    Preferred source is ``inputs_manifest.json``'s ``frequency_index``, because it
    records the *parent label* explicitly.  The filename alone cannot be inverted:
    ``variant_parent_token`` strips underscores and dots, so ``K_parentJ2int_
    frequency.txt`` could in principle come from ``J.2_int`` or ``J2int``.  The
    glob fallback exists only for trees prepared before the index was written, and
    it reports the token rather than guessing a label.
    """
    freq_dir = Path(inputs_dir) / "frequency"
    out: Dict[str, Dict[str, object]] = {}

    manifest_path = Path(inputs_dir) / "inputs_manifest.json"
    if manifest_path.exists():
        try:
            index = json.loads(manifest_path.read_text()).get("frequency_index") or {}
        except json.JSONDecodeError:
            index = {}
        for report_key, entry in index.items():
            if not isinstance(entry, dict) or entry.get("is_primary_parent", True):
                continue
            if str(entry.get("lineage_key")) != lineage_key:
                continue
            path = Path(str(entry.get("frequency_path", "")))
            if not path.exists():
                print(f"@> WARNING: manifest lists {path} for the alternate parent "
                      f"{entry.get('parent_lineage')} but it is missing on disk")
                continue
            out[str(entry["parent_lineage"])] = {
                "path": path,
                "report_key": report_key,
                "meta_path": Path(str(entry.get("meta_path", ""))),
            }
        if out:
            return out

    for path in sorted(freq_dir.glob(f"{lineage_key}_parent*_frequency.txt")):
        stem = path.name[: -len(".txt")]
        token = stem[len(f"{lineage_key}_parent"): -len("_frequency")]
        if not token:  # the primary file, `<key>_parent_frequency.txt`
            continue
        out[token] = {
            "path": path,
            "report_key": stem,
            "meta_path": freq_dir / f"{stem}_meta.tsv",
        }
    return out


def resolve_parent_map(inputs_dir: Path, override: Optional[str] = None) -> Dict[str, str]:
    """Read the parent map stage A resolved, optionally patched by a CLI override.

    We deliberately do not re-derive the clade topology here: stage A already wrote the
    resolved map into ``inputs_manifest.json`` and that is the single source of truth for the
    whole run, so a mismatch between stages is impossible.
    """
    parent_map: Dict[str, str] = {}
    manifest_path = Path(inputs_dir) / "inputs_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        parent_map = dict(manifest.get("parent_map", {}))
    if override:
        for token in override.split(","):
            token = token.strip()
            if not token:
                continue
            child, _, parent = token.partition("=")
            parent_map[child.strip()] = parent.strip()
    return parent_map


def resolve_sensitivity_map(
    inputs_dir: Path,
    override: Optional[str] = None,
    disable: bool = False,
) -> Dict[str, str]:
    """The alternate parent edges to score, defaulting to whatever stage A prepared.

    Defaulting to the manifest rather than to ``{}`` is deliberate: the frequency
    file for an alternate edge is expensive to build and useless if nothing scores
    it, so if stage A went to the trouble of writing one, stage C should use it
    unless explicitly told not to.  ``--sensitivity-parent-map`` overrides the
    manifest outright; ``--no-parent-sensitivity`` wins over both.
    """
    if disable:
        return {}
    if override:
        return dict(parse_edge_spec(override))

    manifest_path = Path(inputs_dir) / "inputs_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return {}
    return {str(child): str(parent)
            for child, parent in (manifest.get("sensitivity_parent_map") or {}).items()}


def process_lineage(
    lineage_label: str,
    inputs_dir: Path,
    escott_root: Path,
    scores_dir: Path,
    *,
    parent_lineage: Optional[str] = None,
    alternate_parents: Sequence[str] = (),
    diagnostics_dir: Optional[Path] = None,
    escott_bin: Path = DEFAULT_ESCOTT_BIN,
    prescott_bin: Path = DEFAULT_PRESCOTT_BIN,
    pdbfile: Optional[Path] = None,
    alphabet: str = "lw-i.7",
    max_coil_length: int = 5,
    cv_radius: Optional[float] = None,
    temperature: float = 1.0,
    coefficients: Sequence[float] = (0.25, 0.5, 1.0),
    equations: Sequence[int] = (2,),
    frequency_cutoff_ks: Sequence[int] = (1,),
    frequency_cutoff_mode: str = "depth_scaled",
    frequency_cutoff_fallback: float = -4.0,
    prescott_ref_dir: Optional[Path] = None,
    force: bool = False,
    env_bin: Optional[Path] = None,
) -> List[Dict[str, object]]:
    """Run ESCOTT for one lineage and emit every score matrix in the variant grid.

    Returns one row per variant, ready to be concatenated into ``scores/score_variants.tsv``.

    ``parent_lineage`` is the PRIMARY basal panel the frequency prior comes from.
    ``alternate_parents`` names additional basal panels to score the same lineage
    against; each needs its own frequency file from ``prepare_inputs.py
    --sensitivity-parent-map`` and produces its own ``_parent<TOK>``-suffixed
    variants, which is the mechanism behind ``--parent-sensitivity``.

    ``diagnostics_dir`` is where the prescott parity table is written.  It is an
    explicit argument rather than ``scores_dir.parent / "tables" / "diagnostics"``
    because ``--scores-dir`` can point outside the output directory (sharing score
    matrices between runs is the whole reason it exists), and the parity table
    belongs with the run that produced it, not with the matrices.
    """
    inputs = resolve_lineage_inputs(inputs_dir, lineage_label)
    key = str(inputs["lineage_key"])
    protein = str(inputs["protein"])
    scores_dir = ensure_dir(Path(scores_dir))

    workdir = escott_workdir(Path(escott_root), key)
    escott_txt = run_escott_for_lineage(
        key,
        Path(inputs["msa_path"]),
        Path(inputs["jet_path"]),
        workdir,
        escott_bin=escott_bin,
        pdbfile=pdbfile,
        alphabet=alphabet,
        max_coil_length=max_coil_length,
        cv_radius=cv_radius,
        force=force,
        env_bin=env_bin,
    )

    matrix = read_escott_matrix(escott_txt, expect_protein=protein)
    n_flat, n_uniform = report_dead_columns(key, matrix)
    write_raw_matrix(matrix, scores_dir / f"{key}_ESCOTT_raw.tsv")

    variant_rows: List[Dict[str, object]] = []

    # --- the ESCOTT baseline variant -------------------------------------------------
    escott_probabilities = escott_to_probability(matrix, temperature=temperature)
    escott_csv = write_score_matrix(escott_probabilities, protein, scores_dir / f"{key}_ESCOTT_score_matrix.csv")
    variant_rows.append(
        {
            "variant": "ESCOTT",
            "lineage": lineage_label,
            "lineage_key": key,
            "parent_lineage": None,
            # Written explicitly rather than left to pandas' NaN fill so that the
            # column has a stable dtype-independent meaning for stage D: the
            # ESCOTT baseline is conditioned on no parent at all, which is not the
            # same as "an unrecorded parent".
            "is_primary_parent": None,
            "frequency_path": None,
            "equation": None,
            "coefficient": None,
            "frequency_cutoff": None,
            "frequency_cutoff_k": None,
            "parent_median_depth": None,
            "temperature": float(temperature),
            "n_flat_columns": n_flat,
            "n_uniform_columns": n_uniform,
            "n_mutants_with_frequency": 0,
            "n_clipped_to_zero": 0,
            "score_matrix_path": str(escott_csv),
            "md5": file_md5(escott_csv),
        }
    )

    # --- the PRESCOTT grid, once per parent edge ---------------------------------------
    frequency_path = inputs["frequency_path"]
    if frequency_path is None:
        print(
            f"@> [{key}] no parent frequency file at {inputs_dir}/frequency/{key}_parent_frequency.txt; "
            "emitting the ESCOTT variant only (expected for input-only lineages such as G.1)"
        )
        return variant_rows

    # Each entry is (parent_label, frequency_file, report_key, meta_path, is_primary).
    # The alternate edges are what makes --parent-sensitivity a real analysis rather
    # than a manifest claim: this loop emits a SECOND, independently named score
    # matrix per grid point, conditioned on a different basal panel, so the two
    # candidate parents for K can be compared as ordinary model rows downstream.
    available_alternates: Dict[str, Dict[str, object]] = dict(
        inputs.get("alternate_frequency_paths") or {}  # type: ignore[arg-type]
    )
    parent_edges: List[Tuple[Optional[str], Path, str, Optional[Path], bool]] = [
        (parent_lineage, Path(frequency_path), key, None, True)
    ]
    for alt_parent in alternate_parents:
        if alt_parent == parent_lineage:
            print(f"@> [{key}] alternate parent {alt_parent} equals the primary parent; skipping")
            continue
        entry = available_alternates.get(alt_parent)
        if entry is None:
            raise FileNotFoundError(
                f"[{key}] alternate parent {alt_parent!r} was requested but stage A wrote no "
                f"frequency file for it. Expected "
                f"{Path(inputs_dir) / 'frequency' / (alternate_frequency_basename(key, alt_parent) + '.txt')}; "
                f"re-run prepare_inputs.py with --sensitivity-parent-map "
                f"'{lineage_label}={alt_parent}'. Available alternates: {sorted(available_alternates)}"
            )
        parent_edges.append(
            (
                alt_parent,
                Path(str(entry["path"])),
                str(entry["report_key"]),
                Path(str(entry["meta_path"])) if entry.get("meta_path") else None,
                False,
            )
        )

    primary_log10_frequency = None
    for edge_parent, edge_freq_path, edge_report_key, edge_meta_path, is_primary in parent_edges:
        frequencies = load_frequency_file(edge_freq_path)
        log10_frequency, frequency_report = build_log10_frequency_matrix(frequencies, matrix)
        assert_frequency_frame(frequency_report, key, edge_parent, edge_freq_path, matrix)
        if is_primary:
            primary_log10_frequency = log10_frequency
        tag = "" if is_primary else " [SENSITIVITY]"
        print(f"@> [{key}] frequency file{tag} ({edge_parent}) {edge_freq_path.name}: {frequency_report}")

        for k in frequency_cutoff_ks:
            cutoff, median_depth = resolve_frequency_cutoff(
                key, k, Path(inputs_dir), frequency_cutoff_fallback, mode=frequency_cutoff_mode,
                report_key=edge_report_key, meta_path_override=edge_meta_path,
            )
            print(f"@> [{key}]{tag} parent={edge_parent} k={k} -> Fc={cutoff:.4f} "
                  f"(median parent depth {median_depth})")
            for equation in equations:
                for coefficient in coefficients:
                    variant = build_variant_name(equation, coefficient, k, edge_parent or "NA")
                    remapped = prescott_v2_scores(
                        matrix, log10_frequency, coefficient, cutoff, equation=equation
                    )
                    probabilities = escott_to_probability(remapped, temperature=temperature)
                    out_csv = write_score_matrix(
                        probabilities, protein, scores_dir / f"{key}_{variant}_score_matrix.csv"
                    )

                    ranked = escott_rank_scores(matrix)
                    penalised = apply_prescott_equation(
                        ranked, log10_frequency, coefficient, cutoff, equation=equation
                    )
                    clipping = count_clipped_to_zero(ranked, penalised, log10_frequency)
                    if clipping["n_mutants_with_frequency"]:
                        fraction = clipping["n_clipped_to_zero"] / clipping["n_mutants_with_frequency"]
                        if fraction > 0.05:
                            print(
                                f"@> [{key}] WARNING: variant {variant} clipped {fraction:.1%} of its "
                                "frequency-carrying mutants to exactly 0; rank information is degraded"
                            )

                    variant_rows.append(
                        {
                            "variant": variant,
                            "lineage": lineage_label,
                            "lineage_key": key,
                            "parent_lineage": edge_parent,
                            "is_primary_parent": bool(is_primary),
                            "frequency_path": str(edge_freq_path),
                            "equation": int(equation),
                            "coefficient": float(coefficient),
                            "frequency_cutoff": float(cutoff),
                            "frequency_cutoff_k": int(k),
                            "parent_median_depth": median_depth,
                            "temperature": float(temperature),
                            "n_flat_columns": n_flat,
                            # Recomputed from THIS variant's matrix, not the ESCOTT one:
                            # the frequency penalty clips cells to exactly 0 in rank
                            # space, and a column whose cells all clip collapses to a
                            # single remapped value, i.e. a dead site the ESCOTT count
                            # cannot know about.
                            "n_uniform_columns": count_uniform_columns(remapped),
                            "score_matrix_path": str(out_csv),
                            "md5": file_md5(out_csv),
                            **clipping,
                        }
                    )

    # The parity cross-check below is a check on OUR arithmetic against the published
    # tool, not on the parent choice, so it runs once against the primary edge only.
    log10_frequency = primary_log10_frequency

    # --- optional cross-check against the published tool -------------------------------
    if prescott_ref_dir is not None:
        reference_k = int(frequency_cutoff_ks[0])
        reference_equation = int(equations[0])
        reference_coefficient = float(coefficients[0])
        cutoff, _ = resolve_frequency_cutoff(
            key, reference_k, Path(inputs_dir), frequency_cutoff_fallback, mode=frequency_cutoff_mode
        )
        # prescott needs the ungapped query; escott already wrote exactly that as <prot>.fasta,
        # but cleanTheMess leaves it in place only sometimes, so prefer stage A's copy.
        details_path = run_prescott_reference(
            key,
            escott_txt,
            Path(inputs["query_path"]),
            Path(frequency_path),
            Path(prescott_ref_dir) / key,
            coefficient=reference_coefficient,
            frequency_cutoff=cutoff,
            equation=reference_equation,
            prescott_bin=prescott_bin,
            force=force,
            env_bin=env_bin,
        )
        parity = prescott_parity_check(
            matrix,
            log10_frequency,
            read_prescott_details(details_path),
            coefficient=reference_coefficient,
            frequency_cutoff=cutoff,
            equation=reference_equation,
        )
        parity_dir = ensure_dir(
            Path(diagnostics_dir) if diagnostics_dir is not None
            else Path(scores_dir).parent / "tables" / "diagnostics"
        )
        parity_path = parity_dir / "prescott_parity_check.tsv"
        parity.insert(0, "lineage", lineage_label)
        # Merge rather than append: reruns are expected (the caches make them cheap) and a
        # blind append would duplicate every mutant on each pass.
        if parity_path.exists():
            previous = pd.read_csv(parity_path, sep="\t")
            merged = pd.concat([previous, parity], ignore_index=True)
            merged = merged.drop_duplicates(subset=["lineage", "mutant"], keep="last")
        else:
            merged = parity
        merged.to_csv(parity_path, sep="\t", index=False)
        print(
            f"@> [{key}] prescott parity: max|dESCOTT|={parity.attrs['max_abs_delta_escott']:.4f} "
            f"max|dPRESCOTT|={parity.attrs['max_abs_delta_prescott']:.4f} "
            f"passed={parity.attrs['passed']}"
        )

    return variant_rows


def write_variants_table(rows: Iterable[Dict[str, object]], scores_dir: Path) -> Path:
    """Append/replace ``scores/score_variants.tsv``, the contract stage D reads."""
    frame = pd.DataFrame(list(rows))
    out_path = ensure_dir(Path(scores_dir)) / "score_variants.tsv"
    if out_path.exists():
        existing = pd.read_csv(out_path, sep="\t")
        combined = pd.concat([existing, frame], ignore_index=True)
        combined = combined.drop_duplicates(subset=["lineage_key", "variant"], keep="last")
    else:
        combined = frame
    combined.to_csv(out_path, sep="\t", index=False)
    return out_path


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ESCOTT/PRESCOTT for the influenza HA lineages and emit PLM-format score matrices.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--inputs-dir", type=Path, required=True, help="stage A/B output root")
    parser.add_argument("--escott-workdir", type=Path, required=True, help="root for per-lineage escott CWDs")
    parser.add_argument("--scores-dir", type=Path, required=True, help="where the score matrices are written")
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        default=None,
        help="where the prescott parity table goes; defaults to <scores-dir>/../tables/diagnostics, "
             "which is only the run's own diagnostics dir while --scores-dir is left at its default. "
             "Pass <output-dir>/tables/diagnostics explicitly whenever --scores-dir is shared",
    )
    parser.add_argument(
        "--prescott-ref-dir",
        type=Path,
        default=None,
        help="if set, also run the published prescott tool here and record a parity check",
    )
    parser.add_argument(
        "--lineage",
        action="append",
        default=None,
        help="repeatable; default is every lineage with stage A/B outputs on disk",
    )
    parser.add_argument("--escott-bin", type=Path, default=DEFAULT_ESCOTT_BIN)
    parser.add_argument("--prescott-bin", type=Path, default=DEFAULT_PRESCOTT_BIN)
    parser.add_argument("--prescott-env-bin", type=Path, default=DEFAULT_PRESCOTT_ENV_BIN,
                        help="prepended to PATH so escott can find Rscript/mkdssp")
    parser.add_argument(
        "--escott-pdbfile",
        type=Path,
        default=None,
        help=(
            "optional structure. WARNING: escott then forces normweightmode="
            "'sstjetormaxtwocomponent' and recomputes the residue weight itself, discarding the "
            "structural terms stage B folded into 'trace'. Requires 100%% residue coverage."
        ),
    )
    parser.add_argument("--alphabet", default="lw-i.7")
    parser.add_argument("--max-coil-length", type=int, default=5)
    parser.add_argument("--cv-radius", type=float, default=None, help="passed to escott --cvrc; only used with --escott-pdbfile")
    parser.add_argument("--escott-temperature", type=float, default=1.0, help="softmax temperature T in P = softmax(E/T)")
    parser.add_argument("--coefficient-grid", default="0.25,0.5,1.0")
    parser.add_argument("--equation-grid", default="2", help="PRESCOTT equations; 1, 2, 3 and 5 only (4 exits upstream)")
    parser.add_argument("--frequency-cutoff-k", default="1", help="Fc = log10(k / median parent depth)")
    parser.add_argument("--frequency-cutoff-mode", choices=("depth_scaled", "fixed"), default="depth_scaled")
    parser.add_argument("--frequency-cutoff", type=float, default=-4.0, help="used only when --frequency-cutoff-mode fixed")
    parser.add_argument("--parent-map", default=None, help='"child=parent,..." overrides on top of inputs_manifest.json')
    parser.add_argument(
        "--sensitivity-parent-map",
        default=None,
        help='"child=alt_parent,..." ALTERNATE parent edges to score IN ADDITION to the primary '
             "map, e.g. 'K=J.2_int'. Each needs a frequency file from prepare_inputs.py "
             "--sensitivity-parent-map and yields extra _parent<TOK> variants. Omit to take "
             "whatever alternate edges inputs_manifest.json says stage A prepared",
    )
    parser.add_argument(
        "--no-parent-sensitivity",
        action="store_true",
        help="ignore the alternate parent edges stage A prepared and score the primary map only",
    )
    parser.add_argument("--force", action="store_true", help="ignore all caches and recompute")
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="one lineage, ESCOTT variant only, no prescott reference run",
    )
    return parser


def discover_lineages(inputs_dir: Path) -> List[str]:
    """Lineages that have the full stage A *and* stage B product set.

    A query FASTA alone is not enough. prepare_inputs.py writes a query for every
    lineage in the guide, including the input-only basal lineage (G.1), which exists
    solely to supply a parent-frequency prior and never gets a jet table. Selecting on
    the query glob therefore always picked up G.1 and made both the bare
    ``run_escott.py`` (no --lineage) and ``--test-mode`` (which takes lineages[:1], and
    G.1 sorts first) die with FileNotFoundError on G.1_surrogate_jet.res.
    """
    inputs_dir = Path(inputs_dir)
    query_dir = inputs_dir / "query"
    if not query_dir.is_dir():
        raise FileNotFoundError(f"{query_dir} does not exist; run prepare_inputs.py first")
    jet_dir = inputs_dir / "jet"
    candidates = sorted(path.name[: -len("_query.fasta")] for path in query_dir.glob("*_query.fasta"))
    scorable = [key for key in candidates if (jet_dir / f"{key}_surrogate_jet.res").exists()]
    if not scorable:
        raise FileNotFoundError(
            f"no lineage under {inputs_dir} has a jet table in {jet_dir}; "
            "run jet_surrogate.py first, or name lineages explicitly with --lineage"
        )
    skipped = [key for key in candidates if key not in scorable]
    if skipped:
        print(f"@> skipping lineages with no jet table (input-only): {skipped}")
    return scorable


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    lineages = args.lineage or discover_lineages(args.inputs_dir)
    parent_map = resolve_parent_map(args.inputs_dir, args.parent_map)
    sensitivity_map = resolve_sensitivity_map(
        args.inputs_dir, args.sensitivity_parent_map, disable=args.no_parent_sensitivity
    )
    if sensitivity_map:
        print(f"@> alternate (sensitivity) parent edges: {sensitivity_map}")

    coefficients = parse_grid(args.coefficient_grid, float)
    equations = parse_grid(args.equation_grid, int)
    cutoff_ks = parse_grid(args.frequency_cutoff_k, int)
    prescott_ref_dir = args.prescott_ref_dir

    if args.test_mode:
        lineages = lineages[:1]
        coefficients = coefficients[:1]
        equations = equations[:1]
        cutoff_ks = cutoff_ks[:1]
        prescott_ref_dir = None
        print(f"@> --test-mode: restricting to lineage {lineages} and a single PRESCOTT grid point")

    if args.test_mode:
        # A sensitivity pass doubles the PRESCOTT work for no extra coverage of the
        # code paths --test-mode exists to smoke-test.
        sensitivity_map = {}

    all_rows: List[Dict[str, object]] = []
    for lineage in lineages:
        alternates = [sensitivity_map[lineage]] if lineage in sensitivity_map else []
        print(f"\n@> ===== lineage {lineage} (parent {parent_map.get(lineage, 'unknown')}"
              f"{', alternates ' + str(alternates) if alternates else ''}) =====")
        all_rows.extend(
            process_lineage(
                lineage,
                args.inputs_dir,
                args.escott_workdir,
                args.scores_dir,
                parent_lineage=parent_map.get(lineage),
                alternate_parents=alternates,
                diagnostics_dir=args.diagnostics_dir,
                escott_bin=args.escott_bin,
                prescott_bin=args.prescott_bin,
                pdbfile=args.escott_pdbfile,
                alphabet=args.alphabet,
                max_coil_length=args.max_coil_length,
                cv_radius=args.cv_radius,
                temperature=args.escott_temperature,
                coefficients=coefficients,
                equations=equations,
                frequency_cutoff_ks=cutoff_ks,
                frequency_cutoff_mode=args.frequency_cutoff_mode,
                frequency_cutoff_fallback=args.frequency_cutoff,
                prescott_ref_dir=prescott_ref_dir,
                force=args.force,
                env_bin=args.prescott_env_bin,
            )
        )

    table_path = write_variants_table(all_rows, args.scores_dir)
    print(f"\n@> wrote {len(all_rows)} variant rows -> {table_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
