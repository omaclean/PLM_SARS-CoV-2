#!/usr/bin/env python3
"""BLAST-based data-leakage detection and PURGE for the ESCOTT/PRESCOTT HA pipeline.

WHY THIS MODULE EXISTS
======================
The ESCOTT evolutionary set is

    /home3/oml4h/my_SC2_finetunes/myflu/full_job4243186/data/
    ncbiflu_HA_all_110424_noX_clu99_filt_HA-80.clean_input.fasta

-- 6433 NCBI influenza A HA proteins with an 11 April 2024 cutoff -- while the
evaluation targets are 2025/26 GISAID panels under
``Sequences/gisaid_data/alignment_based_19feb26/hard``.  If a 2025 GISAID sequence
(or something within a handful of substitutions of one) is *also* sitting in the
deep set, then GEMME/ESCOTT has already seen the answer and every reported
correlation between predicted and observed diversity is inflated by an unknown
amount.

Identifier matching CANNOT detect this.  The two sets use different accession
namespaces: the panels are keyed by nucleotide accessions (``PX902199``,
``ON903285``, ``OQ233153``) and the deep set by protein accessions (``QBM69670``,
``AIY59238``).  The same physical protein appears under two unrelated IDs.

Neither, on its own, does hashing.  Measured on the real files: the deep set
carries the 16-residue signal peptide (``MKTIIALSYILCLVVAQK...``, 547-576 aa
unaligned) and the GISAID panels do not (``QKIPGNDNST...``, 546-551 aa unaligned
inside 594 aligned columns).  A full-length hash of a genuine duplicate therefore
differs, and indeed ``sequence_hash`` finds **zero** collisions between the deep
set and any panel even where BLAST finds near-identical pairs.  Hashing is kept
(check C) because it is nearly free and because an exact collision is
unambiguous, but it is a *floor*, not the instrument.

Sequence-level alignment is the only reliable instrument, which is what this
module is.

THE FOUR CHECKS
===============
A. ``check_msa_vs_target``   -- deep evolutionary set vs each evaluation panel.
   This is the leak that would inflate every reported correlation.
B. ``check_parent_vs_target`` -- the PRESCOTT frequency input (the PARENT panel)
   vs the evaluation target panel.  Both are GISAID panels in the SAME namespace,
   so an accession-overlap check is meaningful here and is done *as well as* the
   alignment check.  Note that the parent panels legitimately contain pre-2024
   sequences (G.1 is 2022-23, J_int 2023-24), so some parent-vs-deep-set overlap
   is expected and is not a defect; parent-vs-*target* overlap is.
C. ``check_hash_duplicates`` -- exact-duplicate detection by sequence hash across
   every panel and the deep set.  Cheap, namespace-blind, and reported with an
   explicit note that a null result means nothing on its own (see above).
D. ``purge_msa_against_target`` -- **the primary deliverable**.  Not a report: a
   filter.  Drops every deep-set row too close to ANY sequence in one target
   panel, before ESCOTT ever sees the alignment.

THE PURGE RULE
==============
A deep-set row is dropped when it has a BLAST hit against the target panel that

    (coverage >= --leakage-min-coverage)   AND
    (identity >= --leakage-min-identity  OR  hamming <= --leakage-max-hamming)

The two thresholds are combined with **OR** -- the conservative reading: a
sequence is removed if EITHER rule fires.  They are NOT equivalent and the
difference matters.  On a ~550 aa mature HA, 10 mismatches is 100*(550-10)/550 =
98.2% identity, so at the defaults the **Hamming rule is the stricter of the
two**: everything the 99% rule catches (<= 5.5 mismatches) the Hamming rule
already caught, and the Hamming rule additionally catches the 6-10 mismatch band
that 99% identity lets through.  A user who sets only ``--leakage-min-identity``
will still be governed by the default Hamming rule unless they also disable it
with ``--leakage-max-hamming none``.  This is stated in ``--help`` and repeated
in every report, because assuming otherwise silently changes what is removed.

Why 99% is the right identity default: the deep set is *already* clustered at 99%
identity -- that is what ``clu99`` in its filename means.  Purging at >= 99%
therefore removes exactly the sequences that its own dedup step would have
collapsed into a target sequence had the target been present when it was built.
It is coherent with the set's own granularity rather than an arbitrary cut.

METRIC DEFINITIONS (trap 2: the two sets are not in a common frame)
==================================================================
Deep-set rows are 530-566 residues after ungapping the 566-column query frame;
panel rows are ~550 residues after ungapping 594 aligned columns.  Hamming
distance is undefined between strings of different length, so both metrics are
derived from the BLAST alignment (``nident``, ``length``, ``qlen``, ``slen``):

    identity = 100 * nident / length          # `length` = aligned columns,
                                              # so gap columns count against it
    coverage = 100 * length / min(qlen, slen) # per BASIS; see below
    hamming  = min(qlen, slen) - nident

``hamming`` is "residues of the SHORTER sequence that are not identically matched
to the other one" -- mismatches, gap columns and unaligned overhang, each counted
once.  It reduces to the ordinary Hamming distance when the two are the same
length and gapless, and is well defined when they are not.  The *shorter* length
is the right denominator because the deep rows carry a 16-residue signal peptide
the panels lack: ``max(qlen,slen) - nident`` would charge a genuine duplicate 16
phantom mismatches and put it straight past a threshold of 10.

``coverage`` defaults to basis ``both``, i.e. ``min(length/qlen, length/slen)``:
BOTH sequences must be almost entirely inside the alignment.  Neither one-sided
choice survives contact with the data, and both failure modes were reproduced:

* ``query`` basis alone lets a real duplicate escape.  550 aligned columns against
  a 566-residue query is 97.2% -- and against the longest raw deep records (576 aa)
  only 95.5%, i.e. within rounding distance of the 95% floor.
* ``shorter`` basis alone lets a fragment in.  A 200-residue HA1 fragment that is
  a byte-exact substring of a panel sequence scores 100% coverage on the shorter
  (itself) and is purged as a "full-length duplicate".  Verified: with basis
  ``shorter`` a spiked 200-aa fragment was dropped; with basis ``both`` it is not.

``both`` passes the duplicate (97.2% and 100%) and blocks the fragment (100% and
36.4%).  ``shorter``/``query``/``subject`` remain selectable for sensitivity work.

The coverage floor is a *gate*, not a rule: it exists so that a short, high
identity local hit (an HA1 fragment, a conserved stalk segment) is never mistaken
for a full-length duplicate.

TRAPS HANDLED HERE
==================
1. **The query is never purged.**  Row 0 of every ``msa_<key>.fasta`` is that
   lineage's own reference protein, which is by construction ~identical to the
   target lineage and trips both thresholds every time.  ``protect_indices``
   defaults to ``(0,)``, the exemption is asserted after the purge, and the
   metrics the query *would* have been dropped on are recorded in the manifest as
   ``query_exempted``.  A purged query means ESCOTT scores a different protein
   than the one being evaluated, or dies outright.
2. **Frame mismatch** -- handled by the metric definitions above.
3. **The purged MSA is target-specific.**  This is a real change to the pipeline's
   shape: with the purge on, ESCOTT can no longer be run once and shared across
   lineages, because J_int's purged alignment and K's purged alignment are
   different alignments.  The pipeline already ran escott once per evaluation
   target (the query is the GEMME epistatic reference, so it had to), so nothing
   breaks -- but the *reason* is now doubled, and ``msa_<key>.fasta`` must never be
   reused across lineages.
4. **MSA depth drives GEMME/ESCOTT quality.**  Depth before and after is reported
   per target; the purge warns loudly above ``--leakage-max-removed-fraction`` and
   raises ``MsaDepthError`` below ``--leakage-min-depth-after``.  Removing too much
   is its own failure mode.

NOT purged: the parent panel.  The parent is the declared frequency input, not a
leak.  Purging the deep set against the *target* will incidentally remove close
parent relatives, and that is correct and expected -- a sequence one substitution
from a 2025 K genome is a leak no matter which panel it also resembles.

Run under /home3/oml4h/miniconda3/envs/PRESCOTT/bin/python.  No torch, no esm.

Examples
--------
    # full report, no purge, gate a slurm job on the exit code
    .../python scripts/prescott_iav/leakage_check.py \
        --report-dir Results/leakage --no-purge-leakage --fail-on-leakage

    # purge one prepared lineage MSA in place
    .../python scripts/prescott_iav/leakage_check.py \
        --purge-only --lineage K \
        --msa Results/run/inputs/msa/msa_K.fasta \
        --report-dir Results/run/inputs/leakage
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from prescott_iav import common  # noqa: E402

try:  # constants.py is the shared authority; a bare-module import must still work
    from prescott_iav import constants as _constants  # noqa: E402
except ImportError:  # pragma: no cover
    import constants as _constants  # type: ignore  # noqa: E402


__all__ = [
    "LeakageError",
    "MsaDepthError",
    "BlastNotFoundError",
    "Hit",
    "DEFAULT_DEEP_FASTA",
    "DEFAULT_MIN_IDENTITY",
    "DEFAULT_MAX_HAMMING",
    "DEFAULT_MIN_COVERAGE",
    "ungap",
    "sequence_hash",
    "accession_of",
    "load_ungapped",
    "dedupe_by_hash",
    "parse_threshold",
    "hit_metrics",
    "blast_records",
    "identity_distribution",
    "check_msa_vs_target",
    "check_parent_vs_target",
    "check_hash_duplicates",
    "purge_msa_against_target",
    "purge_lineage_msa",
    "run_leakage_stage",
    "add_leakage_arguments",
    "write_report",
    "build_parser",
    "main",
]


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #

DEFAULT_DEEP_FASTA = Path(
    "/home3/oml4h/my_SC2_finetunes/myflu/full_job4243186/data/"
    "ncbiflu_HA_all_110424_noX_clu99_filt_HA-80.clean_input.fasta"
)
DEFAULT_GUIDE = REPO_ROOT / "Sequences" / "IAV_lineage_guide.csv"

# See the module docstring: 99% is the deep set's own clu99 dedup granularity.
DEFAULT_MIN_IDENTITY = 99.0
# 10 AA mismatches ~= 98.2% on a 550 aa mature HA, i.e. STRICTER than 99% identity.
DEFAULT_MAX_HAMMING = 10
# A gate, not a rule: below this the hit is a fragment, not a duplicate.
DEFAULT_MIN_COVERAGE = 95.0

# Flagging thresholds for the *report* (check A) are separate from the purge
# thresholds so that a run can report at one stringency and purge at another.
DEFAULT_FLAG_IDENTITY = 99.0
DEFAULT_FLAG_COVERAGE = 95.0

# Purge sanity limits (trap 4).
DEFAULT_MAX_REMOVED_FRACTION = 0.25   # warn loudly above this
DEFAULT_MIN_DEPTH_AFTER = 500         # hard floor: GEMME on less than this is noise

# blastp-fast + no composition-based statistics + no SEG is ~3x faster than plain
# blastp on this workload (measured: 300 deep queries vs the 2745-sequence unique K
# panel, 21.2 s -> 6.8 s at 12 threads) and cannot miss a >= 95%-identity hit --
# near-identical proteins are exactly the regime where a longer seed word is safe.
DEFAULT_BLAST_TASK = "blastp-fast"
DEFAULT_EVALUE = 1e-20
DEFAULT_MAX_TARGET_SEQS = 5

SENTINELS_OFF = {"none", "off", "disable", "disabled", "inf", "nan", ""}

BLAST_OUTFMT_FIELDS = (
    "qseqid", "sseqid", "pident", "length", "mismatch", "gaps",
    "qstart", "qend", "sstart", "send", "qlen", "slen", "nident",
    "bitscore", "evalue",
)

REPORT_COLUMNS = (
    "check", "target_lineage", "query_set", "subject_set",
    "n_query", "n_subject", "metric", "value", "note",
)


class LeakageError(RuntimeError):
    """A leakage gate failed."""


class MsaDepthError(RuntimeError):
    """The purge left the deep MSA too shallow to be worth running GEMME on."""


class BlastNotFoundError(RuntimeError):
    """blastp/makeblastdb are not on PATH."""


# --------------------------------------------------------------------------- #
# Sequence primitives
# --------------------------------------------------------------------------- #

_GAP_CHARS = "-.*~ \t"
_GAP_TABLE = {ord(ch): None for ch in _GAP_CHARS}


def ungap(seq: str) -> str:
    """Aligned protein -> bare residues, uppercased.

    Panels are aligned (594 columns for the 19feb26 sets) and deep MSA rows are
    aligned (566 columns); BLAST wants neither.  ``*`` and ``.`` are stripped as
    well as ``-`` because ``common.IGNORE_ALIGNMENT_CHARS`` treats all three as
    non-residues everywhere else in this pipeline.
    """
    return str(seq).upper().translate(_GAP_TABLE)


def sequence_hash(seq: str) -> str:
    """md5 of the ungapped uppercase sequence -- the check-C identity key."""
    return hashlib.md5(ungap(seq).encode("ascii", "ignore")).hexdigest()


def suffix_hash(seq: str, length: int) -> Optional[str]:
    """md5 of the C-terminal ``length`` residues, or None if the sequence is shorter.

    The one frame difference between the two sets is entirely N-terminal (the deep
    set keeps the 16-residue signal peptide, the panels do not), so a C-terminal
    suffix hash is the cheapest way to catch a duplicate that differs only by that
    trim.  Measured on the real data it finds nothing at 300/400/450/500 -- which is
    itself worth reporting, because it means every overlap BLAST finds is a *near*
    neighbour rather than a byte-identical record.
    """
    residues = ungap(seq)
    if length <= 0 or len(residues) < length:
        return None
    return hashlib.md5(residues[-length:].encode("ascii", "ignore")).hexdigest()


def accession_of(header: str) -> str:
    """First whitespace/pipe-delimited token of a FASTA header.

    Panel headers are bare accessions (``>OQ233153``); lineage reference headers
    are pipe-delimited (``>EPI4748783|HA|A/England/...``).  Both reduce to their
    first field, which is what an accession-overlap check needs.

    A header consisting only of ``>`` and whitespace has no first field; it yields
    ``""`` rather than raising.  That is reachable from a FASTA whose defline is
    ``>>`` -- ``common.read_fasta`` hands on the header ``">"`` -- and a traceback
    there would kill the whole leakage stage over one malformed record.
    """
    token = (str(header).strip().lstrip(">").split() or [""])[0]
    return token.split("|")[0]


def load_ungapped(path: Path) -> List[Tuple[str, str]]:
    """``[(header, ungapped_sequence), ...]`` with empty records dropped."""
    out: List[Tuple[str, str]] = []
    for header, seq in common.read_fasta(Path(path)):
        residues = ungap(seq)
        if residues:
            out.append((header, residues))
    return out


def dedupe_by_hash(
    records: Sequence[Tuple[str, str]],
) -> Tuple[List[Tuple[str, str]], Dict[str, List[str]]]:
    """Collapse identical sequences, keeping the first header as representative.

    This is the single biggest cost saving in the module: the J.2_int panel is
    27452 records but only 5264 distinct sequences, and K is 17898 -> 2745.  BLAST
    subject databases shrink ~5-6x, and because the panels are what we build the
    database from, so does the search.

    Returns ``(unique_records, members)`` where ``members[hash]`` lists every
    header that carried that sequence, so a hit can still be reported with its
    true multiplicity instead of pretending the panel had one copy.
    """
    unique: List[Tuple[str, str]] = []
    members: Dict[str, List[str]] = {}
    for header, seq in records:
        digest = hashlib.md5(seq.encode("ascii", "ignore")).hexdigest()
        if digest in members:
            members[digest].append(header)
            continue
        members[digest] = [header]
        unique.append((header, seq))
    return unique, members


def parse_threshold(value: object, name: str, cast=float) -> Optional[float]:
    """Parse a threshold that may be explicitly disabled.

    ``--leakage-min-identity none`` (also ``off``/``disabled``/``inf``) returns
    ``None``, meaning "this rule never fires".  A silent 0/-1 sentinel was
    rejected on purpose: ``--leakage-min-identity 0`` reads as "purge everything
    at >= 0% identity", which is exactly what it would do if 0 were overloaded to
    mean "off".
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in SENTINELS_OFF:
        return None
    try:
        parsed = cast(text)
    except (TypeError, ValueError):
        raise ValueError(
            f"{name}: expected a number or one of {sorted(SENTINELS_OFF - {''})}, got {value!r}"
        )
    if isinstance(parsed, float) and (math.isnan(parsed) or math.isinf(parsed)):
        return None
    return parsed


# --------------------------------------------------------------------------- #
# BLAST
# --------------------------------------------------------------------------- #

@dataclass
class Hit:
    """One BLAST HSP, re-expressed in the metrics the purge rule is written in.

    ``identity``/``coverage``/``hamming`` are the module docstring's definitions.
    ``pident`` is kept separately because BLAST's own ``pident`` is
    ``100*nident/length`` too but has historically differed by rounding, and a
    report that quotes a number BLAST did not print is hard to audit.
    """

    qid: str
    sid: str
    qheader: str
    sheader: str
    aln_len: int
    nident: int
    mismatch: int
    gaps: int
    qlen: int
    slen: int
    qstart: int
    qend: int
    sstart: int
    send: int
    bitscore: float
    evalue: float
    pident: float

    @property
    def identity(self) -> float:
        return 100.0 * self.nident / self.aln_len if self.aln_len else 0.0

    @property
    def hamming(self) -> int:
        return int(min(self.qlen, self.slen) - self.nident)

    def coverage(self, basis: str = "both") -> float:
        """Alignment coverage under one of four bases; see the module docstring.

        ``both`` (the default) is ``min(query_coverage, subject_coverage)`` -- the
        only basis that simultaneously passes a signal-peptide-offset duplicate and
        rejects an exact-substring fragment.
        """
        if basis == "query":
            denom = self.qlen
        elif basis == "subject":
            denom = self.slen
        elif basis == "shorter":
            denom = min(self.qlen, self.slen)
        else:  # "both"
            if not self.qlen or not self.slen:
                return 0.0
            return 100.0 * self.aln_len / max(self.qlen, self.slen)
        return 100.0 * self.aln_len / denom if denom else 0.0


def hit_metrics(hit: Hit, coverage_basis: str = "both") -> Dict[str, object]:
    return {
        "identity": round(hit.identity, 4),
        "coverage": round(hit.coverage(coverage_basis), 4),
        "hamming": hit.hamming,
        "aln_len": hit.aln_len,
        "nident": hit.nident,
        "qlen": hit.qlen,
        "slen": hit.slen,
        "bitscore": hit.bitscore,
        "evalue": hit.evalue,
    }


def _require_blast(makeblastdb: str, blastp: str) -> None:
    missing = [tool for tool in (makeblastdb, blastp) if shutil.which(tool) is None]
    if missing:
        raise BlastNotFoundError(
            f"BLAST executables not found on PATH: {missing}. Leakage detection needs "
            "makeblastdb and blastp from the PRESCOTT conda env; prepend "
            "/home3/oml4h/miniconda3/envs/PRESCOTT/bin to PATH, or pass "
            "--makeblastdb-bin/--blastp-bin, or turn the check off with --no-leakage-check "
            "(which makes the run unaudited for leakage -- say so if you do)."
        )


def _write_indexed_fasta(path: Path, records: Sequence[Tuple[str, str]], prefix: str) -> Dict[str, str]:
    """Write FASTA under synthetic IDs and return ``{synthetic_id: original_header}``.

    Synthetic IDs are not cosmetic.  makeblastdb rejects duplicate IDs, and BLAST
    still parses ``|``-containing headers as legacy NCBI deflines and rewrites
    them, so a header like ``EPI4748783|HA|A/England/...`` comes back mangled in
    ``qseqid``.  Numbering the records sidesteps both.
    """
    mapping: Dict[str, str] = {}
    out: List[Tuple[str, str]] = []
    for index, (header, seq) in enumerate(records):
        sid = f"{prefix}{index:07d}"
        mapping[sid] = header
        out.append((sid, seq))
    common.write_fasta(Path(path), out)
    return mapping


def blast_records(
    query_records: Sequence[Tuple[str, str]],
    subject_records: Sequence[Tuple[str, str]],
    workdir: Path,
    tag: str,
    *,
    blast_task: str = DEFAULT_BLAST_TASK,
    evalue: float = DEFAULT_EVALUE,
    max_target_seqs: int = DEFAULT_MAX_TARGET_SEQS,
    threads: int = 8,
    makeblastdb_bin: str = "makeblastdb",
    blastp_bin: str = "blastp",
    keep_files: bool = False,
) -> List[Hit]:
    """blastp ``query_records`` against a database built from ``subject_records``.

    ``-max_hsps 1`` is essential, not an optimisation: with multiple HSPs per pair
    the ``nident``/``length`` of any single row is a fragment of the true
    alignment, and both ``identity`` and ``hamming`` would be computed from it.
    One HSP per pair means one honest (identity, coverage, hamming) triple.
    """
    _require_blast(makeblastdb_bin, blastp_bin)
    workdir = common.ensure_dir(Path(workdir))
    if not query_records or not subject_records:
        return []

    query_fasta = workdir / f"{tag}_query.fasta"
    subject_fasta = workdir / f"{tag}_subject.fasta"
    db_prefix = workdir / f"{tag}_db"
    out_tsv = workdir / f"{tag}_blast.tsv"

    qmap = _write_indexed_fasta(query_fasta, query_records, "q")
    smap = _write_indexed_fasta(subject_fasta, subject_records, "s")

    db_cmd = [makeblastdb_bin, "-in", str(subject_fasta), "-dbtype", "prot",
              "-out", str(db_prefix), "-title", tag]
    proc = subprocess.run(db_cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"makeblastdb failed ({proc.returncode}):\n{proc.stderr[-4000:]}")

    blast_cmd = [
        blastp_bin,
        "-task", blast_task,
        "-query", str(query_fasta),
        "-db", str(db_prefix),
        "-outfmt", "6 " + " ".join(BLAST_OUTFMT_FIELDS),
        "-evalue", repr(float(evalue)),
        "-max_target_seqs", str(int(max_target_seqs)),
        "-max_hsps", "1",
        "-comp_based_stats", "0",
        "-seg", "no",
        "-num_threads", str(max(1, int(threads))),
        "-out", str(out_tsv),
    ]
    started = time.time()
    proc = subprocess.run(blast_cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"blastp failed ({proc.returncode}):\n{proc.stderr[-4000:]}")
    elapsed = time.time() - started

    hits: List[Hit] = []
    with open(out_tsv) as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != len(BLAST_OUTFMT_FIELDS):
                continue
            row = dict(zip(BLAST_OUTFMT_FIELDS, parts))
            hits.append(Hit(
                qid=row["qseqid"],
                sid=row["sseqid"],
                qheader=qmap.get(row["qseqid"], row["qseqid"]),
                sheader=smap.get(row["sseqid"], row["sseqid"]),
                aln_len=int(row["length"]),
                nident=int(row["nident"]),
                mismatch=int(row["mismatch"]),
                gaps=int(row["gaps"]),
                qlen=int(row["qlen"]),
                slen=int(row["slen"]),
                qstart=int(row["qstart"]),
                qend=int(row["qend"]),
                sstart=int(row["sstart"]),
                send=int(row["send"]),
                bitscore=float(row["bitscore"]),
                evalue=float(row["evalue"]),
                pident=float(row["pident"]),
            ))
    print(f"@> blastp[{tag}]: {len(query_records)} x {len(subject_records)} -> "
          f"{len(hits)} HSPs in {elapsed:.1f}s ({blast_task}, {threads} threads)")

    if not keep_files:
        for path in (query_fasta, subject_fasta, out_tsv):
            try:
                path.unlink()
            except OSError:  # pragma: no cover
                pass
        for suffix in (".phr", ".pin", ".psq", ".pdb", ".pot", ".ptf", ".pto", ".pjs", ".pos"):
            try:
                Path(str(db_prefix) + suffix).unlink()
            except OSError:
                pass
    return hits


def best_hit_per_query(hits: Iterable[Hit]) -> Dict[str, Hit]:
    """Best HSP per query, ranked by ``nident`` then bitscore.

    Ranked by identical-residue count rather than bitscore because the question is
    "how close is the nearest target sequence", and a longer, slightly less
    identical alignment can outscore the actual near-duplicate.
    """
    best: Dict[str, Hit] = {}
    for hit in hits:
        current = best.get(hit.qid)
        if current is None or (hit.nident, hit.bitscore) > (current.nident, current.bitscore):
            best[hit.qid] = hit
    return best


# --------------------------------------------------------------------------- #
# Distribution reporting
# --------------------------------------------------------------------------- #

IDENTITY_BINS = (0.0, 70.0, 80.0, 90.0, 95.0, 97.0, 98.0, 99.0, 99.5, 99.9, 100.0)


def identity_distribution(values: Sequence[float]) -> Dict[str, object]:
    """Percentiles plus a fixed-bin histogram of an identity vector.

    The bin edges are deliberately dense above 95%: everything below that is
    "different subtype / different clade" and carries no information about
    leakage, while the 99-100 band is the entire question.
    """
    if not values:
        return {"n": 0, "min": None, "max": None, "percentiles": {}, "histogram": {}}
    ordered = sorted(float(v) for v in values)

    def pct(p: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        idx = (len(ordered) - 1) * (p / 100.0)
        lo, hi = int(math.floor(idx)), int(math.ceil(idx))
        if lo == hi:
            return ordered[lo]
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (idx - lo)

    histogram: Dict[str, int] = {}
    for lo, hi in zip(IDENTITY_BINS[:-1], IDENTITY_BINS[1:]):
        label = f"[{lo:g},{hi:g})"
        histogram[label] = sum(1 for v in ordered if lo <= v < hi)
    histogram["[100,100]"] = sum(1 for v in ordered if v >= 100.0)

    return {
        "n": len(ordered),
        "min": round(ordered[0], 4),
        "max": round(ordered[-1], 4),
        "mean": round(sum(ordered) / len(ordered), 4),
        "percentiles": {f"p{p:g}": round(pct(p), 4)
                        for p in (5, 25, 50, 75, 90, 95, 99, 99.9)},
        "histogram": histogram,
    }


# --------------------------------------------------------------------------- #
# Check A -- deep evolutionary set vs evaluation target
# --------------------------------------------------------------------------- #

@dataclass
class BlastCheck:
    """Result of one set-vs-set alignment comparison."""

    name: str
    target_lineage: str
    query_label: str
    subject_label: str
    n_query: int
    n_subject: int
    n_subject_unique: int
    n_query_with_hit: int
    n_exact_full_length: int
    n_flagged: int
    flagged: List[Dict[str, object]] = field(default_factory=list)
    distribution: Dict[str, object] = field(default_factory=dict)
    thresholds: Dict[str, object] = field(default_factory=dict)


def _flag_rows(
    hits: Sequence[Hit],
    members: Dict[str, List[str]],
    subject_seq_by_header: Dict[str, str],
    flag_identity: Optional[float],
    flag_coverage: float,
    coverage_basis: str,
    max_rows: int,
) -> Tuple[List[Dict[str, object]], int, int]:
    """Rows above the *reporting* thresholds, plus exact and flagged counts."""
    rows: List[Dict[str, object]] = []
    n_exact = 0
    n_flagged = 0
    for hit in sorted(hits, key=lambda h: (-h.identity, h.hamming)):
        coverage = hit.coverage(coverage_basis)
        if coverage < flag_coverage:
            continue
        exact = (hit.nident == hit.aln_len == hit.qlen == hit.slen)
        if exact:
            n_exact += 1
        if flag_identity is not None and hit.identity + 1e-9 < flag_identity:
            continue
        n_flagged += 1
        if len(rows) >= max_rows:
            continue
        digest = hashlib.md5(
            subject_seq_by_header.get(hit.sheader, "").encode("ascii", "ignore")
        ).hexdigest()
        rows.append({
            "query": hit.qheader,
            "query_accession": accession_of(hit.qheader),
            "subject": hit.sheader,
            "subject_accession": accession_of(hit.sheader),
            "subject_panel_copies": len(members.get(digest, [hit.sheader])),
            "exact_full_length": exact,
            **hit_metrics(hit, coverage_basis),
        })
    return rows, n_exact, n_flagged


def check_msa_vs_target(
    deep_records: Sequence[Tuple[str, str]],
    target_panel: Path,
    target_lineage: str,
    workdir: Path,
    *,
    deep_label: str = "deep_msa",
    flag_identity: Optional[float] = DEFAULT_FLAG_IDENTITY,
    flag_coverage: float = DEFAULT_FLAG_COVERAGE,
    coverage_basis: str = "both",
    max_report_rows: int = 500,
    **blast_kwargs,
) -> BlastCheck:
    """Check A: how much of the deep evolutionary set is (near-)present in one panel.

    This is the leak that inflates every reported correlation, so it is measured
    over the WHOLE deep set rather than a sample, and the identity distribution of
    the best hit per deep sequence is reported whether or not anything is flagged
    -- a distribution that tops out at 96% is a materially different claim from one
    that tops out at 99.8%, and only the first is reassuring.
    """
    panel_records = load_ungapped(Path(target_panel))
    unique, members = dedupe_by_hash(panel_records)
    subject_seq_by_header = {h: s for h, s in unique}

    hits = blast_records(deep_records, unique, workdir,
                         f"{deep_label}_vs_{common.safe_label(target_lineage)}",
                         **blast_kwargs)
    best = best_hit_per_query(hits)
    rows, n_exact, n_flagged = _flag_rows(
        list(best.values()), members, subject_seq_by_header,
        flag_identity, flag_coverage, coverage_basis, max_report_rows,
    )
    distribution = identity_distribution([h.identity for h in best.values()])

    return BlastCheck(
        name="A_msa_vs_target",
        target_lineage=target_lineage,
        query_label=deep_label,
        subject_label=f"panel:{target_lineage}",
        n_query=len(deep_records),
        n_subject=len(panel_records),
        n_subject_unique=len(unique),
        n_query_with_hit=len(best),
        n_exact_full_length=n_exact,
        n_flagged=n_flagged,
        flagged=rows,
        distribution=distribution,
        thresholds={
            "flag_identity": flag_identity,
            "flag_coverage": flag_coverage,
            "coverage_basis": coverage_basis,
        },
    )


# --------------------------------------------------------------------------- #
# Check B -- parent (frequency input) vs target (evaluation set)
# --------------------------------------------------------------------------- #

def check_parent_vs_target(
    parent_panel: Path,
    target_panel: Path,
    target_lineage: str,
    parent_lineage: str,
    workdir: Path,
    *,
    flag_identity: Optional[float] = DEFAULT_FLAG_IDENTITY,
    flag_coverage: float = DEFAULT_FLAG_COVERAGE,
    coverage_basis: str = "both",
    max_report_rows: int = 500,
    **blast_kwargs,
) -> Tuple[BlastCheck, Dict[str, object]]:
    """Check B: the PRESCOTT frequency input must not contain the target's own sequences.

    Two instruments, both run, because they fail differently:

    * **Accession overlap.**  Parent and target panels are both GISAID panels cut
      from the same alignment, so they share ONE namespace and an accession match
      is exact, free and unambiguous.  This is the only pair in the project where
      ID matching is valid, which is why it is done here and nowhere else.
    * **Alignment.**  Accessions catch the same *record* in both panels; they do
      not catch the same *virus* deposited twice, or a target sequence that is one
      substitution from a parent sequence.  Only alignment does.

    A parent panel that overlaps its target is a direct circularity: the frequency
    prior would be built from the very sequences whose diversity is the evaluation
    target, and PRESCOTT would be scored on its ability to recall its own input.
    """
    parent_records = load_ungapped(Path(parent_panel))
    target_records = load_ungapped(Path(target_panel))

    parent_ids = {accession_of(h) for h, _ in parent_records}
    target_ids = {accession_of(h) for h, _ in target_records}
    shared_ids = sorted(parent_ids & target_ids)

    parent_hashes = {sequence_hash(s) for _, s in parent_records}
    target_hashes = {sequence_hash(s) for _, s in target_records}
    shared_hashes = parent_hashes & target_hashes

    unique_target, members = dedupe_by_hash(target_records)
    unique_parent, _ = dedupe_by_hash(parent_records)
    subject_seq_by_header = {h: s for h, s in unique_target}

    hits = blast_records(unique_parent, unique_target, workdir,
                         f"parent_{common.safe_label(parent_lineage)}_vs_"
                         f"{common.safe_label(target_lineage)}",
                         **blast_kwargs)
    best = best_hit_per_query(hits)
    rows, n_exact, n_flagged = _flag_rows(
        list(best.values()), members, subject_seq_by_header,
        flag_identity, flag_coverage, coverage_basis, max_report_rows,
    )

    check = BlastCheck(
        name="B_parent_vs_target",
        target_lineage=target_lineage,
        query_label=f"panel:{parent_lineage}",
        subject_label=f"panel:{target_lineage}",
        n_query=len(parent_records),
        n_subject=len(target_records),
        n_subject_unique=len(unique_target),
        n_query_with_hit=len(best),
        n_exact_full_length=n_exact,
        n_flagged=n_flagged,
        flagged=rows,
        distribution=identity_distribution([h.identity for h in best.values()]),
        thresholds={
            "flag_identity": flag_identity,
            "flag_coverage": flag_coverage,
            "coverage_basis": coverage_basis,
        },
    )
    accession_report = {
        "parent_lineage": parent_lineage,
        "target_lineage": target_lineage,
        "n_parent_accessions": len(parent_ids),
        "n_target_accessions": len(target_ids),
        "n_shared_accessions": len(shared_ids),
        "shared_accession_fraction_of_target": (
            round(len(shared_ids) / len(target_ids), 6) if target_ids else 0.0
        ),
        "shared_accessions_sample": shared_ids[:100],
        "n_shared_exact_sequences": len(shared_hashes),
        "shared_exact_sequence_fraction_of_target": (
            round(len(shared_hashes) / max(1, len(target_hashes)), 6)
        ),
    }
    return check, accession_report


# --------------------------------------------------------------------------- #
# Check C -- exact duplicates by hash
# --------------------------------------------------------------------------- #

def check_hash_duplicates(
    sets: Dict[str, Path],
    *,
    suffix_length: int = 500,
) -> Dict[str, object]:
    """Check C: exact-duplicate detection by sequence hash across every set.

    Cheap (one pass, no BLAST) and namespace-blind, so a positive result is
    unambiguous evidence of leakage no matter what the identifiers say.

    A NEGATIVE result, however, proves nothing, and the report says so: the deep
    set carries the 16-residue signal peptide and the GISAID panels do not, so a
    genuine duplicate hashes differently.  The C-terminal ``suffix_length`` hash is
    included as a partial remedy -- it is invariant to exactly that N-terminal trim
    -- but only alignment (checks A/B) is decisive.
    """
    loaded: Dict[str, List[Tuple[str, str]]] = {}
    per_set: Dict[str, Dict[str, object]] = {}
    full: Dict[str, Set[str]] = {}
    suffix: Dict[str, Set[str]] = {}

    for label, path in sets.items():
        records = load_ungapped(Path(path))
        loaded[label] = records
        digests = {sequence_hash(s) for _, s in records}
        suffixes = {d for d in (suffix_hash(s, suffix_length) for _, s in records) if d}
        full[label] = digests
        suffix[label] = suffixes
        lengths = sorted(len(s) for _, s in records)
        per_set[label] = {
            "path": str(path),
            "n_records": len(records),
            "n_unique_sequences": len(digests),
            "duplication_factor": round(len(records) / len(digests), 3) if digests else 0.0,
            "min_length": lengths[0] if lengths else 0,
            "max_length": lengths[-1] if lengths else 0,
        }

    pairs: List[Dict[str, object]] = []
    labels = list(sets)
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            shared_full = full[a] & full[b]
            shared_suffix = suffix[a] & suffix[b]
            pairs.append({
                "set_a": a,
                "set_b": b,
                "n_shared_exact_sequences": len(shared_full),
                "n_shared_cterm_suffix": len(shared_suffix),
                "shared_fraction_of_a": (
                    round(len(shared_full) / len(full[a]), 6) if full[a] else 0.0
                ),
                "shared_fraction_of_b": (
                    round(len(shared_full) / len(full[b]), 6) if full[b] else 0.0
                ),
            })
    return {
        "suffix_length": suffix_length,
        "sets": per_set,
        "pairs": pairs,
        "n_pairs_with_exact_overlap": sum(1 for p in pairs if p["n_shared_exact_sequences"]),
        "caveat": (
            "A zero here is NOT evidence of no leakage. The deep set retains the HA "
            "signal peptide and the GISAID panels start at the mature N-terminus, so "
            "identical viruses hash differently. Trust checks A/B."
        ),
    }


# --------------------------------------------------------------------------- #
# D -- THE PURGE (the primary deliverable)
# --------------------------------------------------------------------------- #

def purge_msa_against_target(
    msa_path: Path,
    target_panel: Path,
    out_fasta: Path,
    out_manifest_tsv: Path,
    *,
    target_lineage: str = "",
    min_identity: Optional[float] = DEFAULT_MIN_IDENTITY,
    max_hamming: Optional[int] = DEFAULT_MAX_HAMMING,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    coverage_basis: str = "both",
    protect_indices: Sequence[int] = (0,),
    max_removed_fraction: float = DEFAULT_MAX_REMOVED_FRACTION,
    min_depth_after: int = DEFAULT_MIN_DEPTH_AFTER,
    workdir: Optional[Path] = None,
    keep_alignment_columns: bool = True,
    **blast_kwargs,
) -> Dict[str, object]:
    """Drop every deep-set row too close to ANY sequence in ``target_panel``.

    THE rule (module docstring has the full justification)::

        drop  <=>  coverage >= min_coverage
                   AND (identity >= min_identity  OR  hamming <= max_hamming)

    OR, not AND: a row is removed if EITHER threshold fires.  At the defaults the
    Hamming rule is strictly the stronger one on a ~550 aa HA (10 mismatches =
    98.2% < 99%), so disabling only ``min_identity`` changes almost nothing while
    disabling only ``max_hamming`` loosens the purge substantially.

    ``protect_indices`` (default ``(0,)``, the lineage's own query) is exempted
    BEFORE the rule is applied and asserted present afterwards.  The query is
    ~identical to the target lineage by construction and trips both thresholds
    every single time; purging it would hand ESCOTT a different protein than the
    one being evaluated, or no query at all.

    Returns a report dict; also writes ``out_fasta`` (alignment columns preserved,
    row order preserved) and ``out_manifest_tsv`` (one row per dropped sequence
    naming the target sequence it matched, at what identity, coverage and Hamming
    distance, and which rule fired) so every removal is auditable.

    Raises ``MsaDepthError`` when the survivors fall below ``min_depth_after``;
    warns loudly on stderr above ``max_removed_fraction``.
    """
    msa_path = Path(msa_path)
    rows = list(common.read_fasta(msa_path))
    if not rows:
        raise ValueError(f"MSA {msa_path} is empty")
    if min_identity is None and max_hamming is None:
        raise ValueError(
            "Both purge rules are disabled (--leakage-min-identity none AND "
            "--leakage-max-hamming none), so purge_msa_against_target would drop "
            "nothing. Use --no-purge-leakage to say that on purpose."
        )

    protect = {int(i) % len(rows) for i in protect_indices}
    ungapped = [(header, ungap(seq)) for header, seq in rows]

    panel_records = load_ungapped(Path(target_panel))
    if not panel_records:
        raise ValueError(f"Target panel {target_panel} contains no records")
    unique_panel, members = dedupe_by_hash(panel_records)
    panel_seq_by_header = {h: s for h, s in unique_panel}

    owns_workdir = workdir is None
    workdir = Path(workdir) if workdir is not None else Path(tempfile.mkdtemp(prefix="leakpurge_"))
    try:
        hits = blast_records(
            [(f"{index}", seq) for index, (_h, seq) in enumerate(ungapped)],
            unique_panel,
            workdir,
            f"purge_{common.safe_label(target_lineage or msa_path.stem)}",
            **blast_kwargs,
        )
    finally:
        if owns_workdir:
            shutil.rmtree(workdir, ignore_errors=True)

    # blast_records maps synthetic q-ids back to the header it was given, which
    # here is the stringified row index.
    by_index: Dict[int, List[Hit]] = {}
    for hit in hits:
        by_index.setdefault(int(hit.qheader), []).append(hit)

    dropped: List[Dict[str, object]] = []
    dropped_indices: Set[int] = set()
    query_exempted: Optional[Dict[str, object]] = None
    # Hits that WOULD have been dropped but were stopped by the coverage gate alone.
    # Reported because the gate has less slack than it looks: with basis 'both' a
    # genuine duplicate of a 550-residue panel sequence sitting in a 566-residue deep
    # row scores 97.2%, and in the longest raw deep records (576 aa) only 95.5% -- so a
    # non-zero count here means the gate, not the thresholds, decided the purge, and
    # the operator should look at it before believing a clean result.
    near_misses: List[Dict[str, object]] = []

    for index, (header, _seq) in enumerate(ungapped):
        candidates = by_index.get(index, [])
        worst: Optional[Tuple[Hit, List[str]]] = None
        gated: Optional[Hit] = None
        for hit in candidates:
            if hit.coverage(coverage_basis) + 1e-9 < min_coverage:
                would_fire = (
                    (min_identity is not None and hit.identity + 1e-9 >= min_identity)
                    or (max_hamming is not None and hit.hamming <= max_hamming)
                )
                # One entry per MSA ROW, not per HSP, so the count reads as "rows the
                # gate saved" rather than "hits the gate saw".
                if would_fire and (gated is None or hit.nident > gated.nident):
                    gated = hit
                continue
            reasons: List[str] = []
            if min_identity is not None and hit.identity + 1e-9 >= min_identity:
                reasons.append("identity")
            if max_hamming is not None and hit.hamming <= max_hamming:
                reasons.append("hamming")
            if not reasons:
                continue
            # Keep the closest offender, so the audit row names the strongest
            # evidence rather than whichever hit BLAST happened to print first.
            if worst is None or hit.nident > worst[0].nident:
                worst = (hit, reasons)
        if worst is None:
            if gated is not None:
                near_misses.append({
                    "msa_index": index,
                    "msa_header": header,
                    "matched_target_sequence": gated.sheader,
                    **hit_metrics(gated, coverage_basis),
                })
            continue
        hit, reasons = worst
        digest = hashlib.md5(
            panel_seq_by_header.get(hit.sheader, "").encode("ascii", "ignore")
        ).hexdigest()
        record = {
            "msa_index": index,
            "msa_header": header,
            "msa_accession": accession_of(header),
            "matched_target_sequence": hit.sheader,
            "matched_target_accession": accession_of(hit.sheader),
            "matched_target_panel_copies": len(members.get(digest, [hit.sheader])),
            "rule": "+".join(reasons),
            "target_lineage": target_lineage,
            **hit_metrics(hit, coverage_basis),
        }
        if index in protect:
            # TRAP 1. Not an error -- it is the EXPECTED outcome for row 0 -- but it
            # must be visible, because "the query looked like a leak" is exactly the
            # signal that the query and the target lineage are the same protein.
            record["exempted"] = True
            record["exemption_reason"] = "protected index (lineage query / ESCOTT reference)"
            if query_exempted is None:
                query_exempted = record
            continue
        dropped_indices.add(index)
        dropped.append(record)

    kept = [row for index, row in enumerate(rows) if index not in dropped_indices]
    if not keep_alignment_columns:
        kept = [(h, ungap(s)) for h, s in kept]
        # Row 0 is degapped too, so the survival assertion below must compare against
        # the same transform or it would fire on every gapped query.
        rows = [(h, ungap(s)) for h, s in rows]

    # TRAP 1, asserted rather than assumed. `kept` preserves input order, so a
    # protected index that survived is still at the same position among the
    # survivors as it was among the inputs when nothing before it was dropped --
    # which is exactly the case for index 0.
    if protect and (dropped_indices & protect):  # pragma: no cover - defensive
        raise AssertionError(
            f"Protected MSA rows {sorted(dropped_indices & protect)} were purged. "
            "Row 0 is the lineage query and the ESCOTT/GEMME epistatic reference."
        )
    if protect and (kept[0][0] != rows[0][0] or kept[0][1] != rows[0][1]):
        raise AssertionError(
            "The purge removed or altered row 0 of the MSA. Row 0 is the lineage query "
            "and the ESCOTT/GEMME epistatic reference; it must survive unchanged. "
            f"MSA {msa_path}"
        )
    if len(kept) != len(rows) - len(dropped_indices):
        raise AssertionError("Purge bookkeeping error: kept + dropped != total")

    common.write_fasta(Path(out_fasta), kept)

    manifest_columns = [
        "msa_index", "msa_header", "msa_accession", "target_lineage",
        "matched_target_sequence", "matched_target_accession",
        "matched_target_panel_copies", "rule", "identity", "coverage", "hamming",
        "aln_len", "nident", "qlen", "slen", "bitscore", "evalue",
    ]
    out_manifest_tsv = Path(out_manifest_tsv)
    common.ensure_dir(out_manifest_tsv.parent)
    with out_manifest_tsv.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(manifest_columns) + "\n")
        for record in sorted(dropped, key=lambda r: (-float(r["identity"]), int(r["hamming"]))):
            handle.write("\t".join(str(record.get(col, "")) for col in manifest_columns) + "\n")

    depth_before = len(rows)
    depth_after = len(kept)
    removed_fraction = (depth_before - depth_after) / depth_before if depth_before else 0.0
    removed_identities = [float(r["identity"]) for r in dropped]
    removed_hammings = [int(r["hamming"]) for r in dropped]

    report: Dict[str, object] = {
        "target_lineage": target_lineage,
        "msa_path": str(msa_path),
        "msa_md5_before": common.md5_file(msa_path),
        "target_panel": str(target_panel),
        "target_panel_records": len(panel_records),
        "target_panel_unique": len(unique_panel),
        "purged_path": str(out_fasta),
        "purged_md5": common.md5_file(Path(out_fasta)),
        "drop_manifest_path": str(out_manifest_tsv),
        "depth_before": depth_before,
        "depth_after": depth_after,
        "n_removed": depth_before - depth_after,
        "removed_fraction": round(removed_fraction, 6),
        "n_removed_by_identity_only": sum(1 for r in dropped if r["rule"] == "identity"),
        "n_removed_by_hamming_only": sum(1 for r in dropped if r["rule"] == "hamming"),
        "n_removed_by_both": sum(1 for r in dropped if r["rule"] == "identity+hamming"),
        "n_removed_exact_full_length": sum(
            1 for r in dropped
            if float(r["identity"]) >= 100.0 and int(r["hamming"]) == 0
        ),
        "removed_identity_distribution": identity_distribution(removed_identities),
        # Check A's headline number, computed here for free from the same BLAST the
        # purge already ran: the best-hit identity of EVERY deep row against the target
        # panel, before anything was removed. Without it a clean purge ("0 removed")
        # is uninterpretable -- a deep set topping out at 94% against the target is a
        # materially different claim from one topping out at 98.9%, and only the first
        # is reassuring. Row 0 (the query) is excluded: it is ~100% by construction and
        # would otherwise put a spurious 100 at the top of every distribution.
        "msa_best_hit_identity_distribution": identity_distribution([
            best.identity for index, best in
            ((idx, max(hits_for, key=lambda h: (h.nident, h.bitscore)))
             for idx, hits_for in by_index.items() if hits_for)
            if index not in protect
        ]),
        "n_msa_rows_with_any_hit": sum(1 for idx in by_index if idx not in protect),
        "removed_hamming_min": min(removed_hammings) if removed_hammings else None,
        "removed_hamming_max": max(removed_hammings) if removed_hammings else None,
        "n_coverage_gated_near_misses": len(near_misses),
        "coverage_gated_near_miss_max_coverage": (
            max(float(m["coverage"]) for m in near_misses) if near_misses else None
        ),
        "coverage_gated_near_misses": near_misses[:50],
        "query_header": rows[0][0],
        "query_protected": True,
        "query_exempted": query_exempted,
        "thresholds": {
            "min_identity": min_identity,
            "max_hamming": max_hamming,
            "min_coverage": min_coverage,
            "coverage_basis": coverage_basis,
            "combination": "OR (drop if EITHER fires)",
            "note": (
                "On a ~550 aa mature HA, max_hamming=10 is ~98.2% identity, i.e. "
                "STRICTER than min_identity=99.0. The Hamming rule governs at the "
                "defaults; disabling only min_identity changes little."
            ),
        },
        "warnings": [],
    }

    if near_misses:
        message = (
            f"[LEAKAGE PURGE] {target_lineage or msa_path.stem}: {len(near_misses)} hit(s) met "
            f"a purge threshold but were stopped by the coverage gate alone (best coverage "
            f"{max(float(m['coverage']) for m in near_misses):.2f}% vs floor {min_coverage}%, "
            f"basis '{coverage_basis}'). The gate, not the thresholds, decided these. Inspect "
            "leakage_report.json:coverage_gated_near_misses before treating the result as clean."
        )
        print(message, file=sys.stderr)
        report["warnings"].append(message)

    if removed_fraction > max_removed_fraction:
        message = (
            f"[LEAKAGE PURGE] {target_lineage or msa_path.stem}: removed "
            f"{depth_before - depth_after}/{depth_before} rows "
            f"({removed_fraction:.1%}), above --leakage-max-removed-fraction "
            f"{max_removed_fraction:.1%}. GEMME/ESCOTT quality scales with MSA depth; "
            "a purge this large may degrade the predictions it is protecting."
        )
        print(message, file=sys.stderr)
        report["warnings"].append(message)
    if depth_after < min_depth_after:
        raise MsaDepthError(
            f"{target_lineage or msa_path.stem}: purge left {depth_after} rows, below "
            f"--leakage-min-depth-after {min_depth_after}. GEMME on an alignment this "
            "shallow is not informative. Either loosen the thresholds, lower the floor "
            "deliberately, or accept that this target cannot be evaluated leak-free."
        )
    return report


def purge_lineage_msa(
    msa_path: Path,
    target_panel: Path,
    target_lineage: str,
    leakage_dir: Path,
    *,
    force: bool = False,
    min_identity: Optional[float] = DEFAULT_MIN_IDENTITY,
    max_hamming: Optional[int] = DEFAULT_MAX_HAMMING,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    coverage_basis: str = "both",
    protect_indices: Sequence[int] = (0,),
    **purge_kwargs,
) -> Dict[str, object]:
    """Purge one prepared ``msa_<key>.fasta`` in place, with a cache.

    Layout, chosen so that nothing downstream has to learn a new filename:

    * ``msa_<key>.fasta``          -- what ESCOTT reads.  After this call it is the
      PURGED alignment.  ``run_escott.py`` and ``jet_surrogate.py`` are untouched.
    * ``msa_<key>_prepurge.fasta`` -- the pre-purge alignment, kept verbatim so the
      purge can be re-derived, diffed or undone.
    * ``<leakage_dir>/<key>_purge_dropped.tsv``  -- the audit manifest.
    * ``<leakage_dir>/<key>_purge.json``         -- the cache key + report.

    TRAP 3: because the purge is per-target, ``msa_<key>.fasta`` is now
    target-specific in a second, stronger sense than before.  It was already
    per-lineage (the query row differs), but two lineages' files used to differ in
    exactly one row; now they differ in hundreds.  Never share one across targets.
    """
    msa_path = Path(msa_path)
    leakage_dir = common.ensure_dir(Path(leakage_dir))
    key = common.safe_label(target_lineage) or msa_path.stem
    prepurge = msa_path.with_name(f"{msa_path.stem}_prepurge{msa_path.suffix}")
    drop_tsv = leakage_dir / f"{key}_purge_dropped.tsv"
    cache_json = leakage_dir / f"{key}_purge.json"

    source = prepurge if prepurge.exists() else msa_path
    # The thresholds are named parameters rather than **kwargs precisely so that they
    # reach the cache key even when the caller relied on the defaults. Harvesting them
    # out of **kwargs recorded only what was passed EXPLICITLY, so a run that took the
    # defaults cached under an empty threshold set -- and would then have been served
    # that cache after leakage_check.py's defaults changed, silently handing ESCOTT an
    # alignment purged at the old stringency.
    thresholds = {
        "min_identity": min_identity,
        "max_hamming": max_hamming,
        "min_coverage": float(min_coverage),
        "coverage_basis": coverage_basis,
        "protect_indices": sorted(int(i) for i in protect_indices),
    }
    cache_key = {
        "source_md5": common.md5_file(source),
        "panel_md5": common.md5_file(Path(target_panel)),
        "thresholds": thresholds,
        # Bump whenever the report SCHEMA changes, not just the algorithm. A cached
        # report is replayed verbatim into inputs_manifest.json, so a v1 cache served
        # to v2 code produces a manifest missing fields the driver and CAVEATS read.
        # v2 added msa_best_hit_identity_distribution / n_msa_rows_with_any_hit /
        # n_coverage_gated_near_misses.
        "version": 2,
    }
    cached = common.read_json(cache_json)
    if (not force) and cached and cached.get("cache_key") == cache_key and msa_path.exists():
        report = dict(cached.get("report") or {})
        if report.get("purged_md5") == common.md5_file(msa_path):
            print(f"@> Leakage purge cache hit: {msa_path.name} "
                  f"({report.get('n_removed')} of {report.get('depth_before')} removed)")
            report["cached"] = True
            return report

    if not prepurge.exists():
        shutil.copy2(msa_path, prepurge)

    tmp_out = msa_path.with_suffix(".purged.tmp")
    report = purge_msa_against_target(
        prepurge, Path(target_panel), tmp_out, drop_tsv,
        target_lineage=target_lineage,
        workdir=leakage_dir / "blast_work",
        min_identity=min_identity,
        max_hamming=max_hamming,
        min_coverage=min_coverage,
        coverage_basis=coverage_basis,
        protect_indices=protect_indices,
        **purge_kwargs,
    )
    shutil.move(str(tmp_out), str(msa_path))
    report["purged_path"] = str(msa_path)
    report["purged_md5"] = common.md5_file(msa_path)
    report["prepurge_path"] = str(prepurge)
    report["prepurge_md5"] = common.md5_file(prepurge)
    report["cached"] = False
    common.write_json(cache_json, {"cache_key": cache_key, "report": report})
    return report


# --------------------------------------------------------------------------- #
# The stage-1 entry point: what prepare_inputs.py calls
# --------------------------------------------------------------------------- #

def run_leakage_stage(
    inputs_dir: Path,
    panel_by_label: Dict[str, Path],
    parent_map: Dict[str, str],
    targets: Sequence[str],
    msa_paths: Dict[str, Path],
    *,
    leakage_check: bool = True,
    purge: bool = True,
    fail_on_leakage: bool = False,
    min_identity: Optional[float] = DEFAULT_MIN_IDENTITY,
    max_hamming: Optional[int] = DEFAULT_MAX_HAMMING,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    coverage_basis: str = "both",
    max_removed_fraction: float = DEFAULT_MAX_REMOVED_FRACTION,
    min_depth_after: int = DEFAULT_MIN_DEPTH_AFTER,
    hash_suffix_length: int = 500,
    deep_fasta: Optional[Path] = None,
    force: bool = False,
    report_dir: Optional[Path] = None,
    **blast_kwargs,
) -> Dict[str, object]:
    """Everything ``prepare_inputs.py`` needs, in one call, in the right order.

    Order matters and is not arbitrary:

    1. **Check C** (hash) first, because it costs one file pass and a positive
       result is unambiguous -- if it fires there is no point spending ten minutes
       on BLAST to confirm it.
    2. **Check B** (parent vs target) next.  It is small (unique-vs-unique, both
       panels collapse ~5x) and it is the check the purge CANNOT fix: the parent
       panel is the declared frequency input, so an overlap there is a design
       error to be reported, not filtered away.
    3. **The purge** last, per target, because it is the expensive step and because
       its result -- the post-purge depth -- is what the rest of stage 1 depends on.

    Check A is not run separately when ``purge`` is on: the purge performs exactly
    the same deep-set-vs-panel BLAST and its report carries the same identity
    distribution and exact-match counts, so running both would double the cost to
    print the same numbers twice.  With ``purge=False`` check A runs on its own.

    Returns the block that goes straight into ``inputs_manifest.json`` under
    ``"leakage"``.  Raises :class:`LeakageError` when ``fail_on_leakage`` and
    anything failed, so a slurm job dies instead of producing an inflated number.
    """
    inputs_dir = Path(inputs_dir)
    report_dir = common.ensure_dir(Path(report_dir) if report_dir else inputs_dir / "leakage")
    workdir = common.ensure_dir(report_dir / "blast_work")

    result: Dict[str, object] = {
        "enabled": bool(leakage_check or purge),
        "leakage_check": bool(leakage_check),
        "purge": bool(purge),
        "report_dir": str(report_dir),
        "thresholds": {
            "min_identity": min_identity,
            "max_hamming": max_hamming,
            "min_coverage": min_coverage,
            "coverage_basis": coverage_basis,
            "combination": "OR (drop if EITHER fires)",
            "max_removed_fraction": max_removed_fraction,
            "min_depth_after": min_depth_after,
            "note": (
                "On a ~550 aa mature HA, max_hamming=10 is ~98.2% identity, i.e. STRICTER "
                "than min_identity=99.0; the Hamming rule governs at the defaults."
            ),
        },
        "blast": {k: (str(v) if isinstance(v, Path) else v) for k, v in blast_kwargs.items()},
        "targets": list(targets),
        "checks": {},
        "purges": {},
        "failures": [],
    }
    if not (leakage_check or purge):
        result["status"] = "SKIPPED"
        return result

    failures: List[str] = result["failures"]  # type: ignore[assignment]

    if leakage_check:
        sets: Dict[str, Path] = {}
        # The deep set belongs in check C even though it is the expensive pair: an
        # exact hash collision between it and any panel is unambiguous leakage and
        # costs one file pass to rule out, which is much cheaper than the BLAST that
        # would otherwise be the only evidence.
        if deep_fasta is not None and Path(deep_fasta).exists():
            sets["deep_msa"] = Path(deep_fasta)
        for label, path in sorted(panel_by_label.items()):
            sets[f"panel:{label}"] = path
        hash_report = check_hash_duplicates(sets, suffix_length=hash_suffix_length)
        result["checks"]["C_hash_duplicates"] = hash_report
        for pair in hash_report["pairs"]:
            if pair["set_a"] == "deep_msa" and pair["n_shared_exact_sequences"]:
                failures.append(
                    f"check C: {pair['n_shared_exact_sequences']} byte-identical sequences "
                    f"are in BOTH the deep evolutionary set and {pair['set_b']}"
                )

        b_reports: Dict[str, object] = {}
        for target in targets:
            parent = parent_map.get(target)
            if not parent or parent not in panel_by_label or target not in panel_by_label:
                continue
            check_b, accessions = check_parent_vs_target(
                panel_by_label[parent], panel_by_label[target], target, parent, workdir,
                coverage_basis=coverage_basis, **blast_kwargs,
            )
            _write_hits_tsv(
                report_dir / f"leakage_B_{common.safe_label(parent)}_vs_"
                             f"{common.safe_label(target)}_hits.tsv",
                check_b.flagged,
            )
            b_reports[target] = {
                "parent": parent,
                "n_parent_records": check_b.n_query,
                "n_target_records": check_b.n_subject,
                "n_exact_full_length": check_b.n_exact_full_length,
                "n_flagged": check_b.n_flagged,
                "distribution": check_b.distribution,
                "accessions": accessions,
            }
            print(f"@> Leakage B {parent} -> {target}: "
                  f"{accessions['n_shared_accessions']} shared accessions, "
                  f"{accessions['n_shared_exact_sequences']} shared exact sequences, "
                  f"{check_b.n_flagged} alignment-flagged")
            if accessions["n_shared_accessions"] or accessions["n_shared_exact_sequences"]:
                failures.append(
                    f"check B: the frequency-input panel {parent} shares "
                    f"{accessions['n_shared_accessions']} accessions / "
                    f"{accessions['n_shared_exact_sequences']} exact sequences with its "
                    f"evaluation target {target}. The purge cannot fix this -- the parent "
                    "panel is a declared input, not a filterable set."
                )
        result["checks"]["B_parent_vs_target"] = b_reports

    for target in targets:
        msa_path = msa_paths.get(target)
        panel = panel_by_label.get(target)
        if msa_path is None or panel is None:
            continue
        if purge:
            report = purge_lineage_msa(
                Path(msa_path), Path(panel), target, report_dir,
                force=force,
                min_identity=min_identity, max_hamming=max_hamming,
                min_coverage=min_coverage, coverage_basis=coverage_basis,
                max_removed_fraction=max_removed_fraction,
                min_depth_after=min_depth_after,
                **blast_kwargs,
            )
            result["purges"][target] = report
            dist = report.get("msa_best_hit_identity_distribution") or {}
            print(f"@> Leakage purge {target}: depth {report['depth_before']} -> "
                  f"{report['depth_after']} ({report['n_removed']} removed, "
                  f"{report['removed_fraction']:.2%}); "
                  f"pre-purge best-hit identity max {dist.get('max')}% "
                  f"over {report.get('n_msa_rows_with_any_hit')} rows with any hit; "
                  f"query {'WOULD have been purged (exempted)' if report.get('query_exempted') else 'not near any target'}")
        else:
            deep_records = load_ungapped(Path(msa_path))
            check_a = check_msa_vs_target(
                deep_records, Path(panel), target, workdir,
                coverage_basis=coverage_basis, **blast_kwargs,
            )
            _write_hits_tsv(report_dir / f"leakage_A_{common.safe_label(target)}_hits.tsv",
                            check_a.flagged)
            result["checks"].setdefault("A_msa_vs_target", {})[target] = {
                "n_query": check_a.n_query,
                "n_subject": check_a.n_subject,
                "n_exact_full_length": check_a.n_exact_full_length,
                "n_flagged": check_a.n_flagged,
                "distribution": check_a.distribution,
            }
            print(f"@> Leakage A {target}: {check_a.n_flagged} deep sequences flagged, "
                  f"{check_a.n_exact_full_length} exact full-length matches "
                  f"(PURGE IS OFF -- these remain in the alignment ESCOTT will see)")
            # The query row is index 0 and is ~identical to the target by
            # construction, so it is always one of the flagged rows. Subtracting it
            # is what makes the count mean "leak" rather than "leak plus the query".
            residual = max(0, check_a.n_flagged - 1)
            if residual:
                failures.append(
                    f"check A: {residual} deep-set sequences (excluding the query row) are "
                    f"near-identical to a {target} panel sequence and --no-purge-leakage "
                    "left them in the alignment ESCOTT scores"
                )

    shutil.rmtree(workdir, ignore_errors=True)

    result["status"] = "FAIL" if failures else "PASS"
    summary_lines = [
        "LEAKAGE STAGE (prepare_inputs.py)",
        f"status: {result['status']}",
        f"rule  : coverage >= {min_coverage}% ({coverage_basis}) AND "
        f"(identity >= {min_identity} OR hamming <= {max_hamming})",
    ]
    for target, report in sorted(result["purges"].items()):  # type: ignore[union-attr]
        summary_lines.append(
            f"  purge {target:10s} {report['depth_before']} -> {report['depth_after']} "
            f"({report['n_removed']} removed, {report['removed_fraction']:.2%})"
        )
    for failure in failures:
        summary_lines.append(f"  ! {failure}")
    (report_dir / "leakage_summary.txt").write_text("\n".join(summary_lines) + "\n",
                                                    encoding="utf-8")
    common.write_json(report_dir / "leakage_report.json", result)

    if failures and fail_on_leakage:
        raise LeakageError(
            "Leakage gates failed and --fail-on-leakage is set:\n  - "
            + "\n  - ".join(failures)
            + f"\nFull report: {report_dir / 'leakage_report.json'}"
        )
    return result


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def _report_rows_from_check(check: BlastCheck) -> List[Dict[str, object]]:
    base = {
        "check": check.name,
        "target_lineage": check.target_lineage,
        "query_set": check.query_label,
        "subject_set": check.subject_label,
        "n_query": check.n_query,
        "n_subject": check.n_subject,
    }
    rows = [
        {**base, "metric": "n_subject_unique", "value": check.n_subject_unique, "note": ""},
        {**base, "metric": "n_query_with_hit", "value": check.n_query_with_hit, "note": ""},
        {**base, "metric": "n_exact_full_length", "value": check.n_exact_full_length,
         "note": "100% identity over 100% of both sequences"},
        {**base, "metric": "n_flagged", "value": check.n_flagged,
         "note": f"identity >= {check.thresholds.get('flag_identity')} and coverage >= "
                 f"{check.thresholds.get('flag_coverage')}"},
    ]
    dist = check.distribution or {}
    for stat in ("max", "mean"):
        if dist.get(stat) is not None:
            rows.append({**base, "metric": f"best_hit_identity_{stat}",
                         "value": dist[stat], "note": "over best hit per query"})
    for label, count in (dist.get("percentiles") or {}).items():
        rows.append({**base, "metric": f"best_hit_identity_{label}", "value": count,
                     "note": "over best hit per query"})
    for label, count in (dist.get("histogram") or {}).items():
        rows.append({**base, "metric": f"best_hit_identity_hist_{label}", "value": count,
                     "note": "queries whose best hit falls in this identity bin"})
    return rows


def write_report(
    report_dir: Path,
    rows: Sequence[Dict[str, object]],
    payload: Dict[str, object],
    summary_lines: Sequence[str],
) -> Dict[str, Path]:
    """Write the machine-readable TSV, the JSON payload and the human summary."""
    report_dir = common.ensure_dir(Path(report_dir))
    tsv_path = report_dir / "leakage_report.tsv"
    with tsv_path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(REPORT_COLUMNS) + "\n")
        for row in rows:
            handle.write("\t".join(str(row.get(col, "")) for col in REPORT_COLUMNS) + "\n")
    json_path = report_dir / "leakage_report.json"
    common.write_json(json_path, payload)
    txt_path = report_dir / "leakage_summary.txt"
    txt_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return {"tsv": tsv_path, "json": json_path, "txt": txt_path}


def _write_hits_tsv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        common.ensure_dir(Path(path).parent)
        Path(path).write_text(
            "query\tquery_accession\tsubject\tsubject_accession\tsubject_panel_copies\t"
            "exact_full_length\tidentity\tcoverage\thamming\taln_len\tnident\tqlen\tslen\t"
            "bitscore\tevalue\n",
            encoding="utf-8",
        )
        return
    columns = ["query", "query_accession", "subject", "subject_accession",
               "subject_panel_copies", "exact_full_length", "identity", "coverage",
               "hamming", "aln_len", "nident", "qlen", "slen", "bitscore", "evalue"]
    common.ensure_dir(Path(path).parent)
    with Path(path).open("w", encoding="utf-8") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join(str(row.get(col, "")) for col in columns) + "\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def add_leakage_arguments(parser: argparse.ArgumentParser, *, prefix: str = "") -> None:
    """Attach the shared leakage flags to any parser.

    Used verbatim by ``prepare_inputs.py`` and mirrored by
    ``run_prescott_diversity.py``, so the three CLIs cannot drift in the meaning of
    a threshold -- which matters more than usual here, because a threshold that
    means one thing to the driver and another to stage 1 silently changes which
    sequences ESCOTT sees.
    """
    group = parser.add_argument_group("leakage detection / purge")
    group.add_argument(f"--{prefix}leakage-check", dest="leakage_check",
                       action=argparse.BooleanOptionalAction, default=True,
                       help="Run BLAST-based leakage detection (checks A/B/C). Default on.")
    group.add_argument(f"--{prefix}purge-leakage", dest="purge_leakage",
                       action=argparse.BooleanOptionalAction, default=True,
                       help="REMOVE near neighbours of each evaluation target from the deep "
                            "MSA before ESCOTT sees it. This is the point of the module: a "
                            "report tells you the result is contaminated, the purge stops it "
                            "being contaminated. Default on.")
    group.add_argument(f"--{prefix}fail-on-leakage", dest="fail_on_leakage",
                       action="store_true",
                       help="Exit non-zero when residual leakage exceeds the gates, so a "
                            "slurm job stops instead of producing an inflated number.")
    group.add_argument(f"--{prefix}leakage-min-identity", dest="leakage_min_identity",
                       default=str(DEFAULT_MIN_IDENTITY),
                       help=f"%%AA identity at or above which a deep sequence is a near "
                            f"neighbour of the target and is DROPPED (default "
                            f"{DEFAULT_MIN_IDENTITY}; 'none' disables this rule). Identity is "
                            f"nident/aligned_length from the BLAST alignment. 99%% matches the "
                            f"deep set's own clu99 clustering granularity. COMBINED WITH "
                            f"--leakage-max-hamming BY *OR*.")
    group.add_argument(f"--{prefix}leakage-max-hamming", dest="leakage_max_hamming",
                       default=str(DEFAULT_MAX_HAMMING),
                       help=f"AA mismatches at or below which a deep sequence is DROPPED "
                            f"(default {DEFAULT_MAX_HAMMING}; 'none' disables this rule). "
                            f"Hamming = min(qlen,slen) - nident, i.e. mismatches + gap columns "
                            f"+ unaligned overhang. NOT equivalent to the identity rule: on a "
                            f"~550 aa HA, 10 mismatches is ~98.2%% identity, so at the defaults "
                            f"THIS rule is the stricter one and governs the purge.")
    group.add_argument(f"--{prefix}leakage-min-coverage", dest="leakage_min_coverage",
                       type=float, default=DEFAULT_MIN_COVERAGE,
                       help=f"Minimum alignment coverage (per --leakage-coverage-basis, default "
                            f"'both' = both sequences) before either purge rule may fire "
                            f"(default {DEFAULT_MIN_COVERAGE}). A GATE, not a rule: it stops a "
                            f"short high-identity local hit -- an HA1 fragment, a conserved stalk "
                            f"segment -- being read as a full-length duplicate.")
    group.add_argument(f"--{prefix}leakage-coverage-basis", dest="leakage_coverage_basis",
                       choices=("both", "shorter", "query", "subject"), default="both",
                       help="Denominator for coverage. 'both' (default) requires BOTH sequences "
                            "to be almost entirely inside the alignment, i.e. "
                            "min(aln/qlen, aln/slen). 'query' alone lets a real duplicate escape "
                            "(the deep set keeps the 16-aa signal peptide the panels lack, so a "
                            "duplicate reaches only ~97%% query coverage); 'shorter' alone lets a "
                            "200-aa HA1 fragment in at 100%%. Both failure modes were reproduced "
                            "on the real data before this default was chosen.")
    group.add_argument(f"--{prefix}leakage-max-removed-fraction",
                       dest="leakage_max_removed_fraction", type=float,
                       default=DEFAULT_MAX_REMOVED_FRACTION,
                       help=f"Warn loudly when the purge removes more than this fraction of the "
                            f"deep MSA (default {DEFAULT_MAX_REMOVED_FRACTION}). MSA depth drives "
                            f"GEMME quality, so over-removal is its own failure mode.")
    group.add_argument(f"--{prefix}leakage-min-depth-after", dest="leakage_min_depth_after",
                       type=int, default=DEFAULT_MIN_DEPTH_AFTER,
                       help=f"Hard floor: refuse to proceed if the purge leaves fewer rows than "
                            f"this (default {DEFAULT_MIN_DEPTH_AFTER}).")
    group.add_argument(f"--{prefix}leakage-threads", dest="leakage_threads", type=int,
                       default=min(8, os.cpu_count() or 4),
                       help="blastp -num_threads.")
    group.add_argument(f"--{prefix}blastp-bin", dest="blastp_bin", default="blastp")
    group.add_argument(f"--{prefix}makeblastdb-bin", dest="makeblastdb_bin", default="makeblastdb")
    group.add_argument(f"--{prefix}blast-task", dest="blast_task",
                       choices=("blastp", "blastp-fast"), default=DEFAULT_BLAST_TASK,
                       help="blastp-fast (default) is ~3x faster and cannot miss a >=95%% "
                            "identity hit; 'blastp' is the exhaustive fallback.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="BLAST-based leakage detection and purge for the ESCOTT/PRESCOTT HA pipeline.",
        epilog=(
            "THRESHOLD INTERACTION\n"
            "  --leakage-min-identity and --leakage-max-hamming are combined with OR:\n"
            "  a deep sequence is dropped if EITHER fires (and coverage clears the gate).\n"
            "  They are not equivalent. On a ~550 aa mature HA:\n"
            "      10 mismatches  ==  98.18% identity\n"
            "      99% identity   ==   5.5 mismatches\n"
            "  So at the defaults the HAMMING rule is the stricter one and is what\n"
            "  actually governs the purge. Setting only --leakage-min-identity will not\n"
            "  loosen the purge unless --leakage-max-hamming none is set too.\n"
        ),
    )
    parser.add_argument("--guide-path", type=Path, default=DEFAULT_GUIDE,
                        help="Lineage guide CSV (month,fasta,reference).")
    parser.add_argument("--deep-fasta", type=Path, default=DEFAULT_DEEP_FASTA,
                        help="Deep pre-cutoff HA protein set used as the ESCOTT MSA source.")
    parser.add_argument("--report-dir", type=Path, required=True,
                        help="Where leakage_report.{tsv,json}, leakage_summary.txt and the "
                             "per-target hit tables are written.")
    parser.add_argument("--target", action="append", default=None,
                        help="Evaluation target lineage (repeatable). Default: every guide "
                             "lineage that has a parent, i.e. every evaluable target.")
    parser.add_argument("--parent-map", default=None,
                        help="Override parent edges, e.g. 'K=J.2_int'.")
    parser.add_argument("--parent-map-preset", choices=tuple(common.PARENT_MAP_PRESETS),
                        default=common.DEFAULT_PARENT_MAP_PRESET)

    parser.add_argument("--msa", type=Path, default=None,
                        help="Purge/inspect THIS alignment instead of --deep-fasta. Use with "
                             "--purge-only --target <lineage> on a prepared "
                             "inputs/msa/msa_<key>.fasta.")
    parser.add_argument("--out-purged", type=Path, default=None,
                        help="Where the purged FASTA goes (default: overwrite --msa in place, "
                             "keeping the original as <stem>_prepurge.fasta).")
    parser.add_argument("--purge-only", action="store_true",
                        help="Skip checks A/B/C and only run the purge.")
    parser.add_argument("--check-only", action="store_true",
                        help="Skip the purge and only report. Equivalent to --no-purge-leakage.")
    parser.add_argument("--flag-identity", type=float, default=DEFAULT_FLAG_IDENTITY,
                        help="Reporting threshold for checks A/B (separate from the purge "
                             "thresholds so a run can report at one stringency and purge at "
                             "another).")
    parser.add_argument("--flag-coverage", type=float, default=DEFAULT_FLAG_COVERAGE,
                        help="Reporting coverage gate for checks A/B.")
    parser.add_argument("--max-report-rows", type=int, default=500,
                        help="Cap on rows written to each per-target hit table.")
    parser.add_argument("--hash-suffix-length", type=int, default=500,
                        help="C-terminal suffix length for the check-C suffix hash "
                             "(0 disables). Invariant to the signal-peptide frame difference.")
    parser.add_argument("--max-target-seqs", type=int, default=DEFAULT_MAX_TARGET_SEQS)
    parser.add_argument("--evalue", type=float, default=DEFAULT_EVALUE)
    parser.add_argument("--force", action="store_true", help="Ignore purge caches.")

    add_leakage_arguments(parser)
    return parser


def _blast_kwargs(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "blast_task": args.blast_task,
        "evalue": args.evalue,
        "max_target_seqs": args.max_target_seqs,
        "threads": args.leakage_threads,
        "makeblastdb_bin": args.makeblastdb_bin,
        "blastp_bin": args.blastp_bin,
    }


def _purge_kwargs(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "min_identity": parse_threshold(args.leakage_min_identity, "--leakage-min-identity"),
        "max_hamming": parse_threshold(args.leakage_max_hamming, "--leakage-max-hamming", int),
        "min_coverage": float(args.leakage_min_coverage),
        "coverage_basis": args.leakage_coverage_basis,
        "max_removed_fraction": float(args.leakage_max_removed_fraction),
        "min_depth_after": int(args.leakage_min_depth_after),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.check_only:
        args.purge_leakage = False
    if args.purge_only:
        args.leakage_check = False

    report_dir = common.ensure_dir(args.report_dir.resolve())
    workdir = common.ensure_dir(report_dir / "blast_work")

    guide_rows = common.read_guide_rows(args.guide_path)
    labels = [row["label"] for row in guide_rows]
    panel_by_label = {row["label"]: Path(row["diversity_path"]) for row in guide_rows}
    parent_map = common.resolve_parent_map(args.parent_map_preset, args.parent_map, labels)

    targets = list(args.target) if args.target else [l for l in labels if l in parent_map]
    unknown = [t for t in targets if t not in labels]
    if unknown:
        raise SystemExit(f"--target names lineages absent from the guide: {unknown}")

    purge_kwargs = _purge_kwargs(args)
    blast_kwargs = _blast_kwargs(args)

    rows: List[Dict[str, object]] = []
    summary: List[str] = []
    payload: Dict[str, object] = {
        "generated_by": "scripts/prescott_iav/leakage_check.py",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "deep_fasta": str(args.deep_fasta),
        "deep_fasta_md5": common.md5_file(args.deep_fasta) if args.deep_fasta.exists() else None,
        "guide_path": str(args.guide_path),
        "parent_map": parent_map,
        "targets": targets,
        "purge_thresholds": {**purge_kwargs,
                             "combination": "OR (drop if EITHER fires)"},
        "blast": blast_kwargs,
        "checks": {},
        "purges": {},
    }

    summary.append("LEAKAGE AUDIT -- ESCOTT/PRESCOTT influenza HA pipeline")
    summary.append(f"generated {payload['generated_at']}")
    summary.append("")
    summary.append(f"deep evolutionary set : {args.deep_fasta}")
    summary.append(f"targets               : {', '.join(targets)}")
    summary.append(f"parent map            : {parent_map}")
    summary.append(
        f"purge rule            : coverage >= {purge_kwargs['min_coverage']}% "
        f"({purge_kwargs['coverage_basis']} basis) AND "
        f"(identity >= {purge_kwargs['min_identity']} OR hamming <= {purge_kwargs['max_hamming']}); "
        "the two are combined with OR, and at these values the Hamming rule is the stricter."
    )
    summary.append("")

    failures: List[str] = []

    # ---- Check C (cheap, first) ------------------------------------------ #
    if args.leakage_check:
        sets: Dict[str, Path] = {"deep_msa": args.deep_fasta}
        for label in labels:
            sets[f"panel:{label}"] = panel_by_label[label]
        hash_report = check_hash_duplicates(sets, suffix_length=args.hash_suffix_length)
        payload["checks"]["C_hash_duplicates"] = hash_report
        summary.append("CHECK C -- exact duplicates by sequence hash")
        for label, info in hash_report["sets"].items():
            summary.append(f"  {label:20s} {info['n_records']:7d} records, "
                           f"{info['n_unique_sequences']:6d} unique, "
                           f"lengths {info['min_length']}-{info['max_length']}")
        for pair in hash_report["pairs"]:
            if pair["n_shared_exact_sequences"] or pair["n_shared_cterm_suffix"]:
                summary.append(f"  OVERLAP {pair['set_a']} <-> {pair['set_b']}: "
                               f"{pair['n_shared_exact_sequences']} exact, "
                               f"{pair['n_shared_cterm_suffix']} C-term-suffix")
            rows.append({
                "check": "C_hash_duplicates", "target_lineage": "",
                "query_set": pair["set_a"], "subject_set": pair["set_b"],
                "n_query": "", "n_subject": "",
                "metric": "n_shared_exact_sequences",
                "value": pair["n_shared_exact_sequences"], "note": "",
            })
            rows.append({
                "check": "C_hash_duplicates", "target_lineage": "",
                "query_set": pair["set_a"], "subject_set": pair["set_b"],
                "n_query": "", "n_subject": "",
                "metric": "n_shared_cterm_suffix",
                "value": pair["n_shared_cterm_suffix"],
                "note": f"C-terminal {args.hash_suffix_length} aa",
            })
        summary.append(f"  {hash_report['n_pairs_with_exact_overlap']} of "
                       f"{len(hash_report['pairs'])} set pairs share an exact sequence")
        summary.append(f"  NOTE: {hash_report['caveat']}")
        summary.append("")
        for pair in hash_report["pairs"]:
            if pair["set_a"] == "deep_msa" and pair["n_shared_exact_sequences"]:
                failures.append(
                    f"check C: {pair['n_shared_exact_sequences']} exact duplicate sequences "
                    f"between the deep set and {pair['set_b']}"
                )

    # ---- Checks A and B, per target -------------------------------------- #
    if args.leakage_check:
        deep_records = load_ungapped(args.deep_fasta)
        for target in targets:
            summary.append(f"CHECK A -- deep evolutionary set vs panel:{target}")
            check_a = check_msa_vs_target(
                deep_records, panel_by_label[target], target, workdir,
                flag_identity=args.flag_identity, flag_coverage=args.flag_coverage,
                coverage_basis=args.leakage_coverage_basis,
                max_report_rows=args.max_report_rows, **blast_kwargs,
            )
            rows.extend(_report_rows_from_check(check_a))
            payload["checks"].setdefault("A_msa_vs_target", {})[target] = {
                "n_query": check_a.n_query, "n_subject": check_a.n_subject,
                "n_subject_unique": check_a.n_subject_unique,
                "n_query_with_hit": check_a.n_query_with_hit,
                "n_exact_full_length": check_a.n_exact_full_length,
                "n_flagged": check_a.n_flagged,
                "distribution": check_a.distribution,
                "thresholds": check_a.thresholds,
            }
            _write_hits_tsv(report_dir / f"leakage_A_{common.safe_label(target)}_hits.tsv",
                            check_a.flagged)
            dist = check_a.distribution
            summary.append(f"  {check_a.n_query} deep sequences vs {check_a.n_subject} panel "
                           f"records ({check_a.n_subject_unique} unique)")
            summary.append(f"  {check_a.n_query_with_hit} deep sequences have any hit; "
                           f"best-hit identity max {dist.get('max')}%, median "
                           f"{(dist.get('percentiles') or {}).get('p50')}%")
            summary.append(f"  exact full-length matches: {check_a.n_exact_full_length}")
            summary.append(f"  flagged (>= {args.flag_identity}% id over >= "
                           f"{args.flag_coverage}% coverage): {check_a.n_flagged}")
            summary.append("")
            if check_a.n_flagged and not args.purge_leakage:
                failures.append(
                    f"check A: {check_a.n_flagged} deep sequences are >= {args.flag_identity}% "
                    f"identical to a {target} panel sequence and the purge is OFF"
                )

            parent = parent_map.get(target)
            if parent:
                summary.append(f"CHECK B -- panel:{parent} (frequency input) vs panel:{target}")
                check_b, accessions = check_parent_vs_target(
                    panel_by_label[parent], panel_by_label[target], target, parent, workdir,
                    flag_identity=args.flag_identity, flag_coverage=args.flag_coverage,
                    coverage_basis=args.leakage_coverage_basis,
                    max_report_rows=args.max_report_rows, **blast_kwargs,
                )
                rows.extend(_report_rows_from_check(check_b))
                for metric in ("n_shared_accessions", "shared_accession_fraction_of_target",
                               "n_shared_exact_sequences"):
                    rows.append({
                        "check": "B_parent_vs_target_accessions",
                        "target_lineage": target,
                        "query_set": f"panel:{parent}", "subject_set": f"panel:{target}",
                        "n_query": accessions["n_parent_accessions"],
                        "n_subject": accessions["n_target_accessions"],
                        "metric": metric, "value": accessions[metric],
                        "note": "same accession namespace, so ID matching is valid HERE",
                    })
                payload["checks"].setdefault("B_parent_vs_target", {})[target] = {
                    "parent": parent,
                    "n_query": check_b.n_query, "n_subject": check_b.n_subject,
                    "n_exact_full_length": check_b.n_exact_full_length,
                    "n_flagged": check_b.n_flagged,
                    "distribution": check_b.distribution,
                    "accessions": accessions,
                }
                _write_hits_tsv(
                    report_dir / f"leakage_B_{common.safe_label(parent)}_vs_"
                                 f"{common.safe_label(target)}_hits.tsv",
                    check_b.flagged,
                )
                summary.append(f"  accession overlap: {accessions['n_shared_accessions']} of "
                               f"{accessions['n_target_accessions']} target accessions "
                               f"({accessions['shared_accession_fraction_of_target']:.4%})")
                summary.append(f"  exact shared sequences: "
                               f"{accessions['n_shared_exact_sequences']}")
                summary.append(f"  alignment-flagged parent sequences: {check_b.n_flagged} "
                               f"(>= {args.flag_identity}% id)")
                summary.append("")
                if accessions["n_shared_accessions"]:
                    failures.append(
                        f"check B: parent panel {parent} shares "
                        f"{accessions['n_shared_accessions']} accessions with its evaluation "
                        f"target {target} -- the frequency prior contains the answer"
                    )

    # ---- D: THE PURGE ----------------------------------------------------- #
    if args.purge_leakage:
        summary.append("PURGE -- deep MSA vs each evaluation target")
        summary.append(f"  {'target':10s} {'before':>8s} {'removed':>8s} {'after':>8s} "
                       f"{'removed%':>9s}  {'max id%':>8s} {'min ham':>8s}")
        for target in targets:
            panel = panel_by_label[target]
            if args.msa is not None:
                msa_path = args.msa
                out_fasta = args.out_purged or (
                    report_dir / f"msa_{common.safe_label(target)}_purged.fasta"
                )
                report = purge_msa_against_target(
                    msa_path, panel, out_fasta,
                    report_dir / f"{common.safe_label(target)}_purge_dropped.tsv",
                    target_lineage=target, workdir=workdir,
                    **purge_kwargs, **blast_kwargs,
                )
            else:
                # No prepared MSA given: purge the raw deep set itself, which is the
                # honest standalone answer to "how much of this file is a leak".
                # There is no query row to protect, so nothing is exempted.
                out_fasta = args.out_purged or (
                    report_dir / f"deep_purged_vs_{common.safe_label(target)}.fasta"
                )
                report = purge_msa_against_target(
                    args.deep_fasta, panel, out_fasta,
                    report_dir / f"{common.safe_label(target)}_purge_dropped.tsv",
                    target_lineage=target, workdir=workdir,
                    protect_indices=(), keep_alignment_columns=True,
                    **purge_kwargs, **blast_kwargs,
                )
            payload["purges"][target] = report
            dist = report["removed_identity_distribution"]
            summary.append(
                f"  {target:10s} {report['depth_before']:8d} {report['n_removed']:8d} "
                f"{report['depth_after']:8d} {report['removed_fraction']:9.2%}  "
                f"{str(dist.get('max')):>8s} {str(report['removed_hamming_min']):>8s}"
            )
            for metric in ("depth_before", "n_removed", "depth_after", "removed_fraction",
                           "n_removed_by_identity_only", "n_removed_by_hamming_only",
                           "n_removed_by_both", "n_removed_exact_full_length"):
                rows.append({
                    "check": "D_purge", "target_lineage": target,
                    "query_set": "deep_msa", "subject_set": f"panel:{target}",
                    "n_query": report["depth_before"], "n_subject": report["target_panel_records"],
                    "metric": metric, "value": report[metric], "note": "",
                })
        summary.append("")
        summary.append("  NOTE (trap 3): the purge is PER TARGET, so the purged alignment is "
                       "target-specific. ESCOTT must be run once per evaluation target; a single "
                       "shared ESCOTT run is no longer valid.")
        summary.append("")

    # ---- verdict ---------------------------------------------------------- #
    status = "FAIL" if failures else "PASS"
    payload["status"] = status
    payload["failures"] = failures
    summary.append(f"VERDICT: {status}")
    for failure in failures:
        summary.append(f"  ! {failure}")
    if not failures:
        summary.append("  no residual leakage above the configured gates")

    paths = write_report(report_dir, rows, payload, summary)
    print("\n".join(summary))
    print(f"@> Wrote {paths['tsv']}")
    print(f"@> Wrote {paths['json']}")
    print(f"@> Wrote {paths['txt']}")

    shutil.rmtree(workdir, ignore_errors=True)

    if failures and args.fail_on_leakage:
        return 3
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (LeakageError, MsaDepthError, BlastNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
