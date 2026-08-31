#!/usr/bin/env python3
"""Stage A of the PRESCOTT/ESCOTT influenza HA pipeline: build every ESCOTT input.

Four products, all written under ``--inputs-dir``:

``structure/``   6WXB (or a per-lineage model) reduced to a single renumbered
                 chain, plus the trimer, with construct artefacts stripped and
                 the residue numbering shifted into the 566-aa HA0 query frame.
``query/``       one 566-aa protein FASTA per lineage, translated from the
                 per-lineage nucleotide CDS with exactly the same code path the
                 evaluation half uses, and headed ``>HA<TAG>`` so that escott's
                 own ``prot`` token is predictable.
``msa/``         one deep MSA (6433 pre-cutoff NCBI HA sequences + the anchor
                 query) reduced to the query's 566 columns, then re-anchored per
                 lineage, then PURGED of every row that is a near neighbour of the
                 lineage's own evaluation panel (``msa_<key>_prepurge.fasta`` keeps
                 the unpurged original).  See ``leakage_check.py``: the purge is
                 what stops a 2025 GISAID sequence sitting in a 2024-cutoff
                 evolutionary set and inflating every reported correlation.  It is
                 PER TARGET, so ``msa_<key>.fasta`` may never be shared between
                 lineages.
``leakage/``     the audit trail for the above: which rows were dropped, which
                 target sequence each one matched, at what identity and Hamming
                 distance, plus the parent-vs-target and hash checks.
``frequency/``   one PRESCOTT custom frequency file per *evaluable* lineage,
                 built from the panel of its BASAL lineage rather than its own.

Run under /home3/oml4h/miniconda3/envs/PRESCOTT/bin/python.  Nothing here imports
torch, esm or Functions_HuggingFace -- see scripts/prescott_iav/common.py for why.

Example
-------
    /home3/oml4h/miniconda3/envs/PRESCOTT/bin/python \\
        scripts/prescott_iav/prepare_inputs.py \\
        --inputs-dir Results/iav_prescott_diversity/smoke/inputs
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from prescott_iav import common  # noqa: E402
from prescott_iav import leakage_check  # noqa: E402


DEFAULT_GUIDE = REPO_ROOT / "Sequences" / "IAV_lineage_guide.csv"
DEFAULT_STRUCTURE = REPO_ROOT / "Sequences" / "6WXB-assembly1.cif"
DEFAULT_EXTRA_STRUCTURE = (
    REPO_ROOT
    / "Sequences"
    / "EPI4748783_HA_A_England_01837755_2025_EPI_ISL_20210731_J_2_4_1_model.cif"
)
DEFAULT_DEEP_FASTA = Path(
    "/home3/oml4h/my_SC2_finetunes/myflu/full_job4243186/data/"
    "ncbiflu_HA_all_110424_noX_clu99_filt_HA-80.clean_input.fasta"
)
DEFAULT_MAFFT = Path("/home3/oml4h/miniconda3/envs/PRESCOTT/bin/mafft")

# 6WXB is a stabilised ectodomain construct.  The deposited coordinates happen to
# stop before these, but we test for them explicitly rather than trusting that:
# a silently retained foldon would be read as 34 extra "conserved" HA2 residues.
CONSTRUCT_MOTIFS = {
    "linker": "GGGGTGGGGT",
    "foldon": "GRMKQIEDKIEEILSKIYHIENEIARIKKLIGER",
    "his_tag": "HHHHHH",
}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare structure, query, MSA and basal-frequency inputs for ESCOTT/PRESCOTT.",
    )
    parser.add_argument("--guide-path", type=Path, default=DEFAULT_GUIDE,
                        help="Lineage guide CSV with columns month,fasta,reference.")
    parser.add_argument("--deep-fasta", type=Path, default=DEFAULT_DEEP_FASTA,
                        help="Deep pre-cutoff HA protein set used as the ESCOTT MSA source.")
    parser.add_argument("--inputs-dir", type=Path, required=True,
                        help="Output root, normally <output-dir>/inputs.")

    parser.add_argument("--structure", type=Path, default=DEFAULT_STRUCTURE,
                        help="Primary structure (mmCIF or PDB) used by the JET2 surrogate.")
    parser.add_argument("--structure-chain", default="A",
                        help="Chain to reduce to for the per-residue structural terms.")
    parser.add_argument("--structure-offset", default="auto",
                        help="Integer added to author residue numbers to reach query numbering, or 'auto'.")
    parser.add_argument("--extra-structure", type=Path, default=DEFAULT_EXTRA_STRUCTURE,
                        help="Optional second structure (full-coverage contemporary model) prepared alongside.")
    parser.add_argument("--no-extra-structure", action="store_true",
                        help="Skip the secondary structure entirely.")
    parser.add_argument("--structure-min-identity", type=float, default=0.60,
                        help="Abort if the best offset gives less sequence identity than this.")

    parser.add_argument("--msa-anchor", default="G.1",
                        help="Lineage whose query anchors the single mafft run.")
    parser.add_argument("--mafft-bin", type=Path, default=DEFAULT_MAFFT,
                        help="mafft executable (pinned to the PRESCOTT env by default).")
    parser.add_argument("--mafft-threads", type=int, default=8)
    parser.add_argument("--mafft-mode", choices=("auto", "keeplength"), default="auto",
                        help="'auto' aligns everything then strips query-gap columns; "
                             "'keeplength' adds the deep set onto the fixed-length query.")
    parser.add_argument("--msa-nonstandard", choices=("to-x", "to-gap", "keep"), default="to-x",
                        help="What to do with non-standard residues (B/Z/J/U/O) in the deep rows. "
                             "'to-x' matches the shipped PRESCOTT BLAT alignment, which contains X.")

    parser.add_argument("--parent-map", default=None,
                        help="Override individual edges, e.g. 'K=J.2_int,J.2.4=J.2_int'.")
    parser.add_argument("--parent-map-preset", choices=tuple(common.PARENT_MAP_PRESETS),
                        default=common.DEFAULT_PARENT_MAP_PRESET)
    parser.add_argument("--sensitivity-parent-map", default=None,
                        help="ALTERNATE parent edges to prepare IN ADDITION to the primary map, "
                             "e.g. 'K=J.2_int'. Each writes a second frequency file "
                             "<key>_parent<TOK>_frequency.txt (K_parentJ2int_frequency.txt) which "
                             "run_escott.py --sensitivity-parent-map turns into a _parent<TOK> "
                             "score-matrix variant. This is what makes --parent-sensitivity real.")
    parser.add_argument("--sensitivity-preset", choices=tuple(common.PARENT_MAP_PRESETS),
                        default=None,
                        help="Derive the alternate edges automatically as the edges on which this "
                             "preset disagrees with --parent-map-preset (edges where they agree "
                             "would only produce byte-identical duplicates).")
    parser.add_argument("--parent-min-count", type=int, default=1,
                        help="Minimum observed count in the basal panel for a mutant to be emitted.")
    parser.add_argument("--parent-min-depth", type=int, default=50,
                        help="Minimum valid-residue depth at a column for it to contribute mutants.")
    parser.add_argument("--parent-freq-max", type=float, default=0.95,
                        help="Drop mutants at or above this basal frequency (reverse-substitution guard).")
    parser.add_argument("--drop-parent-reversions", dest="drop_parent_reversions",
                        action="store_true", default=True,
                        help="Also drop, regardless of frequency, any mutant whose mutant residue is the "
                             "PARENT reference's residue at that site (i.e. an exact reversion).")
    parser.add_argument("--no-drop-parent-reversions", dest="drop_parent_reversions",
                        action="store_false")

    parser.add_argument("--frequency-cutoff-mode", choices=("depth_scaled", "fixed"),
                        default="depth_scaled")
    parser.add_argument("--frequency-cutoff-k", default="1",
                        help="Comma-separated k values; Fc = log10(k / median_parent_depth).")
    parser.add_argument("--frequency-cutoff", type=float, default=-4.0,
                        help="Fc used when --frequency-cutoff-mode fixed.")

    parser.add_argument("--only-lineage", action="append", default=None,
                        help="Restrict work to these lineage labels (repeatable).")
    parser.add_argument("--force", action="store_true",
                        help="Recompute everything even if cached outputs are present.")
    parser.add_argument("--seed", type=int, default=20260805,
                        help="Recorded in the manifest and handed to the jet surrogate.")

    # Leakage detection and PURGE. Attached from leakage_check so that the flag
    # names, defaults and help text cannot drift between the three CLIs that carry
    # them (this one, run_prescott_diversity.py and the slurm wrapper) -- a
    # threshold that means one thing to the driver and another to stage 1 would
    # silently change which sequences ESCOTT is allowed to see.
    leakage_check.add_leakage_arguments(parser)
    parser.add_argument("--leakage-hash-suffix-length", type=int, default=500,
                        help="C-terminal suffix length for the check-C hash (0 disables).")
    return parser


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #

def load_structure(path: Path):
    """Parse an mmCIF or PDB into a prody AtomGroup.

    prody is imported lazily so that the query/MSA/frequency stages still run on a
    machine without it (they never touch coordinates).
    """
    import prody

    prody.confProDy(verbosity="none")
    suffix = path.suffix.lower()
    if suffix in (".cif", ".mmcif"):
        return prody.parseMMCIF(str(path))
    if suffix in (".pdb", ".ent"):
        return prody.parsePDB(str(path))
    raise ValueError(f"Unsupported structure format for {path}")


def _chain_ca(struct, chain: str):
    sel = struct.select(f"protein and chain {chain} and name CA and not hetero")
    if sel is None:
        raise ValueError(f"No protein CA atoms in chain {chain}")
    return sel


def resolve_structure_offset(
    struct,
    chain: str,
    query_protein: str,
    min_identity: float,
    search: Tuple[int, int] = (-40, 41),
) -> Dict[str, object]:
    """Find the integer shift taking author residue numbers into query numbering.

    A positional scan rather than a pairwise alignment on purpose: HA crystal
    structures are numbered in the mature-HA1 convention with no indels relative
    to the HA0 reference, so the correct answer is a pure translation and a scan
    both finds it and *proves* there are no indels (a wrong offset scores near
    chance, ~5%).  For 6WXB the winner is +16 at 84.5% identity; for the
    contemporary 566-residue model it is 0.
    """
    ca = _chain_ca(struct, chain)
    resnums = ca.getResnums()
    seq = ca.getSequence()

    scored: List[Tuple[float, int, int, int]] = []
    for offset in range(*search):
        matches = 0
        compared = 0
        for resnum, aa in zip(resnums, seq):
            pos = int(resnum) + offset
            if 1 <= pos <= len(query_protein):
                compared += 1
                matches += query_protein[pos - 1] == aa
        if compared == 0:
            continue
        scored.append((matches / compared, offset, matches, compared))

    if not scored:
        raise ValueError("No offset places any residue inside the query frame")

    # Rank by absolute match count, not fraction: a tiny overhang can be 100%
    # identical by luck, whereas the true offset maximises matched residues.
    scored.sort(key=lambda item: (item[2], item[0]), reverse=True)
    identity, offset, matches, compared = scored[0]
    if identity < min_identity:
        raise ValueError(
            f"Best offset {offset:+d} only reaches {identity:.1%} identity over {compared} residues "
            f"(threshold {min_identity:.0%}); the structure and the query are probably not the same protein"
        )
    return {
        "offset": int(offset),
        "identity": float(identity),
        "matched_residues": int(matches),
        "compared_residues": int(compared),
    }


def _find_construct_truncation(struct, chain: str) -> Optional[int]:
    """Return the first author resnum belonging to a construct artefact, if any.

    Everything from that residue onward is dropped.  Returns None when the chain
    is clean, which is the case for 6WXB (coordinates stop at author 501, i.e.
    query 517, well before any linker would start).
    """
    ca = _chain_ca(struct, chain)
    seq = ca.getSequence()
    resnums = ca.getResnums()
    first: Optional[int] = None
    for name, motif in CONSTRUCT_MOTIFS.items():
        idx = seq.find(motif)
        if idx >= 0:
            resnum = int(resnums[idx])
            print(f"@> Construct artefact {name!r} found in chain {chain} at author resnum {resnum}; truncating.")
            first = resnum if first is None else min(first, resnum)
    return first


def _residue_report(resnums: Sequence[int]) -> Dict[str, object]:
    ordered = sorted(set(int(r) for r in resnums))
    gaps = [[a, b] for a, b in zip(ordered[:-1], ordered[1:]) if b != a + 1]
    return {
        "n_residues": len(ordered),
        "resnum_min": ordered[0] if ordered else None,
        "resnum_max": ordered[-1] if ordered else None,
        "gaps": gaps,
    }


def _uncovered_runs(covered: Sequence[int], length: int) -> List[List[int]]:
    covered_set = set(int(p) for p in covered)
    runs: List[List[int]] = []
    for pos in range(1, length + 1):
        if pos in covered_set:
            continue
        if runs and pos == runs[-1][1] + 1:
            runs[-1][1] = pos
        else:
            runs.append([pos, pos])
    return runs


def write_renumbered_pdb(
    struct,
    chains: Sequence[str],
    offset: int,
    query_length: int,
    out_path: Path,
    truncate_at: Optional[int] = None,
    collapse_to_chain_a: bool = False,
) -> Dict[str, object]:
    """Write a renumbered protein-only PDB and report its coverage.

    Glycans go away via ``not hetero`` (6WXB's only heteroatoms are NAG/BMA);
    construct artefacts via ``truncate_at``; anything that still falls outside
    1..query_length after the shift is dropped, which is the generic guard
    against tags numbered off the end of the real sequence.
    """
    import prody

    chain_spec = " ".join(chains)
    sel = struct.select(f"protein and chain {chain_spec} and not hetero")
    if sel is None:
        raise ValueError(f"No protein atoms in chains {chain_spec} of the structure")
    sel = sel.copy()

    if truncate_at is not None:
        kept = sel.select(f"resnum < {truncate_at}")
        if kept is None:
            raise ValueError("Construct truncation removed the entire chain")
        sel = kept.copy()

    sel.setResnums(sel.getResnums() + offset)
    sel = sel.select(f"resnum 1 to {query_length}")
    if sel is None:
        raise ValueError("No residues fall inside the query frame after renumbering")
    sel = sel.copy()

    if collapse_to_chain_a:
        # The jet surrogate keys purely on residue number, so the single-chain
        # file is forced to chain A to keep escott's own PDB reader unambiguous.
        sel.setChids(["A"] * sel.numAtoms())

    icodes = set(sel.getIcodes())
    if icodes - {""}:
        raise ValueError(f"Insertion codes present ({sorted(icodes)}); the renumbering assumption is broken")

    common.ensure_dir(out_path.parent)
    prody.writePDB(str(out_path), sel)

    report: Dict[str, object] = {"path": str(out_path), "chains": list(chains)}
    per_chain: Dict[str, Dict[str, object]] = {}
    for chid in sorted(set(sel.getChids())):
        ca = sel.select(f"chain {chid} and name CA")
        resnums = [int(r) for r in ca.getResnums()]
        if len(resnums) != len(set(resnums)):
            raise ValueError(f"Chain {chid} has more than one CA per residue number (altloc problem?)")
        per_chain[chid] = _residue_report(resnums)
    report["per_chain"] = per_chain
    return report


def prepare_structure(
    path: Path,
    chain: str,
    offset_arg: str,
    query_protein: str,
    out_dir: Path,
    stem: str,
    min_identity: float,
) -> Dict[str, object]:
    """Prepare the single-chain and trimer PDBs for one source structure."""
    struct = load_structure(path)

    if str(offset_arg).lower() == "auto":
        offset_info = resolve_structure_offset(struct, chain, query_protein, min_identity)
    else:
        offset_info = {"offset": int(offset_arg), "identity": None,
                       "matched_residues": None, "compared_residues": None}
    offset = int(offset_info["offset"])

    truncate_at = _find_construct_truncation(struct, chain)

    mono_path = out_dir / f"{stem}_chain{chain}_qnum.pdb"
    mono = write_renumbered_pdb(
        struct, [chain], offset, len(query_protein), mono_path,
        truncate_at=truncate_at, collapse_to_chain_a=True,
    )

    # The trimer exists purely as the SASA/circular-variance *environment*: the
    # HA trimer interface must read as buried, which it cannot in a monomer.
    all_chains = sorted({
        chid for chid in set(struct.getChids())
        if struct.select(f"protein and chain {chid} and name CA and not hetero") is not None
    })
    trimer_path = out_dir / f"{stem}_trimer_qnum.pdb"
    trimer = write_renumbered_pdb(
        struct, all_chains, offset, len(query_protein), trimer_path,
        truncate_at=truncate_at, collapse_to_chain_a=False,
    )

    covered = sorted(int(r) for r in _chain_ca(load_structure(mono_path), "A").getResnums())
    uncovered = _uncovered_runs(covered, len(query_protein))

    return {
        "source_path": str(path),
        "source_md5": common.md5_file(path),
        "chain": chain,
        "offset": offset,
        "offset_identity": offset_info["identity"],
        "offset_matched_residues": offset_info["matched_residues"],
        "construct_truncated_at_author_resnum": truncate_at,
        "monomer": mono,
        "trimer": trimer,
        "query_length": len(query_protein),
        "covered_positions": covered,
        "n_covered": len(covered),
        "coverage_fraction": len(covered) / len(query_protein),
        "uncovered_runs": uncovered,
    }


# --------------------------------------------------------------------------- #
# Queries
# --------------------------------------------------------------------------- #

def build_query_fastas(
    guide_rows: Sequence[Dict[str, str]],
    out_dir: Path,
) -> Dict[str, Dict[str, object]]:
    """Translate each lineage's nucleotide CDS into a 566-aa ``>HA<TAG>`` FASTA.

    The header is never ``HA<TAG>.fasta``-shaped by accident: escott writes its own
    ``<prot>.fasta`` into the CWD (escott.py:77-85) and would clobber a query file
    that shared that name, so the file is called ``<lineage_key>_query.fasta`` and
    only the *header* carries the tag.
    """
    common.ensure_dir(out_dir)
    queries: Dict[str, Dict[str, object]] = {}
    report_lines = ["lineage\tlineage_key\ttag\tprot_token\tlength\tmd5\treference_path"]

    for row in guide_rows:
        label = row["label"]
        # `read_guide_rows` requires only `label` and `fasta`, so a guide row with an
        # empty `reference` cell reaches here as "".  Path("") is PosixPath("."),
        # whose .exists() is True -- so an .exists() guard is bypassed entirely and
        # the run used to die with IsADirectoryError(".") from inside
        # load_reference_cds, naming neither the lineage nor the guide.  Test the
        # raw cell first, then demand a regular file rather than merely something
        # that exists.
        raw_reference = str(row.get("reference_path") or "").strip()
        if not raw_reference:
            raise FileNotFoundError(
                f"Reference CDS for {label} not found: the guide's `reference` column "
                f"is empty for this row. Every lineage needs a nucleotide CDS FASTA."
            )
        reference_path = Path(raw_reference)
        if not reference_path.is_file():
            what = " (it is a directory)" if reference_path.is_dir() else ""
            raise FileNotFoundError(
                f"Reference CDS for {label} not found: {reference_path}{what}"
            )

        payload = common.load_reference_cds(reference_path, label)
        protein = payload["protein"]

        illegal = sorted(set(protein) - set(common.STANDARD_AMINO_ACIDS))
        if illegal:
            raise ValueError(
                f"{label}: translated reference contains non-standard residues {illegal}; "
                "ESCOTT's alphabet cannot represent them"
            )

        tag = common.lineage_tag(label)
        header = f"HA{tag}"
        token = common.escott_prot_token(header)
        if token != header:
            raise ValueError(f"{label}: escott would truncate header {header!r} to {token!r}")

        key = common.safe_label(label)
        out_path = out_dir / f"{key}_query.fasta"
        common.write_fasta(out_path, [(header, protein)])

        queries[label] = {
            "lineage_key": key,
            "tag": tag,
            "header": header,
            "prot_token": token,
            "protein": protein,
            "length": len(protein),
            "path": str(out_path),
            "md5": common.md5_text(protein),
            "reference_path": str(reference_path),
            "reference_header": payload["header"],
            "nucleotide_length": len(payload["nucleotide"]),
        }
        report_lines.append(
            f"{label}\t{key}\t{tag}\t{token}\t{len(protein)}\t{common.md5_text(protein)}\t{reference_path}"
        )

    lengths = {info["length"] for info in queries.values()}
    if len(lengths) > 1:
        # Not fatal in principle, but every downstream assertion (566 columns in
        # the MSA, 566 columns in the jet table, one shared alignment frame)
        # assumes a single length, so make the violation loud here.
        raise ValueError(f"Lineage references translate to differing lengths: {sorted(lengths)}")

    (out_dir / "query_report.tsv").write_text("\n".join(report_lines) + "\n")
    return queries


# --------------------------------------------------------------------------- #
# Deep MSA
# --------------------------------------------------------------------------- #

def _sanitise_nonstandard(rows: List[Tuple[str, str]], policy: str) -> Tuple[List[Tuple[str, str]], int]:
    """Replace residues outside the 20-letter alphabet in the non-query rows.

    The shipped PRESCOTT alignment (PRESCOTT/data/aliBLAT.fasta) contains 45 'X'
    characters, so GEMME demonstrably tolerates X; it has no B/Z/J/U/O at all and
    we would rather not be the first to find out how ``lw-i.7`` indexes them.  The
    deep NCBI set carries 22 B and 3 Z across 3.6 M residues, so the cost of
    normalising them is nil and the risk of not doing it is a mid-run R traceback.
    Row 0 (the query) is never touched -- it is asserted pure 20-AA upstream.
    """
    if policy == "keep":
        return rows, 0
    replacement = "X" if policy == "to-x" else "-"
    allowed = set(common.STANDARD_AMINO_ACIDS) | {"-"}
    n_replaced = 0
    out = [rows[0]]
    for header, seq in rows[1:]:
        offenders = set(seq) - allowed
        if not offenders:
            out.append((header, seq))
            continue
        table = str.maketrans({char: replacement for char in offenders})
        n_replaced += sum(seq.count(char) for char in offenders)
        out.append((header, seq.translate(table)))
    return out, n_replaced


def _strip_query_gap_columns(rows: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Project every row onto the columns where row 0 (the query) is ungapped."""
    _header0, anchor = rows[0]
    keep = np.frombuffer(anchor.encode("ascii"), dtype=np.uint8) != ord("-")
    stripped: List[Tuple[str, str]] = []
    for header, seq in rows:
        arr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
        stripped.append((header, arr[keep].tobytes().decode("ascii")))
    return stripped


def build_deep_msa(
    deep_fasta: Path,
    anchor_label: str,
    anchor_info: Dict[str, object],
    out_dir: Path,
    mafft_bin: Path,
    threads: int,
    mode: str,
    nonstandard_policy: str,
    force: bool,
) -> Dict[str, object]:
    """Align the deep pre-cutoff set against the anchor query, query-first.

    Query-first and query-ungapped because GEMME/ESCOTT treats row 1 of the input
    as the query and indexes every prediction by its columns (computePred.R:31-34),
    so the alignment frame *is* the query frame and must be exactly 566 columns.
    """
    common.ensure_dir(out_dir)
    report_path = out_dir / "msa_report.json"
    msa_566 = out_dir / "deep_msa_566.fasta"

    deep_md5 = common.md5_file(deep_fasta)
    cache_key = {
        "deep_fasta_md5": deep_md5,
        "anchor_protein_md5": anchor_info["md5"],
        "mafft_mode": mode,
        "mafft_threads": int(threads),
        "msa_nonstandard": nonstandard_policy,
    }
    cached = common.read_json(report_path)
    if (not force) and cached and msa_566.exists() and cached.get("cache_key") == cache_key:
        print(f"@> MSA cache hit: {msa_566}")
        return cached

    anchor_header = str(anchor_info["header"])
    anchor_protein = str(anchor_info["protein"])
    deep_records = list(common.read_fasta(deep_fasta))
    if not deep_records:
        raise ValueError(f"Deep MSA source is empty: {deep_fasta}")

    raw_path = out_dir / "deep_msa_raw.fasta"
    started = time.time()

    if mode == "auto":
        combined = out_dir / "deep_plus_query.fasta"
        common.write_fasta(combined, [(anchor_header, anchor_protein)] + deep_records)
        cmd = [str(mafft_bin), "--auto", "--anysymbol", "--thread", str(threads), "--quiet", str(combined)]
    else:
        # keeplength pins the output to the query's own length, so no column
        # stripping is needed -- but insertions relative to the query are lost.
        query_only = out_dir / "deep_plus_query.fasta"
        common.write_fasta(query_only, [(anchor_header, anchor_protein)])
        deep_only = out_dir / "deep_only.fasta"
        common.write_fasta(deep_only, deep_records)
        cmd = [str(mafft_bin), "--keeplength", "--add", str(deep_only), "--anysymbol",
               "--thread", str(threads), "--quiet", str(query_only)]

    print(f"@> mafft: {' '.join(cmd)}")
    with open(raw_path, "w") as handle:
        proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"mafft failed ({proc.returncode}):\n{proc.stderr[-4000:]}")
    elapsed = time.time() - started

    rows = list(common.read_fasta(raw_path))
    if len(rows) != len(deep_records) + 1:
        raise ValueError(f"mafft returned {len(rows)} rows, expected {len(deep_records) + 1}")
    widths = {len(seq) for _h, seq in rows}
    if len(widths) != 1:
        raise ValueError(f"mafft output is ragged: widths {sorted(widths)}")
    raw_columns = widths.pop()

    rows = _strip_query_gap_columns(rows)
    rows, n_nonstandard = _sanitise_nonstandard(rows, nonstandard_policy)
    if rows[0][1] != anchor_protein:
        raise ValueError(
            "Row 1 of the stripped MSA does not reproduce the anchor query; "
            "the alignment frame would be wrong"
        )

    common.write_fasta(msa_566, rows)

    report = {
        "cache_key": cache_key,
        "deep_fasta": str(deep_fasta),
        "deep_fasta_md5": deep_md5,
        "n_deep_sequences": len(deep_records),
        "n_rows": len(rows),
        "anchor_lineage": anchor_label,
        "anchor_header": anchor_header,
        "raw_columns": raw_columns,
        "n_columns": len(rows[0][1]),
        "mafft_bin": str(mafft_bin),
        "mafft_version": _mafft_version(mafft_bin),
        "mafft_mode": mode,
        "mafft_threads": int(threads),
        "mafft_seconds": round(elapsed, 2),
        "msa_nonstandard": nonstandard_policy,
        "n_nonstandard_residues_replaced": int(n_nonstandard),
        "msa_566_path": str(msa_566),
        "msa_566_md5": common.md5_file(msa_566),
        "raw_path": str(raw_path),
    }
    common.write_json(report_path, report)
    return report


def _mafft_version(mafft_bin: Path) -> str:
    try:
        proc = subprocess.run([str(mafft_bin), "--version"], capture_output=True, text=True)
        return (proc.stderr or proc.stdout).strip().splitlines()[0]
    except Exception:  # pragma: no cover - purely informational
        return "unknown"


def materialise_lineage_msa(
    msa_566: Path,
    lineage_label: str,
    query_info: Dict[str, object],
    out_path: Path,
    anchor_label: str,
) -> Dict[str, object]:
    """Re-anchor the shared MSA on one lineage's query.

    Legal only because the five references are the same length with no indels
    between them, so swapping row 1 changes residues but never columns.  The
    number of residues that actually change is reported, since it is the entire
    justification for running escott five times instead of once.
    """
    rows = list(common.read_fasta(msa_566))
    protein = str(query_info["protein"])
    if len(protein) != len(rows[0][1]):
        raise ValueError(
            f"{lineage_label}: query is {len(protein)} aa but the MSA has {len(rows[0][1])} columns"
        )
    n_diff = sum(1 for a, b in zip(protein, rows[0][1]) if a != b)
    rows[0] = (str(query_info["header"]), protein)
    common.write_fasta(out_path, rows)
    return {
        "path": str(out_path),
        "md5": common.md5_file(out_path),
        "n_rows": len(rows),
        "n_columns": len(protein),
        "anchor_lineage": anchor_label,
        "n_query_residues_changed_vs_anchor": n_diff,
    }


# --------------------------------------------------------------------------- #
# Basal-lineage frequency files
# --------------------------------------------------------------------------- #

def _column_residue_counts(sequences: Sequence[str]) -> Tuple[np.ndarray, List[str], int]:
    """Per-column counts of the 20 standard residues in an aligned panel."""
    matrix, aln_len = common.aligned_byte_matrix(sequences)
    allowed = sorted(set(common.STANDARD_AMINO_ACIDS) - set(common.IGNORE_ALIGNMENT_CHARS))
    counts = np.zeros((len(allowed), aln_len), dtype=np.int64)
    for idx, aa in enumerate(allowed):
        counts[idx] = (matrix == ord(aa)).sum(axis=0)
    return counts, allowed, aln_len


def build_parent_frequency_file(
    child_label: str,
    parent_label: str,
    child_protein: str,
    parent_panel_fasta: Path,
    out_txt: Path,
    out_meta: Path,
    min_count: int,
    min_depth: int,
    freq_max: float,
    parent_protein: Optional[str] = None,
    drop_parent_reversions: bool = True,
) -> Dict[str, object]:
    """Write the PRESCOTT custom frequency file for ``child_label`` from its parent panel.

    Three decisions here are load-bearing and easy to get wrong:

    1. Unobserved mutants are OMITTED, never written as ``0.0``.  prescott.py seeds
       every mutant with the sentinel 999.0 (line 968) and skips those (line 735);
       a literal zero would become ``log10(0) = -inf`` at line 1113 and, under
       equation 3, drive PRESCOTT to maximal deleteriousness for every variant the
       parent never happened to sample.
    2. Mutants at or above ``freq_max`` are dropped.  At the handful of positions
       where the parent's consensus residue differs from the child's reference, the
       *ancestral* residue reaches frequency ~1.0 in the parent panel, and feeding
       that in would tell PRESCOTT that reverting the lineage-defining substitution
       is maximally tolerated -- the opposite of what we are trying to predict.
    3. ``drop_parent_reversions`` closes the hole left by (2).  A frequency
       threshold is the wrong instrument for a categorical problem: measured on the
       real data, K's reversion N160S sits at 0.932 and K176I at 0.897, so a 0.95
       cutoff lets two of the seven K-defining sites through while a cutoff low
       enough to catch them would start deleting genuine standing variation.  When
       the parent reference is available we therefore delete reversions by identity
       -- ``mut == parent_reference[pos]`` at a site where parent and child differ --
       which is exact and threshold-free.
    """
    sequences = common.read_fasta_sequences(parent_panel_fasta)
    if not sequences:
        raise ValueError(f"Parent panel {parent_panel_fasta} contains no records")

    ref_to_aln, aln_len, matched_pairs, consensus = common.map_reference_to_alignment_columns(
        child_protein, sequences
    )
    counts, allowed, counts_len = _column_residue_counts(sequences)
    if counts_len != aln_len:
        raise ValueError("Alignment length disagreement between consensus map and count matrix")

    depths = counts.sum(axis=0)
    mapped_positions = sorted(ref_to_aln)
    mapped_depths = [int(depths[ref_to_aln[pos] - 1]) for pos in mapped_positions]
    positive_depths = [d for d in mapped_depths if d > 0]
    median_depth = float(np.median(positive_depths)) if positive_depths else 0.0

    use_reversion_filter = bool(drop_parent_reversions and parent_protein)
    if use_reversion_filter and len(parent_protein) != len(child_protein):
        raise ValueError(
            f"{child_label}: parent reference is {len(parent_protein)} aa but child is "
            f"{len(child_protein)} aa; the reversion filter assumes a shared frame"
        )

    lines: List[str] = []
    meta_lines = ["mutant\tposition\twt\tmut\tcount\tdepth\tfrequency\tparent_lineage"]
    reverted: List[Dict[str, object]] = []
    parent_reversions: List[Dict[str, object]] = []
    n_shallow = 0
    n_below_min_count = 0

    for pos in mapped_positions:
        col = ref_to_aln[pos] - 1
        depth = int(depths[col])
        if depth < min_depth:
            n_shallow += 1
            continue
        wt = child_protein[pos - 1]
        parent_wt = parent_protein[pos - 1] if use_reversion_filter else None
        for idx, aa in enumerate(allowed):
            if aa == wt:
                continue
            count = int(counts[idx, col])
            if count <= 0:
                continue
            if count < min_count:
                n_below_min_count += 1
                continue
            freq = count / depth
            if parent_wt is not None and aa == parent_wt and parent_wt != wt:
                parent_reversions.append({"mutant": f"{wt}{pos}{aa}", "frequency": round(freq, 6),
                                          "count": count, "depth": depth})
                continue
            if freq >= freq_max:
                # Almost always the ancestral residue at a lineage-defining site.
                reverted.append({"mutant": f"{wt}{pos}{aa}", "frequency": round(freq, 6),
                                 "count": count, "depth": depth})
                continue
            mutant = f"{wt}{pos}{aa}"
            lines.append(f"{mutant} {freq:.10e}")
            meta_lines.append(
                f"{mutant}\t{pos}\t{wt}\t{aa}\t{count}\t{depth}\t{freq:.10e}\t{parent_label}"
            )

    if len(lines) != len(set(line.split(" ")[0] for line in lines)):
        raise ValueError("Duplicate mutant keys in the frequency file; prescott would take the first silently")

    common.ensure_dir(out_txt.parent)
    out_txt.write_text("\n".join(lines) + ("\n" if lines else ""))
    out_meta.write_text("\n".join(meta_lines) + "\n")

    return {
        "child_lineage": child_label,
        "parent_lineage": parent_label,
        "parent_panel": str(parent_panel_fasta),
        "parent_panel_records": len(sequences),
        "alignment_length": int(aln_len),
        "consensus_length": len(consensus),
        "mapped_ref_sites": len(ref_to_aln),
        "matched_pairs": int(matched_pairs),
        "median_mapped_depth": median_depth,
        "n_mutants": len(lines),
        "n_positions_below_min_depth": n_shallow,
        "n_mutants_below_min_count": n_below_min_count,
        "n_reverted_ancestral_mutants": len(reverted),
        "reverted_ancestral_mutants": reverted,
        "n_parent_reversion_mutants": len(parent_reversions),
        "parent_reversion_mutants": parent_reversions,
        "drop_parent_reversions": bool(use_reversion_filter),
        "frequency_path": str(out_txt),
        "frequency_md5": common.md5_file(out_txt),
        "meta_path": str(out_meta),
        "min_count": int(min_count),
        "min_depth": int(min_depth),
        "freq_max": float(freq_max),
    }


def resolve_frequency_cutoffs(
    mode: str,
    k_values: Sequence[int],
    fixed_cutoff: float,
    median_depth: float,
) -> Dict[str, float]:
    """Fc per k.

    ``depth_scaled`` sets ``Fc = log10(k / median_depth)``, which makes the v2
    penalty exactly ``c * log_N(count)``: zero for a singleton and exactly ``c``
    for a fixed variant, *independent of panel depth*.  That is the only way a
    229-sequence panel and a 27452-sequence panel can share one coefficient grid.
    """
    cutoffs: Dict[str, float] = {}
    for k in k_values:
        if mode == "fixed":
            cutoffs[str(k)] = float(fixed_cutoff)
        else:
            if median_depth <= 0:
                raise ValueError("Cannot depth-scale the frequency cutoff: median parent depth is zero")
            cutoffs[str(k)] = float(np.log10(k / median_depth))
    return cutoffs


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    inputs_dir = common.ensure_dir(args.inputs_dir.resolve())
    structure_dir = common.ensure_dir(inputs_dir / "structure")
    query_dir = common.ensure_dir(inputs_dir / "query")
    msa_dir = common.ensure_dir(inputs_dir / "msa")
    freq_dir = common.ensure_dir(inputs_dir / "frequency")

    guide_rows = common.read_guide_rows(args.guide_path)
    if not guide_rows:
        raise ValueError(f"No usable rows in guide {args.guide_path}")
    all_labels = [row["label"] for row in guide_rows]

    # The parent map is resolved against the FULL guide before --only-lineage is
    # applied, so restricting the run cannot silently reassign a parent.
    parent_map = common.resolve_parent_map(args.parent_map_preset, args.parent_map, all_labels)

    # --- alternate ("sensitivity") parent edges ---------------------------- #
    # The single contested edge in this project is K's parent, so the pipeline has
    # to be able to score BOTH candidates in one run rather than asserting one.
    # These edges are prepared alongside the primary map, each producing its own
    # frequency file; nothing here changes the primary map.
    sensitivity_map: Dict[str, str] = {}
    if args.sensitivity_preset:
        sensitivity_map.update(
            common.sensitivity_edges_between_presets(args.parent_map_preset, args.sensitivity_preset)
        )
    if args.sensitivity_parent_map:
        sensitivity_map.update(common.parse_edge_spec(args.sensitivity_parent_map))
    for child, alt_parent in sorted(sensitivity_map.items()):
        if child not in all_labels:
            raise ValueError(f"--sensitivity-parent-map names an unknown child lineage {child!r}")
        if alt_parent not in all_labels:
            raise ValueError(f"--sensitivity-parent-map names an unknown parent lineage {alt_parent!r}")
        if child not in parent_map:
            raise ValueError(
                f"{child!r} has no primary parent, so an alternate edge is meaningless "
                "(input-only lineages are never scored)"
            )
    # An alternate edge identical to the primary would write a byte-identical
    # second file under a different name and inflate every downstream table with
    # a duplicate model; drop it here rather than downstream.
    redundant = {c for c, p in sensitivity_map.items() if parent_map.get(c) == p}
    for child in redundant:
        print(f"@> Sensitivity edge {child} <- {sensitivity_map[child]} equals the primary map; skipping")
        sensitivity_map.pop(child)

    selected = set(args.only_lineage) if args.only_lineage else set(all_labels)
    unknown = selected - set(all_labels)
    if unknown:
        raise ValueError(f"--only-lineage names labels absent from the guide: {sorted(unknown)}")
    # Parents of selected lineages are needed even when not selected themselves --
    # including alternate parents, whose reference protein the reversion filter and
    # whose panel the frequency counts both come from.
    needed = set(selected)
    for child in list(selected):
        if child in parent_map:
            needed.add(parent_map[child])
        if child in sensitivity_map:
            needed.add(sensitivity_map[child])
    work_rows = [row for row in guide_rows if row["label"] in needed]

    print(f"@> Lineages: {[row['label'] for row in work_rows]}")
    print(f"@> Parent map ({args.parent_map_preset}): {parent_map}")
    if sensitivity_map:
        print(f"@> Alternate (sensitivity) parent edges: {sensitivity_map}")

    # ---- queries ---------------------------------------------------------- #
    queries = build_query_fastas(work_rows, query_dir)
    for label, info in queries.items():
        print(f"@> Query {label}: {info['length']} aa, header >{info['header']}, {info['path']}")

    # ---- structures ------------------------------------------------------- #
    anchor_label = args.msa_anchor if args.msa_anchor in queries else sorted(queries)[0]
    anchor_protein = str(queries[anchor_label]["protein"])

    structures: Dict[str, Dict[str, object]] = {}
    structures["primary"] = prepare_structure(
        args.structure, args.structure_chain, args.structure_offset,
        anchor_protein, structure_dir, args.structure.stem.split("-")[0].split("_")[0],
        args.structure_min_identity,
    )
    print(
        f"@> Structure {args.structure.name}: offset {structures['primary']['offset']:+d}, "
        f"{structures['primary']['n_covered']}/{len(anchor_protein)} covered "
        f"({structures['primary']['coverage_fraction']:.1%}), "
        f"uncovered runs {structures['primary']['uncovered_runs']}"
    )

    if not args.no_extra_structure and args.extra_structure and args.extra_structure.exists():
        structures["extra"] = prepare_structure(
            args.extra_structure, args.structure_chain, args.structure_offset,
            anchor_protein, structure_dir, "model_J241",
            args.structure_min_identity,
        )
        print(
            f"@> Structure {args.extra_structure.name}: offset {structures['extra']['offset']:+d}, "
            f"{structures['extra']['n_covered']}/{len(anchor_protein)} covered"
        )
    common.write_json(structure_dir / "structure_report.json", structures)

    # ---- MSA -------------------------------------------------------------- #
    msa_report = build_deep_msa(
        args.deep_fasta, anchor_label, queries[anchor_label], msa_dir,
        args.mafft_bin, args.mafft_threads, args.mafft_mode, args.msa_nonstandard, args.force,
    )
    print(f"@> Deep MSA: {msa_report['n_rows']} rows x {msa_report['n_columns']} columns "
          f"(raw {msa_report['raw_columns']}, "
          f"{msa_report['n_nonstandard_residues_replaced']} non-standard residues normalised)")

    msa_566 = Path(str(msa_report["msa_566_path"]))
    lineage_msas: Dict[str, Dict[str, object]] = {}
    for label, info in queries.items():
        out_path = msa_dir / f"msa_{info['lineage_key']}.fasta"
        lineage_msas[label] = materialise_lineage_msa(msa_566, label, info, out_path, anchor_label)
        print(f"@> MSA {label}: {out_path.name} "
              f"({lineage_msas[label]['n_query_residues_changed_vs_anchor']} query residues differ from anchor)")

    panel_by_label = {row["label"]: Path(row["diversity_path"]) for row in guide_rows}

    # ---- leakage detection and PURGE -------------------------------------- #
    # MUST happen here: after the per-lineage MSAs exist and BEFORE anything hands
    # one to jet_surrogate.py or run_escott.py. The purge rewrites
    # msa/msa_<key>.fasta in place (keeping msa_<key>_prepurge.fasta), so every
    # downstream consumer picks up the purged alignment without knowing this stage
    # exists -- which is the point: there is no code path that reaches ESCOTT with
    # the unpurged alignment while the purge is on.
    #
    # Targets are the SELECTED lineages that have a parent, i.e. exactly the
    # lineages that will be scored. A parent-only lineage (G.1) is never an
    # evaluation target, so purging the deep set against its panel would remove
    # sequences for no reason and cost depth. Its own MSA is prepared but unused.
    leakage_targets = [label for label in sorted(selected) if label in parent_map]
    leakage_report: Dict[str, object] = {"enabled": False, "status": "SKIPPED"}
    if (args.leakage_check or args.purge_leakage) and leakage_targets:
        leakage_report = leakage_check.run_leakage_stage(
            inputs_dir=inputs_dir,
            panel_by_label={label: panel_by_label[label] for label in
                            sorted(set(leakage_targets) | {parent_map[t] for t in leakage_targets
                                                           if t in parent_map})},
            parent_map=parent_map,
            targets=leakage_targets,
            msa_paths={label: Path(str(lineage_msas[label]["path"]))
                       for label in leakage_targets if label in lineage_msas},
            leakage_check=bool(args.leakage_check),
            purge=bool(args.purge_leakage),
            fail_on_leakage=bool(args.fail_on_leakage),
            min_identity=leakage_check.parse_threshold(
                args.leakage_min_identity, "--leakage-min-identity"),
            max_hamming=leakage_check.parse_threshold(
                args.leakage_max_hamming, "--leakage-max-hamming", int),
            min_coverage=float(args.leakage_min_coverage),
            coverage_basis=args.leakage_coverage_basis,
            max_removed_fraction=float(args.leakage_max_removed_fraction),
            min_depth_after=int(args.leakage_min_depth_after),
            hash_suffix_length=int(args.leakage_hash_suffix_length),
            deep_fasta=args.deep_fasta,
            force=bool(args.force),
            threads=int(args.leakage_threads),
            blast_task=args.blast_task,
            blastp_bin=args.blastp_bin,
            makeblastdb_bin=args.makeblastdb_bin,
        )
        # The purge rewrote the files, so every md5/row count recorded above is now
        # stale. Refreshing them here rather than trusting the earlier values is the
        # difference between a manifest that describes what ESCOTT read and one that
        # describes what mafft produced.
        for label, purge in (leakage_report.get("purges") or {}).items():
            if label in lineage_msas:
                lineage_msas[label].update({
                    "md5": purge["purged_md5"],
                    "n_rows": purge["depth_after"],
                    "n_rows_before_purge": purge["depth_before"],
                    "n_rows_removed_by_purge": purge["n_removed"],
                    "purged": True,
                    "prepurge_path": purge.get("prepurge_path"),
                    "drop_manifest_path": purge.get("drop_manifest_path"),
                })
    elif not leakage_targets:
        print("@> Leakage stage skipped: no evaluation targets selected (parents only)")
    else:
        print("@> Leakage stage DISABLED (--no-leakage-check --no-purge-leakage). "
              "The deep MSA has NOT been screened against the evaluation panels; "
              "any correlation this run reports is unaudited for leakage.")

    # ---- frequency files -------------------------------------------------- #
    k_values = [int(chunk) for chunk in str(args.frequency_cutoff_k).split(",") if chunk.strip()]

    frequency_reports: Dict[str, Dict[str, object]] = {}
    # Which frequency file belongs to which (child, parent) edge.  run_escott.py
    # reads this off inputs_manifest.json rather than reconstructing the filename,
    # so the two stages cannot disagree about which panel a variant was
    # conditioned on -- the exact class of bug the _parent<TOK> suffix exists to
    # make visible in the first place.
    frequency_index: Dict[str, Dict[str, object]] = {}

    for label in sorted(selected):
        if label not in parent_map:
            print(f"@> {label}: no basal panel (input-only lineage); no frequency file written")
            continue

        key = str(queries[label]["lineage_key"])
        # (parent_label, report_key, file_stem, is_primary).  The primary keeps the
        # historical unsuffixed name so existing trees stay readable.
        edges: List[Tuple[str, str, str, bool]] = [
            (parent_map[label], label, f"{key}_parent_frequency", True)
        ]
        if label in sensitivity_map:
            alt = sensitivity_map[label]
            stem = common.alternate_frequency_basename(key, alt)
            edges.append((alt, stem, stem, False))

        for parent, report_key, stem, is_primary in edges:
            panel = panel_by_label[parent]
            if not panel.exists():
                raise FileNotFoundError(f"Parent panel for {label} not found: {panel}")

            report = build_parent_frequency_file(
                child_label=label,
                parent_label=parent,
                child_protein=str(queries[label]["protein"]),
                parent_panel_fasta=panel,
                out_txt=freq_dir / f"{stem}.txt",
                out_meta=freq_dir / f"{stem}_meta.tsv",
                min_count=args.parent_min_count,
                min_depth=args.parent_min_depth,
                freq_max=args.parent_freq_max,
                parent_protein=str(queries[parent]["protein"]),
                drop_parent_reversions=args.drop_parent_reversions,
            )
            report["frequency_cutoffs"] = resolve_frequency_cutoffs(
                args.frequency_cutoff_mode, k_values, args.frequency_cutoff,
                float(report["median_mapped_depth"]),
            )
            report["frequency_cutoff_mode"] = args.frequency_cutoff_mode
            report["is_primary_parent"] = is_primary
            report["parent_token"] = common.variant_parent_token(parent)
            report["report_key"] = report_key
            frequency_reports[report_key] = report
            frequency_index[report_key] = {
                "child_lineage": label,
                "lineage_key": key,
                "parent_lineage": parent,
                "parent_token": common.variant_parent_token(parent),
                "is_primary_parent": is_primary,
                "frequency_path": str(report["frequency_path"]),
                "meta_path": str(report["meta_path"]),
                "median_mapped_depth": report["median_mapped_depth"],
                "n_mutants": report["n_mutants"],
                "drop_parent_reversions": report["drop_parent_reversions"],
            }
            print(
                f"@> Frequency {label} <- {parent}"
                f"{'' if is_primary else ' [SENSITIVITY]'}: {report['n_mutants']} mutants from "
                f"{report['parent_panel_records']} sequences, {report['mapped_ref_sites']} mapped sites, "
                f"median depth {report['median_mapped_depth']:.0f}, "
                f"Fc {report['frequency_cutoffs']}, "
                f"drop_parent_reversions={report['drop_parent_reversions']}, "
                f"{report['n_parent_reversion_mutants']} parent reversions + "
                f"{report['n_reverted_ancestral_mutants']} near-fixed mutants dropped"
            )

    common.write_json(freq_dir / "frequency_report.json", frequency_reports)

    # ---- manifest --------------------------------------------------------- #
    manifest = {
        "generated_by": "scripts/prescott_iav/prepare_inputs.py",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.version.split()[0],
        "args": {key: (str(value) if isinstance(value, Path) else value)
                 for key, value in sorted(vars(args).items())},
        "guide_path": str(args.guide_path),
        "guide_md5": common.md5_file(args.guide_path),
        "parent_map": parent_map,
        "parent_map_preset": args.parent_map_preset,
        # The alternate edges actually prepared, and the file each one wrote.  An
        # empty dict here is the ONLY honest way for a caller to learn that no
        # sensitivity analysis is available -- the driver must not advertise one
        # in run_manifest.json or CAVEATS.md unless this is non-empty.
        "sensitivity_parent_map": sensitivity_map,
        "sensitivity_preset": args.sensitivity_preset,
        "frequency_index": frequency_index,
        # Hoisted out of `args` on purpose. --drop-parent-reversions defaults ON
        # and is the mechanism that actually removes the ancestral-reversion
        # artefact (--parent-freq-max 0.95 does NOT: K's N160S sits at 0.932 and
        # K176I at 0.897, so the threshold misses two of the seven K-defining
        # sites). Flipping it changes every PRESCOTT score, so it must be legible
        # at the top level of the manifest rather than buried in an args dump.
        "drop_parent_reversions": bool(args.drop_parent_reversions),
        "n_parent_reversion_mutants_dropped": {
            report_key: int(rep["n_parent_reversion_mutants"])
            for report_key, rep in frequency_reports.items()
        },
        "parent_freq_max": float(args.parent_freq_max),
        "input_only_lineages": sorted(common.INPUT_ONLY_LINEAGES & set(all_labels)),
        "anchor_lineage": anchor_label,
        "queries": {label: {k: v for k, v in info.items() if k != "protein"}
                    for label, info in queries.items()},
        "structures": structures,
        "msa": msa_report,
        "lineage_msas": lineage_msas,
        # Hoisted to the top level, like drop_parent_reversions and for the same
        # reason: this block decides which sequences ESCOTT was allowed to see. A
        # run whose `leakage.purge` is false is not comparable with one whose is
        # true, and the driver copies these numbers into run_manifest.json and
        # CAVEATS.md so the difference cannot be lost between stages.
        "leakage": leakage_report,
        "frequency": {label: {k: v for k, v in rep.items()
                              if k not in ("reverted_ancestral_mutants", "parent_reversion_mutants")}
                      for label, rep in frequency_reports.items()},
        "seed": args.seed,
    }
    manifest_path = inputs_dir / "inputs_manifest.json"
    common.write_json(manifest_path, manifest)
    print(f"@> Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
