"""Shared plumbing for the ordered-mutation ("epistasis order") scans.

Both `epistasis_order_scan.py` (PLM probability shifts) and `plant_order_scan.py`
(PLANT 3D embedding trajectories) walk the same object: the mutational hypercube
between a start lineage and an end lineage.

For n mutations that hypercube has

  * ``2**n`` distinct genotypes (every subset of the n mutations), and
  * ``n!`` orderings, i.e. monotone paths from the empty set to the full set.

Every ordering is just a chain of n+1 genotypes through that hypercube, so both
scans embed/score the ``2**n`` genotypes once and then reconstruct all ``n!``
paths from the cached per-genotype results. For the J -> J.2.4 case (n = 4) that
is 16 model evaluations instead of 24 x 5 = 120, and the numbers are identical
because the model is deterministic and depends only on the genotype, never on
the route taken to reach it.

Nothing in this module imports torch, esm or Functions_HuggingFace at module
level, so it stays importable from any of the project conda environments.
"""

from __future__ import annotations

import argparse
import itertools
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_FASTA = REPO_ROOT / "Sequences/huH3N2_HA_CDS.translated.fas"
DEFAULT_H3_REFERENCE = REPO_ROOT / "Sequences/H3N2_canonical.fa"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "Results/JtoJ.2.4_scan"

# Characters that never count as a substitution: they mark missing/ambiguous
# data, and both input sequences here end in a trailing X.
AMBIGUOUS_CHARS = frozenset("X-.*BZJUO")

# n! grows fast enough that enumerating every ordering stops being a "quick
# scan" well before it stops being representable. Past this, --max-orders is
# required so the user makes the sampling decision explicitly.
MAX_MUTATIONS_FOR_FULL_ENUMERATION = 8


@dataclass(frozen=True, order=True)
class Mutation:
    """A single amino-acid substitution in 1-based coordinates of the input FASTA."""

    pos: int
    wt: str
    alt: str

    @property
    def name(self) -> str:
        return f"{self.wt}{self.pos}{self.alt}"

    def label(self, h3_map: Optional[Dict[int, str]] = None) -> str:
        """Name in canonical H3 numbering when a map is available, else raw numbering."""
        if not h3_map:
            return self.name
        canonical = h3_map.get(self.pos)
        if canonical is None:
            return self.name
        if canonical.startswith("HA2:"):
            return f"HA2:{self.wt}{canonical.split(':', 1)[1]}{self.alt}"
        return f"{self.wt}{canonical}{self.alt}"

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.name


###############################################################################
# FASTA handling
###############################################################################
def read_fasta(path: Path) -> "Dict[str, str]":
    """Read a FASTA into an insertion-ordered {header: sequence} dict.

    Deliberately hand-rolled rather than Bio.SeqIO: the headers here are
    pipe-delimited GISAID strings that must survive verbatim, and this keeps the
    module free of a Biopython import so it loads in every environment.
    """
    sequences: Dict[str, str] = {}
    header: Optional[str] = None
    chunks: List[str] = []

    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    sequences[header] = "".join(chunks)
                header = line[1:].strip()
                if header in sequences:
                    raise ValueError(f"Duplicate FASTA header in {path}: {header!r}")
                chunks = []
            else:
                chunks.append(line.upper())
    if header is not None:
        sequences[header] = "".join(chunks)

    if not sequences:
        raise ValueError(f"No sequences parsed from {path}")
    return sequences


def resolve_record(
    sequences: Dict[str, str],
    selector: Optional[str],
    index: Optional[int],
    role: str,
) -> Tuple[str, str]:
    """Pick one record by lineage field, header substring, or positional index.

    Resolution order, first match wins:
      1. exact match on the full header;
      2. exact match on the final pipe-delimited field (the lineage), which is
         what makes ``--end-id J.2.4`` select J.2.4 and *not* J.2.4.1;
      3. unique substring match anywhere in the header;
      4. ``index`` into the file order, when no selector was given.
    """
    headers = list(sequences)

    if selector:
        if selector in sequences:
            return selector, sequences[selector]

        lineage_hits = [h for h in headers if h.rsplit("|", 1)[-1] == selector]
        if len(lineage_hits) == 1:
            return lineage_hits[0], sequences[lineage_hits[0]]
        if len(lineage_hits) > 1:
            raise ValueError(
                f"{role} selector {selector!r} matches {len(lineage_hits)} records by lineage: "
                f"{lineage_hits}. Pass a full header instead."
            )

        substring_hits = [h for h in headers if selector in h]
        if len(substring_hits) == 1:
            return substring_hits[0], sequences[substring_hits[0]]
        if not substring_hits:
            raise ValueError(f"{role} selector {selector!r} matched no record. Available: {headers}")
        raise ValueError(
            f"{role} selector {selector!r} is ambiguous, matching {substring_hits}. "
            "Pass a full header instead."
        )

    if index is None:
        raise ValueError(f"No {role} selector and no {role} index supplied.")
    if not -len(headers) <= index < len(headers):
        raise ValueError(f"{role} index {index} out of range for {len(headers)} records.")
    header = headers[index]
    return header, sequences[header]


###############################################################################
# Mutations and the hypercube
###############################################################################
def mutations_between(
    start_seq: str,
    end_seq: str,
    ambiguous_chars: Iterable[str] = AMBIGUOUS_CHARS,
) -> List[Mutation]:
    """Substitutions turning `start_seq` into `end_seq`, skipping ambiguous sites.

    The two sequences must already be the same length (these are pre-aligned
    translated CDS records); a length mismatch is an error rather than something
    to silently truncate, because a frame or trimming problem upstream would
    otherwise surface as a plausible-looking but wrong mutation list.
    """
    if len(start_seq) != len(end_seq):
        raise ValueError(
            f"Sequences differ in length ({len(start_seq)} vs {len(end_seq)}); "
            "align them before running the scan."
        )

    ambiguous = set(ambiguous_chars)
    mutations = []
    for offset, (wt, alt) in enumerate(zip(start_seq, end_seq)):
        if wt == alt or wt in ambiguous or alt in ambiguous:
            continue
        mutations.append(Mutation(pos=offset + 1, wt=wt, alt=alt))
    return mutations


def apply_mutations(sequence: str, mutations: Sequence[Mutation]) -> str:
    """Apply substitutions to `sequence`, checking each wild-type residue first."""
    chars = list(sequence)
    for mutation in mutations:
        index = mutation.pos - 1
        if not 0 <= index < len(chars):
            raise ValueError(f"{mutation.name} lies outside a sequence of length {len(chars)}")
        if chars[index] != mutation.wt:
            raise ValueError(
                f"{mutation.name}: expected {mutation.wt} at position {mutation.pos}, "
                f"found {chars[index]}"
            )
        chars[index] = mutation.alt
    return "".join(chars)


def all_subsets(
    mutations: Sequence[Mutation],
    max_size: Optional[int] = None,
) -> List[Tuple[Mutation, ...]]:
    """Every subset of `mutations`, ordered by size then by genomic position.

    `max_size` caps the number of mutations in a background. `max_size=1` yields
    the root plus every single mutant — the whole basis for the pairwise
    epistasis matrix, at n+1 genotypes instead of 2**n.
    """
    ordered = sorted(mutations)
    limit = len(ordered) if max_size is None else min(max_size, len(ordered))
    subsets: List[Tuple[Mutation, ...]] = []
    for size in range(limit + 1):
        subsets.extend(itertools.combinations(ordered, size))
    return subsets


def all_orderings(
    mutations: Sequence[Mutation],
    max_orders: Optional[int] = None,
    seed: int = 0,
) -> List[Tuple[Mutation, ...]]:
    """Every ordering of `mutations`, or a reproducible random sample of them.

    Sampling is over the permutation space itself rather than a truncation of
    `itertools.permutations`, whose lexicographic order would otherwise bias the
    sample towards paths that start with the earliest mutation.
    """
    ordered = sorted(mutations)
    n = len(ordered)

    if max_orders is None:
        if n > MAX_MUTATIONS_FOR_FULL_ENUMERATION:
            raise ValueError(
                f"{n} mutations means {_factorial(n):,} orderings. Pass --max-orders to sample a "
                f"subset for the per-path tables, or --max-mutations to restrict the mutation set. "
                f"The best and worst ordering are still reported exactly over all {_factorial(n):,} "
                f"via the hypercube DP, and the full epistasis table covers every one of the "
                f"{2 ** n} genotypes."
            )
        return list(itertools.permutations(ordered))

    total = _factorial(n)
    if max_orders >= total:
        return list(itertools.permutations(ordered))

    rng = random.Random(seed)
    seen = set()
    sampled: List[Tuple[Mutation, ...]] = []
    while len(sampled) < max_orders:
        candidate = tuple(rng.sample(ordered, n))
        if candidate in seen:
            continue
        seen.add(candidate)
        sampled.append(candidate)
    return sampled


def subset_to_mask(mutations: Sequence[Mutation], subset: Sequence[Mutation]) -> int:
    """Bitmask of `subset` over the sorted `mutations` list."""
    index = {m: i for i, m in enumerate(sorted(mutations))}
    mask = 0
    for mutation in subset:
        mask |= 1 << index[mutation]
    return mask


def mask_to_subset(mutations: Sequence[Mutation], mask: int) -> Tuple[Mutation, ...]:
    """Inverse of `subset_to_mask`."""
    ordered = sorted(mutations)
    return tuple(m for i, m in enumerate(ordered) if mask & (1 << i))


def hypercube_path_extremes(
    mutations: Sequence[Mutation],
    step_score: "callable",
) -> Dict[str, object]:
    """Exact best- and worst-scoring ordering, without enumerating n! of them.

    An ordering's total score is a sum of per-step terms, and each term depends
    only on the set of mutations already fixed — never on the order they fixed
    in. So the extremes are a longest/shortest path on the subset lattice, which
    dynamic programming solves in O(2**n · n). At n = 11 that is ~22k operations
    instead of 39,916,800 orderings; the answer is exact over all of them, not a
    sample.

    ``step_score(background_mask, mutation_index) -> float`` scores adding one
    mutation to a background. Higher is better.
    """
    ordered = sorted(mutations)
    n = len(ordered)
    size = 1 << n

    best = [0.0] * size
    worst = [0.0] * size
    best_from = [-1] * size
    worst_from = [-1] * size

    # Clearing a bit always yields a smaller integer, so ascending mask order is
    # already a valid topological order for the lattice.
    for mask in range(1, size):
        best_value = float("-inf")
        worst_value = float("inf")
        remaining = mask
        while remaining:
            bit = remaining & -remaining
            index = bit.bit_length() - 1
            previous = mask ^ bit
            score = step_score(previous, index)
            if best[previous] + score > best_value:
                best_value = best[previous] + score
                best_from[mask] = index
            if worst[previous] + score < worst_value:
                worst_value = worst[previous] + score
                worst_from[mask] = index
            remaining ^= bit
        best[mask] = best_value
        worst[mask] = worst_value

    def traceback(parents: List[int]) -> Tuple[Mutation, ...]:
        mask = size - 1
        reversed_order = []
        while mask:
            index = parents[mask]
            reversed_order.append(ordered[index])
            mask ^= 1 << index
        return tuple(reversed(reversed_order))

    full = size - 1
    return {
        "best_ordering": traceback(best_from),
        "best_score": best[full],
        "worst_ordering": traceback(worst_from),
        "worst_score": worst[full],
        "n_orderings_covered": _factorial(n),
    }


def _factorial(n: int) -> int:
    total = 1
    for k in range(2, n + 1):
        total *= k
    return total


def genotype_id(subset: Sequence[Mutation], h3_map: Optional[Dict[int, str]] = None) -> str:
    """Stable name for a hypercube node: 'root' or '+'-joined mutations by position."""
    if not subset:
        return "root"
    return "+".join(m.label(h3_map) for m in sorted(subset))


def order_id(ordering: Sequence[Mutation], h3_map: Optional[Dict[int, str]] = None) -> str:
    """Stable name for a path: mutations joined by '>' in the order they fix."""
    return ">".join(m.label(h3_map) for m in ordering)


def path_genotypes(ordering: Sequence[Mutation]) -> List[Tuple[Mutation, ...]]:
    """The n+1 hypercube nodes an ordering passes through, root first."""
    return [tuple(sorted(ordering[:k])) for k in range(len(ordering) + 1)]


###############################################################################
# H3 canonical numbering
###############################################################################
def build_h3_map(
    query_seq: str,
    reference_path: Optional[Path],
    signal_peptide_length: int = 16,
    ha2_start: Optional[int] = 330,
) -> Dict[int, str]:
    """Map 1-based query positions to canonical H3 labels.

    Delegates to ``Functions_HuggingFace.create_h3_numbering_map`` (0-based) and
    re-keys to 1-based so it lines up with `Mutation.pos`. Falls back to a plain
    signal-peptide offset when the canonical reference is unavailable, so the
    scan still runs (with clearly-labelled approximate numbering) on a machine
    that lacks it.
    """
    if reference_path is not None and Path(reference_path).exists():
        try:
            sys.path.insert(0, str(REPO_ROOT))
            from Functions_HuggingFace import create_h3_numbering_map  # noqa: WPS433

            reference_seq = next(iter(read_fasta(Path(reference_path)).values()))
            zero_based = create_h3_numbering_map(
                query_seq,
                reference_seq,
                signal_peptide_length=signal_peptide_length,
                HA2_start=ha2_start,
            )
            return {pos + 1: label for pos, label in zero_based.items()}
        except Exception as error:  # pragma: no cover - environment dependent
            print(
                f"[warning] Could not build canonical H3 map from {reference_path}: {error}. "
                "Falling back to a fixed signal-peptide offset.",
                file=sys.stderr,
            )

    return {
        pos: str(pos - signal_peptide_length)
        for pos in range(1, len(query_seq) + 1)
        if pos > signal_peptide_length
    }


###############################################################################
# Argument plumbing shared by both scans
###############################################################################
def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach the input/lineage-selection/hypercube options both scans share."""
    sequences = parser.add_argument_group("input sequences")
    sequences.add_argument(
        "--fasta",
        type=Path,
        default=DEFAULT_FASTA,
        help="Aligned translated-CDS FASTA holding both lineages (default: %(default)s)",
    )
    sequences.add_argument(
        "--start-id",
        default=None,
        help="Start lineage: full header, exact lineage field (e.g. 'J'), or unique substring.",
    )
    sequences.add_argument(
        "--end-id",
        default="J.2.4",
        help="End lineage, resolved the same way as --start-id (default: %(default)s). "
             "An exact lineage match wins, so 'J.2.4' will not pick up J.2.4.1.",
    )
    sequences.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="0-based index into the FASTA, used only when --start-id is omitted (default: %(default)s)",
    )
    sequences.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="0-based index into the FASTA, used only when --end-id is omitted.",
    )

    numbering = parser.add_argument_group("H3 numbering")
    numbering.add_argument(
        "--h3-reference",
        type=Path,
        default=DEFAULT_H3_REFERENCE,
        help="Canonical H3 reference FASTA for site labelling (default: %(default)s)",
    )
    numbering.add_argument(
        "--signal-peptide-length",
        type=int,
        default=16,
        help="Signal peptide length used to anchor H3 numbering (default: %(default)s)",
    )
    numbering.add_argument(
        "--ha2-start",
        type=int,
        default=330,
        help="Canonical H3 position where HA2 begins; use -1 to disable HA2 labelling "
             "(default: %(default)s)",
    )

    cube = parser.add_argument_group("mutation set")
    cube.add_argument(
        "--mutations",
        nargs="+",
        default=None,
        help="Override the derived mutation set, in raw FASTA numbering (e.g. N138D T151K). "
             "Each is validated against the start sequence.",
    )
    cube.add_argument(
        "--max-mutations",
        type=int,
        default=None,
        help="Keep only the first N mutations by position. Useful for a fast smoke test.",
    )
    cube.add_argument(
        "--max-orders",
        type=int,
        default=None,
        help="Sample this many orderings instead of enumerating all n! of them.",
    )
    cube.add_argument(
        "--max-background-size",
        type=int,
        default=None,
        help="Only score backgrounds with at most this many mutations already fixed. "
             "1 gives the root plus every single mutant, which is exactly what the pairwise "
             "epistasis matrix needs and keeps a large mutation set cheap (n+1 genotypes "
             "instead of 2**n). Ordering/trajectory tables need the full hypercube and are "
             "skipped when this is set.",
    )
    cube.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for --max-orders sampling (default: %(default)s)",
    )
    return parser


def parse_mutation_string(text: str) -> Mutation:
    """Parse 'N138D' into a Mutation."""
    text = text.strip()
    if len(text) < 3 or not text[1:-1].isdigit():
        raise ValueError(f"Cannot parse mutation {text!r}; expected the form N138D.")
    return Mutation(pos=int(text[1:-1]), wt=text[0].upper(), alt=text[-1].upper())


def resolve_mutation_set(args: argparse.Namespace) -> Tuple[str, str, str, str, List[Mutation]]:
    """Read the FASTA, resolve both lineages, and return the mutation set to scan.

    Returns ``(start_header, start_seq, end_header, end_seq, mutations)``.
    """
    sequences = read_fasta(args.fasta)
    # An explicit index wins over the selector's default value, so
    # `--end-index 3` does what it says instead of being silently overridden by
    # the default `--end-id J.2.4`.
    start_selector = None if args.start_index is not None and args.start_id is None else args.start_id
    end_selector = None if args.end_index is not None else args.end_id

    start_header, start_seq = resolve_record(sequences, start_selector, args.start_index, "start")
    end_header, end_seq = resolve_record(sequences, end_selector, args.end_index, "end")

    if start_header == end_header:
        raise ValueError(f"Start and end resolved to the same record: {start_header}")

    if args.mutations:
        mutations = [parse_mutation_string(text) for text in args.mutations]
        for mutation in mutations:
            index = mutation.pos - 1
            if not 0 <= index < len(start_seq):
                raise ValueError(f"{mutation.name} lies outside the start sequence.")
            if start_seq[index] != mutation.wt:
                raise ValueError(
                    f"{mutation.name}: start sequence has {start_seq[index]} at position "
                    f"{mutation.pos}, not {mutation.wt}."
                )
    else:
        mutations = mutations_between(start_seq, end_seq)

    if args.max_mutations is not None:
        mutations = sorted(mutations)[: args.max_mutations]

    if not mutations:
        raise ValueError(f"No unambiguous substitutions between {start_header} and {end_header}.")

    return start_header, start_seq, end_header, end_seq, sorted(mutations)


def write_fasta(path: Path, records: Sequence[Tuple[str, str]], line_width: int = 60) -> None:
    """Write (header, sequence) pairs as a wrapped FASTA."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for header, sequence in records:
            handle.write(f">{header}\n")
            for start in range(0, len(sequence), line_width):
                handle.write(sequence[start : start + line_width] + "\n")


def describe_mutation_set(
    mutations: Sequence[Mutation],
    h3_map: Dict[int, str],
    start_header: str,
    end_header: str,
) -> str:
    """Human-readable summary printed at the top of every run."""
    n = len(mutations)
    factorial = 1
    for k in range(2, n + 1):
        factorial *= k
    lines = [
        f"Start : {start_header}",
        f"End   : {end_header}",
        f"Mutations ({n}): " + ", ".join(f"{m.name} [{m.label(h3_map)}]" for m in mutations),
        f"Hypercube: {2 ** n} genotypes, {factorial} orderings",
    ]
    return "\n".join(lines)
