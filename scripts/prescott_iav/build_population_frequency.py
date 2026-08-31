#!/usr/bin/env python3
"""Build a PRESCOTT population-frequency file from a DATE-WINDOWED sequence set.

WHY THIS EXISTS
---------------
The pipeline's default population prior for a lineage is its *parent lineage panel*. For J
that is G.1: 229 sequences, yielding 61 mutant records, of which exactly ONE of the eleven
J->K substitutions appears. The population term therefore has almost nothing to say about the
substitutions we actually care about, and PRESCOTT collapses onto ESCOTT.

This module replaces that panel with a real population: every human H3N2 HA (segment 4)
record in a collection-date window, which for 2021-2023 is 14,058 sequences -- 61x the depth,
and selected by *date* rather than by lineage assignment, so nothing is excluded for being
hard to classify.

THE FILTERS, AND WHY THEY ARE SEPARABLE
---------------------------------------
``build_parent_frequency_file`` applies two suppressions, and they are not the same thing:

``freq_max``
    a blunt threshold: drop any mutant at or above frequency X. Measured on real data this
    is the wrong instrument -- it lets through reversions at 0.93 while deleting genuine
    standing variation that happens to be common.

``drop_parent_reversions``
    surgical and threshold-free: drop a mutant only when it IS the population's own residue
    at a site where the population and the query genuinely differ. This removes the
    "reverting the lineage-defining substitution is maximally tolerated" artefact and
    nothing else.

That distinction matters much more with a date window than with a parent panel. A 2021-2023
population is ANCESTRAL to J: at every position where J carries a residue that arose after
2023, the pre-J population shows the ancestral residue at frequency ~1.0. Left in, those
positions tell PRESCOTT that undoing J is the best available move, at a strength that swamps
the low-frequency standing variation the window was built to expose.

So this script emits several variants of the same population rather than one, and the
comparison between them is the point:

    unfiltered   freq_max=1.0, no reversion drop   -- the population exactly as observed
    reversions   freq_max=1.0, reversion drop      -- standing variation, artefact removed
    strict       freq_max=0.95, reversion drop     -- the pipeline's default suppression
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from prescott_iav import common  # noqa: E402
from prescott_iav.prepare_inputs import build_parent_frequency_file  # noqa: E402

DEFAULT_MAFFT = Path("/home3/oml4h/miniconda3/envs/PRESCOTT/bin/mafft")

# Variants emitted by default. (name, freq_max, drop_parent_reversions)
DEFAULT_VARIANTS: Tuple[Tuple[str, float, bool], ...] = (
    ("unfiltered", 1.0, False),
    ("reversions", 1.0, True),
    ("strict", 0.95, True),
)


def ungapped(sequence: str) -> str:
    return sequence.replace("-", "").replace(".", "")


def combine_nucleotide(sources: Sequence[Path], out_fasta: Path) -> int:
    """One de-duplicated, ungapped nucleotide FASTA from every source.

    De-duplication is by accession, not by sequence: identical sequences from different
    isolates are real population observations and collapsing them would destroy exactly the
    frequency signal this file exists to carry.
    """
    seen: set = set()
    written = 0
    with out_fasta.open("w", encoding="utf-8") as handle:
        for source in sources:
            if not source.exists():
                print(f"@> skipping missing source {source}")
                continue
            for header, sequence in common.read_fasta(source):
                accession = header.split()[0].split(".")[0].strip()
                if accession in seen:
                    continue
                seen.add(accession)
                bases = ungapped(sequence).upper()
                if len(bases) < 1000:          # truncated record; cannot carry HA1
                    continue
                handle.write(f">{accession}\n{bases}\n")
                written += 1
    return written


def mafft_keeplength(reference_fasta: Path, population_fasta: Path, out_fasta: Path,
                     mafft_bin: Path, threads: int) -> None:
    """Force every population sequence into the reference CDS frame.

    ``--keeplength`` deletes columns the reference does not have, so the output is exactly
    the reference's own length and translation is a plain codon walk. Without it the added
    sequences bring insertions and the reading frame drifts per record.
    """
    command = [str(mafft_bin), "--keeplength", "--add", str(population_fasta),
               "--thread", str(threads), "--quiet", str(reference_fasta)]
    print(f"@> {' '.join(command)}", flush=True)
    with out_fasta.open("w", encoding="utf-8") as handle:
        subprocess.run(command, stdout=handle, check=True)


def translate_alignment(aligned_nt: Path, out_protein: Path, drop_first: bool,
                        expected_aa: int) -> int:
    """Codon-walk a frame-locked nucleotide alignment into protein.

    A codon that is entirely gap becomes '-', anything ambiguous or partial becomes 'X'.
    Both are alphabet members the downstream column counter already ignores, so a ragged
    record contributes at the positions it covers and nowhere else.
    """
    written = 0
    with out_protein.open("w", encoding="utf-8") as handle:
        for index, (header, sequence) in enumerate(common.read_fasta(aligned_nt)):
            if drop_first and index == 0:
                continue
            residues: List[str] = []
            for start in range(0, len(sequence) - 2, 3):
                codon = sequence[start:start + 3].upper()
                if codon == "---":
                    residues.append("-")
                elif "-" in codon or "N" in codon or len(codon) < 3:
                    residues.append("X")
                else:
                    residues.append(common.CODON_TABLE.get(codon, "X")
                                    if hasattr(common, "CODON_TABLE")
                                    else _translate_codon(codon))
            protein = "".join(residues)[:expected_aa]
            if len(protein) < expected_aa:
                protein = protein + "-" * (expected_aa - len(protein))
            handle.write(f">{header.split()[0]}\n{protein}\n")
            written += 1
    return written


_CODONS = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L", "CTA": "L",
    "CTG": "L", "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M", "GTT": "V", "GTC": "V",
    "GTA": "V", "GTG": "V", "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S", "CCT": "P",
    "CCC": "P", "CCA": "P", "CCG": "P", "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A", "TAT": "Y", "TAC": "Y", "TAA": "*",
    "TAG": "*", "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q", "AAT": "N", "AAC": "N",
    "AAA": "K", "AAG": "K", "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E", "TGT": "C",
    "TGC": "C", "TGA": "*", "TGG": "W", "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R", "GGT": "G", "GGC": "G", "GGA": "G",
    "GGG": "G",
}


def _translate_codon(codon: str) -> str:
    return _CODONS.get(codon, "X")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--population-dir", type=Path, required=True,
                        help="working directory holding the assembled population")
    parser.add_argument("--source", type=Path, action="append", required=True,
                        help="repeatable; nucleotide FASTA contributing to the population")
    parser.add_argument("--target-lineage", default="J_int",
                        help="the lineage whose reference frame the frequencies are relative to")
    parser.add_argument("--lineage-dir", type=Path,
                        default=REPO_ROOT / "Sequences" / "IAV_lineage_files")
    parser.add_argument("--population-label", default="pop2021_2023")
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--min-depth", type=int, default=50)
    parser.add_argument("--mafft-bin", type=Path, default=DEFAULT_MAFFT)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--variant", action="append", default=None,
                        metavar="NAME:FREQMAX:DROPREV",
                        help="repeatable, e.g. unfiltered:1.0:false; default emits all three")
    parser.add_argument("--force", action="store_true",
                        help="rebuild the alignment even if it already exists")
    return parser


def parse_variants(specs: Optional[Sequence[str]]) -> List[Tuple[str, float, bool]]:
    if not specs:
        return list(DEFAULT_VARIANTS)
    parsed: List[Tuple[str, float, bool]] = []
    for spec in specs:
        name, freq_max, drop = spec.split(":")
        parsed.append((name, float(freq_max), drop.strip().lower() in ("1", "true", "yes")))
    return parsed


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    work = common.ensure_dir(args.population_dir)

    reference = common.load_reference_cds(
        args.lineage_dir / f"{args.target_lineage}.nt.fa", args.target_lineage)
    protein = reference["protein"]
    reference_fasta = work / f"{args.target_lineage}_cds.fasta"
    common.write_fasta(reference_fasta, [(args.target_lineage, reference["nucleotide"])])

    combined = work / "population_nt.fasta"
    aligned_nt = work / "population_nt_aligned.fasta"
    aligned_aa = work / "population_protein_aligned.fasta"

    if args.force or not combined.exists():
        count = combine_nucleotide(args.source, combined)
        print(f"@> combined {count:,} nucleotide records -> {combined}", flush=True)
    if args.force or not aligned_nt.exists():
        mafft_keeplength(reference_fasta, combined, aligned_nt, args.mafft_bin, args.threads)
    if args.force or not aligned_aa.exists():
        n = translate_alignment(aligned_nt, aligned_aa, drop_first=True,
                                expected_aa=len(protein))
        print(f"@> translated {n:,} records -> {aligned_aa}", flush=True)

    reports: Dict[str, object] = {}
    for name, freq_max, drop_reversions in parse_variants(args.variant):
        out_txt = work / f"{args.target_lineage}_{args.population_label}_{name}_frequency.txt"
        out_meta = work / f"{args.target_lineage}_{args.population_label}_{name}_frequency_meta.tsv"
        report = build_parent_frequency_file(
            child_label=args.target_lineage,
            parent_label=args.population_label,
            child_protein=protein,
            parent_panel_fasta=aligned_aa,
            out_txt=out_txt,
            out_meta=out_meta,
            min_count=args.min_count,
            min_depth=args.min_depth,
            freq_max=freq_max,
            # The population's own consensus is what a "reversion" is measured against.
            parent_protein=None if not drop_reversions else consensus_protein(aligned_aa, protein),
            drop_parent_reversions=drop_reversions,
        )
        reports[name] = report
        print(f"@> [{name}] freq_max={freq_max} drop_reversions={drop_reversions} -> "
              f"{report.get('n_written', '?')} records, median depth "
              f"{report.get('median_depth', '?')}", flush=True)

    (work / "frequency_reports.json").write_text(json.dumps(reports, indent=2, default=str),
                                                 encoding="utf-8")
    print(json.dumps(reports, indent=2, default=str))
    return 0


def consensus_protein(aligned_aa: Path, child_protein: str) -> str:
    """The population's own consensus residue at each of the child's positions.

    This is what ``drop_parent_reversions`` compares against. Using the population consensus
    rather than a named parent lineage reference is the right call here: the population is a
    date window, not a clade, so it has no single reference sequence -- but it does have a
    well-defined majority residue at every site.
    """
    sequences = common.read_fasta_sequences(aligned_aa)
    ref_to_aln, _, _, consensus = common.map_reference_to_alignment_columns(
        child_protein, sequences)
    residues = []
    for position in range(1, len(child_protein) + 1):
        column = ref_to_aln.get(position)
        residues.append(consensus[column - 1] if column else child_protein[position - 1])
    return "".join(residues)


if __name__ == "__main__":
    raise SystemExit(main())
