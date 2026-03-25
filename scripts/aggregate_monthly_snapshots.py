#!/usr/bin/env python3
"""
Aggregate monthly SARS-CoV-2 protein snapshots into a unified long table.
Mimics the Flu pipeline's combined long table format for GLM analysis.
"""

import os
import re
import argparse
import csv
import pandas as pd
import numpy as np
import itertools
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Data import CodonTable
from Bio import Align

# SARS-CoV-2 (SC2) TRANSITION MATRIX
# Data source: De Maio (2021) Table 1
SC2_TRANSITIONS = np.array([
    [0.0, 3.19e-8, 2.54e-7, 1.01e-7],
    [1.15e-7, 0.0, 1.80e-8, 2.48e-6],
    [6.11e-7, 9.25e-8, 0.0, 2.42e-6],
    [4.58e-8, 2.14e-7, 2.95e-8, 0.0]
])
BASES = ['A', 'C', 'G', 'T']

def get_mut_prob_matrix(nuc_seq):
    """Calculate mutational accessibility (nucl -> AA) for a given nucleotide sequence."""
    trans = SC2_TRANSITIONS.copy()
    for i in range(4):
        trans[i, i] = 1.0 - np.sum(SC2_TRANSITIONS[i, :])
    
    standard_table = CodonTable.unambiguous_dna_by_id[1]
    genetic_code = standard_table.forward_table.copy()
    for stop_codon in standard_table.stop_codons:
        genetic_code[stop_codon] = '*'
    
    amino_acids = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
    
    # Pre-calculate codon-to-AA probabilities for all 64 codons
    codons = ["".join(c) for c in itertools.product(BASES, repeat=3)]
    codon_to_aa = {} # (codon, aa) -> prob
    for codon_from in codons:
        for aa_to in amino_acids:
            target_codons = [c for c, a in genetic_code.items() if a == aa_to]
            total_p = 0.0
            for codon_to in target_codons:
                p = 1.0
                for k in range(3):
                    idx_from = BASES.index(codon_from[k])
                    idx_to = BASES.index(codon_to[k])
                    p *= trans[idx_from, idx_to]
                total_p += p
            codon_to_aa[(codon_from, aa_to)] = total_p

    mut_prob_rows = []
    nuc_seq_upper = str(nuc_seq).upper().replace("U", "T")
    for i in range(0, len(nuc_seq_upper) - 2, 3):
        codon = nuc_seq_upper[i:i+3]
        if len(codon) < 3 or any(b not in BASES for b in codon):
            mut_prob_rows.append({aa: 0.0 for aa in amino_acids})
            continue
        mut_prob_rows.append({aa: codon_to_aa[(codon, aa)] for aa in amino_acids})
    
    return pd.DataFrame(mut_prob_rows)

def build_reference_to_alignment_column_map(reference_protein, variant_records):
    """
    Aligns each variant to the reference and builds a map:
    ref_idx (0-based) -> list of AAs observed across all variants at that ref position.
    """
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    
    # Store observations: ref_pos (0-based) -> list of AAs
    obs_at_pos = {i: [] for i in range(len(reference_protein))}
    
    for rec in variant_records:
        var_seq = str(rec.seq).upper().replace("*", "")
        alignments = aligner.align(reference_protein, var_seq)
        best = alignments[0]
        
        # alignment.aligned returns tuples of (target_indices, query_indices)
        ref_ranges, var_ranges = best.aligned
        
        # Keep track of which ref positions were covered in this variant
        covered_ref_pos = set()
        for (r_start, r_end), (v_start, v_end) in zip(ref_ranges, var_ranges):
            for r_idx, v_idx in zip(range(r_start, r_end), range(v_start, v_end)):
                aa = var_seq[v_idx]
                obs_at_pos[r_idx].append(aa)
                covered_ref_pos.add(r_idx)
                
        # For positions NOT covered (deletions/gaps in variant), we could add '-'
        # for i in range(len(reference_protein)):
        #     if i not in covered_ref_pos:
        #         obs_at_pos[i].append('-')
                
    return obs_at_pos

def main():
    parser = argparse.ArgumentParser(description="Aggregate monthly SC2 snapshots.")
    parser.add_argument("--ref-nuc", type=Path, required=True, help="Path to reference nucleotide FASTA.")
    parser.add_argument("--plm-matrix", type=Path, help="Optional: Path to PLM probability matrix CSV.")
    parser.add_argument("--snapshots-dir", type=Path, help="Directory containing snapshots to automatically discover.")
    parser.add_argument("--snapshots-csv", type=Path, help="CSV file mapping month/label to FASTA path (columns: month, fasta).")
    parser.add_argument("--output", type=Path, required=True, help="Path to output CSV.")
    args = parser.parse_args()

    if not args.snapshots_dir and not args.snapshots_csv:
        print("Error: Either --snapshots-dir or --snapshots-csv must be provided.")
        sys.exit(1)

    # 1. Load Reference and calculate mut_prob
    print(f"Loading reference: {args.ref_nuc}")
    ref_records = list(SeqIO.parse(args.ref_nuc, "fasta"))
    if not ref_records:
        print(f"Error: No sequences in {args.ref_nuc}")
        sys.exit(1)
    ref_record = ref_records[0]
    ref_nuc = str(ref_record.seq)
    ref_prot = str(ref_record.seq.translate(to_stop=True))
    mut_prob_df = get_mut_prob_matrix(ref_nuc)

    # 2. Load PLM Matrix (optional)
    plm_score_map = {} # (pos, aa_target) -> score
    plm_seq_len = 0
    
    if args.plm_matrix and os.path.exists(args.plm_matrix):
        print(f"Loading PLM matrix: {args.plm_matrix}")
        plm_raw = pd.read_csv(args.plm_matrix, header=None)
        # Row 0: ['sequence', 'M', 'F', 'V'...]
        plm_seq = plm_raw.iloc[0, 1:].tolist()
        plm_aas = plm_raw.iloc[1:, 0].tolist()
        plm_values = plm_raw.iloc[1:, 1:].values.astype(float)
        plm_seq_len = len(plm_seq)
        
        for col_idx, ref_aa_at_pos in enumerate(plm_seq):
            pos = col_idx + 1
            for row_idx, aa_target in enumerate(plm_aas):
                plm_score_map[(pos, aa_target)] = plm_values[row_idx, col_idx]
    else:
        print("Warning: No PLM matrix provided or found. 'plm_prob' column will be NaN.")
        plm_seq_len = len(ref_prot)

    # 2.5 Coordinate mapping for PLM if trimmed
    # (Assuming PLM matrix was generated on a prefix of ref_prot)
    # If the PLM matrix header 'sequence' (row 0) exists, we should use it to align.
    # In this script's current implementation, it assumes 1-based indexing 'pos'.

    # 3. Process Monthly Snapshots
    month_fasta_map = {} # label -> path

    if args.snapshots_csv:
        print(f"Loading snapshot map from {args.snapshots_csv}")
        with open(args.snapshots_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row.get('month') or row.get('label')
                path = row.get('fasta') or row.get('path')
                if label and path:
                    month_fasta_map[label] = Path(path)
    
    if args.snapshots_dir:
        print(f"Discovering snapshots in {args.snapshots_dir}")
        # Search for .fa, .fasta, .fna (standard or in subdirectories)
        for ext in ["*.fa", "*.fasta", "*.fna"]:
            for path in args.snapshots_dir.rglob(ext):
                # Avoid metadata CSVs or other files
                if path.suffix == ".csv": continue
                
                # Extract month (YYYY-MM) from filename if not already in map
                match = re.search(r'(\d{4}-\d{2})', path.name)
                if match:
                    label = match.group(1)
                    if label not in month_fasta_map:
                        month_fasta_map[label] = path
                else:
                    print(f"  Warning: Could not extract YYYY-MM from {path.name}. Skipping.")

    if not month_fasta_map:
        print(f"Error: No monthly snapshots found.")
        sys.exit(1)

    all_data = []
    amino_acids = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

    for month_label, fasta_path in sorted(month_fasta_map.items()):
        print(f"Processing {month_label} ({fasta_path.name})...")
        
        month_records = list(SeqIO.parse(fasta_path, "fasta"))
        if not month_records:
            print(f"  Warning: No records in {fasta_path}")
            continue
        
        # Ensure month records are protein for alignment
        for rec in month_records:
             if any(b in "T" for b in str(rec.seq).upper()[:100]): # simple heuristic
                  rec.seq = rec.seq.translate(to_stop=True)

        depth = len(month_records)
        obs_at_pos = build_reference_to_alignment_column_map(ref_prot, month_records)
        
        for i in range(len(ref_prot)):
            pos = i + 1
            ref_aa = ref_prot[i]
            
            # Count observations at this reference position
            counts = {aa: 0 for aa in amino_acids}
            for aa_obs in obs_at_pos[i]:
                if aa_obs in counts:
                    counts[aa_obs] += 1
            
            for aa in amino_acids:
                if aa == ref_aa:
                    continue # Skip synonymous
                
                obs_count = counts.get(aa, 0)
                obs_freq = obs_count / depth
                
                plm_prob = plm_score_map.get((pos, aa), 0.0)
                mut_prob = mut_prob_df.loc[i, aa] if i < len(mut_prob_df) else 0.0
                
                all_data.append({
                    'lineage': month_label,
                    'position': pos,
                    'ref_aa': ref_aa,
                    'aa': aa,
                    'plm_prob': plm_prob,
                    'mut_prob': mut_prob,
                    'obs_freq': obs_freq,
                    'depth': depth
                })

    # 4. Save
    df = pd.DataFrame(all_data)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Successfully aggregated {len(df)} mutation rows.")
    print(f"Output saved to: {args.output}")

if __name__ == "__main__":
    main()
