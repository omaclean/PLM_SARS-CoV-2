#!/usr/bin/env python3
import sys
import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Add REPO_ROOT to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Functions_HuggingFace import compute_codon_distances_for_df, build_codon_aa_mutation_tables

def main():
    parser = argparse.ArgumentParser(description="Explore mutational distances and print low-probability pairs.")
    parser.add_argument(
        "--table-path",
        type=str,
        default="/home3/oml4h/PLM_SARS-CoV-2/Results/iav_mutational_accessibility/Lytras_OG/tables/combined_long_table.csv",
        help="Path to combined_long_table.csv"
    )
    parser.add_argument(
        "--guide-path",
        type=str,
        default="Sequences/IAV_lineage_guide.csv",
        help="Path to the lineage guide CSV"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="H3N2",
        choices=["SC2", "H1N1", "H3N2"],
        help="Model transition matrix to load"
    )
    args = parser.parse_args()

    table_path = Path(args.table_path)
    if not table_path.exists():
        alt_path = Path("/home3/oml4h/PLM_SARS-CoV-2/Results/iav_mutational_accessibility/Lytras_OG/tables/combined_long_table.csv")
        if alt_path.exists():
            table_path = alt_path
        else:
            print(f"Error: Table path {table_path} does not exist.")
            sys.exit(1)

    guide_path = Path(args.guide_path)
    if not guide_path.exists():
        print(f"Error: Guide path {guide_path} does not exist.")
        sys.exit(1)

    print(f"Loading table from {table_path}...")
    df = pd.read_csv(table_path)
    print(f"Loaded table with {len(df)} rows.")

    print(f"Loading lineage guide from {guide_path}...")
    guide_df = pd.read_csv(guide_path)
    
    # Diagnostic print for J.2_int translation
    from Bio import SeqIO
    from Bio.Seq import Seq
    try:
        j2_record = next(SeqIO.parse("Sequences/IAV_lineage_files/J.2_int.nt.fa", "fasta"))
        j2_nt = str(j2_record.seq).upper().replace("U", "T")
        j2_prot = str(Seq(j2_nt).translate(to_stop=False))
        print("\n--- DIAGNOSTIC TRANSLATION OF J.2_int.nt.fa ---")
        print("Nucleotide sequence length:", len(j2_nt))
        print("Translated protein length:", len(j2_prot))
        print("Stop codons in translation:", [i for i, x in enumerate(j2_prot) if x == "*"])
        print("Amino acid at index 160 (0-indexed):", j2_prot[160])
        print("Codon at index 160:", j2_nt[160*3:160*3+3])
        print("Protein segment around 160:", j2_prot[150:170])
        print("Protein segment around 176:", j2_prot[170:190])
        print("-----------------------------------------------\n")
    except Exception as e:
        print(f"Diagnostic error: {e}")

    lineage_cache = {}
    for _, row in guide_df.iterrows():
        label = str(row.get("month", row.get("lineage", "")))
        ref_path = str(row.get("reference", row.get("reference_path", "")))
        if label and ref_path:
            lineage_cache[label] = {
                "reference_path": ref_path,
            }

    print("Building mutation tables...")
    tables = build_codon_aa_mutation_tables(args.model)
    aa_to_codons = tables.get("aa_to_codons_all", tables.get("aa_to_codons"))

    # Define standard genetic code for printing translated amino acid
    standard_genetic_code = {
        "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
        "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
        "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
        "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
        "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
        "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
        "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
        "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
        "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
        "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
        "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
        "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
        "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
        "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
        "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
        "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
    }

    print("Computing codon distances...")
    # Keep model/checkpoint identifiers to explain row origins
    id_cols = [c for c in ["model", "checkpoint_label", "epoch_value"] if c in df.columns]
    select_cols = id_cols + ["lineage", "position", "ref_aa", "aa", "mut_prob", "plm_prob"]
    df_unique = df[select_cols].drop_duplicates().copy()
    processed_df = compute_codon_distances_for_df(df_unique, lineage_cache, aa_to_codons)

    # Translate the retrieved codons for validation
    def get_codon_aa(codon):
        if not codon or pd.isna(codon) or len(codon) != 3:
            return "X"
        return standard_genetic_code.get(codon.upper().replace("U", "T"), "X")
    processed_df["ref_codon_aa"] = processed_df["ref_codon"].apply(get_codon_aa)

    print("\n--- RESULTS ---")
    
    output_cols = id_cols + ["lineage", "position", "ref_codon", "ref_codon_aa", "ref_aa", "aa", "mut_prob", "plm_prob"]

    # 1. mut distance = 1 and mut_prob < 10^-6
    print("\n--- Mut distance = 1 and mut_prob < 10^-6 ---")
    cond1 = (processed_df["nt_mutations"] == 1) & (processed_df["mut_prob"] < 1e-6)
    subset1 = processed_df.loc[cond1]
    if subset1.empty:
        print("No matching pairs found.")
    else:
        print(subset1[output_cols].to_string(index=False))

    # 2. mut distance = 2 and mut_prob < 10^-11
    print("\n--- Mut distance = 2 and mut_prob < 10^-11 ---")
    cond2 = (processed_df["nt_mutations"] == 2) & (processed_df["mut_prob"] < 1e-11)
    subset2 = processed_df.loc[cond2]
    if subset2.empty:
        print("No matching pairs found.")
    else:
        print(subset2[output_cols].to_string(index=False))

if __name__ == "__main__":
    main()
