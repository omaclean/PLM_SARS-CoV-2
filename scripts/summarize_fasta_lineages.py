#!/usr/bin/env python3
"""
Summarize FASTA Lineages
-------------------------
This script parses FASTA files in a directory that match a specified pattern,
extracts their headers, looks them up in a guide TSV file (e.g., nextclade_id_clade.tsv),
and generates a summary table of assigned lineages for each file.

Author: Antigravity Pair Programmer
Date: June 2026
"""

import os
import glob
import argparse
import sys
from collections import Counter, defaultdict

def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize assigned lineages for sequences in FASTA files using a guide TSV mapping file."
    )
    parser.add_argument(
        "-d", "--fasta-dir",
        default="/home3/oml4h/PLM_SARS-CoV-2/Sequences/gisaid_data/alignment_based_19feb26/hard",
        help="Directory containing the FASTA files (default: alignment_based_19feb26/hard)"
    )
    parser.add_argument(
        "-p", "--pattern",
        default="H3N2_*_hard_nextle2_max5.fasta",
        help="File pattern/glob to match FASTA files (default: H3N2_*_hard_nextle*.fasta)"
    )
    parser.add_argument(
        "-t", "--tsv",
        default="/home3/oml4h/PLM_SARS-CoV-2/Sequences/nextclade_id_clade.tsv",
        help="Path to the guide TSV file mapping sequence IDs to lineages (default: nextclade_id_clade.tsv)"
    )
    parser.add_argument(
        "--key-col",
        help="Header name or index (0-based) for the sequence ID column in the TSV. Auto-detected if not specified."
    )
    parser.add_argument(
        "--val-col",
        help="Header name or index (0-based) for the lineage/clade column in the TSV. Auto-detected if not specified."
    )
    parser.add_argument(
        "-o", "--output-prefix",
        default="lineage_summary",
        help="Prefix for output summary files (saves as prefix.csv and prefix.md) (default: lineage_summary)"
    )
    parser.add_argument(
        "--pivot",
        action="store_true",
        help="Generate a pivot table output (files as rows, lineages as columns)"
    )
    return parser.parse_args()

def clean_id(identifier):
    """
    Cleans sequence identifiers to make lookup more robust by extracting
    the core ID (e.g., before pipes or whitespace).
    """
    if not identifier:
        return ""
    identifier = identifier.strip().lstrip(">")
    for delim in ["|", "\t", " ", "/", "_"]:
        if delim in identifier:
            identifier = identifier.split(delim)[0].strip()
    return identifier

def load_tsv_mapping(tsv_path, key_col_arg=None, val_col_arg=None):
    """
    Loads TSV file and returns two dictionaries:
      1. exact_map: exact sequence name -> lineage
      2. clean_map: cleaned sequence name -> lineage
    """
    if not os.path.exists(tsv_path):
        print(f"Error: TSV file not found at {tsv_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading guide TSV from {tsv_path}...")
    
    exact_map = {}
    clean_map = {}
    
    with open(tsv_path, "r", encoding="utf-8") as f:
        # Read header
        header_line = f.readline()
        if not header_line:
            print("Error: TSV file is empty.", file=sys.stderr)
            sys.exit(1)
            
        headers = [h.strip() for h in header_line.split("\t")]
        
        # Determine column indices
        key_idx = 0
        val_idx = 1
        
        # Auto-detect column headers if headers are present and look like strings
        common_keys = {"seqname", "seq_name", "record_id", "recordid", "id", "accession", "sequence"}
        common_vals = {"clade", "lineage", "assigned_lineage", "assignedclade", "subclade"}
        
        detected_key_idx = None
        detected_val_idx = None
        
        for idx, h in enumerate(headers):
            h_lower = h.lower()
            if h_lower in common_keys:
                detected_key_idx = idx
            elif h_lower in common_vals:
                detected_val_idx = idx
                
        if key_col_arg is not None:
            if key_col_arg.isdigit():
                key_idx = int(key_col_arg)
            else:
                try:
                    key_idx = headers.index(key_col_arg)
                except ValueError:
                    print(f"Warning: Key column '{key_col_arg}' not found in TSV headers. Defaulting to first column.")
                    key_idx = 0
        elif detected_key_idx is not None:
            key_idx = detected_key_idx
            
        if val_col_arg is not None:
            if val_col_arg.isdigit():
                val_idx = int(val_col_arg)
            else:
                try:
                    val_idx = headers.index(val_col_arg)
                except ValueError:
                    print(f"Warning: Value column '{val_col_arg}' not found in TSV headers. Defaulting to second column.")
                    val_idx = 1
        elif detected_val_idx is not None:
            val_idx = detected_val_idx

        print(f"Using column index {key_idx} ('{headers[key_idx]}') for Sequence ID")
        print(f"Using column index {val_idx} ('{headers[val_idx]}') for Lineage/Clade")
        
        # Parse TSV rows
        row_count = 0
        for line in f:
            parts = line.strip("\n").split("\t")
            if not parts or len(parts) <= max(key_idx, val_idx):
                continue
            
            key_val = parts[key_idx].strip()
            val_val = parts[val_idx].strip()
            
            if not key_val:
                continue
                
            exact_map[key_val] = val_val
            
            c_id = clean_id(key_val)
            if c_id:
                clean_map[c_id] = val_val
            row_count += 1
            
    print(f"Loaded {row_count} mapping entries.")
    return exact_map, clean_map

def parse_fasta_headers(filepath):
    """
    Parses a FASTA file and returns a list of all headers.
    """
    headers = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(">"):
                headers.append(line[1:].strip())
    return headers

def process_fasta_files(fasta_dir, pattern, exact_map, clean_map):
    """
    Globs FASTA files and processes each one to count lineage assignments.
    """
    search_path = os.path.join(fasta_dir, pattern)
    fasta_files = glob.glob(search_path)
    
    if not fasta_files:
        print(f"Error: No FASTA files found matching pattern '{pattern}' in directory '{fasta_dir}'")
        sys.exit(1)
        
    print(f"Found {len(fasta_files)} FASTA files to process.")
    
    results = {}
    
    for filepath in sorted(fasta_files):
        filename = os.path.basename(filepath)
        print(f"  Processing {filename}...")
        
        headers = parse_fasta_headers(filepath)
        lineage_counts = Counter()
        unassigned_headers = []
        
        for header in headers:
            # Try exact lookup first
            lineage = exact_map.get(header)
            
            if not lineage:
                # Try cleaning the header ID and looking up
                c_header = clean_id(header)
                lineage = clean_map.get(c_header)
                if not lineage:
                    # Try looking up cleaned header in exact map as fallback
                    lineage = exact_map.get(c_header)
                    
            if lineage:
                lineage_counts[lineage] += 1
            else:
                lineage_counts["Unassigned"] += 1
                unassigned_headers.append(header)
                
        results[filename] = {
            "total": len(headers),
            "counts": lineage_counts,
            "unassigned_list": unassigned_headers
        }
        
    return results

def generate_summary_table(results):
    """
    Builds a summary table showing File, Total Sequences, and Lineages (Counts).
    """
    table_data = []
    for filename, data in results.items():
        total = data["total"]
        counts = data["counts"]
        
        # Format lineages as "Lineage1 (count), Lineage2 (count)..." sorted by count descending
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        lineage_str = ", ".join([f"{lin} ({count})" for lin, count in sorted_counts])
        
        table_data.append({
            "File": filename,
            "Total Sequences": total,
            "Assigned Lineages (Counts)": lineage_str
        })
    return table_data

def generate_pivot_table(results):
    """
    Builds a pivot table with files as rows and lineages as columns.
    """
    # Find all unique lineages across all files
    all_lineages = set()
    for data in results.values():
        all_lineages.update(data["counts"].keys())
    
    # Sort lineages (keep 'Unassigned' at the end if present)
    sorted_lineages = sorted([l for l in all_lineages if l != "Unassigned"])
    if "Unassigned" in all_lineages:
        sorted_lineages.append("Unassigned")
        
    table_data = []
    for filename, data in results.items():
        row = {"File": filename, "Total": data["total"]}
        for lin in sorted_lineages:
            row[lin] = data["counts"].get(lin, 0)
        table_data.append(row)
        
    return sorted_lineages, table_data

def write_outputs(summary_table, pivot_data, output_prefix):
    """
    Writes output tables to CSV and Markdown formats.
    """
    # 1. Write Summary Table (Markdown)
    md_path = f"{output_prefix}.md"
    print(f"Writing Markdown summary to {md_path}...")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Lineage Assignment Summary\n\n")
        f.write("| File | Total Sequences | Assigned Lineages (Counts) |\n")
        f.write("| :--- | :---: | :--- |\n")
        for row in summary_table:
            f.write(f"| {row['File']} | {row['Total Sequences']} | {row['Assigned Lineages (Counts)']} |\n")
            
        if pivot_data:
            lineages, pivot_rows = pivot_data
            f.write("\n## Lineage Distribution Pivot Table\n\n")
            headers = ["File", "Total"] + lineages
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("| :--- | :---: | " + " | ".join([":---:" for _ in lineages]) + " |\n")
            for row in pivot_rows:
                row_vals = [row["File"], str(row["Total"])] + [str(row[lin]) for lin in lineages]
                f.write("| " + " | ".join(row_vals) + " |\n")

    # 2. Write Summary Table (CSV)
    csv_path = f"{output_prefix}.csv"
    print(f"Writing CSV summary to {csv_path}...")
    import csv
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["File", "Total Sequences", "Assigned Lineages (Counts)"])
        for row in summary_table:
            writer.writerow([row["File"], row["Total Sequences"], row["Assigned Lineages (Counts)"]])

    # 3. Write Pivot Table (CSV) if pivot_data exists
    if pivot_data:
        pivot_csv_path = f"{output_prefix}_pivot.csv"
        print(f"Writing CSV pivot table to {pivot_csv_path}...")
        lineages, pivot_rows = pivot_data
        with open(pivot_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["File", "Total"] + lineages)
            for row in pivot_rows:
                row_vals = [row["File"], row["Total"]] + [row[lin] for lin in lineages]
                writer.writerow(row_vals)

def print_pretty_table(summary_table):
    """
    Helper to print a clean text table to stdout.
    """
    col_widths = {
        "File": max(len(row["File"]) for row in summary_table),
        "Total": max(len(str(row["Total Sequences"])) for row in summary_table),
    }
    col_widths["File"] = max(col_widths["File"], 4)
    col_widths["Total"] = max(col_widths["Total"], 5)
    
    header_str = f"| {'File':<{col_widths['File']}} | {'Total':^{col_widths['Total']}} | Assigned Lineages (Counts)"
    separator = f"+{'-' * (col_widths['File'] + 2)}+{'-' * (col_widths['Total'] + 2)}+{'-' * 30}"
    
    print(separator)
    print(header_str)
    print(separator)
    for row in summary_table:
        print(f"| {row['File']:<{col_widths['File']}} | {row['Total Sequences']:^{col_widths['Total']}} | {row['Assigned Lineages (Counts)']}")
    print(separator)

def main():
    args = parse_args()
    
    # Load mapping
    exact_map, clean_map = load_tsv_mapping(args.tsv, args.key_col, args.val_col)
    
    # Process FASTA files
    results = process_fasta_files(args.fasta_dir, args.pattern, exact_map, clean_map)
    
    # Generate tables
    summary_table = generate_summary_table(results)
    
    pivot_data = None
    # Generate pivot data by default if we want comprehensive results
    pivot_data = generate_pivot_table(results)
    
    # Write to files
    write_outputs(summary_table, pivot_data, args.output_prefix)
    
    # Print to console
    print("\n--- Summary Table ---")
    print_pretty_table(summary_table)
    print(f"\nSummary successfully saved to {args.output_prefix}.md and {args.output_prefix}.csv!")

if __name__ == "__main__":
    main()
