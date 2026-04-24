#!/bin/bash

# Define file paths
LINEAGES_FILE="/home3/oml4h/PLM_SARS-CoV-2/Sequences/SC2_lineages_of_interest.csv"
FASTA_INPUT="/home3/oml4h/PLM_SARS-CoV-2/Sequences/SC2_lineage_references_spike_alignment.fasta"
FASTA_OUTPUT="/home3/oml4h/PLM_SARS-CoV-2/Sequences/SC2_lineage_references_spike_aln_LoA_subset.fasta"

FASTA_INPUT='/home3/oml4h/PLM_SARS-CoV-2/Sequences/SC2_lineage_references_spike_nt_aligned.fasta'
FASTA_OUTPUT='/home3/oml4h/PLM_SARS-CoV-2/Sequences/SC2_lineage_references_spike_nt_aligned_focal_lineages.fasta'

FASTA_INPUT='/home3/oml4h/PLM_SARS-CoV-2/Sequences/SC2_lineage_references_spike_nt_no_gaps.fasta'
FASTA_OUTPUT='/home3/oml4h/PLM_SARS-CoV-2/Sequences/SC2_lineage_references_spike_nt_no_gaps_focal_lineages.fasta'

# Clear output file if it exists
> "$FASTA_OUTPUT"

# Loop through each lineage in the CSV
while IFS= read -r lineage || [[ -n "$lineage" ]]; do
    # Remove potential carriage returns if file was edited in Windows
    lineage=$(echo "$lineage" | tr -d '\r')
    
    # Skip empty lines
    [ -z "$lineage" ] && continue

    # Search for the exact lineage header and capture the following sequence line
    # Using -F for fixed strings to avoid issues with dots in lineage names
    grep -E -A1 ">${lineage}$" "$FASTA_INPUT" >> "$FASTA_OUTPUT"
done < "$LINEAGES_FILE"