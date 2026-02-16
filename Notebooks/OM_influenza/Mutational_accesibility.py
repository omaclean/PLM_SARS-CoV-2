# %%
# %load_ext autoreload
# %autoreload 2

import sys
import importlib
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import pearsonr, spearmanr
from adjustText import adjust_text

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Data import CodonTable

# Access Functions
sys.path.append('../../')
module_name = "Functions"
if module_name in sys.modules:
    del sys.modules[module_name]
# Functions = importlib.import_module(module_name)
from Functions_HuggingFace import create_h3_numbering_map

# %% [markdown]
# # Run code

# get fasta imported as nuc

# /home3/oml4h/PLM_SARS-CoV-2/Sequences/huH3N2_HA_CDS.fasta

#  focal sequence is >EPI4551140|HA|A/England/415/2024|EPI_ISL_20080368|J.2.4

# /home3/oml4h/PLM_SARS-CoV-2/Results/test/J.2.4/J.2.4_probability_matrix.csv


fasta_file='/home3/oml4h/PLM_SARS-CoV-2/Sequences/huH3N2_HA_CDS.fasta'
nuc_sequences = list(SeqIO.parse(fasta_file, "fasta"))
seq_keys=[record.id for record in nuc_sequences]
base_lineage_index=seq_keys.index('EPI4551140|HA|A/England/415/2024|EPI_ISL_20080368|J.2.4')
print("Base lineage index:",base_lineage_index)
# translate to protein and confirm matches header of probasbility matrix
protein_sequences=[record.seq.translate(to_stop=True) for record in nuc_sequences]
# get base sequence
base_sequence=protein_sequences[base_lineage_index]
print("Base sequence:",base_sequence)

probability_matrix_file='/home3/oml4h/PLM_SARS-CoV-2/Results/test/J.2.4/J.2.4_probability_matrix.csv'
probability_matrix_file='/home3/oml4h/PLM_SARS-CoV-2/Results/test/ESM2_OG/J.2.4_probability_matrix.csv'
probability_matrix_file="/home3/oml4h/PLM_SARS-CoV-2/Results/test/J.2.4/J.2.4_probability_matrix.csv"
#/home3/oml4h/PLM_SARS-CoV-2/Results/test/ESM2_OG/J.2_int_probability_matrix.csv

outdir='/home3/oml4h/PLM_SARS-CoV-2/Results/test/J.2.4/prob_mutations'
outdir='/home3/oml4h/PLM_SARS-CoV-2/Results/test/J.2.4/prob_mutations_OG_ESM2'
outdir='/home3/oml4h/PLM_SARS-CoV-2/Results/test/J.2.4/prob_mutations_HA3_fine'

import os
os.makedirs(outdir, exist_ok=True)
# import without header, have it as it's own row
probability_matrix=pd.read_csv(probability_matrix_file, index_col=0, header=None)
print("Probability matrix shape:", probability_matrix.shape)

print(probability_matrix.iloc[0,1:20])

# %%
print(19*probability_matrix.shape[1])
# %%
# compare sequences match
# convert prob matrix header to a string
prob_matrix_seq="".join(seq_chars for seq_chars in probability_matrix.iloc[0,:])

print("Probability matrix sequence:",prob_matrix_seq[base_lineage_index])

# assert str(base_sequence)==prob_matrix_seq, "Sequences do not match!"
#find mismatches:
bs = str(base_sequence)

ms = prob_matrix_seq
ms=ms[0:len(bs)]  # trim to length of base sequence
for i in range(len(ms)):
    if bs[i] != ms[i]:
        print(f"Difference at index {i}: Base='{bs[i]}', Matrix='{ms[i]}'")
assert str(bs) == ms, "Sequences do not match!"
# %%
#think ms includes a stop codon at the end?
#print final column of prob matrix
print(probability_matrix.iloc[:, -1])
# %%

# ------------------------------------------------------------------
# NOTATION AND ORDERING
# ------------------------------------------------------------------
# Base Order: A, C, G, T (where T represents T/U)
# Rows (i): Source Nucleotide (From)
# Columns (j): Destination Nucleotide (To)
# Units: Mutation rate per round of replication
# Diagonals: Set to 0.0 as requested
# ------------------------------------------------------------------

# Labels for reference
bases = ['A', 'C', 'G', 'T']

# ------------------------------------------------------------------
# H1N1 TRANSITION MATRIX
# ------------------------------------------------------------------
# Data source: Table S1 (Pauly et al., (14)) - H1N1 Column
h1n1_transitions = np.array([
    # From A: [ A->A,   A->C,    A->G,    A->T ]
    [           0.0,    1.5e-5,  2.0e-4,  1.8e-5 ],
    
    # From C: [ C->A,   C->C,    C->G,    C->T ]
    [           7.7e-6, 0.0,     5.1e-6,  2.7e-5 ],
    
    # From G: [ G->A,   G->C,    G->G,    G->T ]
    [           3.1e-5, 5.4e-5,  0.0,     3.5e-5 ],
    
    # From T: [ T->A,   T->C,    T->G,    T->T ]
    [           1.4e-5, 2.3e-4,  3.5e-5,  0.0    ]
])

# ------------------------------------------------------------------
# H3N2 TRANSITION MATRIX
# ------------------------------------------------------------------
# Data source: Table S1 (Pauly et al., (14)) - H3N2 Column
h3n2_transitions = np.array([
    # From A: [ A->A,   A->C,    A->G,    A->T ]
    [           0.0,    3.4e-5,  3.0e-4,  1.3e-5 ],
    
    # From C: [ C->A,   C->C,    C->G,    C->T ]
    [           1.7e-5, 0.0,     9.7e-6,  4.6e-5 ],
    
    # From G: [ G->A,   G->C,    G->G,    G->T ]
    [           7.2e-5, 2.8e-5,  0.0,     6.0e-5 ],
    
    # From T: [ T->A,   T->C,    T->G,    T->T ]
    [           4.5e-6, 3.1e-4,  3.6e-5,  0.0    ]
])

# ------------------------------------------------------------------
# VERIFICATION PRINT
# ------------------------------------------------------------------
print(f"H1N1 Matrix Shape: {h1n1_transitions.shape}")
print("H1N1 Matrix:\n", h1n1_transitions)
print("-" * 30)
print(f"H3N2 Matrix Shape: {h3n2_transitions.shape}")
print("H3N2 Matrix:\n", h3n2_transitions)
# %%
# Heatmaps for H3N2 transition matrix (4x4)
h3n2_heat = h3n2_transitions.astype(float).copy()
np.fill_diagonal(h3n2_heat, np.nan)

cmap_linear = plt.cm.viridis.copy()
cmap_linear.set_bad(color="white")

plt.figure(figsize=(5, 4.5))
ax = sns.heatmap(
    h3n2_heat,
    annot=True,
    fmt=".1e",
    cmap=cmap_linear,
    xticklabels=bases,
    yticklabels=bases,
    cbar_kws={"label": "Mutation rate"}
)
ax.set_xlabel("To")
ax.set_ylabel("From")
ax.set_title("H3N2 Nucleotide Transition Matrix (Diagonal Masked)")
plt.tight_layout()
plt.savefig(f"{outdir}/h3n2_transition_matrix_heatmap.png", dpi=300)
plt.show()

# Log-scaled palette (diagonals masked to white)
nonzero_vals = h3n2_heat[~np.isnan(h3n2_heat) & (h3n2_heat > 0)]
vmin = nonzero_vals.min() if nonzero_vals.size else 1e-8
vmax = nonzero_vals.max() if nonzero_vals.size else 1.0

cmap_log = plt.cm.magma.copy()
cmap_log.set_bad(color="white")

plt.figure(figsize=(5, 4.5))
ax = sns.heatmap(
    h3n2_heat,
    annot=True,
    fmt=".1e",
    cmap=cmap_log,
    norm=LogNorm(vmin=vmin, vmax=vmax),
    xticklabels=bases,
    yticklabels=bases,
    cbar_kws={"label": "Mutation rate (log scale)"}
)
ax.set_xlabel("To")
ax.set_ylabel("From")
ax.set_title("H3N2 Nucleotide Transition Matrix (Log Scale)")
plt.tight_layout()
plt.savefig(f"{outdir}/h3n2_transition_matrix_heatmap_log.png", dpi=300)
plt.show()
# %%

# %%

# create a 64 x 64 matrix for codon to codon mutation rates based on nucleotide mutation rates above  for H3N2 - each codon is made of 3 nucleotides, so the mutation rate from one codon to another is the product of the mutation rates of the individual nucleotides 

# Calculate probabilities including 'staying same' for diagonal
# For H3N2 input matrix (approx 1 - sum(row))
h3n2_probs = h3n2_transitions.copy()
for i in range(4):
    h3n2_probs[i, i] = 1.0 - np.sum(h3n2_transitions[i, :])

# Generate all 64 codons using bases ['A', 'C', 'G', 'T']
codons = ["".join(trip) for trip in itertools.product(bases, repeat=3)]
n_codons = len(codons)

# Initialize matrix
codon_mutation_matrix = np.zeros((n_codons, n_codons))

# Compute transition probabilities assuming independence of positions
for i, codon_from in enumerate(codons):
    for j, codon_to in enumerate(codons):
        prob = 1.0
        for k in range(3):
            # Find index of base at position k
            idx_from = bases.index(codon_from[k])
            idx_to = bases.index(codon_to[k])
            # Multiply by probability of that specific base transition
            prob *= h3n2_probs[idx_from, idx_to]
        codon_mutation_matrix[i, j] = prob

# Convert to DataFrame
codon_mutation_df = pd.DataFrame(codon_mutation_matrix, index=codons, columns=codons)
codon_mutation_df.to_csv(f"{outdir}/codon_mutation_matrix.csv")

print("Codon Mutation Matrix (H3N2) shape:", codon_mutation_df.shape)
print("Example transitions:")
print(f"P(AAA -> AAG): {codon_mutation_df.loc['AAA', 'AAG']}")
print(f"P(AAA -> GGG): {codon_mutation_df.loc['AAA', 'GGG']}") 

# %% [markdown]
# # Mutational Probability Matrix Calculation
# %%
# Define Genetic Code (DNA -> Amino Acid)
standard_table = CodonTable.unambiguous_dna_by_id[1]
genetic_code = standard_table.forward_table.copy()
# Add stop codons mapped to '*'
for stop_codon in standard_table.stop_codons:
    genetic_code[stop_codon] = '*'


# Reverse map: Amino Acid -> List of Codons
aa_to_codons = {}
for codon, aa in genetic_code.items():
    if aa == '_': continue # Skip stop codons per instruction
    if aa not in aa_to_codons:
        aa_to_codons[aa] = []
    aa_to_codons[aa].append(codon)

# Build codon -> amino acid (including stop) probability matrix (64 x 21)
aa_to_codons_all = {}
for codon, aa in genetic_code.items():
    if aa not in aa_to_codons_all:
        aa_to_codons_all[aa] = []
    aa_to_codons_all[aa].append(codon)
# %% 
target_aas = sorted(aa_to_codons_all.keys(), key=lambda x: (x == '*', x))

# Order codons so rows follow the amino-acid ordering (diagonal-like grouping)
ordered_codons = []
for aa in target_aas:
    ordered_codons.extend(sorted(aa_to_codons_all[aa]))

codon_to_aa_matrix = pd.DataFrame(0.0, index=ordered_codons, columns=target_aas)

for codon_from in ordered_codons:
    for aa in target_aas:
        total_prob = 0.0
        for codon_to in aa_to_codons_all[aa]:
            total_prob += codon_mutation_df.loc[codon_from, codon_to]
        codon_to_aa_matrix.loc[codon_from, aa] = total_prob

# Mask codons mapping to their own amino acid
for codon_from in ordered_codons:
    own_aa = genetic_code.get(codon_from)
    if own_aa in codon_to_aa_matrix.columns:
        codon_to_aa_matrix.loc[codon_from, own_aa] = np.nan

plt.figure(figsize=(18, 10))
ax = sns.heatmap(
    codon_to_aa_matrix.T,
    cmap="viridis",
    cbar_kws={"label": "Codon → Amino Acid probability"}
)
ax.set_xlabel("Starting Codon", fontsize=16)
ax.set_ylabel("Target Amino Acid (including stop '*')", fontsize=16)
ax.set_title("H3N2 Codon → Amino Acid Probability Matrix (64×21)", fontsize=20)
ax.tick_params(axis="both", labelsize=12)
ytick_labels = ["*" if lab == "-" else lab for lab in target_aas]
ax.set_yticklabels(ytick_labels, rotation=279)
plt.tight_layout()
plt.savefig(f"{outdir}/codon_to_amino_acid_matrix_heatmap.png", dpi=300)
plt.show()

# Ratio of most to least likely non-synonymous change
codon_to_aa_values = codon_to_aa_matrix.values
valid_non_syn = codon_to_aa_values[np.isfinite(codon_to_aa_values) & (codon_to_aa_values > 0)]
if valid_non_syn.size:
    ratio_max_min = valid_non_syn.max() / valid_non_syn.min()
    print(f"Ratio (max/min) of non-synonymous change probabilities: {ratio_max_min:.3e}")
else:
    print("No non-synonymous probabilities found for ratio calculation.")
    
# %% 
# Manual sanity checks for codon→AA probabilities
def _raw_codon_to_aa_prob(codon_from, aa):
    return sum(
        codon_mutation_df.loc[codon_from, codon_to]
        for codon_to in aa_to_codons_all.get(aa, [])
    )

test_codons = ["AAA", "ATG", "TGG", "TAA"]
test_aas = ["A", "G", "L", "*", "W"]

print("\nManual codon→AA checks (raw vs matrix):")
for codon_from in test_codons:
    if codon_from not in codon_to_aa_matrix.index:
        continue
    for aa in test_aas:
        if aa not in codon_to_aa_matrix.columns:
            continue
        raw_val = _raw_codon_to_aa_prob(codon_from, aa)
        matrix_val = codon_to_aa_matrix.loc[codon_from, aa]
        diff = np.abs(raw_val - matrix_val)
        print(f"  {codon_from} → {aa}: raw={raw_val:.3e}, matrix={matrix_val:.3e}, |Δ|={diff:.3e}")

# %% 
# Get reference nucleotide sequence
ref_nuc_seq = str(nuc_sequences[base_lineage_index].seq)
# Ensure it's uppercase and uses T instead of U
ref_nuc_seq = ref_nuc_seq.upper().replace('U', 'T')

# PLM Matrix Setup
# Extract the PLM probabilities (exclude the first row which is the sequence header from file read)
# probability_matrix was read with header=None, so row 0 is the sequence info
plm_matrix = probability_matrix.iloc[1:, :].copy()
plm_matrix = plm_matrix.apply(pd.to_numeric, errors='coerce')

# Initialize Output Matrix
# Index: Amino Acids (same as PLM matrix)
# Columns: Same columns as PLM matrix
print("Index dump:", list(plm_matrix.index))
mutational_prob_matrix = pd.DataFrame(0.0, index=plm_matrix.index, columns=plm_matrix.columns)
print("P in index:", 'P' in mutational_prob_matrix.index)

# Verify alignment and populate matrix
mismatch_count = 0
aligned_positions = 0
num_plm_cols = plm_matrix.shape[1]

print(f"Processing {num_plm_cols} positions...")
print("--- [COPILOT FIX] RE-CALCULATING MUTATIONAL MATRIX ---") # Proof of execution

for j in range(num_plm_cols):
    # Get column name/label in the dataframe
    col_idx = plm_matrix.columns[j]
    
    # Map column index to 0-based sequence index
    # Assuming col_idx is 1-based position (e.g. 1, 2, 3...)
    try:
        seq_idx = int(col_idx) - 1
    except ValueError:
        print(f"Skipping non-integer column: {col_idx}")
        continue

    # Bounds check
    if seq_idx < 0 or seq_idx >= len(ref_nuc_seq) // 3:
        # print(f"Column {col_idx} (SeqIdx {seq_idx}) out of bounds")
        continue

    # Expected Amino Acid from PLM header (row 0 of original loaded df)
    # The header row value at column j
    expected_aa = probability_matrix.iloc[0, j]
    
    # Codon range in nucleotide sequence
    nuc_start = 3 * seq_idx
    nuc_end = 3 * seq_idx + 3
    
    current_codon = ref_nuc_seq[nuc_start:nuc_end]
    
    if len(current_codon) < 3:
        continue
        
    translated_aa = genetic_code.get(current_codon, 'X')
    
    # Alignment Check
    # Skip check if expected_aa is nan or not a single char AA
    if isinstance(expected_aa, str) and len(expected_aa) == 1:
        if translated_aa != expected_aa:
            mismatch_count += 1
            if mismatch_count < 5:
                print(f"Alignment Mismatch at col {col_idx} (Pos {seq_idx+1}): Seq Codon {current_codon}->{translated_aa}, PLM expects {expected_aa}")
        else:
            aligned_positions += 1
    
    # Calculate Probability of Mutation to each Target AA
    # Iterate over the amino acids in the PLM matrix (rows)
    for target_aa in mutational_prob_matrix.index:
        
        # 1. RESET MUST HAPPEN INSIDE THIS LOOP
        current_target_total_prob = 0.0  
        
        # Skip if index is NaN or not in our map
        if not isinstance(target_aa, str) or target_aa not in aa_to_codons:
            continue
            
        target_codons = aa_to_codons[target_aa]
        
        # 2. Summation Logic
        if current_codon in codon_mutation_df.index:
            for t_codon in target_codons:
                if t_codon in codon_mutation_df.columns:
                    current_target_total_prob += codon_mutation_df.loc[current_codon, t_codon]
        
        # 3. Assignment
        mutational_prob_matrix.loc[target_aa, col_idx] = current_target_total_prob

print(f"Alignment Check: {aligned_positions} matches, {mismatch_count} mismatches.")

def validate_mutational_matrix(matrix):
    print("\n--- Validating Mutational Probability Matrix ---")
    # Sum over rows (amino acids) for each column (position)
    column_sums = matrix.sum(axis=0) 
    
    print(f"Mean column sum: {column_sums.mean():.6f}")
    print(f"Min column sum: {column_sums.min():.6f}")
    print(f"Max column sum: {column_sums.max():.6f}")
    
    # Check for deviations -> assuming sums should be <= 1.0 (some prob lost to stop codons)
    # But usually very close to 1.0
    deviations = np.abs(column_sums - 1.0)
    # Allow small tolerance
    significant_deviations = deviations[deviations > 1e-3]
    if not significant_deviations.empty:
         print(f"Warning: {len(significant_deviations)} columns sum to != 1.0 (+/- 0.001)")
         print("Top 5 deviations:")
         print(significant_deviations.sort_values(ascending=False).head())
    else:
         print("Validation Passed: All columns sum to approximately 1.0")

validate_mutational_matrix(mutational_prob_matrix)
mutational_prob_matrix.to_csv(f"{outdir}/mutational_prob_matrix.csv")

# %%
# Calculate Combined Matrices
# 1. P_plm * P_mut (Element-wise multiplication)
combined_prob_matrix = plm_matrix * mutational_prob_matrix

# 2. P_plm * sqrt(P_mut)
combined_prob_sqrt_matrix = plm_matrix * np.sqrt(mutational_prob_matrix)

combined_prob_matrix.to_csv(f"{outdir}/combined_prob_matrix.csv")
combined_prob_sqrt_matrix.to_csv(f"{outdir}/combined_prob_sqrt_matrix.csv")

print("Combined Matrix Shape:", combined_prob_matrix.shape)
print("Combined Sqrt Matrix Shape:", combined_prob_sqrt_matrix.shape)

# Example output
print("\nExample (Top 5 rows, first 5 cols) of Combined Matrix:")
print(combined_prob_matrix.iloc[:5, :5])

# %%
# Rank Analysis and Plotting

# 1. Identify Mutations between Focal and Final Sequence
# Focal index is base_lineage_index
final_lineage_index = len(nuc_sequences) - 1
print(f"Focal Index: {base_lineage_index}, Final Index: {final_lineage_index}")

target_record = nuc_sequences[final_lineage_index]
print(f"Target Sequence ID: {target_record.id}")

# Translate Target
target_protein_seq = str(target_record.seq.translate(to_stop=True))
focal_protein_seq = str(base_sequence) # Already translated earlier

# Load Canonical Reference and Create Map
reference_path = "/home3/oml4h/PLM_SARS-CoV-2/Sequences/H3N2_canonical.fa"
ref_record = next(SeqIO.parse(reference_path, "fasta"))
ref_seq_str = str(ref_record.seq)

h3_map_with_ha2 = create_h3_numbering_map(focal_protein_seq, ref_seq_str, HA2_start=330)

# Find differences
observed_mutations = [] # List of (0-based-index, target_aa)
min_len = min(len(focal_protein_seq), len(target_protein_seq))

print(f"Comparing Focal ({len(focal_protein_seq)}) vs Target ({len(target_protein_seq)})")

for i in range(min_len):
    ref_aa = focal_protein_seq[i]
    tgt_aa = target_protein_seq[i]
    if ref_aa != tgt_aa:
        observed_mutations.append((i, tgt_aa))
        # print(f"Mutation at {i}: {ref_aa} -> {tgt_aa}")

print(f"Found {len(observed_mutations)} mutations.")


def get_ranked_mutations(prob_matrix, ref_seq, obs_muts):
    """
    Flattens the probability matrix (excluding self-mutations),
    ranks them, and identifies where the observed mutations fall.
    """
    all_mutations_data = []
    
    # Iterate through columns (positions)
    # prob_matrix columns are string indices or similar, ensure alignment
    num_cols = prob_matrix.shape[1]
    
    for j in range(min(num_cols, len(ref_seq))):
        col_label = prob_matrix.columns[j]
        ref_aa = ref_seq[j]
        
        # Iterate rows (amino acids)
        for aa in prob_matrix.index:
            if not isinstance(aa, str) or len(aa) != 1: continue
            if aa == ref_aa: continue # Skip reference
            
            val = prob_matrix.loc[aa, col_label]
            all_mutations_data.append({
                'Position': j,
                'AA': aa,
                'Probability': val
            })
            
    # Create DataFrame and Rank
    df = pd.DataFrame(all_mutations_data)
    df = df.sort_values(by='Probability', ascending=False).reset_index(drop=True)
    df['Rank'] = df.index + 1
    
    # Find observed
    obs_points = []
    for pos, target_aa in obs_muts:
        # Filter for this specific mutation
        # Note: Position j corresponds to index j in sequence
        match = df[(df['Position'] == pos) & (df['AA'] == target_aa)]
        if not match.empty:
            obs_points.append(match.iloc[0])
            
    return df, pd.DataFrame(obs_points)

# Prepare Matrices
matrices_to_plot = {
    'PLM Probability': plm_matrix,
    'Mutational Probability': mutational_prob_matrix,
    'P_plm * P_mut': combined_prob_matrix,
    'P_plm * sqrt(P_mut)': combined_prob_sqrt_matrix
}

# %%

# Create histogram of mutation_prob_matrix values
plt.figure(figsize=(10, 6))
sns.histplot(mutational_prob_matrix.values.flatten(), bins=100, log_scale=(True, True))
plt.title('Histogram of Mutational Probability Matrix Values')
plt.xlabel('Mutational Probability')
plt.savefig(f"{outdir}/histogram_mutational_prob.png")

# which have values of >0.1
high_values = mutational_prob_matrix.values.flatten() > 0.1
num_high = np.sum(high_values)
print(f"Number of mutations with probability > 0.1: {num_high}")


# %%

# Create Plot
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes_flat = axes.flatten()

for i, (name, matrix) in enumerate(matrices_to_plot.items()):
    ax = axes_flat[i]
    
    ranked_df, obs_df = get_ranked_mutations(matrix, focal_protein_seq, observed_mutations)
    ranked_df['log10Probability'] = np.log10(ranked_df['Probability'])
    obs_df['log10Probability'] = np.log10(obs_df['Probability'])
    
    # Plot all ranks (Line)
    ax.plot(ranked_df['Rank'], ranked_df['log10Probability'], label='All Mutations', color='lightgray', linewidth=1)
    
    # Highlight observed
    if not obs_df.empty:
        sc = ax.scatter(obs_df['Rank'], obs_df['log10Probability'], color='red', zorder=5, label='Observed Diff', s=20)
        
        texts = []
        for idx, row in obs_df.iterrows():
            pos_idx = int(row['Position'])
            ref_aa = focal_protein_seq[pos_idx]
            mut_aa = row['AA']
            rank_val = int(row['Rank'])
            # Label format: RefPosMut using H3 numbering
            h3_label = h3_map_with_ha2.get(pos_idx, str(pos_idx + 1))
            
            if ":" in h3_label:
                # e.g. HA2:49 -> HA2:S49N
                prefix, num = h3_label.split(":", 1)
                label = f"{prefix}:{ref_aa}{num}{mut_aa}_R{rank_val}"
            else:
                label = f"{ref_aa}{h3_label}{mut_aa}_R{rank_val}"
            
            texts.append(ax.text(row['Rank'], row['log10Probability'], label, fontsize=8))
        
        # Use adjust_text to repel labels
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='grey', lw=0.5), ax=ax)

    ax.set_title(name)
    ax.set_xlabel('Rank (1 = Highest Prob)')
    ax.set_ylabel('log10(Probability)')
    #ax.set_xscale('log') # User requested log scale implicitly via "rank each possible mutation" usually implies log-log or semi-log
    #ax.set_yscale('log')
    ax.grid(True, which="both", ls="-", alpha=0.2)

plt.tight_layout()
plt.savefig(f"{outdir}/ranked_mutations.png")
plt.show()


# %%
# correlate the mutation vs plm probabilities plot them and give pearson and spearman ranks

#need to make diagonals NaN in both matrices to avoid self-mutation bias- create a copy first
plm_matrix_no_diag = plm_matrix.copy()
mutational_prob_matrix_no_diag = mutational_prob_matrix.copy()
for j in range(plm_matrix_no_diag.shape[1]):
    if j >= len(focal_protein_seq):
        break
    col_idx = plm_matrix_no_diag.columns[j]
    ref_aa = focal_protein_seq[j]
    if ref_aa in plm_matrix_no_diag.index:
        plm_matrix_no_diag.loc[ref_aa, col_idx] = np.nan
        mutational_prob_matrix_no_diag.loc[ref_aa, col_idx] = np.nan    
# Flatten the matrices to 1D arrays
plm_flat = plm_matrix.values.flatten()
mut_flat = mutational_prob_matrix_no_diag.values.flatten()

#drop NaN values in both
valid_indices = ~np.isnan(plm_flat) & ~np.isnan(mut_flat)
plm_flat = plm_flat[valid_indices]
mut_flat = mut_flat[valid_indices]

# Calculate correlations
pearson_corr, p_p = pearsonr(plm_flat, mut_flat)
spearman_corr, p_s = spearmanr(plm_flat, mut_flat)

plm_flat_log=np.log10(plm_flat + 1e-20)
mut_flat_log=np.log10(mut_flat + 1e-20)
# Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=plm_flat_log, y=mut_flat_log, alpha=0.3)
plt.title('PLM Probability vs Mutational Probability (log10 scale)\n spearman: {:.3f} (p={:.2e}), pearson: {:.3f} (p={:.2e})'.format(spearman_corr, p_s, pearson_corr, p_p))
plt.xlabel('log10(PLM Probability)')
plt.ylabel('log10(Mutational Probability)')
plt.savefig(f"{outdir}/plm_vs_mut_correlation.png")

# which ones have values greater than0.1 in either?
high_plm = plm_flat > 0.1
high_mut = mut_flat > 0.1
high_either = high_plm | high_mut



# %%
# Investigation of "High" Mutational Probabilities
# The user noticed "odd ones" with high probability.
# We look for mutations that have a probability > max(single_nucleotide_mutation_rate).
# This implies summations of multiple paths or multiple synonymous target codons.

max_raw_prob = np.max(h3n2_transitions) # Max off-diagonal element since diagonal was 0 in definition
print(f"\nMax raw nucleotide mutation probability: {max_raw_prob:.2e}")

print("\n--- Investigating High Mutational Probabilities ---")
prob_p_202 = mutational_prob_matrix.loc['P', plm_matrix.columns[201]] # Check 202 via same key used in assign
print(f"[PRE-INVESTIGATION CHECK] P at 202: {prob_p_202}")

print(f"Listing non-synonymous mutations with P > {max_raw_prob:.2e}")

# Iterate through the matrix (it's small enough: ~20 AA rows * ~560 Cols)
high_prob_muts = []

# We need the reference sequence to check for synonymous vs non-synonymous
# focal_protein_seq (AA) and ref_nuc_seq (DNA) are available.
# But mutational_prob_matrix columns align with ref_nuc_seq codons.

for col_idx in mutational_prob_matrix.columns:
    # Get column index as integer (assuming 0, 1, 2...)
    # In the code above: col_idx = plm_matrix.columns[j]
    # plm_matrix.columns are '0', '1', '2' etc (strings) from read_csv
    
    try:
        col_int = int(col_idx) # visual column index
    except ValueError:
        continue # skip if not integer index
        
    seq_idx = col_int - 1 ## FIX: Map 1-based col to 0-based seq
    if seq_idx < 0 or seq_idx >= len(focal_protein_seq):
        continue
        
    ref_aa = focal_protein_seq[seq_idx]
    
    # Get current codon (need to reconstruct logic from above loop)
    nuc_start = 3 * seq_idx
    nuc_end = 3 * seq_idx + 3
    if nuc_end > len(ref_nuc_seq): continue
    current_codon = ref_nuc_seq[nuc_start:nuc_end]
    
    # Iterate rows (target AA)
    for target_aa in mutational_prob_matrix.index:
        if not isinstance(target_aa, str) or len(target_aa) != 1: continue
        
        # Skip Synonymous / Identity
        if target_aa == ref_aa: continue

        prob = mutational_prob_matrix.loc[target_aa, col_idx]
        
        if prob > max_raw_prob:
            high_prob_muts.append({
                'Position': seq_idx + 1,
                'Ref_AA': ref_aa,
                'Ref_Codon': current_codon,
                'Target_AA': target_aa,
                'Probability': prob
            })

# Sort by probability descending
high_prob_df = pd.DataFrame(high_prob_muts)
if not high_prob_df.empty:
    high_prob_df = high_prob_df.sort_values('Probability', ascending=False)
    high_prob_df.to_csv(f"{outdir}/high_prob_mutations.csv")
    
    print(f"Found {len(high_prob_df)} mutations with high probability.")
    print(high_prob_df.head(20).to_string(index=False))
    
    # Detailed breakdown for the top case
    top_case = high_prob_df.iloc[0]
    print(f"\nBreakdown for top case: Pos {top_case['Position']} {top_case['Ref_Codon']}({top_case['Ref_AA']}) -> {top_case['Target_AA']}")
    
    ref_c = top_case['Ref_Codon']
    tgt_aa = top_case['Target_AA']
    tgt_codons = aa_to_codons[tgt_aa]
    
    print(f"Target Codons for {tgt_aa}: {tgt_codons}")
    for tc in tgt_codons:
        if tc in codon_mutation_df.columns:
             p = codon_mutation_df.loc[ref_c, tc]
             if p > 0:
                 print(f"  P({ref_c} -> {tc}) = {p:.2e}")
                 # Identify nucleotide changes
                 changes = []
                 base_probs = []
                 for k in range(3):
                     if ref_c[k] != tc[k]:
                         tr_prob = h3n2_probs[bases.index(ref_c[k]), bases.index(tc[k])]
                         changes.append(f"{ref_c[k]}->{tc[k]} ({tr_prob:.2e})")
                 print(f"    Changes: {', '.join(changes)}")

else:
    print("No mutations found exceeding the max raw probability threshold.")


# %%

# %% [markdown]
# # Lineage-wide mutation accessibility vs PLM probability panel
# %%
import glob
import re
from pathlib import Path

import esm
import torch
from transformers import EsmForMaskedLM

from Functions_HuggingFace import get_mutation_prob_matrix


RUN_LINEAGE_PANEL = True

LINEAGE_CLUSTER_FASTA = "/home3/oml4h/PLM_SARS-CoV-2/Sequences/OM_list_cluster_nuc_plus.fa"
LINEAGE_DIVERSITY_DIR = "/home3/oml4h/PLM_SARS-CoV-2/Sequences/gisaid_data/alignment_based_16feb26_dryrun"
LINEAGE_PANEL_OUTDIR = "/home3/oml4h/PLM_SARS-CoV-2/Results/test/lineage_panel_mutability_vs_plm"

MODEL_RUNS = [
    {
        "tag": "ESM2-HA80",
        "mode": "finetuned",
        "base_model": "esm2_t36_3B_UR50D",
        "layer": 36,
        "checkpoint_dir": "/home3/oml4h/hugging_face_downloads/model_weights_topublish/ESM2-HA80",
        "enabled": True,
    },
    {
        "tag": "OG_ESM2_t36_3B",
        "mode": "raw",
        "base_model": "esm2_t36_3B_UR50D",
        "layer": 36,
        "enabled": True,
    },
]

ALPHA_GRID = np.round(np.arange(-1.0, 1.01, 0.1), 2)
PSEUDOCOUNT = 1e-12

# Keep consistent with lineage handling in build_lineage_subalignments.py
LINEAGE_ALIAS = {
    "J.2.4.1": "K",
}


def _safe_label(label: str) -> str:
    return label.strip().replace(" ", "_").replace("/", "-")


def _is_probably_nucleotide(seq: str) -> bool:
    cleaned = seq.replace("-", "").replace(".", "").upper()
    if not cleaned:
        return False
    nuc_chars = set("ACGTUN")
    frac = sum(1 for char in cleaned if char in nuc_chars) / len(cleaned)
    return frac >= 0.95


def _translate_nt_to_protein(seq: str) -> str:
    cleaned = seq.replace("-", "").replace(".", "").upper().replace("U", "T")
    trimmed = cleaned[: (len(cleaned) // 3) * 3]
    if not trimmed:
        return ""
    return str(Seq(trimmed).translate(to_stop=False)).replace("*", "")


def parse_lineage_references(cluster_fasta: str):
    refs = {}
    for record in SeqIO.parse(cluster_fasta, "fasta"):
        header = record.id.strip()
        lineage_raw = header.split("|")[-1] if "|" in header else header
        lineage = LINEAGE_ALIAS.get(lineage_raw, lineage_raw)

        raw_seq = str(record.seq)
        nt_seq = raw_seq.replace("-", "").replace(".", "").upper().replace("U", "T")
        if _is_probably_nucleotide(raw_seq):
            protein_seq = _translate_nt_to_protein(raw_seq)
        else:
            protein_seq = raw_seq.replace("-", "")

        if lineage not in refs:
            refs[lineage] = {
                "header": header,
                "lineage": lineage,
                "nucleotide": nt_seq,
                "protein": protein_seq,
            }
    return refs


def load_lineage_diversity_fastas(diversity_dir: str):
    lineage_files = {}
    pattern = os.path.join(diversity_dir, "H3N2_*_max*.fasta")
    for path in glob.glob(pattern):
        filename = os.path.basename(path)
        m = re.match(r"H3N2_(.+)_max\d+\.fasta$", filename)
        if not m:
            continue
        lineage_key = m.group(1)
        records = list(SeqIO.parse(path, "fasta"))
        lineage_files[lineage_key] = {
            "path": path,
            "records": records,
        }
    return lineage_files


def compute_lineage_mutation_profile(reference_nt: str, reference_protein: str):
    profile = pd.DataFrame(
        0.0,
        index=list(aa_to_codons.keys()),
        columns=list(range(1, len(reference_protein) + 1)),
    )

    for pos1 in range(1, len(reference_protein) + 1):
        codon = reference_nt[(pos1 - 1) * 3 : pos1 * 3]
        if len(codon) != 3 or codon not in codon_mutation_df.index:
            continue
        for target_aa, target_codons in aa_to_codons.items():
            total = 0.0
            for tc in target_codons:
                if tc in codon_mutation_df.columns:
                    total += float(codon_mutation_df.loc[codon, tc])
            profile.loc[target_aa, pos1] = total

    return profile


def compute_observed_diversity_profile(records, reference_protein: str):
    aa_order = sorted(list(aa_to_codons.keys()))
    n_pos = len(reference_protein)

    counts = pd.DataFrame(0, index=aa_order, columns=list(range(1, n_pos + 1)))
    valid_depth = pd.Series(0, index=list(range(1, n_pos + 1)), dtype=float)

    for record in records:
        seq = str(record.seq)
        seq = seq[:n_pos] if len(seq) >= n_pos else seq + ("-" * (n_pos - len(seq)))
        for pos1 in range(1, n_pos + 1):
            aa = seq[pos1 - 1]
            if aa == "-":
                continue
            if aa in counts.index:
                counts.loc[aa, pos1] += 1
            valid_depth[pos1] += 1

    freqs = counts.copy().astype(float)
    for pos1 in freqs.columns:
        depth = valid_depth[pos1]
        if depth > 0:
            freqs[pos1] = freqs[pos1] / depth
        else:
            freqs[pos1] = 0.0

    return freqs, valid_depth


def softmax_from_log_scores(log_scores: np.ndarray) -> np.ndarray:
    shifted = log_scores - np.nanmax(log_scores)
    ex = np.exp(shifted)
    denom = np.nansum(ex)
    if denom <= 0:
        return np.zeros_like(log_scores)
    return ex / denom


def evaluate_alpha_sweep(combined_df: pd.DataFrame, alpha_grid: np.ndarray) -> pd.DataFrame:
    alpha_results = []
    for alpha in alpha_grid:
        working = combined_df.copy()
        working["log_plm"] = np.log(working["plm_prob"].clip(lower=PSEUDOCOUNT))
        working["log_mut"] = np.log(working["mut_prob"].clip(lower=PSEUDOCOUNT))
        working["combined_log_score"] = working["log_plm"] + alpha * working["log_mut"]

        global_spearman = spearmanr(working["combined_log_score"], working["obs_freq"])
        global_pearson = pearsonr(working["combined_log_score"], working["obs_freq"])

        # scipy API compatibility across versions
        if hasattr(global_spearman, "correlation"):
            sp_r = global_spearman.correlation
            sp_p = global_spearman.pvalue
        elif hasattr(global_spearman, "statistic"):
            sp_r = global_spearman.statistic
            sp_p = global_spearman.pvalue
        else:
            sp_r, sp_p = global_spearman

        if hasattr(global_pearson, "correlation"):
            pr_r = global_pearson.correlation
            pr_p = global_pearson.pvalue
        elif hasattr(global_pearson, "statistic"):
            pr_r = global_pearson.statistic
            pr_p = global_pearson.pvalue
        else:
            pr_r, pr_p = global_pearson

        top_frac = 0.05
        n_top = max(1, int(len(working) * top_frac))
        ranked = working.sort_values("combined_log_score", ascending=False)
        top_hits = ranked.head(n_top)
        baseline_prevalence = float(working["obs_present"].mean()) if len(working) > 0 else np.nan
        top_prevalence = float(top_hits["obs_present"].mean()) if len(top_hits) > 0 else np.nan
        top_enrichment = top_prevalence / baseline_prevalence if baseline_prevalence and baseline_prevalence > 0 else np.nan

        site_nlls = []
        site_rhos = []
        grouped = working.groupby(["lineage", "position", "ref_aa"], sort=False)
        for (_, _, _), site_df in grouped:
            obs_vec = site_df["obs_freq"].to_numpy(dtype=float)
            score_vec = site_df["combined_log_score"].to_numpy(dtype=float)
            if np.nansum(obs_vec) <= 0:
                continue

            obs_norm = obs_vec / np.nansum(obs_vec)
            pred_prob = softmax_from_log_scores(score_vec)
            nll = -np.nansum(obs_norm * np.log(pred_prob.clip(min=PSEUDOCOUNT)))
            site_nlls.append(float(nll))

            if np.nanstd(obs_vec) > 0 and np.nanstd(score_vec) > 0:
                rho_result = spearmanr(score_vec, obs_vec)
                if hasattr(rho_result, "correlation"):
                    rho = rho_result.correlation
                elif hasattr(rho_result, "statistic"):
                    rho = rho_result.statistic
                else:
                    rho = rho_result[0]
                if np.isfinite(rho):
                    site_rhos.append(float(rho))

        alpha_results.append({
            "alpha": float(alpha),
            "global_spearman_r": float(sp_r) if np.isfinite(sp_r) else np.nan,
            "global_spearman_p": float(sp_p) if np.isfinite(sp_p) else np.nan,
            "global_pearson_r": float(pr_r) if np.isfinite(pr_r) else np.nan,
            "global_pearson_p": float(pr_p) if np.isfinite(pr_p) else np.nan,
            "top5pct_enrichment": float(top_enrichment) if np.isfinite(top_enrichment) else np.nan,
            "mean_site_nll": float(np.mean(site_nlls)) if len(site_nlls) > 0 else np.nan,
            "median_site_spearman": float(np.median(site_rhos)) if len(site_rhos) > 0 else np.nan,
            "n_sites_used": int(len(site_nlls)),
        })
    return pd.DataFrame(alpha_results).sort_values("alpha")


if RUN_LINEAGE_PANEL:
    os.makedirs(LINEAGE_PANEL_OUTDIR, exist_ok=True)

    if "codon_mutation_df" not in globals() or "aa_to_codons" not in globals():
        raise RuntimeError(
            "codon_mutation_df / aa_to_codons missing. Run earlier mutational matrix blocks first."
        )

    print("Loading lineage references and circulating diversity...")
    lineage_refs = parse_lineage_references(LINEAGE_CLUSTER_FASTA)
    lineage_diversity = load_lineage_diversity_fastas(LINEAGE_DIVERSITY_DIR)

    available_lineages = []
    for lineage in lineage_refs:
        key = _safe_label(lineage)
        if key in lineage_diversity and len(lineage_diversity[key]["records"]) > 0:
            available_lineages.append(lineage)

    print(f"Lineages with both ref+diversity data: {available_lineages}")

    # cache lineage-level non-PLM components once
    lineage_cache = {}
    for lineage in available_lineages:
        ref = lineage_refs[lineage]
        lineage_key = _safe_label(lineage)
        records = lineage_diversity[lineage_key]["records"]
        reference_protein = ref["protein"]
        reference_nt = ref["nucleotide"]
        if not reference_protein:
            print(f"Skipping {lineage}: empty translated protein")
            continue
        mut_profile = compute_lineage_mutation_profile(reference_nt, reference_protein)
        obs_freq, obs_depth = compute_observed_diversity_profile(records, reference_protein)
        lineage_cache[lineage] = {
            "lineage_key": lineage_key,
            "records": records,
            "reference_protein": reference_protein,
            "mut_profile": mut_profile,
            "obs_freq": obs_freq,
            "obs_depth": obs_depth,
            "diversity_path": lineage_diversity[lineage_key]["path"],
        }

    all_alpha_frames = []
    model_status_rows = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for run_cfg in MODEL_RUNS:
        if not run_cfg.get("enabled", True):
            continue

        model_tag = run_cfg["tag"]
        model_outdir = os.path.join(LINEAGE_PANEL_OUTDIR, model_tag)
        os.makedirs(model_outdir, exist_ok=True)

        try:
            print(f"\nLoading model config: {model_tag}")
            model_raw, alphabet = esm.pretrained.load_model_and_alphabet(run_cfg["base_model"])
            model_raw.eval()
            batch_converter = alphabet.get_batch_converter()

            if run_cfg["mode"] == "finetuned":
                loaded = EsmForMaskedLM.from_pretrained(run_cfg["checkpoint_dir"])
                model = loaded[0] if isinstance(loaded, tuple) else loaded
            else:
                model = model_raw

            model = model.eval().to(device)
            model_status_rows.append({"model": model_tag, "status": "loaded", "reason": ""})
        except Exception as exc:
            print(f"Skipping {model_tag}: failed to load in this environment. Reason: {exc}")
            model_status_rows.append({"model": model_tag, "status": "skipped", "reason": str(exc)})
            continue

        combined_rows = []
        per_lineage_summaries = []

        for lineage, data in lineage_cache.items():
            reference_protein = data["reference_protein"]
            print(f"Processing lineage {lineage} with {model_tag}: n_seq={len(data['records'])}, ref_len={len(reference_protein)}")

            plm_out = get_mutation_prob_matrix(
                reference_protein,
                model,
                run_cfg["layer"],
                device,
                batch_converter,
                alphabet,
            )

            plm_matrix = pd.DataFrame(
                plm_out["mutation_matrix"],
                index=plm_out["amino_acids"],
                columns=plm_out["positions"],
            )

            for pos_label in plm_matrix.columns:
                try:
                    pos1 = int(pos_label)
                except (TypeError, ValueError):
                    continue
                if pos1 > len(reference_protein) or pos1 < 1:
                    continue

                ref_aa = reference_protein[pos1 - 1]
                for aa in plm_matrix.index:
                    if aa == ref_aa:
                        continue
                    if aa not in data["mut_profile"].index or aa not in data["obs_freq"].index:
                        continue

                    plm_prob = float(plm_matrix.loc[aa, pos_label])
                    mut_prob = float(data["mut_profile"].loc[aa, pos1])
                    obs = float(data["obs_freq"].loc[aa, pos1])
                    combined_rows.append({
                        "model": model_tag,
                        "lineage": lineage,
                        "position": int(pos1),
                        "ref_aa": ref_aa,
                        "aa": aa,
                        "plm_prob": plm_prob,
                        "mut_prob": mut_prob,
                        "obs_freq": obs,
                        "obs_present": 1 if obs > 0 else 0,
                        "depth": float(data["obs_depth"][pos1]),
                    })

            data["mut_profile"].to_csv(
                os.path.join(model_outdir, f"{data['lineage_key']}_mutation_accessibility_profile.csv")
            )
            plm_matrix.to_csv(
                os.path.join(model_outdir, f"{data['lineage_key']}_plm_probability_profile.csv")
            )
            data["obs_freq"].to_csv(
                os.path.join(model_outdir, f"{data['lineage_key']}_observed_diversity_profile.csv")
            )

            per_lineage_summaries.append({
                "model": model_tag,
                "lineage": lineage,
                "n_sequences": len(data["records"]),
                "reference_length": len(reference_protein),
                "diversity_fasta": data["diversity_path"],
            })

        combined_df = pd.DataFrame(combined_rows)
        if combined_df.empty:
            print(f"No combined rows produced for {model_tag}; skipping alpha sweep.")
            continue

        combined_df.to_csv(
            os.path.join(model_outdir, "lineage_combined_long_table.csv"),
            index=False,
        )

        lineage_meta_df = pd.DataFrame(per_lineage_summaries)
        lineage_meta_df.to_csv(
            os.path.join(model_outdir, "lineage_panel_metadata.tsv"),
            sep="\t",
            index=False,
        )

        alpha_df = evaluate_alpha_sweep(combined_df, ALPHA_GRID)
        alpha_df["model"] = model_tag
        alpha_df.to_csv(
            os.path.join(model_outdir, "alpha_sweep_fit_metrics.tsv"),
            sep="\t",
            index=False,
        )

        all_alpha_frames.append(alpha_df)

        print(f"Saved per-model alpha sweep for {model_tag} in {model_outdir}")

    status_df = pd.DataFrame(model_status_rows)
    status_df.to_csv(
        os.path.join(LINEAGE_PANEL_OUTDIR, "model_run_status.tsv"),
        sep="\t",
        index=False,
    )

    if len(all_alpha_frames) > 0:
        alpha_all_df = pd.concat(all_alpha_frames, ignore_index=True)
        alpha_all_df.to_csv(
            os.path.join(LINEAGE_PANEL_OUTDIR, "alpha_sweep_fit_metrics_all_models.tsv"),
            sep="\t",
            index=False,
        )

        # Cross-model overlay plots
        metric_cols = [
            "global_spearman_r",
            "global_pearson_r",
            "median_site_spearman",
            "mean_site_nll",
            "top5pct_enrichment",
        ]
        fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True)
        axes = axes.flatten()
        for i, metric_col in enumerate(metric_cols):
            ax = axes[i]
            for model_tag, sub in alpha_all_df.groupby("model"):
                ax.plot(sub["alpha"], sub[metric_col], marker="o", label=model_tag)
            ax.set_title(metric_col)
            ax.set_xlabel("alpha")
            ax.grid(alpha=0.3)
            if i == 0:
                ax.legend()

        # panel 6: best-alpha summary by metric and model (text)
        axes[5].axis("off")
        best_lines = []
        for model_tag, sub in alpha_all_df.groupby("model"):
            if sub.empty:
                continue
            best_sp = sub.loc[sub["global_spearman_r"].idxmax(), "alpha"]
            best_nll = sub.loc[sub["mean_site_nll"].idxmin(), "alpha"]
            best_lines.append(f"{model_tag}: best Spearman alpha={best_sp:.1f}, best NLL alpha={best_nll:.1f}")
        axes[5].text(0.02, 0.98, "\n".join(best_lines), va="top")

        plt.tight_layout()
        plt.savefig(
            os.path.join(LINEAGE_PANEL_OUTDIR, "alpha_sweep_model_comparison.png"),
            dpi=300,
        )
        plt.show()

        print("\nLineage panel complete.")
        print(f"Saved outputs in: {LINEAGE_PANEL_OUTDIR}")
        print(alpha_all_df)
    else:
        print("No model runs completed successfully. Check model_run_status.tsv for details.")

    