# %%
%load_ext autoreload
%autoreload 2

from Bio import SeqIO
from Bio.Seq import Seq
import pandas as pd 
import numpy as np

from adjustText import adjust_text


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from Bio import SeqIO

from Bio import Entrez
from Bio import SeqIO

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

# import without header, have it as it's own row
probability_matrix=pd.read_csv(probability_matrix_file, index_col=0, header=None)
print("Probability matrix shape:", probability_matrix.shape)

probability_matrix.shape
probability_matrix.columns[1:20]
print(probability_matrix.iloc[0,1:20])

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
import numpy as np

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

# create a 64 x 64 matrix for codon to codon mutation rates based on nucleotide mutation rates above  for H3N2 - each codon is made of 3 nucleotides, so the mutation rate from one codon to another is the product of the mutation rates of the individual nucleotides 

import itertools

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

print("Codon Mutation Matrix (H3N2) shape:", codon_mutation_df.shape)
print("Example transitions:")
print(f"P(AAA -> AAG): {codon_mutation_df.loc['AAA', 'AAG']}")
print(f"P(AAA -> GGG): {codon_mutation_df.loc['AAA', 'GGG']}") 

# %% [markdown]
# # Mutational Probability Matrix Calculation
# %%
# Define Genetic Code (DNA -> Amino Acid)
genetic_code = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
    'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
}

# Reverse map: Amino Acid -> List of Codons
aa_to_codons = {}
for codon, aa in genetic_code.items():
    if aa == '_': continue # Skip stop codons per instruction
    if aa not in aa_to_codons:
        aa_to_codons[aa] = []
    aa_to_codons[aa].append(codon)

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
mutational_prob_matrix = pd.DataFrame(0.0, index=plm_matrix.index, columns=plm_matrix.columns)

# Verify alignment and populate matrix
mismatch_count = 0
aligned_positions = 0
num_plm_cols = plm_matrix.shape[1]

print(f"Processing {num_plm_cols} positions...")

for j in range(num_plm_cols):
    # Get column name/label in the dataframe
    col_idx = plm_matrix.columns[j]
    
    # Expected Amino Acid from PLM header (row 0 of original loaded df)
    # The header row value at column j
    expected_aa = probability_matrix.iloc[0, j]
    
    # Codon range in nucleotide sequence
    # 0-indexed AA position implies 3*j to 3*j+3 in nucleotides
    nuc_start = 3 * j
    nuc_end = 3 * j + 3
    
    # Handle end of sequence cases
    if nuc_end > len(ref_nuc_seq):
        # Depending on if the PLM matrix has extra columns (like 'X' mentioned before)
        # or if sequences simply length mismatch.
        # We try to process valid codons only.
        if nuc_start >= len(ref_nuc_seq):
             # Completely out of bounds
             break
        current_codon = ref_nuc_seq[nuc_start:]
    else:
        current_codon = ref_nuc_seq[nuc_start:nuc_end]
    
    if len(current_codon) < 3:
        # Partial codon at end or alignment issue?
        # print(f"Partial codon at {j}: {current_codon}")
        continue
        
    translated_aa = genetic_code.get(current_codon, 'X')
    
    # Alignment Check
    # Skip check if expected_aa is nan or not a single char AA
    if isinstance(expected_aa, str) and len(expected_aa) == 1:
        if translated_aa != expected_aa:
            mismatch_count += 1
            if mismatch_count < 5:
                print(f"Alignment Mismatch at col {j} (Pos {j+1}): Seq Codon {current_codon}->{translated_aa}, PLM expects {expected_aa}")
        else:
            aligned_positions += 1
    
    # Calculate Probability of Mutation to each Target AA
    # Iterate over the amino acids in the PLM matrix (rows)
    for target_aa in mutational_prob_matrix.index:
        # Skip if index is NaN or not in our map
        if not isinstance(target_aa, str) or target_aa not in aa_to_codons:
            continue
            
        target_codons = aa_to_codons[target_aa]
        total_prob = 0.0
        
        # Sum P(current_codon -> potential_target_codon)
        # Requires current_codon to be valid codon (A,C,G,T)
        if current_codon in codon_mutation_df.index:
            for t_codon in target_codons:
                # Safety check if t_codon exist in matrix
                if t_codon in codon_mutation_df.columns:
                    total_prob += codon_mutation_df.loc[current_codon, t_codon]
        
        mutational_prob_matrix.loc[target_aa, col_idx] = total_prob

print(f"Alignment Check: {aligned_positions} matches, {mismatch_count} mismatches.")

# %%
# Calculate Combined Matrices
# 1. P_plm * P_mut (Element-wise multiplication)
combined_prob_matrix = plm_matrix * mutational_prob_matrix

# 2. P_plm * sqrt(P_mut)
combined_prob_sqrt_matrix = plm_matrix * np.sqrt(mutational_prob_matrix)

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

# Create Plot
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes_flat = axes.flatten()

for i, (name, matrix) in enumerate(matrices_to_plot.items()):
    ax = axes_flat[i]
    
    ranked_df, obs_df = get_ranked_mutations(matrix, focal_protein_seq, observed_mutations)
    
    # Plot all ranks (Line)
    ax.plot(ranked_df['Rank'], ranked_df['Probability'], label='All Mutations', color='lightgray', linewidth=1)
    
    # Highlight observed
    if not obs_df.empty:
        sc = ax.scatter(obs_df['Rank'], obs_df['Probability'], color='red', zorder=5, label='Observed Diff', s=20)
        
        texts = []
        for idx, row in obs_df.iterrows():
            pos_idx = int(row['Position'])
            ref_aa = focal_protein_seq[pos_idx]
            mut_aa = row['AA']
            rank_val = int(row['Rank'])
            # Label format: RefPosMut
            label = f"{ref_aa}{pos_idx + 1}{mut_aa}"
            
            texts.append(ax.text(row['Rank'], label, fontsize=8))
        
        # Use adjust_text to repel labels
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='grey', lw=0.5), ax=ax)

    ax.set_title(name)
    ax.set_xlabel('Rank (1 = Highest Prob)')
    ax.set_ylabel('Probability')
    # ax.set_xscale('log') # User requested log scale implicitly via "rank each possible mutation" usually implies log-log or semi-log
    ax.set_yscale('log')
    ax.grid(True, which="both", ls="-", alpha=0.2)

plt.tight_layout()
plt.show()


# %%
