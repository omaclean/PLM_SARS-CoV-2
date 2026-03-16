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

from Functions_HuggingFace import (
    create_h3_numbering_map, 
    load_plm_probability_matrix,
    parse_lineage_references,
    load_lineage_diversity_fastas,
)
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
key_matrix_dir = os.path.join(outdir, "key_probability_matrices")
os.makedirs(key_matrix_dir, exist_ok=True)


def _save_key_matrix(matrix_like, filename, index=True):
    if isinstance(matrix_like, pd.DataFrame):
        matrix_like.to_csv(os.path.join(key_matrix_dir, filename), index=index)
        return
    if isinstance(matrix_like, np.ndarray):
        pd.DataFrame(matrix_like).to_csv(os.path.join(key_matrix_dir, filename), index=index)
        return
    pd.DataFrame(matrix_like).to_csv(os.path.join(key_matrix_dir, filename), index=index)


# import without header, have it as it's own row
probability_matrix=pd.read_csv(probability_matrix_file, index_col=0, header=None)
print("Probability matrix shape:", probability_matrix.shape)
_save_key_matrix(probability_matrix, "plm_probability_matrix_raw_with_header_row.csv")

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
# SARS-CoV-2 (SC2) TRANSITION MATRIX
# ------------------------------------------------------------------
# Data source:  De Maio (2021) Table 1 https://pmc.ncbi.nlm.nih.gov/articles/PMC8135539/pdf/evab087.pdf
# Base order remains: A, C, G, T
SC2_transitions = np.array([
    # From A: [ A->A,  A->C,  A->G,  A->T ]
    [          0.0,    0.039, 0.310, 0.123 ],

    # From C: [ C->A,  C->C,  C->G,  C->T ]
    [          0.140,  0.0,   0.022, 3.028 ],

    # From G: [ G->A,  G->C,  G->G,  G->T ]
    [          0.747,  0.113, 0.0,   2.953 ],

    # From T(U): [ T->A, T->C, T->G, T->T ]
    [          0.056,  0.261, 0.036, 0.0   ]
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

_save_key_matrix(pd.DataFrame(h3n2_transitions, index=bases, columns=bases), "h3n2_nucleotide_transition_rates.csv")
_save_key_matrix(pd.DataFrame(h3n2_probs, index=bases, columns=bases), "h3n2_nucleotide_transition_probabilities.csv")

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
_save_key_matrix(codon_mutation_df, "codon_mutation_matrix_h3n2.csv")

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

_save_key_matrix(codon_to_aa_matrix, "codon_to_aa_matrix_h3n2_with_stop.csv")

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
# Information loss from codon-level (64x20) -> amino-acid-level (20x20) aggregation
#
# Build a 20x20 transition matrix by averaging codon->AA probabilities across all
# codons encoding each source amino acid. Then ask: how much variance in the original
# codon-level 64x20 table is retained by this amino-acid-level compression?
aa20 = [aa for aa in target_aas if aa != "*"]
codon_to_aa_20 = codon_to_aa_matrix.loc[ordered_codons, aa20].copy()
_save_key_matrix(codon_to_aa_20, "codon_to_aa_matrix_h3n2_aa20.csv")

def _build_aa20_average_and_reconstruction(codon_to_aa_20_df: pd.DataFrame):
    aa20_transition = pd.DataFrame(np.nan, index=aa20, columns=aa20, dtype=float)
    source_counts = {}

    for source_aa in aa20:
        source_codons = [codon for codon in ordered_codons if genetic_code.get(codon) == source_aa]
        source_counts[source_aa] = len(source_codons)
        if len(source_codons) == 0:
            continue
        aa20_transition.loc[source_aa, :] = codon_to_aa_20_df.loc[source_codons, :].mean(axis=0, skipna=True)

    reconstructed = pd.DataFrame(np.nan, index=ordered_codons, columns=aa20, dtype=float)
    for codon in ordered_codons:
        source_aa = genetic_code.get(codon)
        if source_aa in aa20_transition.index:
            reconstructed.loc[codon, :] = aa20_transition.loc[source_aa, :]

    return aa20_transition, reconstructed, source_counts


def _flattened_fit_metrics(observed_df: pd.DataFrame, predicted_df: pd.DataFrame):
    obs_vals = observed_df.to_numpy(dtype=float).ravel()
    pred_vals = predicted_df.to_numpy(dtype=float).ravel()
    valid_mask = np.isfinite(obs_vals) & np.isfinite(pred_vals)

    if not np.any(valid_mask):
        return {
            "n_entries": 0,
            "total_var": np.nan,
            "residual_var": np.nan,
            "retained_pct": np.nan,
            "corr": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
        }

    obs_valid = obs_vals[valid_mask]
    pred_valid = pred_vals[valid_mask]

    total_var = float(np.var(obs_valid))
    residuals = obs_valid - pred_valid
    residual_var = float(np.var(residuals))
    retained_pct = float(100.0 * (1.0 - residual_var / total_var)) if total_var > 0 else np.nan

    corr_matrix = np.corrcoef(obs_valid, pred_valid)
    corr_val = float(corr_matrix[0, 1]) if corr_matrix.shape == (2, 2) else np.nan
    rmse = float(np.sqrt(np.mean(np.square(residuals))))
    mae = float(np.mean(np.abs(residuals)))

    return {
        "n_entries": int(valid_mask.sum()),
        "total_var": total_var,
        "residual_var": residual_var,
        "retained_pct": retained_pct,
        "corr": corr_val,
        "rmse": rmse,
        "mae": mae,
    }


aa20_transition_avg, codon_to_aa_20_reconstructed, source_codon_counts = _build_aa20_average_and_reconstruction(codon_to_aa_20)
flu_self_metrics = _flattened_fit_metrics(codon_to_aa_20, codon_to_aa_20_reconstructed)

print("\n--- Codon->AA compression analysis (flu-specific 64x20 -> 20x20) ---")
print("Methodological setup:")
print("  1) Construct target matrix Y (flu 64x20): rows=source codons, columns=target amino acids (excluding stop).")
print("  2) Construct compressed matrix A (20x20): for each source amino acid, average Y across synonymous source codons.")
print("  3) Reconstruct codon-level predictor Y_hat (64x20): assign each codon row the corresponding amino-acid row from A.")
print("  4) Compare Y vs Y_hat entry-wise over finite cells only (self-mutation NaNs excluded).")
print("  5) Report variance decomposition + fit metrics.")
print("Metric definitions used:")
print("  total_var = Var(Y)")
print("  residual_var = Var(Y - Y_hat)")
print("  retained_variation_percent = 100 * (1 - residual_var / total_var)")
print("  flattened_correlation_r = Corr(Y, Y_hat)")
print("  RMSE = sqrt(mean((Y - Y_hat)^2)); MAE = mean(abs(Y - Y_hat))")
print(f"Finite entries compared: {flu_self_metrics['n_entries']}")
print(f"Total variance in original 64x20 table: {flu_self_metrics['total_var']:.6e}")
print(f"Residual variance after 20x20 compression: {flu_self_metrics['residual_var']:.6e}")
print(f"Variation retained by 20x20 aggregation: {flu_self_metrics['retained_pct']:.2f}%")
print(f"Flattened correlation (original vs reconstructed): r={flu_self_metrics['corr']:.4f}")
print(f"RMSE between original and reconstructed entries: {flu_self_metrics['rmse']:.6e}")
print(f"MAE between original and reconstructed entries: {flu_self_metrics['mae']:.6e}")

# Generic baseline model: Kimura-80-like nucleotide process
# - AT content ~50% (uniform A/C/G/T frequencies in this simple implementation)
# - transition:transversion bias = 2:1
transition_pairs = {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}
row_total_mut_rate = float(np.mean(np.sum(h3n2_transitions, axis=1)))

kimura80_transitions = np.zeros((4, 4), dtype=float)
for i, src_base in enumerate(bases):
    weights = {}
    weight_sum = 0.0
    for j, dst_base in enumerate(bases):
        if i == j:
            continue
        weight = 2.0 if (src_base, dst_base) in transition_pairs else 1.0
        weights[j] = weight
        weight_sum += weight
    for j, weight in weights.items():
        kimura80_transitions[i, j] = row_total_mut_rate * (weight / weight_sum)

kimura80_probs = kimura80_transitions.copy()
for i in range(4):
    kimura80_probs[i, i] = 1.0 - np.sum(kimura80_transitions[i, :])

# Build codon->codon and codon->AA matrices for Kimura80 generic baseline
codon_mutation_matrix_k80 = np.zeros((n_codons, n_codons), dtype=float)
for i, codon_from in enumerate(codons):
    for j, codon_to in enumerate(codons):
        prob = 1.0
        for k in range(3):
            idx_from = bases.index(codon_from[k])
            idx_to = bases.index(codon_to[k])
            prob *= kimura80_probs[idx_from, idx_to]
        codon_mutation_matrix_k80[i, j] = prob

codon_mutation_df_k80 = pd.DataFrame(codon_mutation_matrix_k80, index=codons, columns=codons)

codon_to_aa_matrix_k80 = pd.DataFrame(0.0, index=ordered_codons, columns=target_aas)
for codon_from in ordered_codons:
    for aa in target_aas:
        total_prob = 0.0
        for codon_to in aa_to_codons_all[aa]:
            total_prob += codon_mutation_df_k80.loc[codon_from, codon_to]
        codon_to_aa_matrix_k80.loc[codon_from, aa] = total_prob

for codon_from in ordered_codons:
    own_aa = genetic_code.get(codon_from)
    if own_aa in codon_to_aa_matrix_k80.columns:
        codon_to_aa_matrix_k80.loc[codon_from, own_aa] = np.nan

codon_to_aa_20_k80 = codon_to_aa_matrix_k80.loc[ordered_codons, aa20].copy()
aa20_transition_avg_k80, codon_to_aa_20_reconstructed_k80, _ = _build_aa20_average_and_reconstruction(codon_to_aa_20_k80)

generic_self_metrics = _flattened_fit_metrics(codon_to_aa_20_k80, codon_to_aa_20_reconstructed_k80)
generic_to_flu_metrics = _flattened_fit_metrics(codon_to_aa_20, codon_to_aa_20_reconstructed_k80)
aa20_flu_vs_k80_metrics = _flattened_fit_metrics(aa20_transition_avg, aa20_transition_avg_k80)

if np.isfinite(generic_to_flu_metrics["residual_var"]) and generic_to_flu_metrics["residual_var"] > 0:
    error_reduction_pct = 100.0 * (
        (generic_to_flu_metrics["residual_var"] - flu_self_metrics["residual_var"])
        / generic_to_flu_metrics["residual_var"]
    )
else:
    error_reduction_pct = np.nan

print("\n--- Generic Kimura-80 baseline vs flu-specific matrix ---")
print("Methodological setup:")
print("  1) Build generic nucleotide model Q_K80 with AT=50% and Ti:Tv=2:1.")
print("  2) Match its total per-row mutation load to the mean flu H3N2 row-sum for scale comparability.")
print("  3) Generate generic codon->codon probabilities (64x64) by independent per-site multiplication.")
print("  4) Aggregate to generic codon->AA matrix Y_K80 (64x20), then compress to A_K80 (20x20), reconstruct Y_hat_K80.")
print("  5) Evaluate two tests against the flu target Y_flu:")
print("     - within-model compression test: Y_K80 vs Y_hat_K80")
print("     - cross-model approximation test: Y_flu vs Y_hat_K80")
print("  6) Quantify gain from flu-specific model using residual-variance reduction:")
print("     gain_percent = 100 * (resid_var(Y_flu vs Y_hat_K80) - resid_var(Y_flu vs Y_hat_flu)) / resid_var(Y_flu vs Y_hat_K80)")
print(
    f"Generic model retained variation in its own 64x20->20x20 compression: "
    f"{generic_self_metrics['retained_pct']:.2f}%"
)
print(
    f"How much of flu 64x20 variation is captured by generic 20x20 reconstruction: "
    f"{generic_to_flu_metrics['retained_pct']:.2f}%"
)
print(
    f"Flu-specific 20x20 error reduction vs generic 20x20 (against flu 64x20 target): "
    f"{error_reduction_pct:.2f}%"
)
print(
    f"Flu vs generic 20x20 matrix similarity (flattened r): "
    f"{aa20_flu_vs_k80_metrics['corr']:.4f}"
)

compression_summary = pd.DataFrame([
    {
        "comparison": "flu64_to_flu20_reconstruction",
        "finite_entries_compared": flu_self_metrics["n_entries"],
        "total_variance": flu_self_metrics["total_var"],
        "residual_variance": flu_self_metrics["residual_var"],
        "retained_variation_percent": flu_self_metrics["retained_pct"],
        "flattened_correlation_r": flu_self_metrics["corr"],
        "rmse": flu_self_metrics["rmse"],
        "mae": flu_self_metrics["mae"],
    },
    {
        "comparison": "k80_64_to_k80_20_reconstruction",
        "finite_entries_compared": generic_self_metrics["n_entries"],
        "total_variance": generic_self_metrics["total_var"],
        "residual_variance": generic_self_metrics["residual_var"],
        "retained_variation_percent": generic_self_metrics["retained_pct"],
        "flattened_correlation_r": generic_self_metrics["corr"],
        "rmse": generic_self_metrics["rmse"],
        "mae": generic_self_metrics["mae"],
    },
    {
        "comparison": "flu64_to_k80_20_reconstruction",
        "finite_entries_compared": generic_to_flu_metrics["n_entries"],
        "total_variance": generic_to_flu_metrics["total_var"],
        "residual_variance": generic_to_flu_metrics["residual_var"],
        "retained_variation_percent": generic_to_flu_metrics["retained_pct"],
        "flattened_correlation_r": generic_to_flu_metrics["corr"],
        "rmse": generic_to_flu_metrics["rmse"],
        "mae": generic_to_flu_metrics["mae"],
    },
    {
        "comparison": "flu20_vs_k80_20",
        "finite_entries_compared": aa20_flu_vs_k80_metrics["n_entries"],
        "total_variance": aa20_flu_vs_k80_metrics["total_var"],
        "residual_variance": aa20_flu_vs_k80_metrics["residual_var"],
        "retained_variation_percent": aa20_flu_vs_k80_metrics["retained_pct"],
        "flattened_correlation_r": aa20_flu_vs_k80_metrics["corr"],
        "rmse": aa20_flu_vs_k80_metrics["rmse"],
        "mae": aa20_flu_vs_k80_metrics["mae"],
    },
])

gain_summary = pd.DataFrame([
    {
        "flu_specific_error_reduction_vs_generic_percent": error_reduction_pct,
        "generic_model_assumption": "AT50_TiTv2to1",
        "generic_row_total_mutation_rate": row_total_mut_rate,
    }
])

compression_summary.to_csv(f"{outdir}/codon_to_aa_compression_summary.csv", index=False)
gain_summary.to_csv(f"{outdir}/flu_vs_k80_gain_summary.csv", index=False)

aa20_transition_avg.to_csv(f"{outdir}/aa20_transition_matrix_from_codon_averages.csv")
aa20_transition_avg_k80.to_csv(f"{outdir}/aa20_transition_matrix_k80_generic.csv")

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
_save_key_matrix(plm_matrix, "plm_probability_matrix_numeric.csv")

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
_save_key_matrix(mutational_prob_matrix, "mutational_probability_matrix_from_codon_model.csv")

# %%
# Calculate Combined Matrices
# 1. P_plm * P_mut (Element-wise multiplication)
combined_prob_matrix = plm_matrix * mutational_prob_matrix

# 2. P_plm * sqrt(P_mut)
combined_prob_sqrt_matrix = plm_matrix * np.sqrt(mutational_prob_matrix)

combined_prob_matrix.to_csv(f"{outdir}/combined_prob_matrix.csv")
combined_prob_sqrt_matrix.to_csv(f"{outdir}/combined_prob_sqrt_matrix.csv")
_save_key_matrix(combined_prob_matrix, "combined_probability_matrix_plm_times_mut.csv")
_save_key_matrix(combined_prob_sqrt_matrix, "combined_probability_matrix_plm_times_sqrt_mut.csv")

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
##
##
##
##
##
##
##
##
##
##
# %% [markdown]
# # Lineage-wide mutation accessibility vs PLM probability panel
# %%
import glob
import re
import os
from pathlib import Path
from Bio import pairwise2
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import esm
import torch
from transformers import EsmForMaskedLM

from Functions_HuggingFace import get_mutation_prob_matrix
# Keep consistent with lineage handling in build_lineage_subalignments.py
LINEAGE_ALIAS = {
    "J.2.4.1": "K",
}

RUN_LINEAGE_PANEL = True
TEST_MODE = False
TEST_MAX_LINEAGES = 3
LINEAGE_CLUSTER_FASTA = "/home3/oml4h/PLM_SARS-CoV-2/Sequences/OM_list_cluster_nuc_plus.fa"


LINEAGE_DIVERSITY_DIR = "/home3/oml4h/PLM_SARS-CoV-2/Sequences/gisaid_data/alignment_based_19feb26_"
LINEAGE_DIVERSITY_FILE_PATTERN = "H3N2_*_max10.fasta"

LINEAGE_DIVERSITY_DIR="/home3/oml4h/PLM_SARS-CoV-2/Sequences/gisaid_data/alignment_based_19feb26/hard"
LINEAGE_DIVERSITY_FILE_PATTERN = 'H3N2*hard_nextle2_max10.fasta'

LINEAGE_PANEL_OUTDIR = "/home3/oml4h/PLM_SARS-CoV-2/Results/test/lineage_panel_mutability_vs_plm/gisaidinc"
# Flexible input selector for diversity FASTAs.
# Examples:
# - "H3N2_*_max5.fasta"
# - "H3N2_*_max10.fasta"
# - "H3N2_*_max10_unique.fasta"
# - "H3N2_*_max*.fasta"  (all max variants)
# 


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
ALPHA_SWEEP_PARALLEL = True
ALPHA_SWEEP_MIN_GRID = 8
ALPHA_SWEEP_MAX_WORKERS = None
METHOD2_SCATTER_ALPHAS = [-1.0, 0.0, 1.0]
METHOD2_SCATTER_MAX_POINTS = 200000



IGNORE_ALIGNMENT_CHARS = {"-", "*", "."}


def _safe_label(label: str) -> str:
    return label.strip().replace(" ", "_").replace("/", "-")


def _clean_pattern_tag(file_pattern: str) -> str:
    tag = file_pattern.replace("*", "")
    tag = tag.replace(".fasta", "")
    tag = re.sub(r"_+", "_", tag).strip("_")
    return _safe_label(tag) if tag else "pattern"


RUN_MODE_TAG = "test" if TEST_MODE else "full"
DIVERSITY_PATTERN_TAG = _clean_pattern_tag(LINEAGE_DIVERSITY_FILE_PATTERN)
OUTPUT_TAG = f"{RUN_MODE_TAG}_{DIVERSITY_PATTERN_TAG}"


# %%
if RUN_LINEAGE_PANEL:
    os.makedirs(LINEAGE_PANEL_OUTDIR, exist_ok=True)

    if "codon_mutation_df" not in globals() or "aa_to_codons" not in globals():
        raise RuntimeError(
            "codon_mutation_df / aa_to_codons missing. Run earlier mutational matrix blocks first."
        )

    print("Loading lineage references and circulating diversity...")
    lineage_refs = parse_lineage_references(LINEAGE_CLUSTER_FASTA, LINEAGE_ALIAS)
    lineage_diversity = load_lineage_diversity_fastas(
        LINEAGE_DIVERSITY_DIR,
        LINEAGE_DIVERSITY_FILE_PATTERN,
    )

    available_lineages = []
    for lineage in lineage_refs:
        key = _safe_label(lineage)
        if key in lineage_diversity and len(lineage_diversity[key]["records"]) > 0:
            available_lineages.append(lineage)

    print(f"Lineages with both ref+diversity data: {available_lineages}")
    print(f"Diversity FASTA selector pattern: {LINEAGE_DIVERSITY_FILE_PATTERN}")
    print(f"Output tag: {OUTPUT_TAG}")

    if TEST_MODE:
        available_lineages = available_lineages[:TEST_MAX_LINEAGES]
        print(f"TEST_MODE enabled: limiting to first {len(available_lineages)} lineage(s)")

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
        mut_profile = compute_lineage_mutation_profile(
            reference_nt, reference_protein, aa_to_codons, codon_mutation_df
        )
        ref_to_aln_col, aln_len, matched_pairs = build_reference_to_alignment_column_map(
            reference_protein, records, aa_to_codons, IGNORE_ALIGNMENT_CHARS
        )
        obs_freq, obs_depth, diversity_stats = compute_observed_diversity_profile_fast(
            records,
            reference_protein,
            ref_to_aln_col,
            aln_len,
            aa_to_codons,
            IGNORE_ALIGNMENT_CHARS,
        )

        print(
            f"[ALIGNMENT DIFF] {lineage}: any_differing_sites={diversity_stats['differing_sites']}, "
            f"fixed_differing_sites={diversity_stats['fixed_differing_sites']} "
            f"/ compared_sites={diversity_stats['compared_sites']} "
            f"(mapped_ref_sites={diversity_stats['mapped_sites']}/{len(reference_protein)}, "
            f"matched_pairs={matched_pairs}, alignment_len={diversity_stats['alignment_length']}; "
            "ignoring gaps/stops)"
        )

        lineage_cache[lineage] = {
            "lineage_key": lineage_key,
            "records": records,
            "reference_protein": reference_protein,
            "mut_profile": mut_profile,
            "obs_freq": obs_freq,
            "obs_depth": obs_depth,
            "ref_to_aln_col": ref_to_aln_col,
            "alignment_diff_stats": diversity_stats,
            "diversity_path": lineage_diversity[lineage_key]["path"],
            "diversity_tag": lineage_diversity[lineage_key]["diversity_tag"],
        }

    all_alpha_frames = []
    model_status_rows = []
    per_lineage_best_rows = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for run_cfg in MODEL_RUNS:
        if not run_cfg.get("enabled", True):
            continue

        model_tag = run_cfg["tag"]
        model_outdir = os.path.join(LINEAGE_PANEL_OUTDIR, model_tag)
        os.makedirs(model_outdir, exist_ok=True)

        model = None
        alphabet = None
        batch_converter = None
        model_ready = False
        used_cached_plm = False
        model_load_attempted = False
        model_load_failed_reason = ""
        model_runtime_failed = False
        model_runtime_failed_reason = ""

        combined_rows = []
        per_lineage_summaries = []

        for lineage, data in lineage_cache.items():
            reference_protein = data["reference_protein"]
            print(f"Processing lineage {lineage} with {model_tag}: n_seq={len(data['records'])}, ref_len={len(reference_protein)}")

            plm_profile_path = os.path.join(
                model_outdir,
                _tag_output_name(f"{data['lineage_key']}_plm_probability_profile.csv", TEST_MODE, DIVERSITY_PATTERN_TAG, OUTPUT_TAG),
            )

            plm_matrix = None
            if os.path.exists(plm_profile_path):
                try:
                    plm_matrix = pd.read_csv(plm_profile_path, index_col=0)
                    used_cached_plm = True
                    print(f"Using existing PLM matrix: {plm_profile_path}")
                except Exception as exc:
                    print(f"Failed to load existing PLM matrix for {lineage} ({plm_profile_path}): {exc}")
                    print("Recomputing PLM matrix for this lineage.")

            if plm_matrix is None:
                if not model_ready and not model_load_attempted:
                    model_load_attempted = True
                    try:
                        print(f"\nLoading model config: {model_tag}")
                        model_raw, alphabet_local = esm.pretrained.load_model_and_alphabet(run_cfg["base_model"])
                        model_raw.eval()
                        batch_converter_local = alphabet_local.get_batch_converter()

                        if run_cfg["mode"] == "finetuned":
                            loaded = EsmForMaskedLM.from_pretrained(run_cfg["checkpoint_dir"])
                            model_local = loaded[0] if isinstance(loaded, tuple) else loaded
                        else:
                            model_local = model_raw

                        model = model_local.eval().to(device)
                        alphabet = alphabet_local
                        batch_converter = batch_converter_local
                        model_ready = True
                    except Exception as exc:
                        model_load_failed_reason = str(exc)
                        print(f"Skipping PLM generation for {model_tag}: failed to load in this environment. Reason: {exc}")

                if not model_ready:
                    continue

                try:
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
                    plm_matrix.to_csv(plm_profile_path)
                except Exception as exc:
                    model_runtime_failed = True
                    model_runtime_failed_reason = str(exc)
                    print(f"Skipping model {model_tag}: runtime failure during PLM generation on lineage {lineage}. Reason: {exc}")
                    break

            if model_runtime_failed:
                break

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
                os.path.join(model_outdir, _tag_output_name(f"{data['lineage_key']}_mutation_accessibility_profile.csv", TEST_MODE, DIVERSITY_PATTERN_TAG, OUTPUT_TAG))
            )
            data["obs_freq"].to_csv(
                os.path.join(model_outdir, _tag_output_name(f"{data['lineage_key']}_observed_diversity_profile.csv", TEST_MODE, DIVERSITY_PATTERN_TAG, OUTPUT_TAG))
            )

            per_lineage_summaries.append({
                "model": model_tag,
                "lineage": lineage,
                "n_sequences": len(data["records"]),
                "reference_length": len(reference_protein),
                "mapped_ref_sites": int(data["alignment_diff_stats"]["mapped_sites"]),
                "compared_sites_non_gap_non_stop": int(data["alignment_diff_stats"]["compared_sites"]),
                "differing_sites_vs_reference_non_gap_non_stop": int(data["alignment_diff_stats"]["differing_sites"]),
                "fixed_differing_sites_vs_reference_non_gap_non_stop": int(data["alignment_diff_stats"]["fixed_differing_sites"]),
                "diversity_fasta": data["diversity_path"],
                "diversity_tag": data["diversity_tag"],
                "plm_profile": plm_profile_path,
            })

        if model_runtime_failed:
            combined_rows = []
            per_lineage_summaries = []
            model_status_rows.append({"model": model_tag, "status": "skipped", "reason": f"runtime failure: {model_runtime_failed_reason}"})
            continue

        if model_ready:
            model_status_rows.append({"model": model_tag, "status": "loaded", "reason": "used for PLM generation"})
        elif used_cached_plm:
            model_status_rows.append({"model": model_tag, "status": "cached_only", "reason": "all PLM profiles reused from disk"})
        elif model_load_attempted:
            model_status_rows.append({"model": model_tag, "status": "skipped", "reason": model_load_failed_reason})

        combined_df = pd.DataFrame(combined_rows)
        if combined_df.empty:
            print(f"No combined rows produced for {model_tag}; skipping alpha sweep.")
            continue

        combined_df.to_csv(
            os.path.join(model_outdir, _tag_output_name("lineage_combined_long_table.csv", TEST_MODE, DIVERSITY_PATTERN_TAG, OUTPUT_TAG)),
            index=False,
        )

        lineage_meta_df = pd.DataFrame(per_lineage_summaries)
        lineage_meta_df.to_csv(
            os.path.join(model_outdir, _tag_output_name("lineage_panel_metadata.tsv", TEST_MODE, DIVERSITY_PATTERN_TAG, OUTPUT_TAG)),
            sep="\t",
            index=False,
        )

        use_parallel_alpha = ALPHA_SWEEP_PARALLEL and len(ALPHA_GRID) >= ALPHA_SWEEP_MIN_GRID
        if use_parallel_alpha:
            print(f"Running alpha sweep in parallel for {model_tag} (n_alpha={len(ALPHA_GRID)})")

        alpha_df = evaluate_alpha_sweep(
            combined_df,
            ALPHA_GRID,
            parallel=use_parallel_alpha,
            max_workers=ALPHA_SWEEP_MAX_WORKERS,
        )
        alpha_df["model"] = model_tag
        alpha_df.to_csv(
            os.path.join(model_outdir, _tag_output_name("alpha_sweep_fit_metrics.tsv", TEST_MODE, DIVERSITY_PATTERN_TAG, OUTPUT_TAG)),
            sep="\t",
            index=False,
        )

        scatter_alphas = list(dict.fromkeys([float(a) for a in METHOD2_SCATTER_ALPHAS]))

        if len(scatter_alphas) > 0 and len(combined_df) > 0:
            lineage_names = sorted(combined_df["lineage"].dropna().unique().tolist())
            lineage_seq_counts = {
                lineage_name: len(lineage_cache.get(lineage_name, {}).get("records", []))
                for lineage_name in lineage_names
            }
            n_lineages = len(lineage_names)
            if n_lineages > 0:
                nrows = n_lineages
                ncols = len(scatter_alphas)
                fig_sc, axes_sc = plt.subplots(
                    nrows,
                    ncols,
                    figsize=(4.5 * ncols, 3.8 * nrows),
                    sharex=None,
                    sharey="row",
                )

                axes_sc = np.array(axes_sc)
                if axes_sc.ndim == 1:
                    if nrows == 1:
                        axes_sc = axes_sc.reshape(1, -1)
                    else:
                        axes_sc = axes_sc.reshape(-1, 1)

                for row_idx, lineage_name in enumerate(lineage_names):
                    n_seq_lineage = int(lineage_seq_counts.get(lineage_name, 0))
                    lineage_scatter_df = combined_df.loc[
                        combined_df["lineage"] == lineage_name,
                        ["obs_freq", "plm_prob", "mut_prob"],
                    ].copy()

                    if len(lineage_scatter_df) > METHOD2_SCATTER_MAX_POINTS:
                        lineage_scatter_df = lineage_scatter_df.sample(METHOD2_SCATTER_MAX_POINTS, random_state=0)

                    for col_idx, alpha_value in enumerate(scatter_alphas):
                        ax = axes_sc[row_idx, col_idx]

                        if len(lineage_scatter_df) == 0:
                            ax.set_title(f"alpha={alpha_value:.2f}\nno data")
                            ax.grid(alpha=0.2)
                            continue

                        x_vals = np.log10(
                            lineage_scatter_df["plm_prob"].clip(lower=PSEUDOCOUNT)
                            * np.power(lineage_scatter_df["mut_prob"].clip(lower=PSEUDOCOUNT), alpha_value)
                        )
                        y_vals = np.log10(lineage_scatter_df["obs_freq"].clip(lower=PSEUDOCOUNT))

                        sns.scatterplot(
                            x=x_vals,
                            y=y_vals,
                            ax=ax,
                            s=8,
                            alpha=0.25,
                            edgecolor=None,
                        )

                        corr_result = spearmanr(x_vals, y_vals)
                        corr_r, _ = _extract_corr_pvalue(corr_result)
                        ax.set_title(
                            f"alpha={alpha_value:.2f}\n"
                            f"ρ={corr_r:.3f}, n_mut={len(lineage_scatter_df)}, n_seq={n_seq_lineage}"
                        )
                        ax.grid(alpha=0.25)

                        if row_idx == nrows - 1:
                            ax.set_xlabel("log10(PLM × mut^alpha)")
                        else:
                            ax.set_xlabel("")

                        if col_idx == 0:
                            ax.set_ylabel(f"{lineage_name}\nlog10(observed freq)")
                        else:
                            ax.set_ylabel("")

                fig_sc.suptitle(
                    "Method B (mutation-level): observed mutation frequency vs PLM×mutation accessibility score\n"
                    "rows = lineages, columns = alpha values"
                )
                plt.tight_layout(rect=(0, 0, 1, 0.95))
                plt.savefig(
                    os.path.join(
                        model_outdir,
                        _tag_output_name("method2_obsfreq_vs_plm_mut_scatter_by_lineage_grid.png", TEST_MODE, DIVERSITY_PATTERN_TAG, OUTPUT_TAG),
                    ),
                    dpi=300,
                )
                plt.show()

        # Per-lineage best-alpha extraction for explicit Method A vs Method B overlays
        for lineage_name, lineage_df in combined_df.groupby("lineage"):
            lineage_alpha = evaluate_alpha_sweep(
                lineage_df,
                ALPHA_GRID,
                parallel=use_parallel_alpha,
                max_workers=ALPHA_SWEEP_MAX_WORKERS,
            )
            if lineage_alpha.empty:
                continue

            idx_a = lineage_alpha["site_top10pct_mutated_enrichment"].idxmax()
            idx_b = lineage_alpha["mut_flat_global_spearman_r"].idxmax()

            per_lineage_best_rows.append({
                "model": model_tag,
                "lineage": lineage_name,
                "method": "Method A (Site-level)",
                "criterion": "max site_top10pct_mutated_enrichment",
                "best_alpha": float(lineage_alpha.loc[idx_a, "alpha"]),
                "best_value": float(lineage_alpha.loc[idx_a, "site_top10pct_mutated_enrichment"]),
            })
            per_lineage_best_rows.append({
                "model": model_tag,
                "lineage": lineage_name,
                "method": "Method B (Mutation-level flattened)",
                "criterion": "max mut_flat_global_spearman_r",
                "best_alpha": float(lineage_alpha.loc[idx_b, "alpha"]),
                "best_value": float(lineage_alpha.loc[idx_b, "mut_flat_global_spearman_r"]),
            })

        all_alpha_frames.append(alpha_df)

        print(f"Saved per-model alpha sweep for {model_tag} in {model_outdir}")

    status_df = pd.DataFrame(model_status_rows)
    status_df.to_csv(
        os.path.join(LINEAGE_PANEL_OUTDIR, _tag_output_name("model_run_status.tsv", TEST_MODE, DIVERSITY_PATTERN_TAG, OUTPUT_TAG)),
        sep="\t",
        index=False,
    )

    if len(all_alpha_frames) > 0:
        alpha_all_df = pd.concat(all_alpha_frames, ignore_index=True)
        alpha_all_df.to_csv(
            os.path.join(LINEAGE_PANEL_OUTDIR, _tag_output_name("alpha_sweep_fit_metrics_all_models.tsv", TEST_MODE, DIVERSITY_PATTERN_TAG, OUTPUT_TAG)),
            sep="\t",
            index=False,
        )

        print(
            "Method A (site-level) context: each site gets a single score from the max mutation-level score at that site; "
            "site precision = fraction of top-scored sites (top 10%) that have any observed mutation in the lineage set."
        )

        # Cross-model overlay plots
        metric_cols = [
            "site_top10pct_mutated_enrichment",
            "site_top10pct_mutated_precision",
            "site_rank_spearman_r",
            "mut_flat_global_spearman_r",
            "mut_flat_global_pearson_r",
            "mut_flat_mean_site_nll",
        ]
        fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True)
        axes = axes.flatten()
        for i, metric_col in enumerate(metric_cols):
            ax = axes[i]
            for model_tag, sub in alpha_all_df.groupby("model"):
                ax.plot(sub["alpha"], sub[metric_col], marker="o", label=model_tag)
            title_map = {
                "site_top10pct_mutated_enrichment": "Method A (site-level): \nenrichment of mutated sites in top 10% scored sites",
                "site_top10pct_mutated_precision": "Method A (site-level): f\nraction of top 10% scored sites that are observed mutated",
                "site_rank_spearman_r": "Method A (site-level): \nSpearman(site score vs observed site mutation burden)",
                "mut_flat_global_spearman_r": "Method B (mutation-level):\n Spearman(pred score vs observed mutation frequency)",
                "mut_flat_global_pearson_r": "Method B (mutation-level): \nPearson(pred score vs observed mutation frequency)",
                "mut_flat_mean_site_nll": "Method B (mutation-level):\n mean site-level NLL of observed residue distribution",
            }
            ylabel_map = {
                "site_top10pct_mutated_enrichment": "Enrichment ratio (top-10% sites / baseline)",
                "site_top10pct_mutated_precision": "Precision (fraction of top-10% sites with any observed mutation)",
                "site_rank_spearman_r": "Spearman correlation coefficient",
                "mut_flat_global_spearman_r": "Spearman correlation coefficient",
                "mut_flat_global_pearson_r": "Pearson correlation coefficient",
                "mut_flat_mean_site_nll": "Mean site-level negative log-likelihood (lower is better)",
            }
            ax.set_title(title_map.get(metric_col, metric_col))
            ax.set_xlabel("Alpha weight on mutation accessibility in log-space combination")
            ax.set_ylabel(ylabel_map.get(metric_col, "Metric value"))
            ax.grid(alpha=0.3)
            if i == 0:
                ax.legend()

        plt.tight_layout()
        plt.savefig(
            os.path.join(LINEAGE_PANEL_OUTDIR, _tag_output_name("alpha_sweep_model_comparison.png", TEST_MODE, DIVERSITY_PATTERN_TAG, OUTPUT_TAG)),
            dpi=300,
        )
        plt.show()

        # Explicit two-method best-alpha summary table
        best_rows = []
        for model_tag, sub in alpha_all_df.groupby("model"):
            if sub.empty:
                continue
            best_rows.append({
                "model": model_tag,
                "method": "Method A (Site-level)",
                "criterion": "max site_top10pct_mutated_enrichment",
                "best_alpha": float(sub.loc[sub["site_top10pct_mutated_enrichment"].idxmax(), "alpha"]),
            })
            best_rows.append({
                "model": model_tag,
                "method": "Method B (Mutation-level flattened)",
                "criterion": "max mut_flat_global_spearman_r",
                "best_alpha": float(sub.loc[sub["mut_flat_global_spearman_r"].idxmax(), "alpha"]),
            })
            best_rows.append({
                "model": model_tag,
                "method": "Method B (Mutation-level flattened)",
                "criterion": "min mut_flat_mean_site_nll",
                "best_alpha": float(sub.loc[sub["mut_flat_mean_site_nll"].idxmin(), "alpha"]),
            })
        pd.DataFrame(best_rows).to_csv(
            os.path.join(LINEAGE_PANEL_OUTDIR, _tag_output_name("best_alpha_two_methods.tsv", TEST_MODE, DIVERSITY_PATTERN_TAG, OUTPUT_TAG)),
            sep="\t",
            index=False,
        )

        if len(per_lineage_best_rows) > 0:
            per_lineage_best_df = pd.DataFrame(per_lineage_best_rows)
            per_lineage_best_df.to_csv(
                os.path.join(LINEAGE_PANEL_OUTDIR, _tag_output_name("best_alpha_per_lineage_two_methods.tsv", TEST_MODE, DIVERSITY_PATTERN_TAG, OUTPUT_TAG)),
                sep="\t",
                index=False,
            )

            fig_overlay, axes_overlay = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
            method_order = ["Method A (Site-level)", "Method B (Mutation-level flattened)"]

            for i, method_name in enumerate(method_order):
                ax = axes_overlay[i]
                subset = per_lineage_best_df[per_lineage_best_df["method"] == method_name].copy()
                if subset.empty:
                    ax.set_title(f"{method_name} (no data)")
                    ax.axis("off")
                    continue

                sns.stripplot(
                    data=subset,
                    x="lineage",
                    y="best_alpha",
                    hue="model",
                    dodge=True,
                    jitter=0.15,
                    size=7,
                    ax=ax,
                )
                ax.axhline(0.0, linestyle="--", color="black", alpha=0.6)
                ax.set_title(f"Per-lineage best alpha overlay\n{method_name}")
                ax.set_xlabel("Lineage")
                ax.set_ylabel("Best alpha")
                ax.grid(alpha=0.25, axis="y")
                if i > 0:
                    ax.get_legend().remove()

            handles, labels = axes_overlay[0].get_legend_handles_labels()
            fig_overlay.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
            plt.tight_layout(rect=[0, 0, 1, 0.92])
            plt.savefig(
                os.path.join(LINEAGE_PANEL_OUTDIR, _tag_output_name("best_alpha_per_lineage_overlay.png", TEST_MODE, DIVERSITY_PATTERN_TAG, OUTPUT_TAG)),
                dpi=300,
            )
            plt.show()

        print("\nLineage panel complete.")
        print(f"Saved outputs in: {LINEAGE_PANEL_OUTDIR}")
        print(alpha_all_df)
    else:
        print("No model runs completed successfully. Check model_run_status.tsv for details.")

    
    # %%
