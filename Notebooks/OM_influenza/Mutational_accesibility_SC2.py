# %%
# %load_ext autoreload
# %autoreload 2

import sys
import importlib
import itertools
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv
from matplotlib.colors import LogNorm
from scipy.stats import pearsonr, spearmanr
from adjustText import adjust_text

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Data import CodonTable
import esm
import torch
from transformers import EsmForMaskedLM

# Access Functions
try:
    _file = __file__
except NameError:
    _file = "/home3/oml4h/PLM_SARS-CoV-2/Notebooks/OM_influenza/Mutational_accesibility_SC2.py"
SCRIPT_DIR = os.path.dirname(os.path.abspath(_file))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Functions_HuggingFace import (
    _resolve_plm_max_nt_length,
    _build_coordinate_map,
    _is_probably_nucleotide_sequence,
    _load_comparison_protein_sequence,
    _save_key_matrix,
    _load_plm_runtime,
    _write_plm_probability_matrix,
    _ensure_plm_probability_matrix,
    _raw_codon_to_aa_prob,
    _build_aa20_average_and_reconstruction,
    _flattened_fit_metrics,
    validate_mutational_matrix,
    get_ranked_mutations,
    _load_single_focal_reference,
    _tag_output_name,
    _is_probably_nucleotide,
    _translate_nt_to_protein,
    build_reference_to_alignment_column_map,
    compute_lineage_mutation_profile,
    compute_observed_diversity_profile_fast,
    evaluate_alpha_sweep,
    _extract_corr_pvalue,
    _safe_label,
    _clean_pattern_tag,
    load_plm_probability_matrix,
    get_mutation_prob_matrix,
    parse_lineage_references,
    load_lineage_diversity_fastas,
    bases, h1n1_transitions, h3n2_transitions, SC2_transitions, TRANSITION_MATRICES,
)
# # Run code

# get fasta imported as nuc

# Example focal-sequence FASTA path

#  focal sequence is >EPI4551140|HA|A/England/415/2024|EPI_ISL_20080368|J.2.4

# /home3/oml4h/PLM_SARS-CoV-2/Results/test/J.2.4/J.2.4_probability_matrix.csv


# %%
# =====================================================================
# I. OVERALL RUN CONFIGURATION
# =====================================================================

# GPU SAFETY: Set to True to error if no CUDA is available. 
GPU_REQUIRED = True

# ANALYSIS MODE:
# "MONTHLY_GUIDE" -> Processes multiple snapshots from POOLED_DIVERSITY_GUIDE CSV.
# "SINGLE_FASTA"  -> Processes only the file in POOLED_DIVERSITY_FASTA.
ANALYSIS_MODE = "MONTHLY_GUIDE"

# MODEL SELECTION: Choose which PLM model(s) to run and compare.
# Options: "ESMC_600M", "ESMC_600M_FT_SC2_99clus", "ESM2_650M_FT", "ESM2_3B_OG"
MODEL_SELECTION = ["ESMC_600M_FT_SC2_99clus","ESMC_600M_OG"]

#MODEL_SELECTION = ["sarbeco_SC2_FT_ESM2_650M_2023","ESM2_650M_OG","ESM2_3B_OG"]

# TEST SETTINGS
TEST_MODE = False
TEST_MAX_RECORDS = 500

# =====================================================================
# II. DATA PATHS AND LIMITS
# =====================================================================

# PLM sequence length limits
PLM_MAX_AA_LENGTH = 1024 
PLM_MAX_NT_LENGTH = None

# Focal reference sequence (nucleotide)
fasta_file = '/home3/oml4h/PLM_SARS-CoV-2/Sequences/SC2_month_snapshots/Jan25/rootseq_CM7YG5RN.fa'
base_lineage_id = '?'

# Guide CSV mapping months to FASTA paths (used if ANALYSIS_MODE == "MONTHLY_GUIDE")
POOLED_DIVERSITY_GUIDE = "/home3/oml4h/PLM_SARS-CoV-2/Sequences/SC2_month_snapshots/spike_month_file_guide.csv"

# Single diversity FASTA (used if ANALYSIS_MODE == "SINGLE_FASTA")
POOLED_DIVERSITY_FASTA = "/home3/oml4h/PLM_SARS-CoV-2/Sequences/SC2_month_snapshots/spike_2025-06_aa.fa"

# Filtering
FILTER_FIXED_MUTATIONS = False

# =====================================================================
# III. MODEL PROFILES (Definitions)
# =====================================================================
MODEL_PROFILES = {
    "ESMC_600M_OG": {
        "tag": "ESMC_600M_OG",
        "base_model": "esm-c600m",
        "layer": 36,
        "checkpoint_dir": None,
        "force_recompute": False,
    },
    "ESMC_600M_FT_SC2_99clus": {
        "tag": "magma_ESMC_600M_99_95perc",
        "base_model": "esm-c600m",
        "layer": 36,
        "checkpoint_dir": "/home3/oml4h/my_SC2_finetunes/magma_ESMC_600M_99_95perc/",
        "force_recompute": False,
    },
    
    "ESM2_650M_FT_SC2_99clus": {
        "tag": "sarbeco_SC2_FT_ESM2_650M_2023",
        "base_model": "esm2_t33_650M_UR50D",
        "layer": 33,
        "checkpoint_dir": "/home3/oml4h/my_SC2_finetunes/magma_esm-2_650M_95perc/SC2_ft_mod_2023_650M/final_checkpoint/",
        "force_recompute": False,
    },
    "ESM2_650M_OG": {
        "tag": "OG_esm2_t33_650M_UR50D",
        "base_model": "esm2_t33_650M_UR50D",
        "layer": 33,
        "checkpoint_dir": None,
        "force_recompute": False,
    },
    "ESM2_3B_OG": {
        "tag": "OG_esm2_t36_3B_UR50D",
        "base_model": "esm2_t36_3B_UR50D",
        "layer": 36,
        "checkpoint_dir": None,
        "force_recompute": False,
    }
}

# ---------------------------------------------------------------------
# Apply selection and build MODEL_RUNS
# ---------------------------------------------------------------------
MODEL_RUNS = []
for choice in MODEL_SELECTION:
    if choice not in MODEL_PROFILES:
        print(f"Warning: Model Choice '{choice}' not found in MODEL_PROFILES. Skipping.")
        continue
    prof = MODEL_PROFILES[choice]
    MODEL_RUNS.append({
        "tag": prof["tag"],
        "mode": "finetuned" if prof.get("checkpoint_dir") else "raw",
        "base_model": prof["base_model"],
        "layer": prof["layer"],
        "checkpoint_dir": prof.get("checkpoint_dir"),
        "force_recompute": prof.get("force_recompute", False),
        "enabled": True,
    })

# Use the first selected model as the primary for shared matrix naming/paths
primary_profile = MODEL_PROFILES[MODEL_SELECTION[0]] if MODEL_SELECTION else {}
PLM_MODEL_TAG = primary_profile.get("tag", "no_model")
PLM_BASE_MODEL = primary_profile.get("base_model", "")
PLM_MODEL_LAYER = primary_profile.get("layer", 0)
PLM_CHECKPOINT_DIR = primary_profile.get("checkpoint_dir")
FORCE_RECOMPUTE_PLM_MATRIX = primary_profile.get("force_recompute", False)

# Global in-memory cache for PLM probability matrices to avoid redundant computation
# Key: (model_tag, reference_protein_sequence)
# Value: result dictionary from get_mutation_prob_matrix
PLM_MATRIX_CACHE = {}


# --- Pooled Panel Runtime Constants ---
ALPHA_GRID = np.round(np.arange(-1.0, 1.01, 0.1), 2)
# PSEUDOCOUNT is dynamically calculated during panel evaluation based on n_seq
ALPHA_SWEEP_PARALLEL = True
ALPHA_SWEEP_MIN_GRID = 8
ALPHA_SWEEP_MAX_WORKERS = None
METHOD2_SCATTER_ALPHAS = [-1.0, 0.0, 1.0]
METHOD2_SCATTER_MAX_POINTS = 200000

IGNORE_ALIGNMENT_CHARS = {"-", "*", "."}

# --- Main Execution Functions ---

# Update PLM_MAX_AA_LENGTH for ESMC if selected

if "esmc" in PLM_BASE_MODEL.lower() or "esm-c" in PLM_BASE_MODEL.lower():
    PLM_MAX_AA_LENGTH = 2048
    print(f"ESM-C model detected. Setting PLM_MAX_AA_LENGTH to {PLM_MAX_AA_LENGTH}")

# Pooled Population Settings
RUN_POOLED_PANEL = True
POOLED_POPULATION_LABEL = os.path.splitext(os.path.basename(fasta_file))[0]
NUCLEOTIDE_MUTATION_MODEL = "SC2"

# Comparison sequence settings for plots
OBSERVED_MUTATION_FASTA = POOLED_DIVERSITY_FASTA if ANALYSIS_MODE == "SINGLE_FASTA" else None
OBSERVED_MUTATION_SEQUENCE_ID = None
OBSERVED_MUTATION_SELECTION = "last" 

nuc_sequences = list(SeqIO.parse(fasta_file, "fasta"))
seq_keys = [record.id for record in nuc_sequences]

base_lineage_index = seq_keys.index(base_lineage_id) if base_lineage_id in seq_keys else 0
print("Base lineage index:", base_lineage_index)
# translate to protein and confirm matches header of probasbility matrix
protein_sequences = [record.seq.translate(to_stop=True) for record in nuc_sequences]
# get base sequence
base_sequence = protein_sequences[base_lineage_index]
print("Base sequence:", base_sequence)

# create PLM input fasta using configurable ESM length limits
plm_max_nt_length = _resolve_plm_max_nt_length(PLM_MAX_AA_LENGTH, PLM_MAX_NT_LENGTH)
if plm_max_nt_length is None:
    cut_fasta = nuc_sequences[base_lineage_index]
else:
    cut_fasta = nuc_sequences[base_lineage_index][:plm_max_nt_length]

plm_trim_tag = (
    f"aa{PLM_MAX_AA_LENGTH}_nt{plm_max_nt_length}"
    if plm_max_nt_length is not None else "full_length"
)

print(
    f"PLM trimming config: max_aa={PLM_MAX_AA_LENGTH}, max_nt={PLM_MAX_NT_LENGTH}, "
    f"effective_nt_limit={plm_max_nt_length}"
)

# save file
cut_fasta_file = f'{fasta_file[:-3]}_{plm_trim_tag}_cut.fasta'
SeqIO.write(cut_fasta, cut_fasta_file, "fasta")
#translate to protein
cut_protein = cut_fasta.translate(to_stop=True)
cut_protein_file = f'{fasta_file[:-3]}_{plm_trim_tag}_cut_protein.fasta'
SeqIO.write(cut_protein, cut_protein_file, "fasta")

trimmed_base_sequence = str(cut_protein.seq) if hasattr(cut_protein, "seq") else str(cut_protein)
print(f"Full protein length: {len(str(base_sequence))}")
print(f"Trimmed PLM protein length: {len(trimmed_base_sequence)}")

probability_matrix_file = f'{fasta_file[:-3]}_{PLM_MODEL_TAG}_{plm_trim_tag}_prot_probability_matrix.csv'

# outdir calculation
outdir = fasta_file.rsplit('/', 1)[0] + '/' + PLM_MODEL_TAG + '/mut_access_calcs'
POOLED_REFERENCE_FASTA = cut_fasta_file
POOLED_PANEL_OUTDIR = os.path.join(outdir, "pooled_panel")

os.makedirs(outdir, exist_ok=True)
global key_matrix_dir
key_matrix_dir = os.path.join(outdir, "key_probability_matrices")
os.makedirs(key_matrix_dir, exist_ok=True)


# --- Function Definitions ---

reference_protein_for_plm = str(cut_protein.seq) if hasattr(cut_protein, "seq") else str(cut_protein)
_ensure_plm_probability_matrix(
        reference_protein=reference_protein_for_plm,
        output_path=probability_matrix_file,
        model_tag=PLM_MODEL_TAG,
        checkpoint_dir=PLM_CHECKPOINT_DIR,
        base_model=PLM_BASE_MODEL,
        model_layer=PLM_MODEL_LAYER,
        force_recompute=FORCE_RECOMPUTE_PLM_MATRIX,
        cache=PLM_MATRIX_CACHE
    )


# import without header, have it as it's own row
probability_matrix=load_plm_probability_matrix(probability_matrix_file)
print("Probability matrix shape:", probability_matrix.shape)
_save_key_matrix(probability_matrix, "plm_probability_matrix_raw_with_header_row.csv", key_matrix_dir)

print(probability_matrix.iloc[0,1:20])

# %%
print(19*probability_matrix.shape[1])
# %%
# compare sequences match
# convert prob matrix header to a string
prob_matrix_seq="".join(seq_chars for seq_chars in probability_matrix.iloc[0,:])
prob_matrix_seq = prob_matrix_seq[:len(reference_protein_for_plm)]

print("Probability matrix sequence preview:", prob_matrix_seq[:20])
print(f"Probability matrix sequence length: {len(prob_matrix_seq)}")

# assert str(base_sequence)==prob_matrix_seq, "Sequences do not match!"
#find mismatches:
bs = str(reference_protein_for_plm)

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

if NUCLEOTIDE_MUTATION_MODEL not in TRANSITION_MATRICES:
    raise ValueError(
        f"Unsupported NUCLEOTIDE_MUTATION_MODEL: {NUCLEOTIDE_MUTATION_MODEL}. "
        f"Available options: {sorted(TRANSITION_MATRICES)}"
    )

active_transition_config = TRANSITION_MATRICES[NUCLEOTIDE_MUTATION_MODEL]
active_transition_name = active_transition_config["display_name"]
active_transition_tag = active_transition_config["tag"]
active_transitions = active_transition_config["matrix"].astype(float).copy()

# ------------------------------------------------------------------
# VERIFICATION PRINT
# ------------------------------------------------------------------
print(f"Active nucleotide mutation model: {active_transition_name}")
print(f"Transition matrix shape: {active_transitions.shape}")
print("Transition matrix:\n", active_transitions)
# %%
# Heatmaps for selected transition matrix (4x4)
active_transition_heat = active_transitions.copy()
np.fill_diagonal(active_transition_heat, np.nan)

cmap_linear = plt.get_cmap("viridis").copy()
cmap_linear.set_bad(color="white")

plt.figure(figsize=(5, 4.5))
ax = sns.heatmap(
    active_transition_heat,
    annot=True,
    fmt=".1e",
    cmap=cmap_linear,
    xticklabels=bases,
    yticklabels=bases,
    cbar_kws={"label": "Mutation rate"}
)
ax.set_xlabel("To")
ax.set_ylabel("From")
ax.set_title(f"{active_transition_name} Nucleotide Transition Matrix (Diagonal Masked)")
plt.tight_layout()
plt.savefig(f"{outdir}/{active_transition_tag}_transition_matrix_heatmap.png", dpi=300)
# plt.show()

# Log-scaled palette (diagonals masked to white)
nonzero_vals = active_transition_heat[~np.isnan(active_transition_heat) & (active_transition_heat > 0)]
vmin = nonzero_vals.min() if nonzero_vals.size else 1e-8
vmax = nonzero_vals.max() if nonzero_vals.size else 1.0

cmap_log = plt.get_cmap("magma").copy()
cmap_log.set_bad(color="white")

plt.figure(figsize=(5, 4.5))
ax = sns.heatmap(
    active_transition_heat,
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
ax.set_title(f"{active_transition_name} Nucleotide Transition Matrix (Log Scale)")
plt.tight_layout()
plt.savefig(f"{outdir}/{active_transition_tag}_transition_matrix_heatmap_log.png", dpi=300)
# plt.show()
# %%

# %%

# Create a 64 x 64 codon-to-codon mutation matrix from the selected nucleotide
# mutation model by assuming independence across the three codon positions.

# Calculate probabilities including 'staying same' for diagonal
# Diagonal is the probability of no nucleotide change.
active_transition_probs = active_transitions.copy()
for i in range(4):
    active_transition_probs[i, i] = 1.0 - np.sum(active_transitions[i, :])

_save_key_matrix(
    pd.DataFrame(active_transitions, index=bases, columns=bases),
    f"{active_transition_tag}_nucleotide_transition_rates.csv",
    key_matrix_dir
)
_save_key_matrix(
    pd.DataFrame(active_transition_probs, index=bases, columns=bases),
    f"{active_transition_tag}_nucleotide_transition_probabilities.csv",
    key_matrix_dir
)

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
            prob *= active_transition_probs[idx_from, idx_to]
        codon_mutation_matrix[i, j] = prob

# Convert to DataFrame
codon_mutation_df = pd.DataFrame(codon_mutation_matrix, index=codons, columns=codons)
codon_mutation_df.to_csv(f"{outdir}/codon_mutation_matrix.csv")
_save_key_matrix(codon_mutation_df, f"codon_mutation_matrix_{active_transition_tag}.csv", key_matrix_dir)

print(f"Codon Mutation Matrix ({active_transition_name}) shape:", codon_mutation_df.shape)
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

_save_key_matrix(codon_to_aa_matrix, f"codon_to_aa_matrix_{active_transition_tag}_with_stop.csv", key_matrix_dir)

plt.figure(figsize=(18, 10))
ax = sns.heatmap(
    codon_to_aa_matrix.T,
    cmap="viridis",
    cbar_kws={"label": "Codon → Amino Acid probability"}
)
ax.set_xlabel("Starting Codon", fontsize=16)
ax.set_ylabel("Target Amino Acid (including stop '*')", fontsize=16)
ax.set_title(f"{active_transition_name} Codon → Amino Acid Probability Matrix (64×21)", fontsize=20)
ax.tick_params(axis="both", labelsize=12)
ytick_labels = ["*" if lab == "-" else lab for lab in target_aas]
ax.set_yticklabels(ytick_labels, rotation=279)
plt.tight_layout()
plt.savefig(f"{outdir}/codon_to_amino_acid_matrix_heatmap.png", dpi=300)
# plt.show()

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

test_codons = ["AAA", "ATG", "TGG", "TAA"]
test_aas = ["A", "G", "L", "*", "W"]

print("\nManual codon→AA checks (raw vs matrix):")
for codon_from in test_codons:
    if codon_from not in codon_to_aa_matrix.index:
        continue
    for aa in test_aas:
        if aa not in codon_to_aa_matrix.columns:
            continue
        raw_val = _raw_codon_to_aa_prob(codon_from, aa, codon_mutation_df, aa_to_codons_all)
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
_save_key_matrix(codon_to_aa_20, f"codon_to_aa_matrix_{active_transition_tag}_aa20.csv", key_matrix_dir)





aa20_transition_avg, codon_to_aa_20_reconstructed, source_codon_counts = _build_aa20_average_and_reconstruction(codon_to_aa_20, aa20, ordered_codons, genetic_code)
selected_self_metrics = _flattened_fit_metrics(codon_to_aa_20, codon_to_aa_20_reconstructed)

print(f"\n--- Codon->AA compression analysis ({active_transition_name}-specific 64x20 -> 20x20) ---")
print("Methodological setup:")
print("  1) Construct target matrix Y (selected-model 64x20): rows=source codons, columns=target amino acids (excluding stop).")
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
print(f"Finite entries compared: {selected_self_metrics['n_entries']}")
print(f"Total variance in original 64x20 table: {selected_self_metrics['total_var']:.6e}")
print(f"Residual variance after 20x20 compression: {selected_self_metrics['residual_var']:.6e}")
print(f"Variation retained by 20x20 aggregation: {selected_self_metrics['retained_pct']:.2f}%")
print(f"Flattened correlation (original vs reconstructed): r={selected_self_metrics['corr']:.4f}")
print(f"RMSE between original and reconstructed entries: {selected_self_metrics['rmse']:.6e}")
print(f"MAE between original and reconstructed entries: {selected_self_metrics['mae']:.6e}")

# Generic baseline model: Kimura-80-like nucleotide process
# - AT content ~50% (uniform A/C/G/T frequencies in this simple implementation)
# - transition:transversion bias = 2:1
transition_pairs = {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}
row_total_mut_rate = float(np.mean(np.sum(active_transitions, axis=1)))

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
aa20_transition_avg_k80, codon_to_aa_20_reconstructed_k80, _ = _build_aa20_average_and_reconstruction(codon_to_aa_20_k80, aa20, ordered_codons, genetic_code)

generic_self_metrics = _flattened_fit_metrics(codon_to_aa_20_k80, codon_to_aa_20_reconstructed_k80)
generic_to_selected_metrics = _flattened_fit_metrics(codon_to_aa_20, codon_to_aa_20_reconstructed_k80)
aa20_selected_vs_k80_metrics = _flattened_fit_metrics(aa20_transition_avg, aa20_transition_avg_k80)

if np.isfinite(generic_to_selected_metrics["residual_var"]) and generic_to_selected_metrics["residual_var"] > 0:
    error_reduction_pct = 100.0 * (
        (generic_to_selected_metrics["residual_var"] - selected_self_metrics["residual_var"])
        / generic_to_selected_metrics["residual_var"]
    )
else:
    error_reduction_pct = np.nan

print(f"\n--- Generic Kimura-80 baseline vs {active_transition_name}-specific matrix ---")
print("Methodological setup:")
print("  1) Build generic nucleotide model Q_K80 with AT=50% and Ti:Tv=2:1.")
print(f"  2) Match its total per-row mutation load to the mean {active_transition_name} row-sum for scale comparability.")
print("  3) Generate generic codon->codon probabilities (64x64) by independent per-site multiplication.")
print("  4) Aggregate to generic codon->AA matrix Y_K80 (64x20), then compress to A_K80 (20x20), reconstruct Y_hat_K80.")
print(f"  5) Evaluate two tests against the {active_transition_tag} target matrix Y_target:")
print("     - within-model compression test: Y_K80 vs Y_hat_K80")
print("     - cross-model approximation test: Y_target vs Y_hat_K80")
print(f"  6) Quantify gain from {active_transition_name}-specific model using residual-variance reduction:")
print("     gain_percent = 100 * (resid_var(Y_target vs Y_hat_K80) - resid_var(Y_target vs Y_hat_target)) / resid_var(Y_target vs Y_hat_K80)")
print(
    "Generic model retained variation in its own 64x20->20x20 compression: "
    f"{generic_self_metrics['retained_pct']:.2f}%"
)
print(
    f"How much of {active_transition_name} 64x20 variation is captured by generic 20x20 reconstruction: "
    f"{generic_to_selected_metrics['retained_pct']:.2f}%"
)
print(
    f"{active_transition_name}-specific 20x20 error reduction vs generic 20x20: "
    f"{error_reduction_pct:.2f}%"
)
print(
    f"{active_transition_name} vs generic 20x20 matrix similarity (flattened r): "
    f"{aa20_selected_vs_k80_metrics['corr']:.4f}"
)

compression_summary = pd.DataFrame([
    {
        "comparison": f"{active_transition_tag}64_to_{active_transition_tag}20_reconstruction",
        "finite_entries_compared": selected_self_metrics["n_entries"],
        "total_variance": selected_self_metrics["total_var"],
        "residual_variance": selected_self_metrics["residual_var"],
        "retained_variation_percent": selected_self_metrics["retained_pct"],
        "flattened_correlation_r": selected_self_metrics["corr"],
        "rmse": selected_self_metrics["rmse"],
        "mae": selected_self_metrics["mae"],
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
        "comparison": f"{active_transition_tag}64_to_k80_20_reconstruction",
        "finite_entries_compared": generic_to_selected_metrics["n_entries"],
        "total_variance": generic_to_selected_metrics["total_var"],
        "residual_variance": generic_to_selected_metrics["residual_var"],
        "retained_variation_percent": generic_to_selected_metrics["retained_pct"],
        "flattened_correlation_r": generic_to_selected_metrics["corr"],
        "rmse": generic_to_selected_metrics["rmse"],
        "mae": generic_to_selected_metrics["mae"],
    },
    {
        "comparison": f"{active_transition_tag}20_vs_k80_20",
        "finite_entries_compared": aa20_selected_vs_k80_metrics["n_entries"],
        "total_variance": aa20_selected_vs_k80_metrics["total_var"],
        "residual_variance": aa20_selected_vs_k80_metrics["residual_var"],
        "retained_variation_percent": aa20_selected_vs_k80_metrics["retained_pct"],
        "flattened_correlation_r": aa20_selected_vs_k80_metrics["corr"],
        "rmse": aa20_selected_vs_k80_metrics["rmse"],
        "mae": aa20_selected_vs_k80_metrics["mae"],
    },
])

gain_summary = pd.DataFrame([
    {
        f"{active_transition_tag}_specific_error_reduction_vs_generic_percent": error_reduction_pct,
        "generic_model_assumption": "AT50_TiTv2to1",
        "generic_row_total_mutation_rate": row_total_mut_rate,
    }
])

compression_summary.to_csv(f"{outdir}/codon_to_aa_compression_summary.csv", index=False)
gain_summary.to_csv(f"{outdir}/{active_transition_tag}_vs_k80_gain_summary.csv", index=False)

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
_save_key_matrix(plm_matrix, "plm_probability_matrix_numeric.csv", key_matrix_dir)

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
    # A mismatch here MUST be a hard fail to prevent nonsense matrix combinations.
    if isinstance(expected_aa, str) and len(expected_aa) == 1:
        if translated_aa != expected_aa:
            raise ValueError(
                f"CRITICAL ALIGNMENT ERROR at col {col_idx} (Pos {seq_idx+1}): "
                f"Reference codon {current_codon} translates to '{translated_aa}', "
                f"but PLM matrix expects '{expected_aa}'. "
                "The focal sequence and PLM matrix have decoupled coordinates."
            )
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


validate_mutational_matrix(mutational_prob_matrix)
mutational_prob_matrix.to_csv(f"{outdir}/mutational_prob_matrix.csv")
_save_key_matrix(mutational_prob_matrix, "mutational_probability_matrix_from_codon_model.csv", key_matrix_dir)

# %%
# Calculate Combined Matrices
# 1. P_plm * P_mut (Element-wise multiplication)
combined_prob_matrix = plm_matrix * mutational_prob_matrix

# 2. P_plm * sqrt(P_mut)
combined_prob_sqrt_matrix = plm_matrix * np.sqrt(mutational_prob_matrix)

combined_prob_matrix.to_csv(f"{outdir}/combined_prob_matrix.csv")
combined_prob_sqrt_matrix.to_csv(f"{outdir}/combined_prob_sqrt_matrix.csv")
_save_key_matrix(combined_prob_matrix, "combined_probability_matrix_plm_times_mut.csv", key_matrix_dir)
_save_key_matrix(combined_prob_sqrt_matrix, "combined_probability_matrix_plm_times_sqrt_mut.csv", key_matrix_dir)

print("Combined Matrix Shape:", combined_prob_matrix.shape)
print("Combined Sqrt Matrix Shape:", combined_prob_sqrt_matrix.shape)

# Example output
print("\nExample (Top 5 rows, first 5 cols) of Combined Matrix:")
print(combined_prob_matrix.iloc[:5, :5])

# %%
# Rank Analysis and Plotting

# 1. Identify observed mutations relative to the focal/root sequence.
# This comparison sequence is only used for highlighting observed changes in the
# ranked plots. It does not change the PLM probabilities computed above.
#
# IMPORTANT: use `trimmed_base_sequence` (PLM-trimmed focal) as the reference so
# that mutation positions align with the probability matrix coordinate space.
# Observed mutations beyond the trimmed length are dropped because the PLM matrix
# does not cover those positions.
comparison_sequence_id, target_protein_seq = _load_comparison_protein_sequence(
    OBSERVED_MUTATION_FASTA,
    sequence_id=OBSERVED_MUTATION_SEQUENCE_ID,
    selection=OBSERVED_MUTATION_SELECTION,
)
# Use the PLM-trimmed focal sequence so positions align with the probability matrix.
focal_protein_seq = trimmed_base_sequence

print(f"Focal sequence ID: {seq_keys[base_lineage_index]}")
print(f"Focal (PLM-trimmed) length: {len(focal_protein_seq)}")
print(f"Observed-mutation comparison FASTA: {OBSERVED_MUTATION_FASTA}")
print(f"Observed-mutation comparison selection: {OBSERVED_MUTATION_SELECTION}")
print(f"Observed-mutation comparison sequence ID: {comparison_sequence_id}")

# Find differences via pairwise alignment to correctly handle indels.
# A naive zip comparison causes a frame-shift cascade when sequences differ in
# length, turning a single indel into hundreds of apparent mutations.
observed_mutations = []  # List of (0-based focal index, target_aa)
if target_protein_seq is not None:
    from Bio import Align as _BioAlign

    _aligner = _BioAlign.PairwiseAligner()
    _aligner.mode = "global"
    _aligner.match_score = 2
    _aligner.mismatch_score = -1
    _aligner.open_gap_score = -10
    _aligner.extend_gap_score = -0.5

    # Align full-length focal against full-length target.
    _full_focal = str(base_sequence)  # full-length for alignment
    _full_target = target_protein_seq
    _aln = next(iter(_aligner.align(_full_focal, _full_target)))

    # Walk alignment blocks and collect substitutions mapped to focal positions.
    _n_subs = 0
    _n_indel_cols = 0
    _focal_pos = 0
    _target_pos = 0

    _focal_aligned = []
    _target_aligned = []

    _curr_focal_idx = 0
    _curr_target_idx = 0

    # _aln.aligned is a tuple of (focal_ranges, target_ranges)
    # where focal_ranges = [[start, end], [start, end], ...]
    _focal_ranges = _aln.aligned[0]
    _target_ranges = _aln.aligned[1]

    for (f_start, f_end), (t_start, t_end) in zip(_focal_ranges, _target_ranges):
        # Handle indels before this aligned block
        if _curr_focal_idx < f_start:
            num_gaps = f_start - _curr_focal_idx
            _focal_aligned.append(_full_focal[_curr_focal_idx:f_start])
            _target_aligned.append('-' * num_gaps)
            _curr_focal_idx = f_start
        if _curr_target_idx < t_start:
            num_gaps = t_start - _curr_target_idx
            _focal_aligned.append('-' * num_gaps)
            _target_aligned.append(_full_target[_curr_target_idx:t_start])
            _curr_target_idx = t_start
        
        # Aligned block
        _focal_aligned.append(_full_focal[f_start:f_end])
        _target_aligned.append(_full_target[t_start:t_end])
        _curr_focal_idx = f_end
        _curr_target_idx = t_end
    
    # Handle indels at the end
    if _curr_focal_idx < len(_full_focal):
        num_gaps = len(_full_focal) - _curr_focal_idx
        _focal_aligned.append(_full_focal[_curr_focal_idx:])
        _target_aligned.append('-' * num_gaps)
    if _curr_target_idx < len(_full_target):
        num_gaps = len(_full_target) - _curr_target_idx
        _focal_aligned.append('-' * num_gaps)
        _target_aligned.append(_full_target[_curr_target_idx:])

    _aln_focal_str = "".join(_focal_aligned)
    _aln_target_str = "".join(_target_aligned)

    for _af, _at in zip(_aln_focal_str, _aln_target_str):
        if _af == "-" and _at == "-":
            continue  # shouldn't happen
        elif _af == "-":
            # Insertion in target (no focal position consumed)
            _n_indel_cols += 1
        elif _at == "-":
            # Deletion in target (focal position consumed, no target residue)
            _n_indel_cols += 1
            _focal_pos += 1
        else:
            # Aligned pair (substitution or match)
            if _af != _at:
                # Only record if within the PLM-trimmed focal region
                if _focal_pos < len(focal_protein_seq):
                    observed_mutations.append((_focal_pos, _at))
                    _n_subs += 1
            _focal_pos += 1
            _target_pos += 1

    print(
        f"Comparing Focal ({len(_full_focal)}) vs Target ({len(_full_target)}) "
        f"[aligned: {len(_aln_focal_str)} cols]"
    )
    print(
        f"Alignment summary: {_n_subs} substitutions, {_n_indel_cols} indel column(s). "
        f"Mutations within PLM-trimmed region ({len(focal_protein_seq)} aa): {len(observed_mutations)}."
    )
else:
    print("No comparison sequence loaded; ranked plots will show only the background mutation scores.")

print(f"Found {len(observed_mutations)} mutations.")



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
plt.title(f'Histogram of Mutational Probability Matrix Values ({PLM_MODEL_TAG})')
plt.xlabel('Mutational Probability')
plt.savefig(f"{outdir}/histogram_mutational_prob_{PLM_MODEL_TAG}.png")

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
    if not obs_df.empty:
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
            pos_label = pos_idx + 1
            label = f"{ref_aa}{pos_label}{mut_aa}_R{rank_val}"
            
            texts.append(ax.text(row['Rank'], row['log10Probability'], label, fontsize=8))
        
        # Use adjust_text to repel labels
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='grey', lw=0.5), ax=ax)

    ax.set_title(f"{name} ({PLM_MODEL_TAG})")
    ax.set_xlabel('Rank (1 = Highest Prob)')
    ax.set_ylabel('log10(Probability)')
    #ax.set_xscale('log') # User requested log scale implicitly via "rank each possible mutation" usually implies log-log or semi-log
    #ax.set_yscale('log')
    ax.grid(True, which="both", ls="-", alpha=0.2)

plt.tight_layout()
plt.savefig(f"{outdir}/ranked_mutations_{PLM_MODEL_TAG}.png")
# plt.show()


# %%
# correlate the mutation vs plm probabilities plot them and give pearson and spearman ranks

#need to make diagonals NaN in both matrices to avoid self-mutation bias- create a copy first
plm_matrix_no_diag = plm_matrix.copy()
mutational_prob_matrix_no_diag = mutational_prob_matrix.copy()

for j in range(plm_matrix_no_diag.shape[1]):
    if j >= len(focal_protein_seq):
        break
    ref_aa = focal_protein_seq[j]
    try:
        row_idx = plm_matrix_no_diag.index.get_loc(ref_aa)
        plm_matrix_no_diag.iloc[row_idx, j] = np.nan
        mutational_prob_matrix_no_diag.iloc[row_idx, j] = np.nan    
    except KeyError:
        pass

# Flatten the matrices to 1D arrays
plm_flat = plm_matrix_no_diag.values.flatten()
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
plt.title(f'PLM ({PLM_MODEL_TAG}) Probability vs Mutational Probability (log10 scale)\n spearman: {spearman_corr:.3f} (p={p_s:.2e}), pearson: {pearson_corr:.3f} (p={p_p:.2e})')
plt.xlabel('log10(PLM Probability)')
plt.ylabel('log10(Mutational Probability)')
plt.savefig(f"{outdir}/plm_vs_mut_correlation_{PLM_MODEL_TAG}.png")

# which ones have values greater than0.1 in either?
high_plm = plm_flat > 0.1
high_mut = mut_flat > 0.1
high_either = high_plm | high_mut



# %%
# Investigation of "High" Mutational Probabilities
# The user noticed "odd ones" with high probability.
# We look for mutations that have a probability > max(single_nucleotide_mutation_rate).
# This implies summations of multiple paths or multiple synonymous target codons.

max_raw_prob = np.max(active_transitions) # Max off-diagonal element since diagonal was 0 in definition
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
                         tr_prob = active_transition_probs[bases.index(ref_c[k]), bases.index(tc[k])]
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
# # Pooled mutation accessibility vs PLM probability panel
# %%
import glob
import re
from pathlib import Path
from Bio import pairwise2
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from Functions_HuggingFace import (
    _extract_corr_pvalue,
    _is_probably_nucleotide,
    _tag_output_name,
    _translate_nt_to_protein,
    build_reference_to_alignment_column_map,
    compute_lineage_mutation_profile,
    compute_observed_diversity_profile_fast,
    evaluate_alpha_sweep,
)




RUN_MODE_TAG = "test" if TEST_MODE else "full"
DIVERSITY_PATTERN_TAG = _clean_pattern_tag(Path(POOLED_DIVERSITY_FASTA).stem)
OUTPUT_TAG = f"{RUN_MODE_TAG}_{_safe_label(POOLED_POPULATION_LABEL)}_{DIVERSITY_PATTERN_TAG}"




# %%
if RUN_POOLED_PANEL:
    os.makedirs(POOLED_PANEL_OUTDIR, exist_ok=True)

    if "codon_mutation_df" not in globals() or "aa_to_codons" not in globals():
        # (Assuming these are global or passed in. In this case, we'll keep them global for simplicity)
        pass

    print("Loading pooled focal reference...")
    pooled_ref = _load_single_focal_reference(POOLED_REFERENCE_FASTA, POOLED_POPULATION_LABEL)
    reference_protein = pooled_ref["protein"]
    reference_nt = pooled_ref["nucleotide"]
    if not reference_protein:
        raise ValueError("Empty translated protein sequence in pooled focal FASTA.")

    # Determine which diversity FASTA(s) and their references to process
    diversity_targets = []
    if ANALYSIS_MODE == "MONTHLY_GUIDE" and POOLED_DIVERSITY_GUIDE and os.path.exists(POOLED_DIVERSITY_GUIDE):
        print(f"Loading diversity guide: {POOLED_DIVERSITY_GUIDE}")
        with open(POOLED_DIVERSITY_GUIDE, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row.get('month') or row.get('label')
                path = row.get('fasta') or row.get('path')
                # Check for lineage-specific reference, fallback to global POOLED_REFERENCE_FASTA
                ref_path = row.get('reference') or POOLED_REFERENCE_FASTA
                if label and path:
                    diversity_targets.append((label, path, ref_path))
    elif ANALYSIS_MODE == "SINGLE_FASTA":
        diversity_targets.append((POOLED_POPULATION_LABEL, POOLED_DIVERSITY_FASTA, POOLED_REFERENCE_FASTA))
    else:
        print(f"Warning: Unsupported ANALYSIS_MODE or missing guide: {ANALYSIS_MODE}")

    lineage_cache = {}
    # (Mutational profile will be computed inside the loop per reference)

    for label, fasta_path, ref_path in diversity_targets:
        if not os.path.exists(fasta_path):
            print(f"Warning: Diversity FASTA does not exist: {fasta_path}. Skipping.")
            continue
            
        print(f"Processing {label} diversity from {fasta_path} using reference {ref_path}...")
        
        # Load the focal reference for THIS lineage/month
        try:
             curr_ref_rec = _load_single_focal_reference(ref_path, label)
             full_ref_nt = curr_ref_rec["nucleotide"]
             full_ref_protein = curr_ref_rec["protein"]
        except Exception as exc:
             print(f"Error loading reference {ref_path} for {label}: {exc}")
             continue

        # Trim for PLM if necessary, ensuring codon-awareness
        plm_max_nt = _resolve_plm_max_nt_length(PLM_MAX_AA_LENGTH, PLM_MAX_NT_LENGTH)
        if plm_max_nt is not None and len(full_ref_nt) > plm_max_nt:
            plm_ref_nt = full_ref_nt[:plm_max_nt]
            plm_ref_protein = _translate_nt_to_protein(plm_ref_nt)
            print(f"  Trimmed PLM reference to {len(plm_ref_protein)} aa ({plm_max_nt} nt)")
        else:
            plm_ref_protein = full_ref_protein
            
        # Build coordinate map: PLM 0-based index -> Full 0-based index
        # Since it's a prefix, it's just identity for the trimmed length.
        coord_map = {i: i for i in range(len(plm_ref_protein))}

        # Compute mutation profile for the FULL reference
        curr_mut_profile = compute_lineage_mutation_profile(
            full_ref_nt, full_ref_protein, aa_to_codons, codon_mutation_df
        )

        records = list(SeqIO.parse(fasta_path, "fasta"))
        if len(records) == 0:
            print(f"Warning: No records found in {fasta_path}. Skipping.")
            continue

        if TEST_MODE:
            records = records[:TEST_MAX_RECORDS]
            
        # Ensure records are protein for the alignment comparison
        processed_records = []
        for rec in records:
            if _is_probably_nucleotide_sequence(str(rec.seq)):
                rec.seq = rec.seq.translate(to_stop=True)
            processed_records.append(rec)

        lineage_key = _safe_label(label)
        ref_to_aln_col, aln_len, matched_pairs = build_reference_to_alignment_column_map(
            full_ref_protein, processed_records, aa_to_codons, IGNORE_ALIGNMENT_CHARS
        )
        obs_freq, obs_depth, diversity_stats = compute_observed_diversity_profile_fast(
            processed_records,
            full_ref_protein,
            ref_to_aln_col,
            aln_len,
            aa_to_codons,
            IGNORE_ALIGNMENT_CHARS,
        )

        print(
            f"  [ALIGNMENT DIFF] {label}: any_diff={diversity_stats['differing_sites']}, "
            f"fixed_diff={diversity_stats['fixed_differing_sites']} "
            f"/ compared={diversity_stats['compared_sites']} "
            f"(mapped_ref={diversity_stats['mapped_sites']}/{len(full_ref_protein)})"
        )

        lineage_cache[label] = {
            "lineage_key": lineage_key,
            "records": processed_records,
            "full_ref_protein": full_ref_protein,
            "plm_ref_protein": plm_ref_protein,
            "coord_map": coord_map,
            "mut_profile": curr_mut_profile,
            "obs_freq": obs_freq,
            "obs_depth": obs_depth,
            "ref_to_aln_col": ref_to_aln_col,
            "alignment_diff_stats": diversity_stats,
            "diversity_path": fasta_path,
            "diversity_tag": Path(fasta_path).stem,
        }

    all_alpha_frames = []
    model_status_rows = []
    per_lineage_best_rows = []

    cuda_available = torch.cuda.is_available()
    if GPU_REQUIRED and not cuda_available:
        raise RuntimeError(
            "GPU_REQUIRED is True but no CUDA GPU was found. "
            "ESM inference on CPU is discouraged due to extreme slowness. "
            "Set GPU_REQUIRED = False if you intentionally want to use CPU."
        )

    device = torch.device("cuda" if cuda_available else "cpu")
    print(f"Using device: {device}")

    for run_cfg in MODEL_RUNS:
        if not run_cfg.get("enabled", True):
            continue

        model_tag = run_cfg["tag"]
        model_outdir = os.path.join(POOLED_PANEL_OUTDIR, model_tag)
        os.makedirs(model_outdir, exist_ok=True)

        model = None
        alphabet = None
        batch_converter = None
        model_ready = False
        used_cached_plm = False
        model_load_attempted = True # will be set in logic
        model_load_failed_reason = ""
        model_runtime_failed = False
        model_runtime_failed_reason = ""

        combined_rows = []
        per_lineage_summaries = []

        for lineage, data in lineage_cache.items():
            plm_ref_protein = data["plm_ref_protein"]
            full_ref_protein = data["full_ref_protein"]
            coord_map = data["coord_map"]
            
            print(f"Processing lineage {lineage} with {model_tag}: n_seq={len(data['records'])}, plm_ref_len={len(plm_ref_protein)}, full_ref_len={len(full_ref_protein)}")

            plm_profile_path = os.path.join(
                model_outdir,
                _tag_output_name(f"{data['lineage_key']}_plm_probability_profile.csv", TEST_MODE, DIVERSITY_PATTERN_TAG, OUTPUT_TAG),
            )

            plm_matrix = None
            if os.path.exists(plm_profile_path) and not run_cfg.get("force_recompute", False):
                try:
                    plm_matrix = pd.read_csv(plm_profile_path, index_col=0)
                    used_cached_plm = True
                    print(f"Using existing PLM matrix from disk: {plm_profile_path}")
                except Exception as exc:
                    print(f"Failed to load existing PLM matrix for {lineage} ({plm_profile_path}): {exc}")

            if plm_matrix is None:
                cache_key = (model_tag, plm_ref_protein)
                plm_out = None
                
                if cache_key in PLM_MATRIX_CACHE and not run_cfg.get("force_recompute", False):
                    print(f"Using in-memory cached PLM matrix for {model_tag}")
                    plm_out = PLM_MATRIX_CACHE[cache_key]
                else:
                    if not model_ready and not model_load_attempted:
                        model_load_attempted = True
                        try:
                            print(f"\nLoading model config: {model_tag}")
                            model, device, batch_converter, alphabet = _load_plm_runtime(
                                run_cfg["base_model"], 
                                checkpoint_dir=run_cfg.get("checkpoint_dir")
                            )
                            model_ready = True
                        except Exception as exc:
                            model_load_failed_reason = str(exc)
                            print(f"Skipping PLM generation for {model_tag}: failed to load. Reason: {exc}")

                    if model_ready:
                        try:
                            plm_out = get_mutation_prob_matrix(
                                plm_ref_protein,
                                model,
                                run_cfg["layer"],
                                device,
                                batch_converter,
                                alphabet,
                            )
                            PLM_MATRIX_CACHE[cache_key] = plm_out
                        except Exception as exc:
                            model_runtime_failed = True
                            model_runtime_failed_reason = str(exc)
                            print(f"Skipping model {model_tag}: runtime failure for {lineage}. Reason: {exc}")
                            break

                if plm_out is not None:
                    plm_matrix = pd.DataFrame(
                        plm_out["mutation_matrix"],
                        index=plm_out["amino_acids"],
                        columns=plm_out["positions"],
                    )
                    plm_matrix.to_csv(plm_profile_path)

            if plm_matrix is None:
                continue

            # --- ROBUST COORDINATE MAPPING ---
            for pos_label in plm_matrix.columns:
                try:
                    pos_plm_1 = int(pos_label) # 1-based index in PLM reference
                except (TypeError, ValueError):
                    continue
                
                # Map PLM position to Full reference position
                pos_plm_0 = pos_plm_1 - 1
                if pos_plm_0 not in coord_map:
                    continue # Skip if not mapped
                
                pos_full_0 = coord_map[pos_plm_0]
                pos_full_1 = pos_full_0 + 1 # 1-based index in full reference
                
                ref_aa = plm_ref_protein[pos_plm_0]
                
                # Verify that this position exists in the mutational accessibility profile
                if pos_full_1 not in data["mut_profile"].columns:
                    continue

                for aa in plm_matrix.index:
                    if aa == ref_aa:
                        continue
                    if aa not in data["mut_profile"].index or aa not in data["obs_freq"].index:
                        continue

                    plm_prob = float(plm_matrix.loc[aa, pos_label])
                    mut_prob = float(data["mut_profile"].loc[aa, pos_full_1])
                    obs = float(data["obs_freq"].loc[aa, pos_full_1])
                    
                    if FILTER_FIXED_MUTATIONS and obs >= 1.0:
                        continue

                    combined_rows.append({
                        "model": model_tag,
                        "lineage": lineage,
                        "position": int(pos_full_1),
                        "ref_aa": ref_aa,
                        "aa": aa,
                        "plm_prob": plm_prob,
                        "mut_prob": mut_prob,
                        "obs_freq": obs,
                        "obs_present": 1 if obs > 0 else 0,
                        "depth": float(data["obs_depth"][pos_full_1]),
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
                "reference_length": len(data["full_ref_protein"]),
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
            os.path.join(model_outdir, _tag_output_name("pooled_combined_long_table.csv", TEST_MODE, DIVERSITY_PATTERN_TAG, OUTPUT_TAG)),
            index=False,
        )

        # Output specific PLM vs Mutation probability comparison for this model
        comparison_columns = ["position", "ref_aa", "aa", "plm_prob", "mut_prob"]
        if all(col in combined_df.columns for col in comparison_columns):
            comparison_df = combined_df[comparison_columns].drop_duplicates()
            comparison_df.to_csv(
                os.path.join(model_outdir, _tag_output_name("plm_vs_mut_prob.csv", TEST_MODE, DIVERSITY_PATTERN_TAG, OUTPUT_TAG)),
                index=False,
            )

            # Generate scatter plot for PLM vs Mut Prob
            try:
                plt.figure(figsize=(6, 5))
                # Add tiny pseudocount for log plotting
                mask = (comparison_df["plm_prob"] > 0) & (comparison_df["mut_prob"] > 0)
                plot_data = comparison_df[mask]
                
                if not plot_data.empty:
                    rho, pval = spearmanr(plot_data["plm_prob"], plot_data["mut_prob"])
                    
                    plt.scatter(plot_data["plm_prob"], plot_data["mut_prob"], alpha=0.3, s=10, edgecolors="none")
                    plt.xscale("log")
                    plt.yscale("log")
                    plt.xlabel("PLM Probability")
                    plt.ylabel("Mutation Probability (Codon Model)")
                    plt.title(f"{model_tag} Correlation\nSpearman rho={rho:.3f}")
                    plt.grid(True, which="both", ls="--", alpha=0.5)
                    
                    plot_path = os.path.join(model_outdir, _tag_output_name("plm_vs_mut_prob_scatter.png", TEST_MODE, DIVERSITY_PATTERN_TAG, OUTPUT_TAG))
                    plt.tight_layout()
                    plt.savefig(plot_path, dpi=300)
                    plt.close()
            except Exception as plot_exc:
                print(f"Warning: Failed to generate comparison plot for {model_tag}: {plot_exc}")

        lineage_meta_df = pd.DataFrame(per_lineage_summaries)
        lineage_meta_df.to_csv(
            os.path.join(model_outdir, _tag_output_name("pooled_panel_metadata.tsv", TEST_MODE, DIVERSITY_PATTERN_TAG, OUTPUT_TAG)),
            sep="\t",
            index=False,
        )

        use_parallel_alpha = ALPHA_SWEEP_PARALLEL and len(ALPHA_GRID) >= ALPHA_SWEEP_MIN_GRID
        if use_parallel_alpha:
            print(f"Running alpha sweep in parallel for {model_tag} (n_alpha={len(ALPHA_GRID)})")

        n_seq_pooled = int(lineage_meta_df["n_sequences"].max()) if not lineage_meta_df.empty else 1000
        dynamic_pseudocount = float(10 ** -round(np.log10(10 * n_seq_pooled)))
        print(f"Using dynamic pseudocount for obs_freq plotting: {dynamic_pseudocount:.1e} based on {n_seq_pooled} max sequences")
        
        # Model probabilities (plm_prob, mut_prob) need a tiny pseudocount (e.g. 1e-16) to not squash the long tail
        alpha_df = evaluate_alpha_sweep(
            combined_df,
            ALPHA_GRID,
            parallel=use_parallel_alpha,
            max_workers=ALPHA_SWEEP_MAX_WORKERS,
            pseudocount=1e-16,
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

                        lineage_pseudocount = float(10 ** -round(np.log10(10 * max(1, n_seq_lineage))))
                        x_vals = np.log10(
                            lineage_scatter_df["plm_prob"].replace(0, 1e-32)
                            * np.power(lineage_scatter_df["mut_prob"].replace(0, 1e-32), alpha_value)
                        )
                        y_vals = np.log10(lineage_scatter_df["obs_freq"].clip(lower=lineage_pseudocount))

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
                    "row = pooled population, columns = alpha values"
                )
                plt.tight_layout(rect=(0, 0, 1, 0.95))
                plt.savefig(
                    os.path.join(
                        model_outdir,
                        _tag_output_name("method2_obsfreq_vs_plm_mut_scatter_pooled_grid.png", TEST_MODE, DIVERSITY_PATTERN_TAG, OUTPUT_TAG),
                    ),
                    dpi=300,
                )

        # Per-lineage best-alpha extraction
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
        os.path.join(POOLED_PANEL_OUTDIR, _tag_output_name("model_run_status.tsv", TEST_MODE, DIVERSITY_PATTERN_TAG, OUTPUT_TAG)),
        sep="\t",
        index=False,
    )

    if len(all_alpha_frames) > 0:
        alpha_all_df = pd.concat(all_alpha_frames, ignore_index=True)
        alpha_all_df.to_csv(
            os.path.join(POOLED_PANEL_OUTDIR, _tag_output_name("alpha_sweep_fit_metrics_all_models.tsv", TEST_MODE, DIVERSITY_PATTERN_TAG, OUTPUT_TAG)),
            sep="\t",
            index=False,
        )

        # Compute baseline for mutation probability alone
        if not combined_df.empty:
            mut_only_df = combined_df.copy()
            mut_only_df["plm_prob"] = 1.0  # log(1.0) = 0, so score = alpha * log_mut
            mut_baseline_df = evaluate_alpha_sweep(
                mut_only_df,
                np.array([1.0]),
                parallel=False,
                pseudocount=1e-16
            )
            if not mut_baseline_df.empty:
                mut_baseline_metrics = mut_baseline_df.iloc[0]

        # Cross-model overlay plots
        metric_cols = [
            "site_top10pct_mutated_enrichment",
            "site_top10pct_mutated_precision",
            "site_rank_spearman_r",
            "mut_flat_global_spearman_r",
            "mut_flat_global_pearson_r",
            "mut_flat_mean_site_nll",
        ]
        
        def plot_overlay(plot_df, file_suffix):
            if len(plot_df) == 0:
                return
            fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True)
            axes = axes.flatten()
            for i, metric_col in enumerate(metric_cols):
                ax = axes[i]
                for model_tag, sub in plot_df.groupby("model"):
                    ax.plot(sub["alpha"], sub[metric_col], marker="o", label=model_tag)
                    
                if "mut_baseline_metrics" in locals() and metric_col in mut_baseline_metrics:
                    ax.axhline(mut_baseline_metrics[metric_col], color="black", linestyle="--", label="Mut Prob Only")

                title_map = {
                    "site_top10pct_mutated_enrichment": "Method A (site-level): enrichment of mutated sites in top 10%",
                    "site_top10pct_mutated_precision": "Method A (site-level): fraction of top 10% sites mutated",
                    "site_rank_spearman_r": "Method A (site-level): Spearman(site score vs burden)",
                    "mut_flat_global_spearman_r": "Method B (mutation-level): Spearman(score vs freq)",
                    "mut_flat_global_pearson_r": "Method B (mutation-level): Pearson(score vs freq)",
                    "mut_flat_mean_site_nll": "Method B (mutation-level): mean site-level NLL",
                }
                ax.set_title(title_map.get(metric_col, metric_col))
                ax.set_xlabel("Alpha weight")
                ax.set_ylabel("Metric value")
                ax.grid(alpha=0.3)
                if i == 0:
                    ax.legend()

            plt.tight_layout()
            plt.savefig(
                os.path.join(POOLED_PANEL_OUTDIR, _tag_output_name(f"alpha_sweep_model_comparison_{file_suffix}.png", TEST_MODE, DIVERSITY_PATTERN_TAG, OUTPUT_TAG)),
                dpi=300,
            )
            plt.close()

        # Plot overlays
        plot_overlay(alpha_all_df, "all")
        
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
        pd.DataFrame(best_rows).to_csv(
            os.path.join(POOLED_PANEL_OUTDIR, _tag_output_name("best_alpha_two_methods.tsv", TEST_MODE, DIVERSITY_PATTERN_TAG, OUTPUT_TAG)),
            sep="\t",
            index=False,
        )

        if len(per_lineage_best_rows) > 0:
            per_lineage_best_df = pd.DataFrame(per_lineage_best_rows)
            per_lineage_best_df.to_csv(
                os.path.join(POOLED_PANEL_OUTDIR, _tag_output_name("best_alpha_per_group_two_methods.tsv", TEST_MODE, DIVERSITY_PATTERN_TAG, OUTPUT_TAG)),
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
                ax.set_title(f"Per-group best alpha overlay\n{method_name}")
                ax.set_xlabel("Group")
                ax.set_ylabel("Best alpha")
                ax.grid(alpha=0.25, axis="y")
                if i > 0:
                    ax.get_legend().remove()

            handles, labels = axes_overlay[0].get_legend_handles_labels()
            fig_overlay.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
            plt.tight_layout(rect=[0, 0, 1, 0.92])
            plt.savefig(
                os.path.join(POOLED_PANEL_OUTDIR, _tag_output_name("best_alpha_per_group_overlay.png", TEST_MODE, DIVERSITY_PATTERN_TAG, OUTPUT_TAG)),
                dpi=300,
            )
            plt.close()

        print("\nPooled panel complete.")
        print(f"Saved outputs in: {POOLED_PANEL_OUTDIR}")
    else:
        print("No model runs completed successfully. Check model_run_status.tsv for details.")


    
    # %%
