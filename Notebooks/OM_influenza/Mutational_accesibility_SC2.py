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

from Bio import SeqIO ,Align, pairwise2
from Bio.Seq import Seq
from Bio.Data import CodonTable
import esm
import torch
from transformers import EsmForMaskedLM
import glob
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import random

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
    bases, TRANSITION_MATRICES,
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
# if in single fasta mode, months/ lineages
ANALYSIS_MODE = "MONTHLY_GUIDE"
# Guide CSV mapping months to FASTA paths (used if ANALYSIS_MODE == "MONTHLY_GUIDE")
POOLED_DIVERSITY_GUIDE = "/home3/oml4h/PLM_SARS-CoV-2/Sequences/SC2_month_snapshots/spike_month_file_guide.csv"

# if in single fasta mode, months/ lineages
ANALYSIS_MODE = "MONTHLY_GUIDE"
# Guide CSV mapping months to FASTA paths (used if ANALYSIS_MODE == "MONTHLY_GUIDE")
POOLED_DIVERSITY_GUIDE = "/home3/oml4h/PLM_SARS-CoV-2/Sequences/SC2_month_snapshots/spike_month_file_guide.csv"


# Focal reference sequence (nucleotide)
fasta_file = '/home3/oml4h/PLM_SARS-CoV-2/Sequences/SC2_month_snapshots/Jan25/rootseq_CM7YG5RN.fa'
base_lineage_id = '?'


# MODEL SELECTION: Choose which PLM model(s) to run and compare.
# Options: "ESMC_600M", "ESMC_600M_FT_SC2_99clus", "ESM2_650M_FT", "ESM2_3B_OG"

ANALYSIS_MODE = "MONTHLY_GUIDE"
# Guide CSV mapping months to FASTA paths (used if ANALYSIS_MODE == "MONTHLY_GUIDE")
POOLED_DIVERSITY_GUIDE = "/home3/oml4h/PLM_SARS-CoV-2/Sequences/SC2_month_snapshots/lineage_spilts/lineage_guide_soft.csv"


NUCLEOTIDE_MUTATION_MODEL = "SC2"
FILTER_SINGLETON_MUTATIONS = False
MODEL_SELECTION = ["ESMC_600M_FT_SC2_99clus","ESMC_600M_OG",'ESMC_300M_OG']
outdir='/home3/oml4h/PLM_SARS-CoV-2/Sequences/SC2_lineage_snapshots/ESMC_runs_aa/soft'
PLM_MAX_AA_LENGTH=1522

# ##SC2 ESM2 block
# FILTER_SINGLETON_MUTATIONS = True
# MODEL_SELECTION = ["sarbeco_SC2_FT_ESM2_650M_2023","ESM2_650M_OG","ESM2_3B_OG"]
# outdir='/home3/oml4h/PLM_SARS-CoV-2/Sequences/SC2_lineage_snapshots/ESM2_runs_aa/1022AA'
# PLM_MAX_AA_LENGTH=1022


# FILTER_SINGLETON_MUTATIONS = False
# MODEL_SELECTION = ["sarbeco_SC2_FT_ESM2_650M_2023","ESM2_650M_OG","ESM2_3B_OG"]
# outdir='/home3/oml4h/PLM_SARS-CoV-2/Sequences/SC2_month_snapshots/ESM2_runs_aa/1022AA'
# PLM_MAX_AA_LENGTH=1022

# #IAV block

# NUCLEOTIDE_MUTATION_MODEL = "H3N2"
# FILTER_SINGLETON_MUTATIONS = False
# POOLED_DIVERSITY_GUIDE = "/home3/oml4h/PLM_SARS-CoV-2/Sequences/IAV_lineage_guide.csv"
# fasta_file = '/home3/oml4h/PLM_SARS-CoV-2/Sequences/IAV_lineage_files/J_int.nt.fa'
# MODEL_SELECTION = ["IAV_Lytras_finetuned_HA80","ESM2_650M_OG","ESM2_3B_OG"]
# outdir='/home3/oml4h/PLM_SARS-CoV-2/Sequences/IAV_lineage_files/IAV_model_comparison_aa'

# TEST SETTINGS


TEST_MODE = False
TEST_MAX_RECORDS = 5
MIN_LINEAGE_SEQUENCE_COUNT = 100

# =====================================================================
# II. DATA PATHS AND LIMITS
# =====================================================================

# PLM sequence length limits
PLM_MAX_NT_LENGTH = None


# Set to True to strictly parse diversity FASTA files as protein, bypassing heuristics.
EXPECT_PROTEIN_DIVERSITY = True 

# Set to True to project a single global PLM matrix onto monthly diversity references
USE_GLOBAL_PLM_REFERENCE = False



# Filtering
FILTER_FIXED_MUTATIONS = True
# Optionally filter out singleton mutations seen in only one sample (per-site count)

SKIP_FILTER=False
# Minimum observed counts required to include a mutation (default 2 -> exclude singletons)
MIN_OBS_COUNT = 2

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
    "ESMC_300M_OG": {
        "tag": "ESMC_300M_OG",
        "base_model": "esm-c300m",
        "layer": 30,
        "checkpoint_dir": None,
        "force_recompute": False,
    },
    "ESMC_600M_FT_SC2_99clus": {
        "tag": "ESMC_600M_FT_SC2_99clus",
        "base_model": "esm-c600m",
        "layer": 36,
        "checkpoint_dir": "/home3/oml4h/my_SC2_finetunes/magma_ESMC_600M_99_95perc/",
        "force_recompute": False,
    },
    "IAV_Lytras_finetuned_HA80": {
        "tag": "IAV_Lytras_finetuned_HA80",
        "base_model": "esm2_t33_650M_UR50D",
        "layer": 33,
        "checkpoint_dir": "/home3/oml4h/hugging_face_downloads/model_weights_topublish/ESM2-HA80/",
        "force_recompute": False,
    },
    "sarbeco_SC2_FT_ESM2_650M_2023": {
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
# Ensure all models in MODEL_SELECTION exist in MODEL_PROFILES before proceeding
missing_models = [m for m in MODEL_SELECTION if m not in MODEL_PROFILES]
if missing_models:
    raise ValueError(
        f"MODEL_SELECTION contains unknown profile(s): {missing_models}. "
        f"Available profiles: {list(MODEL_PROFILES.keys())}"
    )

for choice in MODEL_SELECTION:
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

def generate_verified_coordinate_map(ref_seq: str, target_seq: str) -> dict:
    """
    Generates a 0-based coordinate map from a reference sequence to a target sequence.
    Uses free end gaps to prevent terminal missing data from causing frameshifts.
    """
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    
    # Crucial for viral consensus sequences: do not penalise missing termini
    aligner.target_end_gap_score = 0.0
    aligner.query_end_gap_score = 0.0

    alignments = aligner.align(ref_seq, target_seq)
    best_aln = next(iter(alignments))
    
    coord_map = {}
    ref_pos = 0
    target_pos = 0
    
    for char_ref, char_tgt in zip(best_aln[0], best_aln[1]):
        if char_ref != '-' and char_tgt != '-':
            coord_map[ref_pos] = target_pos
            ref_pos += 1
            target_pos += 1
        elif char_ref == '-':
            target_pos += 1
        elif char_tgt == '-':
            ref_pos += 1
            
    return coord_map, best_aln

def export_rolling_identity_plot(alignment, window_size=50, outdir=".", label=""):
    """
    Calculates and plots a rolling percentage identity between two aligned sequences.
    """
    os.makedirs(outdir, exist_ok=True)
    
    # Extract gapped sequences from the alignment object
    aln_ref, aln_tgt = alignment[0], alignment[1]
    
    identities = []
    positions = []
    
    # Track the ungapped position of the reference sequence for the x-axis
    ref_pos = 0 
    
    for i in range(len(aln_ref) - window_size + 1):
        window_r = aln_ref[i:i+window_size]
        window_t = aln_tgt[i:i+window_size]
        
        # Calculate matches, excluding positions where both are gaps
        valid_pairs = [(r, t) for r, t in zip(window_r, window_t) if not (r == '-' and t == '-')]
        if not valid_pairs:
            identities.append(0.0)
        else:
            matches = sum(1 for r, t in valid_pairs if r == t)
            identities.append((matches / len(valid_pairs)) * 100)
        
        # Advance sequence coordinate if reference is not a gap
        if aln_ref[i] != '-':
            ref_pos += 1
            
        positions.append(ref_pos)
        
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(positions, identities, color='k', linewidth=1.5)
    ax.set_title(f"Rolling Sequence Identity ({window_size}aa window) - {label}")
    ax.set_xlabel("Focal Sequence Position")
    ax.set_ylabel("% Identity")
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    export_publication_figure(os.path.join(outdir, f"rolling_identity_{label}.png"), figure=fig)
    plt.close()


def export_alignment_verification_plot(plm_matrix, ref_seq, target_seq, coord_map, month_label, outdir, max_cols=100):
    """
    Exports a heatmap showing the PLM top prediction vs the mapped sequences.
    Includes type-safe lookups to prevent false 100% mismatch rates.
    """
    os.makedirs(outdir, exist_ok=True)
    plot_data = []

    # Ensure we don't exceed the bounds of the sequence or the matrix
    limit = min(max_cols, len(ref_seq), plm_matrix.shape[1])
    for ref_pos in range(limit):
        target_pos = coord_map.get(ref_pos, None)

        # Bypass the index labels entirely using positional indexing
        top_aa = plm_matrix.iloc[:, ref_pos].idxmax()

        actual_ref_aa = ref_seq[ref_pos]
        actual_tgt_aa = target_seq[target_pos] if target_pos is not None and target_pos < len(target_seq) else '-'

        match_status = 1 if actual_tgt_aa == top_aa else 0

        plot_data.append({
            'Position': ref_pos + 1,
            'PLM_Top': top_aa,
            'Ref_AA': actual_ref_aa,
            'Target_AA': actual_tgt_aa,
            'Match': match_status
        })

    df = pd.DataFrame(plot_data)
    
    fig, ax = plt.subplots(figsize=(20, 4))
    sns.heatmap([df['Match'].tolist()], cmap=['#e74c3c', '#2ecc71'], cbar=False, ax=ax, linewidths=0.5)
    
    for i in range(len(df)):
        text = f"P:{df['PLM_Top'].iloc[i]}\nR:{df['Ref_AA'].iloc[i]}\nT:{df['Target_AA'].iloc[i]}"
        ax.text(i + 0.5, 0.5, text, ha='center', va='center', fontsize=8, color='black')
        
    ax.set_xticks(np.arange(len(df)) + 0.5)
    ax.set_xticklabels(df['Position'], rotation=90, fontsize=8)
    ax.set_yticks([])
    ax.set_title(f"PLM Prediction vs Sequences ({month_label})\nRed = Mismatch, Green = Match")
    
    plt.tight_layout()
    export_publication_figure(os.path.join(outdir, f"alignment_verification_{month_label}.png"), figure=fig)
    plt.close()

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
PLOT_EXPORT_PNG_DPI = 600

IGNORE_ALIGNMENT_CHARS = {"-", "*", "."}


def export_publication_figure(output_path, figure=None, png_dpi=PLOT_EXPORT_PNG_DPI):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = figure or plt.gcf()

    figure.savefig(
        str(output_path),
        dpi=png_dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        str(output_path.with_suffix(".pdf")),
        bbox_inches="tight",
        facecolor="white",
    )

# --- Main Execution Functions ---

# Update PLM_MAX_AA_LENGTH for ESMC if selected

if "esmc" in PLM_BASE_MODEL.lower() or "esm-c" in PLM_BASE_MODEL.lower():
    PLM_MAX_AA_LENGTH = 2048
    print(f"ESM-C model detected. Setting PLM_MAX_AA_LENGTH to {PLM_MAX_AA_LENGTH}")

# Pooled Population Settings
RUN_POOLED_PANEL = True
POOLED_POPULATION_LABEL = os.path.splitext(os.path.basename(fasta_file))[0]

# Comparison sequence settings for plots
# Only set the observed-mutation comparison FASTA when running in SINGLE_FASTA mode.
if ANALYSIS_MODE == "SINGLE_FASTA":
    OBSERVED_MUTATION_FASTA = POOLED_DIVERSITY_FASTA if POOLED_DIVERSITY_FASTA else None
elif ANALYSIS_MODE == "MONTHLY_GUIDE":
    # In MONTHLY_GUIDE mode we don't have a single observed-mutation fasta by default.
    OBSERVED_MUTATION_FASTA = None
else:
    OBSERVED_MUTATION_FASTA = POOLED_DIVERSITY_FASTA if POOLED_DIVERSITY_FASTA else None

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

# outdir calculation if not defined
try:
    outdir
except NameError:
    outdir = fasta_file.rsplit('/', 1)[0] + '/' + PLM_MODEL_TAG + '/mut_access_calcs'
POOLED_REFERENCE_FASTA = cut_fasta_file
POOLED_PANEL_OUTDIR = os.path.join(outdir, "pooled_panel")

os.makedirs(outdir, exist_ok=True)
os.makedirs(POOLED_PANEL_OUTDIR, exist_ok=True)
global key_matrix_dir
key_matrix_dir = os.path.join(outdir, "key_probability_matrices")
os.makedirs(key_matrix_dir, exist_ok=True)
print(outdir)

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
export_publication_figure(f"{outdir}/{active_transition_tag}_transition_matrix_heatmap.png")
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
export_publication_figure(f"{outdir}/{active_transition_tag}_transition_matrix_heatmap_log.png")
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
export_publication_figure(f"{outdir}/codon_to_amino_acid_matrix_heatmap.png")
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
    # Use pure positional indexing: the j-th column corresponds to the j-th residue
    col_idx = plm_matrix.columns[j]
    
    # Bypass string parsing; the j-th column is natively the j-th sequence index
    seq_idx = j 
    
    # Bounds check
    if seq_idx >= len(ref_nuc_seq) // 3:
        continue

    # Expected Amino Acid from PLM header (row 0 of original loaded df)
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
hist_values = mutational_prob_matrix.values.flatten()
hist_values = hist_values[np.isfinite(hist_values) & (hist_values > 0)]

plt.figure(figsize=(10, 6))
if hist_values.size > 0:
    log_bins = np.logspace(np.log10(hist_values.min()), np.log10(hist_values.max()), 100)
    ax = sns.histplot(
        hist_values,
        bins=log_bins,
        stat="count",
        element="bars",
        fill=True,
        color="#4c72b0",
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.5)
else:
    ax = plt.gca()
plt.title(f'Histogram of Mutational Probability Matrix Values ({PLM_MODEL_TAG})')
plt.xlabel('Mutational Probability')
plt.ylabel('Count')
export_publication_figure(f"{outdir}/histogram_mutational_prob_{PLM_MODEL_TAG}.png")

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
export_publication_figure(f"{outdir}/ranked_mutations_{PLM_MODEL_TAG}.png")
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
export_publication_figure(f"{outdir}/plm_vs_mut_correlation_{PLM_MODEL_TAG}.png")

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



RUN_MODE_TAG = "test" if TEST_MODE else "full"
# Derive diversity pattern tag from the active analysis mode so the script
# does not implicitly require the other (unused) path to be defined.
if ANALYSIS_MODE == "MONTHLY_GUIDE" and POOLED_DIVERSITY_GUIDE:
    _pattern_source = POOLED_DIVERSITY_GUIDE
elif ANALYSIS_MODE == "SINGLE_FASTA" and POOLED_DIVERSITY_FASTA:
    _pattern_source = POOLED_DIVERSITY_FASTA
else:
    # Fallback to the pooled population label when no path is provided
    _pattern_source = POOLED_POPULATION_LABEL

DIVERSITY_PATTERN_TAG = _clean_pattern_tag(Path(str(_pattern_source)).stem)
OUTPUT_TAG = f"{RUN_MODE_TAG}_{_safe_label(POOLED_POPULATION_LABEL)}_{DIVERSITY_PATTERN_TAG}"




# %%
if RUN_POOLED_PANEL:
    os.makedirs(POOLED_PANEL_OUTDIR, exist_ok=True)

    # Flag set when any diversity/lineage sequences look like nucleotides
    lineage_sequence_nucleotide_flag = False

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
            if TEST_MODE:
                # In test mode, only read the first data line (if any)
                row = next(reader, None)
                if row:
                    label = row.get('month') or row.get('label')
                    path = row.get('fasta') or row.get('path')
                    # Check for lineage-specific reference, fallback to global POOLED_REFERENCE_FASTA
                    ref_path = row.get('reference') or POOLED_REFERENCE_FASTA
                    if label and path:
                        diversity_targets.append((label, path, ref_path))
            else:
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
    skipped_lineage_rows = []
    # (Mutational profile will be computed inside the loop per reference)

    for label, fasta_path, ref_path in diversity_targets:
        if not os.path.exists(fasta_path):
            print(f"Warning: Diversity FASTA does not exist: {fasta_path}. Skipping.")
            skipped_lineage_rows.append({
                "lineage": label,
                "n_sequences": 0,
                "n_sequences_total": 0,
                "n_sequences_processed": 0,
                "reference_length": np.nan,
                "mapped_ref_sites": np.nan,
                "compared_sites_non_gap_non_stop": np.nan,
                "differing_sites_vs_reference_non_gap_non_stop": np.nan,
                "fixed_differing_sites_vs_reference_non_gap_non_stop": np.nan,
                "diversity_fasta": fasta_path,
                "diversity_tag": Path(fasta_path).stem,
                "plm_profile": "",
                "sequence_threshold_min": MIN_LINEAGE_SEQUENCE_COUNT,
                "sequence_threshold_passed": False,
                "skipped": True,
                "skip_reason": "missing_fasta",
            })
            continue

        total_records = list(SeqIO.parse(fasta_path, "fasta"))
        total_record_count = len(total_records)
        if total_record_count == 0:
            print(f"Warning: No records found in {fasta_path}. Skipping.")
            skipped_lineage_rows.append({
                "lineage": label,
                "n_sequences": 0,
                "n_sequences_total": 0,
                "n_sequences_processed": 0,
                "reference_length": np.nan,
                "mapped_ref_sites": np.nan,
                "compared_sites_non_gap_non_stop": np.nan,
                "differing_sites_vs_reference_non_gap_non_stop": np.nan,
                "fixed_differing_sites_vs_reference_non_gap_non_stop": np.nan,
                "diversity_fasta": fasta_path,
                "diversity_tag": Path(fasta_path).stem,
                "plm_profile": "",
                "sequence_threshold_min": MIN_LINEAGE_SEQUENCE_COUNT,
                "sequence_threshold_passed": False,
                "skipped": True,
                "skip_reason": "empty_fasta",
            })
            continue

        if total_record_count < MIN_LINEAGE_SEQUENCE_COUNT:
            print(
                f"Skipping {label}: {total_record_count} sequences in {fasta_path} "
                f"(< {MIN_LINEAGE_SEQUENCE_COUNT})."
            )
            skipped_lineage_rows.append({
                "lineage": label,
                "n_sequences": total_record_count,
                "n_sequences_total": total_record_count,
                "n_sequences_processed": 0,
                "reference_length": np.nan,
                "mapped_ref_sites": np.nan,
                "compared_sites_non_gap_non_stop": np.nan,
                "differing_sites_vs_reference_non_gap_non_stop": np.nan,
                "fixed_differing_sites_vs_reference_non_gap_non_stop": np.nan,
                "diversity_fasta": fasta_path,
                "diversity_tag": Path(fasta_path).stem,
                "plm_profile": "",
                "sequence_threshold_min": MIN_LINEAGE_SEQUENCE_COUNT,
                "sequence_threshold_passed": False,
                "skipped": True,
                "skip_reason": "below_min_sequence_threshold",
            })
            continue
            
        print(
            f"Processing {label} diversity from {fasta_path} using reference {ref_path} "
            f"(n={total_record_count})..."
        )
        
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

        records = total_records

        if TEST_MODE:
            records = records[:TEST_MAX_RECORDS]
        # Detect nucleotide-like diversity sequences when we expect protein sequences
        if EXPECT_PROTEIN_DIVERSITY:
            try:
                any_nuc = any(_is_probably_nucleotide_sequence(str(rec.seq)) for rec in records)
            except Exception:
                any_nuc = False
            if any_nuc:
                lineage_sequence_nucleotide_flag = True
                banner = "\n" + "!" * 80 + "\n" + (
                    "WARNING: INPUT DIVERSITY SEQUENCES APPEAR TO BE NUCLEOTIDES, NOT AMINO ACIDS"
                ) + "\n" + "!" * 80 + "\n"
                print(banner)
            
        # Ensure records are protein for the alignment comparison
      # Ensure records are protein for the alignment comparison
        processed_records = []
        for rec in records:
            if not EXPECT_PROTEIN_DIVERSITY and _is_probably_nucleotide_sequence(str(rec.seq)):
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
            "n_sequences_total": total_record_count,
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
    baseline_combined_df = None

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
        force_recompute_model = run_cfg.get("force_recompute", False)

        combined_df_path = os.path.join(
            model_outdir,
            _tag_output_name("pooled_combined_long_table.csv", TEST_MODE, DIVERSITY_PATTERN_TAG, OUTPUT_TAG),
        )
        lineage_meta_path = os.path.join(
            model_outdir,
            _tag_output_name("pooled_panel_metadata.tsv", TEST_MODE, DIVERSITY_PATTERN_TAG, OUTPUT_TAG),
        )
        alpha_df_path = os.path.join(
            model_outdir,
            _tag_output_name("alpha_sweep_fit_metrics.tsv", TEST_MODE, DIVERSITY_PATTERN_TAG, OUTPUT_TAG),
        )
        lineage_alpha_path = os.path.join(
            model_outdir,
            _tag_output_name("alpha_sweep_fit_metrics_by_lineage.tsv", TEST_MODE, DIVERSITY_PATTERN_TAG, OUTPUT_TAG),
        )

        model = None
        alphabet = None
        batch_converter = None
        model_ready = False
        used_cached_plm = False
        used_cached_downstream = False
        used_cached_alpha = False
        used_cached_lineage_alpha = False
        model_load_attempted = False # will be set in logic
        model_load_failed_reason = ""
        model_runtime_failed = False
        model_runtime_failed_reason = ""

        combined_df = None
        lineage_meta_df = None
        alpha_df = None
        lineage_alpha_all_df = None

        if not force_recompute_model and os.path.exists(combined_df_path) and os.path.exists(lineage_meta_path):
            try:
                combined_df = pd.read_csv(combined_df_path)
                lineage_meta_df = pd.read_csv(lineage_meta_path, sep="\t")
                required_lineage_meta_cols = {
                    "skipped",
                    "skip_reason",
                    "n_sequences_total",
                    "n_sequences_processed",
                    "sequence_threshold_min",
                    "sequence_threshold_passed",
                }
                if not required_lineage_meta_cols.issubset(lineage_meta_df.columns):
                    combined_df = None
                    lineage_meta_df = None
                    print(
                        f"Cached pooled tables for {model_tag} are missing lineage-threshold metadata; recomputing."
                    )
                else:
                    threshold_matches = pd.to_numeric(
                        lineage_meta_df["sequence_threshold_min"], errors="coerce"
                    ).eq(MIN_LINEAGE_SEQUENCE_COUNT)
                    active_lineage_meta_df = lineage_meta_df.loc[
                        ~lineage_meta_df["skipped"].fillna(False)
                    ].copy()
                    active_counts = pd.to_numeric(
                        active_lineage_meta_df["n_sequences_total"], errors="coerce"
                    )
                    cached_lineages_valid = threshold_matches.all() and not (
                        active_counts < MIN_LINEAGE_SEQUENCE_COUNT
                    ).fillna(False).any()
                    cached_combined_lineages = set(combined_df["lineage"]) if "lineage" in combined_df.columns else set()
                    active_lineages = set(active_lineage_meta_df["lineage"])
                    if not cached_lineages_valid or not cached_combined_lineages.issubset(active_lineages):
                        combined_df = None
                        lineage_meta_df = None
                        print(
                            f"Cached pooled tables for {model_tag} do not satisfy the current lineage threshold; recomputing."
                        )
                    else:
                        used_cached_downstream = True
                        print(f"Reusing cached pooled tables for {model_tag}: {combined_df_path}")
            except Exception as exc:
                combined_df = None
                lineage_meta_df = None
                print(f"Failed to load cached pooled tables for {model_tag}: {exc}")

        combined_rows = []
        per_lineage_summaries = [
            {
                "model": model_tag,
                **row,
            }
            for row in skipped_lineage_rows
        ]

        if combined_df is None or lineage_meta_df is None:
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
                if os.path.exists(plm_profile_path) and not force_recompute_model:
                    try:
                        plm_matrix = pd.read_csv(plm_profile_path, index_col=0)
                        used_cached_plm = True
                        print(f"Using existing PLM matrix from disk: {plm_profile_path}")
                    except Exception as exc:
                        print(f"Failed to load existing PLM matrix for {lineage} ({plm_profile_path}): {exc}")

                if plm_matrix is None:
                    cache_key = (model_tag, plm_ref_protein)
                    plm_out = None
                    
                    if cache_key in PLM_MATRIX_CACHE and not force_recompute_model:
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
                                # Infer per-model maximum input length (aa) and set PLM_MAX_NT_LENGTH (nt)
                                try:
                                    max_aa = None
                                    # Common places to find model/tokenizer length
                                    if hasattr(alphabet, "max_seq_len") and alphabet.max_seq_len:
                                        max_aa = int(alphabet.max_seq_len)
                                    elif hasattr(alphabet, "_tokenizer") and hasattr(alphabet._tokenizer, "model_max_length"):
                                        max_aa = int(alphabet._tokenizer.model_max_length)
                                    elif hasattr(alphabet, "tokenizer") and hasattr(alphabet.tokenizer, "model_max_length"):
                                        max_aa = int(alphabet.tokenizer.model_max_length)
                                    elif hasattr(model, "config") and hasattr(model.config, "max_position_embeddings"):
                                        max_aa = int(model.config.max_position_embeddings)
                                    # Fallback for ESM-C models: allow large context
                                    if max_aa is None and ("esmc" in run_cfg["base_model"].lower() or "esm-c" in run_cfg["base_model"].lower()):
                                        max_aa = 2048

                                    if max_aa is not None:
                                        PLM_MAX_NT_LENGTH = max_aa * 3
                                        print(f"Inferred PLM max aa={max_aa}, setting PLM_MAX_NT_LENGTH={PLM_MAX_NT_LENGTH}")
                                except Exception as exc:
                                    print(f"Warning: failed to infer PLM max length for {model_tag}: {exc}")
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

                # Assuming 'focal_protein_seq' is your global root reference that generated the PLM
                global_to_monthly_map = {}
                if USE_GLOBAL_PLM_REFERENCE:
                    try:
                        # Uses the robust generator we set up previously
                        global_to_monthly_map, alignment_obj = generate_verified_coordinate_map(focal_protein_seq, plm_ref_protein)

                        verify_dir = os.path.join(model_outdir, "alignment_verifications")
                        
                        # 1. Generate the type-safe verification heatmap
                        export_alignment_verification_plot(
                            plm_matrix=plm_matrix,
                            ref_seq=focal_protein_seq,
                            target_seq=plm_ref_protein,
                            coord_map=global_to_monthly_map,
                            month_label=lineage,
                            outdir=verify_dir,
                        )
                        
                        # 2. Generate the rolling % identity curve
                        export_rolling_identity_plot(
                            alignment=alignment_obj,
                            window_size=30,
                            outdir=verify_dir,
                            label=lineage
                        )

                        mapped_target_aas = [plm_ref_protein[global_to_monthly_map[i]] for i in range(len(focal_protein_seq)) if i in global_to_monthly_map]
                        print(f"[{lineage}] Mapping complete. Captured {len(mapped_target_aas)} aligned residues.")
                    except Exception as exc:
                        print(f"[{lineage}] Alignment failed: {exc}")
                

                # --- ROBUST COORDINATE MAPPING ---
                for j, pos_label in enumerate(plm_matrix.columns):
                    pos_plm_0 = j
                    
                    if USE_GLOBAL_PLM_REFERENCE:
                        if pos_plm_0 not in global_to_monthly_map:
                            continue
                        mapped_monthly_0 = global_to_monthly_map[pos_plm_0]
                        
                        if mapped_monthly_0 not in coord_map:
                            continue
                        pos_full_0 = coord_map[mapped_monthly_0]
                    else:
                        if pos_plm_0 not in coord_map:
                            continue
                        pos_full_0 = coord_map[pos_plm_0]

                    pos_full_1 = pos_full_0 + 1
                    ref_aa = plm_ref_protein[pos_plm_0]
                    
                    if pos_full_1 not in data["mut_profile"].columns:
                        continue

                    for aa in plm_matrix.index:
                        if aa == ref_aa:
                            continue
                        if aa not in data["mut_profile"].index or aa not in data["obs_freq"].index:
                            continue

                        row_idx = plm_matrix.index.get_loc(aa)
                        plm_prob = float(plm_matrix.iloc[row_idx, j])
                        mut_prob = float(data["mut_profile"].loc[aa, pos_full_1])
                        obs = float(data["obs_freq"].loc[aa, pos_full_1])
                        
                        depth_here = int(data["obs_depth"].get(pos_full_1, 0)) if pos_full_1 in data["obs_depth"].index else int(data["obs_depth"].loc[pos_full_1])
                        obs_count_est = int(round(obs * depth_here)) if depth_here > 0 else 0

                        if FILTER_FIXED_MUTATIONS and obs >= 1.0:
                            continue

                        obs_final = obs
                        obs_present_final = 1 if obs > 0 else 0
                        if FILTER_SINGLETON_MUTATIONS and obs_count_est < MIN_OBS_COUNT:
                            if SKIP_FILTER:
                                continue
                            else:
                                obs_final = 0.0
                                obs_present_final = 0

                        combined_rows.append({
                            "model": model_tag,
                            "lineage": lineage,
                            "position": int(pos_full_1),
                            "ref_aa": ref_aa,
                            "aa": aa,
                            "plm_prob": plm_prob,
                            "mut_prob": mut_prob,
                            "obs_freq": obs_final,
                            "obs_present": obs_present_final,
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
                    "n_sequences": data["n_sequences_total"],
                    "n_sequences_total": data["n_sequences_total"],
                    "n_sequences_processed": len(data["records"]),
                    "reference_length": len(data["full_ref_protein"]),
                    "mapped_ref_sites": int(data["alignment_diff_stats"]["mapped_sites"]),
                    "compared_sites_non_gap_non_stop": int(data["alignment_diff_stats"]["compared_sites"]),
                    "differing_sites_vs_reference_non_gap_non_stop": int(data["alignment_diff_stats"]["differing_sites"]),
                    "fixed_differing_sites_vs_reference_non_gap_non_stop": int(data["alignment_diff_stats"]["fixed_differing_sites"]),
                    "diversity_fasta": data["diversity_path"],
                    "diversity_tag": data["diversity_tag"],
                    "plm_profile": plm_profile_path,
                    "sequence_threshold_min": MIN_LINEAGE_SEQUENCE_COUNT,
                    "sequence_threshold_passed": True,
                    "skipped": False,
                    "skip_reason": "",
                })

            if model_runtime_failed:
                combined_rows = []
                per_lineage_summaries = []
                model_status_rows.append({
                    "model": model_tag,
                    "status": "skipped",
                    "reason": f"runtime failure: {model_runtime_failed_reason}",
                    "load_failed_reason": model_load_failed_reason,
                    "runtime_failed_reason": model_runtime_failed_reason,
                    "used_cached_plm": used_cached_plm,
                    "used_cached_downstream": used_cached_downstream,
                    "used_cached_alpha": used_cached_alpha,
                    "used_cached_lineage_alpha": used_cached_lineage_alpha,
                })
                continue

            combined_df = pd.DataFrame(combined_rows)
            if combined_df.empty:
                print(f"No combined rows produced for {model_tag}; skipping alpha sweep.")
                continue

            combined_df.to_csv(combined_df_path, index=False)
            lineage_meta_df = pd.DataFrame(per_lineage_summaries)
            lineage_meta_df.to_csv(lineage_meta_path, sep="\t", index=False)
            print(f"Saved cached pooled tables for {model_tag}: {combined_df_path}")

        # Ensure we always append a status row with explicit reasons and diagnostics
        if model_ready:
            status = "loaded"
            reason = "used for PLM generation"
        elif used_cached_downstream:
            status = "cached_only"
            reason = "reused cached pooled tables"
        elif used_cached_plm:
            status = "cached_only"
            reason = "all PLM profiles reused from disk"
        elif model_load_attempted and model_load_failed_reason:
            status = "skipped"
            reason = f"load failed: {model_load_failed_reason}"
        else:
            status = "skipped"
            reason = "no PLM profiles generated and no load error captured (no model loaded, no cached matrices)"

        if combined_df is None or combined_df.empty:
            print(f"No combined rows produced for {model_tag}; skipping alpha sweep.")
            continue

        if baseline_combined_df is None and not combined_df.empty:
            baseline_combined_df = combined_df.copy()

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
                    sp_rho, sp_p = spearmanr(plot_data["plm_prob"], plot_data["mut_prob"])
                    try:
                        pe_r, pe_p = pearsonr(plot_data["plm_prob"], plot_data["mut_prob"])
                    except Exception:
                        pe_r, pe_p = (np.nan, np.nan)

                    plt.scatter(plot_data["plm_prob"], plot_data["mut_prob"], alpha=0.3, s=10, edgecolors="none")
                    plt.xscale("log")
                    plt.yscale("log")
                    plt.xlabel("PLM Probability")
                    plt.ylabel("Mutation Probability (Codon Model)")
                    plt.title(
                        f"{model_tag} Correlation\n"
                        f"Spearman rho={sp_rho:.3f} (p={sp_p:.2e}); "
                        f"Pearson r={pe_r:.3f} (p={pe_p:.2e})"
                    )
                    plt.grid(True, which="both", ls="--", alpha=0.5)
                    
                    plot_path = os.path.join(model_outdir, _tag_output_name("plm_vs_mut_prob_scatter.png", TEST_MODE, DIVERSITY_PATTERN_TAG, OUTPUT_TAG))
                    plt.tight_layout()
                    export_publication_figure(plot_path)
                    plt.close()
            except Exception as plot_exc:
                print(f"Warning: Failed to generate comparison plot for {model_tag}: {plot_exc}")

        if lineage_meta_df is None:
            lineage_meta_df = pd.DataFrame(per_lineage_summaries)
            lineage_meta_df.to_csv(lineage_meta_path, sep="\t", index=False)

        use_parallel_alpha = ALPHA_SWEEP_PARALLEL and len(ALPHA_GRID) >= ALPHA_SWEEP_MIN_GRID
        if use_parallel_alpha:
            print(f"Running alpha sweep in parallel for {model_tag} (n_alpha={len(ALPHA_GRID)})")

        active_lineage_meta_df = lineage_meta_df.loc[~lineage_meta_df["skipped"].fillna(False)].copy()
        n_seq_pooled = int(active_lineage_meta_df["n_sequences"].max()) if not active_lineage_meta_df.empty else 1000
        dynamic_pseudocount = float(10 ** -round(np.log10(10 * n_seq_pooled)))
        print(f"Using dynamic pseudocount for obs_freq plotting: {dynamic_pseudocount:.1e} based on {n_seq_pooled} max sequences")

        required_alpha_cols = {
            "alpha",
            "mut_flat_global_pearson_r",
            "mut_flat_nonzero_spearman_r",
            "mut_flat_nonzero_pearson_r",
            "mut_flat_logfreq_global_pearson_r",
            "mut_flat_logfreq_nonzero_pearson_r",
            "site_top10pct_mutated_enrichment",
            "mut_flat_global_spearman_r",
        }
        if not force_recompute_model and os.path.exists(alpha_df_path):
            try:
                alpha_df = pd.read_csv(alpha_df_path, sep="\t")
                if required_alpha_cols.issubset(alpha_df.columns):
                    used_cached_alpha = True
                    print(f"Reusing cached alpha sweep for {model_tag}: {alpha_df_path}")
                else:
                    alpha_df = None
                    print(f"Cached alpha sweep missing required columns for {model_tag}; recomputing.")
            except Exception as exc:
                alpha_df = None
                print(f"Failed to load cached alpha sweep for {model_tag}: {exc}")

        if alpha_df is None:
            alpha_df = evaluate_alpha_sweep(
                combined_df,
                ALPHA_GRID,
                parallel=use_parallel_alpha,
                max_workers=ALPHA_SWEEP_MAX_WORKERS,
                pseudocount=1e-16,
            )
            alpha_df["model"] = model_tag
            alpha_df.to_csv(alpha_df_path, sep="\t", index=False)
            print(f"Saved alpha sweep cache for {model_tag}: {alpha_df_path}")
        elif "model" not in alpha_df.columns:
            alpha_df["model"] = model_tag

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
                        n_mut_obs = int(lineage_scatter_df["obs_freq"].gt(0).sum())
                        ax.set_title(
                            f"alpha={alpha_value:.2f}\n"
                            f"ρ={corr_r:.3f}, n_mut={n_mut_obs}, n_seq={n_seq_lineage}"
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
                    f"{model_tag}: Method B (mutation-level) — observed mutation frequency vs PLM×mutation accessibility score\n"
                    "row = pooled population, columns = alpha values"
                )
                plt.tight_layout(rect=(0, 0, 1, 0.95))
                export_publication_figure(
                    os.path.join(
                        model_outdir,
                        _tag_output_name("method2_obsfreq_vs_plm_mut_scatter_pooled_grid.png", TEST_MODE, DIVERSITY_PATTERN_TAG, OUTPUT_TAG),
                    ),
                    figure=fig_sc,
                )
                plt.close(fig_sc)

        required_lineage_alpha_cols = {
            "model",
            "lineage",
            "alpha",
            "site_top10pct_mutated_enrichment",
            "mut_flat_global_spearman_r",
        }
        if not force_recompute_model and os.path.exists(lineage_alpha_path):
            try:
                lineage_alpha_all_df = pd.read_csv(lineage_alpha_path, sep="\t")
                if required_lineage_alpha_cols.issubset(lineage_alpha_all_df.columns):
                    used_cached_lineage_alpha = True
                    print(f"Reusing cached per-lineage alpha sweeps for {model_tag}: {lineage_alpha_path}")
                else:
                    lineage_alpha_all_df = None
                    print(f"Cached per-lineage alpha sweeps missing required columns for {model_tag}; recomputing.")
            except Exception as exc:
                lineage_alpha_all_df = None
                print(f"Failed to load cached per-lineage alpha sweeps for {model_tag}: {exc}")

        if lineage_alpha_all_df is None:
            lineage_alpha_frames = []
            for lineage_name, lineage_df in combined_df.groupby("lineage"):
                lineage_alpha = evaluate_alpha_sweep(
                    lineage_df,
                    ALPHA_GRID,
                    parallel=use_parallel_alpha,
                    max_workers=ALPHA_SWEEP_MAX_WORKERS,
                )
                if lineage_alpha.empty:
                    continue
                lineage_alpha["model"] = model_tag
                lineage_alpha["lineage"] = lineage_name
                lineage_alpha_frames.append(lineage_alpha)

            if len(lineage_alpha_frames) > 0:
                lineage_alpha_all_df = pd.concat(lineage_alpha_frames, ignore_index=True)
                lineage_alpha_all_df.to_csv(lineage_alpha_path, sep="\t", index=False)
                print(f"Saved per-lineage alpha sweep cache for {model_tag}: {lineage_alpha_path}")
            else:
                lineage_alpha_all_df = pd.DataFrame()

        def _select_best_alpha_index(df, metric_col):
            if metric_col not in df.columns or "alpha" not in df.columns:
                return None
            valid = df.loc[df[metric_col].notna() & df["alpha"].notna()]
            if valid.empty:
                return None
            return valid[metric_col].idxmax()

        if not lineage_alpha_all_df.empty:
            for lineage_name, lineage_alpha in lineage_alpha_all_df.groupby("lineage"):
                idx_a = _select_best_alpha_index(lineage_alpha, "site_top10pct_mutated_enrichment")
                if idx_a is not None:
                    per_lineage_best_rows.append({
                        "model": model_tag,
                        "lineage": lineage_name,
                        "method": "Method A (Site-level)",
                        "criterion": "max site_top10pct_mutated_enrichment",
                        "best_alpha": float(lineage_alpha.loc[idx_a, "alpha"]),
                        "best_value": float(lineage_alpha.loc[idx_a, "site_top10pct_mutated_enrichment"]),
                    })

                idx_b = _select_best_alpha_index(lineage_alpha, "mut_flat_global_spearman_r")
                if idx_b is not None:
                    per_lineage_best_rows.append({
                        "model": model_tag,
                        "lineage": lineage_name,
                        "method": "Method B (Mutation-level flattened)",
                        "criterion": "max mut_flat_global_spearman_r",
                        "best_alpha": float(lineage_alpha.loc[idx_b, "alpha"]),
                        "best_value": float(lineage_alpha.loc[idx_b, "mut_flat_global_spearman_r"]),
                    })

        model_status_rows.append({
            "model": model_tag,
            "status": status,
            "reason": reason,
            "load_failed_reason": model_load_failed_reason,
            "runtime_failed_reason": model_runtime_failed_reason,
            "used_cached_plm": used_cached_plm,
            "used_cached_downstream": used_cached_downstream,
            "used_cached_alpha": used_cached_alpha,
            "used_cached_lineage_alpha": used_cached_lineage_alpha,
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
        method2_overlay_required_cols = [
            "mut_flat_nonzero_spearman_r",
            "mut_flat_nonzero_pearson_r",
            "mut_flat_logfreq_global_pearson_r",
            "mut_flat_logfreq_nonzero_pearson_r",
        ]

        alpha_frames_by_model = {}
        for alpha_frame in all_alpha_frames:
            if alpha_frame is None or len(alpha_frame) == 0 or "model" not in alpha_frame.columns:
                continue
            frame_models = pd.Series(alpha_frame["model"]).dropna().unique().tolist()
            if len(frame_models) != 1:
                continue
            alpha_frames_by_model[frame_models[0]] = alpha_frame.copy()

        rebuilt_alpha_frames = []
        for run_cfg in MODEL_RUNS:
            model_tag = run_cfg["tag"]
            model_outdir = os.path.join(POOLED_PANEL_OUTDIR, model_tag)
            combined_df_path = os.path.join(
                model_outdir,
                _tag_output_name("pooled_combined_long_table.csv", TEST_MODE, DIVERSITY_PATTERN_TAG, OUTPUT_TAG),
            )
            alpha_df_path = os.path.join(
                model_outdir,
                _tag_output_name("alpha_sweep_fit_metrics.tsv", TEST_MODE, DIVERSITY_PATTERN_TAG, OUTPUT_TAG),
            )

            alpha_frame = alpha_frames_by_model.get(model_tag)
            missing_overlay_cols = [] if alpha_frame is None else [
                col for col in method2_overlay_required_cols if col not in alpha_frame.columns
            ]

            if alpha_frame is None or missing_overlay_cols:
                reason = "missing in-memory alpha sweep" if alpha_frame is None else f"missing metrics {missing_overlay_cols}"
                print(f"Recomputing alpha sweep for {model_tag} ({reason}).")
                if not os.path.exists(combined_df_path):
                    raise FileNotFoundError(
                        f"Cannot recompute alpha sweep for {model_tag}; combined table not found: {combined_df_path}"
                    )
                combined_rebuild_df = pd.read_csv(combined_df_path)
                if combined_rebuild_df.empty:
                    raise ValueError(
                        f"Cannot recompute alpha sweep for {model_tag}; combined table is empty: {combined_df_path}"
                    )
                alpha_frame = evaluate_alpha_sweep(
                    combined_rebuild_df,
                    ALPHA_GRID,
                    parallel=ALPHA_SWEEP_PARALLEL and len(ALPHA_GRID) >= ALPHA_SWEEP_MIN_GRID,
                    max_workers=ALPHA_SWEEP_MAX_WORKERS,
                    pseudocount=1e-16,
                )
                alpha_frame["model"] = model_tag
                alpha_frame.to_csv(alpha_df_path, sep="\t", index=False)
                print(f"Saved recomputed alpha sweep cache for {model_tag}: {alpha_df_path}")
            elif "model" not in alpha_frame.columns:
                alpha_frame = alpha_frame.copy()
                alpha_frame["model"] = model_tag

            rebuilt_alpha_frames.append(alpha_frame)

        alpha_all_df = pd.concat(rebuilt_alpha_frames, ignore_index=True)

        alpha_all_df.to_csv(
            os.path.join(POOLED_PANEL_OUTDIR, _tag_output_name("alpha_sweep_fit_metrics_all_models.tsv", TEST_MODE, DIVERSITY_PATTERN_TAG, OUTPUT_TAG)),
            sep="\t",
            index=False,
        )

        mut_baseline_metrics = None
        mut_baseline_path = os.path.join(
            POOLED_PANEL_OUTDIR,
            _tag_output_name("alpha_sweep_fit_metrics_mutation_only.tsv", TEST_MODE, DIVERSITY_PATTERN_TAG, OUTPUT_TAG),
        )

        # Compute baseline for mutation probability alone
        if baseline_combined_df is not None and not baseline_combined_df.empty:
            mut_baseline_df = None
            if os.path.exists(mut_baseline_path):
                try:
                    mut_baseline_df = pd.read_csv(mut_baseline_path, sep="\t")
                    missing_mut_baseline_cols = [
                        col for col in method2_overlay_required_cols if col not in mut_baseline_df.columns
                    ]
                    if missing_mut_baseline_cols:
                        mut_baseline_df = None
                        print(
                            "Cached mutation-only alpha baseline missing required columns; recomputing. "
                            f"Missing: {missing_mut_baseline_cols}"
                        )
                    else:
                        print(f"Reusing cached mutation-only alpha baseline: {mut_baseline_path}")
                except Exception as exc:
                    mut_baseline_df = None
                    print(f"Failed to load cached mutation-only alpha baseline: {exc}")

            if mut_baseline_df is None:
                mut_only_df = baseline_combined_df.copy()
                mut_only_df["plm_prob"] = 1.0  # log(1.0) = 0, so score = alpha * log_mut
                mut_baseline_df = evaluate_alpha_sweep(
                    mut_only_df,
                    np.array([1.0]),
                    parallel=False,
                    pseudocount=1e-16
                )
                mut_baseline_df.to_csv(mut_baseline_path, sep="\t", index=False)
                print(f"Saved mutation-only alpha baseline cache: {mut_baseline_path}")
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
        
        # Consistent color mapping across plots
        run_tags_order = [r["tag"] for r in MODEL_RUNS]
        def _build_model_palette(from_df):
            models_in_df = list(pd.Series(from_df["model"]).dropna().unique()) if len(from_df) > 0 else []
            # Preserve user-specified run order where possible
            model_order = [t for t in run_tags_order if t in models_in_df] + [m for m in models_in_df if m not in run_tags_order]
            if len(model_order) == 0:
                return {}, []
            colors = sns.color_palette("tab10", n_colors=max(len(model_order), 3))
            palette = {m: colors[i % len(colors)] for i, m in enumerate(model_order)}
            return palette, model_order

        model_palette, model_order = _build_model_palette(alpha_all_df)

        def _choose_legend_corner(ax):
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            x_span = xlim[1] - xlim[0]
            y_span = ylim[1] - ylim[0]
            if not np.isfinite(x_span) or not np.isfinite(y_span) or x_span == 0 or y_span == 0:
                return "best"

            corner_boxes = {
                "upper left": (0.0, 0.35, 0.65, 1.0),
                "upper right": (0.65, 1.0, 0.65, 1.0),
                "lower left": (0.0, 0.35, 0.0, 0.35),
                "lower right": (0.65, 1.0, 0.0, 0.35),
            }
            occupancy = {corner: 0 for corner in corner_boxes}

            def _accumulate_points(x_vals, y_vals):
                x_vals = np.asarray(x_vals, dtype=float)
                y_vals = np.asarray(y_vals, dtype=float)
                valid = np.isfinite(x_vals) & np.isfinite(y_vals)
                if not np.any(valid):
                    return

                x_norm = (x_vals[valid] - xlim[0]) / x_span
                y_norm = (y_vals[valid] - ylim[0]) / y_span
                for corner, (xmin, xmax, ymin, ymax) in corner_boxes.items():
                    in_corner = (
                        (x_norm >= xmin) & (x_norm <= xmax) &
                        (y_norm >= ymin) & (y_norm <= ymax)
                    )
                    occupancy[corner] += int(np.count_nonzero(in_corner))

            for line in ax.get_lines():
                _accumulate_points(line.get_xdata(), line.get_ydata())

            for collection in ax.collections:
                try:
                    offsets = collection.get_offsets()
                except Exception:
                    offsets = None
                if offsets is None or len(offsets) == 0:
                    continue
                _accumulate_points(offsets[:, 0], offsets[:, 1])

            return min(
                occupancy,
                key=lambda corner: (occupancy[corner], list(corner_boxes).index(corner)),
            )

        def _add_plm_only_reference(ax, color="black"):
            ax.axvline(0.0, color=color, linestyle="--", linewidth=1.2, alpha=0.9, zorder=1)
            ax.annotate(
                "PLM_only",
                xy=(0.0, 0.02),
                xycoords=ax.get_xaxis_transform(),
                xytext=(4, 0),
                textcoords="offset points",
                ha="left",
                va="bottom",
                rotation=90,
                fontsize=8,
                color=color,
                clip_on=False,
            )

        def plot_overlay(plot_df, metric_cols, title_map, file_suffix, nrows, ncols, figsize):
            if len(plot_df) == 0:
                return
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
            if isinstance(axes, np.ndarray):
                axes = list(axes.flat)
            else:
                axes = [axes]
            for i, metric_col in enumerate(metric_cols):
                ax = axes[i]
                for model_tag, sub in plot_df.groupby("model"):
                    color = model_palette.get(model_tag)
                    ax.plot(sub["alpha"], sub[metric_col], marker="o", label=model_tag, color=color)

                if mut_baseline_metrics is not None and metric_col in mut_baseline_metrics.index:
                    ax.scatter(
                        [float(mut_baseline_metrics["alpha"])],
                        [float(mut_baseline_metrics[metric_col])],
                        marker="s",
                        s=110,
                        color="#d62728",
                        edgecolors="black",
                        linewidths=1.0,
                        zorder=6,
                        label="Mut Prob Only",
                    )

                _add_plm_only_reference(ax)
                ax.set_title(title_map.get(metric_col, metric_col))
                ax.set_xlabel("Alpha weight")
                ax.set_ylabel("Metric value")
                ax.grid(alpha=0.3)
                legend_loc = _choose_legend_corner(ax)
                try:
                    ax.legend(loc=legend_loc, fontsize=8)
                except Exception:
                    ax.legend()

            for ax in axes[len(metric_cols):]:
                ax.axis("off")

            plt.tight_layout()
            export_publication_figure(
                os.path.join(POOLED_PANEL_OUTDIR, _tag_output_name(f"alpha_sweep_model_comparison_{file_suffix}.png", TEST_MODE, DIVERSITY_PATTERN_TAG, OUTPUT_TAG)),
                figure=fig,
            )
            plt.close()

        # Plot overlays
        plot_overlay(
            alpha_all_df,
            metric_cols=[
                "site_top10pct_mutated_enrichment",
                "site_top10pct_mutated_precision",
                "site_rank_spearman_r",
                "mut_flat_global_spearman_r",
                "mut_flat_global_pearson_r",
                "mut_flat_mean_site_nll",
            ],
            title_map={
                "site_top10pct_mutated_enrichment": "Method A (site-level): enrichment of mutated sites in top 10%",
                "site_top10pct_mutated_precision": "Method A (site-level): fraction of top 10% sites mutated",
                "site_rank_spearman_r": "Method A (site-level): Spearman(site score vs burden)",
                "mut_flat_global_spearman_r": "Method B (mutation-level): Spearman(score vs freq)",
                "mut_flat_global_pearson_r": "Method B (mutation-level): Pearson(score vs freq)",
                "mut_flat_mean_site_nll": "Method B (mutation-level): mean site-level NLL",
            },
            file_suffix="all",
            nrows=2,
            ncols=3,
            figsize=(18, 9),
        )
        plot_overlay(
            alpha_all_df,
            metric_cols=[
                "mut_flat_nonzero_spearman_r",
                "mut_flat_nonzero_pearson_r",
                "mut_flat_logfreq_global_pearson_r",
                "mut_flat_logfreq_nonzero_pearson_r",
            ],
            title_map={
                "mut_flat_nonzero_spearman_r": "Method B (mutation-level): Spearman RANK(score vs freq), non-zero obs only",
                "mut_flat_nonzero_pearson_r": "Method B (mutation-level): Pearson(score vs freq), non-zero obs only",
                "mut_flat_logfreq_global_pearson_r": "Method B (mutation-level): Pearson(score vs log(freq + pc)), zeroes included",
                "mut_flat_logfreq_nonzero_pearson_r": "Method B (mutation-level): Pearson(score vs log(freq)), non-zero obs only",
            },
            file_suffix="nonzero_and_logfreq",
            nrows=2,
            ncols=2,
            figsize=(14, 10),
        )
        
        # Explicit two-method best-alpha summary table
        best_rows = []
        for model_tag, sub in alpha_all_df.groupby("model"):
            if sub.empty:
                continue
            idx_a = _select_best_alpha_index(sub, "site_top10pct_mutated_enrichment")
            if idx_a is not None:
                best_rows.append({
                    "model": model_tag,
                    "method": "Method A (Site-level)",
                    "criterion": "max site_top10pct_mutated_enrichment",
                    "best_alpha": float(sub.loc[idx_a, "alpha"]),
                })
            idx_b = _select_best_alpha_index(sub, "mut_flat_global_spearman_r")
            if idx_b is not None:
                best_rows.append({
                    "model": model_tag,
                    "method": "Method B (Mutation-level flattened)",
                    "criterion": "max mut_flat_global_spearman_r",
                    "best_alpha": float(sub.loc[idx_b, "alpha"]),
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
                    palette=model_palette,
                    hue_order=model_order,
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
            export_publication_figure(
                os.path.join(POOLED_PANEL_OUTDIR, _tag_output_name("best_alpha_per_group_overlay.png", TEST_MODE, DIVERSITY_PATTERN_TAG, OUTPUT_TAG)),
                figure=fig_overlay,
            )
            plt.close()

        # Closing banner if nucleotide-like sequences were detected earlier
        if 'lineage_sequence_nucleotide_flag' in globals() and lineage_sequence_nucleotide_flag:
            end_banner = "\n" + "!" * 80 + "\n" + (
                "END WARNING: INPUT DIVERSITY SEQUENCES WERE DETECTED AS NUCLEOTIDES"
            ) + "\n" + "!" * 80 + "\n"
            print(end_banner)

        print("\nPooled panel complete.")
        print(f"Saved outputs in: {POOLED_PANEL_OUTDIR}")
    else:
        print("No model runs completed successfully. Check model_run_status.tsv for details.")


    
    # %%
    print(outdir)
    print(PLM_MAX_NT_LENGTH)
# %%
