
import sys
import os
import pandas as pd
import numpy as np
import torch
import joblib
import glob
from transformers import AutoTokenizer, EsmConfig
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO

import nbformat
from Bio import Align


# Add PLM_SARS-CoV-2 to path for Functions_HuggingFace
sys.path.append("/home3/oml4h/PLM_SARS-CoV-2")
try:
    from Functions_HuggingFace import (
        create_h3_numbering_map,
        mutations_to_canonical,
        get_mutations,
        plant_trim_to_target_length,
    )
    print("Imported Functions_HuggingFace OK.")
except ImportError:
    print("Could not import Functions_HuggingFace. Check path.")
    sys.exit(1)

# Add PLANT to path
sys.path.append("/home3/oml4h/hugging_face_downloads/PLANT_model/code/src")
try:
    from plant import TextDataset, tokenize_sequences, semanticESM, set_encoders, embed_sequences
    print("Imported plant module OK.")
except ImportError:
    print("Could not import plant module. Check path.")
    sys.exit(1)

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Test Mode Toggle
TEST_MODE = False #True # Set to False to run on all sites

# PLANT Model Config
REPO_DIR = "/home3/oml4h/hugging_face_downloads/PLANT_model"
CKPT_DIR = os.path.join(REPO_DIR, "variants/PLANT_fixed")
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
SCALE_FACTOR = 8

# Input Data
# QUERY_PATH removed - we get sequence from matrix header
PROB_MATRIX_PATH = "/home3/oml4h/PLM_SARS-CoV-2/Results/test/J.2.4/J.2.4_probability_matrix.csv"
BASE_LINEAGE_NAME = "J.2.4" # Hardcoded for output naming since we aren't parsing it
SEQ_FILE_PATH = "/home3/oml4h/PLM_SARS-CoV-2/Sequences/huH3N2_HA_CDS.translated_OM_synth_extra_steps.fas"
CANONICAL_REF_PATH = "/home3/oml4h/PLM_SARS-CoV-2/Sequences/H3N2_canonical.fa"

# Output
OUT_DIR = "/home3/oml4h/PLM_SARS-CoV-2/Results/PLANT_Rankings"
if TEST_MODE:
    OUT_DIR = os.path.join(OUT_DIR, "test_mode")
os.makedirs(OUT_DIR, exist_ok=True)

# Immunity Model Config
INCLUDE_VACCINE = True
CURRENT_YEAR = 2025.5

# Historic Centroids (from flu_model.py)
HISTORY = {
    2019: (2.49067, -2.0354, -1.3839),
    2020: (2.91758, -1.31161, -1.46984),
    2021: (3.42502, 2.18306, -2.05458),
    2022: (3.47224, 2.19141, -1.61720),
    2023: (3.29408, 2.84147, -0.84808),
    2024: (3.19627, 3.24367, -0.61533) 
}

# Population Distribution [Naive, 2019...2024, Vacc] (from flu_model.py)
POP_DIST = [
    5_000_000,   # Naive
    12_000_000,  # Last inf 2019
    2_000_000,   # Last inf 2020
    5_000_000,   # Last inf 2021
    8_000_000,   # Last inf 2022
    10_000_000,  # Last inf 2023
    10_000_000,  # Last inf 2024
    15_000_000   # Current Season Vaccinated
]

VACCINE_COORD = (3.011719, 3.59375, -0.34155) 

# Target Sites (1-based indices in the TRIMMED sequence / HA1)
# If TEST_MODE is True, use restricted sites. Else, will be set in main()
TEST_SITES = [158, 159, 160] 

# Reference Sequence for Trimming (from Plant.run.py)
# Use a J.2 reference sequence (329 aa) for alignment
REFERENCE_SEQ = "QKIPGNDNSTATLCLGHHAVPNGTIVKTITNDRIEVTNATELVQNSSIGKICNSPHQILDGGNCTLIDALLGDPQCDGFQNKEWDLFVERSRANSSCYPYDVPDYASLRSLVASSGTLEFKDESFNWTGVKQNGKSSACKRGSSSSFFSRLNWLTSLNNIYPAQNVTMPNKEQFDKLYIWGVHHPDTDKNQFSLFAQSSGRITVSTTRSQQAVIPNIGSRPRVRDIPSRISIYWTIVKPGDILLINSTGNLIAPRGYFKIRSGKSSIMRSDAPIGECKSECITPNGSIPNDKPFQNVNRITYGACPRYVKQSTLKLATGMRNVPEKQTR"
TARGET_LENGTH = 329

# ==========================================
# 2. MODEL CLASSES
# ==========================================

class WeightedDistanceModel:
    def __init__(self, history, pop_distribution, vaccine_coord=None, current_year=2025.5, waning_rate=0.2):
        self.history = history
        self.vaccine_coord = vaccine_coord
        self.current_year = current_year
        
        # Map populations to years (indices based on POP_DIST provided in script)
        # POP_DIST = [Naive, 2019, 2020, 2021, 2022, 2023, 2024, Vacc]
        self.pop_map = {
            2019: pop_distribution[1],
            2020: pop_distribution[2],
            2021: pop_distribution[3],
            2022: pop_distribution[4],
            2023: pop_distribution[5],
            2024: pop_distribution[6]
        }
        self.vacc_pop = pop_distribution[7]
        
        # Calculate Weights
        self.weights = {}
        total_weight_score = 0
        
        # 1. Historical Strains
        for year, pop in self.pop_map.items():
            # Waning: exp(-rate * years_elapsed)
            years_elapsed = self.current_year - year
            # Using exponential decay for immune pressure
            # Higher weight = More recent/More population = More pressure to escape
            weight_score = pop * np.exp(-waning_rate * years_elapsed)
            self.weights[year] = weight_score
            total_weight_score += weight_score
            
        # 2. Vaccine
        if self.vaccine_coord is not None:
            # Assume vaccine is effectively "current" (0.5 years old or similar)
            weight_score = self.vacc_pop * np.exp(-waning_rate * 0.5)
            self.weights['vaccine'] = weight_score
            total_weight_score += weight_score
            
        # Normalize
        self.normalized_weights = {k: v / total_weight_score for k, v in self.weights.items()}
        
    def print_weights(self):
        print("Calculated Weights for Distance Metric:")
        for k, v in sorted(self.normalized_weights.items(), key=lambda x: str(x[0])):
            print(f"  {k}: {v:.3f}")

    def calculate_weighted_distance(self, current_strain_coord):
        """
        Calculates the weighted Euclidean distance from historical strains and vaccine.
        Higher Distance = Better Escape.
        """
        curr = np.array(current_strain_coord)
        total_dist = 0
        
        # History
        for year, weight in self.normalized_weights.items():
            if year == 'vaccine': continue
            hist_coord = np.array(self.history[year])
            dist = np.linalg.norm(curr - hist_coord)
            total_dist += dist * weight
            
        # Vaccine
        if self.vaccine_coord is not None and 'vaccine' in self.normalized_weights:
            vacc_coord = np.array(self.vaccine_coord)
            dist = np.linalg.norm(curr - vacc_coord)
            total_dist += dist * self.normalized_weights['vaccine']
            
        return total_dist

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def get_observed_mutations(seq_file, h3_map, parent_id="J.2.4", child_id="J.2.4.1", trim_signal=True, limit_length=None):
    """
    Find mutations between parent and child lineages.
    Returns a list of mutation strings (e.g., "N159Y").
    
    trim_signal: If True, removes first 16 AA (Signal Peptide).
    limit_length: If set, limits the comparison to this length (e.g. 329 for HA1).
    """
    print(f"Searching for mutations between {parent_id} and {child_id} in {seq_file}...")
    parent_seq = None
    child_seq = None
    
    for record in SeqIO.parse(seq_file, "fasta"):
        header = record.description
        if header.endswith(f"|{parent_id}"):
            parent_seq = str(record.seq)
        elif header.endswith(f"|{child_id}"):
            child_seq = str(record.seq)
            
    if parent_seq is None or child_seq is None:
        print(f"Warning: Could not find both sequences. Parent found: {parent_seq is not None}, Child found: {child_seq is not None}")
        return []
        
    # Use get_mutations from Functions_HuggingFace
    # It expects aligned sequences. Assuming they are aligned or same length.
    # If lengths differ, we might need to align or trim.
    # For now, let's assume they are aligned as they come from the same alignment file.
    
    # However, get_mutations returns 1-based index based on the input string.
    # If we pass the full sequence, it returns 1-based index on full sequence.
    # We can then map this to canonical using h3_map.
    
    # Note: h3_map maps 0-based index of the QUERY (Parent) to Canonical Label.
    
    raw_mutations = get_mutations(parent_seq, child_seq)
    print(f"Raw mutations (Full Seq): {raw_mutations}")
    
    # Convert to Canonical
    # raw_mutations are like 'A16T' (1-based).
    # We need to convert to 0-based index to lookup in h3_map.
    
    canonical_muts = mutations_to_canonical(raw_mutations, h3_map)
    
    # Filter based on trim_signal and limit_length if needed
    # But since we are using canonical names, maybe we just return them all?
    # The user asked for "highlighting the mutations between J.2.4.1 and J.2.4".
    # If we are plotting HA1 only, we should filter for HA1 mutations.
    
    filtered_muts = []
    for mut in canonical_muts:
        # Check if it's in HA1 range or HA2
        # HA1 canonical usually numbers 1 to ~329.
        # HA2 starts with HA2:
        # SP starts with SP
        
        if mut.startswith("SP") and trim_signal:
            continue
            
        if mut.startswith("HA2:") and limit_length:
            # If limit_length is set (implying HA1 only), skip HA2
            continue
            
        filtered_muts.append(mut)
            
    print(f"Canonical mutations: {filtered_muts}")
    return filtered_muts

def get_all_mutations_from_matrix(prob_matrix, aa_labels, sequence, h3_map, start_offset=0):
    """
    Generate a DataFrame of all possible mutations and their probabilities from the matrix.
    prob_matrix: (20, L)
    sequence: String of length L
    start_offset: Number to add to 0-based index to get canonical position (e.g. 0 if sequence starts at 1)
    """
    # Use aa_labels from the matrix instead of hardcoding
    amino_acids = aa_labels
    all_muts = []
    
    L = prob_matrix.shape[1]
    if len(sequence) != L:
        print(f"Warning: Sequence length {len(sequence)} does not match matrix width {L}")
        return pd.DataFrame()
        
    for j in range(L):
        ref_aa = sequence[j]
        # pos = j + 1 + start_offset # Canonical Position (Old logic)
        
        # New Logic: Use h3_map
        # h3_map maps 0-based index of the full sequence to canonical label.
        # We need to know the 0-based index in the full sequence corresponding to j.
        # If start_offset is 0, then j is the index.
        # If start_offset is 16 (trimmed), then j+16 is the index.
        
        full_idx = j + start_offset
        
        if full_idx in h3_map:
            canon_pos = h3_map[full_idx]
        else:
            canon_pos = f"idx{full_idx}"
            
        # Get probabilities for this position
        probs = prob_matrix[:, j]
        
        for i, mut_aa in enumerate(aa_labels):
            if mut_aa not in amino_acids: continue
            if mut_aa == ref_aa: continue
            
            p = probs[i]
            if p > 0:
                log_p = np.log10(p)
            else:
                log_p = -10
                
            # Construct Canonical Name
            if str(canon_pos).startswith("HA2:"):
                # HA2:49 -> HA2:A49T
                # Extract number
                parts = str(canon_pos).split(':')
                mut_str = f"HA2:{ref_aa}{parts[1]}{mut_aa}"
            else:
                mut_str = f"{ref_aa}{canon_pos}{mut_aa}"
            
            all_muts.append({
                "Mutation": mut_str,
                "Probability": p,
                "Log10_Prob": log_p,
                "Position": canon_pos
            })
            
    df = pd.DataFrame(all_muts)
    df = df.sort_values("Probability", ascending=False)
    df['Rank'] = range(1, len(df) + 1)
    return df

def load_plant_model(device):
    print("Loading PLANT model...")
    # Encoders
    def find_file(patterns):
        for pat in patterns:
            paths = glob.glob(os.path.join(REPO_DIR, "**", pat), recursive=True)
            if len(paths) > 0:
                return paths[0]
        return None

    virus_enc_path = find_file(["virus_encoder.joblib", "**/virus_encoder.joblib"])
    ref_enc_path   = find_file(["ref_encoder.joblib",   "**/ref_encoder.joblib"])
    vp_enc_path    = find_file(["vp_encoder.joblib",    "**/vp_encoder.joblib"])
    rp_enc_path    = find_file(["rp_encoder.joblib",    "**/rp_encoder.joblib"])

    ohe_v  = joblib.load(virus_enc_path)
    ohe_r  = joblib.load(ref_enc_path)
    ohe_vp = joblib.load(vp_enc_path)
    ohe_rp = joblib.load(rp_enc_path)
    set_encoders(ohe_v, ohe_r, ohe_vp, ohe_rp)

    # Model
    esm_config = EsmConfig.from_pretrained(MODEL_NAME, use_safetensors=True)
    model = semanticESM.from_pretrained(
        CKPT_DIR,
        config=esm_config,
        esm_model_name=MODEL_NAME,
        intermediate_dim=256,
        intermediate_dim_encoder=64,
    )
    model.to(device)
    model.eval()
    model.half() # FP16
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer

def trim_to_target_length(seq, reference=REFERENCE_SEQ, target_len=TARGET_LENGTH):
    """Wrapper around shared helper in Functions_HuggingFace."""
    return plant_trim_to_target_length(seq, reference, target_len, return_start_pos=True)

def generate_mutants(trimmed_sequence, start_pos_in_full, prob_matrix, aa_labels, target_sites, h3_map):
    """
    Generate mutants for the trimmed sequence at specified sites.
    prob_matrix: (20, Full_Length)
    target_sites: 1-based indices in trimmed sequence
    """
    # Use aa_labels from the matrix instead of hardcoding
    amino_acids = aa_labels 
    mutants = []
    mutant_names = []
    probs = []
    
    # Add original
    mutants.append(trimmed_sequence)
    mutant_names.append("Original")
    probs.append(1.0)
    
    print(f"Generating mutants for {len(target_sites)} sites...")
    
    for site in tqdm(target_sites, desc="Generating Mutants"):
        # 1-based index in trimmed sequence -> 0-based index
        idx_trimmed = site - 1
        
        if idx_trimmed < 0 or idx_trimmed >= len(trimmed_sequence):
            # print(f"Warning: Site {site} out of bounds for trimmed sequence length {len(trimmed_sequence)}")
            continue
            
        ref_aa = trimmed_sequence[idx_trimmed]
        
        # Map to full sequence index for probability
        idx_full = start_pos_in_full + idx_trimmed
        
        if idx_full >= prob_matrix.shape[1]:
            # print(f"Warning: Mapped index {idx_full} out of bounds for probability matrix length {prob_matrix.shape[1]}")
            continue
            
        # Get Canonical Position
        if idx_full in h3_map:
            canon_pos = h3_map[idx_full]
        else:
            canon_pos = f"idx{idx_full}"
            
        # Get probabilities for this position
        # prob_matrix is (20, L)
        pos_probs = prob_matrix[:, idx_full]
        
        for i, mut_aa in enumerate(aa_labels):
            if mut_aa not in amino_acids: continue
            if mut_aa == ref_aa: continue
            
            # Create mutant
            mut_seq = trimmed_sequence[:idx_trimmed] + mut_aa + trimmed_sequence[idx_trimmed+1:]
            
            # Construct Canonical Name
            if str(canon_pos).startswith("HA2:"):
                # HA2:49 -> HA2:A49T
                parts = str(canon_pos).split(':')
                mut_name = f"HA2:{ref_aa}{parts[1]}{mut_aa}"
            else:
                mut_name = f"{ref_aa}{canon_pos}{mut_aa}"
            
            mutants.append(mut_seq)
            mutant_names.append(mut_name)
            
            # Get probability
            # aa_labels should match rows of prob_matrix
            p = pos_probs[i]
            probs.append(p)
            
    return mutants, mutant_names, probs

# ==========================================
# 4. MAIN EXECUTION
# ==========================================

def run_analysis_mode(mode, X, Y, Z_coord, mutant_names, probs, obs_muts_full, obs_muts_ha1, prob_matrix, aa_labels, base_sequence, h3_map):
    print(f"\n--- Running Analysis Mode: {mode} ---")
    
    # Output directory for this mode
    mode_out_dir = os.path.join(OUT_DIR, mode)
    os.makedirs(mode_out_dir, exist_ok=True)
    
    # Get Original Coordinates
    orig_idx = mutant_names.index("Original")
    orig_coord = (X[orig_idx], Y[orig_idx], Z_coord[orig_idx])
    
    distance_model = None
    orig_dist = 0.0
    
    if mode == "weighted":
        # Use WeightedDistanceModel instead of AntigenicSeirModel
        # Waning rate of 0.2 means ~18% decay per year (exp(-0.2))
        distance_model = WeightedDistanceModel(HISTORY, POP_DIST, VACCINE_COORD if INCLUDE_VACCINE else None, current_year=CURRENT_YEAR, waning_rate=0.2)
        orig_dist = distance_model.calculate_weighted_distance(orig_coord)
        print(f"Original Weighted Distance: {orig_dist:.4f}")
    elif mode == "reference":
        print("Using distance from Original strain as metric.")
        orig_dist = 0.0
        
    results = []
    for i, name in enumerate(mutant_names):
        coord = (X[i], Y[i], Z_coord[i])
        curr_arr = np.array(coord)
        
        if mode == "weighted":
            w_dist = distance_model.calculate_weighted_distance(coord)
        else: # reference
            w_dist = np.linalg.norm(curr_arr - np.array(orig_coord))
            
        # Relative Escape (Difference from Original)
        escape_impact = w_dist - orig_dist
        
        # Probability
        prob = probs[i]
        
        # Composite Score
        if prob > 0:
            log_prob = np.log10(prob)
        else:
            log_prob = -10 # Floor
            
        # Composite: Escape Impact + log10(prob)
        composite = escape_impact + log_prob 
        
        # Individual Distances
        dists = {}
        for year, hist_coord in HISTORY.items():
             dists[f"Dist_{year}"] = np.linalg.norm(curr_arr - np.array(hist_coord))
             
        if INCLUDE_VACCINE and VACCINE_COORD is not None:
             dists["Dist_Vaccine"] = np.linalg.norm(curr_arr - np.array(VACCINE_COORD))

        row = {
            "Mutation": name,
            "Probability": prob,
            "Log10_Prob": log_prob,
            "Weighted_Distance": w_dist,
            "Escape_Impact": escape_impact,
            "Composite_Score": composite,
            "X": X[i],
            "Y": Y[i],
            "Z": Z_coord[i]
        }
        row.update(dists)
        results.append(row)
        
    # 6. Export
    df_all = pd.DataFrame(results)
    
    # Separate Original and Mutants
    df_original = df_all[df_all["Mutation"] == "Original"].copy()
    df = df_all[df_all["Mutation"] != "Original"].copy()
    
    # Calculate Experimental Metrics for Mutants
    # Metric 1: Standard (Composite) - Already done
    
    # Metric 2: Expected Drift (Escape * Prob)
    df['Metric_Expected_Drift'] = df['Escape_Impact'] * df['Probability']
    
    # Metric 4: Risk-Taking (Escape + 0.5 * log10(Prob))
    df['Metric_Risk_Taking'] = df['Escape_Impact'] + (0.5 * df['Log10_Prob'])
    
    # Sort by Composite
    df = df.sort_values("Composite_Score", ascending=False)
    
    # Add Rank
    df['Rank'] = range(1, len(df) + 1)
    
    # Save Standard Subset (No Original)
    out_path = os.path.join(mode_out_dir, f"{BASE_LINEAGE_NAME}_PLANT_rankings_subset.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved rankings to {out_path}")
    
    # Save Subset with Reference (Original at Top)
    # Add missing columns to Original for consistency (Metrics will be NaN or 0)
    for col in ['Metric_Expected_Drift', 'Metric_Risk_Taking', 'Rank']:
        if col not in df_original.columns:
            df_original[col] = 0 if col == 'Rank' else np.nan
            
    df_plus_ref = pd.concat([df_original, df], ignore_index=True)
    out_path_ref = os.path.join(mode_out_dir, f"{BASE_LINEAGE_NAME}_PLANT_rankings_subset_plus_reference.csv")
    df_plus_ref.to_csv(out_path_ref, index=False)
    print(f"Saved rankings with reference to {out_path_ref}")
    
    # 7. Plotting
    print("Generating plots...")
    
    # --- Figure 1: Main Panel (3 Subplots) ---
    fig1, axes1 = plt.subplots(1, 3, figsize=(20, 6))
    
    # Subplot 1: Most Probable Mutations (Full Sequence)
    # We need the full probability ranking
    # prob_matrix is (20, 567). base_sequence is 567.
    # We trim the first 16 to match "Canonical Flu Coordinates" (Signal Peptide removal)
    full_prob_matrix = prob_matrix[:, 16:]
    full_seq_trimmed = base_sequence[16:]
    
    df_all_probs = get_all_mutations_from_matrix(full_prob_matrix, aa_labels, full_seq_trimmed, h3_map, start_offset=16)
    
    # Plot
    axes1[0].plot(df_all_probs['Rank'], df_all_probs['Log10_Prob'], color='blue', linewidth=0.8, alpha=0.5)
    axes1[0].set_title("Most Probable Mutations (Full HA)")
    axes1[0].set_xlabel("Rank")
    axes1[0].set_ylabel("log10(Probability)")
    
    # Highlight Observed (Full)
    obs_data_full = df_all_probs[df_all_probs['Mutation'].isin(obs_muts_full)]
    if len(obs_data_full) > 0:
        axes1[0].scatter(obs_data_full['Rank'], obs_data_full['Log10_Prob'], color='red', s=50, zorder=5)
        texts = [axes1[0].text(r['Rank'], r['Log10_Prob'], r['Mutation'], fontsize=8) for _, r in obs_data_full.iterrows()]
        try:
            from adjustText import adjust_text
            adjust_text(texts, ax=axes1[0], arrowprops=dict(arrowstyle='-', color='black', lw=0.5))
        except ImportError: pass
        
    # Subplot 2: Ranked Escape (HA1 Only)
    # Sort by Weighted_Distance descending
    df_escape = df.sort_values("Weighted_Distance", ascending=False).reset_index(drop=True)
    df_escape['Rank_Escape'] = df_escape.index + 1
    
    axes1[1].plot(df_escape['Rank_Escape'], df_escape['Weighted_Distance'], color='green', linewidth=0.8, alpha=0.5)
    axes1[1].set_title("Ranked Escape (HA1 Only)")
    axes1[1].set_xlabel("Rank")
    axes1[1].set_ylabel("Weighted Distance")
    
    # Highlight Observed (HA1)
    obs_data_esc = df_escape[df_escape['Mutation'].isin(obs_muts_ha1)]
    if len(obs_data_esc) > 0:
        axes1[1].scatter(obs_data_esc['Rank_Escape'], obs_data_esc['Weighted_Distance'], color='red', s=50, zorder=5)
        texts = [axes1[1].text(r['Rank_Escape'], r['Weighted_Distance'], r['Mutation'], fontsize=8) for _, r in obs_data_esc.iterrows()]
        try:
            from adjustText import adjust_text
            adjust_text(texts, ax=axes1[1], arrowprops=dict(arrowstyle='-', color='black', lw=0.5))
        except ImportError: pass

    # Subplot 3: Composite Score (HA1 Only) - Current
    # df is already sorted by Composite
    axes1[2].plot(df['Rank'], df['Composite_Score'], color='purple', linewidth=0.8, alpha=0.5)
    axes1[2].set_title("Composite Score (HA1 Only)")
    axes1[2].set_xlabel("Rank")
    axes1[2].set_ylabel("Score (Escape + log10(Prob))")
    
    # Highlight Observed (HA1)
    obs_data_comp = df[df['Mutation'].isin(obs_muts_ha1)]
    if len(obs_data_comp) > 0:
        axes1[2].scatter(obs_data_comp['Rank'], obs_data_comp['Composite_Score'], color='red', s=50, zorder=5)
        texts = [axes1[2].text(r['Rank'], r['Composite_Score'], r['Mutation'], fontsize=8) for _, r in obs_data_comp.iterrows()]
        try:
            from adjustText import adjust_text
            adjust_text(texts, ax=axes1[2], arrowprops=dict(arrowstyle='-', color='black', lw=0.5))
        except ImportError: pass
        
    plt.tight_layout()
    fig1.savefig(os.path.join(mode_out_dir, f"{BASE_LINEAGE_NAME}_panel_analysis.png"), dpi=300)
    print(f"Saved panel analysis to {os.path.join(mode_out_dir, f'{BASE_LINEAGE_NAME}_panel_analysis.png')}")
    
    # --- Figure 2: Experimental Combinations (4 Subplots) ---
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Standard (Composite)
    axes2[0,0].plot(df['Rank'], df['Composite_Score'], color='purple', alpha=0.5)
    axes2[0,0].set_title("Standard: Escape + log10(Prob)")
    if len(obs_data_comp) > 0:
        axes2[0,0].scatter(obs_data_comp['Rank'], obs_data_comp['Composite_Score'], color='red', s=50)
        
    # 2. Expected Drift (Escape * Prob)
    df_drift = df.sort_values("Metric_Expected_Drift", ascending=False).reset_index(drop=True)
    df_drift['Rank_Drift'] = df_drift.index + 1
    axes2[0,1].plot(df_drift['Rank_Drift'], df_drift['Metric_Expected_Drift'], color='orange', alpha=0.5)
    axes2[0,1].set_title("Expected Drift: Escape * Prob")
    
    obs_data_drift = df_drift[df_drift['Mutation'].isin(obs_muts_ha1)]
    if len(obs_data_drift) > 0:
        axes2[0,1].scatter(obs_data_drift['Rank_Drift'], obs_data_drift['Metric_Expected_Drift'], color='red', s=50)
        texts = [axes2[0,1].text(r['Rank_Drift'], r['Metric_Expected_Drift'], r['Mutation'], fontsize=8) for _, r in obs_data_drift.iterrows()]
        try:
            from adjustText import adjust_text
            adjust_text(texts, ax=axes2[0,1], arrowprops=dict(arrowstyle='-', color='black', lw=0.5))
        except ImportError: pass

    # 3. Scatter: Escape vs Log10(Prob)
    axes2[1,0].scatter(df['Log10_Prob'], df['Escape_Impact'], alpha=0.3, color='gray')
    axes2[1,0].set_title("Scatter: Escape vs Probability")
    axes2[1,0].set_xlabel("log10(Probability)")
    axes2[1,0].set_ylabel("Escape Impact")
    
    if len(obs_data_comp) > 0:
        axes2[1,0].scatter(obs_data_comp['Log10_Prob'], obs_data_comp['Escape_Impact'], color='red', s=50)
        texts = [axes2[1,0].text(r['Log10_Prob'], r['Escape_Impact'], r['Mutation'], fontsize=8) for _, r in obs_data_comp.iterrows()]
        try:
            from adjustText import adjust_text
            adjust_text(texts, ax=axes2[1,0], arrowprops=dict(arrowstyle='-', color='black', lw=0.5))
        except ImportError: pass

    # 4. Risk Taking (Escape + 0.5 * log10(Prob))
    df_risk = df.sort_values("Metric_Risk_Taking", ascending=False).reset_index(drop=True)
    df_risk['Rank_Risk'] = df_risk.index + 1
    axes2[1,1].plot(df_risk['Rank_Risk'], df_risk['Metric_Risk_Taking'], color='magenta', alpha=0.5)
    axes2[1,1].set_title("Risk Taking: Escape + 0.5 * log10(Prob)")
    
    obs_data_risk = df_risk[df_risk['Mutation'].isin(obs_muts_ha1)]
    if len(obs_data_risk) > 0:
        axes2[1,1].scatter(obs_data_risk['Rank_Risk'], obs_data_risk['Metric_Risk_Taking'], color='red', s=50)
        texts = [axes2[1,1].text(r['Rank_Risk'], r['Metric_Risk_Taking'], r['Mutation'], fontsize=8) for _, r in obs_data_risk.iterrows()]
        try:
            from adjustText import adjust_text
            adjust_text(texts, ax=axes2[1,1], arrowprops=dict(arrowstyle='-', color='black', lw=0.5))
        except ImportError: pass
        
    plt.tight_layout()
    fig2.savefig(os.path.join(mode_out_dir, f"{BASE_LINEAGE_NAME}_experimental_metrics.png"), dpi=300)
    print(f"Saved experimental metrics to {os.path.join(mode_out_dir, f'{BASE_LINEAGE_NAME}_experimental_metrics.png')}")
    
    if mode == "weighted":
        print("\nWeights used for Weighted Distance:")
        distance_model.print_weights()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Probability Matrix & Sequence
    print(f"Loading probability matrix from {PROB_MATRIX_PATH}")
    # Read with header=None to get the sequence from the first row
    df_prob = pd.read_csv(PROB_MATRIX_PATH, header=None, index_col=0)
    
    # Extract sequence from the first row (header)
    # The first column (index 0) is the index name (NaN or empty), subsequent columns are the sequence
    seq_chars = df_prob.iloc[0].values
    base_sequence = "".join(seq_chars)
    print(f"Extracted sequence from matrix header (length {len(base_sequence)})")
    
    # Extract probabilities
    prob_matrix = df_prob.iloc[1:].values.astype(float)
    aa_labels = df_prob.index[1:].tolist()
    
    print(f"Probability matrix shape: {prob_matrix.shape}")
    print(f"Amino acid labels: {aa_labels}")
    
    # 2. Trim Sequence
    print("Trimming sequence to HA1 (329 AA)...")
    trimmed_seq, start_pos = trim_to_target_length(base_sequence)
    if trimmed_seq is None:
        print("Error: Could not trim sequence to target length.")
        sys.exit(1)
    print(f"Trimmed sequence length: {len(trimmed_seq)}")
    print(f"Start position in full sequence: {start_pos}")
    
    # Determine Target Sites
    if TEST_MODE:
        print(f"TEST MODE ON: Using restricted sites {TEST_SITES}")
        target_sites = TEST_SITES
    else:
        print("TEST MODE OFF: Scanning all sites in trimmed sequence.")
        target_sites = list(range(1, len(trimmed_seq) + 1))
    
    # --- Create H3 Numbering Map ---
    print(f"Loading canonical reference from {CANONICAL_REF_PATH}")
    try:
        canonical_ref_seq = str(SeqIO.read(CANONICAL_REF_PATH, "fasta").seq)
        print("Creating H3 numbering map...")
        # Map 0-based index of base_sequence to Canonical Label
        # HA2 starts at 330 (after 329 AA of HA1)
        h3_map = create_h3_numbering_map(base_sequence, canonical_ref_seq, HA2_start=330)
    except Exception as e:
        print(f"Error creating H3 map: {e}")
        print("Falling back to index-based numbering.")
        h3_map = {}

    # 3. Generate Mutants
    mutant_seqs, mutant_names, probs = generate_mutants(
        trimmed_seq, 
        start_pos, 
        prob_matrix, 
        aa_labels, 
        target_sites,
        h3_map
    )
    print(f"Generated {len(mutant_seqs)} sequences (including original).")
    
    # 4. Get Embeddings (PLANT)
    plant_model, plant_tokenizer = load_plant_model(device)
    
    print("Embedding sequences with PLANT...")
    BATCH_SIZE = 64
    
    # Tokenize
    enc_seqs = tokenize_sequences(mutant_seqs, plant_tokenizer, len(trimmed_seq))
    dataset = TextDataset(enc_seqs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Add progress bar to embedding loop
    # embed_sequences in plant/inference.py doesn't have a progress bar, so we wrap the loader or modify the function call if possible.
    # Since we can't easily modify the imported function, we'll just print progress if we were iterating manually.
    # But embed_sequences takes the loader.
    # Let's check if we can wrap the loader with tqdm.
    # embed_sequences iterates over the loader. If we wrap it, it should show progress.
    
    loader_with_progress = tqdm(loader, desc="Embedding Batches")
    Z = embed_sequences(plant_model, loader_with_progress, use_fp16=True)
    
    # Scale
    X = Z[:, 0] * SCALE_FACTOR
    Y = Z[:, 1] * SCALE_FACTOR
    Z_coord = Z[:, 2] * SCALE_FACTOR
    
    # 5. Calculate Escape
    print("Calculating escape scores...")
    
    # Get Observed Mutations
    # 1. Full Sequence (for Probability Plot) - Trim signal (16), No length limit
    obs_muts_full = get_observed_mutations(SEQ_FILE_PATH, h3_map, trim_signal=True, limit_length=None)
    
    # 2. HA1 Only (for Escape/Composite Plots) - Trim signal (16), Limit to 329
    obs_muts_ha1 = get_observed_mutations(SEQ_FILE_PATH, h3_map, trim_signal=True, limit_length=329)
    
    # Run Mode 1: Weighted Centroid (Standard)
    run_analysis_mode("weighted", X, Y, Z_coord, mutant_names, probs, obs_muts_full, obs_muts_ha1, prob_matrix, aa_labels, base_sequence, h3_map)
    
    # Run Mode 2: Reference Only
    run_analysis_mode("reference", X, Y, Z_coord, mutant_names, probs, obs_muts_full, obs_muts_ha1, prob_matrix, aa_labels, base_sequence, h3_map)

if __name__ == "__main__":
    main()
