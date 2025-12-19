
import sys
import os
import pandas as pd
import numpy as np
import torch
import joblib
import glob
from transformers import AutoTokenizer, EsmConfig

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
# PLANT Model Config
REPO_DIR = "/home3/oml4h/hugging_face_downloads/PLANT_model"
CKPT_DIR = os.path.join(REPO_DIR, "variants/PLANT_fixed")
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
SCALE_FACTOR = 8

# Input Data
# QUERY_PATH removed - we get sequence from matrix header
PROB_MATRIX_PATH = "/home3/oml4h/PLM_SARS-CoV-2/Results/test/J.2.4/J.2.4_probability_matrix.csv"
BASE_LINEAGE_NAME = "J.2.4" # Hardcoded for output naming since we aren't parsing it

# Output
OUT_DIR = "/home3/oml4h/PLM_SARS-CoV-2/Results/PLANT_Rankings"
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
# User requested key sites like 158, 159, 160
TARGET_SITES = [158, 159, 160] 

# Reference Sequence for Trimming (from Plant.run.py)
REFERENCE_SEQ = "QKIPGNDNSTATLCLGHHAVPNGTIVKTITNDRIEVTNATELVQNSSIGEICGSPHQILDGGNCTLIDALLGDPQCDGFQNKEWDLFVERSRANSNCYPYDVPGYASLRSLVASSGTLEFKNESFNWTGVKQNGTSSACIRGSSSSFFSRLNWLTSINNIYPAQNVTMPNKEQFDKLYIWGVHHPDTDKNQISLFAQSSGRITVSTKRSQQAVIPNIGSRPRIRDIPSRISIYWTIVKPGDILLINSTGNLIAPRGYFKIRNGKSSIMRSDAPIGRCKSECITPNGSIPNDKPFQNVNRITYGACPRYVKQSTLKLATGMRNVPEKQTR"
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
    """
    Trim sequences to match the target length.
    Returns (trimmed_seq, start_index_in_original)
    """
    if pd.isna(seq) or not isinstance(seq, str):
        return None, -1
    
    seq_len = len(seq)
    
    if seq_len == target_len:
        return seq, 0
    elif seq_len < target_len:
        return None, -1
    else:
        # Longer than target - find where the HA1 region starts
        best_score = 0
        best_start = 0
        
        window = min(50, target_len)
        ref_window = reference[:window]
        
        for i in range(seq_len - target_len + 1):
            seq_window = seq[i:i+window]
            score = sum(1 for a, b in zip(ref_window, seq_window) if a == b)
            if score > best_score:
                best_score = score
                best_start = i
        
        return seq[best_start:best_start + target_len], best_start

def generate_mutants(trimmed_sequence, start_pos_in_full, prob_matrix, aa_labels, target_sites):
    """
    Generate mutants for the trimmed sequence at specified sites.
    prob_matrix: (20, Full_Length)
    target_sites: 1-based indices in trimmed sequence
    """
    amino_acids = ["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
    mutants = []
    mutant_names = []
    probs = []
    
    # Add original
    mutants.append(trimmed_sequence)
    mutant_names.append("Original")
    probs.append(1.0)
    
    print(f"Generating mutants for sites: {target_sites}")
    
    for site in target_sites:
        # 1-based index in trimmed sequence -> 0-based index
        idx_trimmed = site - 1
        
        if idx_trimmed < 0 or idx_trimmed >= len(trimmed_sequence):
            print(f"Warning: Site {site} out of bounds for trimmed sequence length {len(trimmed_sequence)}")
            continue
            
        ref_aa = trimmed_sequence[idx_trimmed]
        
        # Map to full sequence index for probability
        idx_full = start_pos_in_full + idx_trimmed
        
        if idx_full >= prob_matrix.shape[1]:
            print(f"Warning: Mapped index {idx_full} out of bounds for probability matrix length {prob_matrix.shape[1]}")
            continue
            
        # Get probabilities for this position
        # prob_matrix is (20, L)
        pos_probs = prob_matrix[:, idx_full]
        
        for i, mut_aa in enumerate(aa_labels):
            if mut_aa not in amino_acids: continue
            if mut_aa == ref_aa: continue
            
            # Create mutant
            mut_seq = trimmed_sequence[:idx_trimmed] + mut_aa + trimmed_sequence[idx_trimmed+1:]
            mut_name = f"{ref_aa}{site}{mut_aa}"
            
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
    
    # 3. Generate Mutants
    mutant_seqs, mutant_names, probs = generate_mutants(
        trimmed_seq, 
        start_pos, 
        prob_matrix, 
        aa_labels, 
        TARGET_SITES
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
    
    Z = embed_sequences(plant_model, loader, use_fp16=True)
    
    # Scale
    X = Z[:, 0] * SCALE_FACTOR
    Y = Z[:, 1] * SCALE_FACTOR
    Z_coord = Z[:, 2] * SCALE_FACTOR
    
    # 5. Calculate Escape
    print("Calculating escape scores...")
    # Use WeightedDistanceModel instead of AntigenicSeirModel
    # Waning rate of 0.2 means ~18% decay per year (exp(-0.2))
    distance_model = WeightedDistanceModel(HISTORY, POP_DIST, VACCINE_COORD if INCLUDE_VACCINE else None, current_year=CURRENT_YEAR, waning_rate=0.2)
    
    # Calculate for Original first
    orig_idx = mutant_names.index("Original")
    orig_coord = (X[orig_idx], Y[orig_idx], Z_coord[orig_idx])
    
    orig_dist = distance_model.calculate_weighted_distance(orig_coord)
    print(f"Original Weighted Distance: {orig_dist:.4f}")
    
    results = []
    for i, name in enumerate(mutant_names):
        coord = (X[i], Y[i], Z_coord[i])
        
        # Weighted Distance
        w_dist = distance_model.calculate_weighted_distance(coord)
        
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
        # Since escape_impact is small (distance), and log_prob is negative (-1 to -4),
        # we might want to weight them. For now, simple sum.
        composite = escape_impact + log_prob 
        
        results.append({
            "Mutation": name,
            "Probability": prob,
            "Log10_Prob": log_prob,
            "Weighted_Distance": w_dist,
            "Escape_Impact": escape_impact,
            "Composite_Score": composite,
            "X": X[i],
            "Y": Y[i],
            "Z": Z_coord[i]
        })
        
    # 6. Export
    df = pd.DataFrame(results)
    # Filter out Original
    df = df[df["Mutation"] != "Original"]
    
    # Sort by Composite
    df = df.sort_values("Composite_Score", ascending=False)
    
    out_path = os.path.join(OUT_DIR, f"{BASE_LINEAGE_NAME}_PLANT_rankings_subset.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved rankings to {out_path}")
    print(df.head(20))

if __name__ == "__main__":
    main()
