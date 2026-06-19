#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Layer-wise Representation Probe: Reconstructing PLANT 3D Coordinates from PLM/ESM-C Layers
"""
import sys
sys.path.append("/home3/oml4h/PLM_SARS-CoV-2")

# --- BULLETPROOF DESCRIPTOR PATCH FOR ESM TOKENIZER CONFLICTS ---
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

# 1. Override tokens with functional properties containing dummy setters
tokens_to_patch = ["cls_token", "pad_token", "eos_token", "unk_token", "bos_token", "mask_token"]
for attr in tokens_to_patch:
    token_str = f"<{attr.split('_')[0]}>"
    setattr(EsmSequenceTokenizer, attr, property(
        fget=lambda self, t=token_str: t,
        fset=lambda self, val: None
    ))

# 2. Directly expose the token ID mappings to bypass the missing __getattr__ completely
setattr(EsmSequenceTokenizer, "pad_token_id", property(lambda self: self._get_token_id("<pad>")))
setattr(EsmSequenceTokenizer, "cls_token_id", property(lambda self: self._get_token_id("<cls>")))
setattr(EsmSequenceTokenizer, "eos_token_id", property(lambda self: self._get_token_id("<eos>")))
setattr(EsmSequenceTokenizer, "unk_token_id", property(lambda self: self._get_token_id("<unk>")))
setattr(EsmSequenceTokenizer, "mask_token_id", property(lambda self: self._get_token_id("<mask>")))
# ----------------------------------------------------------------


from Functions_HuggingFace import embed_sequence, ESMCAlphabetWrapper

import os
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoTokenizer, AutoModel, DataCollatorWithPadding
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Probe internal PLM/ESM-C layer representations via linear coordinate regression.")
    parser.add_argument(
        "--input_csv", 
        type=str, 
        required=True,
        help="Path to the filtered sequences CSV file containing sequence data ('seq')."
    )
    parser.add_argument(
        "--coords", 
        type=str, 
        required=True,
        help="Path to the PLANT embeddings file containing structural coordinates ('X', 'Y', 'Z')."
    )
    parser.add_argument(
        "--model_name_or_path", 
        type=str, 
        default="facebook/esm2_t33_650M_UR50D",
        help="Hugging Face model identification string or absolute local path to an ESM2/ESM-C checkpoint."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="/home3/oml4h/PLM_SARS-CoV-2/Results/PLANT_results/Layer_Analysis",
        help="Directory where target tabular metrics and trajectory plots will be written."
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for generating model hidden states.")
    parser.add_argument("--num_folds", type=int, default=5, help="Number of splits for Cross-Validation.")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed value for deterministic operations.")
    return parser.parse_args()

class SequenceDataset(Dataset):
    """Encapsulates sequence token indices mapping clean items for dynamic collation."""
    def __init__(self, encodings, targets):
        self.encodings = encodings
        self.targets = targets
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(self.encodings[key][idx]) for key in self.encodings}
        return item
        
    def __len__(self):
        return len(self.targets)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    model_safe_name = os.path.basename(args.model_name_or_path.rstrip("/"))
    print(f"Initialising layer coordinate-regression pipeline for model: {args.model_name_or_path}")
    
    # -------------------------------------------------------------------------
    # 1. Load and Merge Datasets
    # -------------------------------------------------------------------------
    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"Sequence input file not found at: {args.input_csv}")
    if not os.path.exists(args.coords):
        raise FileNotFoundError(f"Coordinate target file not found at: {args.coords}")
        
    df_input = pd.read_csv(args.input_csv)
    df_coords = pd.read_csv(args.coords)
    
    if "name" in df_input.columns and "name" in df_coords.columns:
        df = pd.merge(df_input, df_coords, on="name")
    else:
        shared_identifiers = [col for col in df_input.columns if col in df_coords.columns and col not in ["subclade", "seq"]]
        if shared_identifiers:
            df = pd.merge(df_input, df_coords, on=shared_identifiers[0])
        else:
            print("Warning: No matching identifier column discovered. Defaulting to positional alignment.")
            df = pd.concat([df_input.reset_index(drop=True), df_coords.reset_index(drop=True)], axis=1)
            
    df = df.dropna(subset=["X", "Y", "Z", "seq"]).reset_index(drop=True)
    
    sequences = df["seq"].tolist()
    y_targets = df[["X", "Y", "Z"]].values
    
    print(f"Loaded {len(sequences)} records with valid sequence and 3D coordinate combinations.")
    
    # -------------------------------------------------------------------------
    # 2. Map Fallback Hub Proxies for Local Directory Discrepancies
    # -------------------------------------------------------------------------
    model_lower = args.model_name_or_path.lower()
    is_esmc = "esmc" in model_lower or "evolutionaryscale" in model_lower
    
    if "300m" in model_lower:
        fallback_hub_model = "EvolutionaryScale/esmc-300m-2024-12"
    elif "6b" in model_lower or "600m" in model_lower:
        fallback_hub_model = "EvolutionaryScale/esmc-6b-2024-12"
    elif is_esmc:
        fallback_hub_model = "EvolutionaryScale/esmc-300m-2024-12"
    else:
        fallback_hub_model = "facebook/esm2_t33_650M_UR50D"

    # -------------------------------------------------------------------------
    # 3 & 4. Initialise Model Architecture via Custom Scope
    # -------------------------------------------------------------------------
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)
    print(f"Inference execution engine set to: {device}")
    
    model_lower = args.model_name_or_path.lower()
    is_esmc = "esmc" in model_lower or "evolutionaryscale" in model_lower or "esm_c" in model_lower
    
    
    if is_esmc:
        from esm.models.esmc import ESMC
        from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
        
        # 1. Load the model
        model = ESMC.from_pretrained(model_safe_name if "esmc" in model_safe_name else "esmc_300m").to(device)
        
        # 2. Build a stable, concrete tokenizer class that satisfies all requirements
        class StableESMCTokenizer(EsmSequenceTokenizer):
            @property
            def pad_token_id(self):
                return self._get_token_id("<pad>")
            @property
            def pad_token(self):
                return "<pad>"
            @property
            def mask_token(self):
                return "<mask>"
            @property
            def cls_token(self):
                return "<cls>"
            @property
            def eos_token(self):
                return "<eos>"
            @property
            def unk_token(self):
                return "<unk>"
                
            # Mute the Hugging Face initialization mutations
            def __setattr__(self, name, value):
                if name in ["cls_token", "pad_token", "eos_token", "unk_token", "bos_token", "mask_token"]:
                    return
                super().__setattr__(name, value)

        # 3. Instantiate it using the model's existing underlying vocabulary configuration
        stable_tokenizer = StableESMCTokenizer()
        
        # 4. Force-inject it directly into the model to satisfy model._tokenize()
        model.tokenizer = stable_tokenizer
        
        # 5. Connect it to your downstream pipeline components
        alphabet = ESMCAlphabetWrapper(stable_tokenizer)
        batch_converter = alphabet.get_batch_converter()
    else:
        # Standard HF/FAIR ESM Setup
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(args.model_name_or_path, output_hidden_states=True, trust_remote_code=True).to(device)
        alphabet = tokenizer
        batch_converter = lambda x: (None, None, tokenizer([s for _, s in x], padding=True, return_tensors="pt")["input_ids"])

    model.eval()
    # -------------------------------------------------------------------------
    # 5. Dynamic Sequence Length Extraction 
    # -------------------------------------------------------------------------
    if hasattr(model, "config") and hasattr(model.config, "max_position_embeddings"):
        plm_max_aa_length = model.config.max_position_embeddings
    elif is_esmc:
        plm_max_aa_length = 2048
    else:
        plm_max_aa_length = 1022
        
    print(f"Resolved model architecture parameters. Processing sequence truncation threshold at {plm_max_aa_length} AA.")

   # -------------------------------------------------------------------------
    # 6 & 7. Sequence Processing and Representation Harvesting Loop
    # -------------------------------------------------------------------------
    print(f"Extracting layer hidden states for {len(sequences)} records via embed_sequence...")
    
    # We need to find out how many layers the model has by running a single test probe
    test_res, _, _, _ = embed_sequence(sequences[0], model, device, model_layers=0, batch_converter=batch_converter, alphabet=alphabet)
    
    # Hidden states are contained inside results wrapper
    if is_esmc:
        num_extracted_layers = len(test_res.hidden_states)
    else:
        num_extracted_layers = len(getattr(test_res, "hidden_states", test_res[2]))

    # Initialize a dict of lists for each layer matrix
    layer_embeddings = {i: [] for i in range(num_extracted_layers)}

    for seq in sequences:
        # Iterate over every layer systematically to collect layers for the downstream regression loop
        for layer_idx in range(num_extracted_layers):
            _, _, base_mean_embedding, _ = embed_sequence(
                sequence=seq,
                model=model,
                device=device,
                model_layers=layer_idx,
                batch_converter=batch_converter,
                alphabet=alphabet
            )
            layer_embeddings[layer_idx].append(base_mean_embedding.numpy())

    # Convert lists to solid matrix configurations
    for layer_idx in layer_embeddings:
        layer_matrix = np.array(layer_embeddings[layer_idx])
        if not np.isfinite(layer_matrix).all():
            layer_matrix = np.nan_to_num(layer_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        layer_embeddings[layer_idx] = layer_matrix

    print(f"Extracted feature matrices across {num_extracted_layers} distinct layers.")
    # -------------------------------------------------------------------------
    # 8. Layer-wise Linear Coordinate Regression Probing
    # -------------------------------------------------------------------------
    layer_metrics = []
    
    for layer_idx in range(num_extracted_layers):
        X_features = layer_embeddings[layer_idx]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_features)
        
        regressor = Ridge(alpha=1.0, random_state=args.random_seed)
        kf_cv = KFold(n_splits=args.num_folds, shuffle=True, random_state=args.random_seed)
        
        cross_val_scores = cross_validate(
            regressor, 
            X_scaled, 
            y_targets, 
            cv=kf_cv, 
            scoring=["r2", "neg_mean_squared_error"], 
            n_jobs=-1
        )
        
        mean_r2 = np.mean(cross_val_scores["test_r2"])
        std_r2 = np.std(cross_val_scores["test_r2"])
        mean_mse = -np.mean(cross_val_scores["test_neg_mean_squared_error"])
        
        layer_metrics.append({
            "Layer": layer_idx,
            "Mean_R2": mean_r2,
            "Std_R2": std_r2,
            "Mean_MSE": mean_mse
        })
        
        print(f"Layer {layer_idx:02d} | Mean CV R² Score: {mean_r2:.4f} | Mean MSE: {mean_mse:.4f}")
        
    # -------------------------------------------------------------------------
    # 9. Export Statistical Tables and Diagnostic Plots
    # -------------------------------------------------------------------------
    summary_df = pd.DataFrame(layer_metrics)
    summary_export_path = os.path.join(args.output_dir, f"layer_coordinate_probe_metrics_{model_safe_name}.csv")
    summary_df.to_csv(summary_export_path, index=False)
    print(f"Saved metric output tables to: {summary_export_path}")
    
    optimal_row = summary_df.loc[summary_df["Mean_R2"].idxmax()]
    print("-" * 80)
    print(f"OPTIMAL COORDINATE RECONSTRUCTION LAYER IDENTIFIED FOR {model_safe_name}:")
    print(f"Layer Index: {int(optimal_row['Layer'])} (Mean R²: {optimal_row['Mean_R2']:.4f})")
    print("-" * 80)
    
    plt.figure(figsize=(11, 6))
    plt.plot(summary_df["Layer"], summary_df["Mean_R2"], marker="o", linewidth=2, color="#2ca02c", label="Mean R² Score")
    plt.fill_between(
        summary_df["Layer"], 
        summary_df["Mean_R2"] - summary_df["Std_R2"], 
        summary_df["Mean_R2"] + summary_df["Std_R2"], 
        alpha=0.15, 
        color="#2ca02c"
    )
    plt.axvline(x=optimal_row['Layer'], color="red", linestyle="--", alpha=0.7, label=f"Optimal Layer ({int(optimal_row['Layer'])})")
    
    plt.xlabel("Layer Index (Layer 0 = Input Token Embeddings)")
    plt.ylabel(f"Standard {args.num_folds}-Fold Cross-Validation R² (Variance Explained)")
    plt.title(f"Linear Recoverability of PLANT 3D Coordinates Across Model Layers\nModel: {model_safe_name}")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    plot_export_path = os.path.join(args.output_dir, f"layer_coordinate_trajectory_{model_safe_name}.png")
    plt.savefig(plot_export_path, dpi=300)
    plt.close()
    print(f"Diagnostic plot successfully exported to: {plot_export_path}")

if __name__ == "__main__":
    main()