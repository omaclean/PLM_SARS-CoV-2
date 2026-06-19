#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Layer-wise Representation Probe: Reconstructing PLANT 3D Coordinates from PLM/ESM-C Layers
"""
import sys
sys.path.append("/home3/oml4h/PLM_SARS-CoV-2")

# --- CONDITIONAL PATCH: ONLY APPLIES IF RUNNING ESM-C ---
if any(x in "".join(sys.argv).lower() for x in ["esmc", "esm_c", "300m", "6b"]):
    from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

    tokens_to_patch = ["cls_token", "pad_token", "eos_token", "unk_token", "bos_token", "mask_token"]
    for attr in tokens_to_patch:
        token_str = f"<{attr.split('_')[0]}>"
        setattr(EsmSequenceTokenizer, attr, property(
            fget=lambda self, t=token_str: t,
            fset=lambda self, val: None
        ))

    setattr(EsmSequenceTokenizer, "pad_token_id", property(lambda self: self._get_token_id("<pad>")))
    setattr(EsmSequenceTokenizer, "cls_token_id", property(lambda self: self._get_token_id("<cls>")))
    setattr(EsmSequenceTokenizer, "eos_token_id", property(lambda self: self._get_token_id("<eos>")))
    setattr(EsmSequenceTokenizer, "unk_token_id", property(lambda self: self._get_token_id("<unk>")))
    setattr(EsmSequenceTokenizer, "mask_token_id", property(lambda self: self._get_token_id("<mask>")))
# --------------------------------------------------------

from Functions_HuggingFace import embed_sequence, ESMCAlphabetWrapper

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
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
            raise ValueError(
                "CRITICAL DATA ERROR: No matching identifier column discovered between datasets. "
                "Execution halted to prevent flawed positional alignment."
            )
            
    df = df.dropna(subset=["X", "Y", "Z", "seq"]).reset_index(drop=True)
    
# -------------------------------------------------------------------------
    # 2. Framework Routing and Architecture Initialisation
    # -------------------------------------------------------------------------
    model_lower = args.model_name_or_path.lower()
    is_esmc = "esmc" in model_lower or "evolutionaryscale" in model_lower or "esm_c" in model_lower
    
    is_6b = "6b" in model_lower
    is_finetune = "checkpoint" in model_lower or "finetune" in model_lower
    
    # Route vanilla 300m/600m to native esm; route 6B and custom checkpoints to Hugging Face
    use_native_esmc = is_esmc and (not is_6b) and (not is_finetune)

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)
    print(f"Inference execution engine set to: {device}")

    # Runtime Alias Injection: Prevents KeyError: 'esmc' on legacy transformers versions
    if (not use_native_esmc) and is_esmc:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        from transformers.models.auto.modeling_auto import MODEL_FOR_MASKED_LM_MAPPING
        from transformers import EsmConfig, EsmForMaskedLM
        
        if "esmc" not in CONFIG_MAPPING:
            print("Injecting 'esmc' architecture mapping keys into Hugging Face core registry...")
            CONFIG_MAPPING.register("esmc", EsmConfig)
        if EsmConfig not in MODEL_FOR_MASKED_LM_MAPPING:
            MODEL_FOR_MASKED_LM_MAPPING[EsmConfig] = EsmForMaskedLM

    if not use_native_esmc:
        print(f"Loading architecture via Hugging Face AutoModelForMaskedLM from: {args.model_name_or_path}")
        from transformers import AutoModelForMaskedLM
        
        # Safe Tokenizer resolution block to sidestep missing ESMCTokenizer definitions
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
            tokenizer.padding_idx = tokenizer.pad_token_id
            alphabet = tokenizer
            batch_converter = lambda x: (None, None, tokenizer([s for _, s in x], padding=True, return_tensors="pt")["input_ids"])
        except Exception as e:
            # FIX 1: Access type(e).__name__ to avoid dumping the massive list of HF models to stdout
            print(f"AutoTokenizer unresolvable ({type(e).__name__}). Defaulting to direct EsmSequenceTokenizer instantiation...")
            from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
            base_tokenizer = EsmSequenceTokenizer()
            base_tokenizer.padding_idx = base_tokenizer._get_token_id("<pad>")
            
            alphabet = ESMCAlphabetWrapper(base_tokenizer)
            alphabet.padding_idx = base_tokenizer.padding_idx
            
            # FIX 2: Implement manual batch converter to resolve the NoneType callable crash
            def manual_esm_batch_converter(batch):
                # Extract sequence labels and raw strings from the format passed by embed_sequence
                strs = [item[1] for item in batch]
                labels = [item[0] for item in batch]
                
                # Vectorise sequences using the native tokenization rules
                tokenized = [base_tokenizer.encode(s) for s in strs]
                max_len = max(len(t) for t in tokenized)
                pad_token = base_tokenizer._get_token_id("<pad>")
                
                # Construct padded matrices matching typical batch structures
                padded_tokens = [t + [pad_token] * (max_len - len(t)) for t in tokenized]
                return labels, strs, torch.tensor(padded_tokens, dtype=torch.long)
                
            batch_converter = manual_esm_batch_converter
            
        model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path, output_hidden_states=True, trust_remote_code=True).to(device)

    else:
        # Load vanilla ESM-C 300M or 600M checkpoints directly via the local native esm registry
        from esm.models.esmc import ESMC
        if "600m" in model_lower:
            target_variant = "esmc_600m"
        else:
            target_variant = "esmc_300m"
            
        print(f"Loading native ESM-C checkpoint via esm library registry: {target_variant}")
        model = ESMC.from_pretrained(target_variant).to(device)
        
        alphabet = ESMCAlphabetWrapper(model.tokenizer)
        batch_converter = alphabet.get_batch_converter()

    model.eval()
    # -------------------------------------------------------------------------
    # 3. Dynamic Sequence Length Extraction & Truncation Guard
    # -------------------------------------------------------------------------
    if hasattr(model, "config") and hasattr(model.config, "max_position_embeddings"):
        plm_max_aa_length = model.config.max_position_embeddings
    elif is_esmc:
        plm_max_aa_length = 2048
    else:
        plm_max_aa_length = 1022
        
    print(f"Resolved model architecture parameters. Processing sequence truncation threshold at {plm_max_aa_length} AA.")
    
    sequences = [seq[:plm_max_aa_length] for seq in df["seq"].tolist()]
    y_targets = df[["X", "Y", "Z"]].values
    
    print(f"Loaded {len(sequences)} records with valid sequence and 3D coordinate combinations.")

    # -------------------------------------------------------------------------
    # 4. Sequence Processing and Representation Harvesting Loop
    # -------------------------------------------------------------------------
    print(f"Extracting layer hidden states for {len(sequences)} records via embed_sequence...")
    
    # Cold run to dynamically discover total model layer depth
    test_res, _, _, _ = embed_sequence(sequences[0], model, device, model_layers=0, batch_converter=batch_converter, alphabet=alphabet)
    
    if use_native_esmc:
        num_extracted_layers = len(test_res.hidden_states)
    else:
        # Safe attribute verification to avoid eager tuple-indexing evaluation bugs
        if hasattr(test_res, "hidden_states") and test_res.hidden_states is not None:
            num_extracted_layers = len(test_res.hidden_states)
        elif isinstance(test_res, (tuple, list)):
            # Route based on presence/absence of an evaluation loss element in the tuple
            num_extracted_layers = len(test_res[1]) if len(test_res) == 2 else len(test_res[2])
        else:
            raise ValueError("Could not dynamically resolve hidden_states from model output structure.")

    layer_embeddings = {i: [] for i in range(num_extracted_layers)}

    for seq in sequences:
        embeddings_all_layers, _, _, _ = embed_sequence(
            sequence=seq,
            model=model,
            device=device,
            model_layers=0, 
            batch_converter=batch_converter,
            alphabet=alphabet
        )
        
        # Isolate layer state references safely based on framework layout
        if use_native_esmc:
            states = embeddings_all_layers.hidden_states
        else:
            if hasattr(embeddings_all_layers, "hidden_states") and embeddings_all_layers.hidden_states is not None:
                states = embeddings_all_layers.hidden_states
            elif isinstance(embeddings_all_layers, (tuple, list)):
                states = embeddings_all_layers[1] if len(embeddings_all_layers) == 2 else embeddings_all_layers[2]
            else:
                raise ValueError("Could not parse hidden states sequence array during loops.")
        
        for layer_idx in range(num_extracted_layers):
            layer_tensor = states[layer_idx]
            
            if isinstance(layer_tensor, torch.Tensor):
                if layer_tensor.ndim == 3:  
                    layer_tensor = layer_tensor.squeeze(0)
                if layer_tensor.ndim == 2:  
                    layer_tensor = layer_tensor.mean(dim=0)
                
                layer_numpy = layer_tensor.detach().cpu().float().numpy()
            else:
                layer_numpy = np.array(layer_tensor)
                if layer_numpy.ndim == 2:
                    layer_numpy = layer_numpy.mean(axis=0)

            layer_embeddings[layer_idx].append(layer_numpy)

    for layer_idx in layer_embeddings:
        layer_matrix = np.array(layer_embeddings[layer_idx])
        if not np.isfinite(layer_matrix).all():
            layer_matrix = np.nan_to_num(layer_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        layer_embeddings[layer_idx] = layer_matrix

    print(f"Extracted feature matrices across {num_extracted_layers} distinct layers.")
    # -------------------------------------------------------------------------
    # 5. Layer-wise Linear Coordinate Regression Probing
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
    # 6. Export Statistical Tables and Diagnostic Plots
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