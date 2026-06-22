#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Layer-wise Representation Probe: Pairwise Distance Correlation between PLM/ESM-C Layers and PLANT 3D Coordinates
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
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr

def parse_args():
    parser = argparse.ArgumentParser(description="Probe internal PLM/ESM-C layer representations via pairwise distance correlation.")
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
    parser.add_argument("--num_folds", type=int, default=5, help="Retained for CLI compatibility; unused in distance calculation.")
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
    print(f"Initialising layer distance-correlation pipeline for model: {args.model_name_or_path}")
    
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
    is_finetune = "checkpoint" in model_lower or "finetune" in model_lower or "my_sc2" in model_lower or "magma" in model_lower
    
    use_native_esmc = is_esmc and (not is_6b) and (not is_finetune)

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)
    print(f"Inference execution engine set to: {device}")

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
        import json
        from transformers import EsmConfig, EsmForMaskedLM
        
        num_layers, hidden_size, num_heads, max_pos, vocab_size = 33, 1280, 20, 1026, 64000
        if is_esmc:
            num_layers, hidden_size, num_heads, max_pos, vocab_size = 30, 960, 15, 2048, 33
            if is_6b:
                num_layers, hidden_size, num_heads = 80, 4096, 32
            elif "600m" in model_lower:
                num_layers, hidden_size, num_heads = 36, 1152, 18

        config_file = os.path.join(args.model_name_or_path, "config.json")
        if os.path.exists(config_file):
            try:
                with open(config_file, "r") as f:
                    cfg = json.load(f)
                
                def safe_int(val):
                    if isinstance(val, (list, tuple)):
                        val = val[0] if len(val) > 0 else 0
                    return int(float(val))

                if "num_layers" in cfg or "num_hidden_layers" in cfg or "layers" in cfg:
                    num_layers = safe_int(cfg.get("num_layers", cfg.get("num_hidden_layers", cfg.get("layers"))))
                if "d_model" in cfg or "hidden_size" in cfg:
                    hidden_size = safe_int(cfg.get("d_model", cfg.get("hidden_size")))
                if "num_heads" in cfg or "num_attention_heads" in cfg or "n_heads" in cfg:
                    num_heads = safe_int(cfg.get("num_heads", cfg.get("num_attention_heads", cfg.get("n_heads"))))
                if "max_position_embeddings" in cfg or "max_seq_len" in cfg:
                    max_pos = safe_int(cfg.get("max_position_embeddings", cfg.get("max_seq_len")))
                if "vocab_size" in cfg:
                    vocab_size = safe_int(cfg.get("vocab_size"))
            except Exception as e:
                print(f"Warning: Could not read local config.json ({e}). Defaulting to safety parameters.")

        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
            tokenizer.padding_idx = tokenizer.pad_token_id
            alphabet = tokenizer
            batch_converter = lambda x: (None, None, tokenizer([s for _, s in x], padding=True, return_tensors="pt")["input_ids"])
            pad_token_id = tokenizer.pad_token_id
        except Exception as e:
            print(f"AutoTokenizer unresolvable ({type(e).__name__}). Enforcing direct EsmSequenceTokenizer instantiation...")
            from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
            base_tokenizer = EsmSequenceTokenizer()
            base_tokenizer.padding_idx = base_tokenizer._get_token_id("<pad>")
            alphabet = ESMCAlphabetWrapper(base_tokenizer)
            alphabet.padding_idx = base_tokenizer.padding_idx
            
            def manual_esm_batch_converter(batch):
                strs = [item[1] for item in batch]
                labels = [item[0] for item in batch]
                tokenized = [base_tokenizer.encode(s) for s in strs]
                max_len = max(len(t) for t in tokenized)
                pad_token = base_tokenizer._get_token_id("<pad>")
                padded_tokens = [t + [pad_token] * (max_len - len(t)) for t in tokenized]
                return labels, strs, torch.tensor(padded_tokens, dtype=torch.long)
                
            batch_converter = manual_esm_batch_converter
            pad_token_id = base_tokenizer.padding_idx

        hf_config = EsmConfig(
            vocab_size=vocab_size,
            num_hidden_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            max_position_embeddings=max_pos,
            pad_token_id=pad_token_id,
            output_hidden_states=True
        )
        hf_config.intermediate_size = 4 * hidden_size
        
        print(f"Loading model via explicit HF configuration scheme -> Layers: {num_layers}, Dimension: {hidden_size}, Vocab: {vocab_size}")

        config_bak = config_file + ".bak"
        has_config_to_hide = os.path.exists(config_file)
        
        if has_config_to_hide:
            os.rename(config_file, config_bak)
            
        try:
            model = EsmForMaskedLM.from_pretrained(
                args.model_name_or_path, 
                config=hf_config, 
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if device_type == "cuda" else torch.float32
            ).to(device)
        finally:
            if has_config_to_hide:
                os.rename(config_bak, config_file)

    else:
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
    
    test_res, _, _, _ = embed_sequence(sequences[0], model, device, model_layers=0, batch_converter=batch_converter, alphabet=alphabet)
    
    if use_native_esmc:
        num_extracted_layers = len(test_res.hidden_states)
    else:
        if hasattr(test_res, "hidden_states") and test_res.hidden_states is not None:
            num_extracted_layers = len(test_res.hidden_states)
        elif isinstance(test_res, (tuple, list)):
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
    # 5. Layer-wise Pairwise Distance Correlation Analysis
    # -------------------------------------------------------------------------
    print("Computing pairwise Euclidean distances for target antigenic 3D coordinate space...")
    # Computes compressed upper-triangular distance vector to avoid redundant pairings and self-distance zeros
    flat_coords_dist = pdist(y_targets, metric="euclidean")
    
    layer_metrics = []
    
    print("Analysing pairwise representation distances across hidden layers...")
    for layer_idx in range(num_extracted_layers):
        X_features = layer_embeddings[layer_idx]
        
        # Standardise representation vectors to ensure distance calculations are numerically stable
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_features)
        
        # Compute pairwise Euclidean distances in embedding space
        flat_emb_dist = pdist(X_scaled, metric="euclidean")
        
        # Calculate linear (Pearson) and monotonic rank-based (Spearman) relationship matrices
        p_corr, _ = pearsonr(flat_emb_dist, flat_coords_dist)
        s_corr, _ = spearmanr(flat_emb_dist, flat_coords_dist)
        
        layer_metrics.append({
            "Layer": layer_idx,
            "Pearson_R": p_corr,
            "Spearman_R": s_corr
        })
        
        print(f"Layer {layer_idx:02d} | Pearson r: {p_corr:.4f} | Spearman rho: {s_corr:.4f}")
        
    # -------------------------------------------------------------------------
    # 6. Export Statistical Tables and Diagnostic Plots
    # -------------------------------------------------------------------------
    summary_df = pd.DataFrame(layer_metrics)
    summary_export_path = os.path.join(args.output_dir, f"layer_distance_correlation_{model_safe_name}.csv")
    summary_df.to_csv(summary_export_path, index=False)
    print(f"Saved metric output tables to: {summary_export_path}")
    
    # Identify optimal tracking layer based on maximal monotonic trend preservation
    optimal_row = summary_df.loc[summary_df["Spearman_R"].idxmax()]
    print("-" * 80)
    print(f"OPTIMAL GEOMETRIC DISTANCE CONGRUENCE LAYER IDENTIFIED FOR {model_safe_name}:")
    print(f"Layer Index: {int(optimal_row['Layer'])} (Spearman rho: {optimal_row['Spearman_R']:.4f} | Pearson r: {optimal_row['Pearson_R']:.4f})")
    print("-" * 80)
    
    plt.figure(figsize=(11, 6))
    plt.plot(summary_df["Layer"], summary_df["Pearson_R"], marker="o", linewidth=2, color="#1f77b4", label="Pearson r")
    plt.plot(summary_df["Layer"], summary_df["Spearman_R"], marker="s", linewidth=2, color="#2ca02c", label="Spearman rho")
    
    plt.axvline(x=optimal_row['Layer'], color="red", linestyle="--", alpha=0.7, label=f"Optimal Layer ({int(optimal_row['Layer'])})")
    
    plt.xlabel("Layer Index (Layer 0 = Input Token Embeddings)")
    plt.ylabel("Pairwise Distance Correlation Coefficient")
    plt.title(f"Geometric Congruence of PLM Hidden Spaces with PLANT 3D Coordinate Manifold\nModel: {model_safe_name}")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    plot_export_path = os.path.join(args.output_dir, f"layer_distance_trajectory_{model_safe_name}.png")
    plt.savefig(plot_export_path, dpi=300)
    plt.close()
    print(f"Diagnostic plot successfully exported to: {plot_export_path}")

if __name__ == "__main__":
    main()