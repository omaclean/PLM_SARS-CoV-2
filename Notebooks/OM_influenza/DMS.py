# %% [markdown]
# 

# %%

import sys
sys.path.append('../../')
import os
import importlib
module_name = "Functions"
if module_name in sys.modules:
    del sys.modules[module_name]
Functions = importlib.import_module(module_name)

from Functions_HuggingFace import *

import esm
from Bio import SeqIO
from Bio.Seq import Seq
import pandas as pd 
import numpy as np
import torch



import matplotlib.pyplot as plt
import pandas as pd
from Bio import SeqIO

from Bio import Entrez
from Bio import SeqIO

from transformers import EsmForMaskedLM, T5EncoderModel,T5Tokenizer,DataCollatorForLanguageModeling
from transformers.modeling_outputs import MaskedLMOutput


# %%
model_layers = 36

model_raw, alphabet = esm.pretrained.load_model_and_alphabet('esm2_t36_3B_UR50D')
model_raw.eval()
batch_converter = alphabet.get_batch_converter()

sub_mod='ESM2-H3'
#sub_mod='ESM2-HA80'
modnam="/home3/oml4h/hugging_face_downloads/model_weights_topublish/{}".format(sub_mod)
out='/home3/oml4h/PLM_SARS-CoV-2/Results/DMS_investigation/'

model = EsmForMaskedLM.from_pretrained(modnam)



if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Transferred model to GPU")

else:
    print("CUDA is not available. Using CPU instead.")
    device = torch.device("cpu")
model =  model.to(device)


# %%
OG_reference_path = "/home3/oml4h/PLM_SARS-CoV-2/Sequences/H3N2_canonical.fa"
ref_dict=read_sequences_to_dict(OG_reference_path)
ref_seq_str = list(ref_dict.values())[0]

focal_sequence="/home3/oml4h/PLM_SARS-CoV-2/Sequences/Mass_DMS_2022_prot.fa"
DMS_data=pd.read_csv("/home3/oml4h/PLM_SARS-CoV-2/Sequences/DMS_bloom_Mass_2022_Phenotypes.csv")
sequences = read_sequences_to_dict(focal_sequence)
reference_sequence=list(sequences.values())[0]

h3_map_with_ha2 = create_h3_numbering_map(reference_sequence, ref_seq_str, HA2_start=330)


# %%

result=get_mutation_prob_matrix(list(sequences.values())[0], model, model_layers, device, batch_converter, alphabet)

mutation_prob_matrix = result['mutation_matrix']
amino_acids = result['amino_acids']
positions = result['positions']

# %%
# Add mutation probabilities, canonical coordinates, and reference amino acid to DMS_data
mutation_probs = []
canonical_coords = []
reference_aas = []

# Get the DMS sequence
dms_sequence = list(sequences.values())[0]

for idx, row in DMS_data.iterrows():
    site = row['site']
    wildtype = row['wildtype']
    mutant = row['mutant']
    
    # Get the reference amino acid at this site (1-indexed)
    if site <= len(dms_sequence):
        ref_aa = dms_sequence[site - 1]  # Convert to 0-indexed
        reference_aas.append(ref_aa)
    else:
        reference_aas.append(np.nan)
    
    # Get canonical coordinate using the H3 numbering map
    # Create a mutation string to convert
    mutation_str = f"{wildtype}{site}{mutant}"
    try:
        canonical_mut = mutations_to_canonical([mutation_str], h3_map_with_ha2)
        canonical_coords.append(canonical_mut[0])
    except:
        # If conversion fails, just use original
        canonical_coords.append(mutation_str)
    
    # Check if site is within the positions range
    if site in positions:
        # Get the position index
        pos_idx = positions.index(site)
        
        # Check if mutant amino acid is in our amino_acids list
        if mutant in amino_acids:
            # Get the amino acid index
            aa_idx = amino_acids.index(mutant)
            
            # Get the probability from the matrix
            prob = mutation_prob_matrix[aa_idx, pos_idx]
            mutation_probs.append(prob)
        else:
            # Mutant not in standard amino acids
            mutation_probs.append(np.nan)
    else:
        # Site not in positions (shouldn't happen if sequences match)
        mutation_probs.append(np.nan)

# Add the new columns
DMS_data['mutation_probability'] = mutation_probs
DMS_data['canonical_mutation'] = canonical_coords
DMS_data["mut_in_fasta"] = DMS_data.apply(lambda row: f"{row['wildtype']}{row['site']}{row['mutant']}", axis=1)
DMS_data['dms_sequence_aa'] = reference_aas


# Display summary
print(f"Added mutation probabilities to {len(DMS_data)} rows")
print(f"Non-null probabilities: {DMS_data['mutation_probability'].notna().sum()}")
print(f"\nFirst few rows:")


# %%
DMS_data.head(10)

# %%
# Save the updated dataframe
DMS_data.to_csv(os.path.join(out, f'{sub_mod}_DMS_with_probabilities.csv'), index=False)


# %%
# also get gramaticallity
backbone=reference_sequence
backbone_name="Mass_DMS_2022_prot"
mut_info_rows=[]

for row, mut in enumerate(DMS_data['mut_in_fasta']):
    print(mut)
    print(reference_sequence,[mut])
    mutated_seq_i=mutate_sequence(reference_sequence,[mut])
    
    mutations = embed_protein_sequences(
    [[mut,mutated_seq_i.replace("-","")]],
    backbone.replace("-",""),
    'S:0',
    model,
    model_layers,
    device,
    batch_converter,
    alphabet,
    scores=True)
    
    
    #append mutation info to dataframe
    new_row={"Mutation":mut,"rel_grammar":mutations[mut]["S:0"]["relative_grammaticality"],
                                "rel_seq_grammar":mutations[mut]["S:0"]["relative_sequence_grammaticality"],
                                "narrow_seq_grammar":mutations[mut]["S:0"]["narrow_sequence_grammaticality"],
                                "relative_narrow_seq_grammar":mutations[mut]["S:0"]["relative_narrow_sequence_grammaticality"],
                                "semantic_score":mutations[mut]["S:0"]["semantic_score"],
                                "probability":mutations[mut]["S:0"]["probability"],
                                "Backbone": backbone_name,
                                }
    
    mut_info_rows.append(new_row)

mut_info_combos=pd.DataFrame(mut_info_rows)
DMS_data["relative_grammaticality"]=mut_info_combos["rel_grammar"]
DMS_data["mut_prob2"]=mut_info_combos["probability"]
DMS_data["semantic_score"]=mut_info_combos["semantic_score"]
DMS_data.to_csv(os.path.join(out, f'{sub_mod}_DMS_with_probabilities_grammar.csv'), index=False)



