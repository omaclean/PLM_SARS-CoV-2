# %%
%load_ext autoreload
%autoreload 2
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


# %% [markdown]
# # Run code

# %%
model_layers = 36

model_raw, alphabet = esm.pretrained.load_model_and_alphabet('esm2_t36_3B_UR50D')
model_raw.eval()
batch_converter = alphabet.get_batch_converter()
out='/home3/oml4h/PLM_SARS-CoV-2/Results/test/'
sub_mod='ESM2-H3'
sub_mod='ESM2-HA80'
# sub_mod="ESM_C_600M" <- doesn't work with old esm libs

modnam="/home3/oml4h/hugging_face_downloads/model_weights_topublish/{}".format(sub_mod)

# if(sub_mod=='ESM_C_600M'):
#     modnam="/home3/oml4h/hugging_face_downloads/ESM_C_600M"
model = EsmForMaskedLM.from_pretrained(modnam)



if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Transferred model to GPU")

else:
    print("CUDA is not available. Using CPU instead.")
    device = torch.device("cpu")
    # set max threads for cpu
    torch.set_num_threads(16)
model =  model.to(device)


# %%
query_path = "/home3/oml4h/PLM_SARS-CoV-2/Sequences/huH3N2_HA_CDS.translated_extra_steps.fas"

reference_path = "/home3/oml4h/PLM_SARS-CoV-2/Sequences/H3N2_canonical.fa"

sequences = read_sequences_to_dict(query_path)

sequences



# 1. Read the reference sequence (Assuming single sequence in file)
# We use 'next' to get the first item from the iterator
ref_record = next(SeqIO.parse(reference_path, "fasta"))
ref_seq_str = str(ref_record.seq)



# 2. Read the query sequences
# We parse the file and pick the first one as a test case
query_iterator = SeqIO.parse(query_path, "fasta")
first_query_record = next(query_iterator)

h3_map_with_ha2 = create_h3_numbering_map(first_query_record, ref_seq_str, HA2_start=330)




# %%
ids=list(sequences.keys())




# %%
J_indexed_muts = [m for m in get_reference_mutations(ref = sequences[ids[0]],mut = sequences[ids[len(ids)-1]]) if "-" not in m  ] 
K_indexed_muts = [m for m in get_mutations(sequences[ids[0]],sequences[ids[len(ids)-1]]) if "del" not in m and '-' not in m  ] 
# Convert your mutations to canonical numbering
canonical_mutations = mutations_to_canonical(K_indexed_muts, h3_map_with_ha2)
print("K->J mutations")
print(K_indexed_muts)

# %%
mutation_dictionary  = {k:J_indexed_muts[i] for i,k in enumerate(K_indexed_muts)}

# %%
# get info on model layers:
model_layers_info = model.config.to_dict()
print(model_layers_info)
#print how many layers the model has
print("Number of layers in the model: ", model_layers_info['num_hidden_layers'] + 1)  # +1 for the embedding layer

# %%
#pd.DataFrame(columns=["Mutation","rel_grammar","rel_seq_grammar","semanatic_score","probability","Backbone"])
mut_info_rows=[]

#loop through each node in the tree except the last one (lineage of interest)
for backbone_i in range(len(ids)-1):
    backbone = ids[backbone_i]
    print("Calculating mutations on backbone:",backbone)
    reference_spike_sequence = sequences[backbone]
    #calculate mutation differences between node and K lineage
    J_indexed_muts = [m for m in get_reference_mutations(ref = sequences[backbone],mut = sequences[ids[len(ids)-1]]) if "-" not in m  ] 
    K_indexed_muts = [m for m in get_mutations(sequences[backbone],sequences[ids[len(ids)-1]]) if "del" not in m and '-' not in m  ] 
    # Convert your mutations to canonical numbering
    canonical_mutations = mutations_to_canonical(K_indexed_muts, h3_map_with_ha2)
    mutation_dictionary  = {k:J_indexed_muts[i] for i,k in enumerate(K_indexed_muts)}
    canon_dict={k:canonical_mutations[i] for i,k in enumerate(K_indexed_muts)}
    print("K->J mutations")
    print(K_indexed_muts)
    #loop through each mutation and calculate the scores on the focal node
    for mut in mutation_dictionary.keys():
        if backbone ==  ids[len(ids)-1]:
            sequence = revert_sequence(reference_spike_sequence,[mut])
        else:
            sequence = mutate_sequence(reference_spike_sequence,[mut])
        reference = reference_spike_sequence
        mutations = embed_protein_sequences(
            [[mut,sequence.replace("-","")]],
            reference.replace("-",""),
            'S:0',
            model,
            model_layers,
            device,
            batch_converter,
            alphabet,
            scores=True)
   
        
        #append mutation info to dataframe
        new_row={"Mutation":mut,
                 'canon':canon_dict[mut],
                 'sequence_grammar':mutations[mut]["S:0"]["sequence_grammaticality"],
                    "rel_grammar":mutations[mut]["S:0"]["relative_grammaticality"],
                    "rel_seq_grammar":mutations[mut]["S:0"]["relative_sequence_grammaticality"],
                    "narrow_seq_grammar":mutations[mut]["S:0"]["narrow_sequence_grammaticality"],
                    "relative_narrow_seq_grammar":mutations[mut]["S:0"]["relative_narrow_sequence_grammaticality"],
                    "semanatic_score":mutations[mut]["S:0"]["semantic_score"],
                    "probability":mutations[mut]["S:0"]["probability"],
                    "Backbone":backbone}
        mut_info_rows.append(new_row)

# Add final row: full sequence grammar for the final lineage
final_lineage_name = ids[-1].split("|")[-1]
first_lineage_name = ids[0].split("|")[-1]
print(f"\nCalculating full sequence grammar for {final_lineage_name} (relative to {first_lineage_name})")

final_sequence = sequences[ids[-1]]
first_sequence = sequences[ids[0]]

# Use process_protein_sequence directly instead of embed_protein_sequences
final_result = process_protein_sequence(
    final_sequence.replace("-", ""),
    model,
    model_layers,
    batch_converter,
    alphabet,
    device
)

# Get reference result for comparison
reference_result = process_protein_sequence(
    first_sequence.replace("-", ""),
    model,
    model_layers,
    batch_converter,
    alphabet,
    device
)

# Calculate relative scores
rel_grammar = final_result["sequence_grammaticality"] - reference_result["sequence_grammaticality"]
rel_seq_grammar = final_result["sequence_grammaticality"] - reference_result["sequence_grammaticality"]
semantic_score = semantic_calc(final_result["Mean_Embedding"], reference_result["Mean_Embedding"])

# Add row for full sequence (narrow grammar won't be meaningful here)
new_row = {
    "Mutation": f"{first_lineage_name}_to_{final_lineage_name}",
    'canon': f"{first_lineage_name}_to_{final_lineage_name}",
    'sequence_grammar':final_result["sequence_grammaticality"],
    "rel_grammar": rel_grammar,
    "rel_seq_grammar": rel_seq_grammar,
    "narrow_seq_grammar": np.nan,  # Not meaningful for full sequence
    "relative_narrow_seq_grammar": np.nan,  # Not meaningful for full sequence
    "semanatic_score": semantic_score,
    "probability": np.nan,
    "Backbone": ids[0]  # Reference is the first sequence
}
# and add one for the OG
# Add row for full sequence (narrow grammar won't be meaningful here)
new_row = {
    "Mutation": f"{first_lineage_name}_",
    'canon': f"{first_lineage_name}",
    'sequence_grammar':reference_result["sequence_grammaticality"],
    "rel_grammar":  np.nan,
    "rel_seq_grammar":  np.nan,
    "narrow_seq_grammar": np.nan,  # Not meaningful for full sequence
    "relative_narrow_seq_grammar": np.nan,  # Not meaningful for full sequence
    "semanatic_score": semantic_score,
    "probability":  np.nan,
    "Backbone": ids[0]  # Reference is the first sequence
}
mut_info_rows.append(new_row)

mut_info=pd.DataFrame(mut_info_rows)

# %% [markdown]
# # model each backbone separately just for gramaticality of whole sequence

# %%
# Calculate full sequence grammar for all backbones
backbone_grammar_rows = []

# Get the first sequence as reference
first_sequence = sequences[ids[0]]
first_lineage_name = ids[0].split("|")[-1]

# Process reference sequence once
reference_result = process_protein_sequence(
    first_sequence.replace("-", ""),
    model,
    model_layers,
    batch_converter,
    alphabet,
    device
)

print(f"Processing all backbone sequences relative to {first_lineage_name}...\n")

# Loop through all sequences
for backbone_i, backbone_id in enumerate(ids):
    lineage_name = backbone_id.split("|")[-1]
    print(f"Processing {lineage_name} ({backbone_i+1}/{len(ids)})")
    
    # Get current sequence
    current_sequence = sequences[backbone_id]
    
    # Process current sequence
    current_result = process_protein_sequence(
        current_sequence.replace("-", ""),
        model,
        model_layers,
        batch_converter,
        alphabet,
        device
    )
    
    # Calculate semantic change from first sequence
    if backbone_i == 0:
        semantic_change = 0.0  # First sequence has zero change
    else:
        semantic_change = semantic_calc(
            current_result["Mean_Embedding"], 
            reference_result["Mean_Embedding"]
        )
    
    # Add row to results
    new_row = {
        "Lineage": lineage_name,
        "Full_ID": backbone_id,
        "sequence_grammar": current_result["sequence_grammaticality"],
        "semantic_change_from_first": semantic_change,
        "order": backbone_i
    }
    backbone_grammar_rows.append(new_row)

# Create dataframe and save
backbone_grammar_df = pd.DataFrame(backbone_grammar_rows)
output_file = out + f"{sub_mod}_backbone_sequence_grammar.csv"
backbone_grammar_df.to_csv(output_file, index=False)

print(f"\nSaved backbone sequence grammar to: {output_file}")
print("\nSummary:")
print(backbone_grammar_df)

# %%
#
mut_info["lineage_backbone"]=mut_info["Backbone"].str.split("|").str[-1]
mut_info
mut_info.to_csv(out+"H3_epistasis_mutation_info_spyros_model_{}.csv".format(sub_mod)
,index=False)
print(mut_info.loc[mut_info['Mutation']=='I176K'])


# %% [markdown]
# # do each mutation by itself and then do each pairwise comparison 

# %%
backbone_id=ids[4]
K_indexed_muts = [m for m in get_mutations(sequences[backbone_id],sequences[ids[len(ids)-1]]) if "del" not in m and '-' not in m  ] 
print("K->J mutations")
print(K_indexed_muts)



# %%


# %%
# backbone = "Wuhan-Hu-1"

backbone=backbone_id
reference_spike_sequence = sequences[backbone]

#create dict of point mutation backbones for all of the mutations in  K_indexed_muts, with names as keys
point_mutations={}
for mut in K_indexed_muts:
    point_mutations[mut]=mutate_sequence(reference_spike_sequence,[mut])

mut_info_rows=[]
plm_entropy=[]
plm_probability=[]

#loop through each point mutation
for backbone_i in range(len(K_indexed_muts)+1):
    
    if backbone_i == len(K_indexed_muts):
        backbone = reference_spike_sequence
        backbone_name="Reference"
    else:
        backbone = point_mutations[list(point_mutations.keys())[backbone_i]]
        backbone_name=list(point_mutations.keys())[backbone_i]
    
    for mutation_j in range(len(K_indexed_muts)):
        if backbone_i == mutation_j:
            print(f"Skipping mutation {K_indexed_muts[mutation_j]} on its own backbone")
            continue
        mut=K_indexed_muts[mutation_j]
        mutate_sequence(reference_spike_sequence,[mut])
        
   

        sequence = mutate_sequence(backbone,[mut])

        reference = backbone
        mutations = embed_protein_sequences(
            [[mut,sequence.replace("-","")]],
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
                                    "semanatic_score":mutations[mut]["S:0"]["semantic_score"],
                                    "probability":mutations[mut]["S:0"]["probability"],
                                    "Backbone": backbone_name,
                                    }
        mut_info_rows.append(new_row)

        plm_entropy.append({"Mutation":mut,"Backbone":backbone_name,"entropy":mutations[mut]["S:0"]["entropy"]})
        plm_probability.append({"Mutation":mut,"Backbone":backbone_name,"probability":mutations[mut]["S:0"]["sequence_probabilities"]})   
        
plm_entropy.append({"Mutation":"reference","Backbone":"reference","entropy":mutations["Reference"]["S:0"]["sequence_entropy"]})
plm_probability.append({"Mutation":"reference","Backbone":"reference","probability":mutations["Reference"]["S:0"]["sequence_probabilities"]})

mut_info_combos=pd.DataFrame(mut_info_rows)



# %%


mut_info_combos.to_csv(out+sub_mod+'_mut_info_combos.csv',index=False)   
def make_wide_dataframe(data_list, value_name="site"):
    """
    Converts list of dicts with a list-column into a wide dataframe.
    """
    # Create initial DF
    df = pd.DataFrame(data_list)
    
    # Expand the 'data' column into separate columns
    # resulting shape: (N_samples, Sequence_Length)
    expanded = pd.DataFrame(df[value_name].to_list())
    
    # Rename columns to 1-based site indices (e.g., site_1, site_2...)
    expanded.columns = [f"{value_name}_{i+1}" for i in range(expanded.shape[1])]
    
    # Concatenate ID columns with the expanded data
    # Axis 1 joins columns
    final_df = pd.concat([df[['Mutation', 'Backbone']], expanded], axis=1)
    
    return final_df

# Generate the Wide DataFrames
mut_info_combos = pd.DataFrame(mut_info_rows)
entropy_wide = make_wide_dataframe(plm_entropy, value_name="entropy")
probability_wide = make_wide_dataframe(plm_probability, value_name="probability")

entropy_wide.to_csv(out+sub_mod+'_entropy.csv',index=False)
probability_wide.to_csv(out+sub_mod+'_probability.csv',index=False)



# %% 
# see probability shifts for each mutation on each backbone
backbone=backbone_id
reference_spike_sequence = sequences[backbone]
result = get_mutation_prob_matrix(reference_spike_sequence, model, model_layers, device, batch_converter, alphabet)
mutation_prob_matrix = result['mutation_matrix']
amino_acids = result['amino_acids']
positions = result['positions']

# Get canonical names for mutations
canonical_mutations = mutations_to_canonical(K_indexed_muts, h3_map_with_ha2)
canon_dict = {k: canonical_mutations[i] for i, k in enumerate(K_indexed_muts)}

mut_shifts = []
mut_names = []
all_shift_matrices = []

# Loop through each point mutation
for i, mut in enumerate(K_indexed_muts):
    # Use reference_spike_sequence, not backbone ID
    sequence_i = mutate_sequence(reference_spike_sequence, [mut])
    result = get_mutation_prob_matrix(sequence_i, model, model_layers, device, batch_converter, alphabet)
    
    # calculate shifts for 19xL matrix
    mutation_prob_matrix_i = result['mutation_matrix']
    shift_matrix = mutation_prob_matrix_i - mutation_prob_matrix
    # sum absolute shifts for each position along L amino acids
    total_shifts = np.nansum(np.abs(shift_matrix), axis=0)
    
    mut_shifts.append(total_shifts)
    mut_names.append(canon_dict.get(mut, mut))
    all_shift_matrices.append(shift_matrix)

# Combined mutation
combined_seq = mutate_sequence(reference_spike_sequence, K_indexed_muts)
result = get_mutation_prob_matrix(combined_seq, model, model_layers, device, batch_converter, alphabet)
mutation_prob_matrix_combined = result['mutation_matrix']
shift_matrix_combined = mutation_prob_matrix_combined - mutation_prob_matrix
# sum absolute shifts for each position along L amino acids
total_shifts_combined = np.nansum(np.abs(shift_matrix_combined), axis=0)

mut_shifts.append(total_shifts_combined)
mut_names.append("Combined")
all_shift_matrices.append(shift_matrix_combined)


backbone=backbone_id
backbone_lineage=backbone_id.split("|")[-1]
# Save the lists properly (as CSV)
df_shifts = pd.DataFrame(np.array(mut_shifts).T, columns=mut_names)
df_shifts['Position'] = positions
# Reorder to put Position first
cols = ['Position'] + [c for c in df_shifts.columns if c != 'Position']
df_shifts = df_shifts[cols]
df_shifts.to_csv(os.path.join(out, f"{sub_mod}_mutation_probability_shifts_{backbone_lineage}.csv"), index=False)

# %% 
# plot above stuff

# Create a panel plot of shifts
num_plots = len(mut_shifts)
cols = int(np.ceil(np.sqrt(num_plots)))
rows = int(np.ceil(num_plots / cols))

# Prepare data for plotting (masking focal sites)
masked_shifts_list = []
focal_positions_list = []

# Helper to find index of position in positions list
def get_pos_index(pos_num, positions_list):
    # Try to match integer or string representation
    for idx, p in enumerate(positions_list):
        if str(p) == str(pos_num):
            return idx
    return None

for i in range(num_plots):
    current_shifts = mut_shifts[i].copy()
    current_focal_indices = []
    
    if i < len(K_indexed_muts):
        # Single mutation
        mut = K_indexed_muts[i]
        match = re.search(r'(\d+)', mut)
        if match:
            pos_num = int(match.group(1))
            idx = get_pos_index(pos_num, positions)
            if idx is not None:
                current_shifts[idx] = np.nan # Mask the focal site
                current_focal_indices.append(positions[idx])
    else:
        # Combined - mask all
        for mut in K_indexed_muts:
            match = re.search(r'(\d+)', mut)
            if match:
                pos_num = int(match.group(1))
                idx = get_pos_index(pos_num, positions)
                if idx is not None:
                    current_shifts[idx] = np.nan
                    current_focal_indices.append(positions[idx])
    
    masked_shifts_list.append(current_shifts)
    focal_positions_list.append(current_focal_indices)

# Calculate global max for shared y-axis (ignoring NaNs)
global_max = 0
for shifts in masked_shifts_list:
    m = np.nanmax(shifts)
    if m > global_max:
        global_max = m

print(f"Max epistatic shift observed (excluding focal sites): {global_max}")
print("Note: Total absolute shift is the sum of absolute differences across all 20 amino acids, so the maximum possible value per site is 2.0.")

# Print table of mutations
print("\n=== Mutations Analyzed ===")
mut_table = pd.DataFrame({
    'Mutation': K_indexed_muts,
    'Canonical': [canon_dict.get(m, m) for m in K_indexed_muts]
})
print(mut_table.to_string(index=False))

print("\n=== Detailed Epistatic Shift Analysis ===")
print("For each mutation, showing sites with largest probability shifts.")
print("Showing sites with total shift > 0.3 (or top 2 if none > 0.3).")
print("For each site, showing amino acids with largest probability INCREASE.")

for i, name in enumerate(mut_names):
    print(f"\n--------------------------------------------------")
    print(f"Mutation: {name}")
    
    shifts = masked_shifts_list[i]
    shift_matrix = all_shift_matrices[i]
    
    # Find indices of interest
    # 1. Sites with shift > 0.3
    with np.errstate(invalid='ignore'):
        high_shift_indices = np.where(shifts > 0.3)[0]
    
    target_indices = list(high_shift_indices)
    
    # 2. If fewer than 2 sites, add top sites until we have at least 2
    if len(target_indices) < 2:
        # Get indices sorted by shift value (descending), ignoring NaNs
        # Replace NaNs with -1 for sorting
        shifts_clean = np.nan_to_num(shifts, nan=-1.0)
        sorted_indices = np.argsort(shifts_clean)[::-1]
        
        for idx in sorted_indices:
            if idx not in target_indices and shifts_clean[idx] >= 0: # Ensure we don't pick masked sites
                target_indices.append(idx)
                if len(target_indices) >= 2:
                    break
    
    # Sort indices by position for cleaner output
    target_indices.sort()
    
    for idx in target_indices:
        val = shifts[idx]
        pos = positions[idx]
        
        # Map to canonical numbering
        if 'h3_map_with_ha2' in locals():
            canon_label = h3_map_with_ha2.get(idx, str(pos))
        else:
            canon_label = str(pos)
            
        print(f"\n  Site {canon_label} (Pos {pos}) - Total Shift: {val:.4f}")
        
        # Analyze specific amino acid changes at this site
        # Get the column for this site from the shift matrix (20 rows)
        site_aa_shifts = shift_matrix[:, idx]
        
        # Find amino acids with positive shifts (probability increased)
        # Sort by shift value descending
        aa_indices = np.argsort(site_aa_shifts)[::-1]
        
        # Print top 3 increasing amino acids
        print("    Top probability increases:")
        count = 0
        for aa_idx in aa_indices:
            aa_shift = site_aa_shifts[aa_idx]
            if aa_shift <= 0.01 and count >= 1: # Stop if shift is negligible, but show at least 1
                break
            
            aa = amino_acids[aa_idx]
            print(f"      {aa}: +{aa_shift:.4f}")
            count += 1
            if count >= 3:
                break

fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows), sharey=True, sharex=True)
# Flatten axes for easy iteration if it's a grid
if num_plots > 1:
    axes_flat = axes.flatten()
else:
    axes_flat = [axes]

for i in range(num_plots):
    ax = axes_flat[i]
    
    # Plot the epistatic shifts (focal sites masked)
    ax.plot(positions, masked_shifts_list[i], linewidth=1)
    
    # Plot vertical lines for focal sites
    for focal_pos in focal_positions_list[i]:
        ax.axvline(x=focal_pos, color='orange', alpha=0.5, linewidth=2)
        
    ax.set_title(mut_names[i])
    
    # Only set labels on outer edges since we share axes
    if i >= num_plots - cols:
        ax.set_xlabel('Position')
    if i % cols == 0:
        ax.set_ylabel('Total Absolute Shift')

# Hide empty subplots
for i in range(num_plots, len(axes_flat)):
    axes_flat[i].axis('off')

plt.suptitle(f'Mutation Probability Shifts on Backbone {backbone_id}\n(Focal mutation sites marked with vertical lines)')
plt.tight_layout()
plt.savefig(os.path.join(out, f"{sub_mod}_mutation_probability_shifts_{backbone_lineage}.png"), dpi=300)

# %%
backbone=backbone_id
reference_spike_sequence = sequences[backbone]
result = get_mutation_prob_matrix(reference_spike_sequence, model, model_layers, device, batch_converter, alphabet)
mutation_prob_matrix = result['mutation_matrix']
amino_acids = result['amino_acids']
positions = result['positions']

# Get max non-reference probability for each site
# Loop through each position (column) and mask out the reference amino acid
max_non_reference_probs = []
for site_idx, pos in enumerate(positions):
    ref_aa = reference_spike_sequence[site_idx]  # Get the reference amino acid at this position
    
    # Get all probabilities at this site
    site_probs = mutation_prob_matrix[:, site_idx].copy()
    
    # Only mask out the reference if it's one of the 20 standard amino acids
    # If it's X or another non-standard AA, just take the max of all 20
    if ref_aa in amino_acids:
        ref_aa_idx = amino_acids.index(ref_aa)  # Get its index in the amino_acids list
        # Mask out the reference amino acid
        site_probs[ref_aa_idx] = -np.inf  # Set reference to very low value so it won't be max
    # modify mutation_prob_matrix to make reference sites probabilities zero
    mutation_prob_matrix[ref_aa_idx, site_idx] = np.nan
    # Get the maximum of the remaining (mutant) probabilities
    max_mutant_prob = site_probs.max()
    max_non_reference_probs.append(max_mutant_prob)

max_non_reference_probs = np.array(max_non_reference_probs)

# %%
mutated_sites=set(mut_info_combos["Mutation"])
print(mutated_sites)
#strip out letters
import seaborn
#sns.histplot(entropy_wide.iloc[-1,2:])
sns.histplot(probability_wide.iloc[-1,2:])

numeric_positions = [int(re.search(r'\d+', site).group()) for site in mutated_sites]
print(numeric_positions) # Sort the list for easier reading/comparison


target_series = probability_wide.iloc[-1, 2:]

# Convert the target series to a dictionary for fast lookup: {position: probability}
# Assuming column names (indices) are the residue positions (as in the mock setup)
position_to_prob = target_series.to_dict()

# Calculate Ranks: Assign a rank (1 being highest probability)
# Get the values and sort them in descending order
sorted_probs = target_series.sort_values(ascending=True)
# Create a rank series (using .rank() is safer for ties, but a simple lookup is faster here)
# Rank is 1 + index in the sorted list (since index starts at 0)
position_to_rank = {pos: rank + 1 for rank, pos in enumerate(sorted_probs.index)}
print(position_to_rank)
print(probability_wide.columns)

# Rank the max non-reference probabilities (highest value = rank 1)
# Create position-indexed ranking for max_non_reference_probs
max_nonref_with_positions = [(i+1, val) for i, val in enumerate(max_non_reference_probs)]
# Sort by value descending (highest first)
sorted_max_nonref = sorted(max_nonref_with_positions, key=lambda x: x[1], reverse=True)
# Create rank lookup: position -> rank (1 = highest)
position_to_nonref_rank = {pos: rank + 1 for rank, (pos, val) in enumerate(sorted_max_nonref)}

results = []
for mut_num, pos in enumerate(numeric_positions):
    prob = probability_wide["probability_"+str(pos)].tail(1).item()
    entropy=entropy_wide["entropy_"+str(pos)].tail(1).item()
    rank = position_to_rank["probability_"+str(pos)]
    # rank non-reference peak:
    rank_nonref = position_to_nonref_rank[pos]
    max_nonref_prob = max_non_reference_probs[pos-1]  # Get the actual max non-ref probability

    results.append({
        "mutation": list(mutated_sites)[mut_num],
        "Position": pos,
        "Probability_reference": prob,
        "entropy":entropy,
        "prob_mutation":mut_info_combos["probability"][(mut_info_combos["Mutation"]==list(mutated_sites)[mut_num] ) & 
                                        (mut_info_combos["Backbone"]=="Reference" )].item(),
        "reference_probability_rank": rank,
        "Max_non_reference_probability": max_nonref_prob,
        "Rank_max_non_reference": rank_nonref
    })

# Convert the results list to a DataFrame for clean, tabular output
results_df = pd.DataFrame(results).sort_values(by="Position").reset_index(drop=True)
# flatten mutation_prob_matrix
mutation_prob_flat = mutation_prob_matrix.flatten()
mutation_prob_flat_clean = mutation_prob_flat[~np.isnan(mutation_prob_flat)]

# Calculate percentile of observed mutation probabilities against all possible non-reference mutation probabilities
results_df["probability_percentile"] = np.sum(mutation_prob_flat_clean < results_df["prob_mutation"].values[:, None], axis=1) / len(mutation_prob_flat_clean) * 100

results_df.to_csv(os.path.join(out, sub_mod+'_epistasis_results.csv'),index=False)


# %%
results_df
# note that rank max-non-reference can point to a different mutation

# %%
# make a histogram of reference probabilities
target_series = probability_wide.iloc[-1, 2:]
one_minus_ref_probs = 1 - target_series.values

# Create canonical mutation mapping for focal mutations
canonical_mutations = mutations_to_canonical(K_indexed_muts, h3_map_with_ha2)
canon_dict = {k: canonical_mutations[i] for i, k in enumerate(K_indexed_muts)}

# Create figure with two subplots
plt.figure(figsize=(16, 6))

# Subplot 1: Histogram
plt.subplot(1, 2, 1)
sns.histplot(one_minus_ref_probs, bins=50, kde=True)
plt.xlabel('1 - Reference Probability')
plt.ylabel('Count')
plt.title('Distribution of 1 - Reference Probabilities model{}'.format(sub_mod))

# Subplot 2: Rank vs log10(1 - Reference Probability)
plt.subplot(1, 2, 2)
# Sort by reference probability (ascending) to get ranks
sorted_indices = np.argsort(target_series.values)
sorted_ref_probs = target_series.values[sorted_indices]
sorted_one_minus = 1 - sorted_ref_probs
ranks = np.arange(1, len(sorted_one_minus) + 1)

# Convert to numpy array to avoid pandas Series issue
sorted_one_minus_array = np.array(sorted_one_minus, dtype=float)
plt.plot(ranks, np.log10(sorted_one_minus_array), linewidth=0.8, color='blue', alpha=0.5)

# Add scatter points and text labels for focal mutations
for mut in K_indexed_muts:
    # Extract position from mutation
    pos = int(re.search(r'\d+', mut).group())
    
    # Get the probability column name
    prob_col = f"probability_{pos}"
    
    # Find this position's reference probability
    if prob_col in target_series.index:
        ref_prob = target_series[prob_col]
        one_minus_prob = 1 - ref_prob
        
        # Find the rank of this position
        rank = position_to_rank[prob_col]
        
        # Get canonical name
        canon_name = canon_dict[mut]
        
        # Plot scatter point
        plt.scatter(rank, np.log10(one_minus_prob), color='red', s=50, zorder=5)
        
        # Add text label
        plt.text(rank, np.log10(one_minus_prob), f'  {canon_name}', 
                fontsize=8, ha='left', va='center', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))

plt.xlabel('Rank (1 = lowest reference probability)')
plt.ylabel('log10(1 - Reference Probability)')
plt.title('Rank vs log10(1 - Reference Probability) model{}'.format(sub_mod))
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(out, f"{sub_mod}_reference_probability_analysis.png"), dpi=300)
plt.show()

# Print summary statistics
print("Summary Statistics for 1 - Reference Probability:")
print(f"Mean: {one_minus_ref_probs.mean():.4f}")
print(f"Median: {np.median(one_minus_ref_probs):.4f}")
print(f"Min: {one_minus_ref_probs.min():.4f}")
print(f"Max: {one_minus_ref_probs.max():.4f}")
print(f"Std: {one_minus_ref_probs.std():.4f}")



# %%
# make a plot of listed mutation probabilities on the mutation_prob_matrix, labelling in cannonical form the observed mutations
# Convert to numpy array to avoid pandas Series issue
# %%
# Rank analysis of all possible non-reference mutations for J.2.4 (ids[4])

# 1. Identify the backbone sequence (J.2.4)
target_seq_id = ids[4]
target_sequence = sequences[target_seq_id]
print(f"Calculating mutation probabilities for {target_seq_id}")

# 2. Calculate the full mutation probability matrix (20 x L)
result = get_mutation_prob_matrix(target_sequence, model, model_layers, device, batch_converter, alphabet)
mutation_prob_matrix = result['mutation_matrix']
amino_acids = result['amino_acids']
positions = result['positions']

# 3. Flatten and create a list of all possible non-reference mutations (19 x L)
all_mutations_data = []
for i, pos in enumerate(positions):
    ref_aa = target_sequence[i]
    for j, aa in enumerate(amino_acids):
        # Skip the reference amino acid
        if aa != ref_aa:
            prob = mutation_prob_matrix[j, i]
            # Construct mutation string: Ref+Pos+Alt (e.g., A123T)
            mut_name = f"{ref_aa}{pos}{aa}"
            all_mutations_data.append({
                'mutation': mut_name,
                'probability': prob,
                'log10_prob': np.log10(prob + 1e-10) # Avoid log(0)
            })

# 4. Create DataFrame and rank by probability (descending)
all_muts_df = pd.DataFrame(all_mutations_data)
all_muts_df = all_muts_df.sort_values('probability', ascending=False).reset_index(drop=True)
all_muts_df['rank'] = all_muts_df.index + 1

# 5. Identify and extract data for the observed mutations (K_indexed_muts)
# Ensure canonical mapping is available
canonical_mutations = mutations_to_canonical(K_indexed_muts, h3_map_with_ha2)
canon_dict = {k: canonical_mutations[i] for i, k in enumerate(K_indexed_muts)}

observed_data = []
for mut in K_indexed_muts:
    # Find this mutation in the dataframe
    match = all_muts_df[all_muts_df['mutation'] == mut]
    if not match.empty:
        rank = match.iloc[0]['rank']
        log_prob = match.iloc[0]['log10_prob']
        canon_name = canon_dict.get(mut, mut)
        observed_data.append({
            'mutation': mut,
            'canon': canon_name,
            'rank': rank,
            'log10_prob': log_prob
        })
    else:
        print(f"Warning: Observed mutation {mut} not found in possible mutations list.")

# 6. Plotting
plt.figure(figsize=(12, 8))

# Plot all mutations as a line
plt.plot(all_muts_df['rank'], all_muts_df['log10_prob'], color='blue', linewidth=0.8, alpha=0.5, label='All possible mutations')

# Plot observed mutations as scatter points
for item in observed_data:
    plt.scatter(item['rank'], item['log10_prob'], color='red', s=50, zorder=5)
    plt.text(item['rank'], item['log10_prob'], f"  {item['canon']}", 
             fontsize=9, ha='left', va='center', weight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))

plt.xlabel('Rank (1 = Highest Probability)')
plt.ylabel('log10(Probability)')
plt.title(f'Rank vs log10(Probability) of all non-reference mutations\nBackbone: {target_seq_id.split("|")[-1]}')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out, f"{sub_mod}_all_mutations_rank_analysis.png"), dpi=300)
plt.show()

# Print summary of observed mutations
print("\nObserved Mutations Ranks:")
obs_df = pd.DataFrame(observed_data).sort_values('rank')
print(obs_df[['mutation', 'canon', 'rank', 'log10_prob']])

# %%

# seaborn heatmap of mutation_prob_flat
plt.figure(figsize=(20, 8))
#only label every tenth x axis tick
mutation_prob_matrix2 = np.nan_to_num(mutation_prob_matrix, nan=0.0)
# replace np.nan with zero for visualization

seaborn.heatmap(mutation_prob_matrix2, xticklabels=positions, yticklabels=amino_acids)
plt.xticks(ticks=np.arange(0, len(positions), 10), labels=np.array(positions)[::10])

# %%
# seaborn histogram of mutation_prob_flat panel side by side

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
# log x axis before plotting
mutation_prob_matrix_log=np.log10(mutation_prob_matrix + 1e-10)  # add small constant to avoid log(0)
plt.title("Histogram of log10 mutation probabilities (log x axis)")
sns.histplot(mutation_prob_matrix_log.flatten(), bins=100)
#plot again with log y axis
#log yaxis log x axis
plt.subplot(1, 2, 2)
plt.title("Histogram of mutation probabilities (log y-axis)")
sns.histplot(mutation_prob_matrix.flatten(), bins=100)
plt.yscale('log')


# %%
# Find top 10 non-reference mutations with highest probabilities in mutation_prob_matrix
# Flatten the matrix and get indices of top values
mutation_prob_matrix2 = np.nan_to_num(mutation_prob_matrix, nan=-np.inf)  # Replace NaN with -inf to ignore reference sites
top_10_indices = np.argsort(mutation_prob_matrix2.flatten())[-10:][::-1]  # Get top 10, descending

# Convert flat indices back to 2D coordinates
top_10_coords = np.unravel_index(top_10_indices, mutation_prob_matrix.shape)

# Flatten entropy for all positions to calculate percentiles
entropy_all = entropy_wide.iloc[-1, 2:].values  # All entropy values from reference

# Create a dataframe with the results
top_mutations = []
for i in range(10):
    aa_idx = top_10_coords[0][i]
    pos_idx = top_10_coords[1][i]
    position = positions[pos_idx]
    amino_acid = amino_acids[aa_idx]
    probability = mutation_prob_matrix[aa_idx, pos_idx]
    ref_aa = reference_spike_sequence[pos_idx]
    
    # Format as mutation string
    mutation_str = f"{ref_aa}{position}{amino_acid}"
    
    # Check if this mutation is in the mutations of interest (K_indexed_muts)
    # Look for exact match or matching position
    mutation_of_interest = None
    for mut in K_indexed_muts:
        # Extract position from mutation string (e.g., "S394N" -> 394)
        mut_pos = int(re.search(r'\d+', mut).group())
        if mut_pos == position:
            mutation_of_interest = mut
            break
    
    # Get entropy for this position from entropy_wide (last row = reference)
    entropy_at_position = entropy_wide[f"entropy_{position}"].tail(1).item()
    
    # Calculate entropy percentile (what percentage of sites have lower entropy)
    entropy_percentile = (np.sum(entropy_all < entropy_at_position) / len(entropy_all)) * 100
    
    top_mutations.append({
        'Rank': i + 1,
        'Mutation': mutation_str,
        'Position': position,
        'Reference_AA': ref_aa,
        'Mutant_AA': amino_acid,
        'Probability': probability,
        'Entropy': entropy_at_position,
        'Entropy_Percentile': entropy_percentile,
        'Mutation_of_Interest': mutation_of_interest if mutation_of_interest else 'NA',
        'AA_Index': aa_idx,
        'Position_Index': pos_idx
    })

top_mutations_df = pd.DataFrame(top_mutations)
print("Top 10 non-reference mutations with highest probabilities:")
top_mutations_df

# %%
mut_info_combos.head()


