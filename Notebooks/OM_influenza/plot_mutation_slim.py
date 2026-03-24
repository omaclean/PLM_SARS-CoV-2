# %% [markdown]
# # plot on structure

# %%

from pathlib import Path
import sys
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))
import sys, importlib
# module_name = "Functions"
# if module_name in sys.modules:
#     del sys.modules[module_name]
# Functions = importlib.import_module(module_name)

from Functions_HuggingFace import *


import re
import py3Dmol
from Bio import PDB, Align
from Bio.SeqUtils import seq1
import os
from collections import defaultdict
import argparse

# Global plotting style for larger text
import matplotlib as mpl
mpl.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 18,
    'axes.titleweight': 'bold',
})


def _extract_pdb_chain_sequences(pdb_file):
    """Return chain sequences and residue ids from a PDB file."""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("struct", pdb_file)

    chain_data = {}
    for model in structure:
        for chain in model:
            seq_chars = []
            residue_ids = []
            for residue in chain:
                if PDB.is_aa(residue):
                    try:
                        aa = seq1(residue.get_resname())
                    except Exception:
                        aa = "X"
                    seq_chars.append(aa)
                    residue_ids.append(residue.get_id())
            if seq_chars:
                chain_data[chain.id] = ("".join(seq_chars), residue_ids)
    return chain_data


def _alignment_indices(alignment):
    """Return aligned indices for query/target from a Biopython alignment."""
    try:
        return alignment.indices[0], alignment.indices[1]
    except AttributeError:
        user_indices = []
        pdb_indices = []
        for (u_start, u_end), (p_start, p_end) in zip(*alignment.aligned):
            user_indices.extend(range(u_start, u_end))
            pdb_indices.extend(range(p_start, p_end))
        return user_indices, pdb_indices


def summarize_pdb_alignment(pdb_file, user_sequence, mutation_list=None, threshold_score=50):
    """Print aligned region summary and mutation mapping to PDB residue numbers."""
    chain_data = _extract_pdb_chain_sequences(pdb_file)
    mutation_list = mutation_list or []
    mutation_positions = []
    for mut in mutation_list:
        match = re.search(r"(\d+)", mut)
        if match:
            mutation_positions.append((mut, int(match.group(1))))

    alignment_maps = {}

    for chain_id, (pdb_seq, residue_ids) in chain_data.items():
        alignment = align_sequences(user_sequence, pdb_seq, mode="local", open_gap_score=-10, extend_gap_score=-0.5)
        if alignment.score < threshold_score:
            continue

        user_indices, pdb_indices = _alignment_indices(alignment)
        if user_indices is None:
            continue
        if hasattr(user_indices, "size"):
            if user_indices.size == 0:
                continue
        elif len(user_indices) == 0:
            continue

        user_min = min(user_indices) + 1
        user_max = max(user_indices) + 1
        pdb_min = min(pdb_indices)
        pdb_max = max(pdb_indices)
        pdb_min_res = residue_ids[pdb_min][1]
        pdb_max_res = residue_ids[pdb_max][1]

        print(f"Aligned region (chain {chain_id}): user {user_min}-{user_max} -> PDB {pdb_min_res}-{pdb_max_res}")

        user_to_pdb = dict(zip(user_indices, pdb_indices))
        alignment_maps[chain_id] = (user_to_pdb, residue_ids)

    if mutation_positions:
        print("\nMutation mapping to PDB (by chain):")
        for mut, pos in mutation_positions:
            mapped = False
            for chain_id, (user_to_pdb, residue_ids) in alignment_maps.items():
                user_idx = pos - 1
                if user_idx in user_to_pdb:
                    pdb_idx = user_to_pdb[user_idx]
                    pdb_resnum = residue_ids[pdb_idx][1]
                    print(f"  {mut}: chain {chain_id} -> PDB residue {pdb_resnum}")
                    mapped = True
            if not mapped:
                print(f"  {mut}: OUTSIDE aligned PDB region")

    return alignment_maps


def flag_outside_mutations(mutation_list, alignment_maps):
    """Append ' (OUTSIDE)' to mutation labels that do not map to any PDB residue."""
    mapped_positions = set()
    for user_to_pdb, _ in alignment_maps.values():
        mapped_positions.update(user_to_pdb.keys())

    flagged = []
    for mut in mutation_list:
        match = re.search(r"(\d+)", mut)
        if not match:
            flagged.append(mut)
            continue
        pos = int(match.group(1)) - 1
        if pos in mapped_positions:
            flagged.append(mut)
        else:
            flagged.append(f"{mut} (OUTSIDE)")
    return flagged

# %%


# %%
# mutations = K_indexed_muts # e.g. ['A145K', 'G158E']
# pdb_path = "4FNK.pdb" # Ensure this file is in your directory
pdb_path_default="/home3/oml4h/PLM_SARS-CoV-2/Sequences/4WE4_assembly.pdb"

# view = visualize_mutations_on_pdb(pdb_path, user_k_seq, mutations)
# view.show()
model_name_default="ESM2-HA80"



sequence_path_default="/home3/oml4h/PLM_SARS-CoV-2/Sequences/huH3N2_HA_CDS.translated_OM_synth_extra_steps.fas"
reference_path_default="/home3/oml4h/PLM_SARS-CoV-2/Sequences/H3N2_canonical.fa"
# Allow command-line overrides when executed as a script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise mutations on PDB: accepts paths and indices to automate from Epistasis script")
    parser.add_argument("--pdb_path", help="Path to PDB/mmCIF file", default=pdb_path_default)
    parser.add_argument("--query_path", help="Path to query fasta", default=None)
    parser.add_argument("--reference_path", help="Path to reference fasta", default=None)
    parser.add_argument("--seq1_index", help="Index of reference sequence (0-based)", type=int, default=0)
    parser.add_argument("--seq2_index", help="Index of target sequence (0-based)", type=int, default=1)
    parser.add_argument("--outdir_base", help="Base output directory to create a subdir in", default=None)
    parser.add_argument("--subdir_name", help="Subdirectory name under outdir_base", default="mutation_slim")
    parser.add_argument("--epistasis_dir", help="Directory containing epistasis outputs (probability/entropy/mut_info files)", default=None)
    parser.add_argument("--model_name", help="Model name token to identify epistasis files (e.g. ESM2-HA80)", default=model_name_default)
    
    

    args = parser.parse_args()

    # Override defaults if provided
    if args.pdb_path:
        pdb_path = args.pdb_path
    # prefer CLI-provided query_path; otherwise use default
    if args.query_path:
        query_path = args.query_path
    else:
        query_path = sequence_path_default
        
    sequences = read_sequences_to_dict(query_path)
    ids = list(sequences.keys())
    # prefer CLI-provided reference_path; otherwise use default
    if args.reference_path:
        reference_path = args.reference_path
    else:
        reference_path = reference_path_default

    print(query_path)
    
    seq1_index = args.seq1_index
    seq2_index = args.seq2_index
    lineage_base=ids[seq1_index].split("|")[-1]
    
    model_name=args.model_name

    if args.outdir_base:
        outdir = os.path.join(args.outdir_base, args.subdir_name)
        os.makedirs(outdir, exist_ok=True)
    # optional epistasis outputs directory (from Epistasis_hugging_face)
    if args.epistasis_dir:
        epistasis_dir = args.epistasis_dir

    # print a quick mutation summary using chosen indices
    print(get_mutations(sequences[ids[seq1_index]], sequences[ids[seq2_index]]))
else:
    # when imported, preserve original behaviour but use safe defaults
    print(get_mutations(sequences[ids[seq1_index]], sequences[ids[seq2_index]]))

# Normalize CLI-selected reference/target IDs for the rest of the script
try:
    ref_id_cli = ids[seq1_index]
    target_id_cli = ids[seq2_index]
except Exception:
    ref_id_cli = ids[0]
    target_id_cli = ids[-1]

print(f"Using reference id: {ref_id_cli}; target id: {target_id_cli}")
    
# %%
# import entropy and reference


#find first id in list with lineage base in 
lineages=[str(x).split("|")[-1] for x in ids]
reference = next(id for id, lin in zip(ids, lineages) if lin == lineage_base)

reference_lineage=lineage_base

print(ids[2:(len(ids)-1)])
# Use CLI-selected IDs for mutation comparisons
K_indexed_muts = [m for m in get_mutations(sequences.get(ref_id_cli, sequences[reference]), sequences[target_id_cli]) if "del" not in m and '-' not in m]

# Print aligned region summary and map mutations to PDB residues
alignment_maps = summarize_pdb_alignment(
    pdb_path,
    sequences[target_id_cli],
    mutation_list=K_indexed_muts,
    threshold_score=50,
)

# Flag mutations that fall outside aligned PDB regions
K_indexed_muts_flagged = flag_outside_mutations(K_indexed_muts, alignment_maps)
print("Flagged mutations:", K_indexed_muts_flagged)

# Scale plot sizes relative to the number of mutations (baseline = 8 mutations → multiplier = 1)
plot_size_multiplier = max(1.0, len(K_indexed_muts) / 8)
print(f"Plot size multiplier: {plot_size_multiplier:.2f} ({len(K_indexed_muts)} mutations)")






# Determine source directory for epistasis outputs
ep_dir = None
if 'epistasis_dir' in globals() and epistasis_dir:
    ep_dir = epistasis_dir

if ep_dir:
    # Use user-provided epistasis output directory
    # flexible file discovery: search ep_dir for files containing tokens
    def find_file_with_tokens(directory, tokens, ext='.csv'):
        # Prefer files matching tokens and extension; fall back to any match without ext
        for root, _, files in os.walk(directory):
            for fn in files:
                if ext and not fn.lower().endswith(ext):
                    continue
                if all(t in fn for t in tokens):
                    return os.path.join(root, fn)
        # fallback: ignore extension
        for root, _, files in os.walk(directory):
            for fn in files:
                if all(t in fn for t in tokens):
                    return os.path.join(root, fn)
        return None

    prob_file = find_file_with_tokens(ep_dir, [model_name, 'probability'], ext='.csv')
    ent_file = find_file_with_tokens(ep_dir, [model_name, 'entropy'], ext='.csv')
    mutinfo_file = find_file_with_tokens(ep_dir, [model_name, 'mut_info_combos'], ext='.csv') or find_file_with_tokens(ep_dir, ['mut_info_combos', model_name], ext='.csv')
    backbone_file = find_file_with_tokens(ep_dir, ['H3_epistasis_mutation_info_spyros_model', model_name], ext='.csv') or find_file_with_tokens(ep_dir, ['H3_epistasis_mutation_info_spyros_model'], ext='.csv')

    if not prob_file or not ent_file or not mutinfo_file or not backbone_file:
        raise FileNotFoundError(f"Could not locate required epistasis files in {ep_dir}: found {prob_file}, {ent_file}, {mutinfo_file}, {backbone_file}")

    probability = pd.read_csv(prob_file)
    entropy = pd.read_csv(ent_file)
    backbone_mut_probs = pd.read_csv(backbone_file)
    mut_combos = pd.read_csv(mutinfo_file)
    # If no explicit outdir was provided, use the epistasis dir for outputs
    if 'outdir' not in globals():
        outdir = ep_dir
    os.makedirs(outdir, exist_ok=True)
else:
    try:
        outdir
    except NameError:
        outdir = "/home3/oml4h/PLM_SARS-CoV-2/Results/test/{}/plot_mutation_stuff{}".format(lineage_base,model_name)
        os.makedirs(outdir, exist_ok=True)
    else:
        os.makedirs(outdir, exist_ok=True)
    probability = pd.read_csv(f"/home3/oml4h/PLM_SARS-CoV-2/Results/test/{lineage_base}/{model_name}_probability.csv")
    entropy = pd.read_csv(f"/home3/oml4h/PLM_SARS-CoV-2/Results/test/{lineage_base}/{model_name}_entropy.csv")
    backbone_mut_probs = pd.read_csv(f"/home3/oml4h/PLM_SARS-CoV-2/Results/test/{lineage_base}/H3_epistasis_mutation_info_spyros_model_{model_name}_rel_J.csv")
    mut_combos = pd.read_csv(f"/home3/oml4h/PLM_SARS-CoV-2/Results/test/{lineage_base}/{model_name}_mut_info_combos.csv")

# Backward-compatible grammar columns for older exports
if 'focal_sequence_grammar' not in mut_combos.columns:
    if 'backbone_sequence_grammar' in mut_combos.columns and 'rel_seq_grammar' in mut_combos.columns:
        mut_combos['focal_sequence_grammar'] = mut_combos['backbone_sequence_grammar'] + mut_combos['rel_seq_grammar']
    elif 'rel_seq_grammar' in mut_combos.columns:
        mut_combos['focal_sequence_grammar'] = mut_combos['rel_seq_grammar']
if 'backbone_sequence_grammar' not in mut_combos.columns:
    if 'focal_sequence_grammar' in mut_combos.columns and 'rel_seq_grammar' in mut_combos.columns:
        mut_combos['backbone_sequence_grammar'] = mut_combos['focal_sequence_grammar'] - mut_combos['rel_seq_grammar']

# Drop J.2 lineage and swap to J.2_int where present
if 'lineage_backbone' in backbone_mut_probs.columns:
    backbone_mut_probs['lineage_backbone'] = backbone_mut_probs['lineage_backbone'].replace({'J.2': 'J.2_int'})

os.makedirs(outdir, exist_ok=True)
# Take the final row and extract columns from position 2 onwards as numpy arrays
entropy_vals = entropy.iloc[-1, 2:].values
probability_vals = probability.iloc[-1, 2:].values

print(f"Entropy shape: {entropy_vals.shape}")
print(f"Probability shape: {probability_vals.shape}")

# Extract position numbers from Mutation column in backbone_mut_probs
def extract_position_from_mutation(mutation_str):
    """Extract numeric position from mutation string like 'A145K' or 'HA2:S49N'"""
    match = re.search(r'[A-Z](\d+)', str(mutation_str))
    if match:
        return int(match.group(1))
    return None

# Create a mapping from position to canonical name
position_to_canon = {}
for _, row in backbone_mut_probs.iterrows():
    pos = extract_position_from_mutation(row['Mutation'])
    if pos is not None and pd.notna(row['canon']):
        # Store all canonical names for this position (in case of multiple)
        if pos not in position_to_canon:
            position_to_canon[pos] = []
        position_to_canon[pos]=row['canon']
                
# Create a mapping dictionary from Mutation to canon using backbone_mut_probs
mutation_to_canon_init = dict(zip(backbone_mut_probs['Mutation'], backbone_mut_probs['canon']))
mutation_to_canon = defaultdict(lambda: "Reference", mutation_to_canon_init)

# Map the canonical names to mut_combos
mut_combos["Focal_canon"] = mut_combos["Mutation"].map(mutation_to_canon)

mut_combos["Backbone_canon"] = mut_combos["Backbone"] .map(mutation_to_canon)

# Get unique positions that have mutations
mutated_positions = set(position_to_canon.keys())
print(f"Found {len(mutated_positions)} positions with mutations in backbone_mut_probs")
# %%
# Create color array: red if position is in mutated_positions, blue otherwise
# Positions in entropy/probability arrays are 1-indexed
colors = ['red' if (i+1) in mutated_positions else 'blue' for i in range(len(entropy_vals))]

# Create scatter plot with colors
plt.figure(figsize=(12 * plot_size_multiplier, 8 * plot_size_multiplier))
for color in ['blue', 'red']:
    mask = [c == color for c in colors]
    label = 'Mutated in backbones' if color == 'red' else 'Not mutated'
    plt.scatter(entropy_vals[mask], probability_vals[mask], 
                c=color, alpha=0.6, label=label)

# Add labels for mutated positions

for i in range(len(entropy_vals)):
    pos = i + 1  # 1-indexed
    if pos in position_to_canon:
        # Join multiple canonical names if there are any
        canon_names = position_to_canon[pos]
        plt.annotate(canon_names, 
                    (entropy_vals[i], probability_vals[i]),
                    fontsize=7, alpha=0.7,
                    xytext=(3, 3), textcoords='offset points')

plt.xlabel("Entropy")
plt.ylabel("Reference Probability")
plt.title(f"{model_name} Entropy vs Probability on {lineage_base} (muts vs K lin)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(outdir, f"{lineage_base}_{model_name}_entropy_vs_probability.png"), dpi=300)
plt.show()

# %%
# plot mutations on structure
print(position_to_canon)
print(mutation_to_canon)
print( K_indexed_muts)

view = visualise_mutations_on_pdb(pdb_path, sequences[target_id_cli], 
                                  K_indexed_muts_flagged,
                                  canonical_map=position_to_canon, 
                                  title="{} {} mutations".format(model_name, reference_lineage))
#view.show()
# save to file
print(outdir)
output_path = os.path.join(outdir, "{}_{}_{}_mutations_structure.html".format(lineage_base, model_name, reference_lineage))
print(output_path)
# Generate HTML and save it
html_content = view._make_html()
with open(output_path, 'w') as f:
    f.write(html_content)

#view.show()
# %%
# Create background_values as a dict with 1-based positions
probability_dict = {i+1: val for i, val in enumerate(probability.iloc[-1, 2:].values)}

view = visualise_mutations_on_pdb(
    pdb_path, 
    sequences[target_id_cli], 
    K_indexed_muts_flagged,
    background_values=probability_dict,
    canonical_map=mutation_to_canon,
    title=f"{model_name} Reference Probability"
)
#view.show()
with open(os.path.join(outdir, f"{lineage_base}_{model_name}_reference_probability_structure.html"), 'w') as f:
    f.write(view._make_html())


# %%
# Create background_values as a dict with 1-based positions
probability_dict = {i+1:  np.log10(1-val) for i, val in enumerate(probability.iloc[-1, 2:].values)}

view = visualise_mutations_on_pdb(
    pdb_path, 
    sequences[target_id_cli], 
    mutation_list=[], #K_indexed_muts,
    background_values=probability_dict,
    title=f"{model_name} log10 (1-Reference_Probability)"
)
#view.show()
# save plot as interactive html
# Save plot as interactive html
output_path = "{}{}_{}_{}_lin_mutations_probability.html".format(outdir, lineage_base, model_name, reference_lineage)
print(output_path)

# Generate HTML and save it
html_content = view._make_html()
with open(output_path, 'w') as f:
    f.write(html_content)

print(f"Saved to: {output_path}")

# %%
# Create background_values as a dict with 1-based positions
entropy_dict = {i+1: val for i, val in enumerate(entropy.iloc[-1, 2:].values)}

view = visualise_mutations_on_pdb(
    pdb_path, 
    sequences[target_id_cli], 
    K_indexed_muts_flagged,
    background_values=entropy_dict,
    title=f"{model_name} Reference entropy"
)
#view.show()
with open(os.path.join(outdir, f"{lineage_base}_{model_name}_reference_entropy_structure.html"), 'w') as f:
    f.write(view._make_html())


# %%
# Create background_values as a dict with 1-based positions
entropy_dict = {i+1:  np.log10(val) for i, val in enumerate(entropy.iloc[-1, 2:].values)}

view = visualise_mutations_on_pdb(
    pdb_path, 
    sequences[target_id_cli], 
    K_indexed_muts_flagged,
    background_values=entropy_dict,
    title=f"{model_name} Reference entropy (log10)"
)
#view.show()
with open(os.path.join(outdir, f"{lineage_base}_{model_name}_reference_log10_entropy_structure.html"), 'w') as f:
    f.write(view._make_html())


# %%
from Bio import SeqIO

# 1. Read the reference sequence (Assuming single sequence in file)
# We use 'next' to get the first item from the iterator
ref_record = next(SeqIO.parse(reference_path, "fasta"))
ref_seq_str = str(ref_record.seq)



# 2. Read the query sequences
# We parse the file and pick the first one as a test case
query_iterator = SeqIO.parse(query_path, "fasta")
first_query_record = next(query_iterator)

h3_map_with_ha2 = create_h3_numbering_map(first_query_record, ref_seq_str, HA2_start=330)

# Convert your mutations to canonical numbering
canonical_mutations = mutations_to_canonical(K_indexed_muts, h3_map_with_ha2)


# 3. Run the mapping function
# Note: We pass the whole record for the query, and the string for the reference
h3_map = create_h3_numbering_map(first_query_record, ref_seq_str)

# 4. Verify Output
print(f"Generated H3 map for: {first_query_record.id}")
print(f"Total mapped positions: {len(h3_map)}")
print("Sample (first 5 positions):", list(h3_map.items())[:5])

# %%
# Example: Convert mutations to canonical H3 numbering
# First, create h3_map with HA2_start parameter (HA2 typically starts around position 329)
h3_map_with_ha2 = create_h3_numbering_map(first_query_record, ref_seq_str, HA2_start=330)

# Convert your mutations to canonical numbering
canonical_mutations = mutations_to_canonical(K_indexed_muts, h3_map_with_ha2)

print("Original mutations (sequence numbering):")
print(K_indexed_muts[:10])  # Show first 10
print("\nCanonical mutations (H3 numbering):")
print(canonical_mutations[:10])  # Show first 10


# %%

view = visualise_mutations_on_pdb(
    pdb_path, 
    sequences[target_id_cli], 
    K_indexed_muts_flagged,
    background_values=entropy_dict,
    title=f"{model_name} Reference entropy (Canonical)",
    canonical_map=h3_map  # Now displays H3 canonical numbering in separate legend
)
view.show()
with open(os.path.join(outdir, f"{lineage_base}_{model_name}_reference_entropy_structure_canonical.html"), 'w') as f:
    f.write(view._make_html())

# %%
# 1. Setup Input Data
# The parent header provided
parent_header = "EPI4551140|HA|A/England/415/2024|EPI_ISL_20080368|J.2.4"

# PLACEHOLDER: Replace this string with your actual amino acid sequence for EPI4551140
# I have made this long enough to cover the T328A mutation position
parent_sequence = read_sequences_to_dict('/home2/oml4h/PLM_SARS-CoV-2/Sequences/huH3N2_HA_CDS.translated.fas')
parent_sequence=parent_sequence[parent_header]
# 2. Define the mutation steps
# Each step is a list of mutations to apply cumulatively to the previous result

#
steps_canonical = [
    ['I160K'],              # Step 1
    ['N158D', 'T328A'],     # Step 2
    ['S144N']               # Step 3
]

steps = [
    ['I176K'],              # Step 1
    ['N174D', 'T344A'],     # Step 2
    ['S160N']               # Step 3
]


# 3. Iterative Generation
current_sequence = parent_sequence
current_suffix = "" # Used to build the cumulative name part

print(f"Original Header: >{parent_header}\n")

for step_i, step_mutations in enumerate(steps):
    # Apply new mutations to the sequence from the PREVIOUS step
    # This works because mutate_sequence returns a full string, which becomes the input for the next round
    current_sequence = mutate_sequence(current_sequence, step_mutations)
    
    # Build the name suffix (e.g., _I160K then _I160K_N158D_T328A)
    # We join mutations with underscore and add them to the running suffix
    step_suffix = "_" + "_".join(steps_canonical[step_i])
    current_suffix += step_suffix
    
    # Construct the new header
    # We take the parent header and simply append the cumulative suffix
    new_header = f"{parent_header}{current_suffix}"
    
    # Print in FASTA format
    print(f">{new_header}")
    print(current_sequence)
    print() # Newline for readability

# %%
backbone_mut_probs.head()

# %%
# Explore epistatic interactions by comparing mutation probabilities across backbones

# 1. Filter data to get mutations and their probabilities on different backbones
# Assuming backbone_mut_probs has columns: Mutation, probability, Backbone (or lineage_backbone)

# Get the reference backbone (J lineage)
reference_backbone = "J"  # Should be 'J.2.4' or similar
print(f"Reference backbone: {reference_backbone}")

# Get all unique backbones
all_backbones = backbone_mut_probs['lineage_backbone'].unique()
print(f"All backbones: {all_backbones}")

# 2. Create a pivot table: mutations as rows, backbones as columns, probability as values
prob_pivot = backbone_mut_probs.pivot_table(
    index='canon', 
    columns='lineage_backbone', 
    values='probability',
    aggfunc='first'  # In case of duplicates, take first
)

# Order backbone columns: J, then J.2_int, then J.2.4, then the rest
preferred_backbones = [reference_backbone, 'J.2_int', 'J.2.4']
ordered_cols = [c for c in preferred_backbones if c in prob_pivot.columns] + [
    c for c in prob_pivot.columns if c not in preferred_backbones
]
prob_pivot = prob_pivot[ordered_cols]

print(f"\nProbability pivot shape: {prob_pivot.shape}")
print(prob_pivot.head())

# 3. Calculate probability shifts relative to reference backbone
if reference_backbone in prob_pivot.columns:
    # Calculate difference from reference for each backbone
    prob_shifts = prob_pivot.copy()
    gram_shifts = prob_pivot.copy()
    for col in prob_pivot.columns:
        
        prob_shifts[f'{col}_shift'] = prob_pivot[col] - prob_pivot[reference_backbone]
        gram_shifts[f'{col}_shift'] = np.log10(prob_pivot[col] / prob_pivot[reference_backbone])
    
    # Get only the shift columns
    shift_cols = [col for col in prob_shifts.columns if '_shift' in col]
    # Order columns to place J, then J.2_int, then J.2.4 (if present)
    preferred = [f"{reference_backbone}_shift", "J.2_int_shift", "J.2.4_shift"]
    shift_cols = [c for c in preferred if c in shift_cols] + [c for c in shift_cols if c not in preferred]
    prob_shifts_only = prob_shifts[shift_cols].copy()
    
    # 4. Find biggest shifts (epistatic interactions)
    # Calculate max absolute shift across all backbones for each mutation
    # Use skipna=True to ignore NaN values
    prob_shifts_only['max_abs_shift'] = prob_shifts_only[shift_cols].abs().max(axis=1, skipna=True)
    prob_shifts_only['max_shift'] = prob_shifts_only[shift_cols].max(axis=1, skipna=True)
    prob_shifts_only['min_shift'] = prob_shifts_only[shift_cols].min(axis=1, skipna=True)
    epistatic_ranking = prob_shifts_only[prob_shifts_only['max_abs_shift'].notna()].copy()
    # Add reference probability for context
    prob_shifts_only['ref_probability'] = prob_pivot[reference_backbone]
    
    # Filter out rows where all shifts are NaN (mutation not present in any other backbone)
    prob_shifts_only = prob_shifts_only[prob_shifts_only['max_abs_shift'].notna()]
    # Extract position numbers from canonical mutation names for sorting
    def extract_position(mutation_name):
        """Extract numeric position from mutation name for sorting"""
        # Handle HA2 mutations like "HA2:S49N"
        if mutation_name.startswith('HA2:'):
            match = re.search(r'HA2:[A-Z](\d+)', mutation_name)
            if match:
                return 10000 + int(match.group(1))  # Add 10000 to put HA2 after HA1
        # Handle signal peptide like "SP-15"
        elif mutation_name.startswith('SP'):
            match = re.search(r'SP-(\d+)', mutation_name)
            if match:
                return -int(match.group(1))  # Negative so SP comes first
        # Handle regular mutations and insertions like "S158N" or "N158AN"
        else:
            match = re.search(r'[A-Z](\d+)', mutation_name)
            if match:
                return int(match.group(1))
        return 0  # Fallback

    # Sort by number of non-zero shifts (fewest to most), then genomic position
    epistatic_ranking['genomic_position'] = epistatic_ranking.index.map(extract_position)
    epistatic_ranking['nonzero_shifts'] = prob_shifts_only[shift_cols].fillna(0).ne(0).sum(axis=1)
    epistatic_ranking = epistatic_ranking.sort_values(['nonzero_shifts', 'genomic_position'])

    # Update prob_shifts_only with the same order
    prob_shifts_only = prob_shifts_only.loc[epistatic_ranking.index]

    print("\n=== Top 20 Mutations with Largest Epistatic Shifts (Sorted by Genomic Position) ===")
    print(epistatic_ranking.head(20))
 
    
    # 5. Visualize top epistatic interactions
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Heatmap of top 15 mutations across backbones
    top_n = min(15, len(epistatic_ranking))  # Don't try to show more than available
    top_mutations = prob_shifts_only.index
    
    # Get probability data for these mutations
    top_prob_data = prob_pivot.loc[top_mutations]
    
    plt.figure(figsize=(12 * plot_size_multiplier, 8 * plot_size_multiplier))
    sns.heatmap(top_prob_data, annot=True, fmt='.3f', cmap='viridis', 
                center=top_prob_data.mean().mean(), cbar_kws={'label': 'Probability'},
                mask=top_prob_data.isna(), annot_kws={'size': 14})  # Mask NaN values
    plt.title(f'{model_name} Top {top_n} Epistatis search: Probabilities Across Backbones')
    plt.xlabel('Backbone Lineage')
    plt.ylabel('Mutation-canon name')
    plt.tight_layout()
    plt.yticks(rotation=0) 
    plt.savefig(os.path.join(outdir,f"{lineage_base}_{model_name}_epistatic_heatmap.png"), dpi=300)
    plt.show()
    
    # 6. Create shift heatmap (with reference lineage showing zeros)
    plt.figure(figsize=(12 * plot_size_multiplier, 8 * plot_size_multiplier))
    shift_data = prob_shifts_only.loc[top_mutations, shift_cols].copy()
    # Add reference backbone column with calculated zeros (shift from itself)
    #shift_data[f'{reference_backbone}_shift'] = 0.0
    # Reorder columns to put reference first
    cols_ordered =  shift_cols
    shift_data = shift_data[cols_ordered]
    sns.heatmap(shift_data, annot=True, fmt='.4f', cmap='viridis', 
                center=0, cbar_kws={'label': f'Probability Shift from Reference {lineage_base}'},
                mask=shift_data.isna(), annot_kws={'size': 14})  # Mask NaN values
    plt.title(f'{model_name} Top {top_n} Epistatic Mutations: Probability Shifts from Reference {lineage_base}')
    plt.xlabel('Backbone Lineage')
    plt.ylabel('Mutation -canon name')
    plt.yticks(rotation=0) 
    plt.tight_layout()
    plt.savefig(os.path.join(outdir,f"{lineage_base}_{model_name}_epistatic_shifts.png"), dpi=300)
    plt.show()

    # 7. Create grammar heatmap (with reference lineage showing zeros)
    plt.figure(figsize=(12 * plot_size_multiplier, 8 * plot_size_multiplier))
    gram_data = gram_shifts.loc[top_mutations, shift_cols].copy()
    # Add reference backbone column with calculated zeros (log10(1) = 0)
    #gram_data[f'{reference_backbone}_shift'] = 0.0
    # Reorder columns to put reference first
    cols_ordered =  shift_cols
    gram_data = gram_data[cols_ordered]
    sns.heatmap(gram_data, annot=True, fmt='.4f', cmap='viridis', 
                center=0, cbar_kws={'label': f'Mutation Grammaticality Shift from Reference {lineage_base}'},
                mask=gram_data.isna(), annot_kws={'size': 14})  # Mask NaN values
    plt.title(f'{model_name} Top {top_n} Epistatic Mutations: log10(probx/prob_root) Shifts from Reference {lineage_base}')
    plt.xlabel('Backbone Lineage')
    plt.ylabel('Mutation -canon name')
    plt.tight_layout()
    plt.yticks(rotation=0) 
    plt.savefig(os.path.join(outdir,f"{lineage_base}_{model_name}_epistatic_shifts_gram.png"), dpi=300)
    plt.show()
    
    # 8. Identify specific epistatic pairs (which backbone causes biggest shift for each mutation)
    epistatic_pairs = []
    for mutation in top_mutations:
        shifts = prob_shifts_only.loc[mutation, shift_cols]
        # Skip if all shifts are NaN
        if shifts.notna().sum() == 0:
            continue
            
        # Get the column with max absolute shift (ignoring NaN)
        max_shift_col = shifts.abs().idxmax(skipna=True)
        
        # Skip if max_shift_col is NaN
        if pd.isna(max_shift_col):
            continue
            
        max_shift_val = shifts[max_shift_col]
        backbone_name = max_shift_col.replace('_shift', '')
        
        epistatic_pairs.append({
            'Mutation': mutation,
            'Strongest_Epistatic_Backbone': backbone_name,
            'Probability_Shift': max_shift_val,
            'Ref_Probability': prob_shifts_only.loc[mutation, 'ref_probability'],
            'New_Probability': prob_pivot.loc[mutation, backbone_name] if backbone_name in prob_pivot.columns else np.nan
        })
    
    epistatic_pairs_df = pd.DataFrame(epistatic_pairs)
    print("\n=== Strongest Epistatic Pairs ===")
    print(epistatic_pairs_df)
    
    # Save results
    epistatic_ranking.to_csv(os.path.join(outdir,
        f"{lineage_base}_{model_name}_epistatic_ranking.csv")
    )
    epistatic_pairs_df.to_csv(os.path.join(outdir,
        f"{lineage_base}_{model_name}_epistatic_pairs.csv"),
        index=False
    )
    
    print(f"\nSaved epistatic analysis to Results/test/")
    
else:
    print(f"Error: Reference backbone '{reference_backbone}' not found in data")
    print(f"Available backbones: {prob_pivot.columns.tolist()}")

# %%

mut_combos.head()


# Create pivot tables for probability and grammar
mut_combo_probability_matrix = mut_combos.pivot_table(
    index='Focal_canon', 
    columns='Backbone_canon', 
    values='probability'
)

# Absolute grammar on each backbone (focal sequence grammar)
mut_combo_grammar_matrix = mut_combos.pivot_table(
    index='Focal_canon',
    columns='Backbone_canon',
    values='focal_sequence_grammar'
)



# Extract position for sorting
def extract_pos(mut):
    import re
    # Look for the first sequence of digits
    match = re.search(r'(\d+)', str(mut))
    return int(match.group(1)) if match else 999999

# Get unique pairs of Focal_canon and Mutation to determine order
unique_focals = mut_combos[['Focal_canon', 'Mutation']].drop_duplicates('Focal_canon').copy()
unique_focals['position'] = unique_focals['Mutation'].apply(extract_pos)
unique_focals = unique_focals.sort_values('position')

original_order = unique_focals['Focal_canon'].tolist()

print(original_order)
# 2. Filter out 'Reference' if it exists in the original list
#    This ensures 'Reference' only appears once.
filtered_order = [c for c in original_order if c != 'Reference']

# 3. Concatenate the lists to place 'Reference' first (both rows and columns)
cols = ['Reference'] + filtered_order
rows = ['Reference'] + filtered_order  # Reference as first row too
mut_combo_probability_matrix = mut_combo_probability_matrix.reindex(index=rows, columns=cols)
mut_combo_grammar_matrix = mut_combo_grammar_matrix.reindex(index=rows, columns=cols)
# Plot Probability Matrix
plt.figure(figsize=(12 * plot_size_multiplier, 8 * plot_size_multiplier))
sns.heatmap(mut_combo_probability_matrix, annot=True, fmt='.3f', cmap='viridis', 
            cbar_kws={'label': 'Probability'}, annot_kws={'size': 14})
plt.title(f'{model_name} Mutation Probability Matrix')
plt.tight_layout()
plt.savefig(os.path.join(outdir, f"{lineage_base}_{model_name}_mutation_probability_matrix.png"), dpi=300)
plt.show()

# Plot Log10 Probability Matrix
plt.figure(figsize=(12 * plot_size_multiplier, 8 * plot_size_multiplier))
sns.heatmap(np.log10(mut_combo_probability_matrix), annot=True, fmt='.3f', cmap='viridis', 
            cbar_kws={'label': 'Log10 Probability'}, annot_kws={'size': 14})
plt.title(f'{model_name} Log10 Mutation Probability Matrix')
plt.tight_layout()
plt.savefig(os.path.join(outdir, f"{lineage_base}_{model_name}_log10_mutation_probability_matrix.png"), dpi=300)
plt.show()

# Plot Grammar on Backbone (absolute)
plt.figure(figsize=(12 * plot_size_multiplier, 8 * plot_size_multiplier))
sns.heatmap(mut_combo_grammar_matrix, annot=True, fmt='.3f', cmap='viridis', center=0,
            cbar_kws={'label': 'Focal Sequence Grammar'}, annot_kws={'size': 14})
plt.title(f'{model_name} Focal Sequence Grammar on Backbone')
plt.tight_layout()
plt.savefig(os.path.join(outdir, f"{lineage_base}_{model_name}_focal_sequence_grammar_matrix.png"), dpi=300)
plt.show()

# %%

# --- Relative Shift Plots ---

# 1. Probability Shift (Prob - Ref_Prob)
prob_shift_matrix = mut_combo_probability_matrix.subtract(mut_combo_probability_matrix['Reference'], axis=0)


plt.figure(figsize=(12 * plot_size_multiplier, 8 * plot_size_multiplier))
sns.heatmap(prob_shift_matrix, annot=True, fmt='.3f', cmap='viridis', center=0,
            cbar_kws={'label': f'Probability Shift (vs Reference {lineage_base})'}, annot_kws={'size': 14})
plt.title(f'{model_name} Probability Shift Matrix (Relative to Reference {lineage_base})')
plt.tight_layout()
plt.savefig(os.path.join(outdir, f"{lineage_base}_{model_name}_probability_shift_matrix.png"), dpi=300)
plt.show()
 
# 2. log the probability shifts
# Calculate difference of logs (Log Fold Change) instead of log of difference
# This avoids NaNs for negative shifts (where probability decreases)
log_mut_combo_probability_matrix = np.log10(mut_combo_probability_matrix + 1e-10)
log_prob_shift_matrix = log_mut_combo_probability_matrix.subtract(log_mut_combo_probability_matrix['Reference'], axis=0)
log_prob_shift_matrix.iloc[:,0]= np.nan


plt.figure(figsize=(12 * plot_size_multiplier, 8 * plot_size_multiplier))
sns.heatmap(log_prob_shift_matrix, annot=True, fmt='.3f', cmap='viridis', center=0,
            cbar_kws={'label': f'Log10 Probability Shift (vs Reference {lineage_base})'}, annot_kws={'size': 14})
plt.title(f'{model_name} Log10 Probability Shift Matrix (Relative to Reference {lineage_base})')
plt.tight_layout()
plt.savefig(os.path.join(outdir, f"{lineage_base}_{model_name}_log10_probability_shift_matrix.png"), dpi=300)
plt.show()

# %%
# Grammar delta relative to OG reference
# Align reference backbone grammar values to the backbone columns

reference_backbone_grammar = mut_combos.loc[
    (mut_combos['Mutation'] == "Reference") &
    (mut_combos['Backbone'] == "Reference"),
    "backbone_sequence_grammar"
].values[0]  # or use .iloc[0] or .item()

mut_combo_grammar_delta_matrix = mut_combo_grammar_matrix.subtract(reference_backbone_grammar, axis=1)
# Ensure same ordering (Reference first) as other matrices
try:
    mut_combo_grammar_delta_matrix = mut_combo_grammar_delta_matrix.reindex(index=rows, columns=cols)
except NameError:
    # If rows/cols not defined for some reason, skip reindex
    pass

# 3. Grammar Delta Matrix - Reference row/col in lower half, pairwise in upper half
grammar_shift_matrix = mut_combo_grammar_delta_matrix.copy()

# Populate Reference row with individual mutation effects from Reference column
# The Reference row shows: effect of each mutation (columns) applied alone to Reference
# This is the same as the Reference column data, so copy it to make the row visible
for col_name in grammar_shift_matrix.columns:
    if col_name != 'Reference' and col_name in grammar_shift_matrix.index:
        # Copy the value from Reference column (row=col_name) to Reference row (col=col_name)
        grammar_shift_matrix.loc['Reference', col_name] = grammar_shift_matrix.loc[col_name, 'Reference']

# Configuration for separator lines
SEPARATOR_COLOR = 'red'  # Easy to change: try 'black', 'red', 'white', etc.
SEPARATOR_WIDTH = 4

# Create mask: 
# - Reference row (row 0): show all (no mask)
# - Reference column (col 0): show row 1 onwards (mask row 0 to avoid double-counting)
# - Pairwise section (rows 1+, cols 1+): show upper triangle only
n_rows, n_cols = grammar_shift_matrix.shape
mask = np.zeros((n_rows, n_cols), dtype=bool)

# For pairwise section (excluding Reference row/col): mask lower triangle + diagonal
if n_rows > 1 and n_cols > 1:
    # Create lower triangle mask for the pairwise section (rows 1:, cols 1:)
    pairwise_size = n_rows - 1
    lower_tri_pairwise = np.tril(np.ones((pairwise_size, pairwise_size), dtype=bool), k=0)  # k=0 includes diagonal
    mask[1:, 1:] = lower_tri_pairwise

# Mask the Reference-Reference cell (row 0, col 0) - it's redundant
mask[0, 0] = True

# Ensure Reference column is visible for rows 1+ (not masked)
mask[1:, 0] = False  # Show Reference column for all mutation rows

print(reference_backbone_grammar)
print(mut_combo_grammar_matrix.iloc[0:5,0:5])
print(mut_combo_grammar_delta_matrix.iloc[0:5,0:5])

# Create figure with visual separation between reference and pairwise sections
fig, ax = plt.subplots(figsize=(14 * plot_size_multiplier, 10 * plot_size_multiplier))

# Calculate symmetric vmin/vmax from visible data only (respecting mask)
visible_data = grammar_shift_matrix.values[~mask]
visible_valid = visible_data[~np.isnan(visible_data)]
if len(visible_valid) > 0:
    vmax_grammar = np.abs(visible_valid).max()
    vmin_grammar, vmax_grammar = -vmax_grammar, vmax_grammar
else:
    vmin_grammar, vmax_grammar = None, None

# Plot the heatmap
sns.heatmap(grammar_shift_matrix, annot=True, fmt='.3f', cmap='viridis', center=-0.5,
            vmin=vmin_grammar, vmax=vmax_grammar,
            cbar_kws={'label': f'Grammar Delta (vs Reference {lineage_base})'}, annot_kws={'size': 12},
            mask=mask, ax=ax, linewidths=0.5, linecolor='white')

# Add visual separator lines between Reference row/col and pairwise section
ax.axvline(x=1, color=SEPARATOR_COLOR, linewidth=SEPARATOR_WIDTH)
ax.axhline(y=1, color=SEPARATOR_COLOR, linewidth=SEPARATOR_WIDTH)

# Update axis labels
ax.set_xlabel('First Mutation (Backbone)', fontsize=14, fontweight='bold')
ax.set_ylabel('Extra Mutation (Focal)', fontsize=14, fontweight='bold')
plt.title(f'{model_name} Grammar Delta Matrix (vs Reference {lineage_base})\n(Reference row/col: single effects | Pairwise: upper triangle)')
plt.yticks(rotation=0)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(outdir, f"{lineage_base}_{model_name}_grammar_delta_matrix.png"), dpi=300)
plt.show()

# %%
# 4. Epistasis Detection Plot: Expected vs Observed grammar effects
# Expected = sum of individual mutation effects on reference
# Observed = actual pairwise grammar delta
# Epistasis = Observed - Expected

# Get the individual effects from Reference column (mutations on reference backbone)
individual_effects = grammar_shift_matrix['Reference'].dropna()

# Create epistasis matrix: for each pair (row=extra mut, col=first mut),
# Expected = individual_effects[row] + individual_effects[col]
# Observed = grammar_shift_matrix[row, col]
# Epistasis = Observed - Expected

epistasis_matrix = grammar_shift_matrix.copy()
expected_matrix = grammar_shift_matrix.copy()

for row_idx, row_name in enumerate(grammar_shift_matrix.index):
    for col_idx, col_name in enumerate(grammar_shift_matrix.columns):
        if col_name == 'Reference':
            # Reference column - set epistasis to NaN (no pairwise comparison)
            epistasis_matrix.iloc[row_idx, col_idx] = np.nan
            expected_matrix.iloc[row_idx, col_idx] = np.nan
        elif row_name in individual_effects.index and col_name in individual_effects.index:
            # Calculate expected additive effect
            expected = individual_effects[row_name] + individual_effects[col_name]
            expected_matrix.iloc[row_idx, col_idx] = expected
            observed = grammar_shift_matrix.iloc[row_idx, col_idx]
            if pd.notna(observed):
                epistasis_matrix.iloc[row_idx, col_idx] = observed - expected
            else:
                epistasis_matrix.iloc[row_idx, col_idx] = np.nan
        else:
            epistasis_matrix.iloc[row_idx, col_idx] = np.nan
            expected_matrix.iloc[row_idx, col_idx] = np.nan

# Create mask for epistasis plot (same as pairwise section - upper triangle only, no reference row/col)
epistasis_mask = np.ones((n_rows, n_cols), dtype=bool)
if n_rows > 1 and n_cols > 1:
    # Show upper triangle for pairwise section (rows 1+, cols 1+)
    pairwise_size = n_rows - 1
    upper_tri_pairwise = np.triu(np.ones((pairwise_size, pairwise_size), dtype=bool), k=1)  # k=1 excludes diagonal
    epistasis_mask[1:, 1:] = ~upper_tri_pairwise  # Invert: False = show

# Show Reference row (row 0) for individual effects context
epistasis_mask[0, 1:] = False  # Show Reference row (except Reference-Reference cell)

# Show Reference column for rows 1+ (individual effects)
epistasis_mask[1:, 0] = False

# Populate Reference row AND column with individual effects
for col_name in epistasis_matrix.columns:
    if col_name != 'Reference' and col_name in individual_effects.index:
        # Reference row: individual effect of each mutation
        epistasis_matrix.loc['Reference', col_name] = individual_effects[col_name]

for row_name in epistasis_matrix.index:
    if row_name != 'Reference' and row_name in individual_effects.index:
        # Reference column: individual effect of each mutation  
        epistasis_matrix.loc[row_name, 'Reference'] = individual_effects[row_name]

# Calculate vmin/vmax from pairwise section only (exclude Reference row/col from color scale)
pairwise_data = epistasis_matrix.iloc[1:, 1:].values
pairwise_valid = pairwise_data[~np.isnan(pairwise_data) & ~epistasis_mask[1:, 1:]]
if len(pairwise_valid) > 0:
    vmax_pairwise = np.abs(pairwise_valid).max()
    vmin_epistasis, vmax_epistasis = -vmax_pairwise, vmax_pairwise
else:
    vmin_epistasis, vmax_epistasis = None, None

# Plot Epistasis Matrix
fig, ax = plt.subplots(figsize=(14 * plot_size_multiplier, 10 * plot_size_multiplier))
sns.heatmap(epistasis_matrix, annot=True, fmt='.3f', cmap='viridis', center=0,
            vmin=vmin_epistasis, vmax=vmax_epistasis,
            cbar_kws={'label': 'Epistasis (Observed - Expected)'}, annot_kws={'size': 12},
            mask=epistasis_mask, ax=ax, linewidths=0.5, linecolor='white')

# Add separator lines (same style as grammar delta plot)
ax.axvline(x=1, color=SEPARATOR_COLOR, linewidth=SEPARATOR_WIDTH)
ax.axhline(y=1, color=SEPARATOR_COLOR, linewidth=SEPARATOR_WIDTH)

ax.set_xlabel('First Mutation (Backbone)', fontsize=14, fontweight='bold')
ax.set_ylabel('Extra Mutation (Focal)', fontsize=14, fontweight='bold')
plt.title(f'{model_name} Epistasis Detection\n(Positive = synergistic, Negative = antagonistic)')
plt.yticks(rotation=0)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(outdir, f"{lineage_base}_{model_name}_epistasis_detection.png"), dpi=300)
plt.show()

# Print summary of strongest epistatic interactions
epistasis_flat = epistasis_matrix.unstack().dropna().sort_values()
print("\n=== Top 5 Antagonistic (Negative) Epistatic Pairs ===")
print(epistasis_flat.head(5))
print("\n=== Top 5 Synergistic (Positive) Epistatic Pairs ===")
print(epistasis_flat.tail(5))



# %%
print(outdir)
# %%
# Query-path ordered epistasis heatmaps + evolutionary flow checks

def _extract_position_for_sort(mutation_name):
    """Extract numeric position from canonical mutation names for plotting order."""
    mutation_name = str(mutation_name)
    if mutation_name.startswith('HA2:'):
        match = re.search(r'HA2:[A-Z](\d+)', mutation_name)
        if match:
            return 10000 + int(match.group(1))
    elif mutation_name.startswith('SP'):
        match = re.search(r'SP-?(\d+)', mutation_name)
        if match:
            return -int(match.group(1))
    match = re.search(r'[A-Z](\d+)', mutation_name)
    if match:
        return int(match.group(1))
    return 0


def _query_lineage_order_from_ids(id_list):
    """Return unique lineage names preserving FASTA query order."""
    ordered = []
    seen = set()
    for seq_id in id_list:
        lineage = str(seq_id).split("|")[-1]
        if lineage not in seen:
            seen.add(lineage)
            ordered.append(lineage)
    return ordered


query_lineage_order = _query_lineage_order_from_ids(ids)
reference_backbone = query_lineage_order[0]
print(f"Reference backbone (first in query): {reference_backbone}")
print(f"Query lineage order: {query_lineage_order}")

if 'lineage_backbone' not in backbone_mut_probs.columns and 'Backbone' in backbone_mut_probs.columns:
    backbone_mut_probs['lineage_backbone'] = backbone_mut_probs['Backbone'].astype(str).str.split('|').str[-1]

all_backbones = list(backbone_mut_probs['lineage_backbone'].dropna().unique())
ordered_backbones = [x for x in query_lineage_order if x in all_backbones] + [x for x in all_backbones if x not in query_lineage_order]
print(f"Backbones in data: {all_backbones}")
print(f"Backbones ordered by query path: {ordered_backbones}")

prob_pivot = backbone_mut_probs.pivot_table(
    index='canon',
    columns='lineage_backbone',
    values='probability',
    aggfunc='first'
)
prob_pivot = prob_pivot.reindex(columns=ordered_backbones)

print(f"\nProbability pivot shape: {prob_pivot.shape}")
print(prob_pivot.head())

if reference_backbone in prob_pivot.columns:
    prob_shifts = prob_pivot.copy()
    gram_shifts = prob_pivot.copy()
    ref_vals = prob_pivot[reference_backbone]

    for col in prob_pivot.columns:
        prob_shifts[f'{col}_shift'] = prob_pivot[col] - ref_vals
        ratio = (prob_pivot[col] + 1e-12) / (ref_vals + 1e-12)
        gram_shifts[f'{col}_shift'] = np.log10(ratio)

    shift_cols = [f"{b}_shift" for b in ordered_backbones if f"{b}_shift" in prob_shifts.columns]
    prob_shifts_only = prob_shifts[shift_cols].copy()
    prob_shifts_only['max_abs_shift'] = prob_shifts_only[shift_cols].abs().max(axis=1, skipna=True)
    prob_shifts_only['max_shift'] = prob_shifts_only[shift_cols].max(axis=1, skipna=True)
    prob_shifts_only['min_shift'] = prob_shifts_only[shift_cols].min(axis=1, skipna=True)
    prob_shifts_only['ref_probability'] = ref_vals

    epistatic_ranking = prob_shifts_only[prob_shifts_only['max_abs_shift'].notna()].copy()

    mutation_first_step = {}
    for mutation in epistatic_ranking.index:
        first_step = len(ordered_backbones)
        for step_i, backbone in enumerate(ordered_backbones):
            if backbone in prob_pivot.columns and pd.notna(prob_pivot.loc[mutation, backbone]):
                first_step = step_i
                break
        mutation_first_step[mutation] = first_step

    epistatic_ranking['first_seen_step'] = epistatic_ranking.index.map(mutation_first_step)

    # Row order by mutation-introduction step along query-file sequence order.
    # This intentionally avoids genomic-position ordering.
    introduced_canon_order = []
    seen_canon = set()
    ordered_ids_for_introduction = [seq_id for seq_id in ids if str(seq_id).split('|')[-1] in ordered_backbones]
    root_sequence_for_introduction = sequences[ordered_ids_for_introduction[0]] if ordered_ids_for_introduction else sequences[ids[0]]
    for seq_id in ordered_ids_for_introduction:
        step_muts = [
            m for m in get_mutations(root_sequence_for_introduction, sequences[seq_id])
            if "del" not in m and "-" not in m
        ]
        for mut in step_muts:
            canon_mut = mutation_to_canon.get(mut, mut)
            if canon_mut in epistatic_ranking.index and canon_mut not in seen_canon:
                introduced_canon_order.append(canon_mut)
                seen_canon.add(canon_mut)

    remaining_mutations = [m for m in epistatic_ranking.index if m not in seen_canon]
    ordered_mutations = introduced_canon_order + remaining_mutations
    epistatic_ranking = epistatic_ranking.loc[ordered_mutations]
    top_n = min(40, len(ordered_mutations))
    top_mutations = ordered_mutations[:top_n]

    print("\n=== Ordered Epistatic Mutations (introduction order from query file) ===")
    print(epistatic_ranking.head(20))

    top_prob_data = prob_pivot.loc[top_mutations, ordered_backbones]
    plt.figure(figsize=(14, 10))
    sns.heatmap(top_prob_data, annot=True, fmt='.3f', cmap='viridis',
                center=top_prob_data.mean().mean(), cbar_kws={'label': 'Probability'},
                mask=top_prob_data.isna(), annot_kws={'size': 14})
    plt.title(f'{lineage_base}_{model_name} Probability Heatmap (Query-Ordered Backbones)')
    plt.xlabel('Backbone Lineage (query order)')
    plt.ylabel('Mutation (ordered by first appearance step)')
    plt.tight_layout()
    plt.yticks(rotation=0)
    plt.savefig(os.path.join(outdir, f"{lineage_base}_{model_name}_epistatic_heatmap_query_ordered.png"), dpi=300)
    plt.show()

    shift_data = prob_shifts_only.loc[top_mutations, shift_cols]
    plt.figure(figsize=(14, 10))
    sns.heatmap(shift_data, annot=True, fmt='.3f', cmap='viridis',
                center=0, cbar_kws={'label': f'Probability Shift from Reference {reference_backbone}'},
                mask=shift_data.isna(), annot_kws={'size': 14})
    plt.title(f'{lineage_base}_{model_name} Probability Shifts (Query-Ordered Backbones)')
    plt.xlabel('Backbone Lineage (query order)')
    plt.ylabel('Mutation (ordered by first appearance step)')
    plt.tight_layout()
    plt.yticks(rotation=0)
    plt.savefig(os.path.join(outdir, f"{lineage_base}_{model_name}_epistatic_shifts_query_ordered.png"), dpi=300)
    plt.show()

    gram_data = gram_shifts.loc[top_mutations, shift_cols]
    plt.figure(figsize=(14, 10))
    sns.heatmap(gram_data, annot=True, fmt='.3f', cmap='viridis',
                center=0, cbar_kws={'label': f'log10(Prob/RefProb) from {reference_backbone}'},
                mask=gram_data.isna(), annot_kws={'size': 14})
    plt.title(f'{lineage_base}_{model_name} Grammar-Like Shifts (Query-Ordered Backbones)')
    plt.xlabel('Backbone Lineage (query order)')
    plt.ylabel('Mutation (ordered by first appearance step)')
    plt.tight_layout()
    plt.yticks(rotation=0)
    plt.savefig(os.path.join(outdir, f"{lineage_base}_{model_name}_epistatic_shifts_gram_query_ordered.png"), dpi=300)
    plt.show()

    epistatic_ranking.to_csv(os.path.join(outdir, f"{lineage_base}_{model_name}_epistatic_ranking_query_ordered.csv"))

    # Extra plot: raw PLM probabilities at each node (no shift from reference).
    full_prob_data = prob_pivot.loc[ordered_mutations, ordered_backbones]
    plt.figure(figsize=(14, max(8, int(0.35 * len(ordered_mutations)))))
    sns.heatmap(full_prob_data, annot=True, fmt='.3f', cmap='viridis',
                cbar_kws={'label': 'Raw PLM Probability'},
                mask=full_prob_data.isna(), annot_kws={'size': 14})
    plt.title(f'{lineage_base}_{model_name} Raw PLM Probabilities by Node (Query Order)')
    plt.xlabel('Node / Backbone (query order)')
    plt.ylabel('Mutation (introduction order from query file)')
    plt.tight_layout()
    plt.yticks(rotation=0)
    plt.savefig(os.path.join(outdir, f"{lineage_base}_{model_name}_raw_plm_probabilities_by_node.png"), dpi=300)
    plt.show()

else:
    print(f"Error: Reference backbone '{reference_backbone}' not found in data")
    print(f"Available backbones: {prob_pivot.columns.tolist()}")


# Evolution flow checks: ensure mutations do not dead-end along query sequence order.
print("\n=== Evolution Flow Check (query sequence order) ===")
ordered_ids = ids
root_id = ordered_ids[0]
root_sequence = sequences[root_id]

lineage_by_step = [str(x).split('|')[-1] for x in ordered_ids]
mutation_sets_by_step = []
for seq_id in ordered_ids:
    seq_mutations = {
        m for m in get_mutations(root_sequence, sequences[seq_id])
        if "del" not in m and "-" not in m
    }
    mutation_sets_by_step.append(seq_mutations)

dead_end_rows = []
for step_i, muts_now in enumerate(mutation_sets_by_step[:-1]):
    future_union = set().union(*mutation_sets_by_step[step_i + 1:])
    dead_ends = sorted(m for m in muts_now if m not in future_union)
    if dead_ends:
        dead_end_rows.append({
            'step_index': step_i,
            'lineage': lineage_by_step[step_i],
            'dead_end_mutations': ';'.join(dead_ends),
            'count': len(dead_ends),
        })

if dead_end_rows:
    dead_end_df = pd.DataFrame(dead_end_rows)
    print("WARNING: dead-end mutations detected (present at a step but absent from all subsequent steps).")
    print(dead_end_df[['step_index', 'lineage', 'count', 'dead_end_mutations']])
    dead_end_df.to_csv(os.path.join(outdir, f"{lineage_base}_{model_name}_evolution_dead_end_warnings.csv"), index=False)
else:
    print("No dead-end mutations detected across the query sequence order.")

flow_rows = []
for step_i in range(1, len(mutation_sets_by_step)):
    prev_muts = mutation_sets_by_step[step_i - 1]
    curr_muts = mutation_sets_by_step[step_i]
    gained = sorted(curr_muts - prev_muts)
    lost = sorted(prev_muts - curr_muts)
    flow_rows.append({
        'from_step': step_i - 1,
        'to_step': step_i,
        'from_lineage': lineage_by_step[step_i - 1],
        'to_lineage': lineage_by_step[step_i],
        'gained_count': len(gained),
        'lost_count': len(lost),
        'gained_mutations': ';'.join(gained),
        'lost_mutations': ';'.join(lost),
    })

flow_df = pd.DataFrame(flow_rows)
flow_df.to_csv(os.path.join(outdir, f"{lineage_base}_{model_name}_evolution_flow_summary.csv"), index=False)
print("Saved evolution flow summary.")