# %%
%load_ext autoreload
%autoreload 2
import sys
sys.path.append('../../')

import sys, importlib
module_name = "Functions"
if module_name in sys.modules:
    del sys.modules[module_name]
Functions = importlib.import_module(module_name)

from Functions import *

import esm
from Bio import SeqIO
from Bio.Seq import Seq
import pandas as pd 
import numpy as np
import torch

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from Bio import SeqIO

from Bio import Entrez
from Bio import SeqIO




# %% [markdown]
# # Run code

# %%
model, alphabet = esm.pretrained.load_model_and_alphabet('esm2_t36_3B_UR50D')
model.eval()
batch_converter = alphabet.get_batch_converter()
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Transferred model to GPU")

else:
    print("CUDA is not available. Using CPU instead.")
    device = torch.device("cpu")
model =  model.to(device)


# %%
# Entrez.email = "sample@example.org"

# handle = Entrez.efetch(db="nucleotide",
#                        id="NC_045512.2",
#                        rettype="gb",
#                        retmode="gb")
# whole_sequence = SeqIO.read(handle, "genbank")
model_layers = 36

# %%
sequences = read_sequences_to_dict('/home3/oml4h/PLM_SARS-CoV-2/Sequences/huH3N2_HA_CDS.translated_OM_synth_extra_steps.fas')
sequences
base_lineage_index=4

# %%
ids=list(sequences.keys())




# %%
J_indexed_muts = [m for m in get_reference_mutations(ref = sequences[ids[base_lineage_index]],mut = sequences[ids[len(ids)-1]]) if "-" not in m  ] 
K_indexed_muts = [m for m in get_mutations(sequences[ids[0]],sequences[ids[len(ids)-1]]) if "del" not in m and '-' not in m  ] 
print("K->J mutations")
print(K_indexed_muts)

# %%
mutation_dictionary  = {k:J_indexed_muts[i] for i,k in enumerate(K_indexed_muts)}


# %%
all_mutations = {}
Functions.__file__

# %%
# # backbone = "Wuhan-Hu-1"
# backbone = ids[len(ids)-1]
# all_mutations = {}
# mut_info=pd.DataFrame(columns=["Mutation","rel_grammar","rel_seq_grammar",
#                                "narrow_seq_grammar","relative_narrow_seq_grammar","semantic_score","probability","Backbone"])

# reference_spike_sequence = sequences[backbone]
# #loop through each node in the tree except the last one (lineage of interest)
# for backbone_i in range(len(ids)-1):
#     backbone = ids[backbone_i]
#     print("Calculating mutations on backbone:",backbone)
#     reference_spike_sequence = sequences[backbone]
#     #calculate mutation differences between node and K lineage
#     J_indexed_muts = [m for m in get_reference_mutations(ref = sequences[backbone],mut = sequences[ids[len(ids)-1]]) if "-" not in m  ] 
#     K_indexed_muts = [m for m in get_mutations(sequences[backbone],sequences[ids[len(ids)-1]]) if "del" not in m and '-' not in m  ] 
#     mutation_dictionary  = {k:J_indexed_muts[i] for i,k in enumerate(K_indexed_muts)}
#     print("K->J mutations")
#     print(K_indexed_muts)
#     #loop through each mutation and calculate the scores on the focal node
#     for mut in mutation_dictionary.keys():
#         if backbone ==  ids[len(ids)-1]:
#             sequence = revert_sequence(reference_spike_sequence,[mut])
#         else:
#             sequence = mutate_sequence(reference_spike_sequence,[mut])
#         reference = reference_spike_sequence
#         mutations = embed_protein_sequences(
#             [[mut,sequence.replace("-","")]],
#             reference.replace("-",""),
#             'S:0',
#             model,
#             model_layers,
#             device,
#             batch_converter,
#             alphabet,
#             scores=True)
   
   
#         #append mutation info to dataframe
#         mut_info = mut_info.append({"Mutation":mut,"rel_grammar":mutations[mut]["S:0"]["relative_grammaticality"],
#                                     "rel_seq_grammar":mutations[mut]["S:0"]["relative_sequence_grammaticality"],
#                                     "narrow_seq_grammar":mutations[mut]["S:0"]["narrow_sequence_grammaticality"],
#                                     "relative_narrow_seq_grammar":mutations[mut]["S:0"]["relative_narrow_sequence_grammaticality"],
#                                     "semantic_score":mutations[mut]["S:0"]["semantic_score"],
#                                     "probability":mutations[mut]["S:0"]["probability"],
#                                     "Backbone":backbone},ignore_index=True)
    
#         all_mutations[mut] = mutations
    


# %%
# #
# mut_info["lineage_backbone"]=mut_info["Backbone"].str.split("|").str[-1]
# mut_info
# mut_info.to_csv("/home2/oml4h/PLM_SARS-CoV-2/Results/Influenza_H3N2_HA_epistasis_mutation_effects_raw_ESM2.csv",index=False)

# sns.scatterplot(data=mut_info,x="relative_narrow_seq_grammar",y="rel_seq_grammar",hue="lineage_backbone")#,ylabel="Relative Sequence Grammaticality",xlabel="Relative Narrow Sequence Grammaticality",title="Epistatic Effects of HA Mutations on Sequence Grammaticality")
# # change axis label names to be more informative
# plt.xlabel("Relative Narrow Sequence Grammaticality")
# plt.ylabel("Relative Sequence Grammaticality")
# plt.title("Epistatic Effects of HA Mutations on Sequence Grammaticality")
# # plot again with hue by mutation
# plt.show()
# sns.scatterplot(data=mut_info,x="relative_narrow_seq_grammar",y="rel_seq_grammar",hue="Mutation")#,ylabel="Relative Sequence Grammaticality",xlabel="Relative Narrow Sequence Grammaticality",title="Epistatic Effects of HA Mutations on Sequence Grammaticality")
# # print(mut_info.loc[mut_info['Mutation']=='I176K'])
# # print(mut_info.loc[mut_info['Mutation']=='I151K'])

# %% [markdown]
# # do each mutation by itself and then do each pairwise comparison 

# %%
J_indexed_muts = [m for m in get_reference_mutations(ref = sequences[ids[base_lineage_index]],mut = sequences[ids[len(ids)-1]]) if "-" not in m  ] 
K_indexed_muts = [m for m in get_mutations(sequences[ids[base_lineage_index]],sequences[ids[len(ids)-1]]) if "del" not in m and '-' not in m  ] 
print("K->J mutations")
print(K_indexed_muts)



# %%
# backbone = "Wuhan-Hu-1"

backbone=ids[base_lineage_index]
reference_spike_sequence = sequences[backbone]

#create dict of point mutation backbones for all of the mutations in  K_indexed_muts, with names as keys
point_mutations={}
for mut in K_indexed_muts:
    point_mutations[mut]=mutate_sequence(reference_spike_sequence,[mut])

mut_info_rows=[]

#pd.DataFrame(columns=["Mutation","rel_grammar","rel_seq_grammar","semanatic_score","probability","Backbone"])


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
        mutate_sequence(reference_spike_sequence,[mut])
        
        mut=K_indexed_muts[mutation_j]
     

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
                                    "Backbone": backbone_name}
        mut_info_rows.append(new_row)
        
    
mut_info_combos=pd.DataFrame(mut_info_rows)
# %%
# mutation prob matrix
seq_keys=list(sequences.keys())


base_sequence_name=seq_keys[base_lineage_index]
backbone_id=base_sequence_name
base_lineage=base_sequence_name.split("|")[-1]
out="/home3/oml4h/PLM_SARS-CoV-2/Results/test/ESM2_OG"
print(f"Exporting probability matrix for focal lineage: {base_lineage}")

# Get probability matrix for the focal sequence (defined at top)
# Note: We use replace("-", "") to remove gaps as the model expects continuous sequence
prob_data = get_mutation_prob_matrix(
    sequences[base_sequence_name].replace("-", ""),
    model,
    model_layers,
    device,
    batch_converter,
    alphabet
)

matrix = prob_data["mutation_matrix"]  # (20, L)
amino_acids = prob_data["amino_acids"]  # (20,)
sequence_chars = list(prob_data["sequence"])  # (L,)

# Create DataFrame
# Rows: 20 amino acids
# Columns: sequence characters
prob_df = pd.DataFrame(matrix, index=amino_acids, columns=sequence_chars)

# Define output path
output_file = f"{out}/{base_lineage}_probability_matrix.csv"

# Export to CSV
prob_df.to_csv(output_file)

print(f"Probability matrix exported to: {output_file}")
print(f"Matrix shape: {prob_df.shape}")
prob_df.head()







# %%
mut_info_combos["rel_reference_grammar"]=mut_info_combos["rel_grammar"]-mut_info_combos.merge(mut_info_combos[mut_info_combos['Backbone']=="Reference"][['Mutation','rel_grammar']],on='Mutation',how='left')['rel_grammar_y']
#print most shifted mutations compared to reference
mut_info_combos.sort_values(by="rel_reference_grammar",ascending=False).head(10)
mut_info_combos.to_csv("/home2/oml4h/PLM_SARS-CoV-2/Results/H3_epistasis_mutation_info_OG_ESM2_model.csv",index=False)

# %%
epistatic_sum=mut_info_combos.groupby('Backbone')['rel_reference_grammar'].sum().reset_index().sort_values(by='rel_reference_grammar',ascending=False)
print(epistatic_sum.head(20))
sub_mod='OG_ESM'
epistatic_sum.to_csv("/home2/oml4h/PLM_SARS-CoV-2/Results/{}_epistasis_epistatic_sums.csv".format(sub_mod),index=False)

# %%
# plot the same as above but intearcive using plotly and let hover over show mutation names
import plotly.express as px
fig = px.scatter(mut_info, x="relative_narrow_seq_grammar", y="rel_seq_grammar", color="lineage_backbone",
                 hover_data=['Mutation'])
fig.update_layout(title="Epistatic Effects of HA Mutations on Sequence Grammaticality",
                  xaxis_title="Relative Narrow Sequence Grammaticality",
                  yaxis_title="Relative Sequence Grammaticality")

# %%


# %%


# %%


# %%


# %%


# %%


# %%
megatable = []

for k in mutation_dictionary.keys():
    if k == 'Reference':
        continue
    relative_grammaticality = all_mutations[k][k]['S:0']["relative_grammaticality"]
    relative_sequence_grammaticality = all_mutations[k][k]['S:0']["relative_sequence_grammaticality"]
    semantic_score = all_mutations[k][k]['S:0']["semantic_score"]

    mutations = pd.DataFrame({'mutation':k,
                            'relative_grammaticality':relative_grammaticality,
                            'relative_sequence_grammaticality':relative_sequence_grammaticality,
                            'semantic_score':semantic_score,},index=[0])
    mutations.mutations = k
    megatable.append(mutations)
megatable_df = pd.concat(megatable)

if backbone == ids[len(ids)-1]:
    for mut in  mutation_dictionary.keys():
        megatable_df.mutation = megatable_df.mutation.str.replace(mut,mutation_dictionary[mut])
megatable_df

# %%
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler((0,1))

megatable_df['scaled_relative_grammaticality'] = mms.fit_transform(-megatable_df.relative_grammaticality.values.reshape(-1,1)).ravel()
megatable_df['scaled_relative_sequence_grammaticality'] = mms.fit_transform(-megatable_df.relative_sequence_grammaticality.values.reshape(-1,1)).ravel()

# %%
megatable_df.sort_values('relative_grammaticality',ascending=False)

# %%
import torch
lsoftmax = torch.nn.LogSoftmax(dim=1)

# %%
def outside_std(column):
    filtered_vals = []
    for c in column:
        if c >=(column.mean() + (column.std()*2)) or c <= (column.mean() - (column.std()*2)):
            filtered_vals.append(c)
        else:
            filtered_vals.append(np.nan)
    return filtered_vals


# %%
reference_logits_table = []
for k in all_mutations.keys():
    print(all_mutations[k].keys())
    backbone_logits = all_mutations[k]['Reference']['S:0']['Logits']
    backbone_logits = format_logits(backbone_logits,alphabet)
    backbone_logits = remap_logits(backbone_logits,sequences['Wuhan'], reference_spike_sequence)
    backbone_ref_seq_logits = [row[row.Sequence] if row.Sequence != '-' else np.nan for i,row in backbone_logits.iterrows()]

    for mutation in all_mutations[k].keys():
        if mutation != 'Reference':
            if backbone == 'BA.1':
                mutated_sequence = revert_sequence(reference_spike_sequence,[mutation])
            else:
                mutated_sequence = mutate_sequence(reference_spike_sequence,[mutation])

            mutated_logits = format_logits(all_mutations[k][mutation]['S:0']['Logits'],alphabet)
            mutated_logits = remap_logits(mutated_logits,sequences['Wuhan'], mutated_sequence, )
            mutated_seq_logits = [row[row.Sequence]  if row.Sequence != '-' else np.nan for i,row in mutated_logits.iterrows()]
            
            if backbone == 'BA.1':
                # Difference is backbone - mutation here since the "mutation" is actually a reversion
                difference = np.exp(backbone_ref_seq_logits) -  np.exp(mutated_seq_logits)
                pllr =  np.array(backbone_ref_seq_logits)- np.array(mutated_seq_logits)
            else:
                difference =  np.exp(mutated_seq_logits) - np.exp(backbone_ref_seq_logits)
                pllr =  np.array(mutated_seq_logits) - np.array(backbone_ref_seq_logits)

            df = pd.DataFrame(
                {'source_sequence':list(backbone_logits.Sequence),
                'target_sequence':list(mutated_logits.Sequence),
                'source_logits':backbone_ref_seq_logits,
                'target_logits':mutated_seq_logits,
                'change':difference,
                'ratio':pllr,
                'position':range(1,len(difference)+1)})
                
            df['masked_change'] =[row.change if row.source_sequence == row.target_sequence else np.nan for i, row in df.iterrows()]
            df['masked_ratio'] =[row.ratio if row.source_sequence == row.target_sequence else np.nan for i, row in df.iterrows()]
            
            local_max_change = df.masked_change.mean()+(df.masked_change.std()*2)
            local_min_change = df.masked_change.mean()-(df.masked_change.std()*2)

            local_max_ratio = df.masked_ratio.mean()+(df.masked_ratio.std()*2)
            local_min_ratio = df.masked_ratio.mean()-(df.masked_ratio.std()*2)

            df['significant_masked_change_local'] = [ check_valid(m,local_min_change,local_max_change) for m in df. masked_change]
            df['significant_masked_ratio_local'] = [ check_valid(m,local_min_ratio,local_max_ratio) for m in df.masked_ratio]

            df['mutation'] = mutation_dictionary[mutation]
            reference_logits_table.append(df)
reference_logits_table = pd.concat(reference_logits_table,axis=0)

max_change = reference_logits_table.masked_change.mean()+(reference_logits_table.masked_change.std()*2)
min_change = reference_logits_table.masked_change.mean()-(reference_logits_table.masked_change.std()*2)

max_ratio = reference_logits_table.masked_ratio.mean()+(reference_logits_table.masked_ratio.std()*2)
min_ratio = reference_logits_table.masked_ratio.mean()-(reference_logits_table.masked_ratio.std()*2)

reference_logits_table['significant_masked_change'] = [ check_valid(m,min_change,max_change) for m in reference_logits_table.masked_change]
reference_logits_table['significant_masked_ratio'] = [ check_valid(m,min_ratio,max_ratio) for m in reference_logits_table.masked_ratio]

max_change = reference_logits_table.masked_change.round(4).mean()+(reference_logits_table.masked_change.round(4).std()*2)
min_change = reference_logits_table.masked_change.round(4).mean()-(reference_logits_table.masked_change.round(4).std()*2)

reference_logits_table['significant_masked_change_rounded'] = [ check_valid(m,min_change,max_change) for m in reference_logits_table.masked_change.round(4)]


# %%
import plotly.express as px
fig = px.line(reference_logits_table, x="position", y="masked_change",color="mutation", facet_col='mutation',facet_col_wrap=6,height=1000, width=1500, hover_data=['position','change','mutation'])
fig.update_traces(marker={'size': 3})
fig.show()

# %%
import plotly.express as px
fig = px.scatter(reference_logits_table, x="position", y="significant_masked_change_rounded",color="mutation", facet_col='mutation',facet_col_wrap=6,height=1000, width=1500, hover_data=['position','change','mutation'])
fig.update_traces(marker={'size': 3})
fig.show()

# %%
reference_logits_table.to_csv('./Epistasis/Revisions_Omicron_Epistasis.csv',index=False)

# %%
reference_logits_table = pd.read_csv('./Epistasis/Revisions_Omicron_Epistasis.csv')

# %%
max_change = reference_logits_table.masked_change.mean()+(reference_logits_table.masked_change.std())
min_change = reference_logits_table.masked_change.mean()-(reference_logits_table.masked_change.std())
reference_logits_table['single_significant_masked_change'] = [ check_valid(m,min_change,max_change) for m in reference_logits_table.masked_change]

# %%
reference_logits_table.to_csv('Revisions_Omicron_Epistasis_with_single_std.csv')

# %%
import plotly.express as px
fig = px.scatter(reference_logits_table, x="position", y="single_significant_masked_change",color="mutation", facet_col='mutation',facet_col_wrap=6,height=1000, width=1500, hover_data=['position','change','mutation'])
fig.update_traces(marker={'size': 3})
fig.show()

# %%
ba1_mutated_positions = reference_logits_table.mutation.str[1:-1].astype(int).unique()
ba1_mutated_positions

# %%
one_std_identified_interactions = reference_logits_table[(reference_logits_table.single_significant_masked_change.isna() == False) & (reference_logits_table.position.isin(ba1_mutated_positions)) ]

# %%
two_std_identified_interactions = reference_logits_table[(reference_logits_table.significant_masked_change.isna() == False) & (reference_logits_table.position.isin(ba1_mutated_positions)) ]

# %%
px.scatter(two_std_identified_interactions,x=two_std_identified_interactions.position.astype(str),y=two_std_identified_interactions.mutation.astype(str))

# %%
px.scatter(one_std_identified_interactions,x=one_std_identified_interactions.position.astype(str),y=one_std_identified_interactions.mutation.astype(str))

# %%
identified_interactions = one_std_identified_interactions[['position','mutation']]
identified_interactions['mutated_position'] = identified_interactions.mutation.str[1:-1].astype(int)
swap_interactions = identified_interactions[identified_interactions.position < identified_interactions.mutated_position]
swap_interactions.rename(columns={'position':'position_1','mutated_position':'position_2'},inplace=True)
swap_interactions_2 = identified_interactions[identified_interactions.position >= identified_interactions.mutated_position]
swap_interactions_2.rename(columns={'mutated_position':'position_1','position':'position_2'},inplace=True)

swap_interactions = pd.concat([swap_interactions,swap_interactions_2])
swap_interactions.to_csv('Omicron_One_Std_Interactions.csv',index=False)

# %%
identified_interactions = two_std_identified_interactions[['position','mutation']]
identified_interactions['mutated_position'] = identified_interactions.mutation.str[1:-1].astype(int)
swap_interactions = identified_interactions[identified_interactions.position < identified_interactions.mutated_position]
swap_interactions.rename(columns={'position':'position_1','mutated_position':'position_2'},inplace=True)
swap_interactions_2 = identified_interactions[identified_interactions.position >= identified_interactions.mutated_position]
swap_interactions_2.rename(columns={'mutated_position':'position_1','position':'position_2'},inplace=True)

swap_interactions = pd.concat([swap_interactions,swap_interactions_2])
swap_interactions.to_csv('Omicron_Two_Std_Interactions.csv',index=False)

# %%



