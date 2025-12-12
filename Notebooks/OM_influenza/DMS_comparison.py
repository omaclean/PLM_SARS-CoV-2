# %%
import pandas as pd
import seaborn as sns


import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np
import colorcet as cc
import os 

import statsmodels.api as sm
from scipy import stats

import sys
sys.path.append('/home3/oml4h/PLM_SARS-CoV-2/')
from Functions_HuggingFace import *


def get_me_some_colours(n_colours,sns_pal=True):
    if sns_pal:
        if n_colours >8:
            return sns.color_palette(cc.glasbey, n_colours)
        else:
            return sns.color_palette("Dark2", n_colours)
    else:
        #give warning
        print("Warning: returning non-seaborn colour palette, you might just want to add .as_hex() to the end of your returned seaborn colour palette")
        return cc.glasbey[:n_colours]
    
    
def hexbin_plot(data,x_axis,y_axis,log_scale=True,log_constant=1,nbin=10,plot_type="hex"):

    """function to plot a hexbin plot

    :param data: a pandas dataframe of 
    :param log_scale: boolean log scale the data, defaults to True
    :return: a hexbin plot
    """
    ##tests
    assert isinstance(data, pd.DataFrame)
    assert pd.api.types.is_numeric_dtype(data[x_axis])
    assert pd.api.types.is_numeric_dtype(data[y_axis])
    assert len(data) > 1


    # plot stuff


    cmap = plt.cm.viridis
    values = [cmap(i) for i in range(cmap.N)]
    values[0] = (1,1,1,1)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', values, cmap.N)

    if log_scale:
        data[x_axis+"_log10"]=np.log10(data[x_axis]+log_constant)
        x_axis=x_axis+"_log10"
        data[y_axis+"_log10"]=np.log10(data[x_axis]+log_constant)
        x_axis=y_axis+"_log10"
       

    data.dropna(inplace=True)
    

    #plt.hexbin
    g = sns.jointplot(
        y=data[y_axis],
        x=data[x_axis],
        kind="hex",
        #change colour scale
        cmap=cmap,
        hue_norm=(0,0.000001),
            gridsize=(nbin, nbin),
        vmin=0,
     
        joint_kws={'mincnt': 1}
        
            )
    return g

# %%

model_name = "ESM2-H3"
model_name = "ESM2-HA80"
outdir = f"/home3/oml4h/PLM_SARS-CoV-2/Results/DMS_investigation/{model_name}_plots"
os.makedirs(outdir, exist_ok=True)

data_in=pd.read_csv(f"/home3/oml4h/PLM_SARS-CoV-2/Results/DMS_investigation/{model_name}_DMS_with_probabilities_grammar.csv")
data_in=data_in[data_in["wildtype"]!=data_in["mutant"]]
data_in.head(20)


query_path = "/home3/oml4h/PLM_SARS-CoV-2/Sequences/huH3N2_HA_CDS.translated_extra_steps.fas"

reference_path = "/home3/oml4h/PLM_SARS-CoV-2/Sequences/H3N2_canonical.fa"

sequences = read_sequences_to_dict(query_path)

ids=list(sequences.keys())


# 1. Read the reference sequence (Assuming single sequence in file)
# We use 'next' to get the first item from the iterator
ref_record = next(SeqIO.parse(reference_path, "fasta"))
ref_seq_str = str(ref_record.seq)



# 2. Read the query sequences
# We parse the file and pick the first one as a test case
query_iterator = SeqIO.parse(query_path, "fasta")
first_query_record = next(query_iterator)

h3_map_with_ha2 = create_h3_numbering_map(first_query_record, ref_seq_str, HA2_start=330)


K_indexed_muts = [m for m in get_mutations(sequences[ids[0]],sequences[ids[len(ids)-1]]) if "del" not in m and '-' not in m  ] 
# Convert your mutations to canonical numbering
canonical_mutations = mutations_to_canonical(K_indexed_muts, h3_map_with_ha2)
# %%

# scale y axis log

data_in["log10_mutation_probability"] = np.log10(data_in["mutation_probability"])

# 1. Hexbin plot for MDCKSIAT1 cell entry
g = hexbin_plot(data_in,x_axis="log10_mutation_probability",
                y_axis="MDCKSIAT1 cell entry",log_scale=False,plot_type="hex",
                nbin=30)
g.fig.subplots_adjust(top=0.9)
g.ax_marg_x.set_title(f"{model_name} MDCKSIAT1 cell entry vs log10_mutation_probability")
plt.savefig(os.path.join(outdir, f"{model_name}_MDCKSIAT1_cell_entry_vs_log10_mutation_probability_hexbin.png"), dpi=300)
plt.show()

# 2. Hexbin plot for sera escape
g = hexbin_plot(data_in,x_axis="log10_mutation_probability",
                y_axis="sera escape",log_scale=False,plot_type="hex",
                nbin=30)
g.fig.subplots_adjust(top=0.9)
g.ax_marg_x.set_title(f"{model_name} sera escape vs log10_mutation_probability")
plt.savefig(os.path.join(outdir, f"{model_name}_sera_escape_vs_log10_mutation_probability_hexbin.png"), dpi=300)
plt.show()

# 2.1 Hexbin plot for sera escape
g = hexbin_plot(data_in,x_axis="semantic_score",
                y_axis="sera escape",log_scale=False,plot_type="hex",
                nbin=30)
g.fig.subplots_adjust(top=0.9)
g.ax_marg_x.set_title(f"{model_name} sera escape vs semantic_score")
plt.savefig(os.path.join(outdir, f"{model_name}_sera_escape_vs_semantic_score_hexbin.png"), dpi=300)
plt.show()


# 3. Hexbin plot for pH stability
g = hexbin_plot(data_in,x_axis="log10_mutation_probability",
                y_axis="pH stability",log_scale=False,plot_type="hex",
                nbin=30)
g.fig.subplots_adjust(top=0.9)
g.ax_marg_x.set_title(f"{model_name} pH stability vs log10_mutation_probability")
plt.savefig(os.path.join(outdir, f"{model_name}_pH_stability_vs_log10_mutation_probability_hexbin.png"), dpi=300)
plt.show()

# 4. Pairplot for all variables
vars_to_plot = ["MDCKSIAT1 cell entry", "sera escape", "pH stability", "log10_mutation_probability","semantic_score","relative_grammaticality"]
g = sns.PairGrid(data_in[vars_to_plot].dropna())
g.map_diag(sns.histplot)
g.map_offdiag(plt.hexbin, gridsize=30, cmap='viridis', mincnt=1)
g.fig.subplots_adjust(top=0.95)
plt.suptitle(f"{model_name} Pairplot of DMS variables", y=0.98)
plt.savefig(os.path.join(outdir, f"{model_name}_DMS_variables_pairplot.png"), dpi=300)
plt.show()

# 5. Scatter Pairplot
g = sns.PairGrid(data_in[vars_to_plot].dropna())
g.map_diag(sns.histplot, color='darkred')
g.map_offdiag(sns.scatterplot, color='darkred', alpha=0.2, linewidth=0, s=10)
g.fig.subplots_adjust(top=0.95)
plt.suptitle(f"{model_name} Pairplot of DMS variables (Scatter)", y=0.98)
plt.savefig(os.path.join(outdir, f"{model_name}_DMS_variables_pairplot_scatter.png"), dpi=300)
plt.show()

# %%

# create a plot of semantic change vs mutation probability for the DMS data
# highlight the top 5% escape mutations which don't disrupt binding

plt.figure(figsize=(8,6))
sns.scatterplot(data=data_in, x="relative_grammaticality", y="semantic_score", alpha=0.3, edgecolor=None)
# highlight top 5% escape mutations
escape_threshold = data_in["sera escape"].quantile(0.95)
highlight_data = data_in[(data_in["sera escape"] >= escape_threshold) & (data_in["MDCKSIAT1 cell entry"] >= -1)]
sns.scatterplot(data=highlight_data, x="relative_grammaticality", y="semantic_score", color='red', alpha=0.7, edgecolor=None,
                label='Top 5% Escape & not disruptive')  

plt.yscale('log')
plt.xlabel("log10 Relative Grammaticality")
plt.ylabel("Semantic Score")
plt.title("{} Semantic Score vs Relative Grammaticality\n(Highlighting Top 5% Escape Mutations that are not Disruptive)".format(model_name))
plt.legend()
plt.savefig(os.path.join(outdir, 
                         f"{model_name}_semantic_score_vs_mutation_grammaticality_highlight_escape.png"), dpi=300)

# %%

# create a plot of semantic change vs mutation probability for the DMS data
# highlight the top 5% escape mutations which don't disrupt binding

plt.figure(figsize=(8,6))
sns.scatterplot(data=data_in, x="mutation_probability", y="semantic_score", alpha=0.3, edgecolor=None)
# highlight top 5% escape mutations
escape_threshold = data_in["sera escape"].quantile(0.95)
highlight_data = data_in[(data_in["sera escape"] >= escape_threshold) & (data_in["MDCKSIAT1 cell entry"] >= -1)]
sns.scatterplot(data=highlight_data, x="mutation_probability", y="semantic_score", color='red', alpha=0.7, edgecolor=None,
                label='Top 5% Escape & not disruptive')  
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Mutation Probability (log scale)")
plt.ylabel("Semantic Score")
plt.title("{} Semantic Score vs Mutation Probability\n(Highlighting Top 5% Escape Mutations that are not Disruptive)".format(model_name))
plt.legend()
plt.savefig(os.path.join(outdir, 
                         f"{model_name}_semantic_score_vs_mutation_probability_highlight_escape.png"), dpi=300)


# %%
# do stats and see if log (mutation proabbility ) correlates with glm of the three variables:


# Select the columns and remove rows with NaN or inf values
data_clean = data_in[["MDCKSIAT1 cell entry", "sera escape", "pH stability", "mutation_probability"]].copy()
data_clean = data_clean.replace([np.inf, -np.inf], np.nan)  # Replace inf with NaN
data_clean = data_clean.dropna()  # Drop rows with any NaN values
data_clean["absolute_stability"] = np.abs(data_clean["pH stability"])

X = data_clean[["MDCKSIAT1 cell entry", "sera escape", "pH stability"]]

# X = data_clean[["absolute_stability"]]
#X = data_clean[["pH stability"]]
y = np.log10(data_clean["mutation_probability"])

X = sm.add_constant(X)  # Adds a constant term to the predictors
model = sm.OLS(y, X).fit()

predictions = model.predict(X)


# Print the summary
print(model.summary())
print(f"\nNumber of observations used: {len(data_clean)}")
print(f"Number of observations dropped: {len(data_in) - len(data_clean)}")

model_summary = model.summary().as_text()
with open(os.path.join(outdir, f"{model_name}_DMS_glm_model_summary_logprob.txt"), "w") as f:
    f.write(model_summary)
    

# %%
# now with semantic score


# Select the columns and remove rows with NaN or inf values
data_clean = data_in[["MDCKSIAT1 cell entry", "sera escape", "pH stability", "mutation_probability","semantic_score"]].copy()
data_clean = data_clean.replace([np.inf, -np.inf], np.nan)  # Replace inf with NaN
data_clean = data_clean.dropna()  # Drop rows with any NaN values
data_clean["absolute_stability"] = np.abs(data_clean["pH stability"])

X = data_clean[["MDCKSIAT1 cell entry", "sera escape", "pH stability"]]

# X = data_clean[["absolute_stability"]]
#X = data_clean[["pH stability"]]
y = data_clean["semantic_score"]

X = sm.add_constant(X)  # Adds a constant term to the predictors
model = sm.OLS(y, X).fit()

predictions = model.predict(X)


# Print the summary
print(model.summary())
print(f"\nNumber of observations used: {len(data_clean)}")
print(f"Number of observations dropped: {len(data_in) - len(data_clean)}")

model_summary = model.summary().as_text()
with open(os.path.join(outdir, f"{model_name}_DMS_glm_model_summary_semantic_score.txt"), "w") as f:
    f.write(model_summary)
        

# now with absolute stability
X = data_clean[["MDCKSIAT1 cell entry", "sera escape", "absolute_stability"]]
X = sm.add_constant(X)


model = sm.OLS(y, X).fit()

predictions = model.predict(X)


# Print the summary
print(model.summary())
print(f"\nNumber of observations used: {len(data_clean)}")
print(f"Number of observations dropped: {len(data_in) - len(data_clean)}")

model_summary = model.summary().as_text()
with open(os.path.join(outdir, f"{model_name}_DMS_glm_model_summary_abs_ph_logprob.txt"), "w") as f:
    f.write(model_summary)
    
# %%
# now with semantic score

X = data_clean[["MDCKSIAT1 cell entry", "sera escape", "pH stability"]]
# y is already defined as np.log10(data_clean["mutation_probability"])

# Apply Z-score to each column separately using pandas
# (X - X.mean()) / X.std() ensures it is done column by column
X_Zscore = (X - X.mean()) / X.std()

# Add constant to the Z-scored predictors
X_Zscore = sm.add_constant(X_Zscore)

# Fit the model on the Z-scored data (which now includes a constant)
model = sm.OLS(y, X_Zscore).fit()

# Predict using the same Z-scored data
predictions = model.predict(X_Zscore)


# Print the summary
print("Z-scored model summary:")
print(model.summary())
print(f"\nNumber of observations used: {len(data_clean)}")
print(f"Number of observations dropped: {len(data_in) - len(data_clean)}")  



# %%
# make a heatmap (scaling the params or some other kind of plot of all the values in data_in for the mutations listed in canonical_mutations

# Construct mutation string in data_in if needed. 
# Assuming standard format like "A123T"
if 'mutation' not in data_in.columns:
    # This is a guess based on typical DMS data structures. 
    # If 'site' is numeric and 'wildtype'/'mutant' are single letters:
    data_in['mutation'] = data_in['wildtype'] + data_in['site'].astype(str) + data_in['mutant']

# Filter for mutations of interest
# We need to match the format of canonical_mutations.
# If canonical_mutations are like "A123T", we can filter directly.
heatmap_data = data_in[data_in['mutation'].isin(canonical_mutations)].copy()

if heatmap_data.empty:
    print("Warning: No matching mutations found in DMS data for the canonical list.")
    print("Canonical mutations sample:", canonical_mutations[:5])
    print("Data mutations sample:", data_in['mutation'].head().tolist())
else:
    # Set mutation as index for heatmap
    heatmap_data.set_index('mutation', inplace=True)
    
    # Select columns to plot
    cols_to_plot = ["MDCKSIAT1 cell entry", "sera escape", "pH stability", 
                    "log10_mutation_probability", "semantic_score", "relative_grammaticality"]
    
    # Ensure all columns exist
    cols_to_plot = [c for c in cols_to_plot if c in heatmap_data.columns]
    
    # Z-score normalization for heatmap visualization
    # (x - mean) / std
    heatmap_data_norm = (heatmap_data[cols_to_plot] - heatmap_data[cols_to_plot].mean()) / heatmap_data[cols_to_plot].std()
    
    plt.figure(figsize=(10, len(heatmap_data) * 0.5 + 2))
    sns.heatmap(heatmap_data_norm, cmap="coolwarm", center=0, annot=heatmap_data[cols_to_plot], fmt=".2f", cbar=False)
    plt.yticks(rotation=0) # Make y-axis labels horizontal
    plt.title(f"{model_name} Normalized DMS & Model Metrics for Canonical Mutations")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{model_name}_canonical_mutations_heatmap.png"), dpi=300)
    plt.show()
    



# %%
