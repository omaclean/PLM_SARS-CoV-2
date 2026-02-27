#!/usr/bin/env python3
# %% [markdown]
# # Interactive Exploration: Epistasis vs Diversity Shifts (J.2 vs K)
#
# This script consolidates the analysis of epistatic shifts against changes in circulating diversity.
# It uses the `# %%` cell format so it can be run interactively in modern IDEs (VS Code, Spyder, PyCharm).
#
# Datasets merged here:
# 1. **Epistatic Shifts:** The predicted pairwise shifts (`Combined` and mutation-specific shifts) between J.2 and K.
# 2. **Observed Diversity:** The differential circulating diversity profiles (`H3N2_max10_unique`) between J.2 and K.
# 3. **PLM Probabilities:** The full 20xN mutational probability matrices for both J.2 and K backgrounds.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, ttest_ind, mannwhitneyu
from pathlib import Path

# Configure seaborn styling
sns.set_theme(style="whitegrid")
sns.set_context("talk", font_scale=0.8)

# %% [markdown]
# ### 1. Configure Data Paths
# Ensure these point to the exact outputs from your ESM2 runs and mutational accessibility pipeline.

# %%
# 1. Epistasis shift file (J.2 vs K)
EPISTASIS_PATH = "/home3/oml4h/PLM_SARS-CoV-2/Results/test/J.2_int/ESM2-H3_mutation_probability_shifts_J.2_int.csv"

# 2. Observed diversity profiles (from the Mutational_accessibility.py pipeline)
# NOTE: Update these tags to use "_hard_nextle2_max10" if you have generated the hard-filtered datasets.
# Currently searching for the hard filtered files returned no results, so falling back to 'max10_unique'
FOCAL_DIV_PATH = "/home3/oml4h/PLM_SARS-CoV-2/Results/test/lineage_panel_mutability_vs_plm/gisaidinc/OG_ESM2_t36_3B/J.2_observed_diversity_profile_full_H3N2_max10_unique.csv"
COMP_DIV_PATH = "/home3/oml4h/PLM_SARS-CoV-2/Results/test/lineage_panel_mutability_vs_plm/gisaidinc/OG_ESM2_t36_3B/K_observed_diversity_profile_full_H3N2_max10_unique.csv"

# 3. Full 20xN Probability Matrices
FOCAL_PROB_PATH = "/home3/oml4h/PLM_SARS-CoV-2/Results/test/lineage_panel_mutability_vs_plm/gisaidinc/OG_ESM2_t36_3B/J.2_plm_probability_profile_full_H3N2_max10_unique.csv"
COMP_PROB_PATH = "/home3/oml4h/PLM_SARS-CoV-2/Results/test/lineage_panel_mutability_vs_plm/gisaidinc/OG_ESM2_t36_3B/K_plm_probability_profile_full_H3N2_max10_unique.csv"

OUTDIR = Path("/home3/oml4h/PLM_SARS-CoV-2/Results/test/epistasis_differential_exploration")
OUTDIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ### 2. Load and Preprocess Data Matrices
# Loads 20xN diversity and probability profiles and standardizes headers.

# %%
def load_matrix(path, is_diversity=False):
    df = pd.read_csv(path)
    if df.columns[0] in ["Unnamed: 0", "AA", ""]:
        df = df.rename(columns={df.columns[0]: "AA"})
    df = df.set_index("AA")
    
    # Standardize column headers to int (Position)
    valid_cols = []
    for c in df.columns:
        try:
            valid_cols.append(int(c))
        except ValueError:
            pass
            
    df.columns = [int(c) if str(c).isdigit() else c for c in df.columns]
    
    # Select only integer columns for the spatial matrix 
    numeric_cols = [c for c in df.columns if isinstance(c, int)]
    return df[numeric_cols]

# Load 20xN dataframes
focal_div = load_matrix(FOCAL_DIV_PATH, True)
comp_div = load_matrix(COMP_DIV_PATH, True)

focal_prob = load_matrix(FOCAL_PROB_PATH)
comp_prob = load_matrix(COMP_PROB_PATH)

# Load Epistasis shifts (Site Level)
epi_df = pd.read_csv(EPISTASIS_PATH)
if "Position" not in epi_df.columns:
    epi_df = epi_df.rename(columns={epi_df.columns[0]: "Position"})

# Ensure Combined column exists
if "Combined" not in epi_df.columns:
    numeric = epi_df.select_dtypes(include=["number"]).drop(columns=["Position"], errors="ignore")
    epi_df["Combined"] = numeric.abs().mean(axis=1)
    
epi_df = epi_df.set_index("Position")
print(f"Loaded epistasis metrics for {len(epi_df)} positions.")

# %% [markdown]
# ### 3. Site-Level Diversity Shift Analysis
# Mimics the statistical test: Does high epistatic shift at a site mean high diversity shift?

# %%
def calculate_positional_diversity(div_df):
    """ Calculate total alternate (non-consensus) frequency per site. """
    res = {}
    for pos in div_df.columns:
        col = div_df[pos].dropna()
        if len(col) == 0:
            continue
        max_freq = col.max()
        total_alt_freq = col[col < max_freq].sum()
        res[pos] = {"total_alt_freq": total_alt_freq}
    return pd.DataFrame.from_dict(res, orient="index")

focal_pos_div = calculate_positional_diversity(focal_div).rename(columns={"total_alt_freq": "focal_total_alt"})
comp_pos_div = calculate_positional_diversity(comp_div).rename(columns={"total_alt_freq": "comp_total_alt"})

# Merge site level data
site_df = epi_df[["Combined"]].copy()
site_df = site_df.join(focal_pos_div).join(comp_pos_div).dropna()

# Shift delta
site_df["delta_total_alt"] = site_df["focal_total_alt"] - site_df["comp_total_alt"]
site_df["abs_delta_total_alt"] = site_df["delta_total_alt"].abs()

# Mann Whitney U Test
median_shift = site_df["Combined"].median()
high_shift = site_df[site_df["Combined"] > median_shift]
low_shift = site_df[site_df["Combined"] <= median_shift]
stat_u, p_u = mannwhitneyu(high_shift["abs_delta_total_alt"], low_shift["abs_delta_total_alt"])
site_df["Shift_Group"] = np.where(site_df["Combined"] > median_shift, "High Epistasis", "Low Epistasis")

# Plot Boxplot
plt.figure(figsize=(6, 6))
sns.boxplot(data=site_df, x="Shift_Group", y="abs_delta_total_alt", hue="Shift_Group", legend=False, palette="muted")
sns.stripplot(data=site_df, x="Shift_Group", y="abs_delta_total_alt", color=".25", alpha=0.5, size=4)
plt.title(f"Diversity Shift by Epistatic Shift Group\nMann-Whitney U p={p_u:.3e}")
plt.ylabel("Absolute Shift in Total Alternate Frequency")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4. Flattening to Mutation-Level Resolution
# Instead of aggregating by site, convert 20xN matrices to long tables where each row is a specific amino acid substitution.

# %%
def flatten_matrix(df, value_name):
    df_flat = df.reset_index().melt(id_vars="AA", var_name="Position", value_name=value_name)
    return df_flat

f_div_flat = flatten_matrix(focal_div, "focal_obs_freq")
c_div_flat = flatten_matrix(comp_div, "comp_obs_freq")
f_prob_flat = flatten_matrix(focal_prob, "focal_plm_prob")
c_prob_flat = flatten_matrix(comp_prob, "comp_plm_prob")

# Merge on Position and Amino Acid
mut_df = f_div_flat.merge(c_div_flat, on=["Position", "AA"], how="outer")\
                   .merge(f_prob_flat, on=["Position", "AA"], how="outer")\
                   .merge(c_prob_flat, on=["Position", "AA"], how="outer")

# Differentials
mut_df["delta_obs_freq"] = mut_df["focal_obs_freq"].fillna(0) - mut_df["comp_obs_freq"].fillna(0)
mut_df["abs_delta_obs_freq"] = mut_df["delta_obs_freq"].abs()

mut_df["delta_plm_prob"] = mut_df["focal_plm_prob"] - mut_df["comp_plm_prob"]
mut_df["abs_delta_plm_prob"] = mut_df["delta_plm_prob"].abs()

# Add the site-level 'Combined' epistatic shift to each mutation row
mut_df = mut_df.merge(epi_df[["Combined"]], left_on="Position", right_index=True, how="left")
mut_df = mut_df.dropna(subset=["focal_plm_prob", "comp_plm_prob"])

print(f"Created mutation-level dataframe with {len(mut_df)} possible substitutions.")

# %% [markdown]
# ### 5. Mutation-Level Correlation: Predicted vs Observed Divergence
# Test if specific substitutions with a large predicted epistatic shift correspond to those with large variations in circulating diversity.

# %%
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Identify consensus amino acids (AAs with highest observed frequency at each position)
focal_consensus = focal_div.idxmax()
comp_consensus = comp_div.idxmax()

def is_consensus(row):
    pos = row["Position"]
    aa = row["AA"]
    return (aa == focal_consensus.get(pos)) or (aa == comp_consensus.get(pos))

mut_df["is_consensus"] = mut_df.apply(is_consensus, axis=1)

# Remove trivial non-mutations and exclude fixed/consensus amino acids
mask_active = (mut_df["focal_obs_freq"] > 0) | (mut_df["comp_obs_freq"] > 0)
mask_not_consensus = ~mut_df["is_consensus"]
active_muts = mut_df[mask_active & mask_not_consensus].copy()

r_spear, p_spear = spearmanr(active_muts["abs_delta_plm_prob"], active_muts["abs_delta_obs_freq"])

# Linear Scatter
sns.scatterplot(data=active_muts, x="abs_delta_plm_prob", 
    y="abs_delta_obs_freq", alpha=0.4, edgecolor=None, ax=ax[0])
ax[0].set_title(f"Mutation-Level Matrix Correlation\nSpearman ρ = {r_spear:.3f} (p={p_spear:.2e})")
ax[0].set_xlabel("Absolute Predicted PLM Shift (|Focal - K|)")
ax[0].set_ylabel("Absolute Observed Freq Shift (|Focal - K|)")

# Log-Log Scatter
sns.scatterplot(
    x=np.log10(active_muts["abs_delta_plm_prob"].clip(lower=1e-6)), 
    y=np.log10(active_muts["abs_delta_obs_freq"].clip(lower=1e-6)), 
    alpha=0.4, edgecolor=None, ax=ax[1]
)
ax[1].set_title("Log10 Scale Representation")
ax[1].set_xlabel("Log10(Absolute PLM Shift)")
ax[1].set_ylabel("Log10(Absolute Observed Freq Shift)")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 6. Extract Top 20 High-Deviation Mutations
# Which specific mutations does the model think are undergoing the most epistatic pressure between K and J.2?

# %%
top_predicted_shifts = mut_df.sort_values("abs_delta_plm_prob", ascending=False).head(20)
print("\nTop 20 Substitutions by PLM Epistatic Shift:")
print(top_predicted_shifts[["Position", "AA", "delta_plm_prob", "delta_obs_freq", "Combined"]].to_string(index=False))

top_predicted_shifts.to_csv(OUTDIR / "top_20_epistatic_substitutions.csv", index=False)

# %% [markdown]
# ### 7. Interactive 20xN Heatmap
# Find the cluster of 40 contiguous positions with the highest overall epistatic shift and visualize the full 20xN delta matrix (Focal - K).

# %%
window_size = 40
rolling_sum = epi_df["Combined"].rolling(window=window_size).sum()
max_window_end = rolling_sum.idxmax()

if pd.isna(max_window_end):
    pos_start = epi_df.index.min()
    pos_end = pos_start + window_size
else:
    pos_start = max_window_end - window_size + 1
    pos_end = max_window_end

print(f"\nVisualizing highly epistatic region: sequence positions {int(pos_start)} to {int(pos_end)}")

# Create the 20xN delta probability matrix
delta_matrix = focal_prob - comp_prob
cols_in_region = [c for c in delta_matrix.columns if pos_start <= c <= pos_end]
region_delta = delta_matrix[cols_in_region]

plt.figure(figsize=(18, 6))
# symmetric colormap centered at 0 to show shifts in both directions
sns.heatmap(region_delta, cmap="vlag", center=0, cbar_kws={'label': 'Δ PLM Probability (J.2 - K)'})
plt.title(f"Mutation Probability Matrix Shifts (J.2 vs K) for positions {int(pos_start)}-{int(pos_end)}")
plt.xlabel("Sequence Position")
plt.ylabel("Amino Acid")
plt.show()

# Save final flattened dataset for downstream analysis
mut_df.to_csv(OUTDIR / "flattened_epistasis_diversity_mutations.csv", index=False)
print(f"Data and results saved to {OUTDIR}")

# %%
