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
#model_name = "ESM2-HA80"
outdir = f"/home3/oml4h/PLM_SARS-CoV-2/Results/DMS_investigation/{model_name}_plots"
os.makedirs(outdir, exist_ok=True)

data_in=pd.read_csv(f"/home3/oml4h/PLM_SARS-CoV-2/Results/DMS_investigation/{model_name}_DMS_with_probabilities_grammar.csv")
data_in=data_in[data_in["wildtype"]!=data_in["mutant"]]
data_in.head(20)

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

# 3. Hexbin plot for pH stability
g = hexbin_plot(data_in,x_axis="log10_mutation_probability",
                y_axis="pH stability",log_scale=False,plot_type="hex",
                nbin=30)
g.fig.subplots_adjust(top=0.9)
g.ax_marg_x.set_title(f"{model_name} pH stability vs log10_mutation_probability")
plt.savefig(os.path.join(outdir, f"{model_name}_pH_stability_vs_log10_mutation_probability_hexbin.png"), dpi=300)
plt.show()

# 4. Pairplot for all variables
vars_to_plot = ["MDCKSIAT1 cell entry", "sera escape", "pH stability", "log10_mutation_probability"]
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
# do stats and see if log (mutation proabbility ) correlates with glm of the three variables:

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




