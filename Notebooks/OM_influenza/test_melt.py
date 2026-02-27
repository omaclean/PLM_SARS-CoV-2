import pandas as pd
df = pd.DataFrame({
    "Unnamed: 0": ["A", "C", "D"],
    "1": [0.1, 0.0, 0.2],
    "2": [0.0, 0.5, 0.0]
})
df2 = df.copy()
id_col = df2.index.name or df2.columns[0]
dfm = df2.reset_index().melt(id_vars=id_col, var_name="Position", value_name="obs_freq")
try:
    dfm["Position"] = dfm["Position"].astype(int)
except Exception as e:
    print("Failed with:", type(e).__name__, e)
else:
    print("Success")
