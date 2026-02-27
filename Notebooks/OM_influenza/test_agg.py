import pandas as pd
import numpy as np
df = pd.DataFrame({
    "lineage": ["A", "A", "B"],
    "Position": [1, 1, 2],
    "obs_freq": [np.nan, np.nan, 0.5],
    "obs_present": [np.nan, np.nan, 1.0]
})
agg = df.groupby(["lineage", "Position"]).agg(
    max_obs_freq=("obs_freq", "max"),
    alt_observed_count=("obs_present", "sum")
).reset_index()
print(agg)
