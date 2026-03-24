# %% 
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import root_scalar, minimize_scalar
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import math

# Global Simulation Parameters
SEASONAL_AMPLITUDE = 0.5
# upping seasonal amplitude https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005844#:~:text=Specific%20humidity%2C%20a%20measure%20of,(4)
# https://www.researchgate.net/figure/Absolute-humidity-relative-humidity-temperature-and-wind-speed-in-the-Toronto-area_fig2_330205997
PEAK_DAY = 200  # Jan 14th approx (relative to July 1st start)
INFECTIOUS_PERIOD = 3.0
LATENT_PERIOD = 2.0
IMPORTATION_RATE = 10.0  # Constant daily importations (year-round)
BETA_K = 0.5 # Higher baseline R0 for K lineage
BETA_CF = 0.5  # Baseline R0 for counterfactual lineage
PLANT_ESCAPE_SLOPE_INIT = 0.02  # Initial guess; calibrated at runtime to reproduce target Re gain
POPULATION=67_000_000  # UK Population Approximation
NAIVE=2_500_000  # Approximate naive population size
# % H3N2 circulation by season (used to scale cohort sizes)

CENTROID_FILE="/home3/oml4h/PLM_SARS-CoV-2/Results/PLANT_results/yearly_centroids.csv"
PLANT_EMBEDDINGS_FILE="/home3/oml4h/PLM_SARS-CoV-2/Results/PLANT_results/PLANT_embeddings_with_distance.csv"
PLANT_OG_BACKGROUND_FILE="/home3/oml4h/hugging_face_downloads/PLANT_model/code/examples/backgrounds.csv"

outdir = "/home3/oml4h/PLM_SARS-CoV-2/Results/sim_results/H3N2_partial_flu"
outdir = f"/home3/oml4h/PLM_SARS-CoV-2/Results/sim_results/H3N2_partial_flu_naive_rest/import_{IMPORTATION_RATE}_seasonal_amp_{SEASONAL_AMPLITUDE}"

outdir = f"/home3/oml4h/PLM_SARS-CoV-2/Results/sim_results/H3N2_partial_flu_smear_rest/import_{IMPORTATION_RATE}_seasonal_amp_{SEASONAL_AMPLITUDE}"

outdir = f"/home3/oml4h/PLM_SARS-CoV-2/Results/sim_results_new_cf/H3N2_partial_flu_smear_rest/import_{IMPORTATION_RATE}_seasonal_amp_{SEASONAL_AMPLITUDE}"
outdir = f"/home3/oml4h/PLM_SARS-CoV-2/Results/sim_results_new_cf/H3N2_partial_flu_smear_rest_J.2.4cf/import_{IMPORTATION_RATE}_seasonal_amp_{SEASONAL_AMPLITUDE}"

Saved 3D antigenic map: /home3/oml4h/PLM_SARS-CoV-2/Results/sim_results_new_cf/H3N2_partial_flu_smear_rest_J.2.4cf/import_10.0_seasonal_amp_0.5/flu_antigenic_map_highlighted.html
Background year filter <= 2024: kept 151,785/151,785 points
Saved 3D antigenic map: /home3/oml4h/PLM_SARS-CoV-2/Results/sim_results_new_cf/H3N2_partial_flu_smear_rest_J.2.4cf/import_10.0_seasonal_amp_0.5/flu_antigenic_map_highlighted_og_background.html
Distance from Vaccine to K Lineage: 0.97 units
Distance from K lineage to 2014: 8.02 units
Distance from K lineage to 2015: 8.25 units
Distance from K lineage to 2016: 8.00 units
Distance from K lineage to 2017: 7.48 units
Distance from K lineage to 2018: 6.71 units
Distance from K lineage to 2019: 6.00 units
Distance from K lineage to 2020: 5.29 units
Distance from K lineage to 2021: 2.90 units
Distance from K lineage to 2022: 2.52 units
Distance from K lineage to 2023: 1.54 units
Distance from K lineage to 2024: 1.20 units

--- CALIBRATION ---
Target Re gain per unit distance : 0.0147
Calibrated escape slope          : 0.053144
Antigenic distance K ↔ CF        : 0.649 units
Achieved Re ratio S_eff_K/S_eff_CF : 1.009544  (per-unit gain = 0.0147)

#%% 
class AntigenicSeirModel:
    # ... [Previous Init and Susceptibility methods remain unchanged] ...
    
    def __init__(self, historical_strains, population_size=67_000_000, current_year=2025.5,
                 escape_slope=PLANT_ESCAPE_SLOPE_INIT):
        self.history = historical_strains
        self.pop_size = population_size
        self.current_year = current_year
        self.escape_slope = escape_slope
        self.epochs = sorted(list(self.history.keys()))

    # [Include _sigmoid_susceptibility and _linear_susceptibility here]
    def _linear_susceptibility(self, distance, base_susceptibility=0.3, scaling_factor=None):
        if scaling_factor is None:
            scaling_factor = self.escape_slope
        return min(1.0, base_susceptibility + (scaling_factor * distance))

    def calculate_susceptibility(self, current_strain_coord, vaccine_coord=None):
        curr_coord = np.array(current_strain_coord)
        sigmas = [1.0] # Naive
        for year in self.epochs:
            hist_coord = np.array(self.history[year])
            dist = np.linalg.norm(curr_coord - hist_coord)
            years_elapsed = max(0, self.current_year - year)
            susc = self._linear_susceptibility(dist)
            total_susc = 1.0 - ((1.0 - susc) * np.exp(-0.16 * years_elapsed))
            sigmas.append(total_susc)
            
        if vaccine_coord is not None:
             v_dist = np.linalg.norm(np.array(vaccine_coord) - curr_coord)
             susc_v = self._linear_susceptibility(v_dist)
             # CHANGED: Static waning removed. Only the baseline is appended.
             sigmas.append(susc_v)
        else:
            sigmas.append(1.0)
        return np.array(sigmas)

    def deriv(self, y, t, base_beta, base_sigma_vector, latent_period, infectious_period, 
              seasonal_amplitude, peak_day, importation_rate):
        
        # 1. Seasonal Beta Forcing
        forcing = 1 + seasonal_amplitude * np.cos(2 * np.pi * (t - peak_day) / 365.0)
        beta_t = base_beta * forcing
        
        # 2. Dynamic Vaccine Waning (Ray et al. 2019)
        sigma_t = np.array(base_sigma_vector, dtype=float)
        vacc_idx = -1 # Vaccine is the last cohort
        vax_day = 92.0 # Approx 1st October (assuming 1st July start)
        
        # 16% increase in susceptibility per 28 days
        waning_rate_per_day = np.log(1.16) / 28.0 
        
        if t >= vax_day:
            days_post_vax = t - vax_day
            sigma_t[vacc_idx] = min(1.0, sigma_t[vacc_idx] * np.exp(waning_rate_per_day * days_post_vax))
        else:
            # To avoid a summer peak of unvaccinated individuals, we assume residual 
            # cross-protection from the previous year holds them at a moderate baseline
            # until the new vaccine is administered.
            sigma_t[vacc_idx] = min(1.0, sigma_t[vacc_idx] * 1.5) 
            
        num_compartments = len(sigma_t)
        S_cohorts = y[:num_compartments]
        E = y[num_compartments]
        I = y[num_compartments + 1]
        
        # Standard Force of Infection
        lam = beta_t * I / self.pop_size
        
        dS_dt = -lam * sigma_t * S_cohorts
        new_infections = np.sum(lam * sigma_t * S_cohorts)
        
        dE_dt = new_infections - (1/latent_period) * E + importation_rate
        dI_dt = (1/latent_period) * E - (1/infectious_period) * I
        dR_dt = (1/infectious_period) * I
        
        return np.concatenate([dS_dt, [dE_dt, dI_dt, dR_dt]])

    def run(self, strain_coord, vaccine_coord, pop_distribution, 
            beta=0.6, 
            seasonal_amplitude=0.4,  # INCREASED DEFAULT (Was 0.2)
            seed_infections=0,       # CHANGED: Default to 0
            days=365):
        
        sigmas = self.calculate_susceptibility(strain_coord, vaccine_coord)
        
        # Initial Conditions
        S_init = np.array(pop_distribution, dtype=float)
        # Note: We do NOT subtract seed_infections here anymore if seed is 0
        if seed_infections > 0:
             S_init[0] -= seed_infections
        
        E_init = 0.0
        I_init = float(seed_infections) # Can start at 0 now!
        R_init = 0.0
        
        y0 = np.concatenate([S_init, [E_init, I_init, R_init]])
        
        t = np.linspace(0, days, days*4)
        
        # Pass new args to odeint
        ret = odeint(self.deriv, y0, t, args=(beta, sigmas, LATENT_PERIOD, INFECTIOUS_PERIOD, 
                              seasonal_amplitude, PEAK_DAY, IMPORTATION_RATE))
        
        cols = ['S_Naive'] + [f'S_{yr}' for yr in self.epochs] + ['S_Vacc', 'E', 'I', 'R']
        df = pd.DataFrame(ret, columns=cols)
        df['time'] = t
        return df, sigmas


def calibrate_escape_slope(model, pop_dist, cf_coord, k_coord, vacc_coord,
                           target_re_increase=0.0147):
    """
    Solve for the PLANT_ESCAPE_SLOPE that yields a target per-unit-distance
    relative Re increase between the counterfactual and the K lineage.

    Parameters
    ----------
    model : AntigenicSeirModel
        Model instance (its escape_slope will be updated in-place on success).
    pop_dist : list[int]
        Population distribution [Naive, ..., Vacc].
    cf_coord, k_coord, vacc_coord : tuple
        PLANT coordinates for counterfactual, K lineage, and vaccine strain.
    target_re_increase : float
        Desired *per-unit-distance* relative Re gain, i.e.
        (S_eff_K / S_eff_CF − 1) / ‖K − CF‖ = target_re_increase.

    Returns
    -------
    float
        Calibrated escape slope.
    """
    dist_diff = np.linalg.norm(np.array(k_coord) - np.array(cf_coord))
    target_ratio = 1.0 + (target_re_increase * dist_diff)

    pop = np.array(pop_dist, dtype=float)

    def objective(test_slope):
        model.escape_slope = test_slope

        sigmas_cf = model.calculate_susceptibility(cf_coord, vacc_coord)
        sigmas_k  = model.calculate_susceptibility(k_coord,  vacc_coord)

        S_eff_cf = np.sum(pop * sigmas_cf)
        S_eff_k  = np.sum(pop * sigmas_k)

        return (S_eff_k / S_eff_cf) - target_ratio

    # Try a small bracket first
    a, b = 1e-6, 0.5
    fa = objective(a)
    fb = objective(b)

    if fa * fb < 0:
        result = root_scalar(objective, bracket=[a, b], method='brentq')
        if not result.converged:
            raise ValueError("Could not converge on a valid escape slope.")
        model.escape_slope = result.root
        return result.root

    # If no sign change in the simple bracket, search a wider range for a sign change
    slopes = np.linspace(1e-8, 5.0, 500)
    fvals = [objective(s) for s in slopes]
    bracket_found = None
    for i in range(len(fvals) - 1):
        if fvals[i] == 0:
            bracket_found = (slopes[i], slopes[i])
            break
        if fvals[i] * fvals[i + 1] < 0:
            bracket_found = (slopes[i], slopes[i + 1])
            break

    if bracket_found is not None:
        a, b = bracket_found
        # If a==b then we've hit an exact zero in the sample
        if a == b:
            model.escape_slope = a
            return a
        result = root_scalar(objective, bracket=[a, b], method='brentq')
        if result.converged:
            model.escape_slope = result.root
            return result.root

    # Fallback: minimize the absolute objective over a wide bound and return the best-fit slope
    res = minimize_scalar(lambda s: abs(objective(s)), bounds=(1e-8, 5.0), method='bounded')
    if not res.success:
        raise ValueError(f"Could not find escape slope (fa={fa:.3e}, fb={fb:.3e}).")
    model.escape_slope = res.x
    print(f"Warning: No sign change found; using minimize_scalar fallback, slope={res.x:.6e}, objective={objective(res.x):.3e}")
    return res.x


def plot_antigenic_map_with_highlights(
    history,
    k_coord,
    cf_coord,
    vacc_coord,
    output_html_path,
    background_embeddings_csv=PLANT_EMBEDDINGS_FILE,
    max_background_year=None,
):
    """Create a 3D antigenic map with muted background and highlighted key points."""
    fig = go.Figure()

    # Background: all PLANT embeddings as small, muted dots
    if os.path.exists(background_embeddings_csv):
        bg_df = pd.read_csv(background_embeddings_csv)
        required_cols = {"X", "Y", "Z"}
        if required_cols.issubset(bg_df.columns):
            if max_background_year is not None:
                year_series = None
                if "year" in bg_df.columns:
                    year_series = pd.to_numeric(bg_df["year"], errors="coerce")
                elif "collection date" in bg_df.columns:
                    year_series = pd.to_numeric(
                        bg_df["collection date"].astype(str).str.extract(r"(\d{4})", expand=False),
                        errors="coerce",
                    )
                elif "collection_date" in bg_df.columns:
                    year_series = pd.to_numeric(
                        bg_df["collection_date"].astype(str).str.extract(r"(\d{4})", expand=False),
                        errors="coerce",
                    )

                if year_series is not None:
                    before_n = len(bg_df)
                    bg_df = bg_df.loc[year_series <= max_background_year].copy()
                    print(
                        f"Background year filter <= {max_background_year}: kept {len(bg_df):,}/{before_n:,} points"
                    )

            fig.add_trace(
                go.Scatter3d(
                    x=bg_df["X"],
                    y=bg_df["Y"],
                    z=bg_df["Z"],
                    mode="markers",
                    marker=dict(size=2, color="#c7c7c7", opacity=0.08),
                    name="Background (all PLANT points)",
                    hoverinfo="skip",
                    showlegend=True,
                )
            )
        else:
            print(
                f"Warning: {background_embeddings_csv} missing X/Y/Z columns; background points skipped."
            )
    else:
        print(
            f"Warning: background embeddings file not found at {background_embeddings_csv}; background points skipped."
        )

    # Historical centroids (highlighted, but less dominant than vaccine/K/CF)
    history_years = sorted(history.keys())
    history_xyz = np.array([history[y] for y in history_years], dtype=float)
    fig.add_trace(
        go.Scatter3d(
            x=history_xyz[:, 0],
            y=history_xyz[:, 1],
            z=history_xyz[:, 2],
            mode="markers+text",
            marker=dict(size=8, color="#4d4d4d", opacity=1.0, symbol="diamond"),
            text=[str(y) for y in history_years],
            textposition="top center",
            name="Historic centroids",
            hovertemplate="year: %{text}<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>",
            showlegend=True,
        )
    )

    # Key highlighted points (large, dark, obvious)
    key_points = [
        ("Vaccine", vacc_coord, "#006400", 15),
        ("K lineage", k_coord, "#8B0000", 15),
        (f"Counterfactual \n {cf_name}", cf_coord, "#00008B", 15),
    ]
    shared_label_font = dict(size=14, color="#5DA2C2")
    for name, coord, color, size in key_points:
        text_position = "bottom center" if name == "Counterfactual" else "top center"
        fig.add_trace(
            go.Scatter3d(
                x=[coord[0]],
                y=[coord[1]],
                z=[coord[2]],
                mode="markers+text",
                marker=dict(size=size, color=color, opacity=1.0, symbol="circle"),
                text=[name],
                textposition=text_position,
                textfont=shared_label_font,
                name=name,
                hovertemplate=f"{name}<br>X: %{{x:.3f}}<br>Y: %{{y:.3f}}<br>Z: %{{z:.3f}}<extra></extra>",
                showlegend=True,
            )
        )

    fig.update_layout(
        title="Antigenic Map: Highlighted Vaccine, K, Counterfactual, and Historic Centroids",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
        legend=dict(itemsizing="constant"),
        width=1400,
        height=1100,
    )
    fig.write_html(output_html_path)
    print(f"Saved 3D antigenic map: {output_html_path}")


# ==========================================
# 1. CONFIGURATION & DATA
# ==========================================

# Output directory

os.makedirs(outdir, exist_ok=True)
# %% 
# Historic Centroids (real PLANT output)
history = {
    
    2014: (0.63602, -3.79551, -0.88322),
    2015: (1.43803, -4.35738, -0.52841),
    2016: (2.03978, -4.16784, -0.91984),
    2017: (2.22809, -3.68523, -0.79706),
    2018: (2.48353, -2.99210, -0.45817),
    2019: (2.49067, -2.0354, -1.3839),
    2020: (2.91758, -1.31161, -1.46984),
    2021: (3.42502, 2.18306, -2.05458),
    2022: (3.47224, 2.19141, -1.61720),
    2023: (3.29408, 2.84147, -0.84808),
    2024: (3.19627, 3.24367, -0.61533) 
    
}



#go through each year and print the distances
for year in sorted(history.keys()):
    if year == sorted(history.keys())[0]:
        continue
    dist = np.linalg.norm(np.array(history[year]) - np.array(history[year-1]))
    print(f"Distance from {year-1} to {year}: {dist:.2f} units")
# %% 
# Population Distribution [Naive, 2019...2024, Vacc]
# Each year cohort is scaled by %H3N2 circulation for that season.
#need to sanity check this infl A and B as demoninator
H3N2_FRACTION_BY_YEAR = {
    2019: 0.30, #https://archive.cdc.gov/www_cdc_gov/flu/about/burden/2019-2020/archive-09292021.html#:~:text=seasonal%20influenza%20vaccination.-,2019%E2%80%932020%20Burden%20Estimates,by%20A(H1N1)pdm09%20viruses
    2020: 0.30,
    2021: 0.80, # https://www.gov.uk/government/statistics/annual-flu-reports/surveillance-of-influenza-and-other-seasonal-respiratory-viruses-in-winter-2021-to-2022 figure 15, online analysis
    # of infl A it's 270/(44.7+270)=85.7%, infl A =(44.7+270+635)/(44.7+270+635+69.5)=93%=80%
    2022: 0.28, #https://www.ecdc.europa.eu/en/publications-data/seasonal-influenza-annual-epidemiological-report-20222023
    2023: 0.30, #https://www.rivm.nl/en/flu-and-flu-vaccine/facts-and-figures/annual-reporting-surveillance-2023-2024
    2024: 0.45, #https://www.rivm.nl/en/flu-and-flu-vaccine/facts-and-figures/annual-reporting-surveillance-2024-2025#:~:text=In%20all%20age%20groups%2C%20influenza,%2C%20Meijer%20et%20al.).
}
#citation for these estimates? what about people who get infected in multiple years?
BASE_COHORT_COUNTS = {
    2019: 12_000_000,
    2020: 1_200_000,
    2021: 6_000_000,
    2022: 12_000_000, #12 million each? https://www.sciencedirect.com/science/article/pii/S2213260014700347?via%3Dihub 
    2023: 12_000_000,
    2024: 12_000_000,
    "Vacc": 15_000_000, #https://assets.publishing.service.gov.uk/media/663e246cae748c43d3793925/AB-24-026_Immunisation_Schedule.pdf
}
# 2. Identify which years are explicit (2019-2024) and which are imputed (2014-2018)
history_years = sorted(history.keys())
explicit_years_int = sorted(y for y in BASE_COHORT_COUNTS.keys() if isinstance(y, int))
imputed_years = [y for y in history_years if y not in explicit_years_int]

# 3. Calculate the size of the "Residual" (Smear) population
# Start with Total
residual_pop = POPULATION - NAIVE - BASE_COHORT_COUNTS["Vacc"]

# Subtract the known recent cohorts
current_explicit_sum = 0
for y in explicit_years_int:
    # Safe logic: if data exists, use it, otherwise 0
    fraction = H3N2_FRACTION_BY_YEAR.get(y, 0)
    base = BASE_COHORT_COUNTS.get(y, 0)
    count = int(base * fraction)
    current_explicit_sum += count

residual_pop -= current_explicit_sum

print(f"Residual Population to Smear (2014-{max(imputed_years)}): {residual_pop:,.0f}")

# 4. Distribute the Residual Population
# We divide the residual evenly across the imputed years (2014-2018)
if imputed_years and residual_pop > 0:
    pop_per_year = int(residual_pop / len(imputed_years))
    remainder = residual_pop % len(imputed_years)
else:
    pop_per_year = 0
    remainder = 0
    if residual_pop > 0:
        print("WARNING: No imputed years found to hold residual population. Dumping into oldest explicit year.")
        # Fallback logic would go here, but with your data, imputed_years will be valid.

# 5. Construct the Final Distribution List
cohort_counts = []
for year in history_years:
    if year in imputed_years:
        # Give them the smear share
        val = pop_per_year
        # Add the tiny remainder to the last imputed year to keep sums exact
        if year == imputed_years[-1]:
            val += remainder
        cohort_counts.append(val)
    else:
        # Use the explicit data
        fraction = H3N2_FRACTION_BY_YEAR.get(year, 0)
        base = BASE_COHORT_COUNTS.get(year, 0)
        cohort_counts.append(int(base * fraction))

# Final check
pop_dist = [NAIVE] + cohort_counts + [BASE_COHORT_COUNTS["Vacc"]]

assert sum(pop_dist) == POPULATION, f"Population mismatch! Sum: {sum(pop_dist):,.0f} vs Target: {POPULATION:,.0f}"

model = AntigenicSeirModel(history, population_size=67_000_000, current_year=2025.5)

# %%


model = AntigenicSeirModel(history, population_size=67_000_000, current_year=2025.5)

# ==========================================
# 2. SCENARIO RUNS
# ==========================================
# %% 

# A. Actual Scenario: K Lineage (High Drift)
# [cite_start]Distance from 2024 (3.2, 2.2) is ~2.2 units -> Crosses the 2.0 threshold [cite: 147]
# name,year,subclade,X,Y,Z,Distance_to_Ref
# EPI4748783|HA|A/England/01837755/2025|EPI_ISL_20210731|J.2.4.1,2025,K,3.4980469,3.5703125,0.4946289,0.96760744
# EPI3791586|HA|A/England/2024|J.2.2Eng24,2024,J.2.2Eng24,3.046875,3.5292969,-0.5493164,0.22035405
# EPI4908143|HA|A/Columbia_vaccine/2023|J.2,2023,J.2.vaccine_column,3.0117188,3.59375,-0.34155273,0.0



# croatia onw 
#counterfactual coords of the Eng 24 virus circulating in 2024


antigenic_map_output_path = os.path.join(outdir, "flu_antigenic_map_highlighted.html")
plot_antigenic_map_with_highlights(
    history=history,
    k_coord=k_coord,
    cf_coord=cf_coord,
    vacc_coord=vacc_coord,
    output_html_path=antigenic_map_output_path,
)

antigenic_map_og_background_output_path = os.path.join(outdir, "flu_antigenic_map_highlighted_og_background.html")
plot_antigenic_map_with_highlights(
    history=history,
    k_coord=k_coord,
    cf_coord=cf_coord,
    vacc_coord=vacc_coord,
    output_html_path=antigenic_map_og_background_output_path,
    background_embeddings_csv=PLANT_OG_BACKGROUND_FILE,
    max_background_year=max(history.keys()),
)

#print distance from vaccine to K
v_dist = np.linalg.norm(np.array(k_coord) - np.array(vacc_coord))
print(f"Distance from Vaccine to K Lineage: {v_dist:.2f} units")

for year in sorted(history.keys()):

    dist = np.linalg.norm(np.array(history[year]) - np.array(k_coord ))
    print(f"Distance from K lineage to {year}: {dist:.2f} units")

# B. Counterfactual: No Mutation I160K
# Distance from 2024 is ~1.0 unit -> Within 2.0 threshold (protected)

# ---- Calibrate the escape slope so 1.47 %/unit Re gain is emergent ----
TARGET_RE_GAIN_PER_UNIT = 0.0147   # target: 1.47 % Re increase per PLANT-distance unit
calibrated_slope = calibrate_escape_slope(
    model, pop_dist, cf_coord, k_coord, vacc_coord,
    target_re_increase=TARGET_RE_GAIN_PER_UNIT,
)
dist_kc = np.linalg.norm(np.array(k_coord) - np.array(cf_coord))
print(f"\n--- CALIBRATION ---")
print(f"Target Re gain per unit distance : {TARGET_RE_GAIN_PER_UNIT:.4f}")
print(f"Calibrated escape slope          : {calibrated_slope:.6f}")
print(f"Antigenic distance K ↔ CF        : {dist_kc:.3f} units")
# Verify
_sig_cf = model.calculate_susceptibility(cf_coord, vacc_coord)
_sig_k  = model.calculate_susceptibility(k_coord,  vacc_coord)
_ratio  = np.sum(np.array(pop_dist) * _sig_k) / np.sum(np.array(pop_dist) * _sig_cf)
print(f"Achieved Re ratio S_eff_K/S_eff_CF : {_ratio:.6f}  "
      f"(per-unit gain = {(_ratio - 1) / dist_kc:.4f})")

# %% 
# Vaccine mismatched


res_k, sigmas_k = model.run(k_coord, vacc_coord, pop_dist, beta=BETA_K)

res_cf, sigmas_cf = model.run(cf_coord, vacc_coord, pop_dist, beta=BETA_CF)

# ==========================================
# 3. ANALYSIS & VALIDATION
# ==========================================

peak_k = res_k['I'].max()
peak_cf = res_cf['I'].max()
total_cases_k = res_k['R'].iloc[-1]
total_cases_cf = res_cf['R'].iloc[-1]

print(f"--- EPIDEMIC SUMMARY ---")
print(f"Scenario K (Actual) Peak: {peak_k:,.0f}")
print(f"Scenario CF (No Mut) Peak: {peak_cf:,.0f}")
print(f"Attributable to Mutation: {total_cases_k - total_cases_cf:,.0f} additional cases")

# [cite_start]Check alignment with Paper's Fitness Scalar [cite: 533]
# Paper: 1 unit distance ~= 1.47% fitness advantage (approx)
# We calculate effective susceptible pool S_eff for both
S_eff_k = np.sum(np.array(pop_dist) * sigmas_k)
S_eff_cf = np.sum(np.array(pop_dist) * sigmas_cf)
Re_increase = (S_eff_k / S_eff_cf) - 1.0
dist_diff = np.linalg.norm(np.array(k_coord) - np.array(cf_coord))

print(f"\n--- CALIBRATION CHECK ---")
print(f"Antigenic Shift: {dist_diff:.2f} units")
print(f"Re Increase: {Re_increase:.2%}")
print(f"Implied fitness gain per unit: {(Re_increase/dist_diff):.2%} (Target: ~1.47%)")
print("Note: If Implied > Target, increase sigmoid midpoint or decrease steepness.")

# ==========================================
# 4. PLOTTING
# ==========================================

# Helper function to calculate Rt (effective reproduction number) over time
def calculate_Rt_over_time(df, beta, infectious_period, base_sigma_vector, pop_size, 
                           seasonal_amplitude=SEASONAL_AMPLITUDE, peak_day=PEAK_DAY):
    times = df['time'].values
    R0_vals = []
    
    waning_rate_per_day = np.log(1.16) / 28.0
    vax_day = 92.0
    vacc_idx = -1
    
    for t in times:
        forcing = 1 + seasonal_amplitude * np.cos(2 * np.pi * (t - peak_day) / 365.0)
        beta_t = beta * forcing
        
        # Recreate dynamic sigma for this timestep
        sigma_t = np.array(base_sigma_vector, dtype=float)
        if t >= vax_day:
            sigma_t[vacc_idx] = min(1.0, sigma_t[vacc_idx] * np.exp(waning_rate_per_day * (t - vax_day)))
        else:
            sigma_t[vacc_idx] = min(1.0, sigma_t[vacc_idx] * 1.5)
            
        S_cohorts = df.iloc[int(t*4) if int(t*4) < len(df) else -1][
            [c for c in df.columns if c.startswith('S_')]
        ].values
        
        avg_susceptibility = np.sum(S_cohorts * sigma_t) / np.sum(S_cohorts)
        Rt = beta_t * infectious_period * avg_susceptibility
        R0_vals.append(Rt)
    
    return np.array(R0_vals)

# Helper function to convert day to date string
def day_to_date(day, start_month=7, start_day=1):
    """Convert simulation day to calendar date (assuming start July 1)"""
    from datetime import datetime, timedelta
    start = datetime(2025, start_month, start_day)
    current = start + timedelta(days=int(day))
    return current

# Calculate Rt (time-varying reproduction number) for both scenarios
Rt_k = calculate_Rt_over_time(res_k, BETA_K, infectious_period=INFECTIOUS_PERIOD, 
                               base_sigma_vector=sigmas_k, pop_size=model.pop_size)
Rt_cf = calculate_Rt_over_time(res_cf, BETA_CF, infectious_period=INFECTIOUS_PERIOD, 
                                base_sigma_vector=sigmas_cf, pop_size=model.pop_size)

# Create comprehensive figure
fig = plt.figure(figsize=(18, 14))

# ============ Plot 1: Full SEIR Dynamics over Time ============
ax1 = plt.subplot(4, 2, 1)
ax1.plot(res_k['time'], res_k[['S_' + c for c in ['Naive'] + [str(y) for y in model.epochs] + ['Vacc']]].sum(axis=1) / 1e6, 
         label='Susceptible', color='#1f77b4', lw=2)
ax1.plot(res_k['time'], res_k['E'] / 1e6, label='Exposed', color='#ff7f0e', lw=2)
ax1.plot(res_k['time'], res_k['I'] / 1e6, label='Infected', color='#d62728', lw=2)
ax1.plot(res_k['time'], res_k['R'] / 1e6, label='Recovered', color='#2ca02c', lw=2)
ax1.set_title('SEIR Dynamics - Actual K Lineage', fontsize=12, fontweight='bold')
ax1.set_xlabel('Days')
ax1.set_ylabel('Population (Millions)')
ax1.legend(loc='right')
ax1.grid(alpha=0.3)

# ============ Plot 2: SEIR Counterfactual ============
ax2 = plt.subplot(4, 2, 2)
ax2.plot(res_cf['time'], res_cf[['S_' + c for c in ['Naive'] + [str(y) for y in model.epochs] + ['Vacc']]].sum(axis=1) / 1e6, 
         label='Susceptible', color='#1f77b4', lw=2, alpha=0.7)
ax2.plot(res_cf['time'], res_cf['E'] / 1e6, label='Exposed', color='#ff7f0e', lw=2, alpha=0.7)
ax2.plot(res_cf['time'], res_cf['I'] / 1e6, label='Infected', color='#d62728', lw=2, alpha=0.7)
ax2.plot(res_cf['time'], res_cf['R'] / 1e6, label='Recovered', color='#2ca02c', lw=2, alpha=0.7)
ax2.set_title('SEIR Dynamics - Counterfactual (No Mutation)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Days')
ax2.set_ylabel('Population (Millions)')
ax2.legend(loc='right')
ax2.grid(alpha=0.3)

# ============ Plot 3: Rt with Seasonality Overlay ============
ax3 = plt.subplot(4, 2, 3)
# Calculate baseline seasonality (no depletion)
times = res_k['time'].values
seasonal_forcing = 1 + SEASONAL_AMPLITUDE * np.cos(2 * np.pi * (times - PEAK_DAY) / 365.0)

R0_t_k = BETA_K * INFECTIOUS_PERIOD * seasonal_forcing
R0_t_cf = BETA_CF * INFECTIOUS_PERIOD * seasonal_forcing

# Seasonality forcing on secondary axis - shows multiplicative factor


# Plot R0(t) (same for both) and Rt (different for both)
ax3.plot(times, R0_t_k, 'k--', lw=1.5, alpha=0.5, label='R₀(t) (K baseline)', zorder=5)
ax3.plot(times, R0_t_cf, 'k-.', lw=1.2, alpha=0.5, label='R₀(t) (CF baseline)', zorder=5)

ax3.fill_between(times, 0, R0_t_k, where=(seasonal_forcing > 1.0), 
                    alpha=0.10, color='skyblue', label='Winter boost (>1.0)')
ax3.fill_between(times, 0, R0_t_k, where=(seasonal_forcing <= 1.0), 
                    alpha=0.10, color='orange', label='Summer reduction (<1.0)')

ax3.plot(times, Rt_k, label='Rₜ (K lineage)', color='#d62728', lw=2.5)
ax3.plot(times, Rt_cf, label='Rₜ (Counterfactual)', color='#1f77b4', lw=2.5, ls='--')

ax3.set_title('R₀(t) vs Rₜ: Seasonality vs Susceptibility Depletion', fontsize=11, fontweight='bold')
ax3.set_xlabel('Days')
ax3.set_ylabel('Reproduction Number (Left Axis)', fontsize=10)

ax3.plot(times, seasonal_forcing, 'k:', alpha=0.3, lw=1, label='Seasonal Forcing')
ax3.axhline(y=1.0, color='gray', linestyle='-', lw=0.5, alpha=0.3)
ax3.set_ylabel('Seasonal Multiplier (Right Axis)', fontsize=10, color='gray')
ax3.tick_params(axis='y', labelcolor='gray')


ax3.legend(loc='upper left', fontsize=8)
ax3.grid(alpha=0.3)
# Don't set ylim minimum - let it drop below 1 naturally
ax3.set_ylim([0, max(Rt_k.max(), Rt_cf.max(), R0_t_k.max(), R0_t_cf.max()) * 1.15])

# ============ Plot 4: R₀ → R₀(t) → Rₜ Decomposition ============
ax4 = plt.subplot(4, 2, 4)
# Base R₀ (intrinsic, no seasonality) - same for both scenarios
base_R0_k = np.ones_like(times) * BETA_K * INFECTIOUS_PERIOD
base_R0_cf = np.ones_like(times) * BETA_CF * INFECTIOUS_PERIOD
R0_t_k_line = base_R0_k * seasonal_forcing
R0_t_cf_line = base_R0_cf * seasonal_forcing

ax4.fill_between(times, 0, base_R0_k, alpha=0.08, color='gray', label='Base R₀ (K)')
ax4.fill_between(times, 0, base_R0_cf, alpha=0.08, color='lightgray', label='Base R₀ (CF)')
ax4.fill_between(times, 0, R0_t_k_line, alpha=0.12, color='purple', label='R₀(t) (K)')
ax4.fill_between(times, 0, R0_t_cf_line, alpha=0.12, color='violet', label='R₀(t) (CF)')
ax4.plot(times, Rt_k, color='#d62728', lw=2.5, label='Rₜ (K lineage)', zorder=10)
ax4.plot(times, Rt_cf, color='#1f77b4', lw=2.5, ls='--', label=f'Rₜ (Counterfactual {cf_name})', zorder=10)

ax4.set_title('From R₀ to Rₜ: Decomposition', fontsize=11, fontweight='bold')
ax4.set_xlabel('Days')
ax4.set_ylabel('Reproduction Number')
ax4.legend(loc='best', fontsize=8)
ax4.grid(alpha=0.3)
ax4.text(0.02, 0.98, 'Key: R₀(t) differs by lineage baseline.\nRₜ further differs due to immune escape.', 
         transform=ax4.transAxes, fontsize=8, va='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# ============ Plot 5: Julian Calendar View ============
ax5 = plt.subplot(4, 2, 5)
# Convert days to dates
dates = [day_to_date(d) for d in times]
# julian_days = [(d - datetime(d.year, 1, 1).replace(tzinfo=None)).days + 1 for d in dates]

# Define flu season regions (Northern Hemisphere)
# Typical flu season: October (day 92 from July 1) to May (day 334 from July 1)
ax5.axvspan(92, 334, alpha=0.15, color='blue', label='Typical Flu Season (Oct-May)')
ax5.axvline(x=PEAK_DAY, color='purple', linestyle='--', alpha=0.5, label='Peak Seasonality (Jan 20)')

# Plot infections
ax5.plot(times, res_k['I'] / 1e6, label='K Lineage Infections', 
         color='#d62728', lw=2.5)
ax5.plot(times, res_cf['I'] / 1e6, label=f'Counterfactual {cf_name} Infections', 
         color='#1f77b4', lw=2.5, ls='--')

ax5.set_title('Epidemic Curve on Calendar', fontsize=12, fontweight='bold')
ax5.set_xlabel('Days from July 1st')
ax5.set_ylabel('Infected (Millions)')
ax5.legend(loc='best', fontsize=8)
ax5.grid(alpha=0.3)
ax5.set_xlim([min(times), max(times)])

# Add month labels
# Month starts relative to July 1st (approx)
month_starts = [0, 31, 62, 92, 123, 153, 184, 215, 243, 274, 304, 335]
month_names = ['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
ax5_top = ax5.twiny()
ax5_top.set_xlim(ax5.get_xlim())
visible_months = [i for i, day in enumerate(month_starts) if min(times) <= day <= max(times)]
ax5_top.set_xticks([month_starts[i] for i in visible_months])
ax5_top.set_xticklabels([month_names[i] for i in visible_months])

# ============ Plot 6: Susceptibility Profile ============
ax6 = plt.subplot(4, 2, 6)
cohort_labels = (
    ['Naive']
    + [
        f"{y} (H3N2 {int(H3N2_FRACTION_BY_YEAR.get(y, 0)*100)}%)"
        for y in model.epochs
    ]
    + ['Vacc']
)
x = np.arange(len(cohort_labels))
width = 0.35

ax6.bar(x - width/2, sigmas_k, width, label='Actual K', color='#d62728', alpha=0.7)
ax6.bar(x + width/2, sigmas_cf, width, label='Counterfactual '+cf_name, color='#1f77b4', alpha=0.7)
ax6.set_xticks(x)
ax6.set_xticklabels(cohort_labels, rotation=45)
ax6.set_title('Susceptibility by Immunity Cohort', fontsize=12, fontweight='bold')
ax6.set_ylabel('Susceptibility (σ)')
ax6.legend()
ax6.grid(alpha=0.3, axis='y')

# ============ Plot 7: Epidemic Curve Comparison ============
ax7 = plt.subplot(4, 2, 7)
ax7.fill_between(res_k['time'], res_k['I'] / 1e6, alpha=0.3, color='#d62728', 
                 label='K Lineage (Actual)')
ax7.fill_between(res_cf['time'], res_cf['I'] / 1e6, alpha=0.3, color='#1f77b4',
                 label='No Mutation (CF)')
ax7.plot(res_k['time'], res_k['I'] / 1e6, color='#d62728', lw=2)
ax7.plot(res_cf['time'], res_cf['I'] / 1e6, color='#1f77b4', lw=2, ls='--')
ax7.set_title('Infection Prevalence Comparison', fontsize=12, fontweight='bold')
ax7.set_xlabel('Days')
ax7.set_ylabel('Infected (Millions)')
ax7.legend()
ax7.grid(alpha=0.3)

# Add annotations for key metrics - position away from title
peak_k_idx = res_k['I'].argmax()
peak_cf_idx = res_cf['I'].argmax()
peak_k_time = res_k.iloc[peak_k_idx]['time']
peak_k_val = peak_k/1e6

# Position annotation in middle-right area to avoid title
ax7.annotate(f'K Peak:\n{peak_k_val:.2f}M\n(Day {peak_k_time:.0f})', 
            xy=(peak_k_time, peak_k_val),
            xytext=(peak_k_time + 30, peak_k_val * 0.7),
            bbox=dict(boxstyle='round,pad=0.5', fc='#d62728', alpha=0.5, ec='#d62728'),
            arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5),
            fontsize=9, fontweight='bold')

# ============ Plot 8: Effective Susceptible Population Over Time ============
ax8 = plt.subplot(4, 2, 8)
# Calculate effective susceptible population
S_cols = [c for c in res_k.columns if c.startswith('S_')]
S_eff_k = []
S_eff_cf = []
for idx in range(len(res_k)):
    S_cohorts_k = res_k.iloc[idx][S_cols].values
    S_cohorts_cf = res_cf.iloc[idx][S_cols].values
    S_eff_k.append(np.sum(S_cohorts_k * sigmas_k) / 1e6)
    S_eff_cf.append(np.sum(S_cohorts_cf * sigmas_cf) / 1e6)

ax8.fill_between(res_k['time'], S_eff_k, alpha=0.3, color='#d62728', label='K Lineage')
ax8.fill_between(res_cf['time'], S_eff_cf, alpha=0.3, color='#1f77b4', label=f'Counterfactual {cf_name}')
ax8.plot(res_k['time'], S_eff_k, color='#d62728', lw=2)
ax8.plot(res_cf['time'], S_eff_cf, color='#1f77b4', lw=2, ls='--')
ax8.set_title('Effective Susceptible Population (Why Rₜ Differs)', fontsize=12, fontweight='bold')
ax8.set_xlabel('Days')
ax8.set_ylabel('Effective Susceptibles (Millions)')
ax8.legend()
ax8.grid(alpha=0.3)
ax8.text(0.02, 0.98, f'Initial S_eff:\nK: {S_eff_k[0]:.1f}M\nCF: {S_eff_cf[0]:.1f}M\nRatio: {S_eff_k[0]/S_eff_cf[0]:.2f}x', 
         transform=ax8.transAxes, fontsize=9, va='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()


regime=f"R0_shift_k_{R0_t_k.mean():.2f}_vs_{R0_t_cf.mean():.2f}_escape_param={model.escape_slope:.6f}"
output_path = os.path.join(outdir, f'flu_model_comprehensive_analysis_{regime}.png')

plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("COMPREHENSIVE PLOTS GENERATED")
print("="*60)
print(f"Figure saved as: {output_path}")
print(f"\nPlot 1-2: Full SEIR dynamics for both scenarios")
print(f"Plot 3: R₀(t) [SAME] vs Rₜ [DIFFERENT] over time")
print(f"         R₀(t) varies with seasonality (both scenarios identical)")
print(f"         Rₜ differs due to immune escape affecting susceptibility")
print(f"Plot 4: R₀ → R₀(t) → Rₜ decomposition")
print(f"         (Clarifies: intrinsic transmission SAME, outcome differs)")
print(f"Plot 5: Epidemic curve on Julian calendar with flu season shading")
print(f"Plot 6: Susceptibility profiles by cohort")
print(f"Plot 7: Direct infection comparison with peak annotations")
print(f"Plot 8: Effective susceptible population over time")
print(f"         (K has {S_eff_k[0]/S_eff_cf[0]:.2f}x more susceptibles → higher Rₜ)")
print("="*60)

# %% 

# Tune Beta using the Counterfactual (Baseline) Scenario
target_peak = 3_000_000  # Aim for a moderate flu season
best_beta = 0
min_error = float('inf')

print("Calibrating Beta...")
for b in np.linspace(0.5, 3.0, 20):  # Test betas from 0.5 to 3.0
    res, _ = model.run(cf_coord, vacc_coord, pop_dist, beta=b)
    peak = res['I'].max()
    
    if peak > 100:  # If epidemic takes off
        print(f"Beta: {b:.2f} -> Peak: {peak:,.0f}")
        error = abs(peak - target_peak)
        if error < min_error:
            min_error = error
            best_beta = b

print(f"\nRecommended Beta: {best_beta:.2f}")

#Vaccine Waning Still Simplified (model.test.py)

# Hard-coded 0.5 years is better than before, but should ideally be a parameter
# Beta Same for Both Scenarios (model.test.py, model.test.py)

# Both use beta=0.55 - this is actually correct if you're modeling fitness through susceptibility alone, but the comment at line 115 clarifies this intentional choice
# Calibration Check is Post-Hoc (model.test.py)

# %%

