#%% 
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import math

# Global Simulation Parameters
SEASONAL_AMPLITUDE = 0.2
PEAK_DAY = 204  # Jan 20th approx (relative to July 1st start)
INFECTIOUS_PERIOD = 3.0
LATENT_PERIOD = 2.0

class AntigenicSeirModel:
    """
    A stratified SEIR model where susceptibility is a function of:
    1. Antigenic Distance (Euclidean distance in PLANT space)
    2. Immunological Waning (Time since last infection)
    
    References:
    - PLANT Paper Fig 1G/6B: 2 antigenic units is the standard threshold for drift/mismatch.
    - PLANT Paper Fig 4G: ~1.47% fitness advantage per antigenic unit (used for calibration checks).
    """
    
    def __init__(self, historical_strains, population_size=67_000_000, current_year=2025.5):
        """
        Args:
            historical_strains (dict): Key=Year (int/float), Value=tuple (x, y, z)
            population_size (int): Total N.
            current_year (float): The current decimal year for waning calculations.
        """
        self.history = historical_strains
        self.pop_size = population_size
        self.current_year = current_year
        self.epochs = sorted(list(self.history.keys()))
        
        # Validation
        for year, coords in self.history.items():
            if len(coords) != 3:
                raise ValueError(f"Coordinate for {year} must be 3D (x, y, z).")

    def _sigmoid_susceptibility(self, distance, midpoint=2.0, k=1.5):
        """
        Sigmoid function converting antigenic distance to susceptibility.
        
        Params:
            [cite_start]midpoint (2.0): The '2 antigenic unit' threshold[cite: 147].
                            Below 2 units = high cross-protection. 
                            Above 2 units = escape.
            k (1.5): Steepness. Tunable.
        """
        return 1 / (1 + np.exp(-k * (distance - midpoint)))
    
    def _linear_susceptibility(self, distance, base_susceptibility=0.3, scaling_factor=0.0147):
        """
        Calculates susceptibility based on the linear relationship found in the PLANT paper.
        
        Args:
            distance (float): Euclidean distance in PLANT space.
            base_susceptibility (float): Susceptibility to a homologous strain (0 distance).
                                         This captures that even with 0 distance, immunity isn't 100% 
                                         perfect forever (or acts as a baseline R0 scaler).
            scaling_factor (float): The paper's coefficient (1.47% per unit).
                                    Default = 0.0147.
        """
        # Linear increase: Base + (Slope * Distance)
        # We clip it at 1.0 to prevent probability > 100%
        return min(1.0, base_susceptibility + (scaling_factor * distance))

    def calculate_susceptibility(self, current_strain_coord, vaccine_coord=None, 
                                 waning_rate_natural=0.05, waning_rate_vaccine=0.15):
        """
        Calculates susceptibility vector [Naive, Hist_1, ..., Hist_N, Vaccinated]
        
        Formula:
        Susceptibility = 1 - (Protection_Antigenic * Protection_Temporal)
        """
        curr_coord = np.array(current_strain_coord)
        sigmas = []

        # 0. Naive Cohort (Always 100% susceptible)
        sigmas.append(1.0)

        # 1. Historical Cohorts
        for year in self.epochs:
            hist_coord = np.array(self.history[year])
            
            # Antigenic Protection (1 - susceptibility based on distance)
            dist = np.linalg.norm(curr_coord - hist_coord)
            susc_antigenic = self._linear_susceptibility(dist)
            prot_antigenic = 1.0 - susc_antigenic
            
            # Temporal Protection (Exponential decay of immunity)
            years_elapsed = max(0, self.current_year - year)
            prot_temporal = np.exp(-waning_rate_natural * years_elapsed)
            
            # Combined Susceptibility
            total_susc = 1.0 - (prot_antigenic * prot_temporal)
            sigmas.append(total_susc)

            

        # 2. Vaccinated Cohort
        if vaccine_coord is not None:
            # Distance from vaccine strain to circulating strain
            v_dist = np.linalg.norm(np.array(vaccine_coord) - curr_coord)
            susc_antigenic_v = self._linear_susceptibility(v_dist)
            prot_antigenic_v = 1.0 - susc_antigenic_v
            
            # Vaccine waning (assumed faster than natural infection)
            # Assuming average 0.5 years since vaccination for the cohort
            prot_temporal_v = np.exp(-waning_rate_vaccine * 0.5) 
            
            total_susc_v = 1.0 - (prot_antigenic_v * prot_temporal_v)
            sigmas.append(total_susc_v)
        else:
            sigmas.append(1.0) # If no vaccine logic, treat as naive or exclude

        return np.array(sigmas)

    def deriv(self, y, t, base_beta, sigma_vector, latent_period, infectious_period, 
              seasonal_amplitude=SEASONAL_AMPLITUDE, peak_day=PEAK_DAY):
        """
        Updated with SEASONAL FORCING.
        
        Args:
            seasonal_amplitude (0.2): Variance in R0 due to weather (20% swing).
            peak_day (20): The day of peak transmissibility (e.g., Jan 15th).
                           If simulation starts Nov 1, Jan 15 is ~Day 75.
                           Adjust this relative to your start date.
        """
        # 1. Calculate Seasonal Beta
        # Cosine wave: +20% in winter, -20% in summer
        # Assuming t is days. Period is 365.
        forcing = 1 + seasonal_amplitude * np.cos(2 * np.pi * (t - peak_day) / 365.0)
        
        beta_t = base_beta * forcing
        
        # ... rest of SEIR logic is identical ...
        num_compartments = len(sigma_vector)
        S_cohorts = y[:num_compartments]
        E = y[num_compartments]
        I = y[num_compartments + 1]
        
        lam = beta_t * I / self.pop_size
        
        dS_dt = -lam * sigma_vector * S_cohorts
        new_infections = np.sum(lam * sigma_vector * S_cohorts)
        
        dE_dt = new_infections - (1/latent_period) * E
        dI_dt = (1/latent_period) * E - (1/infectious_period) * I
        dR_dt = (1/infectious_period) * I
        
        return np.concatenate([dS_dt, [dE_dt, dI_dt, dR_dt]])
    
    def run(self, strain_coord, vaccine_coord, pop_distribution, 
            beta=0.6, latent_period=LATENT_PERIOD, infectious_period=INFECTIOUS_PERIOD, 
            seed_infections=100, days=365):
        """
        Run simulation.
        pop_distribution: list [Naive, Hist_1..., Vacc] (Must match history + 2)
        """
        # 1. Validation
        expected_len = 1 + len(self.epochs) + 1 # Naive + History + Vacc
        if len(pop_distribution) != expected_len:
            raise ValueError(f"Pop distribution len {len(pop_distribution)} != expected {expected_len} (Naive + {len(self.epochs)} epochs + Vacc)")
        
        if abs(sum(pop_distribution) - self.pop_size) > 1000:
            print(f"Warning: Sum of compartments ({sum(pop_distribution):,.0f}) != Pop Size ({self.pop_size:,.0f})")

        # 2. Susceptibility
        sigmas = self.calculate_susceptibility(strain_coord, vaccine_coord)
        
        # 3. Initial Conditions
        # Subtract seed infections from Naive pool to conserve N
        S_init = np.array(pop_distribution, dtype=float)
        if S_init[0] > seed_infections:
            S_init[0] -= seed_infections
        
        E_init = 0.0
        I_init = float(seed_infections)
        R_init = 0.0
        
        y0 = np.concatenate([S_init, [E_init, I_init, R_init]])
        
        # 4. Integrate
        t = np.linspace(0, days, days*4) # 4 steps per day for stability
        ret = odeint(self.deriv, y0, t, args=(beta, sigmas, latent_period, infectious_period))
        
        # 5. Package
        cols = ['S_Naive'] + [f'S_{yr}' for yr in self.epochs] + ['S_Vacc', 'E', 'I', 'R']
        df = pd.DataFrame(ret, columns=cols)
        df['time'] = t
        
        # Downsample back to daily for reporting if desired, or keep high res
        return df, sigmas

# ==========================================
# 1. CONFIGURATION & DATA
# ==========================================

# Output directory
outdir = "/home3/oml4h/PLM_SARS-CoV-2/Results/sim_results"
os.makedirs(outdir, exist_ok=True)
# %% 
# Historic Centroids (Mock PLANT output)
history = {
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
# Sum = 67M. Added a 5M "Naive" pool (young children/never infected).
pop_dist = [
    5_000_000,   # Naive
    12_000_000,  # Last inf 2019
    2_000_000,   # Last inf 2020
    5_000_000,   # Last inf 2021
    8_000_000,   # Last inf 2022
    10_000_000,  # Last inf 2023
    10_000_000,  # Last inf 2024
    15_000_000   # Current Season Vaccinated
]

model = AntigenicSeirModel(history, population_size=67_000_000, current_year=2025.5)

# ==========================================
# 2. SCENARIO RUNS
# ==========================================
# %% 
output_path = os.path.join(outdir, 'flu_model_comprehensive_analysis.png')
# A. Actual Scenario: K Lineage (High Drift)
# [cite_start]Distance from 2024 (3.2, 2.2) is ~2.2 units -> Crosses the 2.0 threshold [cite: 147]
k_coord = (3.498047, 3.57003, 0.4946) 

vacc_coord = (3.011719, 3.59375, -0.34155) 

#print distance from vaccine to K
v_dist = np.linalg.norm(np.array(k_coord) - np.array(vacc_coord))
print(f"Distance from Vaccine to K Lineage: {v_dist:.2f} units")

for year in sorted(history.keys()):

    dist = np.linalg.norm(np.array(history[year]) - np.array(k_coord ))
    print(f"Distance from K lineage to {year}: {dist:.2f} units")
# %% 
# Vaccine mismatched

beta=1.16 # Calibrated to match observed growth rates

res_k, sigmas_k = model.run(k_coord, vacc_coord, pop_dist, beta=beta)

# B. Counterfactual: No Mutation I160K
# Distance from 2024 is ~1.0 unit -> Within 2.0 threshold (protected)
cf_coord = (3.011719, 3.59375, -0.34155) # (3.693,3.424,-0.073) # Approx PLANT coord on just first branch with I160K only

res_cf, sigmas_cf = model.run(cf_coord, vacc_coord, pop_dist, beta=beta)

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
def calculate_Rt_over_time(df, beta, infectious_period, sigma_vector, pop_size, 
                           seasonal_amplitude=SEASONAL_AMPLITUDE, peak_day=PEAK_DAY):
    """Calculate Rt (time-varying reproduction number) at each timepoint accounting for seasonality and susceptibility depletion"""
    times = df['time'].values
    R0_vals = []
    
    for t in times:
        # Calculate seasonal forcing
        forcing = 1 + seasonal_amplitude * np.cos(2 * np.pi * (t - peak_day) / 365.0)
        beta_t = beta * forcing
        
        # Calculate average susceptibility weighted by susceptible population
        S_cohorts = df.iloc[int(t*4) if int(t*4) < len(df) else -1][
            [c for c in df.columns if c.startswith('S_')]
        ].values
        avg_susceptibility = np.sum(S_cohorts * sigma_vector) / np.sum(S_cohorts)
        
        # Rt = beta(t) * infectious_period * avg_susceptibility
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
Rt_k = calculate_Rt_over_time(res_k, beta, infectious_period=INFECTIOUS_PERIOD, 
                               sigma_vector=sigmas_k, pop_size=model.pop_size)
Rt_cf = calculate_Rt_over_time(res_cf, beta, infectious_period=INFECTIOUS_PERIOD, 
                                sigma_vector=sigmas_cf, pop_size=model.pop_size)

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

R0_t = beta * INFECTIOUS_PERIOD * seasonal_forcing  # R0(t) - same for both scenarios!


# Seasonality forcing on secondary axis - shows multiplicative factor


# Plot R0(t) (same for both) and Rt (different for both)
ax3.plot(times, R0_t, 'k--', lw=1.5, alpha=0.5, label='R₀(t) (Potential)', zorder=5)

ax3.fill_between(times, 0,R0_t, where=(seasonal_forcing > 1.0), 
                    alpha=0.15, color='skyblue', label='Winter boost (>1.0)')
ax3.fill_between(times,0, R0_t, where=(seasonal_forcing <= 1.0), 
                    alpha=0.15, color='orange', label='Summer reduction (<1.0)')

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
ax3.set_ylim([0, max(Rt_k.max(), Rt_cf.max(), R0_t.max()) * 1.15])

# ============ Plot 4: R₀ → R₀(t) → Rₜ Decomposition ============
ax4 = plt.subplot(4, 2, 4)
# Base R₀ (intrinsic, no seasonality) - same for both scenarios
base_R0_line = np.ones_like(times) * beta * INFECTIOUS_PERIOD
# R₀(t) with seasonality - SAME for both scenarios  
R0_t_line = base_R0_line * seasonal_forcing

ax4.fill_between(times, 0, base_R0_line, alpha=0.1, color='gray', label='Base R₀ (β×D)')
ax4.fill_between(times, 0, R0_t_line, alpha=0.15, color='purple', label='R₀(t) (Seasonality)')
ax4.plot(times, Rt_k, color='#d62728', lw=2.5, label='Rₜ (K lineage)', zorder=10)
ax4.plot(times, Rt_cf, color='#1f77b4', lw=2.5, ls='--', label='Rₜ (Counterfactual)', zorder=10)

ax4.set_title('From R₀ to Rₜ: Decomposition', fontsize=11, fontweight='bold')
ax4.set_xlabel('Days')
ax4.set_ylabel('Reproduction Number')
ax4.legend(loc='best', fontsize=8)
ax4.grid(alpha=0.3)
ax4.text(0.02, 0.98, 'Key: R₀(t) identical for both.\nRₜ differs due to immune escape.', 
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
ax5.plot(times, res_cf['I'] / 1e6, label='Counterfactual Infections', 
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
cohort_labels = ['Naive'] + [str(y) for y in model.epochs] + ['Vacc']
x = np.arange(len(cohort_labels))
width = 0.35

ax6.bar(x - width/2, sigmas_k, width, label='Actual K', color='#d62728', alpha=0.7)
ax6.bar(x + width/2, sigmas_cf, width, label='Counterfactual', color='#1f77b4', alpha=0.7)
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
ax8.fill_between(res_cf['time'], S_eff_cf, alpha=0.3, color='#1f77b4', label='Counterfactual')
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

