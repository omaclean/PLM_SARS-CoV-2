import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class AntigenicSeirModel:
    """
    A stratified SEIR model where susceptibility is a function of:
    1. Antigenic Distance (Euclidean distance in PLANT space)
    2. Immunological Waning (Time since last infection)
    
    References:
    - PLANT Paper Fig 1G/6B: 2 antigenic units is the standard threshold for drift/mismatch.
    - PLANT Paper Fig 4G: ~1.47% fitness advantage per antigenic unit (used for calibration checks).
    """
    
    def __init__(self, historical_strains, population_size=67_000_000, current_year=2025.9):
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

    def deriv(self, y, t, beta, sigma_vector, latent_period=2.0, infectious_period=3.0):
        """
        SEIR System.
        State y: [S_naive, S_hist1...S_histN, S_vacc, E, I, R_total]
        """
        num_compartments = len(sigma_vector)
        
        # Unpack
        S_cohorts = y[:num_compartments]
        E = y[num_compartments]
        I = y[num_compartments + 1]
        R = y[num_compartments + 2]
        
        # Parameters
        delta = 1.0 / latent_period
        gamma = 1.0 / infectious_period
        
        # Force of Infection
        # Note: beta is constant. The fitness advantage is mechanistic via 'sigma_vector'.
        lam = beta * I / self.pop_size
        
        # Derivatives
        dS_dt = -lam * sigma_vector * S_cohorts
        
        # New exposed = sum of flows from all S compartments
        new_infections = np.sum(lam * sigma_vector * S_cohorts)
        
        dE_dt = new_infections - delta * E
        dI_dt = delta * E - gamma * I
        dR_dt = gamma * I
        
        return np.concatenate([dS_dt, [dE_dt, dI_dt, dR_dt]])

    def run(self, strain_coord, vaccine_coord, pop_distribution, 
            beta=0.6, latent_period=2.0, infectious_period=3.0, 
            seed_infections=100, days=180):
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

# Historic Centroids (Mock PLANT output)
history = {
    2019: (0.5, 0.5, 0.0),
    2020: (1.2, 0.8, 0.1),
    2021: (1.5, 1.0, 0.2),
    2022: (2.5, 1.5, 0.5),
    2023: (3.0, 2.0, 0.8),
    2024: (3.2, 2.2, 0.9) 
}

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

model = AntigenicSeirModel(history, population_size=67_000_000, current_year=2025.9)

# ==========================================
# 2. SCENARIO RUNS
# ==========================================
# %% 

# A. Actual Scenario: K Lineage (High Drift)
# [cite_start]Distance from 2024 (3.2, 2.2) is ~2.2 units -> Crosses the 2.0 threshold [cite: 147]
k_coord = (5.0, 3.5, 1.5) 
vacc_coord = (3.5, 2.5, 1.0) # Vaccine mismatched

beta=1.16 # Calibrated to match observed growth rates

res_k, sigmas_k = model.run(k_coord, vacc_coord, pop_dist, beta=beta)

# B. Counterfactual: No Mutation I160K
# Distance from 2024 is ~1.0 unit -> Within 2.0 threshold (protected)
cf_coord = (3.8, 2.8, 1.2) 

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
plt.figure(figsize=(12, 5))

# Plot 1: Curves
plt.subplot(1, 2, 1)
plt.plot(res_k['time'], res_k['I'], label='Actual (K Lineage)', color='#d62728', lw=2)
plt.plot(res_cf['time'], res_cf['I'], label='Counterfactual (No Mutation)', color='#1f77b4', ls='--')
plt.title('Epidemic Curve Comparison')
plt.xlabel('Days')
plt.ylabel('Infected (I)')
plt.legend()
plt.grid(alpha=0.3)

# Plot 2: Susceptibility Profile
plt.subplot(1, 2, 2)
cohort_labels = ['Naive'] + [str(y) for y in model.epochs] + ['Vacc']
x = np.arange(len(cohort_labels))
width = 0.35

plt.bar(x - width/2, sigmas_k, width, label='Actual Susc.', color='#d62728', alpha=0.7)
plt.bar(x + width/2, sigmas_cf, width, label='Counterfactual Susc.', color='#1f77b4', alpha=0.7)
plt.xticks(x, cohort_labels, rotation=45)
plt.title('Susceptibility by Cohort')
plt.ylabel('Probability of Infection (sigma)')
plt.legend()

plt.tight_layout()
plt.show()

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

