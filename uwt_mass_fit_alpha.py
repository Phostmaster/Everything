import numpy as np
from scipy.optimize import curve_fit

# Observed fermion masses (GeV) - placeholder values from SM, 3 generations per family
observed_masses = {
    'leptons': [0.000511, 0.105, 1.777],  # e, mu, tau
    'up_quarks': [0.0022, 1.27, 172.5],   # u, c, t
    'down_quarks': [0.0047, 0.095, 4.18]  # d, s, b
}

# Generation indices (1, 2, 3)
generations = [1, 2, 3]

# Alpha model: m_T(g) = y_T * S * r_T^(g-1)
def alpha_model(g, y_T, r_T, S=1.0):
    return y_T * S * (r_T ** (g - 1))

# Fit function for each family
def fit_alpha_family(masses):
    g_data = np.array(generations)
    popt, _ = curve_fit(lambda g, y, r: alpha_model(g, y, r), g_data, masses, p0=[0.1, 0.5])
    return popt

# Beta model: ln(m_T(g)) = a_T + b_T(g-1) + c_T(g-1)^2
def beta_model(g, a_T, b_T, c_T):
    return np.exp(a_T + b_T * (g - 1) + c_T * (g - 1) ** 2)

# Fit function for Beta model
def fit_beta_family(masses):
    g_data = np.array(generations)
    popt, _ = curve_fit(lambda g, a, b, c: beta_model(g, a, b, c), g_data, np.log(masses), p0=[0.0, 0.0, 0.0])
    return popt

# Boson calculations with effective VEV
g = 0.6529  # SM gauge coupling
gp = 0.3572  # Hypercharge coupling
m_Z_obs = 91.1876  # GeV
v_eff = m_Z_obs * 2 / np.sqrt(g**2 + gp**2)  # From m_Z = (1/2) * sqrt(g^2 + g'^2) * v_eff
lambda_eff = (125.18**2) / (2 * v_eff**2)  # m_H = sqrt(2 * lambda_eff * v_eff^2), m_H = 125.18 GeV

# Run fits and print results
print("Running UWT Mass Law Fit (Alpha and Beta Versions)...")
for family, masses in observed_masses.items():
    y_T, r_T = fit_alpha_family(masses)
    a_T, b_T, c_T = fit_beta_family(masses)
    print(f"\nFamily: {family}")
    print(f"Alpha Fit - y_T: {y_T:.6f}, r_T: {r_T:.6f}")
    print(f"Beta Fit - a_T: {a_T:.6f}, b_T: {b_T:.6f}, c_T: {c_T:.6f}")

print(f"\nBoson Results:")
print(f"v_eff: {v_eff:.6f} GeV")
print(f"lambda_eff: {lambda_eff:.6f}")

# Optional: Predict masses for verification
for family, masses in observed_masses.items():
    print(f"\nPredicted {family} masses (Alpha with S=1):")
    for g in generations:
        pred_mass = alpha_model(g, y_T, r_T)
        print(f"g={g}: {pred_mass:.6f} GeV (Observed: {masses[g-1]:.6f} GeV)")