import numpy as np
from scipy.integrate import solve_bvp, trapezoid
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# Constants in natural units (GeV)
HBARC_GEV_FM = 0.1973269804  # GeV * fm
FM_TO_GEV_INV = 1.0 / HBARC_GEV_FM  # fm -> GeV^-1
r0_fm = 1.2  # typical nuclear radius parameter in fm
r0 = r0_fm * FM_TO_GEV_INV  # GeV^-1 (~6.08 GeV^-1)
print(f"r0 (GeV^-1) = {r0:.6f}")
U_TO_GEV = 0.93149410242  # u -> GeV
M_E = 0.00051099895  # electron mass in GeV
M_P = 0.9382720813  # proton mass in GeV
M_N = 0.9395654133  # neutron mass in GeV
v_unified = 0.226  # GeV (UWT vacuum scale)
lambda_h_prime = 0.1  # Further reduced Higgs-like coupling for stability
m_s_r0 = 0.1  # Dimensionless scalar mass parameter
initial_g_s = 1.0  # Scalar coupling
initial_g_v = 1.0  # Vector coupling
rho_max = 2.0 * r0  # Reduced radial cutoff (~12.16 GeV^-1)
n_points = 400  # Increased points for finer resolution
phi_eps = 1e-12  # Small epsilon for stability
print("Launching adjusted ToE model with refined solver and parameters.")

# Observed Nuclear Data (normalized to nuclear masses in GeV)
def normalize_masses(nuclei):
    """Normalize masses to GeV, subtracting electron masses for nuclear values."""
    fixed = []
    for n in nuclei:
        A, Z, m = n["A"], n["Z"], n["mass_obs"]
        ratio = m / A
        if 0.9 < ratio < 1.1:  # Likely in atomic mass units (u)
            m_GeV = m * U_TO_GEV - Z * M_E
        else:
            m_GeV = m
        fixed.append({"A": A, "mass_obs_GEV": m_GeV, "Z": Z})  # Fixed typo: mass_obs_GeV
    return fixed

nuclei = normalize_masses([
    {"A": 1, "mass_obs": 0.938, "Z": 1},  # H-1
    {"A": 2, "mass_obs": 1.875, "Z": 1},  # H-2
    {"A": 3, "mass_obs": 2.808, "Z": 1},  # H-3
    {"A": 4, "mass_obs": 3.727, "Z": 2},  # He-4
    {"A": 6, "mass_obs": 5.605, "Z": 3},  # Li-6
    {"A": 7, "mass_obs": 6.533, "Z": 3},  # Li-7
    {"A": 9, "mass_obs": 8.394, "Z": 4},  # Be-9
    {"A": 10, "mass_obs": 9.321, "Z": 5},  # B-10
    {"A": 11, "mass_obs": 10.069, "Z": 5},  # B-11
    {"A": 12, "mass_obs": 11.178, "Z": 6},  # C-12
    {"A": 13, "mass_obs": 12.099, "Z": 6},  # C-13
    {"A": 14, "mass_obs": 13.114, "Z": 7},  # N-14
    {"A": 15, "mass_obs": 14.003, "Z": 7},  # N-15
    {"A": 16, "mass_obs": 14.911, "Z": 8},  # O-16
    {"A": 17, "mass_obs": 15.995, "Z": 8},  # O-17
    {"A": 18, "mass_obs": 17.156, "Z": 8},  # O-18
    {"A": 19, "mass_obs": 18.996, "Z": 9},  # F-19
    {"A": 20, "mass_obs": 19.992, "Z": 10},  # Ne-20
    {"A": 23, "mass_obs": 22.994, "Z": 11},  # Na-23
    {"A": 24, "mass_obs": 23.985, "Z": 12},  # Mg-24
    {"A": 27, "mass_obs": 26.981, "Z": 13},  # Al-27
    {"A": 28, "mass_obs": 27.977, "Z": 14},  # Si-28
    {"A": 31, "mass_obs": 30.973, "Z": 15},  # P-31
    {"A": 32, "mass_obs": 31.972, "Z": 16},  # S-32
    {"A": 35, "mass_obs": 34.969, "Z": 17},  # Cl-35
    {"A": 40, "mass_obs": 39.962, "Z": 18},  # Ar-40
    {"A": 56, "mass_obs": 55.935, "Z": 26},  # Fe-56
    {"A": 59, "mass_obs": 58.934, "Z": 27},  # Co-59
    {"A": 64, "mass_obs": 63.929, "Z": 28},  # Ni-64
    {"A": 91, "mass_obs": 90.924, "Z": 40},  # Zr-91
    {"A": 120, "mass_obs": 119.904, "Z": 50},  # Sn-120
    {"A": 150, "mass_obs": 149.920, "Z": 62},  # Sm-150
    {"A": 180, "mass_obs": 179.947, "Z": 72},  # Hf-180
    {"A": 200, "mass_obs": 199.968, "Z": 80},  # Hg-200
    {"A": 220, "mass_obs": 219.964, "Z": 88},  # Ra-220
    {"A": 238, "mass_obs": 221.644, "Z": 92}  # U-238, corrected
])

# Precompute BVP solutions with parameter-aware cache
solution_cache = {}
def uwt_ode(rho, y, A, g_s, g_v, m_s_r0, lambda_h_prime, v_unified):
    """ODE system with simplified cubic Higgs term."""
    phi_0, phi_0_prime, V_0, V_0_prime = y
    rho_density = A * np.exp(-rho / max(r0, 1e-12))  # Scaled density with safeguard
    dphi_0_drho = phi_0_prime
    dV_0_drho = V_0_prime
    higgs_term = -lambda_h_prime * phi_0**3  # Simplified to cubic term
    dphi_0_prime_drho = -2.0 / (rho + phi_eps) * phi_0_prime + m_s_r0**2 * phi_0 - g_s * rho_density + higgs_term
    dV_0_prime_drho = -2.0 / (rho + phi_eps) * V_0_prime - g_v * rho_density
    return np.vstack((dphi_0_drho, dphi_0_prime_drho, dV_0_drho, dV_0_prime_drho))

def bc(ya, yb, A):
    """Boundary conditions."""
    return np.array([ya[1], ya[3], yb[0], yb[2]])  # phi'(0)=0, V'(0)=0, phi(rho_max)=0, V(rho_max)=0

def solve_bvp_for_A(A, g_s=initial_g_s, g_v=initial_g_v, m_s_r0=m_s_r0, lambda_h_prime=lambda_h_prime, v_unified=v_unified):
    """Precompute BVP solution with parameter-aware cache key."""
    key = (A, g_s, g_v, m_s_r0, lambda_h_prime, v_unified, rho_max, n_points)
    if key in solution_cache:
        return solution_cache[key]
    rho = np.linspace(1e-4, rho_max, n_points)  # Increased initial step
    phi_0_init = 0.01 * (A ** (1/3)) * np.exp(- (rho / (5.0 * r0))**2)  # Reduced amplitude
    V_0_init = 0.005 * (A ** (1/3)) * np.exp(- (rho / (5.0 * r0))**2)
    y_init = np.vstack((phi_0_init, np.gradient(phi_0_init, rho), V_0_init, np.gradient(V_0_init, rho)))
    try:
        sol = solve_bvp(lambda r, y: uwt_ode(r, y, A, g_s, g_v, m_s_r0, lambda_h_prime, v_unified),
                        lambda ya, yb: bc(ya, yb, A), rho, y_init, tol=1e-5, max_nodes=20000)
        if sol.status != 0:
            print(f"Solver did not converge for A = {A}, status = {sol.status}")
            solution_cache[key] = None
            return None
        print(f"A = {A}: phi_0(0) = {sol.y[0, 0]:.6e} GeV, V_0(0) = {sol.y[2, 0]:.6e} GeV")
        solution_cache[key] = sol
        return sol
    except Exception as e:
        print(f"Error solving for A = {A}: {e}")
        solution_cache[key] = None
        return None

# Precompute all BVP solutions
unique_As = sorted({n["A"] for n in nuclei})
print("Precomputing BVPs for A values:", unique_As)
for A in unique_As:
    solve_bvp_for_A(A)

# SEMF Binding Energy
def binding_energy_semf(A, Z, params):
    """Bethe-Weizsäcker SEMF in GeV."""
    av, aS, aC, aA, aP = params
    N = A - Z
    delta = 0.0 if A % 2 == 1 else aP / (A ** 0.5) * (-1 if (Z % 2 == 1 or N % 2 == 1) else 1)
    B = av * A - aS * A**(2/3) - aC * Z * (Z - 1) / (A ** (1/3)) - aA * (A - 2 * Z)**2 / A + delta
    return B

# UWT Correction
def uwt_correction(sol, A, theta_y):
    """Global UWT correction in GeV with adjusted scaling."""
    if sol is None:
        return 0.0
    phi0 = sol.y[0, 0]
    V0 = sol.y[2, 0]
    c_y, A0, p = theta_y
    scale = (A ** (1/3)) / (1.0 + (A / max(A0, 1e-6)) ** max(p, 0.01))
    correction = c_y * scale * (phi0 * V0)
    return correction

# Mass Prediction
def predict_mass(A, Z, sol, semf_params, theta_y):
    """Predict nuclear mass using SEMF and UWT correction."""
    mass_free = M_P * Z + M_N * (A - Z)
    B = binding_energy_semf(A, Z, semf_params)
    delta = uwt_correction(sol, A, theta_y)
    return mass_free - B + delta

# Optimization with Global Parameters
def residuals_global(params):
    """Residuals for least_squares fitting 8 global params (5 SEMF + 3 UWT)."""
    semf_params = params[:5]  # av, aS, aC, aA, aP
    theta_y = params[5:]  # c_y, A0, p
    residuals = []
    for n in nuclei:
        A, Z, obs = n["A"], n["Z"], n["mass_obs_GEV"]  # Fixed typo: mass_obs_GEV
        sol = solution_cache.get((A, initial_g_s, initial_g_v, m_s_r0, lambda_h_prime, v_unified, rho_max, n_points), None)
        pred = predict_mass(A, Z, sol, semf_params, theta_y)
        residuals.append(pred - obs)
    return np.array(residuals)

# Initial guesses and bounds (GeV units, widened for flexibility)
SEMF_INIT = [15.5e-3, 17.0e-3, 0.72e-3, 23.0e-3, 11.0e-3]
theta0 = [*SEMF_INIT, 0.001, 150.0, 1.0]
lower = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.01]
upper = [0.05, 0.05, 0.01, 0.05, 0.05, 0.05, 10000.0, 5.0]

print("Starting global parameter optimization (SEMF + UWT)...")
result = least_squares(residuals_global, theta0, bounds=(lower, upper), verbose=2, xtol=1e-6, ftol=1e-9, gtol=1e-10)

# Extract fitted parameters
theta_hat = result.x
av, aS, aC, aA, aP, c_y, A0, p = theta_hat
print("Fitted parameters:")
print(f"SEMF: av={av:.6e} GeV, aS={aS:.6e}, aC={aC:.6e}, aA={aA:.6e}, aP={aP:.6e}")
print(f"UWT: c_y={c_y:.6e} (GeV^-? scaled), A0={A0:.3f}, p={p:.3f}")

# Predict and compare
predicted_masses = []
observed_masses = []
A_values = []
for n in nuclei:
    A, Z, obs = n["A"], n["Z"], n["mass_obs_GEV"]
    sol = solution_cache.get((A, initial_g_s, initial_g_v, m_s_r0, lambda_h_prime, v_unified, rho_max, n_points), None)
    pred = predict_mass(A, Z, sol, (av, aS, aC, aA, aP), (c_y, A0, p))
    predicted_masses.append(pred)
    observed_masses.append(obs)
    A_values.append(A)
    print(f"A = {A}: Observed {obs:.6f} GeV, Predicted {pred:.6f} GeV, Error = {(pred - obs):.6e} GeV")

# Compute RMS error
errors = np.array([p - o for p, o in zip(predicted_masses, observed_masses)])
rms_error = np.sqrt(np.mean(errors**2))
print(f"RMS Error: {rms_error:.6e} GeV")

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(A_values, observed_masses, color='blue', label='Observed')
plt.scatter(A_values, predicted_masses, color='red', label='Predicted')
plt.xlabel("A (Nucleon Number)")
plt.ylabel("Mass (GeV)")
plt.title("Observed vs Predicted Nuclear Masses (SEMF + UWT correction)")
plt.legend()
plt.grid(True)
plt.show()