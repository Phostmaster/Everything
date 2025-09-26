import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gradient

# Parameters from UWT paper (aligned with Golden Spark)
k_damp = 0.0001         # Reduced damping
alpha = 150             # Adjusted coupling
f_ALD = 2.0             # ALD factor
g_wave = 19.5           # Scalar-boosted gravity (paper value)
epsilon = 1e-30         # Metric correction (m^2, paper value)
phi_1_init = 9.5e-4     # Initial Phi_1 (GeV, paper value)
phi_2_init = 0.5        # Initial Phi_2 (GeV, paper value)
phi_product_init = phi_1_init * phi_2_init  # Initial |Phi_1 Phi_2| (GeV^2)
v = 0.226               # Vacuum expectation value (GeV)
lambda_param = 1e-8     # Adjusted density constraint
dt = 1e-4               # Initial time step
dt_max = 1e-2           # Maximum time step
dt_min = 1e-6           # Minimum time step
t_max = 1.5e18          # Extended max time (s), ~0.5 Gyr with buffer (tweak)
n_points = 100          # Number of spatial points
dx = 1.0 / (n_points - 1)  # Spatial step
convergence_threshold = 1e-5  # Convergence criterion
driving_force = 5e-9    # Increased driving force (tweak)
kwave = 2.35e-3         # Wave number (paper value)
epsilon_CP = 2.58e-41   # CP phase (paper value)

# Initialize arrays with density wave pattern
x = np.linspace(0, 1, n_points)
phi_1 = np.full(n_points, phi_1_init) * (1 + 1e-5 * np.cos(kwave * x))  # Golden Spark perturbation
phi_2 = np.full(n_points, phi_2_init) * (1 + 1e-5 * np.sin(kwave * x + epsilon_CP * np.pi))  # Phase-locked
phi_product = np.abs(phi_1 * phi_2)
t = 0.0
time_steps = []

# Effective gravitational constant
G = 6.67430e-11  # m^3 kg^-1 s^-2
M_Pl = 1.22e19   # Planck mass (GeV)
G_eff_factor = 1 + 5.2e-37  # From SBG approximation

# Simulation loop
while t < t_max:
    # Compute spatial gradient
    grad_phi_1 = gradient(phi_1, dx)
    grad_phi_2 = gradient(phi_2, dx)

    # Pulsing damping term with driving force and SBG
    damping_term = -k_damp * (grad_phi_2 * phi_1 + grad_phi_1 * phi_2) - \
                   alpha * phi_1 * phi_2 * f_ALD / (1 + np.exp(-phi_product / 1e4)) + \
                   driving_force * (1 + g_wave * np.abs(phi_1 * phi_2))  # SBG boost

    # Update scalar fields
    phi_1_new = phi_1 + dt * damping_term
    phi_2_new = phi_2 + dt * damping_term

    # Adaptive time step based on field magnitude
    phi_product = np.abs(phi_1_new * phi_2_new)
    dt_new = max(min(0.01 / (1 + np.max(phi_product) / 1e4), dt_max), dt_min)

    # Apply density constraint to prevent singularity
    rho_phi = phi_product**2
    max_rho = lambda_param * v**4
    phi_1_new = np.where(rho_phi > max_rho, phi_1_new * np.sqrt(max_rho / rho_phi), phi_1_new)
    phi_2_new = np.where(rho_phi > max_rho, phi_2_new * np.sqrt(max_rho / rho_phi), phi_2_new)

    # Update fields
    phi_1 = phi_1_new
    phi_2 = phi_2_new

    # Record time step
    time_steps.append(t)
    t += dt_new  # Use updated dt

    # Monitor convergence and divergence
    max_phi_change = np.max(np.abs(phi_1_new - phi_1))
    if max_phi_change < convergence_threshold:
        print(f"Simulation converged at t = {t:.2e} s with max change {max_phi_change:.2e}")
        break
    if np.max(np.abs(phi_1)) > 1e8 or np.max(np.abs(phi_2)) > 1e8:
        print(f"Simulation stopped at t = {t:.2e} s due to divergence.")
        break

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(x, phi_1, label=r'$\Phi_1$')
plt.plot(x, phi_2, label=r'$\Phi_2$')
plt.plot(x, phi_product, label=r'$|\Phi_1 \Phi_2|$')
plt.xlabel('Spatial Coordinate (normalized)')
plt.ylabel('Field Value (GeV)')
plt.title('Scalar Field Evolution in UWT (JWST Context)')
plt.legend()
plt.grid(True)
plt.show()

# Print final phi_product for black hole metric
print(f"Final |Phi_1 Phi_2| = {np.max(phi_product):.2e} GeV^2")
print(f"Simulation ran for {len(time_steps)} steps up to t = {t:.2e} s")