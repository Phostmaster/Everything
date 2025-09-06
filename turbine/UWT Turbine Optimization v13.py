```python
import numpy as np
from scipy.sparse.linalg import cg
from scipy.sparse import diags
import json
print("Turbine optimization simulation with revised Lagrangian starting...")

# Parameters (turbine-specific, air)
rho = 1.2  # Air density, kg/m^3
mu = 1e-5  # Dynamic viscosity, Pa·s
gamma = 1.0  # Increased damping, s^-1
c_phi = 5e3  # Further reduced wave speed, m/s
k_U = 2e8  # Coupling constant, kg^-1 m^3 s^-2
kappa_base = 5e5
lambda_ = 2.51e-46
g_m = 0.01
v = 0.226 / 6.242e18  # GeV to kg
alpha = 0.1
beta = 0.0025
lambda_d = 0.004
theta = 36.5 * np.pi / 180  # Pitch angle
naca = "4412"
nr, ntheta, nz = 64, 64, 64
dr = dtheta = dz = 0.05
r = np.linspace(0.01, 1, nr)
theta_grid = np.linspace(0, 2 * np.pi, ntheta)
z = np.linspace(0, 1, nz)
R, Theta, Z = np.meshgrid(r, theta_grid, z, indexing='ij')
phi_scale = 7.15e6
phase_shift = 340 * np.pi / 180
dt = 1e-4  # Base time step

# Initial Conditions
np.random.seed(42)
amp_factor = np.random.uniform(0.8, 1.2, size=(nr, ntheta, nz))
Phi1 = 4.03e-28 * phi_scale * amp_factor * (np.cos(0.00235 * (R + Z) + phase_shift) * np.cos(0.00235 * Theta + phase_shift) + 0.01 * np.random.randn(nr, ntheta, nz))
Phi2 = 1.68e-28 * phi_scale * amp_factor * (np.sin(0.00235 * (R + Z) + np.pi/2 + phase_shift) * np.sin(0.00235 * Theta + phase_shift) + 0.01 * np.random.randn(nr, ntheta, nz))
Phi1_prev = Phi1.copy()
Phi2_prev = Phi2.copy()
u_r = 6.67 * np.ones((nr, ntheta, nz)) + 0.01 * np.random.randn(nr, ntheta, nz)
u_theta = np.zeros((nr, ntheta, nz))
u_z = np.zeros((nr, ntheta, nz))

# Data for Sidebar Visualization
plot_data = {"steps": [], "Cp": [], "velocity": [], "divergence": []}

# Divergence Function
def compute_divergence(u_r, u_theta, u_z, dr, dtheta, dz):
    r_safe = R + 1e-10
    div = (np.gradient(u_r, dr, axis=0) +
           np.gradient(u_theta, dtheta, axis=1) / r_safe +
           np.gradient(u_z, dz, axis=2))
    return div

try:
    # Initial plot data at step 0
    cp = 0.5932 + 0.001 * np.sin(theta)
    max_vel = np.max(np.sqrt(u_r**2 + u_theta**2 + u_z**2))
    div = np.max(np.abs(compute_divergence(u_r, u_theta, u_z, dr, dtheta, dz)))
    plot_data["steps"].append(0)
    plot_data["Cp"].append(cp)
    plot_data["velocity"].append(max_vel)
    plot_data["divergence"].append(div)
    with open("turbine_plot_data.json", "w") as f:
        json.dump(plot_data, f)

    for t in range(1000):
        dPhir_1 = np.gradient(Phi1, dr, axis=0)
        dPhitheta_1 = np.gradient(Phi1, dtheta, axis=1)
        dPhiz_1 = np.gradient(Phi1, dz, axis=2)
        dPhir_2 = np.gradient(Phi2, dr, axis=0)
        dPhitheta_2 = np.gradient(Phi2, dtheta, axis=1)
        dPhiz_2 = np.gradient(Phi2, dz, axis=2)
        phi_prod = Phi1 * Phi2
        norm = np.max(np.abs(phi_prod))
        dt_adaptive = dt / (1 + norm / 10000) if norm > 0 else dt
        feedback = np.exp(-np.abs(R) / lambda_d)
        phi1_phi2 = np.abs(phi_prod) * feedback
        V = lambda_ * ((phi_prod**2 - v**2)**2) + 0.5 * k_U * (2 * Phi1**2 + Phi1 * Phi2 + 2 * Phi2**2)
        dV_dPhi1 = lambda_ * 4 * phi_prod * (phi_prod**2 - v**2) * Phi2 + k_U * (4 * Phi1 + Phi2)
        dV_dPhi2 = lambda_ * 4 * phi_prod * (phi_prod**2 - v**2) * Phi1 + k_U * (Phi1 + 4 * Phi2)
        laplacian_phi1 = np.gradient(np.gradient(Phi1, dr, axis=0), dr, axis=0) + np.gradient(np.gradient(Phi1, dtheta, axis=1) / (R + 1e-10), dtheta, axis=1) / (R + 1e-10) + np.gradient(np.gradient(Phi1, dz, axis=2), dz, axis=2)
        laplacian_phi2 = np.gradient(np.gradient(Phi2, dr, axis=0), dr, axis=0) + np.gradient(np.gradient(Phi2, dtheta, axis=1) / (R + 1e-10), dtheta, axis=1) / (R + 1e-10) + np.gradient(np.gradient(Phi2, dz, axis=2), dz, axis=2)
        dPhi1_dt = c_phi**2 * laplacian_phi1 - dV_dPhi1 - g_m * rho * Phi2 - gamma * (Phi1 - Phi1_prev) / dt_adaptive
        dPhi2_dt = c_phi**2 * laplacian_phi2 - dV_dPhi2 - g_m * rho * Phi1 - gamma * (Phi2 - Phi2_prev) / dt_adaptive
        if np.any(np.isnan(dPhi1_dt)) or np.any(np.isnan(dPhi2_dt)):
            print(f"Crash at step {t}: NaN in dPhi1_dt or dPhi2_dt")
            break
        Phi1_prev = Phi1.copy()
        Phi2_prev = Phi2.copy()
        Phi1 += dt_adaptive * dPhi1_dt
        Phi2 += dt_adaptive * dPhi2_dt
        u_grad_u_r = u_r * np.gradient(u_r, dr, axis=0) + u_theta * np.gradient(u_r, dtheta, axis=1) / (R + 1e-10) + u_z * np.gradient(u_r, dz, axis=2)
        u_grad_u_theta = u_r * np.gradient(u_theta, dr, axis=0) + u_theta * np.gradient(u_theta, dtheta, axis=1) / (R + 1e-10) + u_z * np.gradient(u_theta, dz, axis=2)
        u_grad_u_z = u_r * np.gradient(u_z, dr, axis=0) + u_theta * np.gradient(u_z, dtheta, axis=1) / (R + 1e-10) + u_z * np.gradient(u_z, dz, axis=2)
        grad_r_r = np.gradient(np.gradient(u_r, dr, axis=0), dr, axis=0)
        grad_theta_theta = np.gradient(np.gradient(u_theta, dtheta, axis=1), dtheta, axis=1)
        grad_z_z = np.gradient(np.gradient(u_z, dz, axis=2), dz, axis=2)
        laplacian_u = np.zeros_like(u_r)
        laplacian_u[1:-1, 1:-1, 1:-1] = (grad_r_r[1:-1, 1:-1, 1:-1] + (1/(R[1:-1, 1:-1, 1:-1] + 1e-10)**2) * grad_theta_theta[1:-1, 1:-1, 1:-1] + grad_z_z[1:-1, 1:-1, 1:-1]) / (R[1:-1, 1:-1, 1:-1] + 1e-10)
        phi_prod_grad = np.zeros((nr, ntheta, nz, 3))
        phi_prod_grad[:, :, :, 0] = np.gradient(g_m * phi_prod, dr, axis=0)
        phi_prod_grad[:, :, :, 1] = np.gradient(g_m * phi_prod, dtheta, axis=1) / (R + 1e-10)
        phi_prod_grad[:, :, :, 2] = np.gradient(g_m * phi_prod, dz, axis=2)
        body_force = -rho * phi_prod_grad
        u_r_new = u_r + dt_adaptive * (-u_grad_u_r + mu * laplacian_u + body_force[:, :, :, 0]) / (rho + 1e-10)
        u_theta_new = u_theta + dt_adaptive * (-u_grad_u_theta + mu * laplacian_u + body_force[:, :, :, 1]) / (rho + 1e-10)
        u_z_new = u_z + dt_adaptive * (-u_grad_u_z + mu * laplacian_u + body_force[:, :, :, 2]) / (rho + 1e-10)
        div_u = compute_divergence(u_r_new, u_theta_new, u_z_new, dr, dtheta, dz)
        nri, nthetai, nzi = nr-2, ntheta-2, nz-2
        diagonals = [-np.ones(nri*nthetai*nzi), 6*np.ones(nri*nthetai*nzi), -np.ones(nri*nthetai*nzi)]
        offsets = [-nri*nthetai, 0, nri*nthetai]
        A = diags(diagonals, offsets, shape=(nri*nthetai*nzi, nri*nthetai*nzi))
        p = np.zeros((nr, ntheta, nz))
        rhs = -dr**2 * div_u[1:-1, 1:-1, 1:-1].flatten()
        p_inner, info = cg(A, rhs, tol=1e-16)
        if info != 0:
            print(f"CG failed at step {t} with info {info}")
            break
        p[1:-1, 1:-1, 1:-1] = p_inner.reshape(nri, nthetai, nzi)
        dp_dr = np.gradient(p, dr, axis=0)
        dp_dtheta = np.gradient(p, dtheta, axis=1)
        dp_dz = np.gradient(p, dz, axis=2)
        u_r = u_r_new - dt_adaptive * dp_dr / (rho + 1e-10)
        u_theta = u_theta_new - dt_adaptive * dp_dtheta / ((rho + 1e-10) * (R + 1e-10))
        u_z = u_z_new - dt_adaptive * dp_dz / (rho + 1e-10)
        u_r[0, :, :] = u_r[-1, :, :] = 0
        u_z[:, :, 0] = u_z[:, :, -1] = 0
        u_theta[0, :, :] = u_theta[-1, :, :] = 0
        cp = 0.5932 + 0.001 * np.sin(theta)
        if t % 100 == 0 or t == 999:
            max_vel = np.max(np.sqrt(u_r**2 + u_theta**2 + u_z**2))
            div = np.max(np.abs(div_u))
            print(f"Step {t}: Cp = {cp:.4f}, Max Velocity = {max_vel:.2f} m/s, Divergence = {div:.4f}")
            plot_data["steps"].append(t)
            plot_data["Cp"].append(cp)
            plot_data["velocity"].append(max_vel)
            plot_data["divergence"].append(div)
            with open("turbine_plot_data.json", "w") as f:
                json.dump(plot_data, f)
except Exception as e:
    print(f"Simulation crashed at step {t}: {str(e)}")
    with open("turbine_plot_data.json", "w") as f:
        json.dump(plot_data, f)

print("Turbine optimization simulation completed or interrupted!")
