```python
import numpy as np
from scipy.sparse.linalg import cg
from scipy.sparse import diags
print("Fusion plasma flow test with revised Lagrangian starting...")

# Parameters (fusion plasma, tokamak-like, SI units)
rho = 1e-5  # Plasma density, kg/m^3
B = 5.0  # Magnetic field, T
mu_0 = 4 * np.pi * 1e-7  # Permeability, H/m
mu = 1e-5  # Dynamic viscosity, Pa·s
gamma = 0.001  # Scalar damping, s^-1
c_phi = 1e8  # Scalar wave speed, m/s (adjusted for plasma)
k_U = 2e8  # Coupling constant, kg^-1 m^3 s^-2
lambda_ = 2.51e-46
g_m = 0.01
v = 0.226 / 6.242e18  # GeV to kg conversion
nr, ntheta, nz = 64, 64, 64
dr = dtheta = dz = 0.05
r = np.linspace(0.01, 1, nr)
theta = np.linspace(0, 2 * np.pi, ntheta)
z = np.linspace(0, 1, nz)
R, Theta, Z = np.meshgrid(r, theta, z, indexing='ij')
phi_scale = 7.15e6
phase_shift = 340 * np.pi / 180
dt = 1e-7

# Initial Conditions
np.random.seed(42)
amp_factor = np.random.uniform(0.8, 1.2, size=(nr, ntheta, nz))
Phi1 = 4.03e-28 * phi_scale * amp_factor * (np.cos(0.00235 * (R + Z) + phase_shift) * np.cos(0.00235 * Theta + phase_shift) + 0.01 * np.random.randn(nr, ntheta, nz))
Phi2 = 1.68e-28 * phi_scale * amp_factor * (np.sin(0.00235 * (R + Z) + np.pi/2 + phase_shift) * np.sin(0.00235 * Theta + phase_shift) + 0.01 * np.random.randn(nr, ntheta, nz))
u_r = 1e4 * np.ones((nr, ntheta, nz)) + 0.01 * np.random.randn(nr, ntheta, nz)
u_theta = np.zeros((nr, ntheta, nz))
u_z = np.zeros((nr, ntheta, nz))
B_field = np.zeros((nr, ntheta, nz, 3))
B_field[:, :, :, 2] = B

# Divergence Function
def compute_divergence(u_r, u_theta, u_z, dr, dtheta, dz):
    r_safe = R + 1e-10
    div = (np.gradient(u_r, dr, axis=0) +
           np.gradient(u_theta, dtheta, axis=1) / r_safe +
           np.gradient(u_z, dz, axis=2))
    return div

try:
    for t in range(1000):
        # Scalar Gradients and Potential
        dPhir_1 = np.gradient(Phi1, dr, axis=0)
        dPhitheta_1 = np.gradient(Phi1, dtheta, axis=1)
        dPhiz_1 = np.gradient(Phi1, dz, axis=2)
        dPhir_2 = np.gradient(Phi2, dr, axis=0)
        dPhitheta_2 = np.gradient(Phi2, dtheta, axis=1)
        dPhiz_2 = np.gradient(Phi2, dz, axis=2)
        phi_grad_1_sq = dPhir_1**2 + dPhitheta_1**2 / (R + 1e-10)**2 + dPhiz_1**2
        phi_grad_2_sq = dPhir_2**2 + dPhitheta_2**2 / (R + 1e-10)**2 + dPhiz_2**2
        phi_prod = Phi1 * Phi2
        V = lambda_ * ((phi_prod**2 - v**2)**2) + 0.5 * k_U * (2 * Phi1**2 + Phi1 * Phi2 + 2 * Phi2**2)
        dV_dPhi1 = lambda_ * 4 * phi_prod * (phi_prod**2 - v**2) * Phi2 + k_U * (4 * Phi1 + Phi2)
        dV_dPhi2 = lambda_ * 4 * phi_prod * (phi_prod**2 - v**2) * Phi1 + k_U * (Phi1 + 4 * Phi2)
        dPhi1_dt = np.zeros_like(Phi1)
        dPhi2_dt = np.zeros_like(Phi2)
        dPhi1_dt[1:-1, 1:-1, 1:-1] = c_phi**2 * (np.gradient(dPhir_1, dr, axis=0) + np.gradient(dPhitheta_1 / (R + 1e-10), dtheta, axis=1) + np.gradient(dPhiz_1, dz, axis=2))[1:-1, 1:-1, 1:-1] - dV_dPhi1[1:-1, 1:-1, 1:-1] - g_m * rho * Phi2[1:-1, 1:-1, 1:-1] - gamma * np.gradient(Phi1, dt, axis=0)[1:-1, 1:-1, 1:-1]
        dPhi2_dt[1:-1, 1:-1, 1:-1] = c_phi**2 * (np.gradient(dPhir_2, dr, axis=0) + np.gradient(dPhitheta_2 / (R + 1e-10), dtheta, axis=1) + np.gradient(dPhiz_2, dz, axis=2))[1:-1, 1:-1, 1:-1] - dV_dPhi2[1:-1, 1:-1, 1:-1] - g_m * rho * Phi1[1:-1, 1:-1, 1:-1] - gamma * np.gradient(Phi2, dt, axis=0)[1:-1, 1:-1, 1:-1]
        if np.any(np.isnan(dPhi1_dt)) or np.any(np.isnan(dPhi2_dt)):
            print(f"Crash at step {t}: NaN in dPhi1_dt or dPhi2_dt")
            break
        Phi1 += dt * dPhi1_dt
        Phi2 += dt * dPhi2_dt
        # Fluid Update (Navier-Stokes with MHD)
        u_grad_u_r = u_r * np.gradient(u_r, dr, axis=0) + u_theta * np.gradient(u_r, dtheta, axis=1) / (R + 1e-10) + u_z * np.gradient(u_r, dz, axis=2)
        u_grad_u_theta = u_r * np.gradient(u_theta, dr, axis=0) + u_theta * np.gradient(u_theta, dtheta, axis=1) / (R + 1e-10) + u_z * np.gradient(u_theta, dz, axis=2)
        u_grad_u_z = u_r * np.gradient(u_z, dr, axis=0) + u_theta * np.gradient(u_z, dtheta, axis=1) / (R + 1e-10) + u_z * np.gradient(u_z, dz, axis=2)
        grad_r_r = np.gradient(np.gradient(u_r, dr, axis=0), dr, axis=0)
        grad_theta_theta = np.gradient(np.gradient(u_theta, dtheta, axis=1), dtheta, axis=1)
        grad_z_z = np.gradient(np.gradient(u_z, dz, axis=2), dz, axis=2)
        laplacian_u = np.zeros_like(u_r)
        laplacian_u[1:-1, 1:-1, 1:-1] = (grad_r_r[1:-1, 1:-1, 1:-1] + (1/(R[1:-1, 1:-1, 1:-1] + 1e-10)**2) * grad_theta_theta[1:-1, 1:-1, 1:-1] + grad_z_z[1:-1, 1:-1, 1:-1]) / (R[1:-1, 1:-1, 1:-1] + 1e-10)
        body_force = -rho * np.gradient(g_m * phi_prod, dr, axis=0)
        J = np.zeros((nr, ntheta, nz, 3))
        J[:, :, :, 0] = (1 / mu_0) * (np.gradient(B_field[:, :, :, 2], dtheta, axis=1) / (R + 1e-10) - np.gradient(B_field[:, :, :, 1], dz, axis=2))
        J[:, :, :, 1] = (1 / mu_0) * (np.gradient(B_field[:, :, :, 0], dz, axis=2) - np.gradient(B_field[:, :, :, 2], dr, axis=0))
        lorentz_force = np.zeros((nr, ntheta, nz, 3))
        lorentz_force[:, :, :, 0] = J[:, :, :, 1] * B_field[:, :, :, 2]
        lorentz_force[:, :, :, 1] = -J[:, :, :, 0] * B_field[:, :, :, 2]
        u_r_new = u_r + dt * (-u_grad_u_r + mu * laplacian_u + body_force + lorentz_force[:, :, :, 0]) / (rho + 1e-10)
        u_theta_new = u_theta + dt * (-u_grad_u_theta + mu * laplacian_u + lorentz_force[:, :, :, 1]) / (rho + 1e-10)
        u_z_new = u_z + dt * (-u_grad_u_z + mu * laplacian_u) / (rho + 1e-10)
        div_u = compute_divergence(u_r_new, u_theta_new, u_z_new, dr, dtheta, dz)
        nri, nthetai, nzi = nr-2, ntheta-2, nz-2
        diagonals = [-np.ones(nri*nthetai*nzi), 6*np.ones(nri*nthetai*nzi), -np.ones(nri*nthetai*nzi)]
        offsets = [-nri*nthetai, 0, nri*nthetai]
        A = diags(diagonals, offsets, shape=(nri*nthetai*nzi, nri*nthetai*nzi))
        p = np.zeros((nr, ntheta, nz))
        rhs = -dr**2 * div_u[1:-1, 1:-1, 1:-1].flatten()
        p_inner, info = cg(A, rhs, rtol=1e-12)
        if info != 0:
            print(f"CG failed at step {t} with info {info}")
            break
        p[1:-1, 1:-1, 1:-1] = p_inner.reshape(nri, nthetai, nzi)
        dp_dr = np.gradient(p, dr, axis=0)
        dp_dtheta = np.gradient(p, dtheta, axis=1)
        dp_dz = np.gradient(p, dz, axis=2)
        u_r = u_r_new - dt * dp_dr / (rho + 1e-10)
        u_theta = u_theta_new - dt * dp_dtheta / ((rho + 1e-10) * (R + 1e-10))
        u_z = u_z_new - dt * dp_dz / (rho + 1e-10)
        u_r[0, :, :] = u_r[-1, :, :] = 0
        u_z[:, :, 0] = u_z[:, :, -1] = 0
        u_theta[0, :, :] = u_theta[-1, :, :] = 0
        if t % 100 == 0 or t == 999:
            max_vel = np.max(np.sqrt(u_r**2 + u_theta**2 + u_z**2))
            div = np.max(np.abs(div_u))
            enthalpy = (rho / 2) * max_vel**2 + k_U * np.max(np.abs(Phi1 + Phi2)**2)
            coh = np.max(kappa_base / (1 + 1e3 * np.max(np.abs(Phi1 + Phi2))) * (0.001 + 0.0005 * np.abs(Phi1 + Phi2)))
            print(f"Step {t}: Max Velocity = {max_vel:.3e} m/s, Divergence = {div:.3e}, Coherence = {coh:.3f}sigma, Enthalpy = {enthalpy:.3e} J/m^3")
except Exception as e:
    print(f"Simulation crashed at step {t}: {str(e)}")
print("Fusion plasma test completed or interrupted!")
