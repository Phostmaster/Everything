import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import cg
from scipy.sparse import diags

print("Simulation starting...")

# Parameters (Updated per To-Do List)
g_wave = 1e-6  # Boosted from 5e-7 for velocity
kappa = 1e4    # Increased from 5e3 for coherence
k_U = 2e8
alpha = 2.0
nr, ntheta, nz = 128, 128, 128  # Refined grid from 64x64x64
dr = dtheta = dz = 1.0 / 20      # Adjust if needed post-refine
r = np.linspace(0.01, 1, nr)
theta = np.linspace(0, 2*np.pi, ntheta)
z = np.linspace(0, 1, nz)
R, Theta, Z = np.meshgrid(r, theta, z, indexing='ij')
k_damp = 5e-5
k = 0.0047
nu = 1e-5
print(f"g_wave={g_wave}, k_damp={k_damp}, k_U={k_U}, alpha={alpha}, kappa={kappa}, nu={nu}")

# Initial Conditions (Updated Phi1 amplitude)
np.random.seed(42)
Phi1 = 0.95 * np.cos(k * (R + Z)) * np.cos(k * Theta) + 0.01 * np.random.randn(nr, ntheta, nz)  # Boosted from 0.095
Phi2 = 5.0 * np.sin(k * (R + Z) + np.pi/2) * np.sin(k * Theta) + 0.01 * np.random.randn(nr, ntheta, nz)
u_r = -np.gradient(Phi1, dz, axis=2)
u_theta = np.gradient(Phi1, dr, axis=0)
u_z = np.gradient(Phi1, dtheta, axis=1)

# Simulation Parameters
nt = 4000
dt = 5e-13
rho = 1000.0

# Compute Divergence Function
def compute_divergence(u_r, u_theta, u_z, dr, dtheta, dz):
    r_safe = R + 1e-10
    div = (np.gradient(u_r, dr, axis=0) + 
           np.gradient(u_theta, dtheta, axis=1) / r_safe + 
           np.gradient(u_z, dz, axis=2))
    return div

try:
    # Time Loop
    for t in range(nt):
        # Gradients
        dPhir_1 = np.gradient(Phi1, dr, axis=0)
        dPhitheta_1 = np.gradient(Phi1, dtheta, axis=1)
        dPhiz_1 = np.gradient(Phi1, dz, axis=2)
        dPhir_2 = np.gradient(Phi2, dr, axis=0)
        dPhitheta_2 = np.gradient(Phi2, dtheta, axis=1)
        dPhiz_2 = np.gradient(Phi2, dz, axis=2)
        
        # SNR and Theta Field
        SNR = np.abs(Phi1 + Phi2) / (np.std(Phi1 + Phi2) + 1e-10)
        grad_mag = np.sqrt(dPhir_1**2 + dPhitheta_1**2 + dPhiz_1**2 + dPhir_2**2 + dPhitheta_2**2 + dPhiz_2**2)
        theta_field = np.pi + 100 * 0.2 * np.pi / 180 * SNR * (1 + 0.3 * grad_mag)
        
        # Laplacian
        laplacian = np.zeros_like(Phi1)
        phi_sum = Phi1 + Phi2
        grad_r_r = np.gradient(np.gradient(phi_sum, dr, axis=0), dr, axis=0)
        grad_theta_theta = np.gradient(np.gradient(phi_sum, dtheta, axis=1), dtheta, axis=1)
        grad_z_z = np.gradient(np.gradient(phi_sum, dz, axis=2), dz, axis=2)
        laplacian[1:-1, 1:-1, 1:-1] = (grad_r_r[1:-1, 1:-1, 1:-1] + 
                                       (1/(R[1:-1, 1:-1, 1:-1] + 1e-10)**2) * grad_theta_theta[1:-1, 1:-1, 1:-1] + 
                                       grad_z_z[1:-1, 1:-1, 1:-1]) / (R[1:-1, 1:-1, 1:-1] + 1e-10)
        
        # Time Derivatives
        r_safe = R + 1e-10
        dPhi1_dt = np.zeros_like(Phi1)
        dPhi2_dt = np.zeros_like(Phi2)
        dPhi1_dt[1:-1, 1:-1, 1:-1] = ((np.gradient(r_safe[1:-1, 1:-1, 1:-1] * dPhir_1[1:-1, 1:-1, 1:-1], axis=0) + 
                                       np.gradient(dPhitheta_1[1:-1, 1:-1, 1:-1], axis=1) + 
                                       np.gradient(dPhiz_1[1:-1, 1:-1, 1:-1], axis=2)) / r_safe[1:-1, 1:-1, 1:-1] - 
                                      k_U * (2 * Phi1[1:-1, 1:-1, 1:-1] + Phi2[1:-1, 1:-1, 1:-1]) - 
                                      alpha * Phi2[1:-1, 1:-1, 1:-1] * np.cos(0.0047 * R[1:-1, 1:-1, 1:-1] + theta_field[1:-1, 1:-1, 1:-1]) - 
                                      g_wave * laplacian[1:-1, 1:-1, 1:-1] - k_damp * Phi1[1:-1, 1:-1, 1:-1])
        dPhi2_dt[1:-1, 1:-1, 1:-1] = ((np.gradient(r_safe[1:-1, 1:-1, 1:-1] * dPhir_2[1:-1, 1:-1, 1:-1], axis=0) + 
                                       np.gradient(dPhitheta_2[1:-1, 1:-1, 1:-1], axis=1) + 
                                       np.gradient(dPhiz_2[1:-1, 1:-1, 1:-1], axis=2)) / r_safe[1:-1, 1:-1, 1:-1] - 
                                      k_U * (Phi1[1:-1, 1:-1, 1:-1] + 2 * Phi2[1:-1, 1:-1, 1:-1]) - 
                                      alpha * Phi1[1:-1, 1:-1, 1:-1] * np.cos(0.0047 * R[1:-1, 1:-1, 1:-1] + theta_field[1:-1, 1:-1, 1:-1]) - 
                                      g_wave * laplacian[1:-1, 1:-1, 1:-1] - k_damp * Phi2[1:-1, 1:-1, 1:-1])
        
        # 3D Advection
        u_r_grad = u_r * np.gradient(Phi1, dr, axis=0) + (u_theta / (R + 1e-10)) * np.gradient(Phi1, dtheta, axis=1) + u_z * np.gradient(Phi1, dz, axis=2)
        dPhi1_dt[1:-1, 1:-1, 1:-1] += nu * laplacian[1:-1, 1:-1, 1:-1] - u_r_grad[1:-1, 1:-1, 1:-1]
        dPhi2_dt[1:-1, 1:-1, 1:-1] += nu * laplacian[1:-1, 1:-1, 1:-1] - u_r_grad[1:-1, 1:-1, 1:-1]
        
        # Crash Detection
        if np.any(np.isnan(dPhi1_dt)) or np.any(np.isnan(dPhi2_dt)):
            print(f"Crash detected at step {t + 19000}")
            break
        
        # Time Step Update
        Phi1 += dPhi1_dt * dt
        Phi2 += dPhi2_dt * dt
        
        # Velocity Update
        u_r_new = -np.gradient(Phi1, dz, axis=2)
        u_theta_new = np.gradient(Phi1, dr, axis=0)
        u_z_new = np.gradient(Phi1, dtheta, axis=1)
        div_u = compute_divergence(u_r_new, u_theta_new, u_z_new, dr, dtheta, dz)
        
        # Vorticity Calculation
        omega_r = np.gradient(u_z_new, dtheta, axis=1) - np.gradient(u_theta_new, dz, axis=2)
        omega_theta = np.gradient(u_r_new, dz, axis=2) - np.gradient(u_z_new, dr, axis=0)
        omega_z = np.gradient(u_theta_new, dr, axis=0) - np.gradient(u_r_new, dtheta, axis=1)
        omega_max = np.max(np.sqrt(omega_r**2 + omega_theta**2 + omega_z**2))
        
        # Boundary Conditions
        u_r[0, :, :] = u_r[-1, :, :] = 0
        u_z[:, :, 0] = u_z[:, :, -1] = 0
        u_theta[0, :, :] = u_theta[-1, :, :] = 0
        
        # Diagnostics (every 100 steps or end)
        if t % 100 == 0 or t == nt - 1:
            max_vel = np.max(np.sqrt(u_r**2 + u_theta**2 + u_z**2))
            div = np.max(np.abs(div_u))
            phi_sum = Phi1 + Phi2
            enthalpy = (rho / 2) * max_vel**2 + k_U * np.max(np.abs(phi_sum)**2)
            coh = np.max(kappa * (0.001 + 0.0005 * np.abs(phi_sum)))
            print(f"Step {t + 19000}: Max Velocity = {max_vel:.3e} m/s, Divergence = {div:.3e}, Coherence = {coh:.3f}sigma, Enthalpy = {enthalpy:.3e} J/m³, Vorticity Max = {omega_max:.3e} s^-1")
        
        # Pressure Correction
        nri, nthetai, nzi = nr-2, ntheta-2, nz-2
        diagonals = [-np.ones(nri*nthetai*nzi), 6*np.ones(nri*nthetai*nzi), -np.ones(nri*nthetai*nzi)]
        offsets = [-nri*nthetai, 0, nri*nthetai]
        A = diags(diagonals, offsets, shape=(nri*nthetai*nzi, nri*nthetai*nzi))
        p = np.zeros((nr, ntheta, nz))
        rhs = -dr**2 * div_u[1:-1, 1:-1, 1:-1].flatten()
        p_inner, info = cg(A, rhs, rtol=1e-10)  # Tightened tolerance to 1e-10
        if info != 0:
            print(f"CG failed at step {t + 19000} with info {info}")
            break
        p[1:-1, 1:-1, 1:-1] = p_inner.reshape(nri, nthetai, nzi)
        
        # Pressure Gradients
        dp_dr = np.gradient(p, dr, axis=0)
        dp_dtheta = np.gradient(p, dtheta, axis=1)
        dp_dz = np.gradient(p, dz, axis=2)
        u_r = u_r_new - dp_dr
        u_theta = u_theta_new - (1 / (R + 1e-10)) * dp_dtheta
        u_z = u_z_new - dp_dz

    # Save and Complete
    if t == nt - 1:
        np.save('3D_velocity_field.npy', np.sqrt(u_r**2 + u_theta**2 + u_z**2))
        print("3D velocity field saved as 3D_velocity_field.npy")

except KeyboardInterrupt:
    print(f"Simulation interrupted at step {t + 19000}. Saving current state...")
    np.save('3D_velocity_field_partial.npy', np.sqrt(u_r**2 + u_theta**2 + u_z**2))
    print("Partial 3D velocity field saved as 3D_velocity_field_partial.npy")

print("Simulation completed or interrupted!")
