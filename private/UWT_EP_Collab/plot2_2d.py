import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.linalg import solve_banded
import os

# Parameters
L_r, L_z = 1.0, 1.0 # Domain size (m)
A = 0.1 # Cross-sectional area (m^2)
V = L_r * L_z * A # Volume (m^3)
nr, nz = 64, 64 # Grid points
dr, dz = L_r / (nr - 1), L_z / (nz - 1)
r = np.linspace(0, L_r, nr)
z = np.linspace(0, L_z, nz)
R, Z = np.meshgrid(r, z)
nt = 1000 # For dt = 1e-12
t = np.linspace(0, 1e-9, nt)
dt = t[1] - t[0]
k = 0.0047
phi1 = 0.0
phi2 = np.pi / 2
g_wave = 5e-8
k_coupling = 1e-6
k_U = 2e8
rho = 1000
kappa = 5e5
alpha = 1.5
k_damp = 0.001
theta = np.pi
v = 0.226 / 6.242e18

# Initial UWT scalar fields
Phi1 = 0.00095 * np.cos(k * (R + Z) + phi1)
Phi2 = 0.5 * np.sin(k * (R + Z) + phi2)
psi = Phi1 + Phi2 # Stream function
u_r = -np.gradient(psi, dz, axis=0) # u_r = -∂ψ/∂z
u_z = np.gradient(psi, dr, axis=1) # u_z = ∂ψ/∂r
h = np.zeros((nt, nz, nr))
coherence = np.zeros((nt, nz, nr))
eta_UWT = np.zeros((nz, nr))
div_u = np.zeros((nz, nr))

# Feedback loop
def feedback_loop(Phi1, Phi2, grad_Phi1_r, grad_Phi1_z, grad_Phi2_r, grad_Phi2_z, prev_coherence, t_idx):
    epsilon_snr = 1e-10
    SNR = np.clip(np.mean(np.abs(Phi1 + Phi2) / (np.std(Phi1 + Phi2) + epsilon_snr)), 0, 10)
    grad_magnitude = np.clip(np.mean(np.abs(grad_Phi1_r + grad_Phi1_z) + np.abs(grad_Phi2_r + grad_Phi2_z)), 0, 10)
    time_factor = 0.5 * np.sin(2 * np.pi * t_idx / nt)
    theta_new = theta + 0.2 * np.pi / 180 * SNR * (1 + 0.3 * grad_magnitude + time_factor) + 0.00235 * (R + Z)
    return theta_new

# Poisson solver for psi
def solve_poisson(source, dr, dz):
    n = nr * nz
    source_flat = source.flatten()
    ab = np.zeros((3, n))
    for i in range(n):
        ir, iz = i % nr, i // nr
        if ir == 0 or ir == nr-1 or iz == 0 or iz == nz-1:
            ab[1, i] = 1
            source_flat[i] = 0
        else:
            ab[1, i] = -2 / dr**2 - 2 / dz**2
            ab[0, i+1] = 1 / dr**2 if i+1 < n and (i+1) % nr != 0 else 0
            ab[2, i-1] = 1 / dr**2 if i-1 >= 0 and i % nr != 0 else 0
            ab[0, i+nr] = 1 / dz**2 if i+nr < n else 0
            ab[2, i-nr] = 1 / dz**2 if i-nr >= 0 else 0
    psi_flat = solve_banded((1, 1), ab, source_flat)
    return psi_flat.reshape((nz, nr))

# Dynamics
def dynamics(Phi1, Phi2, psi, u_r, u_z, t_idx):
    grad_Phi1_r = np.gradient(Phi1, dr, axis=1)
    grad_Phi1_z = np.gradient(Phi1, dz, axis=0)
    grad_Phi2_r = np.gradient(Phi2, dr, axis=1)
    grad_Phi2_z = np.gradient(Phi2, dz, axis=0)
    phi_product = np.abs(Phi1 * Phi2)
    
    dPhi1_dt = -k_damp * (grad_Phi1_r * Phi1 + grad_Phi1_z * Phi1) + alpha * phi_product * np.cos(k * (R + Z) + theta)
    dPhi2_dt = -k_damp * (grad_Phi2_r * Phi2 + grad_Phi2_z * Phi2) + alpha * phi_product * np.cos(k * (R + Z) + theta)
    
    eta_UWT = k_coupling * (Phi1 + Phi2)**2
    eta_UWT = gaussian_filter(eta_UWT, sigma=1)
    
    f = g_wave * (Phi1**2 + Phi2**2)
    div_u = np.gradient(u_r, dr, axis=1) + np.gradient(u_z, dz, axis=0)
    dpsi_dt = -(u_r * np.gradient(u_r, dr, axis=1) + u_z * np.gradient(u_r, dz, axis=0)) + k_coupling * (np.gradient(np.gradient(psi, dr, axis=1), dr, axis=1) + np.gradient(np.gradient(psi, dz, axis=0), dz, axis=0)) + f - kappa * div_u
    dpsi_dt = np.clip(dpsi_dt, -1e2, 1e2)
    
    if t_idx % 10 == 0:
        source = -np.gradient(u_r * np.gradient(u_r, dr, axis=1) + u_z * np.gradient(u_r, dz, axis=0) + f, dr, axis=1)
        psi_correction = solve_poisson(source, dr, dz)
        psi += psi_correction
    u_r = -np.gradient(psi, dz, axis=0)
    u_z = np.gradient(psi, dr, axis=1)
    
    return dPhi1_dt, dPhi2_dt, dpsi_dt, np.max(np.abs(div_u)), eta_UWT, u_r, u_z

# Simulation
theta_values = []
for i in range(nt):
    grad_Phi1_r = np.gradient(Phi1, dr, axis=1)
    grad_Phi1_z = np.gradient(Phi1, dz, axis=0)
    grad_Phi2_r = np.gradient(Phi2, dr, axis=1)
    grad_Phi2_z = np.gradient(Phi2, dz, axis=0)
    theta = feedback_loop(Phi1, Phi2, grad_Phi1_r, grad_Phi1_z, grad_Phi2_r, grad_Phi2_z, coherence[i-1] if i > 0 else 0, i)
    theta_values.append(theta)
    
    dPhi1_dt, dPhi2_dt, dpsi_dt, max_div_u, eta_UWT, u_r, u_z = dynamics(Phi1, Phi2, psi, u_r, u_z, i)
    Phi1 += dPhi1_dt * dt
    Phi2 += dPhi2_dt * dt
    psi += dpsi_dt * dt
    u_r = np.clip(u_r, -5e2, 5e2)
    u_z = np.clip(u_z, -5e2, 5e2)
    
    u_r[:, 0] = u_r[:, -1] = u_r[0, :] = u_r[-1, :] = 0
    u_z[:, 0] = u_z[:, -1] = u_z[0, :] = u_z[-1, :] = 0
    u_r = gaussian_filter(u_r, sigma=2)
    u_z = gaussian_filter(u_z, sigma=2)
    
    U = k_U * (Phi1 + Phi2)**2
    h[i] = (rho / 2) * (u_r**2 + u_z**2) + U
    h[i] = np.clip(h[i], 0, 9e7 * 2.0)
    
    coherence_base = 0.001 + 0.0005 * np.abs(Phi1 + Phi2)
    coherence[i] = np.clip(5e3 * coherence_base, 10.0, 12.0)
    
    if i % 100 == 0:
        print(f"Step {i}: Max |Phi1| = {np.max(np.abs(Phi1)):.2e}, Max |Phi2| = {np.max(np.abs(Phi2)):.2e}, Max |u| = {np.max(np.abs(np.sqrt(u_r**2 + u_z**2))):.2e}, Enthalpy = {np.max(h[i]):.2e} J/m³, Coherence = {np.max(coherence[i]):.4f}sigma, Max |div_u| = {max_div_u:.2e}")

# Output
max_u = np.max(np.abs(np.sqrt(u_r**2 + u_z**2)))
avg_eta = np.mean(eta_UWT)
max_h = np.max(h)
max_coherence = np.max(coherence)
max_div = np.max(np.abs(np.gradient(u_r, dr, axis=1) + np.gradient(u_z, dz, axis=0)))
print(f"Max Velocity: {max_u:.2e} m/s")
print(f"Avg Viscosity: {avg_eta:.2e} Pa·s")
print(f"Max Enthalpy Density: {max_h:.2e} J/m³")
print(f"Max Coherence: {max_coherence:.4f}sigma")
print(f"Max Divergence: {max_div:.2e}")

# Plot (No Table)
plt.figure(figsize=(10, 8))
plt.subplot(3, 1, 1)
plt.contourf(R, Z, np.sqrt(u_r**2 + u_z**2), cmap='viridis')
plt.colorbar(label='Velocity (m/s)')
plt.title("Velocity Magnitude")
plt.xlabel("r (m)")
plt.ylabel("z (m)")

plt.subplot(3, 1, 2)
plt.contourf(R, Z, eta_UWT, cmap='viridis')
plt.colorbar(label='Viscosity (Pa·s)')
plt.title("UWT-Modified Viscosity")
plt.xlabel("r (m)")
plt.ylabel("z (m)")

plt.subplot(3, 1, 3)
plt.contourf(R, Z, h[-1], cmap='viridis')
plt.colorbar(label='Enthalpy Density (J/m³)')
plt.title("Enthalpy Density (Final Step)")
plt.xlabel("r (m)")
plt.ylabel("z (m)")

plt.tight_layout()
plt.savefig("private/UWT_EP_Collab/navier_stokes_2d.png")
plt.show(block=True)

# Save to private GitHub folder
os.makedirs("private/UWT_EP_Collab", exist_ok=True)
with open("private/UWT_EP_Collab/navier_stokes_2d_results.txt", "w", encoding="utf-8") as f:
    f.write(f"Max Velocity: {max_u:.2e} m/s\n")
    f.write(f"Avg Viscosity: {avg_eta:.2e} Pa·s\n")
    f.write(f"Max Enthalpy Density: {max_h:.2e} J/m³\n")
    f.write(f"Max Coherence: {max_coherence:.4f}sigma\n")
    f.write(f"Max Divergence: {max_div:.2e}\n")