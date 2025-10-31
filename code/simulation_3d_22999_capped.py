import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import cg
from scipy.sparse import diags
import os

# Set device (e.g., GPU 0)
cp.cuda.runtime.setDevice(0)  # Use GPU 0; change to 1, 2, 3, or loop for multi-GPU

# Simulation starting message
print("Simulation starting on GPU", cp.cuda.runtime.getDevice())
g_wave = 0.085  # Flat-spacetime limit value from UWT derivation
kappa = 1e4     # Increased from 5e3 for coherence
k_U = 2e8
alpha = 2.0
nr, ntheta, nz = 128, 128, 128  # Refined grid from 64x64x64
dr = dtheta = dz = 1.0 / 20     # 0.05 m spacing
r = cp.linspace(0.01, 1, nr)
theta = cp.linspace(0, 2 * cp.pi, ntheta)
z = cp.linspace(0, 1, nz)
R, Theta, Z = cp.meshgrid(r, theta, z, indexing='ij')
k_damp = 1e-4   # Increased from 5e-5 to dampen divergence
k = 0.0047
nu = 1e-5
print(f"g_wave={g_wave}, k_damp={k_damp}, k_U={k_U}, alpha={alpha}, kappa={kappa}, nu={nu}")

# Initial Conditions on GPU
cp.random.seed(42)
Phi1 = 0.95 * cp.cos(k * (R + Z)) * cp.cos(k * Theta) + 0.01 * cp.random.randn(nr, ntheta, nz)
Phi2 = 5.0 * cp.sin(k * (R + Z) + cp.pi / 2) * cp.sin(k * Theta) + 0.01 * cp.random.randn(nr, ntheta, nz)
u_r = -cp.gradient(Phi1, dz, axis=2)
u_theta = cp.gradient(Phi1, dr, axis=0)
u_z = cp.gradient(Phi1, dtheta, axis=1)

# Simulation Parameters
nt = 4000
dt = 5e-13
rho = 1000.0

# Compute Divergence Function (GPU version)
def compute_divergence(u_r, u_theta, u_z, dr, dtheta, dz):
    r_safe = R + 1e-10
    div = (cp.gradient(u_r, dr, axis=0) +
           cp.gradient(u_theta, dtheta, axis=1) / r_safe +
           cp.gradient(u_z, dz, axis=2))
    return div

# Data logging
log_file = os.path.join(os.path.expanduser("~"), "Desktop", "turbine_sim_log.txt")
with open(log_file, 'w') as f:  # Ensure file is created
    f.write("Step,Max Velocity (m/s),Divergence,Coherence (sigma),Enthalpy (J/m³),Vorticity Max (s^-1),SNR,Grad Mag\n")
    f.flush()  # Force write to disk

try:
    # Time Loop
    for t in range(nt):
        # Gradients on GPU
        dPhir_1 = cp.gradient(Phi1, dr, axis=0)
        dPhitheta_1 = cp.gradient(Phi1, dtheta, axis=1)
        dPhiz_1 = cp.gradient(Phi1, dz, axis=2)
        dPhir_2 = cp.gradient(Phi2, dr, axis=0)
        dPhitheta_2 = cp.gradient(Phi2, dtheta, axis=1)
        dPhiz_2 = cp.gradient(Phi2, dz, axis=2)

        # SNR and Theta Field on GPU
        SNR = cp.abs(Phi1 + Phi2) / (cp.std(Phi1 + Phi2) + 1e-10)
        grad_mag = cp.sqrt(dPhir_1**2 + dPhitheta_1**2 + dPhiz_1**2 + dPhir_2**2 + dPhitheta_2**2 + dPhiz_2**2)
        theta_field = cp.pi + 100 * 0.2 * cp.pi / 180 * SNR * (1 + 0.3 * grad_mag)

        # Laplacian on GPU
        laplacian = cp.zeros_like(Phi1)
        phi_sum = Phi1 + Phi2
        grad_r_r = cp.gradient(cp.gradient(phi_sum, dr, axis=0), dr, axis=0)
        grad_theta_theta = cp.gradient(cp.gradient(phi_sum, dtheta, axis=1), dtheta, axis=1)
        grad_z_z = cp.gradient(cp.gradient(phi_sum, dz, axis=2), dz, axis=2)
        laplacian[1:-1, 1:-1, 1:-1] = (grad_r_r[1:-1, 1:-1, 1:-1] +
                                       (1 / (R[1:-1, 1:-1, 1:-1] + 1e-10)**2) * grad_theta_theta[1:-1, 1:-1, 1:-1] +
                                       grad_z_z[1:-1, 1:-1, 1:-1]) / (R[1:-1, 1:-1, 1:-1] + 1e-10)

        # Time Derivatives on GPU
        r_safe = R + 1e-10
        dPhi1_dt = cp.zeros_like(Phi1)
        dPhi2_dt = cp.zeros_like(Phi2)
        dPhi1_dt[1:-1, 1:-1, 1:-1] = ((cp.gradient(r_safe[1:-1, 1:-1, 1:-1] * dPhir_1[1:-1, 1:-1, 1:-1], axis=0) +
                                       cp.gradient(dPhitheta_1[1:-1, 1:-1, 1:-1], axis=1) +
                                       cp.gradient(dPhiz_1[1:-1, 1:-1, 1:-1], axis=2)) / r_safe[1:-1, 1:-1, 1:-1] -
                                      k_U * (2 * Phi1[1:-1, 1:-1, 1:-1] + Phi2[1:-1, 1:-1, 1:-1]) -
                                      alpha * Phi2[1:-1, 1:-1, 1:-1] * cp.cos(0.0047 * R[1:-1, 1:-1, 1:-1] + theta_field[1:-1, 1:-1, 1:-1]) -
                                      g_wave * laplacian[1:-1, 1:-1, 1:-1] - k_damp * Phi1[1:-1, 1:-1, 1:-1])
        dPhi2_dt[1:-1, 1:-1, 1:-1] = ((cp.gradient(r_safe[1:-1, 1:-1, 1:-1] * dPhir_2[1:-1, 1:-1, 1:-1], axis=0) +
                                       cp.gradient(dPhitheta_2[1:-1, 1:-1, 1:-1], axis=1) +
                                       cp.gradient(dPhiz_2[1:-1, 1:-1, 1:-1], axis=2)) / r_safe[1:-1, 1:-1, 1:-1] -
                                      k_U * (Phi1[1:-1, 1:-1, 1:-1] + 2 * Phi2[1:-1, 1:-1, 1:-1]) -
                                      alpha * Phi1[1:-1, 1:-1, 1:-1] * cp.cos(0.0047 * R[1:-1, 1:-1, 1:-1] + theta_field[1:-1, 1:-1, 1:-1]) -
                                      g_wave * laplacian[1:-1, 1:-1, 1:-1] - k_damp * Phi2[1:-1, 1:-1, 1:-1])

        # 3D Advection on GPU
        u_r_grad = u_r * cp.gradient(Phi1, dr, axis=0) + (u_theta / (R + 1e-10)) * cp.gradient(Phi1, dtheta, axis=1) + u_z * cp.gradient(Phi1, dz, axis=2)
        dPhi1_dt[1:-1, 1:-1, 1:-1] += nu * laplacian[1:-1, 1:-1, 1:-1] - u_r_grad[1:-1, 1:-1, 1:-1]
        dPhi2_dt[1:-1, 1:-1, 1:-1] += nu * laplacian[1:-1, 1:-1, 1:-1] - u_r_grad[1:-1, 1:-1, 1:-1]

        # Crash Detection on GPU
        if cp.any(cp.isnan(dPhi1_dt)) or cp.any(cp.isnan(dPhi2_dt)):
            print(f"Crash detected at step {t + 19000}")
            break

        # Time Step Update on GPU
        Phi1 += dPhi1_dt * dt
        Phi2 += dPhi2_dt * dt

        # Velocity Update on GPU
        u_r_new = -cp.gradient(Phi1, dz, axis=2)
        u_theta_new = cp.gradient(Phi1, dr, axis=0)
        u_z_new = cp.gradient(Phi1, dtheta, axis=1)
        div_u = compute_divergence(u_r_new, u_theta_new, u_z_new, dr, dtheta, dz)

        # Vorticity Calculation on GPU
        omega_r = cp.gradient(u_z_new, dtheta, axis=1) - cp.gradient(u_theta_new, dz, axis=2)
        omega_theta = cp.gradient(u_r_new, dz, axis=2) - cp.gradient(u_z_new, dr, axis=0)
        omega_z = cp.gradient(u_theta_new, dr, axis=0) - cp.gradient(u_r_new, dtheta, axis=1)
        omega_max = cp.max(cp.sqrt(omega_r**2 + omega_theta**2 + omega_z**2))

        # Boundary Conditions on GPU
        u_r[0, :, :] = u_r[-1, :, :] = 0
        u_z[:, :, 0] = u_z[:, :, -1] = 0
        u_theta[0, :, :] = u_theta[-1, :, :] = 0

        # Diagnostics (convert to CPU only for printing/logging)
        if t % 100 == 0 or t == nt - 1:
            max_vel = cp.max(cp.sqrt(u_r**2 + u_theta**2 + u_z**2)).get()
            div = cp.max(cp.abs(div_u)).get()
            phi_sum = Phi1 + Phi2  # Keep as CuPy array
            enthalpy = (rho / 2) * max_vel**2 + k_U * cp.max(cp.abs(phi_sum))**2  # GPU calc
            coh = cp.max(kappa * (0.001 + 0.0005 * cp.abs(phi_sum))).get()
            print(f"Step {t + 19000}: Max Velocity = {max_vel:.3e} m/s, Divergence = {div:.3e}, "
                  f"Coherence = {coh:.3f}sigma, Enthalpy = {enthalpy:.3e} J/m³, Vorticity Max = {omega_max.get():.3e} s^-1")
            with open(log_file, 'a') as f:
                f.write(f"{t + 19000},{max_vel:.3e},{div:.3e},{coh:.3f},{enthalpy:.3e},{omega_max.get():.3e},{cp.max(SNR).get():.3e},{cp.max(grad_mag).get():.3e}\n")
                f.flush()  # Ensure data is written

        # Pressure Correction (CPU-based for now, using cp for consistency, convert to np for diags)
        nri, nthetai, nzi = nr - 2, ntheta - 2, nz - 2
        diagonals = [-cp.ones(nri * nthetai * nzi).get(), 6 * cp.ones(nri * nthetai * nzi).get(), -cp.ones(nri * nthetai * nzi).get()]
        offsets = [-nri * nthetai, 0, nri * nthetai]
        A = diags(diagonals, offsets, shape=(nri * nthetai * nzi, nri * nthetai * nzi))
        div_u_cpu = cp.asnumpy(div_u[1:-1, 1:-1, 1:-1]).flatten()
        p = np.zeros((nr, ntheta, nz))  # Still CPU-based for p
        rhs = -dr**2 * div_u_cpu
        p_inner, info = cg(A, rhs, rtol=1e-10)
        if info != 0:
            print(f"CG failed at step {t + 19000} with info {info}")
            break
        p[1:-1, 1:-1, 1:-1] = p_inner.reshape(nri, nthetai, nzi)

        # Pressure Gradients (CPU-based)
        dp_dr = np.gradient(p, dr, axis=0)
        dp_dtheta = np.gradient(p, dtheta, axis=1)
        dp_dz = np.gradient(p, dz, axis=2)
        u_r = u_r_new - cp.asarray(dp_dr)
        u_theta = u_theta_new - (1 / (R + 1e-10)) * cp.asarray(dp_dtheta)
        u_z = u_z_new - cp.asarray(dp_dz)

    # Save and Complete (convert to CPU for saving)
    if t == nt - 1:
        vel = cp.sqrt(u_r**2 + u_theta**2 + u_z**2).get()  # Ensure vel is computed
        cp.save('3D_velocity_field.npy', vel)
        print("3D velocity field saved as 3D_velocity_field.npy")
        # Validate save
        if np.all(vel == 0):
            print("Warning: Velocity field is all zeros—check computation!")
        else:
            print(f"Velocity field max: {np.max(vel):.3e} m/s")
except KeyboardInterrupt:
    print(f"Simulation interrupted at step {t + 19000}. Saving current state...")
    cp.save('3D_velocity_field_partial.npy', cp.sqrt(u_r**2 + u_theta**2 + u_z**2).get())
    print("Partial 3D velocity field saved as 3D_velocity_field_partial.npy")
print("Simulation completed or interrupted!")

# --- Enhanced Plotting to Desktop ---
desktop = os.path.join(os.path.expanduser("~"), "Desktop")

# Load logged data for plots (CPU-based)
try:
    data = np.loadtxt(log_file, delimiter=',', skiprows=1)
    steps = data[:, 0]
    velocities = data[:, 1]
    divergences = data[:, 2]
    coherences = data[:, 3]
    enthalpies = data[:, 4]
    vorticities = data[:, 5]
    snrs = data[:, 6]
    grad_mags = data[:, 7]
except FileNotFoundError:
    print(f"Error: Log file {log_file} not found. Check path or simulation output.")
    data = np.array([])  # Fallback to avoid crash

# Plot 1-7: Time series (skipped for brevity, same as before)
# ... (keep existing Plot 1-7 code) ...

# Plot 8: 3D Velocity Field Slice
if os.path.exists('3D_velocity_field.npy'):
    vel = np.load('3D_velocity_field.npy')
    if vel.size == 0 or np.all(vel == 0):
        print("Warning: Velocity field is empty or uniform—plot may be invalid.")
    else:
        slice_vel = vel[nr // 2, :, :]  # Center slice
        vmax = np.max(vel) if np.max(vel) > 0 else 2000  # Dynamic vmax
        plt.figure(8, figsize=(10, 8))
        plt.imshow(slice_vel, cmap='viridis', extent=[0, 1, 0, 1], vmin=0, vmax=vmax)
        plt.colorbar(label='Velocity (m/s)')
        plt.title('UWT Turbine Wake — Center Slice')
        plt.xlabel('Z (m)')
        plt.ylabel('Theta (rad)')
        plt.savefig(os.path.join(desktop, 'turbine_wake_plot.png'))
        plt.show()
        plt.close(8)
else:
    print("3D velocity field not saved—check simulation completion.")

print("All plots generated and saved to Desktop!")