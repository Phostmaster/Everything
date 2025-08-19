import numpy as np
import os

# Parameters for antigravity (760 Starship lifts, no tunnel)
L = 1.0
dx = 0.0001
x = np.arange(-1, 1 + dx, dx)
t_steps = 2000
epsilons = [0.9115]
phi1 = 12 * np.exp(-(x / L)**2)
phi2 = 0.5 * np.ones_like(x)  # No tunnel, uniform phi2
lambda_d = 0.004
mass = 0.001
alpha = 1000.0  # Max coupling
beta = 0.0025
eta = 1e9  # Capped energy density (J/m³)
f_ALD = 2.0  # Max deposition
k_damp = 0.001  # Damping coefficient
dt = 0.01  # Time step
mu = 1e-40  # Magnetic moment
results = []
target_dmm = -1e-3  # Negative for antigravity

# Run for epsilon
try:
    for epsilon in epsilons:
        phi1_temp = phi1.copy()
        phi2_temp = phi2.copy()
        delta_m = []
        energy_out = []
        print(f"Initial phi1 max={np.max(np.abs(phi1)):.2e}")
        for t in range(t_steps):
            grad_phi1 = np.gradient(phi1_temp, dx)
            grad_phi2 = np.gradient(phi2_temp, dx)
            norm = np.max(np.abs(phi1_temp * phi2_temp))
            dt_adaptive = dt / (1 + norm / 10) if norm > 0 else dt
            phi1_new = phi1_temp + dt_adaptive * (-k_damp * grad_phi2 * phi1_temp - alpha * phi1_temp * phi2_temp * f_ALD)  # No tunnel term
            phi2_new = phi2_temp + dt_adaptive * (-k_damp * grad_phi1 * phi2_temp - alpha * phi1_temp * phi2_temp * f_ALD)
            phi1_temp = phi1_new
            phi2_temp = phi2_new
            feedback = np.exp(-np.abs(x) / lambda_d)
            phi1_phi2 = np.abs(phi1_temp * phi2_temp) * feedback
            delta_m_t = epsilon * phi1_phi2**2 * mass * (eta / 1e9) * -1
            energy_t = eta * phi1_phi2 * f_ALD
            delta_m.append(np.mean(delta_m_t))
            energy_out.append(np.mean(energy_t))
            if t % 500 == 0:
                print(f"t={t}, phi1_temp max={np.max(np.abs(phi1_temp)):.2e}, phi1_phi2 mean={np.mean(phi1_phi2):.2e}, delta_m_t mean={np.mean(delta_m_t):.2e}, energy mean={np.mean(energy_t):.2e}")
        dmm = np.mean(delta_m) / mass
        energy_mean = np.mean(energy_out)
        results.append((epsilon, dmm, energy_mean))
        print(f"epsilon={epsilon:.4f}, Average delta_m/m: {dmm}, Energy: {energy_mean:.2e} J/m³")
        if dmm <= target_dmm:
            print(f"Target antigravity Δm/m <= {target_dmm} reached!")
            break

    # Save results
    output_file = 'squid_bec_antigrav_760x_no_tunnel_results.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        for epsilon, dmm, energy in results:
            f.write(f"epsilon={epsilon:.4f}, Average delta_m/m: {dmm}, Energy: {energy:.2e} J/m³\n")
    print(f"Results saved to {output_file}")

except Exception as e:
    print(f"Error: {e}")