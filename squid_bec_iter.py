import numpy as np
import os

# Parameters
L, dx = 1.0, 0.0001
x = np.arange(-1, 1 + dx, dx)
t_steps = 2000
k_wave = 0.00235
epsilons = [1.0]
phi1 = 0.095 * np.exp(-(x / L)**2)  # Increased phi1 amplitude
phi2 = 0.5 * np.sin(k_wave * x)
lambda_d, mass = 0.004, 0.001
alpha = 10
beta = 0.01
results = []
target_dmm = 1e-3

# Run for k_wave and epsilon
try:
    for epsilon in epsilons:
        phi1_temp = phi1.copy()
        phi2_temp = phi2.copy()
        delta_m = []
        for t in range(t_steps):
            grad_phi1 = np.gradient(phi1_temp, dx)
            grad_phi2 = np.gradient(phi2_temp, dx)
            norm = np.max(np.abs(phi1_temp * phi2_temp))
            dt = 0.0001 / (1 + norm / 10) if norm > 0 else 0.0001
            phi1_new = phi1_temp + dt * (-0.001 * grad_phi2 * phi1_temp + alpha * phi1_temp * phi2_temp * np.cos(k_wave * np.abs(x)))
            phi2_new = phi2_temp + dt * (-0.001 * grad_phi1 * phi2_temp + alpha * phi1_temp * phi2_temp * np.cos(k_wave * np.abs(x)))
            phi1_temp = np.clip(phi1_new, -1e5, 1e5)
            phi2_temp = np.clip(phi2_new, -1e5, 1e5)
            feedback = np.exp(-np.abs(x) / lambda_d)
            phi1_phi2 = np.abs(phi1_temp * phi2_temp) * feedback
            delta_m_t = epsilon * phi1_phi2**2 * mass
            delta_m.append(np.mean(delta_m_t))
        dmm = np.mean(delta_m) / mass
        results.append((k_wave, epsilon, dmm))
        print(f"k_wave={k_wave:.5f}, epsilon={epsilon:.0e}, Average delta_m/m: {dmm}")
        if dmm >= target_dmm:
            print(f"Target Δm/m ≥ {target_dmm} reached!")
            break

    # Save Δm/m results
    output_file = 'squid_bec_results.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        for k_wave, epsilon, dmm in results:
            f.write(f"k_wave={k_wave:.5f}, epsilon={epsilon:.0e}, Average delta_m/m: {dmm}\n")
    print(f"Results saved to {output_file}")

except Exception as e:
    print(f"Error: {e}")