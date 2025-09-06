import numpy as np
from scipy.sparse.linalg import cg
from scipy.sparse import diags, kron, eye, lil_matrix
import json
print("Fusion simulation with Lorentz force, temperature diffusion, radiation losses, neutron transport, alpha particle heating, impurity transport, bootstrap current, gyrokinetic turbulence, MHD stability, neoclassical transport, pedestal physics, ELM modeling, divertor modeling, RF heating, neutral beam injection, pellet fueling, current drive, SOL modeling, plasma rotation, and fast particle effects starting...")

# Parameters (fusion-specific, plasma, with units)
rho = 1.0e-6  # Plasma density, kg/m^3
mu = 1e-3  # Dynamic viscosity, Pa·s
gamma = 5e-5  # Damping, s^-1
c_phi = 5e3  # Wave speed, m/s
k_U = 2e10  # Coupling constant, kg^-1 m^3 s^-2
kappa_base = 5e7  # Pressure coefficient, Pa
lambda_ = 2.51e-46  # Potential coefficient, kg m^3 s^-2
g_m = 0.01  # Coupling, dimensionless
v = 0.226 / 6.242e18  # Scalar field reference, kg
alpha = 0.1  # Feedback parameter, dimensionless
beta = 0.0025  # Feedback parameter, dimensionless
lambda_d = 0.004  # Damping length, dimensionless
q = 4e-4  # Charge density, C/m^3
B_theta = 0.4  # Toroidal magnetic field, T
kappa_T = 7.5e-3  # Thermal diffusivity, m^2/s
C_r = 8e-29  # Radiation loss coefficient, W m^3 K^-5/2
eta_alpha = 4e-12  # Alpha heating rate, J/m^3/s per m^-3
D = 7.5e-3  # Neutron diffusion coefficient, m^2/s
Sigma_a = 7.5e-4  # Neutron absorption cross-section, m^-1
D_Z = 4e-3  # Impurity diffusion coefficient, m^2/s
Sigma_Z = 8e-4  # Impurity loss rate, s^-1
k_p = 7.5e4  # Bootstrap current coefficient, A/m^2/Pa
epsilon_turb = 8e-4  # Gyrokinetic turbulence intensity, m/s^2
kappa_MHD = 1.2e5  # MHD stability coefficient, N/m^3
D_neo = 3e-3  # Neoclassical diffusion coefficient, m^2/s
kappa_ped = 1.5e4  # Pedestal transport barrier coefficient, m/s^2
alpha_ELM = 1.2e3  # ELM amplitude, m/s^2
p_crit = 1e7  # Critical pressure gradient for ELM, Pa/m
ped_width = 0.1  # Pedestal width, normalized radius
sigma_div = 1.2e3  # Divertor sink coefficient, s^-1
div_width = 0.05  # Divertor region width, normalized z
eta_RF = 1.2e5  # RF heating coefficient, W/m^3
eta_NBI = 1.5e5  # NBI coefficient, W/m^3 and kg/m^2/s^2
eta_pellet = 1.5e13  # Pellet fueling coefficient, m^-3/s
tau_pellet = 0.01  # Pellet injection period, s
eta_CD = 1.2e4  # Current drive coefficient, A/m^2
sigma_SOL = 1.5e3  # SOL decay coefficient, s^-1
SOL_width = 0.05  # SOL region width, normalized radius
eta_rot = 1.5e4  # Rotation coefficient, m/s^2
eta_fast = 1.5e5  # Fast particle coefficient, W/m^3 and kg/m^2/s^2
N_fast_0 = 1e14  # Initial fast particle density, m^-3
Z_0 = 1e14  # Initial impurity density, m^-3
T_0 = 1e7  # Initial temperature, K
N_0 = 1e15  # Initial neutron density, m^-3
nr, ntheta, nz = 32, 16, 16  # Debug grid size (use 128, 64, 64 for full run)
dr = (1 - 0.01) / (nr - 1)  # Radial step, normalized
dtheta = 2 * np.pi / ntheta  # Angular step, radians
dz = 1.0 / (nz - 1)  # Axial step, normalized
r = np.linspace(0.01, 1, nr)
theta_grid = np.linspace(0, 2 * np.pi, ntheta)
z = np.linspace(0, 1, nz)
R, Theta, Z_grid = np.meshgrid(r, theta_grid, z, indexing='ij')
phi_scale = 7.15e6  # Scalar field amplitude, dimensionless
phase_shift = 340 * np.pi / 180  # Phase shift, radians
dt = 1e-4  # Base time step, s
t_total = 0  # Total simulation time, s
rng = np.random.RandomState(42)  # For reproducibility

# Initial Conditions
amp_factor = rng.uniform(0.8, 1.2, size=(nr, ntheta, nz))
Phi1 = 4.03e-28 * phi_scale * amp_factor * (np.cos(0.00235 * (R + Z_grid) + phase_shift) * np.cos(0.00235 * Theta + phase_shift) + 0.01 * rng.randn(nr, ntheta, nz))
Phi2 = 1.68e-28 * phi_scale * amp_factor * (np.sin(0.00235 * (R + Z_grid) + np.pi/2 + phase_shift) * np.sin(0.00235 * Theta + phase_shift) + 0.01 * rng.randn(nr, ntheta, nz))
Phi1_prev = Phi1.copy()
Phi2_prev = Phi2.copy()
u_r = 1e5 * np.ones((nr, ntheta, nz)) + 0.01 * rng.randn(nr, ntheta, nz)  # Initial velocity ~10^5 m/s
u_theta = np.zeros((nr, ntheta, nz))  # Initial toroidal velocity
u_z = np.zeros((nr, ntheta, nz))  # Initial poloidal velocity
T = T_0 * np.ones((nr, ntheta, nz))  # Initial temperature field
N = N_0 * np.ones((nr, ntheta, nz))  # Initial neutron density field
impurity = Z_0 * np.ones((nr, ntheta, nz))  # Initial impurity density field
N_fast = N_fast_0 * np.ones((nr, ntheta, nz))  # Initial fast particle density field

# Data for Sidebar Visualization
plot_data = {"steps": [], "enthalpy": [], "velocity": [], "divergence": [], "temperature": [], "neutron_density": [], "impurity_density": [], "fast_particle_density": []}

# Helper Functions
def compute_divergence(u_r, u_theta, u_z, dr, dtheta, dz, R):
    r_safe = R + 1e-6
    div_r = (r_safe[1:-1, :, :] * u_r[1:-1, :, :])[:, :, :]  # r * u_r
    div_r = (div_r[1:, :, :] - div_r[:-1, :, :]) / dr / r_safe[1:-1, :, :]  # Aligned slices
    div_theta = (u_theta[:, 1:-1, :] - u_theta[:, :-2, :]) / (dtheta * r_safe[:, 1:-1, :])
    div_z = (u_z[:, :, 1:-1] - u_z[:, :, :-2]) / (2 * dz)
    div = np.zeros_like(u_r)
    div[1:-1, 1:-1, 1:-1] = div_r[:, 1:-1, 1:-1] + div_theta[:, :, 1:-1] + div_z[1:-1, 1:-1, :]
    return div

def laplacian_cylindrical(f, R, dr, dtheta, dz):
    invR = 1.0 / (R + 1e-6)
    lap = np.zeros_like(f)
    # Radial term: (1/r) * d/dr(r * df/dr)
    df_dr = (f[2:, :, :] - f[:-2, :, :]) / (2 * dr)
    rdf_dr = R[1:-1, :, :] * df_dr
    lap_r = (rdf_dr[1:, :, :] - rdf_dr[:-1, :, :]) / (dr * R[1:-1, :, :])
    lap[1:-1, :, :] = lap_r
    # Theta term: (1/r^2) * d^2f/dθ^2
    lap_theta = (f[:, 2:, :] - 2 * f[:, 1:-1, :] + f[:, :-2, :]) / ((R[:, 1:-1, :] * dtheta) ** 2)
    lap[:, 1:-1, :] += lap_theta
    # Z term: d^2f/dz^2
    lap_z = (f[:, :, 2:] - 2 * f[:, :, 1:-1] + f[:, :, :-2]) / (dz ** 2)
    lap[:, :, 1:-1] += lap_z
    return lap

def compute_dt(u_r, u_theta, u_z, R, dr, dtheta, dz, c_phi, nu, dt_base):
    u_max = np.max(np.sqrt(u_r**2 + u_theta**2 + u_z**2))
    r_min = np.min(R)
    C_cfl, C_diff, C_wave = 0.3, 0.25, 0.3
    dt_adv = C_cfl * min(dr / (u_max + 1e-12), (r_min * dtheta) / (u_max + 1e-12), dz / (u_max + 1e-12))
    dt_diff = C_diff * min(dr**2 / (nu + 1e-12), (r_min * dtheta)**2 / (nu + 1e-12), dz**2 / (nu + 1e-12))
    dt_wave = C_wave * min(dr / c_phi, (r_min * dtheta) / c_phi, dz / c_phi)
    return max(1e-12, min(dt_base, dt_adv, dt_diff, dt_wave))

def build_poisson_matrix(nr, ntheta, nz, dr, dtheta, dz, R):
    nri, nthetai, nzi = nr - 2, ntheta, nz - 2  # Dirichlet in r, z; periodic in theta
    e_r = np.ones(nri)
    Lr = diags([e_r, -2 * e_r, e_r], offsets=[-1, 0, 1], shape=(nri, nri)) / dr**2
    e_theta = np.ones(ntheta)
    Ltheta = lil_matrix((ntheta, ntheta))
    Ltheta.setdiag(-2 * e_theta / (R[1:-1, :, 1:-1].mean() * dtheta)**2)
    Ltheta[1:, :-1] += diags(e_theta[:-1], offsets=1) / (R[1:-1, :, 1:-1].mean() * dtheta)**2
    Ltheta[:-1, 1:] += diags(e_theta[:-1], offsets=1) / (R[1:-1, :, 1:-1].mean() * dtheta)**2
    Ltheta[0, -1] = Ltheta[-1, 0] = 1 / (R[1:-1, :, 1:-1].mean() * dtheta)**2  # Periodic BCs
    Ltheta = Ltheta.tocsr()  # Convert to CSR for efficient solving
    e_z = np.ones(nzi)
    Lz = diags([e_z, -2 * e_z, e_z], offsets=[-1, 0, 1], shape=(nzi, nzi)) / dz**2
    A = kron(kron(Lr, eye(ntheta)), eye(nzi)) + kron(kron(eye(nri), Ltheta), eye(nzi)) + kron(kron(eye(nri), eye(ntheta)), Lz)
    return A.tocsr()

try:
    # Initial plot data at step 0
    enthalpy = kappa_base * np.mean(np.abs(Phi1 * Phi2))
    max_vel = np.max(np.sqrt(u_r**2 + u_theta**2 + u_z**2))
    div = np.max(np.abs(compute_divergence(u_r, u_theta, u_z, dr, dtheta, dz, R)))
    avg_temp = np.mean(T)
    avg_neutron = np.mean(N)
    avg_impurity = np.mean(impurity)
    avg_fast = np.mean(N_fast)
    plot_data["steps"].append(0)
    plot_data["enthalpy"].append(enthalpy)
    plot_data["velocity"].append(max_vel)
    plot_data["divergence"].append(div)
    plot_data["temperature"].append(avg_temp)
    plot_data["neutron_density"].append(avg_neutron)
    plot_data["impurity_density"].append(avg_impurity)
    plot_data["fast_particle_density"].append(avg_fast)
    with open("fusion_plot_data_step0.json", "w") as f:
        json.dump(plot_data, f)
    np.savez("fusion_arrays_step0.npz", T=T, N=N, impurity=impurity, N_fast=N_fast)

    A_poisson = build_poisson_matrix(nr, ntheta, nz, dr, dtheta, dz, R)

    for t in range(1000):
        dt_adaptive = compute_dt(u_r, u_theta, u_z, R, dr, dtheta, dz, c_phi, mu / rho, dt)
        t_total += dt_adaptive
        # Precompute gradients
        gradients = {
            'Phi1_r': (Phi1[2:, :, :] - Phi1[:-2, :, :]) / (2 * dr),
            'Phi1_theta': (Phi1[:, 2:, :] - Phi1[:, :-2, :]) / (2 * dtheta),
            'Phi1_z': (Phi1[:, :, 2:] - Phi1[:, :, :-2]) / (2 * dz),
            'Phi2_r': (Phi2[2:, :, :] - Phi2[:-2, :, :]) / (2 * dr),
            'Phi2_theta': (Phi2[:, 2:, :] - Phi2[:, :-2, :]) / (2 * dtheta),
            'Phi2_z': (Phi2[:, :, 2:] - Phi2[:, :, :-2]) / (2 * dz),
            'p_r': None, 'p_theta': None, 'p_z': None
        }
        phi_prod = Phi1 * Phi2
        norm = np.max(np.abs(phi_prod))
        feedback = np.exp(-np.abs(R) / lambda_d)
        phi1_phi2 = np.abs(phi_prod) * feedback
        V = lambda_ * ((phi_prod**2 - v**2)**2) + 0.5 * k_U * (2 * Phi1**2 + Phi1 * Phi2 + 2 * Phi2**2)
        dV_dPhi1 = lambda_ * 4 * phi_prod * (phi_prod**2 - v**2) * Phi2 + k_U * (4 * Phi1 + Phi2)
        dV_dPhi2 = lambda_ * 4 * phi_prod * (phi_prod**2 - v**2) * Phi1 + k_U * (Phi1 + 4 * Phi2)
        laplacian_phi1 = laplacian_cylindrical(Phi1, R, dr, dtheta, dz)
        laplacian_phi2 = laplacian_cylindrical(Phi2, R, dr, dtheta, dz)
        dPhi1_dt = c_phi**2 * laplacian_phi1 - dV_dPhi1 - g_m * rho * Phi2 - gamma * (Phi1 - Phi1_prev) / dt_adaptive
        dPhi2_dt = c_phi**2 * laplacian_phi2 - dV_dPhi2 - g_m * rho * Phi1 - gamma * (Phi2 - Phi2_prev) / dt_adaptive
        if np.any(np.isnan(dPhi1_dt)) or np.any(np.isnan(dPhi2_dt)):
            print(f"Crash at step {t}: NaN in dPhi1_dt or dPhi2_dt")
            break
        Phi1_prev = Phi1.copy()
        Phi2_prev = Phi2.copy()
        Phi1 += dt_adaptive * dPhi1_dt
        Phi2 += dt_adaptive * dPhi2_dt
        # Plasma pressure with pedestal enhancement
        p_core = kappa_base * phi1_phi2
        p_ped = p_core * (1 + 5 * np.exp(-(1 - R)**2 / (2 * ped_width**2)))
        gradients['p_r'] = (p_ped[2:, :, :] - p_ped[:-2, :, :]) / (2 * dr)
        gradients['p_theta'] = (p_ped[:, 2:, :] - p_ped[:, :-2, :]) / (2 * dtheta)
        gradients['p_z'] = (p_ped[:, :, 2:] - p_ped[:, :, :-2]) / (2 * dz)
        # Bootstrap current
        J_bs = np.zeros((nr, ntheta, nz, 3))
        B_sq = B_theta**2 + 1e-6
        J_bs[:, :, :, 0] = k_p * (-gradients['p_z'] * B_theta) / B_sq
        J_bs[:, :, :, 2] = k_p * (gradients['p_r'] * B_theta) / B_sq
        # Current drive
        J_CD = np.zeros((nr, ntheta, nz, 3))
        CD_factor = np.exp(-R**2 / (2 * 0.5**2))
        J_CD[:, :, :, 2] = eta_CD * phi1_phi2 * CD_factor
        # Total Lorentz force
        lorentz_force = np.zeros((nr, ntheta, nz, 3))
        lorentz_force[:, :, :, 0] = q * (u_z * B_theta) + (J_bs[:, :, :, 2] + J_CD[:, :, :, 2]) * B_theta
        lorentz_force[:, :, :, 2] = -q * (u_r * B_theta) - (J_bs[:, :, :, 0] + J_CD[:, :, :, 0]) * B_theta
        # Plasma rotation
        rot_factor = np.exp(-R**2 / (2 * 0.5**2))
        rot_force = np.zeros((nr, ntheta, nz, 3))
        rot_force[:, :, :, 1] = eta_rot * phi1_phi2 * rot_factor  # Toroidal rotation
        rot_force[:, :, :, 2] = 0.1 * eta_rot * phi1_phi2 * rot_factor  # Poloidal rotation
        # Fast particle effects
        fast_factor = np.exp(-R**2 / (2 * 0.5**2))
        fast_force = np.zeros((nr, ntheta, nz, 3))
        fast_force[:, :, :, 2] = eta_fast * phi1_phi2 * (N_fast / N_fast_0) * fast_factor  # Coupled to N_fast
        fast_heating = eta_fast * phi1_phi2 * (N_fast / N_fast_0) * fast_factor
        # Gyrokinetic turbulence
        turb_force = np.zeros((nr, ntheta, nz, 3))
        turb_force[:, :, :, 0] = epsilon_turb * rng.randn(nr, ntheta, nz) * np.abs(gradients['p_r']) / (rho + 1e-6)
        turb_force[:, :, :, 1] = epsilon_turb * rng.randn(nr, ntheta, nz) * np.abs(gradients['p_theta']) / ((R + 1e-6) * (rho + 1e-6))
        turb_force[:, :, :, 2] = epsilon_turb * rng.randn(nr, ntheta, nz) * np.abs(gradients['p_z']) / (rho + 1e-6)
        # MHD stability
        mhd_force = np.zeros((nr, ntheta, nz, 3))
        u_mag = np.sqrt(u_r**2 + u_theta**2 + u_z**2 + 1e-6)
        mhd_force[:, :, :, 0] = -kappa_MHD * u_r / u_mag
        mhd_force[:, :, :, 1] = -kappa_MHD * u_theta / u_mag
        mhd_force[:, :, :, 2] = -kappa_MHD * u_z / u_mag
        # Neoclassical transport
        neo_diffusion = np.zeros((nr, ntheta, nz, 3))
        laplacian_ur = laplacian_cylindrical(u_r, R, dr, dtheta, dz)
        laplacian_utheta = laplacian_cylindrical(u_theta, R, dr, dtheta, dz)
        laplacian_uz = laplacian_cylindrical(u_z, R, dr, dtheta, dz)
        neo_diffusion[:, :, :, 0] = D_neo * laplacian_ur
        neo_diffusion[:, :, :, 1] = D_neo * laplacian_utheta
        neo_diffusion[:, :, :, 2] = D_neo * laplacian_uz
        # Pedestal transport barrier
        ped_factor = 1 - 0.9 * np.exp(-(1 - R)**2 / (2 * ped_width**2))
        ped_force = np.zeros((nr, ntheta, nz, 3))
        ped_force[:, :, :, 0] = -kappa_ped * u_r * ped_factor
        ped_force[:, :, :, 1] = -kappa_ped * u_theta * ped_factor
        ped_force[:, :, :, 2] = -kappa_ped * u_z * ped_factor
        # ELM modeling
        elm_force = np.zeros((nr, ntheta, nz, 3))
        dp_mag = np.sqrt(gradients['p_r']**2 + gradients['p_theta']**2 + gradients['p_z']**2 + 1e-6)
        elm_trigger = (dp_mag > p_crit) * np.exp(-(1 - R)**2 / (2 * ped_width**2))
        elm_force[:, :, :, 0] = alpha_ELM * elm_trigger * rng.randn(nr, ntheta, nz) * np.abs(gradients['p_r']) / (rho + 1e-6)
        elm_force[:, :, :, 1] = alpha_ELM * elm_trigger * rng.randn(nr, ntheta, nz) * np.abs(gradients['p_theta']) / ((R + 1e-6) * (rho + 1e-6))
        elm_force[:, :, :, 2] = alpha_ELM * elm_trigger * rng.randn(nr, ntheta, nz) * np.abs(gradients['p_z']) / (rho + 1e-6)
        # Divertor modeling
        div_factor = np.exp(-(Z_grid**2 / (2 * div_width**2)) + -((1 - Z_grid)**2 / (2 * div_width**2)))
        # Neutral beam injection
        NBI_factor = np.exp(-R**2 / (2 * 0.5**2))
        NBI_momentum = np.zeros((nr, ntheta, nz, 3))
        NBI_momentum[:, :, :, 2] = eta_NBI * phi1_phi2 * NBI_factor
        # SOL modeling
        SOL_factor = np.exp(-((R - 0.95)**2 / (2 * SOL_width**2))) * (R > 0.9)
        # Momentum equations
        u_grad_u_r = u_r * (u_r[2:, :, :] - u_r[:-2, :, :]) / (2 * dr) + u_theta * (u_r[:, 2:, :] - u_r[:, :-2, :]) / (2 * dtheta * (R + 1e-6)) + u_z * (u_r[:, :, 2:] - u_r[:, :, :-2]) / (2 * dz)
        u_grad_u_theta = u_r * (u_theta[2:, :, :] - u_theta[:-2, :, :]) / (2 * dr) + u_theta * (u_theta[:, 2:, :] - u_theta[:, :-2, :]) / (2 * dtheta * (R + 1e-6)) + u_z * (u_theta[:, :, 2:] - u_theta[:, :, :-2]) / (2 * dz)
        u_grad_u_z = u_r * (u_z[2:, :, :] - u_z[:-2, :, :]) / (2 * dr) + u_theta * (u_z[:, 2:, :] - u_z[:, :-2, :]) / (2 * dtheta * (R + 1e-6)) + u_z * (u_z[:, :, 2:] - u_z[:, :, :-2]) / (2 * dz)
        phi_prod_grad = np.zeros((nr, ntheta, nz, 3))
        phi_prod_grad[:, :, :, 0] = (g_m * phi_prod[2:, :, :] - g_m * phi_prod[:-2, :, :]) / (2 * dr)
        phi_prod_grad[:, :, :, 1] = (g_m * phi_prod[:, 2:, :] - g_m * phi_prod[:, :-2, :]) / (2 * dtheta * (R + 1e-6))
        phi_prod_grad[:, :, :, 2] = (g_m * phi_prod[:, :, 2:] - g_m * phi_prod[:, :, :-2]) / (2 * dz)
        body_force = -rho * phi_prod_grad
        u_r_new = u_r + dt_adaptive * (-u_grad_u_r + mu * laplacian_ur + body_force[:, :, :, 0] + lorentz_force[:, :, :, 0] + turb_force[:, :, :, 0] + mhd_force[:, :, :, 0] + neo_diffusion[:, :, :, 0] + ped_force[:, :, :, 0] + elm_force[:, :, :, 0] + NBI_momentum[:, :, :, 0] + rot_force[:, :, :, 0] + fast_force[:, :, :, 0]) / (rho + 1e-6)
        u_theta_new = u_theta + dt_adaptive * (-u_grad_u_theta + mu * laplacian_utheta + body_force[:, :, :, 1] + lorentz_force[:, :, :, 1] + turb_force[:, :, :, 1] + mhd_force[:, :, :, 1] + neo_diffusion[:, :, :, 1] + ped_force[:, :, :, 1] + elm_force[:, :, :, 1] + NBI_momentum[:, :, :, 1] + rot_force[:, :, :, 1] + fast_force[:, :, :, 1]) / (rho + 1e-6)
        u_z_new = u_z + dt_adaptive * (-u_grad_u_z + mu * laplacian_uz + body_force[:, :, :, 2] + lorentz_force[:, :, :, 2] + turb_force[:, :, :, 2] + mhd_force[:, :, :, 2] + neo_diffusion[:, :, :, 2] + ped_force[:, :, :, 2] + elm_force[:, :, :, 2] + NBI_momentum[:, :, :, 2] + rot_force[:, :, :, 2] + fast_force[:, :, :, 2]) / (rho + 1e-6)
        # Temperature diffusion
        laplacian_T = laplacian_cylindrical(T, R, dr, dtheta, dz)
        source_T = 1e5 * phi1_phi2
        radiation_loss = -C_r * T**2.5
        alpha_heating = eta_alpha * N
        div_sink_T = -sigma_div * T * div_factor
        RF_heating = eta_RF * phi1_phi2 * np.exp(-R**2 / (2 * 0.5**2))
        NBI_heating = eta_NBI * phi1_phi2 * NBI_factor
        SOL_decay_T = -sigma_SOL * T * SOL_factor
        dT_dt = kappa_T * laplacian_T + source_T + radiation_loss + alpha_heating + div_sink_T + RF_heating + NBI_heating + SOL_decay_T + fast_heating
        if np.any(np.isnan(dT_dt)):
            print(f"Crash at step {t}: NaN in dT_dt")
            break
        T = np.clip(T + dt_adaptive * dT_dt, 1e5, 1e9)  # Prevent negative/unphysical temperatures
        # Neutron transport
        laplacian_N = laplacian_cylindrical(N, R, dr, dtheta, dz)
        source_N = 1e13 * phi1_phi2
        pellet_source = eta_pellet * phi1_phi2 * np.sin(2 * np.pi * t_total / tau_pellet) * np.exp(-R**2 / (2 * 0.5**2))
        absorption_N = -Sigma_a * N
        div_sink_N = -sigma_div * N * div_factor
        SOL_decay_N = -sigma_SOL * N * SOL_factor
        dN_dt = D * laplacian_N + source_N + absorption_N + div_sink_N + pellet_source + SOL_decay_N
        if np.any(np.isnan(dN_dt)):
            print(f"Crash at step {t}: NaN in dN_dt")
            break
        N = np.clip(N + dt_adaptive * dN_dt, 1e12, 1e16)  # Prevent negative/unphysical density
        # Impurity transport
        laplacian_Z = laplacian_cylindrical(impurity, R, dr, dtheta, dz)
        source_Z = 1e12 * phi1_phi2
        loss_Z = -Sigma_Z * impurity
        div_sink_Z = -sigma_div * impurity * div_factor
        SOL_decay_Z = -sigma_SOL * impurity * SOL_factor
        dZ_dt = D_Z * laplacian_Z + source_Z + loss_Z + div_sink_Z + SOL_decay_Z
        if np.any(np.isnan(dZ_dt)):
            print(f"Crash at step {t}: NaN in dZ_dt")
            break
        impurity = np.clip(impurity + dt_adaptive * dZ_dt, 1e12, 1e16)  # Prevent negative/unphysical density
        # Fast particle transport
        laplacian_N_fast = laplacian_cylindrical(N_fast, R, dr, dtheta, dz)
        source_N_fast = eta_fast * phi1_phi2 * (N_fast / N_fast_0) * fast_factor
        loss_N_fast = -Sigma_a * N_fast
        div_sink_N_fast = -sigma_div * N_fast * div_factor
        SOL_decay_N_fast = -sigma_SOL * N_fast * SOL_factor
        dN_fast_dt = D * laplacian_N_fast + source_N_fast + loss_N_fast + div_sink_N_fast + SOL_decay_N_fast
        if np.any(np.isnan(dN_fast_dt)):
            print(f"Crash at step {t}: NaN in dN_fast_dt")
            break
        N_fast = np.clip(N_fast + dt_adaptive * dN_fast_dt, 1e12, 1e16)  # Prevent negative/unphysical density
        # Pressure Poisson
        div_u = compute_divergence(u_r_new, u_theta_new, u_z_new, dr, dtheta, dz, R)
        p = np.zeros((nr, ntheta, nz))
        rhs = -dr**2 * div_u[1:-1, :, 1:-1].flatten()
        p_inner, info = cg(A_poisson, rhs, tol=1e-17)
        if info != 0:
            print(f"CG failed at step {t} with info {info}")
            break
        p[1:-1, :, 1:-1] = p_inner.reshape(nr-2, ntheta, nz-2)
        gradients['p_r'] = (p[2:, :, :] - p[:-2, :, :]) / (2 * dr)
        gradients['p_theta'] = (p[:, 2:, :] - p[:, :-2, :]) / (2 * dtheta)
        gradients['p_z'] = (p[:, :, 2:] - p[:, :, :-2]) / (2 * dz)
        u_r = u_r_new - dt_adaptive * gradients['p_r'] / (rho + 1e-6)
        u_theta = u_theta_new - dt_adaptive * gradients['p_theta'] / ((rho + 1e-6) * (R + 1e-6))
        u_z = u_z_new - dt_adaptive * gradients['p_z'] / (rho + 1e-6)
        # Neumann BCs in SOL region (r > 0.9)
        mask_SOL = R > 0.9
        u_r[mask_SOL] = u_r[np.maximum(1, np.where(mask_SOL)[0] - 1)]  # Zero gradient at r=1
        u_theta[mask_SOL] = u_theta[np.maximum(1, np.where(mask_SOL)[0] - 1)]
        u_z[mask_SOL] = u_z[np.maximum(1, np.where(mask_SOL)[0] - 1)]
        u_r[0, :, :] = u_r[1, :, :]  # Zero gradient at r=0.01
        u_theta[0, :, :] = u_theta[1, :, :]
        u_z[0, :, :] = u_z[1, :, :]
        u_z[:, :, 0] = u_z[:, :, 1]  # Zero gradient at z=0
        u_z[:, :, -1] = u_z[:, :, -2]  # Zero gradient at z=1
        enthalpy = kappa_base * np.mean(np.abs(Phi1 * Phi2))
        if t % 200 == 0 or t == 999:
            max_vel = np.max(np.sqrt(u_r**2 + u_theta**2 + u_z**2))
            div = np.max(np.abs(div_u))
            avg_temp = np.mean(T)
            avg_neutron = np.mean(N)
            avg_impurity = np.mean(impurity)
            avg_fast = np.mean(N_fast)
            print(f"Step {t}: Enthalpy = {enthalpy:.3e} J/m^3, Max Velocity = {max_vel:.2f} m/s, Divergence = {div:.4f}, Temperature = {avg_temp:.2e} K, Neutron Density = {avg_neutron:.2e} m^-3, Impurity Density = {avg_impurity:.2e} m^-3, Fast Particle Density = {avg_fast:.2e} m^-3")
            plot_data["steps"].append(t)
            plot_data["enthalpy"].append(enthalpy)
            plot_data["velocity"].append(max_vel)
            plot_data["divergence"].append(div)
            plot_data["temperature"].append(avg_temp)
            plot_data["neutron_density"].append(avg_neutron)
            plot_data["impurity_density"].append(avg_impurity)
            plot_data["fast_particle_density"].append(avg_fast)
            with open(f"fusion_plot_data_step{t}.json", "w") as f:
                json.dump(plot_data, f)
            np.savez(f"fusion_arrays_step{t}.npz", T=T, N=N, impurity=impurity, N_fast=N_fast)
except Exception as e:
    print(f"Simulation crashed at step {t}: {str(e)}")
    with open(f"fusion_plot_data_step{t}.json", "w") as f:
        json.dump(plot_data, f)
    np.savez(f"fusion_arrays_step{t}.npz", T=T, N=N, impurity=impurity, N_fast=N_fast)
print("Fusion simulation completed or interrupted!")
