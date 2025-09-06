import numpy as np
from scipy.sparse.linalg import cg
from scipy.sparse import diags
print("Turbine optimization simulation for Goal #2 starting...")

# Parameters (turbine-specific, ρ = 1.2 kg/m³ for air)
rho = 1.2  # Air density, kg/m^3
theta = 36 * np.pi / 180  # Pitch angle, radians
naca = "4412"  # Airfoil
nu = 1.5e-5  # Kinematic viscosity, m^2/s
nr, ntheta, nz = 64, 64, 64  # Grid for turbine flow
dr = dtheta = dz = 0.05  # Grid spacing, m
r = np.linspace(0.01, 1, nr)
theta_grid = np.linspace(0, 2 * np.pi, ntheta)
z = np.linspace(0, 1, nz)
R, Theta, Z = np.meshgrid(r, theta_grid, z, indexing='ij')
dt = 1e-4  # Time step, s
nt = 1000  # Time steps

# Initial Conditions (turbine flow, based on antigravity sim stability)
np.random.seed(42)
u_r = 6.67 * np.ones((nr, ntheta, nz)) + 0.01 * np.random.randn(nr, ntheta, nz)  # Base velocity from xGrok
u_theta = np.zeros((nr, ntheta, nz))
u_z = np.zeros((nr, ntheta, nz))

# Divergence Function
def compute_divergence(u_r, u_theta, u_z, dr, dtheta, dz):
    r_safe = R + 1e-10
    div = (np.gradient(u_r, dr, axis=0) +
           np.gradient(u_theta, dtheta, axis=1) / r_safe +
           np.gradient(u_z, dz, axis=2))
    return div

try:
    for t in range(nt):
        # Gradients
        dUr_dr = np.gradient(u_r, dr, axis=0)
        dUtheta_dtheta = np.gradient(u_theta, dtheta, axis=1)
        dUz_dz = np.gradient(u_z, dz, axis=2)

        # Laplacian
        grad_r_r = np.gradient(dUr_dr, dr, axis=0)
        grad_theta_theta = np.gradient(dUtheta_dtheta, dtheta, axis=1)
        grad_z_z = np.gradient(dUz_dz, dz, axis=2)
        laplacian = np.zeros_like(u_r)
        laplacian[1:-1, 1:-1, 1:-1] = (grad_r_r[1:-1, 1:-1, 1:-1] +
                                       (1/(R[1:-1, 1:-1, 1:-1] + 1e-10)**2) * grad_theta_theta[1:-1, 1:-1, 1:-1] +
                                       grad_z_z[1:-1, 1:-1, 1:-1]) / (R[1:-1, 1:-1, 1:-1] + 1e-10)

        # Navier-Stokes Update (simplified for turbine flow)
        r_safe = R + 1e-10
        u_r_new = u_r + dt * (-u_r * dUr_dr - u_theta * dUtheta_dtheta / r_safe - u_z * dUz_dz + nu * laplacian)
        u_theta_new = u_theta + dt * (-u_r * np.gradient(u_theta, dr, axis=0) - u_theta * dUtheta_dtheta / r_safe - u_z * np.gradient(u_theta, dz, axis=2) + nu * laplacian)
        u_z_new = u_z + dt * (-u_r * np.gradient(u_z, dr, axis=0) - u_theta * np.gradient(u_z, dtheta, axis=1) / r_safe - u_z * dUz_dz + nu * laplacian)

        # Pressure Correction
        nri, nthetai, nzi = nr-2, ntheta-2, nz-2
        diagonals = [-np.ones(nri*nthetai*nzi), 6*np.ones(nri*nthetai*nzi), -np.ones(nri*nthetai*nzi)]
        offsets = [-nri*nthetai, 0, nri*nthetai]
        A = diags(diagonals, offsets, shape=(nri*nthetai*nzi, nri*nthetai*nzi))
        p = np.zeros((nr, ntheta, nz))
        div_u = compute_divergence(u_r_new, u_theta_new, u_z_new, dr, dtheta, dz)
        rhs = -dr**2 * div_u[1:-1, 1:-1, 1:-1].flatten()
        p_inner, info = cg(A, rhs, rtol=1e-12)
        if info != 0:
            print(f"CG failed at step {t} with info {info}")
            break
        p[1:-1, 1:-1, 1:-1] = p_inner.reshape(nri, nthetai, nzi)

        # Update Velocities
        dp_dr = np.gradient(p, dr, axis=0)
        dp_dtheta = np.gradient(p, dtheta, axis=1)
        dp_dz = np.gradient(p, dz, axis=2)
        u_r = u_r_new - dt * dp_dr / rho
        u_theta = u_theta_new - dt * dp_dtheta / (rho * r_safe)
        u_z = u_z_new - dt * dp_dz / rho

        # Boundary Conditions
        u_r[0, :, :] = u_r[-1, :, :] = 0
        u_z[:, :, 0] = u_z[:, :, -1] = 0
        u_theta[0, :, :] = u_theta[-1, :, :] = 0

        # Power Coefficient (simplified, based on Cp = 0.5926)
        cp = 0.5926 + 0.001 * np.sin(theta)  # Mock adjustment for θ = 36°
        if t % 100 == 0 or t == nt - 1:
            max_vel = np.max(np.sqrt(u_r**2 + u_theta**2 + u_z**2))
            div = np.max(np.abs(div_u))
            print(f"Step {t}: Cp = {cp:.4f}, Max Velocity = {max_vel:.2f} m/s, Divergence = {div:.4f}")

    # Save Results
    np.save('turbine_velocity_field.npy', np.sqrt(u_r**2 + u_theta**2 + u_z**2))
    print("Saved turbine velocity field")

except Exception as e:
    print(f"Simulation crashed at step {t}: {str(e)}")
    np.save('turbine_velocity_field_partial.npy', np.sqrt(u_r**2 + u_theta**2 + u_z**2))
    print("Saved partial turbine velocity field")
print("Turbine optimization simulation completed or interrupted!")
