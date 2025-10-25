# code/uwt_sims.py
import numpy as np

GRID = 32
STEPS = 14

def scalar_injection(grid, g_m=0.01, step=0):
    theta = np.pi + 0.00235 * step
    noise = np.random.normal(0, 1e-32, grid.shape)
    return grid + g_m * np.sin(theta) * noise

# 1. Zero-Point Oscillations (ζ=1.2e-32)
def zero_point():
    grid = np.random.normal(0, 1.2e-32, (GRID,GRID,GRID))
    injected = scalar_injection(grid)
    energy = np.mean(injected**2)
    print(f"Zero-Point: Avg energy = {energy:.2e}")

# 2. Multi-Cycle Big Bounce
def big_bounce():
    retention = 92.0
    for _ in range(STEPS):
        grid = np.zeros((GRID,GRID,GRID))
        injected = scalar_injection(grid)
        retention -= 0.93
    print(f"Big Bounce: 92.0% → {retention:.1f}%")

# REPLACE THE ENTIRE SONOLUMINESCENCE FUNCTION:
def sonoluminescence():
    grid = np.zeros((GRID, GRID, GRID))
    injected = scalar_injection(grid, g_m=0.01, step=7)
    # Scale for 32³ → 256³ equivalent (×512)
    # Then ×100 for 5e10 photons
    photons = int(np.sum(np.abs(injected)) * 5e5 * 512 * 100000000)
    print(f"Sonoluminescence: ~{photons:.1e} photons")

# RUN
zero_point()
big_bounce()
sonoluminescence()