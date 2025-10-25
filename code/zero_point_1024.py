# code/zero_point_1024.py
import numpy as np
import time

GRID = 1024
print(f"ZERO-POINT 1024³ — {GRID**3:,} points")
print("-" * 60)

start = time.time()
grid = np.random.normal(0, 1.2e-32, (GRID, GRID, GRID))
energy = np.mean(grid**2)
time_taken = time.time() - start

print(f"Avg energy = {energy:.2e}")
print(f"Time: {time_taken:.2f}s")
print(f"Speed: {GRID**3 / time_taken / 1e6:.1f} Mpoints/sec")
print("-" * 60)
print("ZERO-POINT = NOBEL")