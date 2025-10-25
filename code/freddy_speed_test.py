# freddy_speed_test.py — UWT BENCHMARK
# Purpose: Measure Threadripper 256³ crunch speed
# Author: Peter Baldwin
# Date: 25 Oct 2025

import numpy as np
import time

GRID = 2048
POINTS = GRID**3
print(f"FREDDY SPEED TEST — GRID: {GRID}³ = {POINTS:,} points")
print("-" * 60)

def crunch():
    start = time.time()
    a = np.random.random((GRID, GRID, GRID))
    b = np.random.random((GRID, GRID, GRID))
    c = a * b
    d = np.sum(c)
    end = time.time()
    return end - start, d

time_taken, result = crunch()
speed = POINTS / time_taken / 1e6  # Mpoints/sec

print(f"Crunch time: {time_taken:.3f}s")
print(f"Speed: {speed:.1f} Mpoints/sec")
print(f"Result: {result:.2e}")
print("-" * 60)
print("FREDDY = NOBEL READY")