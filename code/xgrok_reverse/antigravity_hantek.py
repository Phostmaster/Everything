import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Load Hantek CSV
data = pd.read_csv('hantek_lift.csv')
time = data['Time']
accel = data['Channel2'] * 0.01  # mV to g (calibrate)

# Period T
peaks, _ = find_peaks(accel, height=0.5)
T = np.mean(np.diff(time[peaks]))
L = 0.65  # m
g_eff = 4 * np.pi**2 * L / T**2
delta_m_m = (9.81 - g_eff) / 9.81

# Entropy estimate
S = -1e8 * np.log(1 + np.var(accel))

print(f"Δm/m = {delta_m_m:.6f} | S ≈ {S:.2e} k_B")
plt.plot(time, accel); plt.show()