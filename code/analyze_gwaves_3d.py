# analyze_gwaves_3d.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
import os

vel = np.load("3D_velocity_field.npy")
nr, ntheta, nz = vel.shape

# Average over theta, take radial profile at r=0.5
vel_r = vel[nr//2, :, :].mean(axis=0)  # Average over theta
# Or take max: vel_r = vel[nr//2, :, :].max(axis=0)

dz = 1.0 / (nz - 1)
N = len(vel_r)
freq = fftfreq(N, d=dz)[:N//2]
power = np.abs(fft(vel_r - vel_r.mean()))[:N//2]**2

# Peak detection
peaks, _ = find_peaks(power, height=0.01*power.max(), distance=3)
peak_freqs = freq[peaks]
peak_power = power[peaks]
idx = np.argsort(peak_power)[::-1][:5]

print("3D-AVERAGED MODES:")
for f, p in zip(peak_freqs[idx], peak_power[idx]):
    print(f"  f = {f:.3f} → λ = {1/f:.3f} m | P = {p:.2e}")

# Add to analyze_gwaves_3d.py
print(f"Wake mode wavelength: {1/1.984:.3f} m")
print(f"Turbine diameter: 1.0 m → D/2 = 0.5 m")
print(f"→ λ ≈ D → Classic near-wake expansion!")

# Plot
plt.semilogy(freq, power, 'k-')
plt.semilogy(peak_freqs[idx], peak_power[idx], 'ro')
plt.axvline(0.749, color='gold', linestyle='--', label='g-wave (0.749)')
plt.xlim(0, 2)
plt.legend()
plt.title("3D-Averaged Axial FFT → Hunting g-wave")
plt.savefig(os.path.join(os.path.expanduser("~"), "Desktop", "gwave_3d_fft.png"), dpi=200)
plt.show()