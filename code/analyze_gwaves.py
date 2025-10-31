import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
import os

# --- Load ---
code_dir = os.path.dirname(os.path.abspath(__file__))
npy_path = os.path.join(code_dir, "3D_velocity_field.npy")
vel = np.load(npy_path)
print(f"Loaded: {vel.shape}, max = {vel.max():.3e} m/s")

nr, ntheta, nz = vel.shape
dz = 1.0 / (nz - 1)  # Correct spacing

# --- Extract centerline ---
r_idx = nr // 2
theta_idx = 0
axial_vel = vel[r_idx, theta_idx, :]

# --- FFT ---
N = len(axial_vel)
freq = fftfreq(N, d=dz)[:N//2]
fft_vals = fft(axial_vel - axial_vel.mean())  # Remove DC
power = np.abs(fft_vals)[:N//2]**2

# --- Smoothing for peak detection ---
from scipy.ndimage import gaussian_filter1d
power_smooth = gaussian_filter1d(power, sigma=1)

# --- Find peaks ---
peaks, properties = find_peaks(power_smooth, height=0.01 * power_smooth.max(), distance=3)
peak_freqs = freq[peaks]
peak_power = power[peaks]

# Sort
idx = np.argsort(peak_power)[::-1]
top_freqs = peak_freqs[idx][:6]
top_power = peak_power[idx][:6]

print("\n" + "="*50)
print("DOMINANT MODES (Axial Velocity FFT)")
print("="*50)
for i, (f, p) in enumerate(zip(top_freqs, top_power)):
    wavelength = 1.0 / f if f > 0 else np.inf
    print(f"Mode {i+1:1d}: f = {f:6.3f} cycles/m → λ = {wavelength:6.3f} m | Power = {p:8.2e}")

# --- Highlight g-wave prediction ---
gwave_freq = 1.0 / 1.336
print(f"\n→ Predicted g-wave: f = {gwave_freq:.3f} cycles/m (λ = 1.336 m)")

# --- Plot ---
plt.figure(figsize=(13, 5))

plt.subplot(1, 2, 1)
z = np.linspace(0, 1, nz)
plt.plot(z, axial_vel, 'b-', linewidth=1.2)
plt.title('Axial Velocity Profile (r=0.5 m, θ=0)')
plt.xlabel('Z (m)')
plt.ylabel('Velocity (m/s)')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.semilogy(freq, power, 'k-', alpha=0.6, label='Raw')
plt.semilogy(freq, power_smooth, 'r-', linewidth=2, label='Smoothed')
plt.semilogy(top_freqs, top_power, 'go', markersize=8, label='Detected Peaks')
for f, p in zip(top_freqs, top_power):
    plt.text(f, p*1.8, f'{f:.2f}', fontsize=9, ha='center', color='green')

# g-wave line
plt.axvline(gwave_freq, color='gold', linestyle='--', linewidth=2, label=f'g-wave (f={gwave_freq:.3f})')

plt.xlim(0, 8)
plt.ylim(power_smooth.min()*0.5, power_smooth.max()*10)
plt.title('FFT Power Spectrum → g-wave Search')
plt.xlabel('Frequency (cycles/m)')
plt.ylabel('Power')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(os.path.expanduser("~"), "Desktop", "gwave_fft_analysis_v2.png")
plt.savefig(plot_path, dpi=200, bbox_inches='tight')
plt.show()

print(f"\nPlot saved: {plot_path}")