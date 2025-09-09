from pathlib import Path
import argparse
import json
import math
import sys
from dataclasses import dataclass
from typing import Tuple, Callable, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# -----------------------------
# Utilities
# -----------------------------

def _safe(val, default):
    return default if pd.isna(val) else val

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["name", "m_obs_MeV", "sigma_MeV"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {path}")
    # Defaults
    if "base_MeV" not in df.columns:
        df["base_MeV"] = df["m_obs_MeV"].astype(float)
    if "k_i" not in df.columns:
        df["k_i"] = 1.0
    if "p_i" not in df.columns:
        df["p_i"] = 1.0
    if "type" not in df.columns:
        df["type"] = "particle"
    # Types
    df["m_obs_MeV"] = df["m_obs_MeV"].astype(float)
    df["sigma_MeV"]  = df["sigma_MeV"].astype(float).replace(0.0, np.nan)
    df["base_MeV"]   = df["base_MeV"].astype(float)
    df["k_i"]        = df["k_i"].astype(float)
    df["p_i"]        = df["p_i"].astype(float)
    # Guard: replace zero/NaN uncertainties with a small floor to avoid inf weights
    sig_floor = max(1e-9, np.nanmedian(df["sigma_MeV"]) * 1e-6 if np.isfinite(np.nanmedian(df["sigma_MeV"])) else 1e-6)
    df["sigma_MeV"]  = df["sigma_MeV"].fillna(sig_floor).replace(0.0, sig_floor)
    return df

# -----------------------------
# Models
# -----------------------------

def model_power(alpha: float, base, k_i, p_i) -> np.ndarray:
    return base * np.power((1.0 + alpha * k_i), p_i)

def model_exp(alpha: float, base, k_i, p_i) -> np.ndarray:
    return base * np.exp(alpha * k_i)

def model_poly(alpha: float, base, k_i, p_i) -> np.ndarray:
    # 2nd order Taylor-like polynomial inside a power
    return base * np.power((1.0 + alpha * k_i + 0.5 * (alpha * k_i)**2), p_i)

MODELS: Dict[str, Callable[[float, np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = {
    "power": model_power,
    "exp":   model_exp,
    "poly":  model_poly,
}

# -----------------------------
# Chi-square
# -----------------------------

def chi2_of_alpha(alpha: float, df: pd.DataFrame, model_name: str) -> float:
    model = MODELS[model_name]
    pred = model(alpha, df["base_MeV"].to_numpy(), df["k_i"].to_numpy(), df["p_i"].to_numpy())
    resid = (df["m_obs_MeV"].to_numpy() - pred) / df["sigma_MeV"].to_numpy()
    return float(np.sum(resid**2))

def fit_alpha(df: pd.DataFrame, model_name: str, alpha0: float = 0.007, bounds: Tuple[float,float] = (-0.2, 0.2)) -> Tuple[float, float, float]:
    """Return (alpha_hat, chi2_min, alpha_err_approx)"""
    def objective(a):
        return chi2_of_alpha(a[0], df, model_name)
    res = minimize(objective, x0=np.array([alpha0], dtype=float), bounds=[bounds], method="L-BFGS-B")
    if not res.success:
        print("[WARN] Minimizer did not converge:", res.message, file=sys.stderr)
    alpha_hat = float(res.x[0])
    # Error estimate from curvature: σ ≈ sqrt(2 / f''(α̂)) using numeric second derivative
    eps = 1e-6 * max(1.0, abs(alpha_hat))
    fpp = (objective([alpha_hat + eps]) - 2*objective([alpha_hat]) + objective([alpha_hat - eps])) / (eps**2)
    alpha_err = math.sqrt(2.0 / fpp) if fpp > 0 else float("nan")
    chi2_min = float(res.fun)
    return alpha_hat, chi2_min, alpha_err

def evaluate(df: pd.DataFrame, model_name: str, alpha: float) -> pd.DataFrame:
    model = MODELS[model_name]
    pred = model(alpha, df["base_MeV"].to_numpy(), df["k_i"].to_numpy(), df["p_i"].to_numpy())
    out = df.copy()
    out["m_pred_MeV"] = pred
    out["residual_MeV"] = out["m_obs_MeV"] - out["m_pred_MeV"]
    out["pull"] = out["residual_MeV"] / out["sigma_MeV"]
    return out

def bootstrap_alpha(df: pd.DataFrame, model_name: str, alpha_hat: float, n_boot: int = 2000, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(df)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        dfb = df.iloc[idx].reset_index(drop=True)
        ah, _, _ = fit_alpha(dfb, model_name, alpha0=alpha_hat)
        boots.append(ah)
    lo = float(np.percentile(boots, 16))
    hi = float(np.percentile(boots, 84))
    return lo, hi

# -----------------------------
# Plotting
# -----------------------------

def plot_obs_vs_pred(df_pred: pd.DataFrame, out_png: str):
    plt.figure()
    x = df_pred["m_obs_MeV"].to_numpy()
    y = df_pred["m_pred_MeV"].to_numpy()
    plt.scatter(x, y)
    lim = [min(x.min(), y.min()), max(x.max(), y.max())]
    plt.plot(lim, lim)  # 1:1 line
    plt.xlabel("Observed mass (MeV)")
    plt.ylabel("Predicted mass (MeV)")
    plt.title("Observed vs Predicted masses")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def plot_residuals(df_pred: pd.DataFrame, out_png: str):
    plt.figure()
    pulls = df_pred["pull"].to_numpy()
    plt.hist(pulls, bins=20)
    plt.xlabel("Pull ( (m_obs - m_pred)/sigma )")
    plt.ylabel("Count")
    plt.title("Residual pulls")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Fit alpha in a UWT-inspired mass model.")
    ap.add_argument("--data", required=True, help="CSV with columns name,m_obs_MeV,sigma_MeV and optional base_MeV,k_i,p_i,type")
    ap.add_argument("--model", choices=list(MODELS.keys()), default="power", help="Mass model form")
    ap.add_argument("--alpha0", type=float, default=0.007, help="Initial guess for alpha")
    ap.add_argument("--alpha-fixed", dest="alpha_fixed", type=float, default=None, help="If given, skip fit and evaluate at this alpha")
    ap.add_argument("--bounds", type=float, nargs=2, default=[-0.05, 0.05], help="Bounds for alpha during fit")
    ap.add_argument("--bootstrap", type=int, default=0, help="Bootstrap resamples for CI (0 to disable)")
    ap.add_argument("--out", default="uwt_mass_fit_out", help="Output directory")
    args = ap.parse_args()
    df = load_data(args.data)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.alpha_fixed is not None:
        alpha_hat = float(args.alpha_fixed)
        chi2_min = chi2_of_alpha(alpha_hat, df, args.model)
        dof = len(df) - 1
        red = chi2_min / max(1, dof)
        alpha_err = float("nan")
        boot_lo = boot_hi = None
    else:
        alpha_hat, chi2_min, alpha_err = fit_alpha(df, args.model, alpha0=args.alpha0, bounds=tuple(args.bounds))
        dof = len(df) - 1
        red = chi2_min / max(1, dof)
        boot_lo = boot_hi = None
        if args.bootstrap and args.bootstrap > 0:
            boot_lo, boot_hi = bootstrap_alpha(df, args.model, alpha_hat, n_boot=args.bootstrap)
    df_pred = evaluate(df, args.model, alpha_hat)
    df_pred.to_csv(outdir / "masses_fit.csv", index=False)
    plot_obs_vs_pred(df_pred, str(outdir / "m_obs_vs_pred.png"))
    plot_residuals(df_pred, str(outdir / "residuals_hist.png"))
    results = {
        "model": args.model,
        "alpha_hat": alpha_hat,
        "alpha_err_curvature": alpha_err,
        "chi2": chi2_min,
        "dof": int(dof),
        "chi2_reduced": red,
        "bootstrap_CI_16_84": [boot_lo, boot_hi] if boot_lo is not None else None,
        "n_points": int(len(df)),
        "data": str(Path(args.data).resolve()),
    }
    with open(outdir / "fit_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(outdir / "fit_report.txt", "w", encoding="utf-8") as f:
        f.write("uwt_mass_fit_alpha.py — fit report\n")
        for k, v in results.items():
            f.write(f"{k}: {v}\n")
    print(f"[OK] alpha_hat = {alpha_hat:.10g} (χ²/dof = {red:.3f})")
    if boot_lo is not None:
        print(f"[boot] 68% CI ≈ [{boot_lo:.10g}, {boot_hi:.10g}]")
    print(f"[OUT] Results in: {outdir}")

if __name__ == "__main__":
    main()