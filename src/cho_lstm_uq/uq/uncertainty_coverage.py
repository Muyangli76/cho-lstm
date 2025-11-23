from __future__ import annotations
from typing import Dict, Tuple, Optional, Sequence
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize


def calibrate_uq(
    df,
    target_prefix: str = "X",
    t_idx: Optional[int] = None,
    *,
    mode: str = "auto",          # "auto" | "nll" | "picp"
    target: float = 0.95,
    tol: float = 0.02,
    shrink: float = 0.5,
    clamp: Tuple[float, float] = (0.7, 1.3),
    true_col_fmt: str = "true_{p}",
    mu_col_fmt: str   = "mu_{p}",
    var_col_fmt: str  = "var_tot_{p}",
    t_idx_col: str = "t_idx",
    plot: bool = True,
    title_prefix: Optional[str] = None,
    model_label: Optional[str] = None,
    annotate: bool = True,
    quantiles: Optional[Sequence[float]] = None,
    verbose: bool = False,
) -> Dict:
    """
    Updated: now includes CRPS (Gaussian closed-form),
    median NLL, and per-horizon NLL.
    """

    import pandas as pd

    # ---------- constants ----------
    EPS = 1e-12
    if quantiles is None:
        QUANTS = np.linspace(0.05, 0.95, 19)
    else:
        QUANTS = np.asarray(quantiles, dtype=float)

    Z_target = norm.ppf(0.5 + target / 2.0)

    # ---------- helpers ----------
    def _prep_cols(frame: pd.DataFrame, p: str):
        y  = frame[true_col_fmt.format(p=p)].to_numpy()
        mu = frame[mu_col_fmt.format(p=p)].to_numpy()
        var = frame[var_col_fmt.format(p=p)].to_numpy()
        sig = np.sqrt(np.clip(var, EPS, None))
        return y, mu, var, sig

    def _nll_gauss(y, mu, var):
        v = np.clip(var, EPS, None)
        return 0.5 * (np.log(2 * np.pi * v) + (y - mu) ** 2 / v)

    def _picp_at(y, mu, sig, z):
        return ((y >= mu - z * sig) & (y <= mu + z * sig)).mean()

    def _coverage_curve(y, mu, sig, quants):
        vals = []
        for q in quants:
            z = norm.ppf(0.5 + q / 2.0)
            inside = (y >= mu - z * sig) & (y <= mu + z * sig)
            vals.append(inside.mean())
        return np.array(vals)

    # ---------- NEW: CRPS closed-form ----------
    def _crps_gaussian(y, mu, sig):
        """Return per-sample CRPS for Normal(μ,σ)."""
        x = (y - mu) / np.maximum(sig, EPS)
        return sig * (
            x * (2 * norm.cdf(x) - 1)
            + 2 * norm.pdf(x)
            - 1 / np.sqrt(np.pi)
        )

    # ---------- NEW: per-horizon NLL ----------
    def _per_horizon_nll(frame: pd.DataFrame, var_scaled=None):
        """Returns {h: mean NLL at horizon h}."""
        out = {}
        if var_scaled is not None:
            v = np.clip(var_scaled, EPS, None)
        else:
            v = np.clip(frame[var_col_fmt.format(p=target_prefix)], EPS, None)

        y = frame[true_col_fmt.format(p=target_prefix)].to_numpy()
        mu = frame[mu_col_fmt.format(p=target_prefix)].to_numpy()
        tvals = frame[t_idx_col].to_numpy()

        nll_vals = _nll_gauss(y, mu, v)

        for h in np.unique(tvals):
            mask = (tvals == h)
            out[int(h)] = float(nll_vals[mask].mean())

        return out

    # ---------- calibration solvers ----------
    def _fit_c_nll(y, mu, var):
        def nll_obj(log_c):
            c = np.exp(log_c)
            return np.mean(_nll_gauss(y, mu, var * c))
        res = minimize(nll_obj, x0=np.log(1.0))
        return float(np.exp(res.x[0]))

    def _fit_c_picp(y, mu, var, tgt):
        sigma = np.sqrt(np.clip(var, EPS, None))
        r = np.abs(y - mu) / np.maximum(sigma, EPS)
        s = np.quantile(r, tgt)
        return float(s ** 2)

    def _shrink_and_clamp(c, shrink_frac, clamp_pair):
        log_c = np.log(max(c, EPS))
        c_shrunk = float(np.exp((1.0 - shrink_frac) * log_c))
        return float(min(max(c_shrunk, clamp_pair[0]), clamp_pair[1]))

    # ---------- slice ----------
    d = df if t_idx is None else df[df[t_idx_col] == t_idx]
    if getattr(d, "empty", False):
        raise ValueError(f"No rows for {t_idx_col}={t_idx}")

    # ---------- data ----------
    y, mu, var, sig = _prep_cols(d, target_prefix)

    # ---------- BEFORE metrics ----------
    picp_b = _picp_at(y, mu, sig, Z_target)
    nll_b_vals = _nll_gauss(y, mu, var)
    nll_b = float(nll_b_vals.mean())
    nll_b_med = float(np.median(nll_b_vals))
    crps_b_vals = _crps_gaussian(y, mu, sig)
    crps_b = float(crps_b_vals.mean())
    crps_b_med = float(np.median(crps_b_vals))
    curve_b = _coverage_curve(y, mu, sig, QUANTS)
    per_h_b = _per_horizon_nll(d, var_scaled=None)

    # ---------- choose calibration ----------
    if mode == "auto":
        if abs(picp_b - target) <= tol:
            c_raw = 1.0
        else:
            c_raw = _fit_c_nll(y, mu, var)
            c_raw = _shrink_and_clamp(c_raw, shrink, clamp)
    elif mode == "nll":
        c_raw = _fit_c_nll(y, mu, var)
    else:
        c_raw = _fit_c_picp(y, mu, var, target)

    c_opt = float(c_raw)

    # ---------- AFTER metrics ----------
    var_a = var * c_opt
    sig_a = np.sqrt(np.clip(var_a, EPS, None))

    picp_a = _picp_at(y, mu, sig_a, Z_target)

    nll_a_vals = _nll_gauss(y, mu, var_a)
    nll_a = float(nll_a_vals.mean())
    nll_a_med = float(np.median(nll_a_vals))

    crps_a_vals = _crps_gaussian(y, mu, sig_a)
    crps_a = float(crps_a_vals.mean())
    crps_a_med = float(np.median(crps_a_vals))

    curve_a = _coverage_curve(y, mu, sig_a, QUANTS)
    per_h_a = _per_horizon_nll(d, var_scaled=var_a)

    # ---------- plotting ----------
    fig = None
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(10.2, 4.4), sharey=True)

        base = title_prefix or f"{target_prefix}"
        ttl_base = f"{model_label} | {base}" if model_label else base

        titles = [
            f"{ttl_base}: UQ before calibration",
            f"{ttl_base}: UQ after calibration",
        ]

        for a, curve, t in zip(ax, [curve_b, curve_a], titles):
            a.plot(QUANTS, curve, marker="o", linewidth=2, label="Empirical")
            a.plot([0, 1], [0, 1], "k--", linewidth=1)
            a.set_xlabel("Nominal confidence")
            a.set_title(t)
            a.grid(True, alpha=0.25)
        ax[0].set_ylabel("Empirical coverage")

        if annotate:
            lbl = f"PICP@{int(round(target*100))}"
            txt_b = (
                f"Scale c = 1.00\n"
                f"{lbl} = {picp_b:.3f}\n"
                f"NLL = {nll_b:.3f}\n"
                f"NLL_med = {nll_b_med:.3f}\n"
                f"CRPS = {crps_b:.3f}\n"
                f"CRPS_med = {crps_b_med:.3f}"
            )
            ax[0].text(0.03, 0.97, txt_b, transform=ax[0].transAxes,
                       va="top", ha="left", fontsize=9,
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7"))

            txt_a = (
                f"Scale c = {c_opt:.2f}\n"
                f"{lbl} = {picp_a:.3f}\n"
                f"NLL = {nll_a:.3f}\n"
                f"NLL_med = {nll_a_med:.3f}\n"
                f"CRPS = {crps_a:.3f}\n"
                f"CRPS_med = {crps_a_med:.3f}"
            )
            ax[1].text(0.03, 0.97, txt_a, transform=ax[1].transAxes,
                       va="top", ha="left", fontsize=9,
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7"))

        plt.tight_layout()
        plt.show()

    # ---------- output ----------
    qlabel = f"PICP@{int(round(target*100))}"

    return {
        "c_opt": c_opt,
        "before": {
            qlabel: float(picp_b),
            "NLL": float(nll_b),
            "NLL_med": nll_b_med,
            "CRPS": float(crps_b),
            "CRPS_med": crps_b_med,
            "curve": curve_b,
            "per_horizon_NLL": per_h_b,
        },
        "after": {
            qlabel: float(picp_a),
            "NLL": float(nll_a),
            "NLL_med": nll_a_med,
            "CRPS": float(crps_a),
            "CRPS_med": crps_a_med,
            "curve": curve_a,
            "per_horizon_NLL": per_h_a,
        },
        "fig": fig,
    }
