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
    # Calibration behavior
    mode: str = "auto",          # "auto" | "nll" | "picp"
    target: float = 0.95,        # desired central coverage for PICP target
    tol: float = 0.02,           # tolerance band for "auto" (skip scaling if within ±tol)
    shrink: float = 0.5,         # shrink scaling in log-space toward 1.0 (0=no shrink, 1=full shrink->1)
    clamp: Tuple[float, float] = (0.7, 1.3),  # clamp scale factor after shrink
    # Column naming / slicing
    true_col_fmt: str = "true_{p}",
    mu_col_fmt: str   = "mu_{p}",
    var_col_fmt: str  = "var_tot_{p}",
    t_idx_col: str = "t_idx",
    # Plotting
    plot: bool = True,
    title_prefix: Optional[str] = None,
    model_label: Optional[str] = None,   # NEW: e.g., "2L", "50L", "TL"
    annotate: bool = True,
    quantiles: Optional[Sequence[float]] = None,
    # Output verbosity
    verbose: bool = False,
) -> Dict:
    """
    Calibrate predictive variance by a single multiplicative factor c (on variance),
    using either NLL-optimal scaling or PICP-matching scaling, with an optional
    'auto' mode that skips or gently scales if coverage is already close.

    Parameters
    ----------
    ...
    model_label : str or None
        Optional short tag for the model/scale, shown in plot titles (e.g., "2L", "50L", "TL").
    ...
    """
    import pandas as pd  # local import to keep the function self-contained for notebooks

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

    def _ce(y, mu, var):
        return abs(np.mean(var) - np.mean((y - mu) ** 2))

    def _coverage_curve(y, mu, sig, quants):
        vals = []
        for q in quants:
            z = norm.ppf(0.5 + q / 2.0)  # central two-sided
            inside = (y >= mu - z * sig) & (y <= mu + z * sig)
            vals.append(inside.mean())
        return np.array(vals)

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
        return float(s ** 2)  # scale variance

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
    nll_b  = _nll_gauss(y, mu, var).mean()
    ce_b   = _ce(y, mu, var)
    curve_b = _coverage_curve(y, mu, sig, QUANTS)

    # ---------- choose calibration ----------
    if mode not in {"auto", "nll", "picp"}:
        raise ValueError("mode must be 'auto', 'nll', or 'picp'")

    if mode == "auto":
        if abs(picp_b - target) <= tol:
            c_raw = 1.0
        else:
            c_raw = _fit_c_nll(y, mu, var)
            c_raw = _shrink_and_clamp(c_raw, shrink, clamp)
    elif mode == "nll":
        c_raw = _fit_c_nll(y, mu, var)
    else:  # "picp"
        c_raw = _fit_c_picp(y, mu, var, target)

    c_opt = float(c_raw)

    # ---------- AFTER metrics ----------
    var_a = var * c_opt
    sig_a = np.sqrt(np.clip(var_a, EPS, None))
    picp_a = _picp_at(y, mu, sig_a, Z_target)
    nll_a  = _nll_gauss(y, mu, var_a).mean()
    ce_a   = _ce(y, mu, var_a)
    curve_a = _coverage_curve(y, mu, sig_a, QUANTS)

    # ---------- plotting ----------
    fig = None
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(10.2, 4.4), sharey=True)

        # Build a clean, informative title block
        base = title_prefix or f"{target_prefix}"
        if model_label:
            ttl_base = f"{model_label} | {base}"
        else:
            ttl_base = base

        titles = [
            f"{ttl_base}: UQ before calibration",
            f"{ttl_base}: UQ after calibration",
        ]

        for a, curve, t in zip(ax, [curve_b, curve_a], titles):
            a.plot(QUANTS, curve, marker="o", linewidth=2, label="Empirical")
            a.plot([0, 1], [0, 1], "k--", linewidth=1, label="Ideal")
            a.set_xlabel("Nominal confidence")
            a.set_title(t)
            a.grid(True, alpha=0.25)
        ax[0].set_ylabel("Empirical coverage")

        if annotate:
            qlabel = f"PICP@{int(round(target*100))}"
            txt_before = (f"Scale c = 1.00\n"
                          f"{qlabel} = {picp_b:.3f}\n"
                          f"NLL = {nll_b:.3f}\n"
                          f"CE = {ce_b:.2e}")
            ax[0].text(0.03, 0.97, txt_before, transform=ax[0].transAxes,
                       va="top", ha="left",
                       fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9))

            txt_after = (f"Scale c = {c_opt:.2f}\n"
                         f"{qlabel} = {picp_a:.3f}\n"
                         f"NLL = {nll_a:.3f}\n"
                         f"CE = {ce_a:.2e}")
            ax[1].text(0.03, 0.97, txt_after, transform=ax[1].transAxes,
                       va="top", ha="left",
                       fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9))

        # Keep a single legend
        handles, labels = ax[1].get_legend_handles_labels()
        if labels:
            ax[1].legend(loc="lower right", frameon=True)

        plt.tight_layout()
        plt.show()

    # ---------- output ----------
    qlabel = f"PICP@{int(round(target*100))}"
    out = {
        "c_opt": c_opt,
        "before": {qlabel: float(picp_b), "NLL": float(nll_b), "CE": float(ce_b), "curve": curve_b},
        "after":  {qlabel: float(picp_a), "NLL": float(nll_a), "CE": float(ce_a), "curve": curve_a},
        "fig": fig,
    }

    if verbose:
        print(
            f"Scale (c) for {target_prefix}"
            + (f" @ {t_idx_col}={t_idx}" if t_idx is not None else "")
            + f" | mode={mode} | target={target} | tol={tol} | shrink={shrink} | clamp={clamp} => {c_opt:.3f}"
        )
        print(f"{'':<10} | {qlabel:>8} | {'NLL':>10} | {'CE':>12}")
        print("-" * 40)
        print(f"Before   | {out['before'][qlabel]:8.3f} | {out['before']['NLL']:10.3f} | {out['before']['CE']:12.3e}")
        print(f"After    | {out['after'][qlabel]:8.3f} | {out['after']['NLL']:10.3f} | {out['after']['CE']:12.3e}")

    return out
