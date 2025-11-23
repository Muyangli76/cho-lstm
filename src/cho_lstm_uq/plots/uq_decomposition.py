import matplotlib.pyplot as plt
import pandas as pd

def plot_uq_decomposition(df: pd.DataFrame, title_prefix: str = "2L", save_path: str = None):
    """
    Plot aleatoric, epistemic, and total variance decomposition over time
    for both X and Ab predictions, aggregated across runs.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns:
        ["run_id", "t_idx",
         "var_ale_X", "var_epi_X", "var_tot_X",
         "var_ale_Ab", "var_epi_Ab", "var_tot_Ab"]
    title_prefix : str
        Label prefix, e.g. "2L" or "50L".
    save_path : str or None
        If provided, saves the two plots to files instead of just showing.
    """
    # 1. collapse overlapping windows inside each run at each t_idx
    per_rt = df.groupby(["run_id", "t_idx"], as_index=False).agg({
        "var_ale_X":"mean", "var_epi_X":"mean", "var_tot_X":"mean",
        "var_ale_Ab":"mean","var_epi_Ab":"mean","var_tot_Ab":"mean",
    })

    # 2. aggregate across runs (mean variance at each t_idx)
    agg = per_rt.groupby("t_idx", as_index=False).agg({
        "var_ale_X":"mean", "var_epi_X":"mean", "var_tot_X":"mean",
        "var_ale_Ab":"mean","var_epi_Ab":"mean","var_tot_Ab":"mean",
    })

    # 3a. Plot X
    plt.figure(figsize=(8,5))
    plt.plot(agg["t_idx"], agg["var_ale_X"], label="Aleatoric", linewidth=3)
    plt.plot(agg["t_idx"], agg["var_epi_X"], label="Epistemic", linewidth=3)
    plt.plot(agg["t_idx"], agg["var_tot_X"], label="Total", linewidth=3)
    plt.title(f"{title_prefix} – X: Uncertainty Quantification Decomposition")
    plt.xlabel("Horizon (h)")
    plt.ylabel("Variance")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path.replace(".png", "_X.png"), dpi=300)
    plt.show()

    # 3b. Plot Ab
    plt.figure(figsize=(8,5))
    plt.plot(agg["t_idx"], agg["var_ale_Ab"], label="Aleatoric", linewidth=3)
    plt.plot(agg["t_idx"], agg["var_epi_Ab"], label="Epistemic", linewidth=3)
    plt.plot(agg["t_idx"], agg["var_tot_Ab"], label="Total", linewidth=3)
    plt.title(f"{title_prefix} – Ab: Uncertainty Quantification Decomposition")
    plt.xlabel("Horizon (h)")
    plt.ylabel("Variance")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path.replace(".png", "_Ab.png"), dpi=300)
    plt.show()
