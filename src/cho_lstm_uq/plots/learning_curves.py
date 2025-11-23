import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_r2_curves(df, title_prefix=None, model_label=None, save_path=None):
    """
    Plot Teacher-Forced (TF) and Open-Loop (OL) R² vs number of training runs.
    Automatically inserts model label (e.g., 2L, 50L, TL) in the title and axis labels.
    """

    # -------- infer model label --------
    inferred_label = "Unknown"
    if model_label:
        inferred_label = model_label
    elif hasattr(df, "attrs") and "model_label" in df.attrs:
        inferred_label = df.attrs["model_label"]
    elif save_path:
        for tag in ["2L", "50L", "TL"]:
            if tag in save_path:
                inferred_label = tag
                break

    prefix = title_prefix or "Model"
    run_label = f"{inferred_label} runs" if inferred_label != "Unknown" else "training runs"

    # -------- TF plot --------
    plt.figure(figsize=(7, 5))
    plt.plot(df["k_runs"], df["TF_R2_X_gL"], marker="o", label="X_gL", linewidth=2)
    plt.plot(df["k_runs"], df["TF_R2_Ab_gL"], marker="s", label="Ab_gL", linewidth=2)
    plt.title(f"{prefix} {inferred_label}: R² vs Number of {run_label}")
    plt.xlabel(f"k (number of {run_label} in training)")
    plt.ylabel("R²")
    plt.ylim(-0.5, 1.1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

    # -------- OL plot --------
    plt.figure(figsize=(7, 5))
    plt.plot(df["k_runs"], df["OL_R2_X_gL"], marker="o", label="X_gL", linewidth=2)
    plt.plot(df["k_runs"], df["OL_R2_Ab_gL"], marker="s", label="Ab_gL", linewidth=2)
    plt.title(f"{prefix} {inferred_label}: R² vs Number of {run_label}")
    plt.xlabel(f"k (number of {run_label} in training)")
    plt.ylabel("R²")
    plt.ylim(-0.5, 1.1)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Force integer ticks on x-axis
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.xticks(sorted(df["k_runs"].unique()))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path.replace(".png", "_openloop.png"), dpi=300)
    plt.show()