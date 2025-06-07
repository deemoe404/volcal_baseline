from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ─────────────────────── paths ───────────────────────
run_dir = Path("outputs/20250519-2111")
csv_path = run_dir / "results.csv"
fig_path = run_dir / "m3c2_noise.pdf"

# ──────────────────── load & filter ──────────────────
df = pd.read_csv(csv_path)
df = df[df["quantile_thr"] == 0.20]

metrics = (
    df.groupby("noise_std")
    .agg(
        precision_mean=("cd_precision", "mean"),
        precision_std=("cd_precision", "std"),
        recall_mean=("cd_recall", "mean"),
        recall_std=("cd_recall", "std"),
        iou_mean=("cd_iou", "mean"),
        iou_std=("cd_iou", "std"),
        miscls_mean=("cd_miscls", "mean"),
        miscls_std=("cd_miscls", "std"),
        lod_in_mean=("LOD_mean_in", "mean"),
        lod_in_std=("LOD_mean_in", "std"),
        lod_out_mean=("LOD_mean_out", "mean"),
        lod_out_std=("LOD_mean_out", "std"),
    )
    .reset_index()
)


# ───────────────────── plotting ──────────────────────
def tag_outliers(group: pd.DataFrame, col: str) -> pd.DataFrame:
    """Flag outliers in *col* via 1.5xIQR rule."""
    q1, q3 = group[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    g = group.copy()
    g["is_outlier"] = ~g[col].between(lower, upper)
    return g


df = df.groupby("noise_std", group_keys=False).apply(tag_outliers, col="cd_miscls")
df_clean, df_out = df[~df.is_outlier], df[df.is_outlier]

g = df_clean.groupby("noise_std")
stats = g.agg(
    {
        "cd_precision": ["mean", "std"],
        "cd_recall": ["mean", "std"],
        "cd_iou": ["mean", "std"],
        "cd_miscls": ["mean", "std"],
        "LOD_mean_in": ["mean", "std"],
        "LOD_mean_out": ["mean", "std"],
    }
)
stats.columns = ["_".join(c) for c in stats.columns]
stats = stats.reset_index()

# ────────────────────── colours ───────────────────────
COL_PREC = "tab:blue"
COL_REC = "tab:orange"
COL_IOU = "tab:green"
COL_MIS = "tab:red"
COL_IN = "tab:purple"
COL_OUT = "tab:brown"

# ────────────────────── figure ────────────────────────
fig, (ax_det, ax_mis, ax_lod) = plt.subplots(
    3,
    1,
    figsize=(14 / 2, 8.25),
    sharex=True,
    gridspec_kw=dict(height_ratios=[1.5, 1, 1]),
)

x = stats["noise_std"]

# — A: detection metrics —
ax_det.errorbar(
    x,
    stats["cd_precision_mean"],
    yerr=stats["cd_precision_std"],
    marker="o",
    color=COL_PREC,
    label="Precision",
)
ax_det.errorbar(
    x,
    stats["cd_recall_mean"],
    yerr=stats["cd_recall_std"],
    marker="s",
    color=COL_REC,
    label="Recall",
)
ax_det.errorbar(
    x,
    stats["cd_iou_mean"],
    yerr=stats["cd_iou_std"],
    marker="^",
    color=COL_IOU,
    label="IoU",
)
# scatter outliers
for col, colour in [
    ("cd_precision", COL_PREC),
    ("cd_recall", COL_REC),
    ("cd_iou", COL_IOU),
]:
    sel = df_out[["noise_std", col]].dropna()
    ax_det.scatter(sel["noise_std"], sel[col], marker="x", s=60, color=colour)
ax_det.set_ylabel("Score")
ax_det.grid(alpha=0.3, which="both", linestyle=":")
ax_det.legend(ncol=2, loc="lower left")

# — B: mis-classification %
ax_mis.errorbar(
    x,
    stats["cd_miscls_mean"] * 100,
    yerr=stats["cd_miscls_std"] * 100,
    marker="d",
    color=COL_MIS,
)
sel = df_out[["noise_std", "cd_miscls"]].dropna()
ax_mis.scatter(
    sel["noise_std"], sel["cd_miscls"] * 100, marker="x", s=60, color=COL_MIS
)
ax_mis.set_ylabel("Misclassification Rate (%)")
ax_mis.grid(alpha=0.3, which="both", linestyle=":")

# — C: LoD —
ax_lod.errorbar(
    x,
    stats["LOD_mean_in_mean"],
    yerr=stats["LOD_mean_in_std"],
    marker="o",
    color=COL_IN,
    label="In-deformation",
)
ax_lod.errorbar(
    x,
    stats["LOD_mean_out_mean"],
    yerr=stats["LOD_mean_out_std"],
    marker="s",
    color=COL_OUT,
    label="Background",
)
for col, colour in [("LOD_mean_in", COL_IN), ("LOD_mean_out", COL_OUT)]:
    sel = df_out[["noise_std", col]].dropna()
    ax_lod.scatter(sel["noise_std"], sel[col], marker="x", s=60, color=colour)
ax_lod.set_ylabel("Mean LoD (m)")
ax_lod.set_xlabel(r"Added Gaussian noise $\sigma$ (m)")
ax_lod.grid(alpha=0.3, linestyle=":")
ax_lod.legend(loc="upper left")

fig.tight_layout()
fig_path = run_dir / "m3c2_noise.pdf"
fig.savefig(fig_path)
print("Saved to", fig_path)

# ───────────────── print nicely ─────────────────
with pd.option_context("display.max_columns", None, "display.precision", 4):
    print("\n=== Detection & LoD stats by noise sigma ===")
    print(metrics.to_string(index=False))
