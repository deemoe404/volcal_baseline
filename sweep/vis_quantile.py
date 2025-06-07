from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ──────────────────────────────────
# 0. File Paths
# ──────────────────────────────────
run_dir = Path("outputs/20250519-2111")
out_pdf = run_dir / "quantile.pdf"

# ──────────────────────────────────
# 1. Data Loading & Derived Columns
# ──────────────────────────────────
df = pd.read_csv(run_dir / "results.csv")
df["start_timestamp"] = pd.to_datetime(df["start_timestamp"])
df["GICP_timestamp"] = pd.to_datetime(df["GICP_timestamp"])
df["runtime_min"] = (
    df["GICP_timestamp"] - df["start_timestamp"]
).dt.total_seconds() / 60
df["volume_cut_acc"] = df["volume_cut_acc"].abs()


# ──────────────────────────────────
# 2. Error Detection by volume_cut_acc
# ──────────────────────────────────
def tag_outliers(group: pd.DataFrame) -> pd.DataFrame:
    q1, q3 = group["volume_cut_acc"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    group = group.copy()
    group["is_outlier"] = ~group["volume_cut_acc"].between(lower, upper)
    return group


try:
    df = df.groupby("quantile_thr", group_keys=False).apply(tag_outliers)
except TypeError:  # pandas < 2.2
    df = df.groupby("quantile_thr", group_keys=False).apply(tag_outliers)

df_clean, df_out = df[~df.is_outlier], df[df.is_outlier]

# ──────────────────────────────────
# 3. Outlier removal by mean/std
# ──────────────────────────────────
g = df_clean.groupby("quantile_thr")
stats = g.agg(
    {
        "rre_refined": ["mean", "std"],
        "rte_refined": ["mean", "std"],
        "volume_cut_acc": ["mean", "std"],
        "runtime_min": ["mean", "std"],
    }
)
stats.columns = ["_".join(c) for c in stats.columns]
stats = stats.reset_index()

# ──────────────────────────────────
# 4. Color Definitions
# ──────────────────────────────────
COL_RRE = "tab:blue"
COL_RTE = "tab:orange"
COL_ACC = "tab:green"
COL_TIME = "tab:red"

# ──────────────────────────────────
# 5. Plotting
# ──────────────────────────────────
fig, (ax_err, ax_acc, ax_time) = plt.subplots(
    nrows=3,
    sharex=True,
    figsize=(14 / 2, 8.25),
    gridspec_kw={"height_ratios": [2, 1, 1]},
)

# Boxplot of volume_cut_acc
ax_rte = ax_err.twinx()
ax_err.errorbar(
    stats["quantile_thr"],
    stats["rre_refined_mean"],
    yerr=stats["rre_refined_std"],
    marker="o",
    color=COL_RRE,
    label="RRE (°)",
)
ax_rte.errorbar(
    stats["quantile_thr"],
    stats["rte_refined_mean"],
    yerr=stats["rte_refined_std"],
    marker="s",
    color=COL_RTE,
    label="RTE (m)",
)
if not df_out.empty:
    ax_err.scatter(
        df_out["quantile_thr"],
        df_out["rre_refined"],
        marker="x",
        s=70,
        color=COL_RRE,
        label="Outlier RRE",
    )
    ax_rte.scatter(
        df_out["quantile_thr"],
        df_out["rte_refined"],
        marker="x",
        s=70,
        color=COL_RTE,
        label="Outlier RTE",
    )
ax_err.set_ylabel("RRE (°)", color=COL_RRE)
ax_err.set_yscale("symlog", linthresh=0.01, linscale=0.5)
ax_err.invert_xaxis()
ax_err.grid(alpha=0.3, which="both", linestyle=":")

ax_rte.set_ylabel("RTE (m)", color=COL_RTE)
ax_rte.set_yscale("symlog", linthresh=0.01, linscale=0.5)
ax_rte.grid(alpha=0.3, which="both", linestyle=":")

# Volume error
ax_acc.errorbar(
    stats["quantile_thr"],
    stats["volume_cut_acc_mean"],
    yerr=stats["volume_cut_acc_std"],
    marker="^",
    color=COL_ACC,
    label="Mean ± std",
)
if not df_out.empty:
    ax_acc.scatter(
        df_out["quantile_thr"],
        df_out["volume_cut_acc"],
        marker="x",
        s=70,
        color=COL_ACC,
        label="Outlier",
    )
ax_acc.set_ylabel("Volume error (%)")
ax_acc.set_yscale("symlog", linthresh=0.01, linscale=1)
ax_acc.invert_xaxis()
ax_acc.grid(alpha=0.3, which="both", linestyle=":")

# Runtime
ax_time.errorbar(
    stats["quantile_thr"],
    stats["runtime_min_mean"],
    yerr=stats["runtime_min_std"],
    marker="d",
    color=COL_TIME,
    label="Mean ± std",
)
if not df_out.empty:
    ax_time.scatter(
        df_out["quantile_thr"],
        df_out["runtime_min"],
        marker="x",
        s=70,
        color=COL_TIME,
        label="Outlier",
    )

# decide the ceiling (here: 1.3 × clean max)
upper_clean = (stats["runtime_min_mean"] + stats["runtime_min_std"]).max()
ymax = upper_clean * 1.05
ax_time.set_ylim(top=ymax, bottom=5.5)

# annotate any outlier that still sits above the ceiling
big_runtimes = df_out[df_out["runtime_min"] > ymax]
for _, row in big_runtimes.iterrows():
    # draw a downward arrow sitting on the top frame
    ax_time.scatter(row["quantile_thr"], ymax, marker="x", s=70, color=COL_TIME)
    ax_time.annotate(
        f"{row['runtime_min']:.0f} min",
        xy=(row["quantile_thr"], ymax),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        va="bottom",
        color=COL_TIME,
        fontsize=8,
        arrowprops=dict(arrowstyle="-|>", fc=COL_TIME, lw=1),
    )
# ---------------------------------------------

ax_time.set_xlabel("Quantile threshold for max correspondence distance")
ax_time.set_ylabel("Runtime (min)")
ax_time.invert_xaxis()
ax_time.grid(alpha=0.3, which="both", linestyle=":")

# save figure
fig.tight_layout()
fig.savefig(out_pdf)
print(f"Saved to: {out_pdf}")
