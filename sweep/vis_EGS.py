from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ───────────── paths ─────────────
run_dir = Path("outputs/20250522-0211")
csv_path = run_dir / "results.csv"

# ───────────── data ─────────────
df = pd.read_csv(csv_path)
g = (
    df.groupby("voxel_size")
    .agg(
        src_mean=("n_points_src", "mean"),
        src_std=("n_points_src", "std"),
        tgt_mean=("n_points_tgt", "mean"),
        tgt_std=("n_points_tgt", "std"),
        gpu_mean=("egs_peak_gpu_gb", "mean"),
        gpu_std=("egs_peak_gpu_gb", "std"),
        rss_mean=("egs_cpu_rss_mb", "mean"),
        rss_std=("egs_cpu_rss_mb", "std"),
        t_mean=("egs_time_s", "mean"),
        t_std=("egs_time_s", "std"),
        rre_mean=("rre_deg", "mean"),
        rre_std=("rre_deg", "std"),
        rte_mean=("rte_m", "mean"),
        rte_std=("rte_m", "std"),
    )
    .reset_index()
)
vox = g["voxel_size"]

# ─────────── Plot A: Memory ───────────
fig_mem, ax_mem = plt.subplots(figsize=(14 / 3, 4))
ax_mem.errorbar(
    vox,
    g["gpu_mean"],
    yerr=g["gpu_std"],
    marker="o",
    label="GPU peak (GiB)",
    color="tab:red",
)
ax_mem.errorbar(
    vox,
    g["rss_mean"] / 1024,
    yerr=g["rss_std"] / 1024,
    marker="s",
    label="CPU RSS (GiB)",
    color="tab:purple",
)
ax_mem.set_ylabel("Memory (GiB)")
ax_mem.grid(alpha=0.3, linestyle=":")
ax_mem.legend(loc="best")

# secondary x-axis for point counts
ax_pts = ax_mem.twiny()
ax_pts.set_xlim(ax_mem.get_xlim())
ax_pts.set_xticks(vox)


def fmt(n):
    return f"{n/1e6:.1f} M" if n >= 1e6 else f"{int(n/1e3)} k"


ax_pts.set_xticklabels([fmt(v) for v in g["src_mean"]])
ax_pts.set_xlabel("# points after down-sampling")
ax_mem.set_xlabel("EGS voxel size (m)")

fig_mem.tight_layout()
mem_path = run_dir / "egs_memory.pdf"
fig_mem.savefig(mem_path)
print("Saved memory plot to:", mem_path)


# ─────────── Plot B: Runtime ───────────
fig_rt, ax_rt = plt.subplots(figsize=(14 / 3, 4))
ax_rt.errorbar(
    vox, g["t_mean"] / 60, yerr=g["t_std"] / 60, marker="d", color="tab:green"
)
ax_rt.set_ylabel("Runtime (min)")
ax_rt.set_xlabel("EGS voxel size (m)")
ax_rt.grid(alpha=0.3, linestyle=":")

# secondary x-axis for point counts
ax_pts = ax_rt.twiny()
ax_pts.set_xlim(ax_rt.get_xlim())
ax_pts.set_xticks(vox)


def fmt(n):
    return f"{n/1e6:.1f} M" if n >= 1e6 else f"{int(n/1e3)} k"


ax_pts.set_xticklabels([fmt(v) for v in g["src_mean"]])
ax_pts.set_xlabel("# points after down-sampling")
ax_rt.set_xlabel("EGS voxel size (m)")

fig_rt.tight_layout()
rt_path = run_dir / "egs_runtime.pdf"
fig_rt.savefig(rt_path)
print("Saved runtime plot to:", rt_path)

# ─────────── Plot C: Accuracy as Boxplots ───────────
import numpy as np

# get unique, sorted voxel sizes
vox = np.array(sorted(df["voxel_size"].unique()))

# collect raw per-run accuracy values
data_rre = [df.loc[df["voxel_size"] == v, "rre_deg"].values for v in vox]
data_rte = [df.loc[df["voxel_size"] == v, "rte_m"].values for v in vox]

# compute offsets for side-by-side boxplots
if len(vox) > 1:
    dmin = np.min(np.diff(vox))
else:
    dmin = 1.0
width = dmin * 0.3
offset = dmin * 0.2

fig_acc, ax_err = plt.subplots(figsize=(14 / 3, 4))
ax_rte = ax_err.twinx()

# RRE boxplots (degrees)
b1 = ax_err.boxplot(
    data_rre,
    positions=vox - offset,
    widths=width,
    patch_artist=True,
    boxprops=dict(facecolor="tab:brown", alpha=0.5),
    medianprops=dict(color="black"),
)

# RTE boxplots (meters)
b2 = ax_rte.boxplot(
    data_rte,
    positions=vox + offset,
    widths=width,
    patch_artist=True,
    boxprops=dict(facecolor="tab:gray", alpha=0.5),
    medianprops=dict(color="black"),
)

# axis labels & styling
ax_err.set_xlabel("EGS voxel size (m)")
ax_err.set_ylabel("RRE (°)", color="tab:brown")
ax_err.tick_params(axis="y", labelcolor="tab:brown")
ax_rte.set_ylabel("RTE (m)", color="tab:gray")
ax_rte.tick_params(axis="y", labelcolor="tab:gray")

# x‐ticks exactly at your voxel sizes
ax_err.set_xticks(vox)
ax_err.set_xticklabels([f"{v:g}" for v in vox])
ax_err.grid(alpha=0.3, linestyle=":")

# secondary x-axis for point counts (reuse your fmt)
ax_pts = ax_err.twiny()
ax_pts.set_xlim(ax_err.get_xlim())
ax_pts.set_xticks(vox)
ax_pts.set_xticklabels([fmt(n) for n in g["src_mean"]])
ax_pts.set_xlabel("# points after down-sampling")

fig_acc.tight_layout()
acc_path = run_dir / "egs_accuracy.pdf"
fig_acc.savefig(acc_path)
print("Saved accuracy boxplot to:", acc_path)
