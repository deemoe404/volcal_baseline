from __future__ import annotations

import argparse
import datetime as _dt
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set

import geopandas as gpd
import laspy
import numpy as np
import open3d as o3d
import pandas as pd
import psutil
import torch

import pipeline

proc = psutil.Process(os.getpid())


# ─────────────────────────────────────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ExperimentConfig:
    dataset_root: Path = Path("/home/sam/Documents/datasets/WHU/3-Mountain")
    output_root: Path = Path("outputs")

    device: torch.device = torch.device("cuda")
    num_workers: int = 24

    # Sweep list – only this changes
    voxel_sizes: List[float] = field(default_factory=lambda: [1.0, 1.5, 2.0, 2.5, 3.0])

    # Fixed parameters (same as your current script)
    padding: str = "same"
    pv: int = 5
    nv: int = -1
    ppv: int = -1
    rotation_choice: str = "gen"

    # Synthetic deformation parameters
    radius: int = 25
    max_depth: int = 15
    noise_std: float = 0.05
    rotation_z: int = 30
    translation: int = 5

    repeat: int = 12  # repetitions per voxel size

    center_offset_xy: float = 15.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. Logging helpers
# ─────────────────────────────────────────────────────────────────────────────


def _setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "run.log"

    logger = logging.getLogger("egs_bench")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_file, mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────────────────────────────────────
# 3. Resume helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_config_id(voxel_size: float, run_idx: int) -> str:
    return f"{voxel_size:.3f}-{run_idx:02d}"


def _load_existing_results(run_dir: Path) -> tuple[List[Dict], Set[str]]:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return [], set()
    df_old = pd.read_csv(csv_path)
    return df_old.to_dict("records"), set(df_old["config_id"].tolist())


def _append_result(run_dir: Path, row: Dict, header: bool = False):
    csv_path = run_dir / "results.csv"
    pd.DataFrame([row]).to_csv(csv_path, mode="a", index=False, header=header)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Data loading
# ─────────────────────────────────────────────────────────────────────────────


def load_point_clouds(cfg: ExperimentConfig, logger: logging.Logger):
    las_path = cfg.dataset_root / "combined.las"
    shp_path = cfg.dataset_root / "combined_stable.shp"

    logger.info("Loading LAS: %s", las_path)
    las = laspy.read(las_path)
    target = np.column_stack((las.x, las.y, las.z))

    logger.info("Loading SHP: %s", shp_path)
    stable = gpd.read_file(shp_path)
    return target, stable


# ─────────────────────────────────────────────────────────────────────────────
# 5. Single run
# ─────────────────────────────────────────────────────────────────────────────


def run_single(
    cfg: ExperimentConfig,
    voxel_size: float,
    run_idx: int,
    target: np.ndarray,
    stable_shp: gpd.GeoDataFrame,
    logger: logging.Logger,
) -> Dict:

    # --- synthetic deformation ----------------------------------------
    x, y, z = target.T
    center_x = x.mean() + cfg.center_offset_xy
    center_y = y.mean() + cfg.center_offset_xy
    sigma = cfg.radius / 2.0

    z_def, _ = pipeline.apply_deformation(
        x, y, z, center_x, center_y, cfg.radius, cfg.max_depth, sigma
    )
    x_n, y_n, z_n = pipeline.apply_noise(x, y, z_def, cfg.noise_std)

    T_rand = pipeline.get_random_transformation(
        angle_range_z=(
            (-(cfg.rotation_z + 2.5), -(cfg.rotation_z - 2.5)),
            (cfg.rotation_z - 2.5, cfg.rotation_z + 2.5),
        ),
        translation_range=(
            (-(cfg.translation + 0.5), -(cfg.translation - 0.5)),
            (cfg.translation - 0.5, cfg.translation + 0.5),
        ),
    )
    gt_T = np.linalg.inv(T_rand)
    x_t, y_t, z_t = pipeline.apply_transformation(x_n, y_n, z_n, T_rand)
    source = np.column_stack((x_t, y_t, z_t))

    # --- memory & timing metrics --------------------------------------
    process = psutil.Process(os.getpid())
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(cfg.device)
    torch.cuda.synchronize()

    t0 = _dt.datetime.now()
    egs_T = pipeline.EGS(
        source=source,
        target=target,
        voxel_size=voxel_size,
        padding=cfg.padding,
        ppv=cfg.ppv,
        pv=cfg.pv,
        nv=cfg.nv,
        num_workers=cfg.num_workers,
        rotation_choice=cfg.rotation_choice,
        rotation_root_path="exhaustive-grid-search/data/rotations",
    )
    torch.cuda.synchronize()
    t1 = _dt.datetime.now()

    peak_gpu_gb = torch.cuda.max_memory_allocated(cfg.device) / 1024**3
    rss_mb = process.memory_info().rss / 1024**2  # MiB

    procs = [proc] + proc.children(recursive=True)
    rss_total = sum(p.memory_info().rss for p in procs) / 1024**2

    egs_secs = (t1 - t0).total_seconds()

    # --- accuracy ------------------------------------------------------
    rre = pipeline.compute_rre(egs_T, gt_T)
    rte = pipeline.compute_rte(egs_T, gt_T)

    # --- point‑count after down‑sampling -------------------------------
    src_ds = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(source))
    src_ds = src_ds.voxel_down_sample(voxel_size)
    tgt_ds = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(target))
    tgt_ds = tgt_ds.voxel_down_sample(voxel_size)

    return {
        "config_id": _make_config_id(voxel_size, run_idx),
        "voxel_size": voxel_size,
        "run_idx": run_idx,
        "n_points_src": len(src_ds.points),
        "n_points_tgt": len(tgt_ds.points),
        "egs_time_s": egs_secs,
        "egs_peak_gpu_gb": peak_gpu_gb,
        "egs_cpu_rss_mb": rss_total,
        "rre_deg": rre,
        "rte_m": rte,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="EGS voxel-size benchmark")
    parser.add_argument(
        "--run-dir", type=Path, default=None, help="Existing output dir → resume mode"
    )
    args = parser.parse_args()

    cfg = ExperimentConfig()

    # --- output dir ----------------------------------------------------
    if args.run_dir is None:
        ts = _dt.datetime.now().strftime("%Y%m%d-%H%M")
        run_dir = cfg.output_root / ts
        resume = False
    else:
        run_dir = args.run_dir
        resume = True

    logger = _setup_logging(run_dir)
    logger.info("%s benchmark → %s", "RESUME" if resume else "START", run_dir.resolve())

    previous, done_ids = _load_existing_results(run_dir)
    if done_ids:
        logger.info("Found %d finished configs – skipping.", len(done_ids))

    # --- data ----------------------------------------------------------
    target, stable_shp = load_point_clouds(cfg, logger)

    # --- sweep ---------------------------------------------------------
    combinations = [(_vs, i) for _vs in cfg.voxel_sizes for i in range(cfg.repeat)]
    total = len(combinations)

    first_write = not (run_dir / "results.csv").exists()
    processed = 0
    for voxel_size, run_idx in combinations:
        cid = _make_config_id(voxel_size, run_idx)
        if cid in done_ids:
            continue
        processed += 1
        logger.info(
            "[%d/%d] v=%.2f m  repeat=%d",
            processed,
            total - len(done_ids),
            voxel_size,
            run_idx,
        )
        try:
            row = run_single(cfg, voxel_size, run_idx, target, stable_shp, logger)
            _append_result(run_dir, row, header=first_write)
            first_write = False
        except Exception:
            logger.exception("Failure in config %s", cid)

    logger.info("✅ All done – results saved to %s", run_dir)


if __name__ == "__main__":
    main()
