from __future__ import annotations

import argparse
import datetime as _dt
import logging
import sys
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Dict, List, Set, Tuple

import geopandas as gpd
import laspy
import numpy as np
import open3d as o3d
import pandas as pd
import py4dgeo
import torch
from sklearn.neighbors import NearestNeighbors

import pipeline


# ─────────────────────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ExperimentConfig:
    dataset_root: Path = Path("datasets/WHU/3-Mountain")
    output_root: Path = Path("outputs")

    device: torch.device = torch.device("cuda")
    num_workers: int = 22

    voxel_size: float = 2.0
    padding: str = "same"
    pv: int = 5
    nv: int = -1
    ppv: int = -1
    rotation_choice: str = "gen"
    max_iter: int = 2048

    # G-ICP quantile list
    quantile_thresholds: List[float] = field(
        default_factory=lambda: [0.90, 0.70, 0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10]
    )

    # noise std list
    noise_std_list: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1])

    # deformation parameters list (will be ziped)
    radius_list: List[int] = field(default_factory=lambda: [25])
    max_depth_list: List[int] = field(default_factory=lambda: [15])

    # rotation/translation parameters list (will be ziped)
    rotation_z_list: List[int] = field(default_factory=lambda: [5, 30, 45])
    translation_list: List[int] = field(default_factory=lambda: [1, 5, 10])

    # Repeat count for each configuration
    repeat: int = 1

    center_offset_xy: float = 15.0
    lodetection_max_distance: float = 15.0


# ─────────────────────────────────────────────────────────────
# 2. One-time pre-computation
# ─────────────────────────────────────────────────────────────
@dataclass
class Precomp:
    adapted_voxel_size: float
    search_radius: float
    neigh: NearestNeighbors
    stable_before: np.ndarray
    epoch_stable_before: py4dgeo.Epoch
    corepoints: np.ndarray
    gt_volumes: Dict[Tuple[int, int], float]


# ─────────────────────────────────────────────────────────────
# 3. Logging
# ─────────────────────────────────────────────────────────────
def _setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "run.log"

    logger = logging.getLogger("experiment")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_file, mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# ─────────────────────────────────────────────────────────────
# 4. Checkpoint resume helper
# ─────────────────────────────────────────────────────────────
def _make_config_id(
    q_thr: float,
    radius: int,
    max_depth: int,
    noise_std: float,
    rot_z: int,
    trans: int,
) -> str:
    return f"{q_thr:.3f},{radius},{max_depth},{noise_std:.3f},{rot_z},{trans}"


def _load_existing_results(run_dir: Path) -> tuple[List[Dict], Set[str]]:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return [], set()

    df_old = pd.read_csv(csv_path)
    done_ids = {
        _make_config_id(
            row["quantile_thr"],
            row["radius"],
            row["max_depth"],
            row["noise_std"],
            row["rotation_z"],
            row["translation"],
        )
        for _, row in df_old.iterrows()
    }
    return df_old.to_dict("records"), done_ids


def _append_result_to_csv(
    run_dir: Path, result_row: Dict, header: bool = False
) -> None:
    csv_path = run_dir / "results.csv"
    pd.DataFrame([result_row]).to_csv(csv_path, mode="a", index=False, header=header)


# ─────────────────────────────────────────────────────────────
# 5. Data loading
# ─────────────────────────────────────────────────────────────
def load_point_clouds(cfg: ExperimentConfig, logger: logging.Logger):
    las_path = cfg.dataset_root / "combined.las"
    shp_path = cfg.dataset_root / "combined_stable.shp"

    logger.info("Loading LAS %s", las_path)
    las = laspy.read(las_path)
    target = np.column_stack((las.x, las.y, las.z))

    logger.info("Loading stable shapefile %s", shp_path)
    stable_shp = gpd.read_file(shp_path)
    return target, stable_shp


# ─────────────────────────────────────────────────────────────
# 6. Run single configuration
# ─────────────────────────────────────────────────────────────
def run_single_configuration(
    cfg: ExperimentConfig,
    pre: Precomp,
    logger: logging.Logger,
    target: np.ndarray,
    stable_shp: gpd.GeoDataFrame,
    *,
    quantile_thr: float,
    radius: int,
    max_depth: int,
    noise_std: float,
    rotation_z: int,
    translation: int,
) -> Dict:
    """Run one param combo; return flat dict."""

    # ---------- Deformation + noise ----------
    x, y, z = target.T
    center_x = x.mean() + cfg.center_offset_xy
    center_y = y.mean() + cfg.center_offset_xy
    sigma = radius / 2.0
    gt_volume = pre.gt_volumes[(radius, max_depth)]

    z_def, _ = pipeline.apply_deformation(
        x, y, z, center_x, center_y, radius, max_depth, sigma
    )
    x_n, y_n, z_n = pipeline.apply_noise(x, y, z_def, noise_std)

    T_rand = pipeline.get_random_transformation(
        angle_range_z=(
            (-(rotation_z + 2.5), -(rotation_z - 2.5)),
            (rotation_z - 2.5, rotation_z + 2.5),
        ),
        translation_range=(
            (-(translation + 0.5), -(translation - 0.5)),
            (translation - 0.5, translation + 0.5),
        ),
    )
    gt_T = np.linalg.inv(T_rand)
    x_t, y_t, z_t = pipeline.apply_transformation(x_n, y_n, z_n, T_rand)
    source = np.column_stack((x_t, y_t, z_t))

    start_ts = _dt.datetime.now()

    # ---------- Initial registration (EGS) ----------
    egs_T = pipeline.EGS(
        source=source,
        target=target,
        voxel_size=cfg.voxel_size,
        padding=cfg.padding,
        ppv=cfg.ppv,
        pv=cfg.pv,
        nv=cfg.nv,
        num_workers=cfg.num_workers,
        rotation_choice=cfg.rotation_choice,
        rotation_root_path="exhaustive-grid-search/data/rotations",
    )
    EGS_ts = _dt.datetime.now()
    rre_init = pipeline.compute_rre(egs_T, gt_T)
    rte_init = pipeline.compute_rte(egs_T, gt_T)

    # ---------- Fine registration (G-ICP) ----------
    gicp_T = pipeline.auto_GICP_TEST(
        source=source,
        target=target,
        T_init=egs_T,
        thr=quantile_thr,
        neigh=pre.neigh,
        max_iter=cfg.max_iter,
    )
    GICP_ts = _dt.datetime.now()
    rre_ref = pipeline.compute_rre(gicp_T, gt_T)
    rte_ref = pipeline.compute_rte(gicp_T, gt_T)

    # ---------- Apply refined transformation ----------
    x_r, y_r, z_r = pipeline.apply_transformation(x_t, y_t, z_t, gicp_T)
    refined = np.column_stack((x_r, y_r, z_r))

    # ---------- Uncertainty estimation ----------
    adapted_voxel_size = pre.adapted_voxel_size
    search_radius = pre.search_radius

    epoch_stable_after = py4dgeo.Epoch(pipeline.isolate_stable(refined, stable_shp)[0])

    m3c2_stable = py4dgeo.M3C2(
        epochs=(pre.epoch_stable_before, epoch_stable_after),
        corepoints=pre.epoch_stable_before.cloud[:],
        normal_radii=(search_radius,),
        cyl_radius=(adapted_voxel_size),
        max_distance=(cfg.lodetection_max_distance),
        registration_error=(0.0),
    )
    m3c2_distances_stable, _ = m3c2_stable.run()
    reg_unc = np.nanstd(m3c2_distances_stable)

    # ---------- Overall scene M3C2 Change Detection ----------
    epoch_before = py4dgeo.Epoch(target)
    epoch_after = py4dgeo.Epoch(refined)

    m3c2_full = py4dgeo.M3C2(
        epochs=(epoch_before, epoch_after),
        corepoints=pre.corepoints,
        normal_radii=(search_radius,),
        cyl_radius=(adapted_voxel_size),
        max_distance=(cfg.lodetection_max_distance),
        registration_error=(reg_unc),
    )
    m3c2_dist, uncert = m3c2_full.run()
    change_sign = np.abs(m3c2_dist) > uncert["lodetection"]

    # ---------- Compare with GT circles ----------
    dists_center = np.hypot(
        pre.corepoints[:, 0] - center_x, pre.corepoints[:, 1] - center_y
    )
    gt_mask = dists_center <= radius

    EPS = 1e-9
    tp = np.count_nonzero(change_sign & gt_mask)
    fp = np.count_nonzero(change_sign & ~gt_mask)
    fn = np.count_nonzero(~change_sign & gt_mask)

    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    iou = tp / (tp + fp + fn + EPS)
    miscls = (fp + fn) / pre.corepoints.shape[0]

    lod_vals = uncert["lodetection"]
    lod_in = np.nanmean(lod_vals[gt_mask])
    lod_out = np.nanmean(lod_vals[~gt_mask])

    # ---------- Cut & fill volume ----------
    hulls = pipeline.segment_changes(pre.corepoints, change_sign)
    sel_idx = [np.argmax([h.area for h in hulls])]
    sel_hulls = [hulls[i] for i in sel_idx]

    in_refined = pipeline.is_inside_selected_hulls_vectorized(sel_hulls, refined[:, :2])
    in_raw = pipeline.is_inside_selected_hulls_vectorized(sel_hulls, target[:, :2])

    dem_b, dem_a, _, _, grid_res = pipeline.reletive_DEM(
        target[in_raw],
        refined[in_refined],
        grid_res=None,
        method="linear",
        mask_hulls=sel_hulls,
    )

    net_vol, cut_vol, fill_vol, _ = pipeline.calculate_volume(
        dem_b, dem_a, grid_res=grid_res, threshold=reg_unc
    )

    vol_cut_acc = (gt_volume + net_vol) / gt_volume * 100

    return {
        # sweep parameters
        "radius": radius,
        "max_depth": max_depth,
        "noise_std": noise_std,
        "rotation_z": rotation_z,
        "translation": translation,
        "quantile_thr": quantile_thr,
        # initial / refined transformation error
        "rre": rre_init,
        "rte": rte_init,
        "rre_refined": rre_ref,
        "rte_refined": rte_ref,
        # volume
        "cut_volume": cut_vol,
        "fill_volume": fill_vol,
        "net_volume": net_vol,
        "volume_cut_acc": vol_cut_acc,
        "reg_target_source": reg_unc,
        # timestamps
        "start_timestamp": start_ts,
        "EGS_timestamp": EGS_ts,
        "GICP_timestamp": GICP_ts,
        # M3C2 benchmarks
        "cd_tp": tp,
        "cd_fp": fp,
        "cd_fn": fn,
        "cd_precision": precision,
        "cd_recall": recall,
        "cd_iou": iou,
        "cd_miscls": miscls,
        # LOD
        "LOD_mean_in": lod_in,
        "LOD_mean_out": lod_out,
    }


# ─────────────────────────────────────────────────────────────
# 7. Main function
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Run & resume batch experiments")
    parser.add_argument(
        "--run-dir", type=Path, default=None, help="Directory to resume from"
    )
    args = parser.parse_args()
    cfg = ExperimentConfig()

    # ----- Output directory / Resume -----
    if args.run_dir is None:
        ts = _dt.datetime.now().strftime("%Y%m%d-%H%M")
        run_dir = cfg.output_root / ts
        is_resume = False
    else:
        run_dir = args.run_dir
        is_resume = True

    logger = _setup_logging(run_dir)
    logger.info(
        "🚀 %s run  →  %s", "RESUME" if is_resume else "START", run_dir.resolve()
    )

    prev_rows, done_ids = _load_existing_results(run_dir)
    if done_ids:
        logger.info("Found %d completed configs – skip them", len(done_ids))

    # ----- Load point clouds -----
    target, stable_shp = load_point_clouds(cfg, logger)

    # ----- Pre-compute parameters -----
    adapted_voxel = pipeline.adaptive_voxel_size(
        target, 0.01, pipeline.estimate_avg_spacing(target) / 0.01, 25, 15, 1
    )
    search_r = adapted_voxel * 2.0
    neigh = NearestNeighbors(n_neighbors=1).fit(target)

    stable_before = pipeline.isolate_stable(target, stable_shp)[0]
    epoch_stable_before = py4dgeo.Epoch(stable_before)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(target)
    pcd = pcd.voxel_down_sample(voxel_size=adapted_voxel)
    corepts = np.asarray(pcd.points)

    gt_vols = {
        (r, d): pipeline.get_analytical_volume(r / 2.0, r, d)
        for r in cfg.radius_list
        for d in cfg.max_depth_list
    }

    pre = Precomp(
        adapted_voxel_size=adapted_voxel,
        search_radius=search_r,
        neigh=neigh,
        stable_before=stable_before,
        epoch_stable_before=epoch_stable_before,
        corepoints=corepts,
        gt_volumes=gt_vols,
    )

    # ----- sweep -----
    paired_rd = list(zip(cfg.radius_list, cfg.max_depth_list))
    paired_mp = list(zip(cfg.rotation_z_list, cfg.translation_list))
    combos = list(
        product(
            cfg.quantile_thresholds,
            paired_rd,
            cfg.noise_std_list,
            paired_mp,
            range(cfg.repeat),
        )
    )
    total = len(combos)
    logger.info("Grid size = %d", total)

    first_write = not (run_dir / "results.csv").exists()
    counter = 0
    for q_thr, (radius, max_depth), noise_std, (rot_z, trans), _ in combos:

        cfg_id = _make_config_id(q_thr, radius, max_depth, noise_std, rot_z, trans)
        if cfg_id in done_ids:
            continue

        counter += 1
        logger.info(
            "[%d/%d] q=%.2f r=%d d=%d n=%.3f rot=%d t=%d",
            counter,
            total - len(done_ids),
            q_thr,
            radius,
            max_depth,
            noise_std,
            rot_z,
            trans,
        )
        try:
            result = run_single_configuration(
                cfg,
                pre,
                logger,
                target,
                stable_shp,
                quantile_thr=q_thr,
                radius=radius,
                max_depth=max_depth,
                noise_std=noise_std,
                rotation_z=rot_z,
                translation=trans,
            )
            _append_result_to_csv(run_dir, result, header=first_write)
            first_write = False
        except Exception:
            logger.exception("Failure in cfg %s – skipped", cfg_id)

    logger.info("✅ Finished – outputs at %s", run_dir)


if __name__ == "__main__":
    main()
