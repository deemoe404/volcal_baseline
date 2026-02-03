import argparse
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

_mpl_tmp = Path(tempfile.gettempdir()) / "matplotlib"
_mpl_tmp.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_tmp))

import numpy as np
import open3d as o3d
import py4dgeo
from plyfile import PlyData
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from shapely.geometry import MultiPoint

import geopandas as gpd
from shapely.ops import unary_union
from shapely.vectorized import contains


DEFAULT_RC_TRANSFORM = np.array(
    [
        [0.016997808591, -0.006867627613, -0.002250073710, 61.350204467773],
        [-0.006642486434, -0.012581477873, -0.011778643355, 815.083801269531],
        [-0.002846842399, -0.011648809537, 0.014048252255, 1039.387695312500],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

DEFAULT_TERRA_TRANSFORM = np.array(
    [
        [-0.014011225663, 0.004350652453, -0.011145155877, 53.568080902100],
        [0.010189454071, -0.004655822180, -0.014627216384, 788.437316894531],
        [0.006270382553, 0.017287345603, -0.001134529477, 1048.019042968750],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

DEFAULT_TERRA_FINE_TRANSFORM = np.array(
    [
        [0.999996066093, -0.002269646619, 0.001640063012, -0.547948300838],
        [0.002271376317, 0.999996840954, -0.001053676475, -1.387025117874],
        [-0.001637666370, 0.001057397574, 0.999998092651, 3.152194261551],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class TerraRcMetrics:
    block: str
    v_net_abs_m3: float
    median_abs_d_m: float
    p95_abs_d_m: float
    rmse95_m: float
    pct_within_010m: float


def _read_ply_xyz(path: Path) -> np.ndarray:
    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    x = np.asarray(vertex["x"], dtype=np.float64)
    y = np.asarray(vertex["y"], dtype=np.float64)
    z = np.asarray(vertex["z"], dtype=np.float64)
    return np.stack([x, y, z], axis=1)


def _apply_affine(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    return points @ R.T + t


def _center(points: np.ndarray) -> np.ndarray:
    return points - points.mean(axis=0, keepdims=True)


def _voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    ds = pcd.voxel_down_sample(voxel_size=float(voxel_size))
    return np.asarray(ds.points)


def _estimate_avg_spacing(points: np.ndarray, sample_size: int, seed: int) -> float:
    if sample_size > 0 and points.shape[0] > sample_size:
        rng = np.random.default_rng(seed)
        idx = rng.choice(points.shape[0], size=sample_size, replace=False)
        pts = points[idx]
    else:
        pts = points
    tree = cKDTree(pts[:, :3])
    distances, _ = tree.query(pts[:, :3], k=2)
    return float(np.mean(distances[:, 1]))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _adaptive_voxel_size(
    points: np.ndarray,
    ref_ratio: float,
    init_voxel_size: float,
    *,
    kp: float = 25,
    ki: float = 15,
    lr: float = 1.0,
    tol: float = 0.001,
    max_iter: int = 64,
) -> float:
    voxel_size = float(init_voxel_size)
    error_sum = 0.0
    scale = 1.0
    for _ in range(max_iter):
        down_points = _voxel_downsample(points, voxel_size)
        ratio = len(down_points) / len(points)
        error = ref_ratio - ratio
        if abs(error) < tol:
            break
        error_sum += error
        diff = kp * error + ki * error_sum
        scale = scale - lr * (_sigmoid(diff) - 0.5)
        voxel_size = init_voxel_size * float(np.exp(scale))
    return float(voxel_size)


def _fine_align_gicp(
    *,
    source: np.ndarray,
    target: np.ndarray,
    voxel_size: float,
    distance_quantile: float = 0.9,
    max_iter: int = 128,
) -> np.ndarray:
    source_ds = _voxel_downsample(source, voxel_size)
    target_ds = _voxel_downsample(target, voxel_size)
    if source_ds.shape[0] < 50 or target_ds.shape[0] < 50:
        raise RuntimeError("Too few points after downsampling for GICP")

    tree = cKDTree(target_ds)
    dists, _ = tree.query(source_ds, k=1)
    max_corr = float(np.quantile(dists, distance_quantile))
    if not np.isfinite(max_corr) or max_corr <= 0:
        raise RuntimeError(f"Invalid max correspondence distance: {max_corr}")

    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_ds)
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_ds)

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=int(max_iter)
    )
    reg = o3d.pipelines.registration.registration_generalized_icp(
        source_pcd,
        target_pcd,
        max_corr,
        np.eye(4),
        criteria=criteria,
    )
    return reg.transformation


def _points_in_polygon(points_xy: np.ndarray, polygon) -> np.ndarray:
    return contains(polygon, points_xy[:, 0], points_xy[:, 1])


def _stable_points(points: np.ndarray, stable_union) -> np.ndarray:
    mask = _points_in_polygon(points[:, :2], stable_union)
    return points[mask]


def _is_inside_selected_hulls_vectorized(selected_hulls, pts_2d: np.ndarray) -> np.ndarray:
    union_hull = unary_union(selected_hulls)
    return contains(union_hull, pts_2d[:, 0], pts_2d[:, 1])


def _segment_changes(
    cp: np.ndarray, cs: np.ndarray, thr: float = -0.2, min_samples: int = 2
) -> list:
    sig_cp_2d = cp[cs][:, :2]
    if sig_cp_2d.shape[0] < 3:
        return []

    neigh = NearestNeighbors(n_neighbors=2)
    neigh.fit(sig_cp_2d)
    dist, _ = neigh.kneighbors(sig_cp_2d)
    second_nn_dists = dist[:, 1]

    average_eps = float(np.mean(second_nn_dists))
    deviation = float(np.std(second_nn_dists))
    adaptive_eps = average_eps + thr * deviation

    db = DBSCAN(eps=adaptive_eps, min_samples=min_samples).fit(sig_cp_2d)
    labels = db.labels_
    unique_labels = [lbl for lbl in set(labels) if lbl != -1]

    hulls = []
    for lbl in unique_labels:
        cluster_points = sig_cp_2d[labels == lbl]
        if len(cluster_points) >= 3:
            hulls.append(MultiPoint(cluster_points).convex_hull)
    return hulls


def _relative_dem(
    cloud_1: np.ndarray,
    cloud_2: np.ndarray,
    *,
    grid_res: float | None = None,
    method: str = "linear",
    mask_hulls: list | None = None,
    grid_res_factor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    if grid_res is None:
        tree_1 = cKDTree(cloud_1[:, :2])
        distances_1, _ = tree_1.query(cloud_1[:, :2], k=2)
        grid_res_1 = float(grid_res_factor * np.median(distances_1[:, 1]))

        tree_2 = cKDTree(cloud_2[:, :2])
        distances_2, _ = tree_2.query(cloud_2[:, :2], k=2)
        grid_res_2 = float(grid_res_factor * np.median(distances_2[:, 1]))
        grid_res = max(grid_res_1, grid_res_2)

    min_z = min(float(cloud_1[:, 2].min()), float(cloud_2[:, 2].min()))
    z1_adjusted = cloud_1[:, 2] - min_z
    z2_adjusted = cloud_2[:, 2] - min_z

    x_min = min(float(cloud_1[:, 0].min()), float(cloud_2[:, 0].min()))
    x_max = max(float(cloud_1[:, 0].max()), float(cloud_2[:, 0].max()))
    y_min = min(float(cloud_1[:, 1].min()), float(cloud_2[:, 1].min()))
    y_max = max(float(cloud_1[:, 1].max()), float(cloud_2[:, 1].max()))

    grid_x, grid_y = np.meshgrid(
        np.arange(x_min, x_max, grid_res), np.arange(y_min, y_max, grid_res)
    )

    dem1 = griddata(
        (cloud_1[:, 0], cloud_1[:, 1]), z1_adjusted, (grid_x, grid_y), method=method
    )
    dem2 = griddata(
        (cloud_2[:, 0], cloud_2[:, 1]), z2_adjusted, (grid_x, grid_y), method=method
    )

    if mask_hulls is not None:
        union = unary_union(mask_hulls)
        mask_grid = contains(union, grid_x, grid_y)
        dem1[~mask_grid] = np.nan
        dem2[~mask_grid] = np.nan

    return dem1, dem2, grid_x, grid_y, float(grid_res)


def _calculate_volume(
    dem_before: np.ndarray,
    dem_after: np.ndarray,
    *,
    grid_res: float,
    threshold: float = 0.0,
) -> tuple[float, float, float, np.ndarray]:
    diff = dem_after - dem_before
    diff[np.abs(diff) < threshold] = 0.0

    cell_area = float(grid_res) ** 2
    cut_volume = float(np.nansum(np.abs(diff[diff < 0])) * cell_area)
    fill_volume = float(np.nansum(diff[diff > 0]) * cell_area)
    net_volume = float(fill_volume - cut_volume)
    return net_volume, cut_volume, fill_volume, diff


def _raster_mean_dem(
    cloud: np.ndarray,
    *,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    min_z: float,
) -> np.ndarray:
    nx = int(len(x_edges) - 1)
    ny = int(len(y_edges) - 1)
    if nx <= 0 or ny <= 0:
        raise ValueError(f"Invalid grid size: nx={nx}, ny={ny}")

    x = cloud[:, 0]
    y = cloud[:, 1]
    z = cloud[:, 2] - float(min_z)

    ix = np.searchsorted(x_edges, x, side="right") - 1
    iy = np.searchsorted(y_edges, y, side="right") - 1
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & np.isfinite(z)
    if not np.any(valid):
        return np.full((ny, nx), np.nan, dtype=np.float32)

    flat = (iy[valid].astype(np.int64) * nx + ix[valid].astype(np.int64)).astype(
        np.int64
    )
    z_valid = z[valid].astype(np.float64)

    sum_z = np.bincount(flat, weights=z_valid, minlength=nx * ny).astype(np.float64)
    count = np.bincount(flat, minlength=nx * ny).astype(np.int64)

    dem = np.full(nx * ny, np.nan, dtype=np.float32)
    nonzero = count > 0
    dem[nonzero] = (sum_z[nonzero] / count[nonzero]).astype(np.float32)
    return dem.reshape((ny, nx))


def _volume_on_overlap(
    terra: np.ndarray, rc: np.ndarray, *, grid_res: float, threshold: float
) -> float:
    x_min = max(float(terra[:, 0].min()), float(rc[:, 0].min()))
    x_max = min(float(terra[:, 0].max()), float(rc[:, 0].max()))
    y_min = max(float(terra[:, 1].min()), float(rc[:, 1].min()))
    y_max = min(float(terra[:, 1].max()), float(rc[:, 1].max()))
    if x_max <= x_min or y_max <= y_min:
        raise RuntimeError("No XY overlap between Terra and RC clouds")

    x_edges = np.arange(x_min, x_max + grid_res, grid_res, dtype=np.float64)
    y_edges = np.arange(y_min, y_max + grid_res, grid_res, dtype=np.float64)
    if len(x_edges) < 2 or len(y_edges) < 2:
        raise RuntimeError("Overlap grid too small to rasterize")

    min_z = min(float(terra[:, 2].min()), float(rc[:, 2].min()))
    dem_terra = _raster_mean_dem(terra, x_edges=x_edges, y_edges=y_edges, min_z=min_z)
    dem_rc = _raster_mean_dem(rc, x_edges=x_edges, y_edges=y_edges, min_z=min_z)

    overlap = np.isfinite(dem_terra) & np.isfinite(dem_rc)
    if not np.any(overlap):
        raise RuntimeError("No overlapping DEM cells with valid data")

    diff = (dem_rc - dem_terra).astype(np.float32)
    diff[~overlap] = np.nan
    diff[np.abs(diff) < float(threshold)] = 0.0

    cell_area = float(grid_res) ** 2
    cut_volume = float(np.nansum(np.abs(diff[diff < 0])) * cell_area)
    fill_volume = float(np.nansum(diff[diff > 0]) * cell_area)
    net_volume = float(fill_volume - cut_volume)
    return float(abs(net_volume))


def _volume_on_roi(
    *,
    terra: np.ndarray,
    rc: np.ndarray,
    corepoints: np.ndarray,
    distances: np.ndarray,
    lodetection: np.ndarray,
    reg_error: float,
    min_roi_points: int = 10,
    roi_buffer_m: float = 0.0,
    roi_buffer_step_m: float | None = None,
    roi_buffer_max_m: float | None = None,
    verbose: bool = False,
) -> float:
    valid = np.isfinite(distances) & np.isfinite(lodetection)
    cs = np.zeros_like(distances, dtype=bool)
    cs[valid] = np.abs(distances[valid]) > lodetection[valid]

    hulls = _segment_changes(corepoints, cs)
    if len(hulls) == 0:
        raise RuntimeError("No change hulls found; cannot compute ROI volume")

    areas = np.array([h.area for h in hulls], dtype=float)
    selected = [hulls[int(np.argmax(areas))]]

    step = float(roi_buffer_step_m) if roi_buffer_step_m is not None else 0.0
    if not np.isfinite(step) or step <= 0:
        step = 0.5

    max_buf = float(roi_buffer_max_m) if roi_buffer_max_m is not None else max(2.0, 4.0 * step)
    if not np.isfinite(max_buf) or max_buf < 0:
        max_buf = 0.0

    requested_buf = float(roi_buffer_m)
    if not np.isfinite(requested_buf) or requested_buf < 0:
        requested_buf = 0.0

    if requested_buf > 0:
        candidates = [requested_buf]
    else:
        candidates = [0.0]
        b = step
        while b <= max_buf + 1e-9:
            candidates.append(float(b))
            b += step

    terra_roi = None
    rc_roi = None
    selected_used = None
    used_buf = None
    last_counts = None

    for b in candidates:
        buffered = selected if b <= 0 else [h.buffer(float(b)) for h in selected]
        inside_terra = _is_inside_selected_hulls_vectorized(buffered, terra[:, :2])
        inside_rc = _is_inside_selected_hulls_vectorized(buffered, rc[:, :2])
        terra_roi_try = terra[inside_terra]
        rc_roi_try = rc[inside_rc]
        last_counts = (int(terra_roi_try.shape[0]), int(rc_roi_try.shape[0]))
        if verbose:
            print(f"[debug] roi_buffer_try={b:.3f} terra_roi={last_counts[0]} rc_roi={last_counts[1]}")
        if terra_roi_try.shape[0] >= min_roi_points and rc_roi_try.shape[0] >= min_roi_points:
            terra_roi = terra_roi_try
            rc_roi = rc_roi_try
            selected_used = buffered
            used_buf = float(b)
            break

    if terra_roi is None or rc_roi is None or selected_used is None:
        terra_n = last_counts[0] if last_counts else 0
        rc_n = last_counts[1] if last_counts else 0
        raise RuntimeError(
            f"Too few ROI points even after buffering (max_buf={candidates[-1]:.3f}): terra={terra_n}, rc={rc_n}"
        )

    dem_before, dem_after, _gx, _gy, grid_res = _relative_dem(
        terra_roi,
        rc_roi,
        grid_res=None,
        method="linear",
        mask_hulls=selected_used,
    )
    net_volume, _cut, _fill, _diff = _calculate_volume(
        dem_before, dem_after, grid_res=float(grid_res), threshold=float(reg_error)
    )
    if verbose and used_buf is not None and used_buf > 0:
        print(f"[debug] roi_buffer_used={used_buf:.3f}")
    return float(abs(net_volume))


def compute_metrics(
    *,
    block: str,
    terra_ply: Path,
    rc_ply: Path,
    stable_shp: Path,
    within_threshold_m: float = 0.10,
    ref_ratio: float = 0.01,
    max_distance_m: float = 15.0,
    spacing_sample: int = 200_000,
    voxel_sample: int = 200_000,
    seed: int = 0,
    initial_transform: Literal["default", "none"] = "default",
    do_center: bool = True,
    fine_align: Literal["gicp", "none", "manual"] = "gicp",
    volume_mode: Literal["roi", "overlap"] = "roi",
    roi_buffer_m: float = 0.0,
    manual_fine_transform: np.ndarray | None = None,
    verbose: bool = False,
) -> TerraRcMetrics:
    terra = _read_ply_xyz(terra_ply)
    rc = _read_ply_xyz(rc_ply)

    if initial_transform == "default":
        terra = _apply_affine(terra, DEFAULT_TERRA_TRANSFORM)
        rc = _apply_affine(rc, DEFAULT_RC_TRANSFORM)
    elif initial_transform == "none":
        pass
    else:
        raise ValueError(f"Unknown initial_transform: {initial_transform}")

    if do_center:
        terra = _center(terra)
        rc = _center(rc)

    if fine_align == "gicp":
        # Estimate a reasonable downsampling voxel for GICP from the unaligned Terra cloud
        avg_spacing_pre = _estimate_avg_spacing(
            terra, sample_size=spacing_sample, seed=seed
        )
        init_voxel_pre = avg_spacing_pre / ref_ratio
        if voxel_sample > 0 and terra.shape[0] > voxel_sample:
            rng = np.random.default_rng(seed)
            idx = rng.choice(terra.shape[0], size=voxel_sample, replace=False)
            voxel_fit_pre = terra[idx]
        else:
            voxel_fit_pre = terra

        adapted_voxel_pre = _adaptive_voxel_size(
            voxel_fit_pre, ref_ratio, init_voxel_pre, kp=25, ki=15, lr=1.0
        )
        if verbose:
            print(
                f"[debug] gicp: avg_spacing_pre={avg_spacing_pre:.6f} adapted_voxel_pre={adapted_voxel_pre:.6f}"
            )

        # Align Terra -> RC (matches exp_test.ipynb flow)
        T_fine = _fine_align_gicp(
            source=terra,
            target=rc,
            voxel_size=max(adapted_voxel_pre, 0.05),
            distance_quantile=0.9,
            max_iter=128,
        )
        terra = _apply_affine(terra, T_fine)
    elif fine_align == "manual":
        T_fine = manual_fine_transform
        if T_fine is None:
            T_fine = DEFAULT_TERRA_FINE_TRANSFORM
        terra = _apply_affine(terra, np.asarray(T_fine, dtype=np.float64))
    elif fine_align == "none":
        pass
    else:
        raise ValueError(f"Unknown fine_align: {fine_align}")

    # After fine alignment, estimate voxel size for M3C2 and DEM computations (matches exp_test.ipynb)
    avg_spacing = _estimate_avg_spacing(terra, sample_size=spacing_sample, seed=seed)
    init_voxel_size = avg_spacing / ref_ratio
    if verbose:
        print(
            f"[debug] avg_spacing={avg_spacing:.6f} init_voxel_size={init_voxel_size:.6f}"
        )

    if voxel_sample > 0 and terra.shape[0] > voxel_sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(terra.shape[0], size=voxel_sample, replace=False)
        voxel_fit_points = terra[idx]
    else:
        voxel_fit_points = terra

    adapted_voxel_size = _adaptive_voxel_size(
        voxel_fit_points, ref_ratio, init_voxel_size, kp=25, ki=15, lr=1.0
    )
    if verbose:
        print(
            f"[debug] adapted_voxel_size={adapted_voxel_size:.6f} (fit_points={len(voxel_fit_points)})"
        )

    stable_gdf = gpd.read_file(stable_shp)
    stable_union = unary_union(list(stable_gdf.geometry))

    stable_terra = _stable_points(terra, stable_union)
    stable_rc = _stable_points(rc, stable_union)
    if stable_terra.shape[0] < 50 or stable_rc.shape[0] < 50:
        raise RuntimeError(
            f"Too few stable points: terra={stable_terra.shape[0]}, rc={stable_rc.shape[0]}"
        )

    epoch_stable_terra = py4dgeo.Epoch(stable_terra)
    epoch_stable_rc = py4dgeo.Epoch(stable_rc)

    stable_corepoints = _voxel_downsample(
        epoch_stable_terra.cloud[::], voxel_size=float(adapted_voxel_size)
    )
    if verbose:
        print(f"[debug] stable_corepoints_n={len(stable_corepoints)}")

    m3c2_stable = py4dgeo.M3C2(
        epochs=(epoch_stable_terra, epoch_stable_rc),
        corepoints=stable_corepoints,
        normal_radii=[adapted_voxel_size * 2.0],
        cyl_radius=float(adapted_voxel_size),
        max_distance=float(max_distance_m),
        registration_error=0.0,
    )
    d_stable, _u_stable = m3c2_stable.run()
    reg_error = float(np.nanstd(d_stable))
    if not np.isfinite(reg_error) or reg_error <= 0:
        raise RuntimeError(f"Invalid registration error from stable M3C2: {reg_error}")
    if verbose:
        print(f"[debug] reg_error={reg_error:.6f} stable_n={len(stable_terra)}/{len(stable_rc)}")

    epoch_terra = py4dgeo.Epoch(terra)
    epoch_rc = py4dgeo.Epoch(rc)

    corepoints = _voxel_downsample(epoch_terra.cloud[::], voxel_size=adapted_voxel_size)
    if verbose:
        print(f"[debug] corepoints_n={len(corepoints)}")
    m3c2 = py4dgeo.M3C2(
        epochs=(epoch_terra, epoch_rc),
        corepoints=corepoints,
        normal_radii=[adapted_voxel_size * 2.0],
        cyl_radius=float(adapted_voxel_size),
        max_distance=float(max_distance_m),
        registration_error=float(reg_error),
    )
    d, uncertainties = m3c2.run()

    d = np.asarray(d)
    valid = np.isfinite(d)
    d_valid = d[valid]
    if d_valid.size == 0:
        raise RuntimeError("No valid M3C2 distances")

    abs_d = np.abs(d_valid)
    median_abs = float(np.percentile(abs_d, 50))
    p95_abs = float(np.percentile(abs_d, 95))

    central = abs_d <= p95_abs
    rmse95 = float(np.sqrt(np.mean((d_valid[central]) ** 2)))
    pct_within = float(np.mean(abs_d <= within_threshold_m) * 100.0)

    if volume_mode == "overlap":
        v_net_abs = _volume_on_overlap(
            terra,
            rc,
            grid_res=float(adapted_voxel_size),
            threshold=float(reg_error),
        )
    elif volume_mode == "roi":
        if isinstance(uncertainties, dict):
            lod = np.asarray(uncertainties.get("lodetection"))
        elif (
            isinstance(uncertainties, np.ndarray)
            and uncertainties.dtype.fields
            and "lodetection" in uncertainties.dtype.fields
        ):
            lod = np.asarray(uncertainties["lodetection"])
        else:
            raise RuntimeError(
                f"Unsupported uncertainties type for LoDetection: {type(uncertainties)}"
            )

        if lod is None:
            raise RuntimeError("Missing 'lodetection' in M3C2 uncertainties")
        if lod.shape != d.shape:
            raise RuntimeError(
                f"lodetection shape mismatch: d={d.shape}, lod={lod.shape}"
            )

        try:
            v_net_abs = _volume_on_roi(
                terra=terra,
                rc=rc,
                corepoints=corepoints,
                distances=d,
                lodetection=lod,
                reg_error=reg_error,
                roi_buffer_m=float(roi_buffer_m),
                roi_buffer_step_m=float(adapted_voxel_size),
                roi_buffer_max_m=None,
                verbose=bool(verbose),
            )
            if verbose:
                print(f"[debug] v_net_abs_roi={v_net_abs}")
        except RuntimeError as e:
            if verbose:
                print(f"[debug] ROI volume failed: {e}. Falling back to overlap.")
            # Fallback to overlap if ROI becomes degenerate
            v_net_abs = _volume_on_overlap(
                terra,
                rc,
                grid_res=float(adapted_voxel_size),
                threshold=float(reg_error),
            )
            if verbose:
                print(f"[debug] v_net_abs_overlap={v_net_abs}")
    else:
        raise ValueError(f"Unknown volume_mode: {volume_mode}")

    return TerraRcMetrics(
        block=block,
        v_net_abs_m3=v_net_abs,
        median_abs_d_m=median_abs,
        p95_abs_d_m=p95_abs,
        rmse95_m=rmse95,
        pct_within_010m=pct_within,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Terra vs on-controller metrics for one panorama.")
    parser.add_argument("--block", default="Panorama", help="Row label (e.g., 'Panorama~2').")
    parser.add_argument("--terra", required=True, type=Path, help="DJI Terra point cloud PLY (reference).")
    parser.add_argument("--rc", required=True, type=Path, help="On-controller point cloud PLY (comparison).")
    parser.add_argument(
        "--stable-shp",
        type=Path,
        default=Path("/Users/sam/Desktop/stable.shp"),
        help="Stable region polygon shapefile (used to estimate registration error).",
    )
    parser.add_argument("--out", type=Path, default=None, help="Write JSON metrics to this path.")
    parser.add_argument("--within", type=float, default=0.10, help="Threshold (m) for %% within ±threshold.")
    parser.add_argument("--ref-ratio", type=float, default=0.01, help="Target downsample ratio for corepoints.")
    parser.add_argument("--max-distance", type=float, default=15.0, help="M3C2 max distance (m).")
    parser.add_argument("--spacing-sample", type=int, default=200_000, help="Sample size for avg spacing estimate (0=all).")
    parser.add_argument("--voxel-sample", type=int, default=200_000, help="Sample size to fit adaptive voxel size (0=all).")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for sampling.")
    parser.add_argument(
        "--initial-transform",
        choices=["default", "none"],
        default="default",
        help="Apply initial RC/Terra transforms from exp_test.ipynb, or skip.",
    )
    parser.add_argument("--no-center", action="store_true", help="Do not subtract per-cloud mean before alignment.")
    parser.add_argument(
        "--fine-align",
        choices=["gicp", "none", "manual"],
        default="gicp",
        help="Fine alignment method after initial transform/centering.",
    )
    parser.add_argument(
        "--volume-mode",
        choices=["roi", "overlap"],
        default="roi",
        help="How to compute |V_net|: ROI-by-change-segmentation (roi) or DEM overlap (overlap).",
    )
    parser.add_argument(
        "--roi-buffer",
        type=float,
        default=0.0,
        help="Buffer (m) applied to ROI hull before volume rasterization. If 0, auto-buffers only when ROI is too sparse.",
    )
    parser.add_argument(
        "--fine-transform",
        type=Path,
        default=None,
        help="Optional 4x4 matrix (.npy or JSON) to use when --fine-align manual.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print debug info to stderr/stdout.")
    return parser.parse_args()


def _load_matrix(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        m = np.load(path)
        return np.asarray(m, dtype=np.float64)
    data = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(data, dtype=np.float64)


def main() -> None:
    args = _parse_args()
    manual_fine = _load_matrix(args.fine_transform) if args.fine_transform else None
    metrics = compute_metrics(
        block=args.block,
        terra_ply=args.terra,
        rc_ply=args.rc,
        stable_shp=args.stable_shp,
        within_threshold_m=float(args.within),
        ref_ratio=float(args.ref_ratio),
        max_distance_m=float(args.max_distance),
        spacing_sample=int(args.spacing_sample),
        voxel_sample=int(args.voxel_sample),
        seed=int(args.seed),
        initial_transform=args.initial_transform,
        do_center=not bool(args.no_center),
        fine_align=args.fine_align,
        volume_mode=args.volume_mode,
        roi_buffer_m=float(args.roi_buffer),
        manual_fine_transform=manual_fine,
        verbose=bool(args.verbose),
    )

    payload = asdict(metrics)
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
