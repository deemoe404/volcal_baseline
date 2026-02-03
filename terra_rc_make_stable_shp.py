#!/usr/bin/env python3
import argparse
import os
import tempfile
from pathlib import Path

_mpl_tmp = Path(tempfile.gettempdir()) / "matplotlib"
_mpl_tmp.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_tmp))

import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import box

import terra_rc_metrics as m


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Auto-generate a stable-area polygon shapefile for a given panorama."
    )
    p.add_argument("--panorama", type=int, required=True, help="Panorama index (1..4).")
    p.add_argument(
        "--terra-root",
        type=Path,
        default=Path("/Users/sam/Developer/Datasets/Tongjiang Arial"),
        help="Folder containing panorama_{i}cloud.ply",
    )
    p.add_argument(
        "--rc-pattern",
        type=str,
        default="/Users/sam/Developer/Android Studio/rebuild_edge/mobile_out/result_2026Feb03/p{i}/dense_depth_out/tsdf_surface.ply",
        help="Pattern for on-controller PLY path, python format with {i}.",
    )
    p.add_argument(
        "--out-shp",
        type=Path,
        default=Path("outputs/terra_rc/stable_pano2_auto.shp"),
        help="Output shapefile path (.shp).",
    )
    p.add_argument(
        "--initial-transform",
        choices=["default", "none"],
        default="default",
        help="Apply initial transforms from exp_test.ipynb or skip.",
    )
    p.add_argument(
        "--fine-align",
        choices=["gicp", "none", "manual"],
        default="gicp",
        help="Fine alignment method (should match metrics run).",
    )
    p.add_argument(
        "--spacing-sample",
        type=int,
        default=0,
        help="Sample size for spacing estimate (0=all).",
    )
    p.add_argument(
        "--voxel-sample",
        type=int,
        default=0,
        help="Sample size for voxel fit (0=all).",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed.")
    p.add_argument(
        "--terra-sample",
        type=int,
        default=400_000,
        help="Number of Terra points to sample for KDTree (0=all).",
    )
    p.add_argument(
        "--terra-count-sample",
        type=int,
        default=200_000,
        help="Number of Terra points to sample for stable-count estimation (0=all).",
    )
    p.add_argument(
        "--keep-frac",
        type=float,
        default=0.3,
        help="Keep this fraction of RC points with smallest 3D NN distance to Terra.",
    )
    p.add_argument(
        "--box-quantiles",
        type=str,
        default="0.10,0.15,0.20,0.25,0.30,0.35,0.40",
        help="Comma-separated list of q values; box uses [q,1-q] quantiles on (x,y). Higher q => smaller box.",
    )
    p.add_argument(
        "--buffer",
        type=float,
        default=0.5,
        help="Buffer (in same XY units) applied to the convex hull.",
    )
    p.add_argument(
        "--min-points",
        type=int,
        default=500,
        help="Minimum number of RC points required inside stable polygon; widens threshold if needed.",
    )
    p.add_argument(
        "--max-terra-points",
        type=int,
        default=200_000,
        help="Try to keep Terra stable points below this number (estimated by sampling).",
    )
    return p.parse_args()


def _sample(points: np.ndarray, n: int, seed: int) -> np.ndarray:
    if n <= 0 or points.shape[0] <= n:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=n, replace=False)
    return points[idx]


def main() -> None:
    args = _parse_args()
    i = int(args.panorama)
    terra_ply = args.terra_root / f"panorama_{i}cloud.ply"
    rc_ply = Path(args.rc_pattern.format(i=i))

    terra = m._read_ply_xyz(terra_ply)
    rc = m._read_ply_xyz(rc_ply)

    if args.initial_transform == "default":
        terra = m._apply_affine(terra, m.DEFAULT_TERRA_TRANSFORM)
        rc = m._apply_affine(rc, m.DEFAULT_RC_TRANSFORM)

    terra = m._center(terra)
    rc = m._center(rc)

    if args.fine_align == "gicp":
        avg_spacing_pre = m._estimate_avg_spacing(
            terra, sample_size=int(args.spacing_sample), seed=int(args.seed)
        )
        init_voxel_pre = avg_spacing_pre / 0.01
        voxel_fit_pre = _sample(terra, int(args.voxel_sample), int(args.seed))
        adapted_voxel_pre = m._adaptive_voxel_size(voxel_fit_pre, 0.01, init_voxel_pre)
        T = m._fine_align_gicp(
            source=terra,
            target=rc,
            voxel_size=max(float(adapted_voxel_pre), 0.05),
            distance_quantile=0.9,
            max_iter=128,
        )
        terra = m._apply_affine(terra, T)
    elif args.fine_align == "manual":
        terra = m._apply_affine(terra, m.DEFAULT_TERRA_FINE_TRANSFORM)
    elif args.fine_align == "none":
        pass
    else:
        raise SystemExit(f"Unknown fine_align: {args.fine_align}")

    terra_kd_pts = _sample(terra, int(args.terra_sample), int(args.seed))
    tree = cKDTree(terra_kd_pts)
    dists, _ = tree.query(rc, k=1)
    dists = np.asarray(dists, dtype=np.float64)

    keep_frac = float(args.keep_frac)
    keep_frac = min(max(keep_frac, 0.01), 1.0)

    rc_good = None
    for frac in [keep_frac, min(1.0, keep_frac * 1.5), min(1.0, keep_frac * 2.0), 1.0]:
        thr = float(np.quantile(dists, frac))
        mask = np.isfinite(dists) & (dists <= thr)
        if int(mask.sum()) >= max(int(args.min_points), 10):
            rc_good = rc[mask]
            break
    if rc_good is None:
        rc_good = rc[np.isfinite(dists)]

    # Pick a stable box around the best-matching RC points, while bounding Terra stable point count.
    terra_count_pts = _sample(terra, int(args.terra_count_sample), int(args.seed) + 123)

    q_values = [float(x) for x in str(args.box_quantiles).split(",") if x.strip()]
    q_values = [q for q in q_values if 0.0 <= q < 0.5]
    if not q_values:
        raise SystemExit("Invalid --box-quantiles; expected values in [0, 0.5).")

    best_poly = None
    best_stats = None
    for q in q_values:
        xs = rc_good[:, 0]
        ys = rc_good[:, 1]
        x0, x1 = np.quantile(xs, [q, 1.0 - q])
        y0, y1 = np.quantile(ys, [q, 1.0 - q])
        poly = box(float(x0), float(y0), float(x1), float(y1))
        if float(args.buffer) != 0.0:
            poly = poly.buffer(float(args.buffer))

        inside_rc = m._points_in_polygon(rc[:, :2], poly)
        rc_in = int(inside_rc.sum())
        if rc_in < int(args.min_points):
            continue

        inside_terra_est = m._points_in_polygon(terra_count_pts[:, :2], poly)
        terra_in_est = int(inside_terra_est.sum())
        terra_in = int(round(terra_in_est / len(terra_count_pts) * len(terra)))

        best_poly = poly
        best_stats = (q, rc_in, terra_in)
        if terra_in <= int(args.max_terra_points):
            break

    if best_poly is None:
        raise SystemExit("Could not find a stable polygon satisfying --min-points.")

    out_shp = Path(args.out_shp)
    out_shp.parent.mkdir(parents=True, exist_ok=True)
    q, rc_in, terra_in = best_stats
    gdf = gpd.GeoDataFrame(
        {"name": [f"stable_pano{i}_auto_q{q:.2f}"], "q": [q], "rc_in": [rc_in], "terra_in_est": [terra_in]},
        geometry=[best_poly],
        crs="EPSG:4326",
    )
    gdf.to_file(out_shp)

    # quick stats
    inside_rc = m._points_in_polygon(rc[:, :2], best_poly)
    inside_terra = m._points_in_polygon(terra[:, :2], best_poly)
    print(
        f"Wrote {out_shp} (q={q:.2f}, terra_in={int(inside_terra.sum())}, rc_in={int(inside_rc.sum())}, rc_total={len(rc)})"
    )


if __name__ == "__main__":
    main()
