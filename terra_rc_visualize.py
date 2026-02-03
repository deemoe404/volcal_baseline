#!/usr/bin/env python3
import argparse
import os
import tempfile
from pathlib import Path

_mpl_tmp = Path(tempfile.gettempdir()) / "matplotlib"
_mpl_tmp.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_tmp))

import numpy as np

import geopandas as gpd
import matplotlib.pyplot as plt
import py4dgeo
from shapely.ops import unary_union

import terra_rc_metrics as m


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize Terra vs on-controller reconstruction alignment and metrics."
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
        "--stable-shp",
        type=Path,
        default=Path("/Users/sam/Desktop/stable.shp"),
        help="Stable region shapefile (same one used in metrics).",
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
        help="Fine alignment method to match metrics.",
    )
    p.add_argument(
        "--volume-mode",
        choices=["roi", "overlap"],
        default="roi",
        help="Volume mode (roi may fall back to overlap if degenerate).",
    )
    p.add_argument(
        "--roi-buffer",
        type=float,
        default=0.0,
        help="Buffer (m) applied to ROI hull before volume rasterization (0=auto only when needed).",
    )
    p.add_argument("--spacing-sample", type=int, default=0, help="Sample size for spacing estimate (0=all).")
    p.add_argument("--voxel-sample", type=int, default=0, help="Sample size for voxel fit (0=all).")
    p.add_argument("--seed", type=int, default=0, help="RNG seed.")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/terra_rc/vis_result_2026Feb03"),
        help="Directory to write PNGs.",
    )
    p.add_argument("--max-plot", type=int, default=50_000, help="Max points to plot per cloud.")
    p.add_argument("--dpi", type=int, default=200, help="PNG DPI.")
    return p.parse_args()


def _sample(points: np.ndarray, n: int, seed: int) -> np.ndarray:
    if points.shape[0] <= n:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=n, replace=False)
    return points[idx]


def _plot_geom_outline(ax, geom, *, color: str, lw: float, label: str | None = None):
    # Handles Polygon / MultiPolygon / GeometryCollection
    if geom is None:
        return
    gtype = getattr(geom, "geom_type", None)
    if gtype == "Polygon":
        x, y = geom.exterior.xy
        ax.plot(x, y, color=color, linewidth=lw, label=label)
        for interior in geom.interiors:
            x, y = interior.xy
            ax.plot(x, y, color=color, linewidth=lw)
    elif gtype == "MultiPolygon":
        first = True
        for g in geom.geoms:
            _plot_geom_outline(
                ax, g, color=color, lw=lw, label=label if first else None
            )
            first = False
    elif gtype == "GeometryCollection":
        first = True
        for g in geom.geoms:
            _plot_geom_outline(
                ax, g, color=color, lw=lw, label=label if first else None
            )
            first = False


def _savefig(path: Path, dpi: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def _compute_overlap_diff_grid(
    terra: np.ndarray, rc: np.ndarray, *, grid_res: float, threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_min = max(float(terra[:, 0].min()), float(rc[:, 0].min()))
    x_max = min(float(terra[:, 0].max()), float(rc[:, 0].max()))
    y_min = max(float(terra[:, 1].min()), float(rc[:, 1].min()))
    y_max = min(float(terra[:, 1].max()), float(rc[:, 1].max()))
    x_edges = np.arange(x_min, x_max + grid_res, grid_res, dtype=np.float64)
    y_edges = np.arange(y_min, y_max + grid_res, grid_res, dtype=np.float64)
    min_z = min(float(terra[:, 2].min()), float(rc[:, 2].min()))

    dem_terra = m._raster_mean_dem(terra, x_edges=x_edges, y_edges=y_edges, min_z=min_z)
    dem_rc = m._raster_mean_dem(rc, x_edges=x_edges, y_edges=y_edges, min_z=min_z)
    overlap = np.isfinite(dem_terra) & np.isfinite(dem_rc)
    diff = (dem_rc - dem_terra).astype(np.float32)
    diff[~overlap] = np.nan
    diff[np.abs(diff) < float(threshold)] = 0.0

    x_centers = (x_edges[:-1] + x_edges[1:]) * 0.5
    y_centers = (y_edges[:-1] + y_edges[1:]) * 0.5
    xx, yy = np.meshgrid(x_centers, y_centers)
    return xx.astype(np.float32), yy.astype(np.float32), diff


def main() -> None:
    args = _parse_args()
    i = int(args.panorama)
    terra_ply = args.terra_root / f"panorama_{i}cloud.ply"
    rc_ply = Path(args.rc_pattern.format(i=i))
    out_dir = args.out_dir / f"p{i}"

    terra = m._read_ply_xyz(terra_ply)
    rc = m._read_ply_xyz(rc_ply)

    # Initial transforms (optional)
    if args.initial_transform == "default":
        terra = m._apply_affine(terra, m.DEFAULT_TERRA_TRANSFORM)
        rc = m._apply_affine(rc, m.DEFAULT_RC_TRANSFORM)

    # Center per cloud (same as metrics)
    terra = m._center(terra)
    rc = m._center(rc)

    # Pre-alignment overview plot
    terra_vis = _sample(terra, args.max_plot, args.seed)
    rc_vis = _sample(rc, args.max_plot, args.seed + 1)
    stable_gdf = gpd.read_file(args.stable_shp)
    stable_union = unary_union(list(stable_gdf.geometry))

    plt.figure(figsize=(8, 6))
    plt.scatter(terra_vis[:, 0], terra_vis[:, 1], s=1, c="0.6", label="Terra (pre)")
    plt.scatter(rc_vis[:, 0], rc_vis[:, 1], s=2, c="tab:red", alpha=0.7, label="RC (pre)")
    _plot_geom_outline(
        plt.gca(), stable_union, color="tab:blue", lw=1.5, label="stable.shp"
    )
    plt.axis("equal")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Panorama {i} - XY overview (pre fine-align)")
    plt.legend(markerscale=6)
    _savefig(out_dir / "xy_pre_align.png", dpi=args.dpi)

    # Fine alignment
    if args.fine_align == "gicp":
        avg_spacing_pre = m._estimate_avg_spacing(
            terra, sample_size=int(args.spacing_sample), seed=int(args.seed)
        )
        init_voxel_pre = avg_spacing_pre / 0.01
        if int(args.voxel_sample) > 0 and terra.shape[0] > int(args.voxel_sample):
            rng = np.random.default_rng(int(args.seed))
            idx = rng.choice(terra.shape[0], size=int(args.voxel_sample), replace=False)
            voxel_fit_pre = terra[idx]
        else:
            voxel_fit_pre = terra
        adapted_voxel_pre = m._adaptive_voxel_size(voxel_fit_pre, 0.01, init_voxel_pre)
        T_fine = m._fine_align_gicp(
            source=terra,
            target=rc,
            voxel_size=max(float(adapted_voxel_pre), 0.05),
            distance_quantile=0.9,
            max_iter=128,
        )
        terra = m._apply_affine(terra, T_fine)
    elif args.fine_align == "manual":
        terra = m._apply_affine(terra, m.DEFAULT_TERRA_FINE_TRANSFORM)
    elif args.fine_align == "none":
        pass

    # Post-alignment overview plot
    terra_vis = _sample(terra, args.max_plot, args.seed)
    rc_vis = _sample(rc, args.max_plot, args.seed + 1)
    plt.figure(figsize=(8, 6))
    plt.scatter(terra_vis[:, 0], terra_vis[:, 1], s=1, c="0.6", label="Terra (post)")
    plt.scatter(rc_vis[:, 0], rc_vis[:, 1], s=2, c="tab:red", alpha=0.7, label="RC (post)")
    _plot_geom_outline(
        plt.gca(), stable_union, color="tab:blue", lw=1.5, label="stable.shp"
    )
    plt.axis("equal")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Panorama {i} - XY overview (post fine-align: {args.fine_align})")
    plt.legend(markerscale=6)
    _savefig(out_dir / "xy_post_align.png", dpi=args.dpi)

    # Stable region detail plot (points inside stable polygon)
    stable_terra = m._stable_points(terra, stable_union)
    stable_rc = m._stable_points(rc, stable_union)
    plt.figure(figsize=(7, 6))
    plt.scatter(
        stable_terra[:, 0],
        stable_terra[:, 1],
        s=2,
        c="0.4",
        label=f"Terra stable (n={len(stable_terra)})",
    )
    plt.scatter(
        stable_rc[:, 0],
        stable_rc[:, 1],
        s=6,
        c="tab:red",
        alpha=0.8,
        label=f"RC stable (n={len(stable_rc)})",
    )
    _plot_geom_outline(plt.gca(), stable_union, color="tab:blue", lw=1.5)
    plt.axis("equal")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Panorama {i} - stable.shp detail")
    plt.legend(markerscale=4)
    _savefig(out_dir / "stable_xy.png", dpi=args.dpi)

    # M3C2 + histogram
    avg_spacing = m._estimate_avg_spacing(
        terra, sample_size=int(args.spacing_sample), seed=int(args.seed)
    )
    init_voxel_size = avg_spacing / 0.01
    if int(args.voxel_sample) > 0 and terra.shape[0] > int(args.voxel_sample):
        rng = np.random.default_rng(int(args.seed))
        idx = rng.choice(terra.shape[0], size=int(args.voxel_sample), replace=False)
        voxel_fit = terra[idx]
    else:
        voxel_fit = terra
    adapted_voxel_size = m._adaptive_voxel_size(voxel_fit, 0.01, init_voxel_size)

    if len(stable_terra) >= 10 and len(stable_rc) >= 10:
        stable_corepoints = m._voxel_downsample(
            stable_terra, voxel_size=float(adapted_voxel_size)
        )
        m3c2_stable = py4dgeo.M3C2(
            epochs=(py4dgeo.Epoch(stable_terra), py4dgeo.Epoch(stable_rc)),
            corepoints=stable_corepoints,
            normal_radii=[adapted_voxel_size * 2.0],
            cyl_radius=float(adapted_voxel_size),
            max_distance=15.0,
            registration_error=0.0,
        )
        d_stable, _ = m3c2_stable.run()
        reg_error = float(np.nanstd(d_stable))
    else:
        reg_error = 0.0

    epoch_terra = py4dgeo.Epoch(terra)
    epoch_rc = py4dgeo.Epoch(rc)
    corepoints = m._voxel_downsample(epoch_terra.cloud[::], voxel_size=adapted_voxel_size)
    m3c2 = py4dgeo.M3C2(
        epochs=(epoch_terra, epoch_rc),
        corepoints=corepoints,
        normal_radii=[adapted_voxel_size * 2.0],
        cyl_radius=float(adapted_voxel_size),
        max_distance=15.0,
        registration_error=float(reg_error),
    )
    d, unc = m3c2.run()
    d = np.asarray(d)
    abs_d = np.abs(d[np.isfinite(d)])

    plt.figure(figsize=(7, 4))
    plt.hist(abs_d, bins=200, color="0.3")
    plt.axvline(np.percentile(abs_d, 50), color="tab:blue", lw=1.5, label="median |d|")
    plt.axvline(np.percentile(abs_d, 95), color="tab:orange", lw=1.5, label="p95 |d|")
    plt.xlabel("|d| (m)")
    plt.ylabel("count")
    plt.title(f"Panorama {i} - |d| histogram (n={abs_d.size})")
    plt.legend()
    _savefig(out_dir / "abs_d_hist.png", dpi=args.dpi)

    if isinstance(unc, np.ndarray) and unc.dtype.fields and "lodetection" in unc.dtype.fields:
        lod = np.asarray(unc["lodetection"])
        valid = np.isfinite(d) & np.isfinite(lod)
        cs = np.zeros_like(d, dtype=bool)
        cs[valid] = np.abs(d[valid]) > lod[valid]
    else:
        lod = None
        cs = np.zeros_like(d, dtype=bool)

    # Corepoints distance map (XY)
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        corepoints[:, 0],
        corepoints[:, 1],
        c=d,
        s=3,
        cmap="RdBu",
        vmin=-np.nanpercentile(np.abs(d), 95),
        vmax=np.nanpercentile(np.abs(d), 95),
    )
    plt.colorbar(sc, label="signed d (m)")
    plt.axis("equal")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(
        f"Panorama {i} - M3C2 distances (reg_err={reg_error:.3f}m, change%={float(cs.mean()*100):.1f}%)"
    )
    _savefig(out_dir / "m3c2_corepoints_xy.png", dpi=args.dpi)

    # ROI hull diagnostic (might be empty on RC side)
    hulls = m._segment_changes(corepoints, cs)
    if hulls:
        areas = np.array([h.area for h in hulls], dtype=float)
        roi = hulls[int(np.argmax(areas))]

        step = float(adapted_voxel_size) if float(adapted_voxel_size) > 0 else 0.5
        requested_buf = float(args.roi_buffer) if float(args.roi_buffer) > 0 else 0.0
        candidates = [requested_buf] if requested_buf > 0 else [0.0, step, 2 * step, 3 * step, 4 * step]

        roi_used = roi
        terra_roi_n = 0
        rc_roi_n = 0
        used_buf = 0.0
        for b in candidates:
            roi_b = roi if b <= 0 else roi.buffer(float(b))
            inside_rc = m._is_inside_selected_hulls_vectorized([roi_b], rc[:, :2])
            inside_terra = m._is_inside_selected_hulls_vectorized([roi_b], terra[:, :2])
            terra_roi_n = int(np.count_nonzero(inside_terra))
            rc_roi_n = int(np.count_nonzero(inside_rc))
            roi_used = roi_b
            used_buf = float(b)
            if terra_roi_n >= 10 and rc_roi_n >= 10:
                break

        plt.figure(figsize=(8, 6))
        cp_sig = corepoints[cs]
        plt.scatter(corepoints[:, 0], corepoints[:, 1], s=1, c="0.8", label="corepoints")
        if cp_sig.shape[0] > 0:
            plt.scatter(cp_sig[:, 0], cp_sig[:, 1], s=4, c="tab:purple", label="|d|>LoD")
        inside_rc_vis = m._is_inside_selected_hulls_vectorized([roi_used], rc[:, :2])
        rc_roi_vis = _sample(rc[inside_rc_vis], min(args.max_plot, 20_000), args.seed + 7)
        if rc_roi_vis.shape[0] > 0:
            plt.scatter(
                rc_roi_vis[:, 0],
                rc_roi_vis[:, 1],
                s=6,
                c="tab:red",
                alpha=0.7,
                label=f"RC in ROI (n={rc_roi_n})",
            )
        label = "ROI hull" if used_buf <= 0 else f"ROI hull (buffer={used_buf:.2f}m)"
        _plot_geom_outline(plt.gca(), roi_used, color="tab:green", lw=2.0, label=label)
        plt.axis("equal")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"Panorama {i} - ROI hull diagnostic (Terra n={terra_roi_n}, RC n={rc_roi_n})")
        plt.legend(markerscale=4)
        _savefig(out_dir / "roi_xy.png", dpi=args.dpi)

    # Overlap DEM diff grid (the fallback volume path)
    try:
        xx, yy, diff = _compute_overlap_diff_grid(
            terra, rc, grid_res=float(adapted_voxel_size), threshold=float(reg_error)
        )
        vmax = float(np.nanpercentile(np.abs(diff), 99)) if np.isfinite(diff).any() else 1.0
        plt.figure(figsize=(8, 6))
        plt.pcolormesh(xx, yy, diff, cmap="RdBu", vmin=-vmax, vmax=vmax, shading="auto")
        plt.colorbar(label="DEM diff (RC - Terra) [m]")
        plt.axis("equal")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"Panorama {i} - overlap DEM diff (grid={adapted_voxel_size:.3f}m)")
        _savefig(out_dir / "dem_diff_overlap.png", dpi=args.dpi)
    except Exception as e:
        (out_dir / "dem_diff_overlap.error.txt").write_text(str(e) + "\n", encoding="utf-8")

    print(f"Wrote visualizations to {out_dir}")


if __name__ == "__main__":
    main()
