#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from terra_rc_metrics import compute_metrics
from terra_rc_metrics import _load_matrix


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-run Terra vs on-controller metrics for panorama 1–4.")
    parser.add_argument(
        "--terra-root",
        type=Path,
        default=Path("/Users/sam/Developer/Datasets/Tongjiang Arial"),
        help="Folder containing panorama_{i}cloud.ply",
    )
    parser.add_argument(
        "--rc-map",
        type=Path,
        default=None,
        help="JSON mapping panorama index -> on-controller PLY path. Example: {\"1\": \"/path/p1.ply\", ...}",
    )
    parser.add_argument(
        "--rc-pattern",
        type=str,
        default=None,
        help="Pattern for on-controller PLYs, e.g. '/path/panorama_{i}_rc.ply'",
    )
    parser.add_argument(
        "--stable-shp",
        type=Path,
        default=Path("/Users/sam/Desktop/stable.shp"),
        help="Stable region shapefile.",
    )
    parser.add_argument(
        "--fine-align",
        choices=["gicp", "none", "manual"],
        default="manual",
        help="Fine alignment method (matches exp_test.ipynb by default).",
    )
    parser.add_argument(
        "--fine-transform",
        type=Path,
        default=None,
        help="Optional 4x4 matrix (.npy or JSON) for --fine-align manual.",
    )
    parser.add_argument(
        "--volume-mode",
        choices=["roi", "overlap"],
        default="roi",
        help="How to compute |V_net|.",
    )
    parser.add_argument(
        "--roi-buffer",
        type=float,
        default=0.0,
        help="Buffer (m) applied to ROI hull before volume rasterization (0=auto only when needed).",
    )
    parser.add_argument("--spacing-sample", type=int, default=0, help="Sample size for spacing estimate (0=all).")
    parser.add_argument("--voxel-sample", type=int, default=0, help="Sample size for voxel fit (0=all).")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for sampling.")
    parser.add_argument("--out", type=Path, default=Path("outputs/terra_rc/results.json"), help="Output JSON path.")
    parser.add_argument("--start", type=int, default=1, help="Start panorama index (inclusive).")
    parser.add_argument("--end", type=int, default=4, help="End panorama index (inclusive).")
    return parser.parse_args()


def _load_rc_map(path: Path | None) -> dict[int, str]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, str] = {}
    for k, v in data.items():
        out[int(k)] = str(v)
    return out


def main() -> None:
    args = _parse_args()
    rc_map = _load_rc_map(args.rc_map)
    manual_fine = _load_matrix(args.fine_transform) if args.fine_transform else None

    results: list[dict] = []
    for i in range(int(args.start), int(args.end) + 1):
        terra = args.terra_root / f"panorama_{i}cloud.ply"
        if i in rc_map:
            rc = Path(rc_map[i])
        elif args.rc_pattern is not None:
            rc = Path(args.rc_pattern.format(i=i))
        else:
            raise SystemExit(
                "Need --rc-map or --rc-pattern to locate on-controller PLYs for each panorama."
            )

        metrics = compute_metrics(
            block=f"Panorama~{i}",
            terra_ply=terra,
            rc_ply=rc,
            stable_shp=args.stable_shp,
            fine_align=args.fine_align,
            manual_fine_transform=manual_fine,
            volume_mode=args.volume_mode,
            roi_buffer_m=float(args.roi_buffer),
            spacing_sample=int(args.spacing_sample),
            voxel_sample=int(args.voxel_sample),
            seed=int(args.seed),
        )
        results.append(metrics.__dict__)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(results)} rows to {args.out}")


if __name__ == "__main__":
    main()
