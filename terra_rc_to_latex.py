#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render Terra/RC metrics JSON as LaTeX table rows.")
    p.add_argument("--in", dest="inp", required=True, type=Path, help="Input JSON from terra_rc_batch.py")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    rows = json.loads(args.inp.read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = [rows]

    for r in rows:
        block = r.get("block", "")
        vnet = r.get("v_net_abs_m3")
        med = r.get("median_abs_d_m")
        p95 = r.get("p95_abs_d_m")
        rmse = r.get("rmse95_m")
        pct = r.get("pct_within_010m")

        def fnum(x, nd=4):
            return "\\dots" if x is None else f"{float(x):.{nd}f}"

        def fpct(x):
            return "\\dots" if x is None else f"{float(x):.2f}"

        print(
            f"{block} & {fnum(vnet)} & {fnum(med)} & {fnum(p95)} & {fnum(rmse)} & {fpct(pct)} \\\\"
        )


if __name__ == "__main__":
    main()
