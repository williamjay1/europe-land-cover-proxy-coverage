"""Path-adapted copy: compare local and EPSG:3035 threshold classifications."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    runs = (Path(__file__).resolve().parents[2] / "runs").resolve()
    if not destination.is_relative_to(runs):
        parser.error("output must be inside replication_draft/runs")
    if destination.exists():
        raise FileExistsError(destination)
    data = pd.read_csv(args.points)
    assert len(data) == 136605
    x, y = Transformer.from_crs(4326, 3035, always_xy=True).transform(data.lon.to_numpy(), data.lat.to_numpy())
    n, e, s, w = (data[k].to_numpy(dtype=float) for k in ("n_m", "e_m", "s_m", "w_m"))
    vertices = np.stack((np.column_stack((x-w,y)), np.column_stack((x,y+n)), np.column_stack((x+e,y)), np.column_stack((x,y-s))), axis=1)
    polygons = shapely.polygons(vertices)
    invalid = ~shapely.is_valid(polygons)
    polygons[invalid] = shapely.buffer(polygons[invalid], 0)
    assert shapely.is_valid(polygons).all()
    result = {"n": len(data), "invalid_fixed_n": int(invalid.sum()), "rule": "pandas Series.round(10) before comparisons", "scales": {}}
    for size in (10,20,30):
        h = size / 2
        boxes = shapely.box(x-h,y-h,x+h,y+h)
        projected = np.clip(shapely.area(shapely.intersection(polygons, boxes)) / size**2, 0, 1)
        local = data[f"nominal_{size}m"].to_numpy()
        a = pd.Series(projected).round(10).to_numpy()
        b = pd.Series(local).round(10).to_numpy()
        tests = {}
        for t in (.25,.5,.75,.95):
            pa, pb = (a >= t, b >= t) if t == .95 else (a < t, b < t)
            tests[str(t)] = {"projected_count": int(pa.sum()), "local_count": int(pb.sum()), "membership_mismatches": int((pa != pb).sum())}
        result["scales"][str(size)] = {"max_abs_difference": float(np.max(np.abs(projected-local))), "thresholds": tests}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as f:
        f.write(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
