"""Compare two complete point tables after pandas round(10), including per-point bins."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_points(path):
    names = ["point_id", "broad_class", "nominal_10m", "nominal_20m", "nominal_30m"]
    frame = pd.read_csv(path, usecols=names, dtype={"point_id": "string", "broad_class": "string"})
    if len(frame) != 136605 or frame.isna().any().any() or frame.point_id.duplicated().any():
        raise ValueError(f"{path}: expected 136605 unique complete records")
    values = frame[names[2:]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError(f"{path}: invalid coverage fraction")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--expected-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = (Path(__file__).resolve().parents[2] / "runs").resolve()
    output = args.output.resolve()
    if not output.is_relative_to(runs):
        parser.error("output must be inside replication_draft/runs")
    ref, cand = read_points(args.reference), read_points(args.candidate)
    paired = ref.merge(cand, on="point_id", how="outer", suffixes=("_reference", "_candidate"), indicator=True, validate="one_to_one")
    if not (paired._merge == "both").all() or not (paired.broad_class_reference == paired.broad_class_candidate).all():
        raise ValueError("Point IDs or broad classes do not match")
    expected = json.loads(args.expected_json.read_text(encoding="utf-8"))
    report = {"rule": "pandas Series.round(10), then <0.25, <0.50, <0.75, >=0.95; ties follow pandas/NumPy rounding", "reference_sha256": sha256(args.reference), "candidate_sha256": sha256(args.candidate), "point_count": len(paired), "point_ids_and_classes_match": True, "pandas": pd.__version__, "numpy": np.__version__, "scales": {}, "pass": True}
    for size in (10, 20, 30):
        field = f"nominal_{size}m"
        raw_a, raw_b = paired[field + "_reference"], paired[field + "_candidate"]
        a, b = raw_a.round(10), raw_b.round(10)
        conditions = {"below_025": (a < .25, b < .25), "below_050": (a < .5, b < .5), "below_075": (a < .75, b < .75), "at_least_095": (a >= .95, b >= .95)}
        scale = {"max_absolute_unrounded_difference": float((raw_a - raw_b).abs().max()), "rounded_value_mismatches": int((a != b).sum()), "thresholds": {}, "by_class": {}}
        for key, (ca, cb) in conditions.items():
            ca_n, cb_n = int(ca.sum()), int(cb.sum())
            mismatch = int((ca != cb).sum())
            expected_n = expected["overall"][str(size)][key]
            scale["thresholds"][key] = {"reference": ca_n, "candidate": cb_n, "expected": expected_n, "point_membership_mismatches": mismatch}
            if ca_n != cb_n or ca_n != expected_n or mismatch:
                report["pass"] = False
        for label in sorted(paired.broad_class_reference.unique()):
            mask = paired.broad_class_reference == label
            counts = {}
            for key, (ca, cb) in conditions.items():
                na, nb = int((ca & mask).sum()), int((cb & mask).sum())
                counts[key] = {"reference": na, "candidate": nb}
                if na != nb:
                    report["pass"] = False
                if size == 10 and key in ("below_025", "at_least_095") and na != expected["class_10m"][label][key]:
                    report["pass"] = False
            scale["by_class"][label] = counts
        report["scales"][str(size)] = scale
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
