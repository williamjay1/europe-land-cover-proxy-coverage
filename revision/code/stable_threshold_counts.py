"""Portable, non-overwriting version of the archived stable threshold calculation."""
import argparse
import json
from pathlib import Path

import pandas as pd


def calculate(points):
    df = pd.read_csv(points, usecols=["broad_class", "nominal_10m", "nominal_20m", "nominal_30m"])
    if len(df) != 136605 or df.isna().any().any():
        raise ValueError("Expected 136605 complete records")
    output = {"rule": "Round fractions to 10 decimal places before applying fixed cutoffs; means and medians use unrounded values.", "overall": {}, "class_10m": {}}
    for size in (10, 20, 30):
        v = df[f"nominal_{size}m"].round(10)
        output["overall"][str(size)] = {"below_025": int((v < .25).sum()), "below_050": int((v < .5).sum()), "below_075": int((v < .75).sum()), "at_least_095": int((v >= .95).sum())}
    for label, group in df.groupby("broad_class"):
        v = group.nominal_10m.round(10)
        output["class_10m"][label] = {"below_025": int((v < .25).sum()), "at_least_095": int((v >= .95).sum())}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = Path(__file__).resolve().parents[2] / "runs"
    output = args.output.resolve()
    if not output.is_relative_to(runs.resolve()):
        parser.error("output must be inside replication_draft/runs")
    result = calculate(args.points)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
