"""Stress test the LUCAS to WorldCover bridge for the 2021 COG experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data_processed" / "lucas_cog_exact_pilot.csv"
DEFAULT_OUTPUT = ROOT / "data_processed" / "lucas_exact_analysis" / "bridge_sensitivity"
PRIMARY_YEAR = "2021"


def portable_path(path: Path) -> str:
    """Return a project relative path for portable analysis metadata."""

    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def baseline_candidates(code: str) -> set[int]:
    broad = code[:1]
    if broad == "A":
        return {50}
    if broad == "B":
        return {40}
    if broad == "C":
        return {10}
    if broad == "D":
        return {20}
    if broad == "E":
        return {30}
    if broad == "F":
        return {100 if code.startswith("F30") else 60}
    if broad == "G":
        return {70 if code.startswith("G50") else 80}
    if broad == "H":
        return {90}
    return set()


def candidate_sets(code: str, scenario: str) -> set[int]:
    baseline = baseline_candidates(code)
    broad = code[:1]
    if scenario == "baseline":
        return baseline
    if scenario == "one_step":
        alternatives = {
            "A": {50},
            "B": {40, 30},
            "C": {10},
            "D": {20, 30},
            "E": {30, 20},
            "F": {60, 100},
            "G": {80, 70},
            "H": {90},
        }
        return alternatives.get(broad, baseline)
    if scenario == "permissive":
        alternatives = {
            "A": {50, 40},
            "B": {40, 30},
            "C": {10, 30},
            "D": {20, 30},
            "E": {30, 20},
            "F": {60, 100},
            "G": {80, 90, 70},
            "H": {90, 80},
        }
        return alternatives.get(broad, baseline)
    raise ValueError(f"Unknown bridge scenario: {scenario}")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if row.get("proxy_pixel_fraction") not in (None, "")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = read_rows(args.input)
    if not rows:
        raise RuntimeError("No exact COG pilot rows found")
    support = np.array([float(row["proxy_pixel_fraction"]) for row in rows], dtype=float)
    prediction = np.array([int(row[f"raw_{PRIMARY_YEAR}"]) for row in rows], dtype=int)
    scenarios = ("baseline", "one_step", "permissive")
    results: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    baseline_match: np.ndarray | None = None
    for scenario in scenarios:
        candidates = [candidate_sets(row["cprn_code"], scenario) for row in rows]
        match = np.array([int(value in allowed) for value, allowed in zip(prediction, candidates)], dtype=float)
        if scenario == "baseline":
            baseline_match = match
        proxy_supported = support * match
        residual_compatible = proxy_supported + (1.0 - support)
        results.append(
            {
                "scenario": scenario,
                "year": PRIMARY_YEAR,
                "n": len(rows),
                "matched_n": int(match.sum()),
                "changed_from_baseline_n": int(np.sum(match != baseline_match)) if baseline_match is not None else 0,
                "hard_match_score": float(match.mean()),
                "proxy_supported_match_score": float(proxy_supported.mean()),
                "residual_compatible_score": float(residual_compatible.mean()),
                "mean_scenario_span": float(np.mean(residual_compatible - proxy_supported)),
            }
        )
        for row, allowed in zip(rows, candidates):
            detail_rows.append(
                {
                    "point_id": row["point_id"],
                    "cprn_code": row["cprn_code"],
                    "broad_class": row["broad_class"],
                    "scenario": scenario,
                    "candidate_worldcover_classes": ";".join(str(value) for value in sorted(allowed)),
                }
            )

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "bridge_sensitivity.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    with (args.output / "bridge_candidate_sets_2021.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)
    summary = {
        "input": portable_path(args.input),
        "year": PRIMARY_YEAR,
        "scenarios": {
            "baseline": "The deterministic published code bridge used for the primary score convention.",
            "one_step": "A stress scenario allowing one adjacent or special WorldCover class for selected broad groups.",
            "permissive": "A wider stress scenario allowing additional adjacent or special classes.",
        },
        "interpretation": (
            "Candidate sets are bridge stress scenarios, not expert validated semantic probabilities. "
            "They test whether the finite sample score conventions depend on the legend crosswalk."
        ),
        "results": results,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
