"""Descriptive analysis for the exact WorldCover 2021 COG experiment.

The input is a class balanced measurement experiment, not a probability sample.
The script consequently reports deterministic finite sample score conventions
and sensitivity patterns.  It does not calculate confidence intervals or make
European population accuracy claims.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data_processed" / "lucas_cog_exact_pilot.csv"
DEFAULT_OUTPUT = ROOT / "data_processed" / "lucas_exact_analysis"
PRIMARY_YEAR = "2021"
LUCAS_CLASSES = tuple("ABCDEFGH")


def portable_path(path: Path) -> str:
    """Return a project relative path for portable analysis metadata."""

    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def as_float(value: str | None) -> float:
    return float(value) if value not in (None, "") else math.nan


def as_bool(value: str | int | bool | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def arrays(rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    prediction = np.array([int(row[f"raw_{PRIMARY_YEAR}"]) for row in rows], dtype=int)
    target = np.array([int(row["target_wc"]) for row in rows], dtype=int)
    support = np.array([as_float(row["proxy_pixel_fraction"]) for row in rows], dtype=float)
    hard_match = (prediction == target).astype(float)
    proxy_supported = support * hard_match
    residual_compatible = proxy_supported + (1.0 - support)
    return {
        "prediction": prediction,
        "target": target,
        "support": support,
        "hard_match": hard_match,
        "proxy_supported": proxy_supported,
        "residual_compatible": residual_compatible,
        "scenario_span": residual_compatible - proxy_supported,
    }


def score_bundle(values: dict[str, np.ndarray]) -> dict[str, float | int]:
    support = values["support"]
    return {
        "n": int(support.size),
        "mean_support": float(support.mean()),
        "median_support": float(np.median(support)),
        "support_p10": float(np.quantile(support, 0.10)),
        "support_p90": float(np.quantile(support, 0.90)),
        "hard_match_score": float(values["hard_match"].mean()),
        "proxy_supported_match_score": float(values["proxy_supported"].mean()),
        "residual_compatible_score": float(values["residual_compatible"].mean()),
        "mean_scenario_span": float(values["scenario_span"].mean()),
    }


def class_metrics(rows: list[dict[str, str]]) -> list[dict[str, float | int | str]]:
    result: list[dict[str, float | int | str]] = []
    for lucas_class in LUCAS_CLASSES:
        subset = [row for row in rows if row["broad_class"] == lucas_class]
        if subset:
            result.append({"class": lucas_class, **score_bundle(arrays(subset))})
    return result


def support_strata(rows: list[dict[str, str]]) -> list[dict[str, float | int | str]]:
    bins = [
        ("0.00 to 0.25", 0.0, 0.25),
        ("0.25 to 0.50", 0.25, 0.50),
        ("0.50 to 0.75", 0.50, 0.75),
        ("0.75 to 1.00", 0.75, 1.0000000001),
    ]
    result: list[dict[str, float | int | str]] = []
    for label, lower, upper in bins:
        subset = [
            row for row in rows if lower <= as_float(row["proxy_pixel_fraction"]) < upper
        ]
        if subset:
            result.append({"stratum": label, **score_bundle(arrays(subset))})
        else:
            result.append({"stratum": label, "n": 0})
    return result


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figures(summary: dict[str, object], output_dir: Path) -> list[str]:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    files: list[str] = []

    support = np.array(summary["support_values"], dtype=float)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.hist(support, bins=np.linspace(0, 1, 21), color="#4c78a8", edgecolor="white")
    ax.axvline(support.mean(), color="#e45756", linewidth=2, label=f"Mean {support.mean():.2f}")
    ax.set(xlim=(0, 1), xlabel="COG intersected proxy fraction", ylabel="Number of points", title="Proxy support")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = figure_dir / "fig2_cog_support.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    files.append(portable_path(path))

    overall = summary["overall"]
    labels = ["Hard match", "Proxy supported", "Residual compatible"]
    keys = ["hard_match_score", "proxy_supported_match_score", "residual_compatible_score"]
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    x = np.arange(len(keys))
    values = [float(overall[key]) for key in keys]
    bars = ax.bar(x, values, color=colors, width=0.64)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set(xticks=x, xticklabels=labels, ylim=(0, 1), ylabel="Score", title="Score conventions")
    ax.tick_params(axis="x", labelrotation=10)
    fig.tight_layout()
    path = figure_dir / "fig3_score_conventions.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    files.append(portable_path(path))

    entries = summary["class_metrics"]
    fig, ax = plt.subplots(figsize=(7.1, 3.8))
    x = np.arange(len(entries))
    proxy = np.array([float(entry["proxy_supported_match_score"]) for entry in entries])
    compatible = np.array([float(entry["residual_compatible_score"]) for entry in entries])
    hard = np.array([float(entry["hard_match_score"]) for entry in entries])
    ax.vlines(x, proxy, compatible, color="#f58518", linewidth=5, alpha=0.75, label="Scenario span")
    ax.scatter(x, hard, color="#4c78a8", s=28, zorder=3, label="Hard match")
    ax.set(xticks=x, xticklabels=[str(entry["class"]) for entry in entries], ylim=(0, 1), xlabel="LUCAS broad class", ylabel="Score", title="Class scores")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    path = figure_dir / "fig4_class_scores.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    files.append(portable_path(path))

    strata = [entry for entry in summary["support_strata"] if int(entry.get("n", 0))]
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    x = np.arange(len(strata))
    for key, color, label in [
        ("hard_match_score", "#4c78a8", "Hard match"),
        ("proxy_supported_match_score", "#f58518", "Proxy supported"),
        ("residual_compatible_score", "#54a24b", "Residual compatible"),
    ]:
        ax.plot(x, [float(entry[key]) for entry in strata], marker="o", color=color, label=label)
    ax.set(xticks=x, xticklabels=[str(entry["stratum"]) for entry in strata], ylim=(0, 1), ylabel="Score", xlabel="Proxy support stratum", title="Proxy strata")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = figure_dir / "fig5_support_strata.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    files.append(portable_path(path))

    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with args.input.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    empty_field = f"intersection_is_empty_{PRIMARY_YEAR}"
    required = {"proxy_pixel_fraction", f"raw_{PRIMARY_YEAR}", "target_wc", "broad_class", empty_field}
    if not rows or required - set(rows[0]):
        raise RuntimeError("The exact COG input does not contain the required 2021 fields")
    rows = [row for row in rows if row.get("proxy_pixel_fraction") not in (None, "")]
    values = arrays(rows)
    support = values["support"]
    nominal_proxy = np.array([as_float(row["nominal_proxy_fraction"]) for row in rows], dtype=float) if "nominal_proxy_fraction" in rows[0] else np.array([], dtype=float)
    countries = sorted({row["country"] for row in rows})
    geometry_fields = {
        "pixel_area_m2": f"pixel_area_m2_{PRIMARY_YEAR}",
        "geodesic_area_m2": f"pixel_area_geodesic_m2_{PRIMARY_YEAR}",
        "densified_relative_error": f"pixel_area_relative_error_vs_geodesic_{PRIMARY_YEAR}",
        "corner_relative_error": f"pixel_area_corner_relative_error_vs_geodesic_{PRIMARY_YEAR}",
    }
    geometry_audit = {
        name: {
            "mean": float(np.mean([as_float(row[field]) for row in rows])),
            "max": float(np.max([as_float(row[field]) for row in rows])),
            "min": float(np.min([as_float(row[field]) for row in rows])),
        }
        for name, field in geometry_fields.items()
        if field in rows[0]
    }
    geometry_audit["empty_intersections"] = int(sum(as_bool(row.get(empty_field)) for row in rows))
    geometry_audit["pixel_geometry_method"] = rows[0].get(f"pixel_geometry_method_{PRIMARY_YEAR}", "")
    geometry_audit["pixel_boundary_segments_per_edge"] = int(float(rows[0].get(f"pixel_boundary_segments_per_edge_{PRIMARY_YEAR}", "0") or 0))
    nonempty_rows = [row for row in rows if not as_bool(row.get(empty_field))]
    if not nonempty_rows:
        raise RuntimeError("The COG input has no nonempty overlays for the diagnostic sensitivity")
    nonempty_values = arrays(nonempty_rows)

    summary: dict[str, object] = {
        "input": portable_path(args.input),
        "n": len(rows),
        "year": PRIMARY_YEAR,
        "sample_design": "Class balanced proxy sensitivity experiment, not a probability sample for European population estimation.",
        "classes": {cls: sum(row["broad_class"] == cls for row in rows) for cls in LUCAS_CLASSES},
        "countries": countries,
        "support_values": [float(value) for value in support],
        "support_counts": {
            "below_025": int((support < 0.25).sum()),
            "at_least_075": int((support >= 0.75).sum()),
            "at_least_095": int((support >= 0.95).sum()),
        },
        "nominal_proxy_comparison": {
            "mean_nominal_proxy_fraction": float(nominal_proxy.mean()) if nominal_proxy.size else math.nan,
            "mean_absolute_difference": float(np.abs(support - nominal_proxy).mean()) if nominal_proxy.size else math.nan,
            "max_absolute_difference": float(np.abs(support - nominal_proxy).max()) if nominal_proxy.size else math.nan,
        },
        "score_definitions": {
            "hard_match_score": "M_i, the binary match between the bridged LUCAS class and raw WorldCover class.",
            "proxy_supported_match_score": "r_i M_i, which credits a hard match only in proportion to the released geometry proxy and does not identify actual reference support.",
            "residual_compatible_score": "r_i M_i+(1-r_i), a deliberately extreme counterfactual convention that credits every unresolved residual area as compatible.",
            "scenario_span": "1-r_i, the algebraic difference between the two conditional score conventions and not an accuracy interval.",
        },
        "overall": score_bundle(values),
        "nonempty_overlay_sensitivity": {
            "excluded_empty_overlay_n": int(sum(as_bool(row.get(empty_field)) for row in rows)),
            "matches": int(nonempty_values["hard_match"].sum()),
            **score_bundle(nonempty_values),
        },
        "class_metrics": class_metrics(rows),
        "support_strata": support_strata(rows),
        "geometry_audit": geometry_audit,
        "point_records": [
            {
                "point_id": row["point_id"],
                "country": row["country"],
                "lon": row["lon"],
                "lat": row["lat"],
                "class": row["broad_class"],
                "support": as_float(row["proxy_pixel_fraction"]),
                "match": bool(int(row[f"raw_{PRIMARY_YEAR}"]) == int(row["target_wc"])),
            }
            for row in rows
        ],
        "interpretation": (
            "These are deterministic finite sample score summaries conditional on a released geometry proxy. They do not validate actual reference support and are not lower or upper bounds on latent pixel agreement, "
            "confidence intervals, product reliability estimates, or probability sampled European accuracy estimates."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / f"class_metrics_{PRIMARY_YEAR}.csv", summary["class_metrics"])
    write_csv(args.output / f"support_strata_{PRIMARY_YEAR}.csv", summary["support_strata"])
    summary["figure_files"] = make_figures(summary, args.output)
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"n": len(rows), "year": PRIMARY_YEAR, "output": str(args.output), "figure_files": summary["figure_files"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
