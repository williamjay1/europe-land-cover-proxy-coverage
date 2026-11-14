"""Audit released LUCAS reference support fields before map comparison.

This script uses every eligible 2022 Copernicus observation retained by the
published filtering rule.  It examines only the released directional distance
fields and a centered nominal 10 m footprint.  It is therefore a descriptive
audit of the released reference records, not a land cover accuracy estimate.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import run_lucas_worldcover_pilot as common  # noqa: E402

VENDOR = SCRIPT_DIR.parent / "vendor"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))
from pyproj import Transformer  # noqa: E402


ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT = ROOT / "data_raw" / "lucas_2022" / "l2022_survey_cop.csv"
DEFAULT_OUTPUT = ROOT / "data_processed" / "lucas_reference_support_audit.json"
DEFAULT_FIGURE_DIR = ROOT / "data_processed" / "lucas_exact_analysis" / "figures"


def portable_path(path: Path) -> str:
    """Return a path relative to the project root for portable audit metadata."""

    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def describe(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
        "below_025_n": int((values < 0.25).sum()),
        "below_050_n": int((values < 0.50).sum()),
        "at_least_075_n": int((values >= 0.75).sum()),
        "at_least_095_n": int((values >= 0.95).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    args = parser.parse_args()

    rows = common.read_rows(args.input)
    if not rows:
        raise RuntimeError("No eligible LUCAS Copernicus records were found")
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    values = np.empty(len(rows), dtype=float)
    by_class: dict[str, list[float]] = {code: [] for code in "ABCDEFGH"}
    for index, row in enumerate(rows):
        value = common.support_fraction(row, transformer)
        values[index] = value
        by_class[str(row["broad_class"])].append(value)

    class_rows = []
    for broad_class in sorted(by_class):
        class_rows.append({"broad_class": broad_class, **describe(np.array(by_class[broad_class]))})

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.figure_dir / "fig1_release_support.png"
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    axes[0].hist(values, bins=np.linspace(0, 1, 21), color="#4c78a8", edgecolor="white")
    axes[0].axvline(values.mean(), color="#e45756", linewidth=2, label=f"Mean {values.mean():.2f}")
    axes[0].set(xlim=(0, 1), xlabel="Nominal proxy fraction", ylabel="Number of records", title="Released records")
    axes[0].legend(frameon=False)
    x = np.arange(len(class_rows))
    class_means = [float(entry["mean"]) for entry in class_rows]
    axes[1].bar(x, class_means, color="#54a24b", width=0.65)
    axes[1].set(xticks=x, xticklabels=[str(entry["broad_class"]) for entry in class_rows], ylim=(0, 1), xlabel="LUCAS broad class", ylabel="Mean nominal proxy fraction", title="Class patterns")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(class_rows[0]))
        writer.writeheader()
        writer.writerows(class_rows)

    summary = {
        "input": portable_path(args.input),
        "filtering_rule": (
            "POINT_COPERNICUS=1, SURVEY_CPRN_CANDO=1, a valid broad LUCAS class and WorldCover bridge, "
            "valid coordinates and four cardinal distance fields, and a 2022 survey date."
        ),
        "support_definition": (
            "Intersection fraction between the released cardinal distance quadrilateral and a centered nominal "
            "10 m by 10 m square in EPSG:3035. This is a release wide reference geometry diagnostic, not the "
            "exact WorldCover COG pixel support used in the map comparison."
        ),
        "overall": describe(values),
        "by_broad_class": class_rows,
        "figure_file": portable_path(figure_path),
        "interpretation": (
            "This is a complete descriptive audit of records retained by the stated released file filter. "
            "It carries no external area weights and is not a European land cover population estimate."
        ),
    }
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {args.output}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
