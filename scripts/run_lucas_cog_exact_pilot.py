"""Create a reproducible COG based reference support experiment.

The experiment is deliberately class balanced.  It is a measurement audit of
released LUCAS support geometry, not a probability sampled estimate of
European land cover accuracy.  The default run samples 800 records directly
from the released 2022 LUCAS Copernicus file and reads the raw WorldCover 2021
v200 class from the original Cloud Optimized GeoTIFFs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import run_lucas_worldcover_pilot as common  # noqa: E402
import validate_worldcover_cog_probe as cog  # noqa: E402

VENDOR = SCRIPT_DIR.parent / "vendor"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))
import rasterio  # noqa: E402
from pyproj import Transformer  # noqa: E402


ROOT = SCRIPT_DIR.parent
DEFAULT_RELEASE = ROOT / "data_raw" / "lucas_2022" / "l2022_survey_cop.csv"
DEFAULT_INPUT = ROOT / "data_processed" / "lucas_worldcover_pilot.csv"
DEFAULT_OUTPUT = ROOT / "data_processed" / "lucas_cog_exact_pilot.csv"


def portable_path(path: Path | str) -> str:
    """Return a project relative path, or a filename for an external input."""

    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return candidate.name


def select_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], str]:
    """Return a deterministic class balanced sample and its source label."""

    if args.from_release:
        rows = common.read_rows(args.release)
        sample = common.stratified_sample(rows, args.n, args.seed)
        return sample, str(args.release)

    with args.input.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    sample = common.stratified_sample(rows, min(args.n, len(rows)), args.seed)
    return sample, str(args.input)


def geometry_summary(records: list[dict[str, object]]) -> dict[str, object]:
    """Summarize area and projection checks without treating them as outcomes."""

    def values(name: str) -> np.ndarray:
        return np.array([float(record[name]) for record in records], dtype=float)

    areas = values("pixel_area_m2")
    relative_errors = values("pixel_area_relative_error_vs_geodesic")
    corner_relative_errors = values("pixel_area_corner_relative_error_vs_geodesic")
    return {
        "pixel_area_m2_min": float(areas.min()),
        "pixel_area_m2_max": float(areas.max()),
        "pixel_area_m2_mean": float(areas.mean()),
        "pixel_area_ratio_to_100m2_mean": float(values("pixel_area_ratio_to_100m2").mean()),
        "pixel_area_relative_error_vs_geodesic_mean": float(relative_errors.mean()),
        "pixel_area_relative_error_vs_geodesic_max": float(relative_errors.max()),
        "pixel_area_corner_relative_error_vs_geodesic_mean": float(corner_relative_errors.mean()),
        "pixel_area_corner_relative_error_vs_geodesic_max": float(corner_relative_errors.max()),
        "empty_intersections": int(sum(bool(record["intersection_is_empty"]) for record in records)),
        "invalid_support_geometries": int(sum(not bool(record["support_geometry_valid"]) for record in records)),
        "invalid_pixel_geometries": int(sum(not bool(record["pixel_geometry_valid"]) for record in records)),
        "pixel_geometry_method": cog.PIXEL_GEOMETRY_METHOD,
        "pixel_boundary_segments_per_edge": cog.PIXEL_BOUNDARY_SEGMENTS,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=800)
    parser.add_argument("--seed", type=int, default=17017)
    parser.add_argument("--year", choices=sorted(cog.WORLD_COVER), default="2021")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--from-release", action="store_true")
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE,
                        help="Path to the external LUCAS CSV when using --from-release.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "data_processed" / "cog_window_cache",
        help="Persistent cache for one raw COG class and geometry record per point.",
    )
    parser.add_argument("--force-reload", action="store_true")
    args = parser.parse_args()
    if args.n < 8:
        raise ValueError("n must allow one observation in each LUCAS broad class")

    rows, input_source = select_rows(args)
    if not rows:
        raise RuntimeError("No eligible LUCAS Copernicus observations were found")
    output_path = args.output
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.cache_dir / "worldcover_cog_pixel_cache.json"
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    except json.JSONDecodeError:
        cache = {}
    if not isinstance(cache, dict):
        cache = {}

    def cache_complete(record: object) -> bool:
        return (
            isinstance(record, dict)
            and "raw_value" in record
            and "support_fraction" in record
            and record.get("pixel_geometry_method") == cog.PIXEL_GEOMETRY_METHOD
        )

    # Permit a deterministic rerun to reuse a completed geometry record without
    # reopening the remote COG.  The conventional cache key also contains row
    # and column, which are unavailable until a COG is opened, so build a safe
    # secondary index from the stored point identifier and tile.
    cache_by_point = {
        "|".join([str(record.get("year", "")), str(record.get("point_id", "")), str(record.get("tile", ""))]): record
        for record in cache.values()
        if cache_complete(record)
    }

    year = args.year
    to_3035 = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    for row in rows:
        row["tile"] = cog.tile_name(float(row["lat"]), float(row["lon"]))
        row["nominal_proxy_fraction"] = common.support_fraction(row, to_3035)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["tile"])].append(row)

    raw: dict[str, int] = {}
    proxy_pixel_fraction: dict[str, float] = {}
    geometry_records: dict[str, dict[str, object]] = {}
    cache_hits = 0
    cache_updates = 0
    raw_reads = 0
    total_groups = len(groups)
    for index, (tile, group) in enumerate(sorted(groups.items()), start=1):
        print(f"COG progress: {index}/{total_groups} {year} {tile}", flush=True)
        direct_records = {
            str(row["point_id"]): cache_by_point.get("|".join([year, str(row["point_id"]), tile]))
            for row in group
        }
        if not args.force_reload and all(cache_complete(record) for record in direct_records.values()):
            for row in group:
                point_id = str(row["point_id"])
                record = direct_records[point_id]
                raw[point_id] = int(record["raw_value"])
                proxy_pixel_fraction[point_id] = float(record["support_fraction"])
                geometry_records[point_id] = record
                cache_hits += 1
            continue
        url = cog.tile_url(year, tile)
        with rasterio.open(url) as src:
            if src.crs is None or src.crs.to_epsg() != 4326:
                raise RuntimeError(f"Expected an EPSG:4326 WorldCover COG, found {src.crs!s} for {url}")
            for row in group:
                lon = float(row["lon"])
                lat = float(row["lat"])
                row_index, col_index = src.index(lon, lat)
                cache_key = "|".join(
                    [year, str(row["point_id"]), tile, str(row_index), str(col_index), url]
                )
                cached = None if args.force_reload else (direct_records.get(str(row["point_id"])) or cache.get(cache_key))
                if cache_complete(cached):
                    record = cached
                    cache_hits += 1
                else:
                    if isinstance(cached, dict) and "raw_value" in cached and not args.force_reload:
                        raw_value = int(cached["raw_value"])
                    else:
                        value = src.read(1, window=((row_index, row_index + 1), (col_index, col_index + 1)))
                        if value.size == 0:
                            raise RuntimeError(f"Empty COG window for {year} {tile} {row['point_id']}")
                        raw_value = int(value[0, 0])
                        raw_reads += 1
                    x, y = to_3035.transform(lon, lat)
                    support = cog.support_polygon(row, x, y)
                    pixel, pixel_audit = cog.pixel_geometry(src, row_index, col_index, to_3035)
                    record = {
                        "year": year,
                        "point_id": str(row["point_id"]),
                        "tile": tile,
                        "row_index": int(row_index),
                        "col_index": int(col_index),
                        "url": url,
                        "raw_value": raw_value,
                        **cog.intersection_metrics(support, pixel, pixel_audit),
                    }
                    cache[cache_key] = record
                    cache_by_point["|".join([year, str(row["point_id"]), tile])] = record
                    cache_updates += 1
                point_id = str(row["point_id"])
                raw[point_id] = int(record["raw_value"])
                proxy_pixel_fraction[point_id] = float(record["support_fraction"])
                geometry_records[point_id] = record

    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    geometry_fields = [
        "pixel_area_m2",
        "pixel_area_ratio_to_100m2",
        "pixel_area_geodesic_m2",
        "pixel_area_corner_3035_m2",
        "pixel_area_relative_error_vs_geodesic",
        "pixel_area_corner_relative_error_vs_geodesic",
        "pixel_boundary_segments_per_edge",
        "pixel_geometry_method",
        "intersection_area_m2",
        "intersection_geometry_type",
        "intersection_is_empty",
        "support_geometry_valid",
        "pixel_geometry_valid",
    ]
    fields = list(rows[0]) + ["proxy_pixel_fraction", f"raw_{year}"] + [
        f"{name}_{year}" for name in geometry_fields
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            point_id = str(row["point_id"])
            row["proxy_pixel_fraction"] = proxy_pixel_fraction[point_id]
            row[f"raw_{year}"] = raw[point_id]
            geometry = geometry_records[point_id]
            for name in geometry_fields:
                row[f"{name}_{year}"] = geometry[name]
            writer.writerow(row)

    support = np.array([proxy_pixel_fraction[str(row["point_id"])] for row in rows], dtype=float)
    prediction = np.array([raw[str(row["point_id"])] for row in rows], dtype=int)
    target = np.array([int(row["target_wc"]) for row in rows], dtype=int)
    hard_match = (prediction == target).astype(float)
    proxy_supported = support * hard_match
    residual_compatible = proxy_supported + (1.0 - support)
    scenario_span = residual_compatible - proxy_supported
    records = [geometry_records[str(row["point_id"])] for row in rows]
    summary = {
        "input": portable_path(input_source),
        "output": portable_path(output_path),
        "sample_n": len(rows),
        "seed": args.seed,
        "year": year,
        "class_balanced": True,
        "sample_class_counts": {
            cls: sum(row["broad_class"] == cls for row in rows) for cls in sorted({row["broad_class"] for row in rows})
        },
        "sample_country_counts": {
            country: sum(row["country"] == country for row in rows) for country in sorted({row["country"] for row in rows})
        },
        "cache_path": portable_path(cache_path),
        "cache_hits": cache_hits,
        "cache_geometry_updates": cache_updates,
        "raw_cog_reads": raw_reads,
        "score_definitions": {
            "hard_match_score": "M_i, the raster class match indicator.",
            "proxy_supported_match_score": "r_i M_i, which credits a hard match by the observed geometric support proxy fraction.",
            "residual_compatible_score": "r_i M_i + (1-r_i), the counterfactual score that treats every unresolved residual area as compatible.",
            "scenario_span": "1-r_i, the algebraic difference between the two conditional score conventions.",
        },
        "metrics": {
            "mean_proxy_pixel_fraction": float(support.mean()),
            "mean_nominal_proxy_fraction": float(np.mean([float(row["nominal_proxy_fraction"]) for row in rows])),
            "mean_absolute_proxy_fraction_difference": float(np.mean(np.abs(support - np.array([float(row["nominal_proxy_fraction"]) for row in rows])))),
            "max_absolute_proxy_fraction_difference": float(np.max(np.abs(support - np.array([float(row["nominal_proxy_fraction"]) for row in rows])))),
            "hard_match_score": float(hard_match.mean()),
            "proxy_supported_match_score": float(proxy_supported.mean()),
            "residual_compatible_score": float(residual_compatible.mean()),
            "mean_scenario_span": float(scenario_span.mean()),
            "support_below_025_n": int((support < 0.25).sum()),
            "support_at_least_075_n": int((support >= 0.75).sum()),
            "support_at_least_095_n": int((support >= 0.95).sum()),
        },
        "geometry_audit": geometry_summary(records),
        "interpretation": (
            "The scores are deterministic summaries of the selected class balanced sample. "
            "They are not confidence intervals, lower or upper bounds on latent pixel agreement, "
            "or probability sampled European accuracy estimates."
        ),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {output_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
