"""Validate WMS class decoding and exact WorldCover COG pixel geometry."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

import rasterio  # noqa: E402
from pyproj import Geod, Transformer  # noqa: E402
from shapely.geometry import Polygon  # noqa: E402


WORLD_COVER = {
    "2020": ("v100", "ESA_WorldCover_10m_2020_v100"),
    "2021": ("v200", "ESA_WorldCover_10m_2021_v200"),
}


# A WorldCover COG cell is defined in geographic coordinates.  Mapping only
# its four corners to an equal area CRS makes its curved latitude and longitude
# edges into straight chords.  The cells are small, but densifying the edges
# makes that approximation explicit and auditable.
PIXEL_BOUNDARY_SEGMENTS = 16
PIXEL_GEOMETRY_METHOD = "EPSG3035_densified_WGS84_cell_edges_v1"
WGS84_GEOD = Geod(ellps="WGS84")


def tile_name(lat: float, lon: float) -> str:
    lat0 = math.floor(lat / 3.0) * 3
    lon0 = math.floor(lon / 3.0) * 3
    lat_prefix = "N" if lat0 >= 0 else "S"
    lon_prefix = "E" if lon0 >= 0 else "W"
    return f"{lat_prefix}{abs(lat0):02d}{lon_prefix}{abs(lon0):03d}"


def tile_url(year: str, tile: str) -> str:
    version, stem = WORLD_COVER[year]
    return (
        "https://esa-worldcover.s3.eu-central-1.amazonaws.com/"
        f"{version}/{year}/map/{stem}_{tile}_Map.tif"
    )


def support_polygon(row: dict[str, str], x: float, y: float) -> Polygon:
    n = min(float(row["n_m"]), 50.0)
    e = min(float(row["e_m"]), 50.0)
    s = min(float(row["s_m"]), 50.0)
    w = min(float(row["w_m"]), 50.0)
    return Polygon([(x - w, y), (x, y + n), (x + e, y), (x, y - s)])


def pixel_corners(src: rasterio.DatasetReader, row: int, col: int) -> list[tuple[float, float]]:
    """Return the four geographic corners of one COG cell in ring order."""

    return [
        src.transform * (col, row),
        src.transform * (col + 1, row),
        src.transform * (col + 1, row + 1),
        src.transform * (col, row + 1),
    ]


def densify_ring(corners: list[tuple[float, float]], segments: int = PIXEL_BOUNDARY_SEGMENTS) -> list[tuple[float, float]]:
    """Densify coordinate aligned geographic cell edges without duplicate corners."""

    if segments < 1:
        raise ValueError("segments must be positive")
    boundary: list[tuple[float, float]] = []
    for start, end in zip(corners, corners[1:] + corners[:1]):
        for step in range(segments):
            fraction = step / segments
            boundary.append(
                (
                    start[0] + fraction * (end[0] - start[0]),
                    start[1] + fraction * (end[1] - start[1]),
                )
            )
    return boundary


def pixel_geometry(
    src: rasterio.DatasetReader,
    row: int,
    col: int,
    transformer: Transformer,
    segments: int = PIXEL_BOUNDARY_SEGMENTS,
) -> tuple[Polygon, dict[str, object]]:
    """Build the cell footprint and record an independent area audit.

    The intersection is calculated in EPSG:3035 from a densified cell boundary.
    EPSG:3035 is equal area over the European analysis region.  A WGS84 geodesic
    calculation along the same densified boundary is retained only as an
    implementation check; it is not mixed into the planar intersection.
    """

    corners = pixel_corners(src, row, col)
    boundary = densify_ring(corners, segments)
    projected = [transformer.transform(lon, lat) for lon, lat in boundary]
    projected_corners = [transformer.transform(lon, lat) for lon, lat in corners]
    pixel = Polygon(projected)
    corner_polygon = Polygon(projected_corners)
    lons, lats = zip(*boundary)
    geodesic_area, _ = WGS84_GEOD.polygon_area_perimeter(lons, lats)
    geodesic_area = abs(float(geodesic_area))
    pixel_area = abs(float(pixel.area))
    corner_area = abs(float(corner_polygon.area))
    return pixel, {
        "pixel_geometry_method": PIXEL_GEOMETRY_METHOD,
        "pixel_boundary_segments_per_edge": int(segments),
        "pixel_area_geodesic_m2": geodesic_area,
        "pixel_area_corner_3035_m2": corner_area,
        "pixel_area_relative_error_vs_geodesic": abs(pixel_area - geodesic_area) / geodesic_area if geodesic_area else math.nan,
        "pixel_area_corner_relative_error_vs_geodesic": abs(corner_area - geodesic_area) / geodesic_area if geodesic_area else math.nan,
    }


def pixel_polygon(src: rasterio.DatasetReader, row: int, col: int, transformer: Transformer) -> Polygon:
    """Backward compatible convenience wrapper for the audited footprint."""

    return pixel_geometry(src, row, col, transformer)[0]


def intersection_metrics(
    support: Polygon,
    pixel: Polygon,
    pixel_audit: dict[str, object] | None = None,
) -> dict[str, object]:
    """Return auditable area metrics for a support and pixel intersection.

    The denominator is the projected area of the actual geographic raster
    pixel.  It is not forced to 100 square metres because a 10 m raster in
    EPSG:4326 has latitude dependent ground area.
    """

    support_geom = support if support.is_valid else support.buffer(0)
    pixel_geom = pixel if pixel.is_valid else pixel.buffer(0)
    try:
        intersection = support_geom.intersection(pixel_geom)
    except Exception:
        intersection = support_geom.buffer(0).intersection(pixel_geom.buffer(0))
    pixel_area = float(pixel_geom.area)
    intersection_area = float(max(0.0, intersection.area))
    fraction = intersection_area / pixel_area if pixel_area > 0 else 0.0
    metrics: dict[str, object] = {
        "support_polygon_area_m2": float(support_geom.area),
        "pixel_area_m2": pixel_area,
        "pixel_area_ratio_to_100m2": pixel_area / 100.0 if pixel_area else math.nan,
        "intersection_area_m2": intersection_area,
        "support_fraction": max(0.0, min(1.0, fraction)),
        "intersection_geometry_type": intersection.geom_type,
        "intersection_is_empty": bool(intersection.is_empty),
        "support_geometry_valid": bool(support_geom.is_valid),
        "pixel_geometry_valid": bool(pixel_geom.is_valid),
    }
    if pixel_audit:
        metrics.update(pixel_audit)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17017)
    args = parser.parse_args()

    input_path = ROOT / "data_processed" / "lucas_worldcover_pilot.csv"
    output_dir = ROOT / "data_processed"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "worldcover_cog_probe.csv"
    summary_path = output_dir / "worldcover_cog_probe_summary.json"

    with input_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: min(args.n, len(rows))]

    to_3035 = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    for row in rows:
        row["tile"] = tile_name(float(row["lat"]), float(row["lon"]))

    raw_by_year: dict[str, dict[str, int | None]] = {year: {} for year in WORLD_COVER}
    exact_support: dict[str, dict[str, float]] = {year: {} for year in WORLD_COVER}
    geometry_records: dict[str, dict[str, dict[str, object]]] = {year: {} for year in WORLD_COVER}
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        for year in WORLD_COVER:
            groups[(year, row["tile"])].append(row)

    total_groups = len(groups)
    for group_index, ((year, tile), group) in enumerate(sorted(groups.items()), start=1):
        print(f"COG progress: {group_index}/{total_groups} {year} {tile}", flush=True)
        url = tile_url(year, tile)
        with rasterio.open(url) as src:
            if src.crs is None or src.crs.to_epsg() != 4326:
                raise RuntimeError(f"Expected an EPSG:4326 WorldCover COG, found {src.crs!s} for {url}")
            for row in group:
                lon = float(row["lon"])
                lat = float(row["lat"])
                r_index, c_index = src.index(lon, lat)
                value = src.read(1, window=((r_index, r_index + 1), (c_index, c_index + 1)))
                raw_by_year[year][row["point_id"]] = int(value[0, 0]) if value.size else None
                x, y = to_3035.transform(lon, lat)
                support = support_polygon(row, x, y)
                pixel, pixel_audit = pixel_geometry(src, r_index, c_index, to_3035)
                geometry = intersection_metrics(support, pixel, pixel_audit)
                exact_support[year][row["point_id"]] = float(geometry["support_fraction"])
                geometry_records[year][row["point_id"]] = geometry

    fields = list(rows[0]) + [
        "raw_2020", "raw_2021", "raw_wms_equal_2020", "raw_wms_equal_2021",
        "proxy_pixel_fraction", "proxy_fraction_difference",
    ]
    for year in WORLD_COVER:
        fields.extend([
            f"pixel_area_m2_{year}", f"pixel_area_ratio_to_100m2_{year}",
            f"pixel_area_geodesic_m2_{year}", f"pixel_area_corner_3035_m2_{year}",
            f"pixel_area_relative_error_vs_geodesic_{year}",
            f"pixel_area_corner_relative_error_vs_geodesic_{year}",
            f"pixel_boundary_segments_per_edge_{year}", f"pixel_geometry_method_{year}",
            f"intersection_area_m2_{year}", f"intersection_geometry_type_{year}",
            f"intersection_is_empty_{year}", f"support_geometry_valid_{year}",
            f"pixel_geometry_valid_{year}",
        ])
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            point_id = row["point_id"]
            raw_2020 = raw_by_year["2020"].get(point_id)
            raw_2021 = raw_by_year["2021"].get(point_id)
            row["raw_2020"] = raw_2020
            row["raw_2021"] = raw_2021
            row["raw_wms_equal_2020"] = int(str(raw_2020) == row.get("wc_2020", ""))
            row["raw_wms_equal_2021"] = int(str(raw_2021) == row.get("wc_2021", ""))
            row["proxy_pixel_fraction"] = exact_support["2021"][point_id]
            row["proxy_fraction_difference"] = exact_support["2021"][point_id] - float(row["support_fraction"])
            for year in WORLD_COVER:
                geometry = geometry_records[year][point_id]
                row[f"pixel_area_m2_{year}"] = geometry["pixel_area_m2"]
                row[f"pixel_area_ratio_to_100m2_{year}"] = geometry["pixel_area_ratio_to_100m2"]
                row[f"pixel_area_geodesic_m2_{year}"] = geometry["pixel_area_geodesic_m2"]
                row[f"pixel_area_corner_3035_m2_{year}"] = geometry["pixel_area_corner_3035_m2"]
                row[f"pixel_area_relative_error_vs_geodesic_{year}"] = geometry["pixel_area_relative_error_vs_geodesic"]
                row[f"pixel_area_corner_relative_error_vs_geodesic_{year}"] = geometry["pixel_area_corner_relative_error_vs_geodesic"]
                row[f"pixel_boundary_segments_per_edge_{year}"] = geometry["pixel_boundary_segments_per_edge"]
                row[f"pixel_geometry_method_{year}"] = geometry["pixel_geometry_method"]
                row[f"intersection_area_m2_{year}"] = geometry["intersection_area_m2"]
                row[f"intersection_geometry_type_{year}"] = geometry["intersection_geometry_type"]
                row[f"intersection_is_empty_{year}"] = int(geometry["intersection_is_empty"])
                row[f"support_geometry_valid_{year}"] = int(geometry["support_geometry_valid"])
                row[f"pixel_geometry_valid_{year}"] = int(geometry["pixel_geometry_valid"])
            writer.writerow(row)

    # Keep the exact support values aligned with the row order used for the
    # nominal support values.  ``exact_support`` is populated by sorted
    # year/tile groups, so relying on dictionary insertion order would pair
    # different points and inflate the geometry difference.
    exact = np.array([exact_support["2021"][row["point_id"]] for row in rows], dtype=float)
    nominal = np.array([float(row["support_fraction"]) for row in rows], dtype=float)
    summary = {
        "n": len(rows),
        "tiles": sorted({row["tile"] for row in rows}),
        "wms_raw_agreement_2020": float(np.mean([str(raw_by_year["2020"].get(row["point_id"])) == row.get("wc_2020", "") for row in rows])),
        "wms_raw_agreement_2021": float(np.mean([str(raw_by_year["2021"].get(row["point_id"])) == row.get("wc_2021", "") for row in rows])),
        "nominal_support_mean": float(nominal.mean()),
        "exact_support_mean": float(exact.mean()),
        "absolute_support_difference_mean": float(np.abs(exact - nominal).mean()),
        "absolute_support_difference_max": float(np.abs(exact - nominal).max()),
        "geometry_audit": {
            year: {
                "pixel_area_m2_min": float(min(geometry_records[year][row["point_id"]]["pixel_area_m2"] for row in rows)),
                "pixel_area_m2_max": float(max(geometry_records[year][row["point_id"]]["pixel_area_m2"] for row in rows)),
                "pixel_area_m2_mean": float(np.mean([geometry_records[year][row["point_id"]]["pixel_area_m2"] for row in rows])),
                "pixel_area_relative_error_vs_geodesic_max": float(max(geometry_records[year][row["point_id"]]["pixel_area_relative_error_vs_geodesic"] for row in rows)),
                "pixel_area_corner_relative_error_vs_geodesic_max": float(max(geometry_records[year][row["point_id"]]["pixel_area_corner_relative_error_vs_geodesic"] for row in rows)),
                "pixel_geometry_method": PIXEL_GEOMETRY_METHOD,
                "intersection_geometry_types": sorted({geometry_records[year][row["point_id"]]["intersection_geometry_type"] for row in rows}),
                "empty_intersections": int(sum(geometry_records[year][row["point_id"]]["intersection_is_empty"] for row in rows)),
                "invalid_support_geometries": int(sum(not geometry_records[year][row["point_id"]]["support_geometry_valid"] for row in rows)),
                "invalid_pixel_geometries": int(sum(not geometry_records[year][row["point_id"]]["pixel_geometry_valid"] for row in rows)),
            }
            for year in WORLD_COVER
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {output_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
