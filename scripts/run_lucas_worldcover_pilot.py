"""Run a small feasibility pilot linking LUCAS 2022 to WorldCover WMS.

The pilot is deliberately conservative.  LUCAS Copernicus observations provide
one central land cover code and four directional distances to the first cover
change.  We convert those distances into a diamond support polygon and measure
its overlap with a nominal 10 m footprint around the theoretical LUCAS point.
This produces a class support fraction, not a claim that the residual area has
any particular alternative class.

The script is a data feasibility tool.  Its output is descriptive and is not a
design based European area estimate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import requests


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

from pyproj import Transformer  # noqa: E402
from shapely.geometry import Polygon, box  # noqa: E402


WMS_URL = "https://titiler.terrascope.be/wms"
LAYERS = {
    "2020": ("esa-worldcover-map-10m-2020-v1_map", "2020-01-01"),
    "2021": ("esa-worldcover-map-10m-2021-v2_map", "2021-01-01"),
}

WORLD_COVER_RGB = {
    10: (0, 100, 0),
    20: (255, 187, 34),
    30: (255, 255, 76),
    40: (240, 150, 255),
    50: (250, 0, 0),
    60: (180, 180, 180),
    70: (240, 240, 240),
    80: (0, 100, 200),
    90: (0, 150, 160),
    95: (0, 207, 117),
    100: (250, 230, 160),
}


def parse_number(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if value in {"", "NA", "N/A", "8", "88", "888", "8888"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def broad_class(code: str | None) -> str | None:
    if not code:
        return None
    code = code.strip()
    if len(code) < 1 or code[0] not in "ABCDEFGH":
        return None
    return code[0]


def target_worldcover(code: str | None) -> int | None:
    """Map LUCAS land cover to a transparent broad WorldCover legend.

    This is intentionally a broad class bridge.  It is not used to claim
    semantic equivalence for detailed LUCAS subclasses.
    """

    cls = broad_class(code)
    if cls == "A":
        return 50
    if cls == "B":
        return 40
    if cls == "C":
        return 10
    if cls == "D":
        return 20
    if cls == "E":
        return 30
    if cls == "F":
        return 100 if str(code).startswith("F30") else 60
    if cls == "G":
        return 70 if str(code).startswith("G50") else 80
    if cls == "H":
        return 90
    return None


def rgb_to_worldcover(rgb: tuple[int, int, int] | None) -> int | None:
    if rgb is None:
        return None
    exact = {v: k for k, v in WORLD_COVER_RGB.items()}
    if rgb in exact:
        return exact[rgb]
    # WMS GetFeatureInfo normally returns the palette value exactly.  A small
    # nearest colour fallback makes the pilot robust to integer conversion.
    distances = {
        cls: sum((int(rgb[i]) - colour[i]) ** 2 for i in range(3))
        for cls, colour in WORLD_COVER_RGB.items()
    }
    cls, distance = min(distances.items(), key=lambda item: item[1])
    return cls if distance <= 5**2 else None


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("POINT_COPERNICUS") != "1":
                continue
            if row.get("SURVEY_CPRN_CANDO") != "1":
                continue
            code = (row.get("SURVEY_CPRN_LC") or "").strip()
            cls = broad_class(code)
            target = target_worldcover(code)
            lat = parse_number(row.get("POINT_LAT"))
            lon = parse_number(row.get("POINT_LONG"))
            n = parse_number(row.get("SURVEY_CPRN_LC1N"))
            e = parse_number(row.get("SURVEY_CPRNC_LC1E"))
            s = parse_number(row.get("SURVEY_CPRNC_LC1S"))
            w = parse_number(row.get("SURVEY_CPRNC_LC1W"))
            if cls is None or target is None or lat is None or lon is None:
                continue
            if any(value is None for value in (n, e, s, w)):
                continue
            survey_date = row.get("SURVEY_DATE", "") or ""
            if not survey_date.startswith("2022-"):
                continue
            rows.append(
                {
                    "point_id": row.get("POINT_ID", ""),
                    "country": row.get("POINT_NUTS0", ""),
                    "lat": lat,
                    "lon": lon,
                    "survey_date": row.get("SURVEY_DATE", ""),
                    "obs_type": row.get("SURVEY_OBS_TYPE", ""),
                    "cprn_code": code,
                    "broad_class": cls,
                    "target_wc": target,
                    "n_m": min(n, 50.0),
                    "e_m": min(e, 50.0),
                    "s_m": min(s, 50.0),
                    "w_m": min(w, 50.0),
                }
            )
    return rows


def stratified_sample(rows: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["broad_class"]][row["country"]].append(row)
    classes = sorted(grouped)
    if not classes:
        return []
    base, remainder = divmod(n, len(classes))
    selected: list[dict[str, Any]] = []
    for index, cls in enumerate(classes):
        quota = base + (1 if index < remainder else 0)
        country_rows = grouped[cls]
        for values in country_rows.values():
            rng.shuffle(values)
        countries = list(country_rows)
        rng.shuffle(countries)
        picked = 0
        while picked < quota and countries:
            progressed = False
            for country in list(countries):
                values = country_rows[country]
                if values:
                    selected.append(values.pop())
                    picked += 1
                    progressed = True
                    if picked >= quota:
                        break
                else:
                    countries.remove(country)
            if not progressed:
                break
    rng.shuffle(selected)
    return selected


def support_fraction(row: dict[str, Any], transformer: Transformer) -> float:
    """Overlap of the LUCAS central class diamond with a nominal 10 m square."""

    x, y = transformer.transform(row["lon"], row["lat"])
    # LUCAS C1 defines N, E, S and W as sector radii to the first cover change.
    diamond = Polygon(
        [(x - row["w_m"], y), (x, y + row["n_m"]), (x + row["e_m"], y), (x, y - row["s_m"])],
    )
    footprint = box(x - 5.0, y - 5.0, x + 5.0, y + 5.0)
    return max(0.0, min(1.0, diamond.intersection(footprint).area / 100.0))


def cache_key(layer: str, lat: float, lon: float) -> str:
    return f"{layer}|{lat:.7f}|{lon:.7f}"


def query_worldcover(
    session: requests.Session,
    layer: str,
    date: str,
    lat: float,
    lon: float,
    retries: int = 3,
) -> dict[str, Any]:
    # WMS 1.3.0 uses latitude then longitude axis order for EPSG:4326.
    delta = 0.001
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetFeatureInfo",
        "LAYERS": layer,
        "QUERY_LAYERS": layer,
        "STYLES": "",
        "FORMAT": "image/png",
        "CRS": "EPSG:4326",
        "BBOX": f"{lat-delta},{lon-delta},{lat+delta},{lon+delta}",
        "WIDTH": 101,
        "HEIGHT": 101,
        "I": 50,
        "J": 50,
        "INFO_FORMAT": "application/geo+json",
        "FEATURE_COUNT": 1,
        "TIME": date,
    }
    last_error = ""
    for attempt in range(retries):
        try:
            response = session.get(WMS_URL, params=params, timeout=45)
            response.raise_for_status()
            payload = response.json()
            features = payload.get("features", [])
            if not features:
                return {"class": None, "rgb": None, "error": "no_feature"}
            properties = features[0].get("properties", {})
            rgb = tuple(int(properties.get(f"band_{i}", -1)) for i in range(1, 4))
            if min(rgb) < 0:
                return {"class": None, "rgb": None, "error": "missing_rgb"}
            return {"class": rgb_to_worldcover(rgb), "rgb": list(rgb), "error": ""}
        except Exception as exc:  # network service errors are recorded, not hidden
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(1.5 * (attempt + 1))
    return {"class": None, "rgb": None, "error": last_error or "unknown_error"}


def bootstrap_mean(values: np.ndarray, seed: int, n_boot: int = 2000) -> tuple[float, float, float]:
    if values.size == 0:
        return (math.nan, math.nan, math.nan)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[indices].mean(axis=1)
    return (float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--seed", type=int, default=17017)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-2021", action="store_true")
    args = parser.parse_args()

    input_path = ROOT / "data_raw" / "lucas_2022" / "l2022_survey_cop.csv"
    output_dir = ROOT / "data_processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "lucas_worldcover_pilot.csv"
    summary_path = output_dir / "lucas_worldcover_pilot_summary.json"
    cache_path = output_dir / "worldcover_wms_cache.json"

    rows = read_rows(input_path)
    sample = stratified_sample(rows, args.n, args.seed)
    if not sample:
        raise RuntimeError("No valid LUCAS Copernicus rows were found.")

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    for row in sample:
        row["support_fraction"] = support_fraction(row, transformer)

    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    except json.JSONDecodeError:
        cache = {}
    cache_lock = Lock()
    query_specs = [(name, *spec) for name, spec in LAYERS.items() if not (args.skip_2021 and name == "2021")]

    def fetch(row: dict[str, Any], name: str, layer: str, date: str) -> tuple[str, str, dict[str, Any]]:
        key = cache_key(layer, row["lat"], row["lon"])
        with cache_lock:
            cached = cache.get(key)
        if cached is not None:
            return str(row["point_id"]), name, cached
        with requests.Session() as session:
            result = query_worldcover(session, layer, date, row["lat"], row["lon"])
        with cache_lock:
            cache[key] = result
        return str(row["point_id"]), name, result

    tasks = []
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for row in sample:
            for name, layer, date in query_specs:
                tasks.append(executor.submit(fetch, row, name, layer, date))
        for future in as_completed(tasks):
            point_id, name, result = future.result()
            row = next(item for item in sample if str(item["point_id"]) == point_id)
            row[f"wc_{name}"] = result.get("class")
            row[f"wc_{name}_rgb"] = ",".join(str(v) for v in result.get("rgb") or [])
            row[f"wc_{name}_error"] = result.get("error", "")
            completed += 1
            if completed == 1 or completed % 20 == 0 or completed == len(tasks):
                print(f"WMS progress: {completed}/{len(tasks)}", flush=True)

    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    fields = [
        "point_id", "country", "lat", "lon", "survey_date", "obs_type", "cprn_code",
        "broad_class", "target_wc", "n_m", "e_m", "s_m", "w_m", "support_fraction",
    ]
    for name, _, _ in query_specs:
        fields.extend([f"wc_{name}", f"wc_{name}_rgb", f"wc_{name}_error"])
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sample:
            writer.writerow({field: row.get(field, "") for field in fields})

    summary: dict[str, Any] = {
        "input_rows_after_filters": len(rows),
        "sample_n": len(sample),
        "seed": args.seed,
        "layers": [name for name, _, _ in query_specs],
        "worldcover_rgb_palette": {str(k): v for k, v in WORLD_COVER_RGB.items()},
        "sample_class_counts": dict(Counter(row["broad_class"] for row in sample)),
        "sample_country_counts": dict(Counter(row["country"] for row in sample)),
        "support_fraction": {
            "mean": float(np.mean([row["support_fraction"] for row in sample])),
            "median": float(np.median([row["support_fraction"] for row in sample])),
            "p10": float(np.quantile([row["support_fraction"] for row in sample], 0.10)),
            "p90": float(np.quantile([row["support_fraction"] for row in sample], 0.90)),
        },
        "metrics": {},
    }

    for name, _, _ in query_specs:
        predictions = np.array([row.get(f"wc_{name}") or np.nan for row in sample], dtype=float)
        targets = np.array([row["target_wc"] for row in sample], dtype=float)
        support = np.array([row["support_fraction"] for row in sample], dtype=float)
        valid = ~np.isnan(predictions)
        if not valid.any():
            summary["metrics"][name] = {"valid_predictions": 0}
            continue
        match = (predictions[valid] == targets[valid]).astype(float)
        support_valid = support[valid]
        lower_agreement = support_valid * match
        upper_agreement = np.where(match == 1.0, 1.0, 1.0 - support_valid)
        metric = {
            "valid_predictions": int(valid.sum()),
            "missing_predictions": int((~valid).sum()),
            "crisp_accuracy": bootstrap_mean(match, args.seed + 1),
            "support_weighted_agreement_lower": bootstrap_mean(lower_agreement, args.seed + 2),
            "possible_agreement_upper": bootstrap_mean(upper_agreement, args.seed + 3),
            "conditional_supported_accuracy": float(lower_agreement.sum() / support_valid.sum()) if support_valid.sum() else math.nan,
            "mean_support_fraction_valid": float(support_valid.mean()),
        }
        area = {}
        for wc_class in sorted(set(int(value) for value in targets)):
            target_class = (targets[valid] == wc_class).astype(float)
            predicted_class = (predictions[valid] == wc_class).astype(float)
            lower_area = support_valid * target_class
            upper_area = lower_area + (1.0 - support_valid)
            area[str(wc_class)] = {
                "reference_support_lower_fraction": bootstrap_mean(lower_area, args.seed + 10 + wc_class),
                "reference_support_upper_fraction": bootstrap_mean(upper_area, args.seed + 30 + wc_class),
                "product_sample_fraction": bootstrap_mean(predicted_class, args.seed + 50 + wc_class),
                "difference_product_minus_lower": bootstrap_mean(predicted_class - lower_area, args.seed + 70 + wc_class),
            }
        metric["descriptive_class_area"] = area
        summary["metrics"][name] = metric

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {output_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
