"""Create the minimal historical 800-cell metadata dependency from an old cache."""
import argparse
import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser()
p.add_argument('--cache', required=True, type=Path, help='Read-only original cache JSON')
p.add_argument('--output', required=True, type=Path, help='New JSON inside revision/inputs')
a = p.parse_args()
out = a.output.resolve()
if not out.is_relative_to((root / 'revision/inputs').resolve()):
    p.error('output must be inside revision/inputs')
with (root / 'data_processed/lucas_cog_exact_pilot.csv').open(encoding='utf8', newline='') as f:
    sample = list(csv.DictReader(f))
if len(sample) != 800 or len({r['point_id'] for r in sample}) != 800:
    raise ValueError('Historical sample is not the unique 800 points')
cache = json.loads(a.cache.read_text(encoding='utf8'))
lookup = {}
for value in cache.values():
    if not isinstance(value, dict) or str(value.get('year')) != '2021':
        continue
    if value.get('pixel_geometry_method') != 'EPSG3035_densified_WGS84_cell_edges_v1':
        continue
    key = (str(value.get('point_id')), str(value.get('tile')))
    if key in lookup:
        raise ValueError(f'duplicate COG metadata {key}')
    lookup[key] = value
fields = ('point_id', 'year', 'tile', 'row_index', 'col_index', 'raw_value', 'pixel_geometry_method')
result = {}
for row in sample:
    rec = lookup[(row['point_id'], row['tile'])]
    if int(rec['raw_value']) != int(row['raw_2021']):
        raise ValueError(f'cached class mismatch: {row["point_id"]}')
    result[row['point_id']] = {key: rec[key] for key in fields}
with out.open('x', encoding='utf8') as f:
    json.dump(result, f, separators=(',', ':'), ensure_ascii=False)
print(f'Wrote {len(result)} historical cells to {out}')
