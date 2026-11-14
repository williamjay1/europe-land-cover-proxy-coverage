"""Independently check historical fixed-800 scores; no raw or geometry rerun."""
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'data_processed' / 'lucas_cog_exact_pilot.csv'
REFERENCE = ROOT / 'data_processed' / 'lucas_cog_exact_pilot_summary.json'
BRIDGE = ROOT / 'runs' / 'bridge' / 'summary.json'
with INPUT.open(encoding='utf-8', newline='') as f:
    rows = list(csv.DictReader(f))
ref = json.loads(REFERENCE.read_text(encoding='utf-8'))['metrics']
assert len(rows) == 800
assert len({r['point_id'] for r in rows}) == 800
assert Counter(r['broad_class'] for r in rows) == {c: 100 for c in 'ABCDEFGH'}
assert len({r['country'] for r in rows}) == 27
r = [float(row['proxy_pixel_fraction']) for row in rows]
assert all(math.isfinite(x) and 0 <= x <= 1 for x in r)
m = [int(row['raw_2021'] == row['target_wc']) for row in rows]
n = len(rows)
metrics = {
    'mean_proxy_pixel_fraction': sum(r) / n,
    'hard_match_score': sum(m) / n,
    'proxy_supported_match_score': sum(x*y for x, y in zip(r, m)) / n,
    'residual_compatible_score': sum(x*y + 1-x for x, y in zip(r, m)) / n,
    'mean_scenario_span': sum(1-x for x in r) / n,
}
for key, value in metrics.items():
    assert math.isclose(value, ref[key], rel_tol=0, abs_tol=1e-11), (key, value, ref[key])
assert sum(m) == 285
assert math.isclose(metrics['residual_compatible_score'] - metrics['proxy_supported_match_score'], metrics['mean_scenario_span'], rel_tol=0, abs_tol=1e-11)
result = {'status': 'PASS', 'historical_only': True, 'input_sha256': hashlib.sha256(INPUT.read_bytes()).hexdigest(), 'n': n, 'countries': 27, 'matches': sum(m), 'metrics': metrics}
if BRIDGE.exists():
    observed = {x['scenario']: (x['matched_n'], x['changed_from_baseline_n']) for x in json.loads(BRIDGE.read_text(encoding='utf-8'))['results']}
    assert observed == {'baseline': (285, 0), 'one_step': (353, 68), 'permissive': (386, 101)}, observed
    result['bridge_recomputed_counts'] = observed
else:
    result['bridge_recomputed_counts'] = 'not run; optional'
print(json.dumps(result, indent=2))
