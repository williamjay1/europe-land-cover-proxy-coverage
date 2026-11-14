"""Check completed revision results; no raw access or remote raster reads."""
import csv
import json
import math
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parents[2]
res = root / 'revision/results'
runs = root / 'runs/revision_check'

def load(path):
    return json.loads(path.read_text(encoding='utf8'))

def table(path):
    with path.open(encoding='utf8', newline='') as f:
        return list(csv.DictReader(f))

def close(a, b, tolerance=1e-11):
    return math.isclose(float(a), float(b), abs_tol=tolerance, rel_tol=0)

corrected = load(res / 'corrected/r3_corrected_full_summary.json')
sample = table(res / 'corrected/r3_corrected_full_resampled_800.csv')
assert corrected['source_rows'] == 149408 and corrected['eligible_rows'] == 136605
assert corrected['old_frame_rows'] == 128082 and corrected['newly_included_vs_old'] == 8523
assert len(sample) == len({r['point_id'] for r in sample}) == 800
assert Counter(r['broad_class'] for r in sample) == {c:100 for c in 'ABCDEFGH'}
assert sum(r['in_old_800'].lower() == 'true' for r in sample) == 134
run_corrected = runs / 'corrected/r3_corrected_full_summary.json'
if run_corrected.exists():
    run = load(run_corrected)
    for k in ('source_rows','eligible_rows','old_frame_rows','newly_included_vs_old','counts_by_class','overall_geometry','old_new_pilot_overlap'):
        assert run[k] == corrected[k], k

gps = table(res / 'gps/r3_gps_cog_comparison.csv')
gps_ref = load(res / 'gps/r3_gps_cog_summary.json')
assert len(gps) == 798 and all(r['gps_raw_known'].lower() == 'true' for r in gps)
assert sum(int(r['old_match']) for r in gps) == 285
assert sum(int(r['gps_match']) for r in gps) == 282
assert sum(r['old_raw'] != r['gps_raw'] for r in gps) == 94
assert sum(r['old_match'] != r['gps_match'] for r in gps) == 39
for match, cover, key in (('old_match','old_coverage_saved','old_proxy_supported'),('gps_match','gps_centered_coverage','gps_proxy_supported')):
    value = sum(int(r[match])*float(r[cover]) for r in gps)/len(gps)
    assert close(value, gps_ref['all_paired'][key]), (key, value)
run_gps = runs / 'gps/r3_gps_offset_summary.json'
if run_gps.exists():
    assert load(run_gps) == load(res / 'gps/r3_gps_offset_summary.json')

radial = table(res / 'radial/paired_800.csv')
radial_ref = load(res / 'radial/summary_800.json')
assert len(radial) == 800
for col in ('quad_r','rad_r','delta_r','hard','quad_p','rad_p','quad_c','rad_c','quad_span','rad_span'):
    value = sum(float(r[col]) for r in radial)/len(radial)
    assert close(value, radial_ref[col]['mean']), (col, value, radial_ref[col]['mean'])
run_radial = runs / 'radial/summary_800.json'
if run_radial.exists():
    other = load(run_radial)
    for col in ('quad_r','rad_r','delta_r','hard','quad_p','rad_p','quad_c','rad_c','quad_span','rad_span'):
        assert close(other[col]['mean'], radial_ref[col]['mean'])
    assert other['counts'] == radial_ref['counts']
print(json.dumps({'status':'PASS','corrected_frame':136605,'corrected_sample':800,
                  'legacy_gps_paired':798,'legacy_radial':800,
                  'checked_reruns':{'corrected':run_corrected.exists(),'gps_offset':run_gps.exists(),'radial':run_radial.exists()},
                  'not_tested':'fresh remote new-sample and legacy GPS COG retrieval; official radial polygon'}, indent=2))
