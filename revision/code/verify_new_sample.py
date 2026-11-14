"""Independent standard-library cross-check of completed new-sample tables."""
import csv
import json
import math
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parents[2]
folder = root / 'revision/results/new_sample'
def read(name):
    with (folder / name).open(encoding='utf8', newline='') as f:
        return list(csv.DictReader(f))
def near(a,b):
    return math.isclose(float(a),float(b),rel_tol=0,abs_tol=1e-10)
sample=read('corrected_800.csv')
cond=read('cog_conditions.csv')
paired=read('cog_theoretical_gps_paired.csv')
ref=json.loads((folder/'comparison_summary.json').read_text(encoding='utf8'))
status=json.loads((folder/'cog_status.json').read_text(encoding='utf8'))
assert len(sample)==800 and len({r['point_id'] for r in sample})==800
assert Counter(r['broad_class'] for r in sample)=={c:100 for c in 'ABCDEFGH'}
assert sum(r['added_by_8']=='1' for r in sample)==60
by_sample={r['point_id']:r for r in sample}
keys={(r['point_id'],r['condition']) for r in cond}
assert len(cond)==1599 and len(keys)==1599
theory={r['point_id']:r for r in cond if r['condition']=='theoretical'}
gps={r['point_id']:r for r in cond if r['condition']=='gps'}
assert len(theory)==800 and len(gps)==799 and set(theory)==set(by_sample) and set(gps)<=set(theory)
assert len(paired)==799 and {r['point_id'] for r in paired}==set(gps)
assert status['completed_conditions']==1599 and status['failures']==[]
for r in cond:
    assert r['broad_class']==by_sample[r['point_id']]['broad_class']
    assert int(r['target_wc'])==int(by_sample[r['point_id']]['target_wc'])
    m=int(r['raw_2021'])==int(r['target_wc'])
    assert int(r['hard_match'])==m
    for shape in ('quad','rad'):
        coverage=float(r[shape+'_r'])
        assert math.isfinite(coverage) and 0 <= coverage <= 1
        assert near(r[shape+'_p'],coverage*m)
        assert near(r[shape+'_c'],coverage*m+1-coverage)
for row in paired:
    for label, records in (('theoretical',theory),('gps',gps)):
        record=records[row['point_id']]
        for metric in ('raw_2021','hard_match','quad_r','rad_r','quad_p','rad_p','quad_c','rad_c'):
            assert near(row[label+'_'+metric],record[metric]), (row['point_id'],label,metric)
for records,key in ((list(theory.values()),'corrected_800_theoretical'),
                    ([theory[k] for k in gps],'paired_theoretical'),
                    (list(gps.values()),'paired_gps')):
    n=len(records);summary=ref[key]
    assert n==summary['n']
    for col,target in (('quad_r','quad_r'),('hard_match','hard'),('quad_p','proxy_score'),('quad_c','compatible_score')):
        assert near(sum(float(r[col]) for r in records)/n,summary[target]), (key,col)
    assert sum(int(r['hard_match']) for r in records)==summary['match_n']
ids=set(gps)
raw_change=sum(theory[k]['raw_2021']!=gps[k]['raw_2021'] for k in ids)
match_change=sum(theory[k]['hard_match']!=gps[k]['hard_match'] for k in ids)
assert raw_change==ref['paired_diagnostics']['different_raw_class_n']==93
assert match_change==ref['paired_diagnostics']['different_hard_match_n']==42
with (root/'revision/results/corrected/r3_corrected_full_resampled_800.csv').open(encoding='utf8',newline='') as f:
    corrected=list(csv.DictReader(f))
assert [r['point_id'] for r in sample]==[r['point_id'] for r in corrected]
print(json.dumps({'status':'PASS','theoretical':len(theory),'gps':len(gps),'paired':len(paired),
                  'conditions':len(cond),'new_8_records':60,'theoretical_matches':sum(int(r['hard_match']) for r in theory.values()),
                  'paired_theoretical_matches':sum(int(theory[k]['hard_match']) for k in ids),
                  'paired_gps_matches':sum(int(gps[k]['hard_match']) for k in ids),
                  'changed_raw_class':raw_change,'changed_match':match_change,
                  'scope':'Stored-table arithmetic and joins; remote raster retrieval and geometry construction not rerun.'},indent=2))
