"""Summarize complete paired COG results without altering existing project data."""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]

def read(p):
    with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def metrics(rows,source):
    if source=='original':
        r=np.array([float(x['proxy_pixel_fraction']) for x in rows]);m=np.array([int(x['raw_2021'])==int(x['target_wc']) for x in rows],dtype=int)
    else:
        r=np.array([float(x['quad_r']) for x in rows]);m=np.array([int(x['hard_match']) for x in rows])
    return {'n':len(rows),'quad_r':float(r.mean()),'quad_r_median':float(np.median(r)),
            'quad_r_p10':float(np.quantile(r,.1)),'quad_r_p90':float(np.quantile(r,.9)),
            'r_below_025':int((r<.25).sum()),'r_ge_075':int((r>=.75).sum()),'hard':float(m.mean()),
            'match_n':int(m.sum()),'proxy_score':float((r*m).mean()),'compatible_score':float((r*m+1-r).mean()),'span':float((1-r).mean())}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--sample',type=Path,default=ROOT/'revision/results/new_sample/corrected_800.csv')
    parser.add_argument('--conditions',type=Path,default=ROOT/'revision/results/new_sample/cog_conditions.csv')
    parser.add_argument('--paired',type=Path,default=ROOT/'revision/results/new_sample/cog_theoretical_gps_paired.csv')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    out=args.output_dir.resolve()
    if not out.is_relative_to((ROOT/'runs').resolve()): parser.error('--output-dir must be inside this package runs/')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'comparison_summary.json').exists(): raise FileExistsError('Refusing to overwrite an existing result')
    original=read(ROOT/'data_processed/lucas_cog_exact_pilot.csv')
    sample=read(args.sample); conditions=read(args.conditions)
    s={r['point_id']:r for r in sample}; t=[r for r in conditions if r['condition']=='theoretical'];g=[r for r in conditions if r['condition']=='gps']
    assert len(t)==800 and len({r['point_id'] for r in t})==800 and {r['point_id'] for r in t}==set(s), 'Incomplete theoretical sample'
    by_t={r['point_id']:r for r in t}; by_g={r['point_id']:r for r in g}; ids=sorted(set(by_t)&set(by_g))
    pair_t=[by_t[k] for k in ids];pair_g=[by_g[k] for k in ids]
    paired=read(args.paired)
    offsets=np.array([float(r['gps_offset_m']) for r in paired]);d=np.array([float(by_g[k]['quad_r'])-float(by_t[k]['quad_r']) for k in ids])
    info={'original_800':metrics(original,'original'),'corrected_800_theoretical':metrics(t,'corrected'),
          'corrected_newly_eligible_8_subset':metrics([r for r in t if s[r['point_id']]['added_by_8']=='1'],'corrected'),
          'corrected_other_subset':metrics([r for r in t if s[r['point_id']]['added_by_8']=='0'],'corrected'),
          'corrected_800_by_class':{c:metrics([r for r in t if r['broad_class']==c],'corrected') for c in 'ABCDEFGH'},
          'paired_theoretical':metrics(pair_t,'corrected'),'paired_gps':metrics(pair_g,'corrected'),
          'paired_diagnostics':{'n':len(ids),'gps_offset_m_median':float(np.median(offsets)),'gps_offset_m_p90':float(np.quantile(offsets,.9)),
              'gps_offset_m_max':float(offsets.max()),'different_raw_class_n':sum(by_t[k]['raw_2021']!=by_g[k]['raw_2021'] for k in ids),
              'different_hard_match_n':sum(by_t[k]['hard_match']!=by_g[k]['hard_match'] for k in ids),
              'mean_gps_minus_theoretical_quad_r':float(d.mean()),'median_gps_minus_theoretical_quad_r':float(np.median(d))},
          'note':'Class-balanced fixed samples; not European accuracy estimates. GPS and theoretical paired values use different point-centred raster cells and geometry, not a pure isolated position or shape effect.'}
    with (out/'comparison_summary.json').open('x',encoding='utf-8') as f: f.write(json.dumps(info,indent=2))
    print(json.dumps(info,indent=2))
if __name__=='__main__':main()
