"""Audit the literal-8 direction-field change, sample reproducibly, read only inputs."""
import argparse, csv, json, math, sys, time
from collections import Counter
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'vendor'),str(ROOT/'scripts'),str(HERE.parent)]
import run_lucas_worldcover_pilot as original
import radial_sensitivity as rad
from shapely.geometry import box

RAW=None
OUT=None
OLD=ROOT/'data_processed/lucas_cog_exact_pilot.csv'
DR=('SURVEY_CPRN_LC1N','SURVEY_CPRNC_LC1E','SURVEY_CPRNC_LC1S','SURVEY_CPRNC_LC1W')

def rows_corrected():
    out=[];field_counts=Counter(); coord8=Counter()
    with RAW.open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            if row.get('POINT_COPERNICUS')!='1' or row.get('SURVEY_CPRN_CANDO')!='1': continue
            code=(row.get('SURVEY_CPRN_LC') or '').strip()
            cls=original.broad_class(code);target=original.target_worldcover(code)
            lat=original.parse_number(row.get('POINT_LAT'));lon=original.parse_number(row.get('POINT_LONG'))
            for field in ('POINT_LAT','POINT_LONG'):
                if (row.get(field) or '').strip()=='8': coord8[field]+=1
            vals=[8. if (row.get(k) or '').strip()=='8' else original.parse_number(row.get(k)) for k in DR]
            if cls is None or target is None or lat is None or lon is None or any(v is None for v in vals): continue
            date=row.get('SURVEY_DATE') or ''
            if not date.startswith('2022-'): continue
            for k,v in zip(DR,vals):
                if v==8: field_counts[k]+=1
            n,e,s,w=(min(v,50.) for v in vals)
            out.append({'point_id':row.get('POINT_ID',''),'country':row.get('POINT_NUTS0',''),
                        'lat':lat,'lon':lon,'survey_date':date,'obs_type':row.get('SURVEY_OBS_TYPE',''),
                        'cprn_code':code,'broad_class':cls,'target_wc':target,'n_m':n,'e_m':e,'s_m':s,'w_m':w})
    return out,field_counts,coord8

def savecsv(name, rows):
    with (OUT/name).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def nominal(row):
    # Use the original projected coordinates, not a recentered origin: tiny
    # floating-point differences can change strict threshold counts at .25/.5.
    return original.support_fraction(row,rad.PROJECT)

def stats(rows):
    vals=np.array([nominal(r) for r in rows]);return {'n':len(vals),'mean_r':float(vals.mean()),'median_r':float(np.median(vals)),
      'below_025':int((vals<.25).sum()),'below_05':int((vals<.5).sum()),'at_least_075':int((vals>=.75).sum()),'at_least_095':int((vals>=.95).sum())}

def main():
    global RAW, OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--cache',type=Path,default=rad.CACHE)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    RAW=args.raw.resolve(); OUT=args.output_dir.resolve(); rad.CACHE=args.cache.resolve()
    if not OUT.is_relative_to((ROOT/'runs').resolve()): parser.error('--output-dir must be inside this package runs/')
    OUT.mkdir(parents=True,exist_ok=True)
    names=('corrected_800.csv','newly_eligible_records.csv','reconstructed_original_800.csv','audit_summary.json')
    if any((OUT/name).exists() for name in names): raise FileExistsError('Refusing to overwrite existing results')
    start=time.monotonic(); old=original.read_rows(RAW); new,fields,coord8=rows_corrected()
    old_ids={r['point_id'] for r in old}; new_ids={r['point_id'] for r in new}
    assert len(old_ids)==len(old) and len(new_ids)==len(new) and old_ids<=new_ids
    added=[r for r in new if r['point_id'] not in old_ids]
    old_draw=[dict(r) for r in original.stratified_sample(old,800,17017)]
    draw=[dict(r) for r in original.stratified_sample(new,800,17017)]
    with OLD.open(encoding='utf-8',newline='') as f: frozen=list(csv.DictReader(f))
    frozen_ids=[r['point_id'] for r in frozen]; old_draw_ids=[r['point_id'] for r in old_draw]
    frozen_set=set(frozen_ids);old_draw_set=set(old_draw_ids); draw_set={r['point_id'] for r in draw}
    if not (len(frozen_set)==len(old_draw_set)==len(draw_set)==800): raise ValueError('nonunique sample')
    cache=json.loads(rad.CACHE.read_text(encoding='utf-8'))
    lookup={(str(v.get('year')),str(v.get('point_id')),str(v.get('tile'))):v for v in cache.values() if isinstance(v,dict)}
    for r in draw:
        r['tile']=f"{'N' if math.floor(r['lat']/3)*3>=0 else 'S'}{abs(math.floor(r['lat']/3)*3):02d}{'E' if math.floor(r['lon']/3)*3>=0 else 'W'}{abs(math.floor(r['lon']/3)*3):03d}"
        r['added_by_8']=int(r['point_id'] not in old_ids)
        r['in_frozen_800']=int(r['point_id'] in frozen_set)
        r['cache_hit_2021']=int(('2021',r['point_id'],r['tile']) in lookup)
        r['nominal_quad_r']=nominal(r)
    savecsv('corrected_800.csv',draw)
    savecsv('newly_eligible_records.csv',added)
    for r in old_draw: r['nominal_quad_r']=nominal(r)
    savecsv('reconstructed_original_800.csv',old_draw)
    info={'raw':str(RAW),'frame_old_n':len(old),'frame_new_n':len(new),'frame_added_n':len(added),
          'added_class_counts':dict(sorted(Counter(r['broad_class'] for r in added).items())),
          'added_country_counts':dict(sorted(Counter(r['country'] for r in added).items())),
          'direction_8_counts_corrected_frame':dict(fields),'eligible_flag_coordinate_literal8_counts_before_other_filters':dict(coord8),
          'old_sample_reconstruction_exact_order':old_draw_ids==frozen_ids,
          'old_sample_reconstruction_same_ids':old_draw_set==frozen_set,
          'old_sample_overlap_n':len(old_draw_set&frozen_set),
          'corrected_sample_classes':dict(sorted(Counter(r['broad_class'] for r in draw).items())),
          'corrected_sample_countries':dict(sorted(Counter(r['country'] for r in draw).items())),
          'corrected_vs_frozen_overlap':len(draw_set&frozen_set),'corrected_vs_reconstructed_old_overlap':len(draw_set&old_draw_set),
          'corrected_sample_newly_eligible_n':sum(r['added_by_8'] for r in draw),
          'corrected_sample_cache_hits_2021_n':sum(r['cache_hit_2021'] for r in draw),
          'corrected_sample_distinct_tiles_n':len({r['tile'] for r in draw}),
          'old_frame_nominal_quad':stats(old),'added_frame_nominal_quad':stats(added),'corrected_frame_nominal_quad':stats(new),
          'old_sample_reconstructed_nominal_quad':stats(old_draw),'corrected_sample_nominal_quad':stats(draw),
          'elapsed_seconds':time.monotonic()-start,
          'caveat':'Only literal 8 in four distance fields changed; original coordinate and other sentinel handling retained. Nominal squares are not COG pixels.'}
    with (OUT/'audit_summary.json').open('x',encoding='utf-8') as f: f.write(json.dumps(info,indent=2,ensure_ascii=False))
    print(json.dumps(info,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
