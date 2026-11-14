"""Field-specific audit, eligible-frame reconstruction and nominal geometry.

Reads the immutable JRC CSV and existing pilot; writes only unique outputs in
this directory. Run with PYTHONDONTWRITEBYTECODE=1. No WorldCover lookups occur.
"""
from __future__ import annotations
import csv
import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Polygon, box
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description='Recompute corrected historical nominal geometry.')
parser.add_argument('--raw', type=Path, required=True, help='Read-only JRC CSV')
parser.add_argument('--output-dir', type=Path, required=True, help='New directory inside this package runs/')
parser.add_argument('--write-point-records', action='store_true', help='Also write large full point and field-frequency tables')
args = parser.parse_args()
RAW = args.raw.resolve()
HERE = args.output_dir.resolve()
if not HERE.is_relative_to((ROOT / 'runs').resolve()):
    parser.error('--output-dir must be inside this package runs/')
HERE.mkdir(parents=True, exist_ok=True)
FROZEN = ROOT / 'data_processed' / 'lucas_cog_exact_pilot.csv'
sys.path.insert(0, str(ROOT / 'scripts'))
import run_lucas_worldcover_pilot as original

prefix = 'r3_corrected_full_'
names = ['filter_flow.csv', 'country_class.csv',
         'class_geometry.csv', 'overall_geometry.csv', 'resampled_800.csv',
         'resampled_country_class.csv', 'summary.json']
if args.write_point_records:
    names += ['points.csv', 'field_values.csv']
paths = {name: HERE / (prefix + name) for name in names}
if any(p.exists() for p in paths.values()):
    raise FileExistsError('A requested output already exists; do not overwrite.')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
expected_sha = 'b5d5414fd8058bb1067496080883138a819700c21c9300c82f4a31cfe1528d6f'
assert sha(RAW) == expected_sha, 'Source differs from the previously identified raw file.'
df = pd.read_csv(RAW, dtype=str, keep_default_na=False, encoding='utf-8-sig')
directions = {'n':'SURVEY_CPRN_LC1N','e':'SURVEY_CPRNC_LC1E',
              's':'SURVEY_CPRNC_LC1S','w':'SURVEY_CPRNC_LC1W'}
audit_fields = ['POINT_ID','POINT_NUTS0','POINT_COPERNICUS','SURVEY_CPRN_CANDO',
                'SURVEY_CPRN_LC','SURVEY_DATE','POINT_LAT','POINT_LONG',
                *directions.values(),'SURVEY_GPS_LAT','SURVEY_GPS_LONG',
                'SURVEY_OBS_TYPE','SURVEY_GPS_PROJ','SURVEY_GPS_PREC']
audit_fields = [f for f in audit_fields if f in df.columns]
# This complete field-value inventory deliberately preserves literal NA/8/51.
if args.write_point_records:
    inventory = []
    for field in audit_fields:
        vals = df[field].str.strip().value_counts(dropna=False, sort=False)
        inventory.extend({'field':field,'literal_value':str(value),'n':int(n)} for value,n in vals.items())
    pd.DataFrame(inventory).to_csv(paths['field_values.csv'], index=False, mode='x', lineterminator='\n')

def numeric(field):
    # NA/blank/unparseable become missing; do not globally classify 8 as missing.
    a = pd.to_numeric(df[field].str.strip(), errors='coerce')
    return a.where(np.isfinite(a))

lat, lon = numeric('POINT_LAT'), numeric('POINT_LONG')
gpslat, gpslon = numeric('SURVEY_GPS_LAT'), numeric('SURVEY_GPS_LONG')
code = df['SURVEY_CPRN_LC'].str.strip()
dist = {k:numeric(v) for k,v in directions.items()}
# Strict domain validation is field-specific; 0 and 8 are valid distances.
valid_dist = pd.Series(True,index=df.index)
for d in dist.values():
    valid_dist &= d.notna() & d.between(0,51) & (d % 1 == 0)
checks = [
    ('source rows',pd.Series(True,index=df.index)),
    ('POINT_COPERNICUS equals 1',df.POINT_COPERNICUS.str.strip().eq('1')),
    ('SURVEY_CPRN_CANDO equals 1',df.SURVEY_CPRN_CANDO.str.strip().eq('1')),
    ('class starts A through H',code.str.match(r'^[A-H]',na=False)),
    ('valid theoretical geographic coordinates',lat.between(-90,90)&lon.between(-180,180)),
    ('four finite integral 0 to 51 m distances',valid_dist),
    ('SURVEY_DATE starts 2022-',df.SURVEY_DATE.str.startswith('2022-')),
]
mask = pd.Series(True,index=df.index)
flow=[]
for name,condition in checks:
    before = int(mask.sum()); mask &= condition
    flow.append({'step':name,'before':before,'excluded_at_step':before-int(mask.sum()),'remaining':int(mask.sum())})
pd.DataFrame(flow).to_csv(paths['filter_flow.csv'],index=False,mode='x',lineterminator='\n')
assert int(mask.sum()) == 136605, f'Unexpected eligible n {mask.sum()}'
sel=df.loc[mask].copy()
assert sel.POINT_ID.is_unique, 'Point ID is not unique in eligible frame.'
sel['point_id']=sel.POINT_ID
sel['country']=sel.POINT_NUTS0
sel['broad_class']=code.loc[mask].str[0]
sel['lat']=lat.loc[mask]
sel['lon']=lon.loc[mask]
sel['gps_lat']=gpslat.loc[mask]
sel['gps_lon']=gpslon.loc[mask]
sel['gps_valid']=gpslat.loc[mask].between(-90,90)&gpslon.loc[mask].between(-180,180)
sel['cprn_code']=code.loc[mask]
sel['obs_type']=sel.SURVEY_OBS_TYPE
sel['survey_date']=sel.SURVEY_DATE
sel['target_wc']=sel.cprn_code.map(original.target_worldcover)
for k,v in dist.items():
    sel[k+'_raw_m']=v.loc[mask].astype(int)
    # Preserve original implementation: cap 51 at 50; not measured exact edge.
    sel[k+'_m']=v.loc[mask].clip(upper=50).astype(float)

# Original class-by-country sampling routine and seed, applied to the corrected
# frame in original CSV order. Check exact reproduction of the frozen pilot
# under the original frame before using this routine for a fresh selection.
orig_rows=original.read_rows(RAW)
orig_sample=original.stratified_sample(orig_rows,800,17017)
frozen=pd.read_csv(FROZEN,dtype={'point_id':str})
old_ids=[str(r['point_id']) for r in orig_sample]
frozen_ids=frozen.point_id.astype(str).tolist()
assert old_ids == frozen_ids, 'Original sampling replay does not match frozen pilot in order.'
frame_records=sel[['point_id','country','broad_class']].to_dict('records')
new_sample=original.stratified_sample(frame_records,800,17017)
new_ids=[str(r['point_id']) for r in new_sample]
assert len(new_ids)==800 and len(set(new_ids))==800
sel['resampled_800'] = sel.point_id.isin(new_ids)
oldset=set(old_ids)
sel['in_old_800']=sel.point_id.isin(oldset)
assert int(sel.in_old_800.sum())==800

# Translation-invariant reconstruction in projected metric coordinates:
# W,N,E,S vertices about (0,0), square of width size co-centered at (0,0).
# Shapely batch geometry is the same as the original per-point construction.
w,n,e,s=(sel[k+'_m'].to_numpy() for k in 'wnes')
vertices=np.stack((np.column_stack((-w,np.zeros(len(w)))),
                   np.column_stack((np.zeros(len(w)),n)),
                   np.column_stack((e,np.zeros(len(w)))),
                   np.column_stack((np.zeros(len(w)),-s))),axis=1)
polys=shapely.polygons(vertices)
sel['proxy_polygon_valid']=shapely.is_valid(polys)
for size in (10,20,30):
    a=shapely.area(shapely.intersection(polys,box(-size/2,-size/2,size/2,size/2)))/size**2
    if not np.isfinite(a).all(): raise ValueError('Non-finite geometry result')
    sel[f'nominal_{size}m']=np.clip(a,0,1)
assert ((sel.nominal_20m-sel.nominal_10m)<1e-9).all()
assert ((sel.nominal_30m-sel.nominal_20m)<1e-9).all()

# Independently validate translation-free calculation against the full
# original projected EPSG:3035 point-centered implementation, on frozen 800.
ix=sel.set_index('point_id')
transform=Transformer.from_crs('EPSG:4326','EPSG:3035',always_xy=True)
diffs=[]
for record in orig_sample:
    x,y=transform.transform(record['lon'],record['lat'])
    p=Polygon([(x-record['w_m'],y),(x,y+record['n_m']),(x+record['e_m'],y),(x,y-record['s_m'])])
    for size in (10,20,30):
        q=p.intersection(box(x-size/2,y-size/2,x+size/2,y+size/2)).area/size**2
        diffs.append(abs(q-float(ix.loc[str(record['point_id']),f'nominal_{size}m'])))
maxdiff=max(diffs)
assert maxdiff < 1e-8, maxdiff

def summarize(part,size):
    x=part[f'nominal_{size}m']
    return {'n':int(len(x)),'mean':float(x.mean()),'median':float(x.median()),
            'p10':float(x.quantile(.1)),'p25':float(x.quantile(.25)),
            'p75':float(x.quantile(.75)),'p90':float(x.quantile(.9)),
            'zero_n':int((x==0).sum()),'below_025_n':int((x<.25).sum()),
            '025_to_below_050_n':int(((x>=.25)&(x<.5)).sum()),
            '050_to_below_075_n':int(((x>=.5)&(x<.75)).sum()),
            '075_to_1_n':int((x>=.75).sum()),'at_least_095_n':int((x>=.95).sum())}

overall=[{'nominal_size_m':size,**summarize(sel,size)} for size in (10,20,30)]
classes=[{'class':cls,'nominal_size_m':size,**summarize(part,size)}
         for cls,part in sel.groupby('broad_class',sort=True) for size in (10,20,30)]
pd.DataFrame(overall).to_csv(paths['overall_geometry.csv'],index=False,mode='x',lineterminator='\n')
pd.DataFrame(classes).to_csv(paths['class_geometry.csv'],index=False,mode='x',lineterminator='\n')
ct=pd.crosstab(sel.country,sel.broad_class).reindex(columns=list('ABCDEFGH'),fill_value=0)
ct['total']=ct.sum(axis=1)
assert int(ct.total.sum())==len(sel)
ct.to_csv(paths['country_class.csv'],mode='x',lineterminator='\n')
sample=sel.set_index('point_id').loc[new_ids].reset_index()
st=pd.crosstab(sample.country,sample.broad_class).reindex(columns=list('ABCDEFGH'),fill_value=0)
st['total']=st.sum(axis=1)
st.to_csv(paths['resampled_country_class.csv'],mode='x',lineterminator='\n')
outcols=['point_id','country','survey_date','obs_type','cprn_code','broad_class','target_wc',
         'lat','lon','gps_lat','gps_lon','gps_valid',*[k+'_raw_m' for k in 'nesw'],
         *[k+'_m' for k in 'nesw'],'proxy_polygon_valid','nominal_10m','nominal_20m',
         'nominal_30m','in_old_800','resampled_800']
if args.write_point_records:
    sel[outcols].to_csv(paths['points.csv'],index=False,mode='x',lineterminator='\n')
sample[outcols].to_csv(paths['resampled_800.csv'],index=False,mode='x',lineterminator='\n')
oldonly = set(r['point_id'] for r in orig_rows)
assert oldonly <= set(sel.point_id)
assert len(sel)-len(oldonly)==8523
summary={
 'raw_path':str(RAW),'raw_sha256':sha(RAW),'frozen_800_sha256':sha(FROZEN),
 'source_rows':len(df),'eligible_rows':len(sel),'old_frame_rows':len(orig_rows),
 'newly_included_vs_old':len(sel)-len(oldonly),'flow':flow,
 'counts_by_class':sel.broad_class.value_counts().sort_index().to_dict(),
 'countries':int(sel.country.nunique()),'gps_valid':int(sel.gps_valid.sum()),
 'gps_missing_or_invalid':int((~sel.gps_valid).sum()),
 'obs_type_counts':sel.obs_type.value_counts().sort_index().to_dict(),
 'directions_full_source_value_counts':{k:df[v].str.strip().value_counts().sort_index().to_dict() for k,v in directions.items()},
 'eligible_distance_8_n':{k:int((sel[k+'_raw_m']==8).sum()) for k in directions},
 'eligible_distance_51_n':{k:int((sel[k+'_raw_m']==51).sum()) for k in directions},
 'eligible_any_8_n':int((sel[[k+'_raw_m' for k in directions]]==8).any(axis=1).sum()),
 'eligible_any_51_n':int((sel[[k+'_raw_m' for k in directions]]==51).any(axis=1).sum()),
 'invalid_proxy_polygon_n':int((~sel.proxy_polygon_valid).sum()),
 'original_sampler_reproduced_frozen_800_in_order':True,
 'resampled_seed':17017,'resampled_class_counts':sample.broad_class.value_counts().sort_index().to_dict(),
 'old_new_pilot_overlap':len(set(new_ids)&oldset),
 'translation_method_max_abs_difference_vs_original_projected_800_3_scales':maxdiff,
 'overall_geometry':overall,
 'definitions_and_limits':[
  'Distance source integer domain 0..51 per C1 section 9.1.12; literal 8 is valid. NA is treated as missing because it is nonnumeric in this source; no claim made about other fields.',
  'Values 51 are capped at 50 m to match the original study geometry. 51 is a censored code, not an exact measured 51 m.',
  'Class and country filters replicate original design, with geographic ranges and direction integer domain explicitly validated. Class first-letter bridge is not a full legend validation.',
  'Quadrilateral vertices at four directional radii; centered square 10/20/30 m; projected-plane translation invariance. Not official radial geometry or actual map pixels.',
  'Unweighted descriptive summaries of selected records; not probability-weighted EU population estimates. 20/30 m values are geometry only, not map accuracy.',
  'New 800 uses unchanged original class-balanced country-round-robin algorithm and seed; it is a distinct sample with no newly queried map classes.',
  'Survey GPS not required for co-centered nominal geometry; missing coordinates remain missing and actual pixel alignment is not evaluated.',
  'NA/null encoding is empirically observed but no comprehensive JRC 2022 field-level missing-value dictionary was located; unverified details are not generalized.'
 ],
 'software':{'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__,'shapely':shapely.__version__}
}
with paths['summary.json'].open('x',encoding='utf8') as f: json.dump(summary,f,indent=2,ensure_ascii=False,allow_nan=False)
print(json.dumps({k:summary[k] for k in ('source_rows','eligible_rows','flow','counts_by_class','gps_valid','gps_missing_or_invalid','obs_type_counts','eligible_any_8_n','eligible_any_51_n','invalid_proxy_polygon_n','old_new_pilot_overlap','translation_method_max_abs_difference_vs_original_projected_800_3_scales','overall_geometry')},indent=2))
