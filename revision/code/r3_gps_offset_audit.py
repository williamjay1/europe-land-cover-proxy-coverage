"""Read immutable JRC CSV and quantify GPS versus theoretical point offsets."""
from pathlib import Path
from collections import Counter
import argparse, hashlib, json, math
import numpy as np
import pandas as pd
from pyproj import Geod

ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser()
parser.add_argument('--raw',type=Path,required=True)
parser.add_argument('--cache',type=Path,default=ROOT/'revision/inputs/legacy_800_cog_metadata.json')
parser.add_argument('--output-dir',type=Path,required=True)
args=parser.parse_args()
RAW=args.raw.resolve()
HERE=args.output_dir.resolve()
if not HERE.is_relative_to((ROOT/'runs').resolve()): parser.error('--output-dir must be inside this package runs/')
HERE.mkdir(parents=True,exist_ok=True)
SRC=ROOT/'data_processed/lucas_cog_exact_pilot.csv'
CACHE=args.cache.resolve()
out_csv=HERE/'r3_gps_offsets.csv'
out_json=HERE/'r3_gps_offset_summary.json'
if out_csv.exists() or out_json.exists(): raise FileExistsError('Output exists; no overwrite.')
digest=hashlib.sha256(RAW.read_bytes()).hexdigest()
assert digest=='b5d5414fd8058bb1067496080883138a819700c21c9300c82f4a31cfe1528d6f'
points=pd.read_csv(SRC,dtype={'point_id':str,'country':str,'broad_class':str})
cols=['POINT_ID','POINT_LAT','POINT_LONG','POINT_NUTS0','SURVEY_DATE','SURVEY_OBS_TYPE','SURVEY_GPS_PROJ','SURVEY_GPS_PREC','SURVEY_GPS_LAT','SURVEY_GPS_LONG','SURVEY_OBS_DIST','SURVEY_CALC_DIST','SURVEY_CPRN_LC','SURVEY_CPRN_LC1N','SURVEY_CPRNC_LC1E','SURVEY_CPRNC_LC1S','SURVEY_CPRNC_LC1W']
raw=pd.read_csv(RAW,usecols=cols,dtype={'POINT_ID':str},low_memory=False)
subset=raw[raw.POINT_ID.isin(points.point_id)].copy()
assert len(points)==800 and points.point_id.is_unique
assert len(subset)==800 and subset.POINT_ID.is_unique, f'Join not one-to-one, found {len(subset)} rows'
d=points.merge(subset,left_on='point_id',right_on='POINT_ID',how='left',validate='one_to_one')
assert (d.country==d.POINT_NUTS0).all() and (d.cprn_code==d.SURVEY_CPRN_LC).all()
assert np.allclose(d.lat,d.POINT_LAT,rtol=0,atol=1e-8) and np.allclose(d.lon,d.POINT_LONG,rtol=0,atol=1e-8)
assert (d.obs_type.astype(str)==d.SURVEY_OBS_TYPE.astype(str)).all()
for old,new in [('n_m','SURVEY_CPRN_LC1N'),('e_m','SURVEY_CPRNC_LC1E'),('s_m','SURVEY_CPRNC_LC1S'),('w_m','SURVEY_CPRNC_LC1W')]:
    assert np.allclose(d[old],np.minimum(d[new].astype(float),50),rtol=0,atol=1e-9)
for col in ['SURVEY_GPS_LAT','SURVEY_GPS_LONG','SURVEY_GPS_PREC','SURVEY_OBS_DIST','SURVEY_CALC_DIST']:
    d[col]=pd.to_numeric(d[col],errors='coerce')
d['gps_numeric_valid']=d.SURVEY_GPS_LAT.between(-90,90)&d.SURVEY_GPS_LONG.between(-180,180)&~((d.SURVEY_GPS_LAT==0)&(d.SURVEY_GPS_LONG==0))
mask=d.gps_numeric_valid
geod=Geod(ellps='WGS84')
azimuth,_,dist=geod.inv(d.loc[mask,'lon'].to_numpy(),d.loc[mask,'lat'].to_numpy(),d.loc[mask,'SURVEY_GPS_LONG'].to_numpy(),d.loc[mask,'SURVEY_GPS_LAT'].to_numpy())
d['gps_offset_m']=np.nan; d.loc[mask,'gps_offset_m']=dist
d['gps_azimuth_deg']=np.nan; d.loc[mask,'gps_azimuth_deg']=azimuth

# Check the grid formula against frozen COG row/column indices for all points.
cache=json.loads(CACHE.read_text(encoding='utf8'))
lookup={str(v['point_id']):v for v in cache.values() if v.get('year')=='2021' and v.get('pixel_geometry_method')=='EPSG3035_densified_WGS84_cell_edges_v1'}
assert set(d.point_id)<=set(lookup)
def grid(lat,lon):
    y0=math.floor(lat/3)*3; x0=math.floor(lon/3)*3
    return (y0,x0,math.floor((y0+3-lat)*12000),math.floor((lon-x0)*12000))
same=[]; checked=[]
for row in d.itertuples(index=False):
    orig=grid(row.lat,row.lon); rec=lookup[row.point_id]
    checked.append(orig[2]==rec['row_index'] and orig[3]==rec['col_index'])
    same.append(grid(row.SURVEY_GPS_LAT,row.SURVEY_GPS_LONG)==orig if row.gps_numeric_valid else None)
assert all(checked),f'Grid does not replicate original cache for {len(checked)-sum(checked)} points'
d['gps_same_worldcover_cell']=same

def counts(series):
    return {str(k):int(v) for k,v in series.fillna('NA').value_counts(dropna=False).sort_index().items()}
def summary(s):
    return {'n':len(s),'min':float(s.min()),'p10':float(s.quantile(.1)),'p25':float(s.quantile(.25)),'median':float(s.median()),'mean':float(s.mean()),'p75':float(s.quantile(.75)),'p90':float(s.quantile(.9)),'p95':float(s.quantile(.95)),'max':float(s.max()),**{f'gt_{x}m':int((s>x).sum()) for x in (1,5,10,20,30,50,100)}}
sm={'original_sha256':digest,'n':len(d),'one_to_one_join_n':len(subset),'gps_numeric_valid':int(mask.sum()),'gps_missing_or_invalid':int((~mask).sum()),
    'gps_proj_counts':counts(d.SURVEY_GPS_PROJ),'obs_type_counts':counts(d.SURVEY_OBS_TYPE),'gps_precision_recorded_n':int(d.SURVEY_GPS_PREC.notna().sum()),
    'geodesic_displacement_m':summary(d.loc[mask,'gps_offset_m']),
    'same_cog_cell':int((d.gps_same_worldcover_cell==True).sum()),'different_cog_cell':int((d.gps_same_worldcover_cell==False).sum()),
    'grid_formula_matches_original_cache_n':sum(checked),
    'by_obs_type':{str(k):summary(g.loc[g.gps_numeric_valid,'gps_offset_m']) for k,g in d.groupby('SURVEY_OBS_TYPE') if g.gps_numeric_valid.any()},
    'coordinate_note':'Numerical validity alone is not GPS positional accuracy; inspect GPS projection, precision and survey observation type. WGS84 geodesic displacement includes survey location differences and GPS error.',
    'same_cell_note':'WorldCover 2021 geographic COG grid derived from tile and row/col, checked against all 800 archived COG entries. Does not validate original COG class or GPS quality.',
    'survey_calc_dist_note':'Reported observed/calculated distance fields are retained for diagnostic comparison, not substituted for coordinates.'}
selected=['point_id','country','broad_class','lat','lon','obs_type','SURVEY_GPS_PROJ','SURVEY_GPS_PREC','SURVEY_GPS_LAT','SURVEY_GPS_LONG','SURVEY_OBS_DIST','SURVEY_CALC_DIST','gps_numeric_valid','gps_offset_m','gps_azimuth_deg','gps_same_worldcover_cell']
d[selected].to_csv(out_csv,index=False,mode='x',lineterminator='\n')
with out_json.open('x',encoding='utf8') as f: json.dump(sm,f,indent=2,allow_nan=False)
print(json.dumps(sm,indent=2,allow_nan=False))
