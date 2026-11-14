"""GPS-centered rerun on original WorldCover COG; no existing file is changed.

For matching original and GPS COG cells the frozen class is reused. For other
cells read the 2021 source COG. Both variants use the original quadrilateral,
not the official radial polygon. GPS quality and observation-type limitations
are reported; this is not a probability sample or a validated reference mask.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse, json, math, os, sys, time
from collections import defaultdict
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from shapely.geometry import Polygon

ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser()
parser.add_argument('--gps',type=Path,default=ROOT/'revision/results/gps/r3_gps_offsets.csv')
parser.add_argument('--output-dir',type=Path,required=True)
args=parser.parse_args()
HERE=args.output_dir.resolve()
if not HERE.is_relative_to((ROOT/'runs').resolve()): parser.error('--output-dir must be inside this package runs/')
HERE.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'scripts'))
import validate_worldcover_cog_probe as cog
SOURCE=ROOT/'data_processed/lucas_cog_exact_pilot.csv'
GPS=args.gps.resolve()
STREAM=HERE/'r3_gps_cog_reads.jsonl'
POINTS=HERE/'r3_gps_cog_comparison.csv'
SUMMARY=HERE/'r3_gps_cog_summary.json'
if any(p.exists() for p in (STREAM, POINTS, SUMMARY)): raise FileExistsError('Output exists; no overwrite.')
df=pd.read_csv(SOURCE,dtype={'point_id':str}).merge(pd.read_csv(GPS,dtype={'point_id':str}),on='point_id',validate='one_to_one',suffixes=('','_gps'))
assert len(df)==800 and int(df.gps_numeric_valid.sum())==798
valid=df[df.gps_numeric_valid].copy()
valid['gps_tile']=[cog.tile_name(lat,lon) for lat,lon in zip(valid.SURVEY_GPS_LAT,valid.SURVEY_GPS_LONG)]
groups=defaultdict(list)
for row in valid.itertuples(index=False):
    if not row.gps_same_worldcover_cell:
        groups[row.gps_tile].append((row.point_id,row.SURVEY_GPS_LAT,row.SURVEY_GPS_LONG))
assert sum(map(len,groups.values()))==415
print(f'GPS valid={len(valid)}; same-cell cache=383; remote GPS reads=415; tiles={len(groups)}',flush=True)

def read_tile(tile, rows):
    url=cog.tile_url('2021',tile)
    last=None
    for attempt in range(2):
        try:
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_HTTP_TIMEOUT='30',GDAL_HTTP_MAX_RETRY='2',GDAL_PAM_ENABLED='NO'):
                with rasterio.open(url) as ds:
                    assert ds.crs.to_epsg()==4326 and ds.width==36000 and ds.height==36000
                    results=[]
                    for pid,lat,lon in rows:
                        ri,ci=ds.index(lon,lat)
                        if not (0<=ri<ds.height and 0<=ci<ds.width): raise ValueError('Out-of-tile index')
                        a=ds.read(1,window=((ri,ri+1),(ci,ci+1)),masked=True)
                        value=None if bool(np.ma.getmaskarray(a)[0,0]) else int(a[0,0])
                        results.append({'point_id':pid,'tile':tile,'row':ri,'col':ci,'raw':value,'error':None})
                    return results
        except Exception as exc:
            last=f'{type(exc).__name__}: {exc}'
            time.sleep(1)
    return [{'point_id':pid,'tile':tile,'raw':None,'error':last} for pid,lat,lon in rows]

raw_by_id={str(row.point_id):int(row.raw_2021) for row in valid.itertuples(index=False) if row.gps_same_worldcover_cell}
failures=[]
with STREAM.open('x',encoding='utf8') as stream:
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures={pool.submit(read_tile,tile,rows):tile for tile,rows in groups.items()}
        for i,future in enumerate(as_completed(futures),1):
            for item in future.result():
                stream.write(json.dumps(item)+'\n')
                if item['error'] is None: raw_by_id[item['point_id']]=item['raw']
                else: failures.append(item)
            stream.flush()
            print(f'tiles={i}/{len(groups)} resolved={len(raw_by_id)} failed={len(failures)}',flush=True)

to_3035=Transformer.from_crs('EPSG:4326','EPSG:3035',always_xy=True)
def grid(lat,lon):
    y0=math.floor(lat/3)*3; x0=math.floor(lon/3)*3
    return y0,x0,math.floor((y0+3-lat)*12000),math.floor((lon-x0)*12000)
def coverage(row,lat,lon):
    y0,x0,ri,ci=grid(lat,lon)
    left=x0+ci/12000; right=x0+(ci+1)/12000
    top=y0+3-ri/12000; bottom=y0+3-(ri+1)/12000
    corners=[(left,top),(right,top),(right,bottom),(left,bottom)]
    pixel=Polygon([to_3035.transform(*xy) for xy in cog.densify_ring(corners,16)])
    x,y=to_3035.transform(lon,lat)
    proxy=cog.support_polygon(row._asdict(),x,y)
    if pixel.area<=0: raise ValueError('Invalid COG pixel area')
    return max(0.,min(1.,proxy.intersection(pixel).area/pixel.area))

out=[]
for row in valid.itertuples(index=False):
    before=coverage(row,row.lat,row.lon)
    new=coverage(row,row.SURVEY_GPS_LAT,row.SURVEY_GPS_LONG)
    value=raw_by_id.get(str(row.point_id))
    known=value in (10,20,30,40,50,60,70,80,90,100)
    match=int(value==row.target_wc) if known else None
    out.append({'point_id':row.point_id,'country':row.country,'broad_class':row.broad_class,'obs_type':row.obs_type,'gps_offset_m':row.gps_offset_m,
        'gps_same_worldcover_cell':row.gps_same_worldcover_cell,'gps_tile':row.gps_tile,
        'old_raw':row.raw_2021,'gps_raw':value,'gps_raw_known':known,'old_match':int(row.raw_2021==row.target_wc),'gps_match':match,
        'old_coverage_saved':row.proxy_pixel_fraction,'old_coverage_reconstructed':before,'gps_centered_coverage':new})
result=pd.DataFrame(out)
maxdiff=float((result.old_coverage_saved-result.old_coverage_reconstructed).abs().max())
print(f'Original pixel geometry max absolute deviation={maxdiff}',flush=True)
assert maxdiff<1e-6
result.to_csv(POINTS,index=False,mode='x',lineterminator='\n')
common=result[result.gps_raw_known].copy()
same=common.gps_same_worldcover_cell
def stats(g):
    if len(g)==0: return {'n':0}
    old_h=g.old_match.mean(); new_h=g.gps_match.mean()
    old_p=(g.old_match*g.old_coverage_saved).mean(); new_p=(g.gps_match*g.gps_centered_coverage).mean()
    return {'n':len(g),'old_matches':int(g.old_match.sum()),'gps_matches':int(g.gps_match.sum()),
        'old_hard':float(old_h),'gps_hard':float(new_h),'hard_difference':float(new_h-old_h),
        'old_proxy_supported':float(old_p),'gps_proxy_supported':float(new_p),'proxy_difference':float(new_p-old_p),
        'old_mean_coverage':float(g.old_coverage_saved.mean()),'gps_mean_coverage':float(g.gps_centered_coverage.mean()),
        'raw_class_changed_n':int((g.old_raw!=g.gps_raw).sum()),'match_changed_n':int((g.old_match!=g.gps_match).sum()),
        'match_0_to_1':int(((g.old_match==0)&(g.gps_match==1)).sum()),'match_1_to_0':int(((g.old_match==1)&(g.gps_match==0)).sum())}
sm={'n_original':800,'n_gps_numeric_valid':798,'n_gps_missing':2,'reused_original_pixel_class_n':383,
    'remote_requested_n':415,'remote_failed_n':len(failures),'no_valid_worldcover_class_n':int((~result.gps_raw_known).sum()),
    'original_geometry_reconstruction_max_abs_difference':maxdiff,'all_paired':stats(common),
    'same_cell':stats(common[same]),'different_cell':stats(common[~same]),
    'by_obs_type':{str(k):stats(g) for k,g in common.groupby('obs_type')},
    'interpretation':'Paired finite-sample sensitivity to GPS relocation changes both map cell and proxy position. Quadrilateral is the original study proxy, not official 2022 radial geometry. GPS quality, field photo interpretation and geolocation differences remain unresolved; no population accuracy inference.'}
with SUMMARY.open('x',encoding='utf8') as f:json.dump(sm,f,indent=2,allow_nan=False)
print(json.dumps(sm,indent=2,allow_nan=False))
