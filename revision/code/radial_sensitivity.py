"""Paired cardinal-quarter-arc sensitivity. Outputs only beside this script.

The reconstruction is NOT the published GeoPackage: it fixes the theoretical
point to isolate shape and approximates the official geography-buffer pipeline.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'vendor'))
sys.path.insert(0, str(ROOT / 'scripts'))
import run_lucas_worldcover_pilot as common
from pyproj import Geod, Transformer
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
import shapely

GEOD = Geod(ellps='WGS84')
PROJECT = Transformer.from_crs(4326, 3035, always_xy=True)
CSV800 = ROOT / 'data_processed/lucas_cog_exact_pilot.csv'
CACHE = ROOT / 'revision/inputs/legacy_800_cog_metadata.json'
RAW = None
OUT = None


def radii(row):
    v = [min(float(row[k]), 50.) for k in ('n_m','e_m','s_m','w_m')]
    if min(v) < 0 or not all(math.isfinite(x) for x in v):
        raise ValueError(f'invalid distances {row["point_id"]}: {v}')
    return v


def quad(v, x, y):
    n, e, s, w = v
    p = Polygon([(x-w,y),(x,y+n),(x+e,y),(x,y-s)])
    return p if p.is_valid else p.buffer(0)


def radial(v, lon, lat, x, y, segments, smooth=True):
    # Geodesic bearings centred at N,E,S,W; sector edges at NE,SE,SW,NW.
    # Official uses geography buffer with 50 segments/quadrant and a 0.1 m
    # geography mitre buffer after union; ours approximates those in EPSG:3035.
    pieces = []
    for center, radius in zip((0, 90, 180, 270), v):
        if radius == 0:
            continue
        bearings = np.linspace(center-45, center+45, segments+1)
        n = len(bearings)
        lons, lats, _ = GEOD.fwd(np.full(n,lon), np.full(n,lat), bearings, np.full(n,radius))
        xs, ys = PROJECT.transform(lons,lats)
        pieces.append(Polygon([(x,y), *zip(xs,ys), (x,y)]))
    if not pieces:
        return Polygon()
    g = unary_union(pieces)
    if not g.is_valid:
        g = g.buffer(0)
    if smooth:
        g = g.buffer(.1, join_style='mitre', mitre_limit=1.)
    return g


def pixel(row, rec):
    # WorldCover 3 x 3 degree tile has 36000 x 36000 pixels. Reconstructed
    # corners checked pointwise against original cached pixel area and overlay.
    tile = rec['tile']
    lat0 = int(tile[1:3]) * (1 if tile[0]=='N' else -1)
    lon0 = int(tile[4:7]) * (1 if tile[3]=='E' else -1)
    ri, ci = int(rec['row_index']), int(rec['col_index'])
    step = 3/36000
    left, top = lon0+ci*step, lat0+3-ri*step
    corners = [(left,top),(left+step,top),(left+step,top-step),(left,top-step)]
    ring = []
    for a,b in zip(corners,corners[1:]+corners[:1]):
        for t in range(16):
            f=t/16
            ring.append(PROJECT.transform(a[0]+f*(b[0]-a[0]),a[1]+f*(b[1]-a[1])))
    p=Polygon(ring)
    if not p.is_valid: raise ValueError('invalid pixel')
    return p


def fraction(g,p):
    return min(1.,max(0.,g.intersection(p).area/p.area))


def describe(a):
    a=np.asarray(a,dtype=float)
    return {'n':len(a),'mean':float(np.mean(a)),'median':float(np.median(a)),
            'p10':float(np.quantile(a,.1)),'p25':float(np.quantile(a,.25)),
            'p75':float(np.quantile(a,.75)),'p90':float(np.quantile(a,.9)),
            'min':float(a.min()),'max':float(a.max())}


def summary(rows):
    result={'n':len(rows)}
    for k in ('quad_area','rad_area','quad_r','rad_r','delta_r','hard','quad_p','rad_p','quad_c','rad_c','quad_span','rad_span'):
        result[k]=describe([r[k] for r in rows])
    result['counts']={
        'quad_r_below_025':sum(r['quad_r']<.25 for r in rows),
        'rad_r_below_025':sum(r['rad_r']<.25 for r in rows),
        'quad_r_ge_075':sum(r['quad_r']>=.75 for r in rows),
        'rad_r_ge_075':sum(r['rad_r']>=.75 for r in rows),
        'quad_r_ge_095':sum(r['quad_r']>=.95 for r in rows),
        'rad_r_ge_095':sum(r['rad_r']>=.95 for r in rows),
        'rad_r_greater_1e-6':sum(r['delta_r']>1e-6 for r in rows),
        'rad_r_lower_1e-6':sum(r['delta_r']< -1e-6 for r in rows),
        'quad_zero_area':sum(r['quad_area']<1e-9 for r in rows),
        'rad_zero_area':sum(r['rad_area']<1e-9 for r in rows)}
    result['strata']={}
    for label,lo,hi in [('0-.25',0,.25),('.25-.5',.25,.5),('.5-.75',.5,.75),('.75-1',.75,1.00000001)]:
        result['strata'][label]={shape:sum(lo<=r[shape+'_r']<hi for r in rows) for shape in ('quad','rad')}
    return result


def save_json(name, obj):
    with (OUT/name).open('x',encoding='utf-8') as f:
        f.write(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False))


def save_csv(name, rows):
    with (OUT/name).open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def run800():
    with CSV800.open(encoding='utf-8',newline='') as f: inp=list(csv.DictReader(f))
    cache=json.loads(CACHE.read_text(encoding='utf-8'))
    lookup={(str(v.get('year')),str(v.get('point_id')),str(v.get('tile'))):v for v in cache.values() if isinstance(v,dict)}
    if len(inp)!=800 or len({r['point_id'] for r in inp})!=800: raise ValueError('Unexpected sample size/ID duplicates')
    rows=[]; tests=[]
    for j,row in enumerate(inp):
        rec=lookup[('2021',row['point_id'],row['tile'])]
        if int(rec['raw_value'])!=int(row['raw_2021']): raise ValueError('cached label mismatch')
        v=radii(row); lon=float(row['lon']);lat=float(row['lat']); x,y=PROJECT.transform(lon,lat)
        p=pixel(row,rec); q=quad(v,x,y); qr=fraction(q,p)
        err=abs(qr-float(row['proxy_pixel_fraction']))
        areaerr=abs(p.area-float(row['pixel_area_m2_2021']))
        if err>2e-6 or areaerr>1e-4: raise ValueError(f'pixel reconstruction failed {row["point_id"]}: {err} {areaerr}')
        g=radial(v,lon,lat,x,y,100); rr=fraction(g,p)
        hard=int(int(row['raw_2021'])==int(row['target_wc']))
        out={'point_id':row['point_id'],'country':row['country'],'broad_class':row['broad_class'],
             'quad_area':q.area,'rad_area':g.area,'quad_r':qr,'rad_r':rr,'delta_r':rr-qr,
             'hard':hard,'quad_p':qr*hard,'rad_p':rr*hard,
             'quad_c':qr*hard+1-qr,'rad_c':rr*hard+1-rr,'quad_span':1-qr,'rad_span':1-rr,
             'pixel_area_error_m2':areaerr,'quad_r_original_abs_error':err}
        rows.append(out)
        for seg in (25,50,200):
            gg=radial(v,lon,lat,x,y,seg)
            tests.append({'point_id':row['point_id'],'segments_per_quarter':seg,
                          'area_abs_difference_vs_100_m2':abs(gg.area-g.area),
                          'r_abs_difference_vs_100':abs(fraction(gg,p)-rr)})
        if (j+1)%200==0: print(f'800 COG: {j+1}/800',flush=True)
    save_csv('paired_800.csv',rows);save_csv('arc_convergence_800.csv',tests)
    result=summary(rows)
    result['by_class']={c:summary([r for r in rows if r['broad_class']==c]) for c in 'ABCDEFGH'}
    result['pixel_reconstruction_max_abs_r_error']=max(r['quad_r_original_abs_error'] for r in rows)
    result['pixel_reconstruction_max_abs_area_error_m2']=max(r['pixel_area_error_m2'] for r in rows)
    result['convergence']={str(s):{k:describe([t[k] for t in tests if t['segments_per_quarter']==s]) for k in ('area_abs_difference_vs_100_m2','r_abs_difference_vs_100')} for s in (25,50,200)}
    result['method']='theoretical centre, equal area projected overlay; geodesic quarter arcs and projected 0.1m mitre buffer approximate official geography pipeline; not released geometry'
    save_json('summary_800.json',result)
    print(json.dumps({'summary_800':{k:result[k] for k in ('n','quad_r','rad_r','delta_r','hard','quad_p','rad_p','quad_c','rad_c','quad_span','rad_span','counts','convergence','pixel_reconstruction_max_abs_r_error')}},ensure_ascii=False),flush=True)


def run_full():
    rows=common.read_rows(RAW)
    print(f'Full eligible input: {len(rows)}',flush=True)
    if len(rows)!=128082: raise ValueError('full input filter count differs from manuscript; inspect first')
    result=[]
    for i,row in enumerate(rows):
        v=radii(row); lon=float(row['lon']);lat=float(row['lat']);x,y=PROJECT.transform(lon,lat)
        square=box(x-5,y-5,x+5,y+5)
        q=quad(v,x,y);g=radial(v,lon,lat,x,y,50)
        qr=fraction(q,square);rr=fraction(g,square)
        result.append({'point_id':row['point_id'],'country':row['country'],'broad_class':row['broad_class'],
                       'quad_area':q.area,'rad_area':g.area,'quad_r':qr,'rad_r':rr,'delta_r':rr-qr})
        if (i+1)%10000==0: print(f'Full nominal: {i+1}/{len(rows)}',flush=True)
    save_csv('paired_full_nominal.csv',result)
    def full_summary(sub):
        res={k:describe([r[k] for r in sub]) for k in ('quad_area','rad_area','quad_r','rad_r','delta_r')}
        res['strata']={lab:{shape:sum(lo<=r[shape+'_r']<hi for r in sub) for shape in ('quad','rad')} for lab,lo,hi in [('0-.25',0,.25),('.25-.5',.25,.5),('.5-.75',.5,.75),('.75-1',.75,1.00000001)]}
        res['at_least_095']={shape:sum(r[shape+'_r']>=.95 for r in sub) for shape in ('quad','rad')}
        return res
    info={'overall':full_summary(result),'by_class':{c:full_summary([r for r in result if r['broad_class']==c]) for c in 'ABCDEFGH'},
          'segments_per_quarter':50,'note':'nominal EPSG:3035 square, not actual WorldCover pixel; same theoretical point; approximate official radial geometry'}
    save_json('summary_full_nominal.json',info)
    print(json.dumps(info['overall'],ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--full',action='store_true');p.add_argument('--only-full',action='store_true');p.add_argument('--raw',type=Path,help='required for unverified legacy full-frame mode');p.add_argument('--cache',type=Path,default=CACHE);p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
    OUT=args.output_dir.resolve(); CACHE=args.cache.resolve(); RAW=args.raw.resolve() if args.raw else None
    if not OUT.is_relative_to((ROOT/'runs').resolve()): p.error('--output-dir must be inside this package runs/')
    if (args.full or args.only_full) and RAW is None: p.error('--raw is required for legacy full-frame mode')
    OUT.mkdir(parents=True,exist_ok=True)
    output_names=[]
    if not args.only_full: output_names += ['paired_800.csv','arc_convergence_800.csv','summary_800.json']
    if args.full or args.only_full: output_names += ['paired_full_nominal.csv','summary_full_nominal.json']
    if any((OUT/name).exists() for name in output_names): raise FileExistsError('Refusing to overwrite an existing output')
    if not args.only_full: run800()
    if args.full or args.only_full: run_full()
