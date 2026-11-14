"""Read paired WorldCover pixels for corrected sample; task-owned checkpoint only."""
import argparse, csv, json, math, os, sys, time
from collections import defaultdict,Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'vendor'),str(ROOT/'scripts'),str(HERE.parent),str(HERE)]
import numpy as np
import radial_sensitivity as rad
import validate_worldcover_cog_probe as cog
import audit_sample as audit
import rasterio
from pyproj import Geod

OUT=None
GEOD=Geod(ellps='WGS84')

def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--limit-tiles',type=int,default=0);parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--raw',type=Path,required=True);parser.add_argument('--sample',type=Path,default=ROOT/'revision/results/new_sample/corrected_800.csv')
    parser.add_argument('--cache',type=Path,default=rad.CACHE);parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    output_dir=args.output_dir.resolve()
    if not output_dir.is_relative_to((ROOT/'runs').resolve()): parser.error('--output-dir must be inside this package runs/')
    output_dir.mkdir(parents=True,exist_ok=True)
    names=('cog_checkpoint.json','cog_conditions.csv','cog_theoretical_gps_paired.csv','cog_status.json')
    if any((output_dir/name).exists() for name in names): raise FileExistsError('Refusing to overwrite an existing run')
    audit.RAW=args.raw.resolve();rad.CACHE=args.cache.resolve();OUT=output_dir/'cog_checkpoint.json'
    start=time.monotonic()
    with args.sample.open(encoding='utf-8',newline='') as f: sample=list(csv.DictReader(f))
    byid={r['point_id']:r for r in sample}; gps={}
    with audit.RAW.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if r['POINT_ID'] in byid:
                try:
                    lon=float(r['SURVEY_GPS_LONG']);lat=float(r['SURVEY_GPS_LAT'])
                    if not (-30<=lon<=45 and 25<=lat<=75) or r['SURVEY_GPS_PROJ']!='1': continue
                    gps[r['POINT_ID']]={'lat':lat,'lon':lon,'gps_proj':r['SURVEY_GPS_PROJ']}
                except (TypeError,ValueError): pass
    tasks={}
    for r in sample:
        for condition in ('theoretical','gps'):
            if condition=='gps' and r['point_id'] not in gps: continue
            pos={'lat':float(r['lat']),'lon':float(r['lon'])} if condition=='theoretical' else gps[r['point_id']]
            tile=cog.tile_name(pos['lat'],pos['lon']);key=r['point_id']+'|'+condition
            tasks[key]={'key':key,'condition':condition,'row':r,**pos,'tile':tile}
    old=json.loads(rad.CACHE.read_text(encoding='utf-8'))
    cache={(str(v.get('year')),str(v.get('point_id')),str(v.get('tile'))):v for v in old.values() if isinstance(v,dict)}
    done=json.loads(OUT.read_text(encoding='utf-8')) if OUT.exists() else {}
    failures=[]
    def metrics(task,raw,pixel,source):
        r=task['row'];lon=task['lon'];lat=task['lat'];x,y=rad.PROJECT.transform(lon,lat)
        v=rad.radii(r);q=rad.quad(v,x,y);g=rad.radial(v,lon,lat,x,y,100)
        qr=rad.fraction(q,pixel);rr=rad.fraction(g,pixel);match=int(raw==int(r['target_wc']))
        return {'point_id':r['point_id'],'condition':task['condition'],'broad_class':r['broad_class'],'country':r['country'],
                'lon':lon,'lat':lat,'tile':task['tile'],'raw_2021':raw,'target_wc':int(r['target_wc']),
                'hard_match':match,'pixel_area_m2':pixel.area,'quad_area_m2':q.area,'rad_area_m2':g.area,
                'quad_r':qr,'rad_r':rr,'quad_p':qr*match,'rad_p':rr*match,
                'quad_c':qr*match+1-qr,'rad_c':rr*match+1-rr,'source':source}
    # Only exact theoretical-point records are reused. GPS always uses its own position.
    for key,t in tasks.items():
        if key in done or t['condition']!='theoretical': continue
        rec=cache.get(('2021',t['row']['point_id'],t['tile']))
        if rec and 'row_index' in rec and 'raw_value' in rec:
            p=rad.pixel(t['row'],rec)
            value=metrics(t,int(rec['raw_value']),p,'existing_cache')
            if 'support_fraction' in rec and abs(value['quad_r']-float(rec['support_fraction']))>2e-6:
                raise ValueError('cached overlay mismatch '+key)
            done[key]=value
    groups=defaultdict(list)
    for key,t in tasks.items():
        if key not in done: groups[t['tile']].append(t)
    print(json.dumps({'sample':len(sample),'gps_valid_wgs84':len(gps),'total_conditions':len(tasks),'reused_checkpoint_and_cache':len(done),'remaining_conditions':sum(map(len,groups.values())),'remaining_tiles':len(groups),'workers':args.workers}),flush=True)
    tilelist=sorted(groups)
    if args.limit_tiles: tilelist=tilelist[:args.limit_tiles]
    def process(tile):
        t0=time.monotonic();res=[];errors=[]
        try:
            url=cog.tile_url('2021',tile)
            with rasterio.Env(GDAL_HTTP_TIMEOUT='20',GDAL_HTTP_CONNECTTIMEOUT='10',GDAL_HTTP_MAX_RETRY='1',GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR'):
                with rasterio.open(url) as src:
                    if src.crs.to_epsg()!=4326: raise ValueError('COG CRS differs from EPSG:4326')
                    for task in groups[tile]:
                        try:
                            row,col=src.index(task['lon'],task['lat'])
                            a=src.read(1,window=((row,row+1),(col,col+1)))
                            if a.size!=1: raise ValueError('empty raster cell')
                            pixel,_=cog.pixel_geometry(src,row,col,rad.PROJECT)
                            item=metrics(task,int(a[0,0]),pixel,'remote_cog')
                            item['row_index']=row;item['col_index']=col
                            res.append((task['key'],item))
                        except Exception as e: errors.append({'key':task['key'],'error':str(e)})
        except Exception as e:
            errors += [{'key':task['key'],'error':str(e)} for task in groups[tile]]
        return tile,res,errors,time.monotonic()-t0
    run_start=time.monotonic();tile_times=[]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for tile,results,errs,secs in ex.map(process,tilelist):
            done.update(results);failures+=errs;tile_times.append(secs)
            OUT.write_text(json.dumps(done,ensure_ascii=False),encoding='utf-8')
            print(f'tile {tile}: success={len(results)} failed={len(errs)} tile_s={secs:.1f} total_done={len(done)}',flush=True)
    with (output_dir/'cog_conditions.csv').open('x',encoding='utf-8',newline='') as f:
        columns=['point_id','condition','broad_class','country','lon','lat','tile','row_index','col_index','raw_2021','target_wc','hard_match','pixel_area_m2','quad_area_m2','rad_area_m2','quad_r','rad_r','quad_p','rad_p','quad_c','rad_c','source']
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(done.values())
    paired=[]
    for r in sample:
        a=done.get(r['point_id']+'|theoretical');b=done.get(r['point_id']+'|gps')
        if not a or not b: continue
        _,_,distance=GEOD.inv(a['lon'],a['lat'],b['lon'],b['lat'])
        row={'point_id':r['point_id'],'broad_class':r['broad_class'],'country':r['country'],'gps_offset_m':distance}
        for m in ('raw_2021','hard_match','quad_r','rad_r','quad_p','rad_p','quad_c','rad_c'):
            row['theoretical_'+m]=a[m];row['gps_'+m]=b[m]
        paired.append(row)
    if paired:
        with (output_dir/'cog_theoretical_gps_paired.csv').open('x',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
    status={'gps_valid_wgs84_n':len(gps),'condition_count':len(tasks),'completed_conditions':len(done),
            'completed_theoretical':sum(k.endswith('|theoretical') for k in done),'completed_gps':sum(k.endswith('|gps') for k in done),
            'paired_n':len(paired),'failures':failures,'tiles_attempted_this_run':len(tilelist),
            'tile_seconds_sum':sum(tile_times),'wall_seconds_fetch':time.monotonic()-run_start,
            'wall_seconds_total':time.monotonic()-start,'workers':args.workers}
    for cond in ('theoretical','gps'):
        rows=[x for x in done.values() if x['condition']==cond]
        if rows:
            status[cond+'_metrics']={m:float(np.mean([r[m] for r in rows])) for m in ('hard_match','quad_r','rad_r','quad_p','rad_p','quad_c','rad_c')}
            status[cond+'_metrics']['n']=len(rows)
            status[cond+'_metrics']['quad_r_below_025']=sum(r['quad_r']<.25 for r in rows)
    with (output_dir/'cog_status.json').open('x',encoding='utf-8') as f: f.write(json.dumps(status,indent=2,ensure_ascii=False))
    print(json.dumps(status,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
