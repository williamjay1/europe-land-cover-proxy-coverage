"""Create compact plotting data and old/new frame comparison without overwrite."""
from pathlib import Path
import argparse, sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser()
parser.add_argument('--raw',type=Path,required=True)
parser.add_argument('--points',type=Path,required=True,help='points.csv produced with --write-point-records')
parser.add_argument('--output-dir',type=Path,required=True)
parser.add_argument('--write-plot-points',action='store_true',help='Also write the large point-level plot table')
args=parser.parse_args()
RAW=args.raw.resolve()
HERE=args.output_dir.resolve()
if not HERE.is_relative_to((ROOT/'runs').resolve()): parser.error('--output-dir must be inside this package runs/')
HERE.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'scripts'))
import run_lucas_worldcover_pilot as original

src=args.points.resolve()
names=['class','frame_comparison','frame_class_comparison']
if args.write_plot_points: names.append('points')
paths={k:HERE/('r3_corrected_plot_'+k+'.csv') for k in names}
if any(p.exists() for p in paths.values()): raise FileExistsError('Refusing to overwrite existing results')
d=pd.read_csv(src,dtype={'point_id':str,'country':str,'broad_class':str})
old=set(str(r['point_id']) for r in original.read_rows(RAW))
assert len(d)==136605 and d.point_id.is_unique and len(old)==128082
d['frame_group']=np.where(d.point_id.isin(old),'old_128082','added_8523')
assert d.frame_group.value_counts().to_dict()=={'old_128082':128082,'added_8523':8523}
distcols=[f'{k}_raw_m' for k in 'nesw']
assert (d.loc[d.frame_group.eq('added_8523'),distcols]==8).any(axis=1).all()
assert not (d.loc[d.frame_group.eq('old_128082'),distcols]==8).any(axis=1).any()
if args.write_plot_points:
    d[['point_id','country','broad_class','frame_group','nominal_10m','nominal_20m','nominal_30m']].to_csv(paths['points'],index=False,mode='x',lineterminator='\n')

def stats(part):
    out={'n':len(part)}
    for size in (10,20,30):
        x=part[f'nominal_{size}m']
        out.update({f'mean_{size}m':x.mean(),f'median_{size}m':x.median(),
                    f'p10_{size}m':x.quantile(.1),f'p90_{size}m':x.quantile(.9),
                    f'below_025_n_{size}m':int((x<.25).sum()),
                    f'ge_095_n_{size}m':int((x>=.95).sum())})
    return out

groups=[{'broad_class':c,**stats(p)} for c,p in d.groupby('broad_class',sort=True)]
pd.DataFrame(groups).to_csv(paths['class'],index=False,mode='x',lineterminator='\n')
groups=[{'frame_group':c,**stats(p)} for c,p in d.groupby('frame_group',sort=True)]
pd.DataFrame(groups).to_csv(paths['frame_comparison'],index=False,mode='x',lineterminator='\n')
groups=[{'frame_group':g,'broad_class':c,**stats(p)} for (g,c),p in d.groupby(['frame_group','broad_class'],sort=True)]
pd.DataFrame(groups).to_csv(paths['frame_class_comparison'],index=False,mode='x',lineterminator='\n')
print(pd.read_csv(paths['frame_comparison']).to_string(index=False))
print(pd.read_csv(paths['class']).to_string(index=False))
