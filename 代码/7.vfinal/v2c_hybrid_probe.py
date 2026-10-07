# -*- coding: utf-8 -*-
"""hybrid_probe.py — 混合方案：主模型保 OA + 浅树模型只供 5 个弱类（限制在合理宿主类内）"""
import os, sys, json, time
import numpy as np, pandas as pd
PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
sys.path.insert(0, os.path.join(PROJ, '代码', '4.全国清洗训练'))
import e6_spatial_eval as E6
from sklearn.ensemble import RandomForestClassifier
WEAK=[52,62,91,140,11]
HOST={52:[51,61,71,72,120,121,62,92],62:[61,51,71,82,121,52],91:[71,61,92,82,51,72],
      140:[150,201,121,130,120,220],11:[10,12,120,121,130,71]}
col=['row_id','lon','lat','class_new','train_weight']+E6.FEATS
df=pd.read_parquet(os.path.join(E6.SUB_DIR,'r7_train_2022.parquet'),columns=col)
hold=np.asarray(json.load(open(E6.HOLDOUT))['block_keys'],dtype=np.int64)
df=df[~np.isin(E6.albers_bk(df.lon.to_numpy(),df.lat.to_numpy()),hold)].reset_index(drop=True)
X=df[E6.FEATS].to_numpy(np.float32); y=df.class_new.to_numpy(int)
w=np.nan_to_num(df.train_weight.to_numpy(np.float32),nan=1.0,posinf=1.0,neginf=1.0)
val=E6.load_val_ids().merge(pd.read_parquet(os.path.join(E6.SUB_DIR,'r7_train_2022.parquet'),
     columns=['row_id']+E6.FEATS),on='row_id',how='inner')
VX=val[E6.FEATS].to_numpy(np.float32); ct=val.class_new.to_numpy(int)
def fit(depth,seed=42):
    rf=RandomForestClassifier(n_estimators=150,n_jobs=13,random_state=seed,max_depth=depth,
        min_samples_leaf=2,max_features='sqrt',class_weight='balanced_subsample')
    rf.fit(X,y,sample_weight=w); return rf
t0=time.time(); MD=None
gens={}
for d in [None,20,15]:
    gens[d]=fit(d); print(f'depth={d} 训练完成 {time.time()-t0:.0f}s',flush=True)
main=gens[None].predict(VX)
lab0=sorted(set(ct)|set(main))
def score(pred,tag):
    labs=sorted(set(lab0)|set(np.unique(pred))); ix={c:i for i,c in enumerate(labs)}
    cm=np.zeros((len(labs),len(labs)),np.int64)
    for a,b in zip(ct,pred): cm[ix[a],ix[b]]+=1
    met,per=E6.metrics_from_cm(cm,labs)
    row={'tag':tag,'OA':round(met['OA'],4),'macroF1':round(met['macro_F1'],4),
         'bal':round(met['balanced_accuracy'],4),'nz':int((per.n_pred>0).sum())}
    for c in WEAK:
        r=per[per['class']==c]
        row[f'c{c}']=(int(r.n_pred.iloc[0]) if len(r) else -1, round(float(r.F1.iloc[0]),3) if len(r) else -1)
    return row
rows=[score(main,'main_depthNone')]
print(f"\n{'方案':<22}{'OA':>8}{'macroF1':>9}{'balAcc':>8}{'非零':>8}   "+'  '.join(f'{c}(pred,F1)' for c in WEAK))
for d in [20,15]:
    wp=gens[d].predict(VX)
    fin=main.copy()
    for c in WEAK:
        sel=(wp==c)&np.isin(main,HOST[c])
        fin[sel]=c
    rows.append(score(fin,f'hybrid_d{d}'))
for r in rows:
    desc='  '.join(f"{r[f'c{c}'][0]:>5},{r[f'c{c}'][1]:.2f}" for c in WEAK)
    print(f"{r['tag']:<22}{r['OA']:>8.4f}{r['macroF1']:>9.4f}{r['bal']:>8.4f}{r['nz']:>6}/30  {desc}",flush=True)
json.dump(rows,open(r'F:/lc_work/hybrid_probe.json','w',encoding='utf-8'),ensure_ascii=False,indent=1)
