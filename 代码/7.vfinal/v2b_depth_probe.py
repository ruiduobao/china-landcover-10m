# -*- coding: utf-8 -*-
"""depth_probe.py — 树深/叶大小杠杆（全深度树→小叶→稀有类票被淹没？）"""
import os, sys, json, time
import numpy as np, pandas as pd
PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
sys.path.insert(0, os.path.join(PROJ, '代码', '4.全国清洗训练'))
import e6_spatial_eval as E6
from sklearn.ensemble import RandomForestClassifier
WEAK = [52, 62, 91, 140, 11]
col = ['row_id', 'lon', 'lat', 'class_new', 'train_weight'] + E6.FEATS
df = pd.read_parquet(os.path.join(E6.SUB_DIR, 'r7_train_2022.parquet'), columns=col)
hold = np.asarray(json.load(open(E6.HOLDOUT))['block_keys'], dtype=np.int64)
df = df[~np.isin(E6.albers_bk(df.lon.to_numpy(), df.lat.to_numpy()), hold)].reset_index(drop=True)
X = df[E6.FEATS].to_numpy(np.float32); y = df.class_new.to_numpy(int)
w = np.nan_to_num(df.train_weight.to_numpy(np.float32), nan=1.0, posinf=1.0, neginf=1.0)
val = E6.load_val_ids().merge(pd.read_parquet(os.path.join(E6.SUB_DIR, 'r7_train_2022.parquet'),
                             columns=['row_id'] + E6.FEATS), on='row_id', how='inner')
VX = val[E6.FEATS].to_numpy(np.float32); ct = val.class_new.to_numpy(int)
lab0 = sorted(set(ct))
print(f'训练 {len(df):,} | 验证 {len(val):,}', flush=True)
print(f"\n{'depth/leaf':<14}{'OA':>8}{'macroF1':>9}{'balAcc':>8}{'非零':>7}   " + '  '.join(f'{c}(pred,F1)' for c in WEAK))
out = []
for depth, leaf in [(None,2),(25,2),(15,2),(15,10),(10,20),(6,50)]:
    t0=time.time()
    rf = RandomForestClassifier(n_estimators=150, n_jobs=13, random_state=42, max_depth=depth,
                                min_samples_leaf=leaf, max_features='sqrt',
                                class_weight='balanced_subsample')
    rf.fit(X, y, sample_weight=w)
    P = rf.predict(VX)
    labs = sorted(set(lab0) | set(np.unique(P)))
    ix = {c:i for i,c in enumerate(labs)}
    cm = np.zeros((len(labs),len(labs)), np.int64)
    for t_,p_ in zip(ct,P): cm[ix[t_], ix[p_]] += 1
    met, per = E6.metrics_from_cm(cm, labs)
    row={'depth':str(depth),'leaf':leaf,'OA':round(met['OA'],4),'macroF1':round(met['macro_F1'],4),
         'bal':round(met['balanced_accuracy'],4),'nz':int((per.n_pred>0).sum())}
    desc=[]
    for c in WEAK:
        r=per[per['class']==c]; row[f'c{c}']=(int(r.n_pred.iloc[0]) if len(r) else -1,
                                              round(float(r.F1.iloc[0]),3) if len(r) else -1)
        desc.append(f"{row[f'c{c}'][0]:>5},{row[f'c{c}'][1]:.2f}")
    print(f"{str(depth)+'/'+str(leaf):<14}{row['OA']:>8.4f}{row['macroF1']:>9.4f}{row['bal']:>8.4f}"
          f"{row['nz']:>5}/30  " + '  '.join(desc), flush=True)
    out.append(row)
json.dump(out, open(r'F:/lc_work/depth_probe.json','w',encoding='utf-8'), ensure_ascii=False, indent=1)
