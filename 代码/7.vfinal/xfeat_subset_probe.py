# -*- coding: utf-8 -*-
"""xfeat_subset.py — 只给森林族点提特征（验证 52/62 的郁闭度假设，先验证再开跑）"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, r'F:/lc_work')
import xfeat as X

SUB = r'Z:/地理所/论文/中国土地覆盖数据_2017-2024/数据/本地处理/全国清洗训练/年度子集_含稀有类/r7_train_2022.parquet'
VAL = r'Z:/地理所/论文/中国土地覆盖数据_2017-2024/数据/本地处理/全国清洗训练/评估_P3.4/val_predictions_2022.parquet'
FAM = [51, 52, 61, 62, 71, 72, 81, 82, 91, 92]
CAP = 360_000

df = pd.read_parquet(SUB, columns=['row_id', 'lon', 'lat', 'class_new'])
df = df[df.class_new.isin(FAM)]
if len(df) > CAP:
    df = df.groupby('class_new', group_keys=False).apply(
        lambda g: g.sample(min(len(g), max(int(CAP * len(g) / len(df)), 1500)), random_state=9))
print('森林族抽样', len(df), df.class_new.value_counts().to_dict(), flush=True)
v = pd.read_parquet(VAL, columns=['row_id', 'lon', 'lat', 'class_new'])
pts = pd.concat([df, v[['row_id', 'lon', 'lat']]], ignore_index=True).drop_duplicates('row_id')
pts = pts.sort_values('row_id').reset_index(drop=True)
print('总点', len(pts), flush=True)

ee = X.bring_up('seqsiu', 'electric-orbit-508407-g4')
st = X.build_stack(ee, with_gedi=False)
os.makedirs(X.XDIR, exist_ok=True)
# 按 2° 区地理分组后再切批：全国范围的点会让 GEE 在整幅影像上算合成 → 超时
pts['zx'] = (pts.lon // 2 * 2).astype(int); pts['zy'] = (pts.lat // 2 * 2).astype(int)
pts = pts.sort_values(['zx', 'zy', 'row_id']).reset_index(drop=True)
outs = []; nb = 0
for (zx, zy), g in pts.groupby(['zx', 'zy'], sort=True):
    for s in range(0, len(g), 6000):
        sub = g.iloc[s:s + 6000]; nb += 1
        try:
            outs.append(X.sample_pts(ee, st, sub, 'fam', scale=30, tile=4))
            print(f'  批 {nb} 区({zx},{zy}): {len(outs[-1])}/{len(sub)}', flush=True)
        except Exception as e:
            print(f'  批 {nb} 区({zx},{zy}) 失败 {str(e)[:70]}', flush=True)
out = pd.concat(outs, ignore_index=True) if outs else pd.DataFrame()
fp = r'F:/lc_work/xfeat/xfeat_fam.parquet'
out.to_parquet(fp, index=False)
print('保存', len(out), '→', fp, flush=True)
print(out[X.FEAT_COLS].describe().T[['count','mean','min','max']].to_string(), flush=True)
