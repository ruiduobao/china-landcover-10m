# -*- coding: utf-8 -*-
"""g7_screen.py — 用判别器筛查全池（521k 灌丛层点 + 其余 154 万点），扩增四省灌丛样本

* 输入：F:/lc_work/prod5p_2023/data/train2023_clean.parquet（含 AEF64/src/class_new）
        data/shrub/judged_all.csv（78 确认 + 328 否定 → 训练判别器）
* 规则源（预注册）：
  S-A 「灌丛层内筛查」：对 src='glc_fcs10_2023_shrub' 的 521,400 点打分 → 该层内 top-k（提纯该层）
  S-B 「全池筛查」：对非灌丛层点打分 → 发现被其他类标签覆盖的灌丛（co-training）
  取点门槛：p_shrub ≥ 0.85；每省每源 ≤200；省内空间间距 ≥300 m；须落在四省 bbox 内
* 输出：data/shrub/screened.csv（point_id,prov,variant,lon,lat,p_shrub,src,class_new）
* 用法：python g7_screen.py
"""
import collections, os, sys, time
sys.path.insert(0, r'F:/lc_work/v31_exp/code')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
import v31_common as VC
from sklearn.ensemble import RandomForestClassifier

W = r'F:/lc_work/v31_exp'
OUTD = os.path.join(W, 'data', 'shrub')
POOL = r'F:/lc_work/prod5p_2023/data/train2023_clean.parquet'
SHRUB_SRC = 'glc_fcs10_2023_shrub'
PROV = {'宁夏': (104.0, 35.0, 108.0, 40.0), '四川': (97.0, 26.0, 109.0, 34.5),
        '黑龙江': (121.0, 43.0, 135.5, 53.5), '福建': (115.5, 23.0, 120.5, 28.5)}
SEED = 7
P_MIN = 0.85
MAX_PER = 200
MIN_KM_M = 300.0

def emit(m): print(time.strftime('[%m-%d %H:%M:%S] ')+str(m), flush=True)

def thin(df, n_max, min_m=MIN_KM_M):
    df = df.sort_values('p_shrub', ascending=False)
    used, out = [], []
    for r in df.itertuples():
        if len(out) >= n_max: break
        if any(((r.lon - u) ** 2 * 100 + (r.lat - v) ** 2 * 120) < (min_m / 111000.0) ** 2 for u, v in used):
            continue
        used.append((r.lon, r.lat)); out.append(r.Index)
    return df.loc[out]

def main():
    j = pd.read_csv(os.path.join(OUTD, 'judged_all.csv'), encoding='utf-8-sig')
    cand = pd.read_csv(os.path.join(OUTD, 'cand_all.csv'), encoding='utf-8-sig')
    miss = [c for c in VC.FEATS if c not in j.columns]
    if miss:
        j = j.merge(cand[['point_id'] + miss], on='point_id', how='left')
    tr = j[j.Q1_shrub.isin(['是', '否'])].copy()
    y = (tr.Q1_shrub == '是').astype(int).to_numpy()
    clf = RandomForestClassifier(n_estimators=300, class_weight='balanced_subsample',
                                 n_jobs=15, random_state=SEED).fit(
        np.nan_to_num(tr[VC.FEATS].to_numpy('float32'), nan=-999), y)
    emit('判别器训练完成（正 %d / 负 %d）' % (int(y.sum()), int((1 - y).sum())))

    pool = pd.read_parquet(POOL, columns=['lon', 'lat', 'class_new', 'src'] + VC.FEATS)
    emit('池 %d 点' % len(pool))
    pool['in_prov'] = ''
    for p, (x0, y0, x1, y1) in PROV.items():
        m = (pool.lon >= x0) & (pool.lon <= x1) & (pool.lat >= y0) & (pool.lat <= y1)
        pool.loc[m, 'in_prov'] = p
    sub = pool[pool.in_prov != ''].copy()
    emit('四省池点 %d' % len(sub))
    X = np.nan_to_num(sub[VC.FEATS].to_numpy('float32'), nan=-999)
    sub['p_shrub'] = clf.predict_proba(X)[:, 1]
    rows = []
    for p in PROV:
        for variant, mask in (('A_灌丛层内', sub.src.fillna('').astype(str) == SHRUB_SRC),
                              ('B_全池发现', sub.src.fillna('').astype(str) != SHRUB_SRC)):
            g = sub[(sub.in_prov == p) & mask & (sub.p_shrub >= P_MIN)]
            k = thin(g, MAX_PER)
            emit('  %s %s：p≥%.2f 命中 %d → 取 %d' % (p, variant, P_MIN, len(g), len(k)))
            k = k.assign(prov=p, variant=variant)
            rows.append(k[['prov', 'variant', 'lon', 'lat', 'p_shrub', 'src', 'class_new']])
    out = pd.concat(rows, ignore_index=True)
    out.insert(0, 'point_id', ['SCR-%s-%s-%04d' % (r.prov, r.variant[:1], i)
                               for i, r in enumerate(out.itertuples())])
    out.to_csv(os.path.join(OUTD, 'screened.csv'), index=False, encoding='utf-8-sig')
    emit('筛查结果 %d → screened.csv' % len(out))
    emit('  %s' % {('%s|%s' % k): int(v) for k, v in out.groupby(['prov', 'variant']).size().items()})

if __name__ == '__main__': main()
