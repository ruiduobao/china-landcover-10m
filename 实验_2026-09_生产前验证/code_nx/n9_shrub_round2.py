# -*- coding: utf-8 -*-
"""n9_shrub_round2.py — 第 2 轮小样本：S1 结构特征补测 + 四种标签策略的省域对比（预注册）

* 输入：gsnx_ref_judged.csv（303 点真值）+ gsnx_ref_embed.csv（AEF）+ train2023_clean.parquet（2023 池）
* 规则源（**预注册**）：
  A. S1 补测：GEE COPERNICUS/S1_GRD 2023 冬(DJF)/夏(JJA) VV/VH 中位（4 波段）→ 与 AEF+结构 一起做同一套 5 折 CV。
     判据同第 1 轮（AUC ≥0.85 且 F1 ≥0.55 才算"可分"）。
  B. 标签策略四臂（省域级模型，box=省域+2°；每臂抽 20,000 点；RF 100/leaf2/nodes5000/seed7）：
     R0=全池；R4=筛灌丛层（现状最优）；R6=**训练端合并**（灌丛层点标签改 11 草地，特征保留）；
     R7=灌丛层标签改 13 稀疏植被。
     评价：在 303 点参考集上 24 类 OA、9 大类 OA、灌草合并 OA、灌丛/草地 per-class（PA=召回 UA=精确）。
     判据：以 **9 大类 OA** 为主判据选臂；若 R6/R7 相对 R4 提升 ≥1.5pp 且灌草合并口径不降，判"合并可用"。
* 输出：results/d2/shrub_round2.json + S1 特征缓存 gsnx_ref_s1.csv
* 用法：python n9_shrub_round2.py
"""
import csv
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import v31_common as VC
from sklearn.ensemble import RandomForestClassifier

M3D = r'F:/lc_work/v31_exp/data/m3'
REF = os.path.join(M3D, 'gsnx_ref_judged.csv')
EMB = os.path.join(M3D, 'gsnx_ref_embed.csv')
S1F = os.path.join(M3D, 'gsnx_ref_s1.csv')
POOL = r'F:/lc_work/prod5p_2023/data/train2023_clean.parquet'
SHRUB_SRC = 'glc_fcs10_2023_shrub'
STRUCT = ['ch_mean', 'tc2000', 'ndvi_amp', 'ndvi_winter', 'ndvi_summer']
S1B = ['s1_vv_w', 's1_vh_w', 's1_vv_s', 's1_vh_s']
PROV_BOX = {'甘肃': (92.5, 32.5, 108.8, 43.0), '宁夏': (104.0, 35.0, 108.0, 40.0)}
N_TRAIN = 20000


def log(m):
    print('[%s] %s' % (time.strftime('%m-%d %H:%M:%S'), m), flush=True)


def get_s1(df, acct='zixen8v8'):
    if os.path.exists(S1F):
        s = pd.read_csv(S1F, encoding='utf-8-sig')
        if len(s) >= len(df):
            return s
    import ee
    VC.ensure_ctx(acct)
    pts = [ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]), {'point_id': r.point_id})
           for r in df.itertuples()]
    col = (ee.ImageCollection('COPERNICUS/S1_GRD')
           .filter(ee.Filter.eq('instrumentMode', 'IW'))
           .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
           .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
           .filterDate('2023-01-01', '2024-01-01').select(['VV', 'VH']))
    img = ee.Image.cat([
        col.filterDate('2023-01-01', '2023-03-01').median().rename(['s1_vv_w', 's1_vh_w']),
        col.filterDate('2023-06-01', '2023-09-01').median().rename(['s1_vv_s', 's1_vh_s']),
    ])
    got = img.sampleRegions(collection=ee.FeatureCollection(pts), scale=10, geometries=False,
                            tileScale=4).getInfo()['features']
    recs = [f['properties'] for f in got if all(k in f['properties'] for k in S1B)]
    s = pd.DataFrame(recs)[['point_id'] + S1B]
    s.to_csv(S1F, index=False, encoding='utf-8-sig')
    log('S1 取样 %d/%d → %s' % (len(s), len(df), S1F))
    return s


def cv_auc(X, y, seed=7):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, f1_score
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    p = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        clf = RandomForestClassifier(n_estimators=300, class_weight='balanced_subsample',
                                     n_jobs=15, random_state=seed).fit(X[tr], y[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
    best = max(((f1_score(y, (p >= t).astype(int)), t) for t in np.arange(0.2, 0.75, 0.05)))
    return round(float(roc_auc_score(y, p)), 4), round(float(best[0]), 4)


def apply_policy(pool, pol):
    src = pool['src'].fillna('').astype(str)
    if pol == 'R0':
        return pool
    if pol == 'R4':
        return pool[src != SHRUB_SRC]
    if pol in ('R6', 'R7'):
        df = pool.copy()
        m = (df['src'].fillna('') == SHRUB_SRC)
        df.loc[m, 'class_new'] = 11 if pol == 'R6' else 13     # 灌丛层点：标签改草地/稀疏植被（特征保留）
        return df
    raise ValueError(pol)


def main():
    t0 = time.time()
    df = pd.read_csv(REF, encoding='utf-8-sig')
    df = df[df.Q1 != '无法判读'].copy()
    df['y'] = (df.Q1 == '是').astype(int)
    for c in STRUCT:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.merge(pd.read_csv(EMB, encoding='utf-8-sig'), on='point_id', how='left')
    s1 = get_s1(df)
    df = df.merge(s1, on='point_id', how='left')
    df = df[df[VC.FEATS + S1B].notna().all(axis=1) & df[STRUCT].notna().all(axis=1)].copy()
    log('样本 %d（灌丛 %d）' % (len(df), df.y.sum()))

    out = {'n': int(len(df)), 'n_shrub': int(df.y.sum()), 's1_probe': {}, 'arms': {}}
    # ---- A. S1 补测 ----
    Xaef = np.nan_to_num(df[VC.FEATS].to_numpy('float32'), nan=-999)
    Xst = np.nan_to_num(df[STRUCT].to_numpy('float32'), nan=-999)
    Xs1 = np.nan_to_num(df[S1B].to_numpy('float32'), nan=-999)
    y = df.y.to_numpy()
    for name, X in (('AEF+S1', np.hstack([Xaef, Xs1])),
                    ('AEF+结构+S1', np.hstack([Xaef, Xst, Xs1])),
                    ('仅S1', Xs1)):
        auc, f1 = cv_auc(X, y)
        out['s1_probe'][name] = dict(AUC=auc, F1=f1)
        log('  %-12s AUC=%.3f F1=%.3f' % (name, auc, f1))
    best = max(out['s1_probe'].values(), key=lambda m: m['AUC'])
    out['s1_verdict'] = ('可分' if best['AUC'] >= 0.85 and best['F1'] >= 0.55 else '仍不可分')
    log('S1 判定：%s（最佳 AUC %.3f / F1 %.3f）' % (out['s1_verdict'], best['AUC'], best['F1']))

    # ---- B. 四臂标签策略 ----
    log('载入 2023 池 …')
    pool = pd.read_parquet(POOL, columns=['lon', 'lat', 'class_new', 'src'] + VC.FEATS)
    names = VC.V31_NAMES()
    truth24 = df['Q2'].map(lambda s: None)  # 占位
    # 真值类码：复用 n2 的规则映射
    import n2_nx_exp as N2
    df['truth'] = df['Q2'].map(lambda s: N2.code_of(str(s) or ''))
    df = df[df.truth.notna()].copy()
    log('可映射真值 %d 点' % len(df))
    for pol in ('R0', 'R4', 'R6', 'R7'):
        tp = apply_policy(pool, pol)
        preds = np.zeros(len(df), dtype=int)
        for prov, box in PROV_BOX.items():
            m = (df.prov == prov).to_numpy()
            if not m.sum():
                continue
            x0, y0, x1, y1 = box
            sub = tp[(tp.lon >= x0 - 2) & (tp.lon <= x1 + 2) & (tp.lat >= y0 - 2) & (tp.lat <= y1 + 2)]
            tr = sub.sample(n=min(N_TRAIN, len(sub)), random_state=7)
            clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, max_leaf_nodes=5000,
                                         max_features='sqrt', n_jobs=15, random_state=7).fit(
                tr[VC.FEATS].to_numpy('float32'), VC.to_v31(tr['class_new'].to_numpy(int)))
            preds[m] = clf.predict(df.loc[m, VC.FEATS].to_numpy('float32'))
        t, p = df.truth.to_numpy(int), preds
        tm, pm = VC.to_macro(t), VC.to_macro(p)
        mer = np.where(np.isin(t, (9, 10, 11)), -1, t), np.where(np.isin(p, (9, 10, 11)), -1, p)
        rec = dict(n=int(len(t)), OA24=round(float((t == p).mean()), 4), OA9=round(float((tm == pm).mean()), 4),
                   OA_sg=round(float((mer[0] == mer[1]).mean()), 4),
                   shrub_pred=int((p == 10).sum() + (p == 9).sum()), shrub_true=int((t == 10).sum() + (t == 9).sum()),
                   grass_pred=int((p == 11).sum()), grass_true=int((t == 11).sum()))
        per = VC.metrics(tm, pm)['per']
        rec['per9'] = {str(k): v for k, v in per.items()}
        out['arms'][pol] = rec
        log('  %-3s OA24=%.3f OA9=%.3f 灌草合并=%.3f | 预测灌丛 %d/真值 %d | 预测草地 %d/真值 %d | %.1f min' % (
            pol, rec['OA24'], rec['OA9'], rec['OA_sg'], rec['shrub_pred'], rec['shrub_true'],
            rec['grass_pred'], rec['grass_true'], (time.time() - t0) / 60))
    best_arm = max(out['arms'].items(), key=lambda kv: kv[1]['OA9'])
    out['verdict'] = '最佳臂 = %s（OA9 %.4f）' % (best_arm[0], best_arm[1]['OA9'])
    log('判定：%s' % out['verdict'])
    os.makedirs(os.path.join(VC.RES, 'd2'), exist_ok=True)
    VC.jsave(out, os.path.join(VC.RES, 'd2', 'shrub_round2.json'))


if __name__ == '__main__':
    main()
