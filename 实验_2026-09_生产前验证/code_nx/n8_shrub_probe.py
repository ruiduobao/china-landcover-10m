# -*- coding: utf-8 -*-
"""n8_shrub_probe.py — 第 1 轮小样本：灌丛可分性探针（甘肃+宁夏 304 点参考集，预注册判据）

* 输入：F:/lc_work/v31_exp/data/m3/gsnx_ref_judged.csv（304 点影像判读真值 + GEDI/S2/Hansen 指标）
* 规则源（**预注册，先写死再跑**）：
  任务：二分类"真灌丛(Q1=是) vs 其它"，5 折分层交叉验证（StratifiedKFold, shuffle, seed 7）
  特征集：F1=AEF 64 维；F2=AEF+结构旁证[GEDI 冠层高 ch_mean、Hansen 树覆盖 tc2000、S2 物候 ndvi_amp/ndvi_winter/ndvi_summer]；
          F3=仅结构旁证（对照）
  模型：RandomForest(300, minLeaf1, sqrt, class_weight='balanced_subsample')
  判据：**若 max(AUC) ≥ 0.85 且 灌丛 F1 ≥ 0.55 → "10 m 可分，值得建窄口径灌丛库"；否则 → 走"灌草合并+辅助层"路线**
  另报：按 WorldCover 分层看 WC=20 点被判是 的比例（外部产品灌丛类在本区的实测精度）
* 输出：results/d2/shrub_probe.json + 打印 AUC/F1 表
* 用法：python n8_shrub_probe.py
"""
import csv
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import v31_common as VC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

M3D = r'F:/lc_work/v31_exp/data/m3'
REF = os.path.join(M3D, 'gsnx_ref_judged.csv')
EMB = os.path.join(M3D, 'gsnx_ref_embed.csv')
AEF = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
YEAR = 2023
STRUCT = ['ch_mean', 'tc2000', 'ndvi_amp', 'ndvi_winter', 'ndvi_summer']


def log(m):
    import time
    print('[%s] %s' % (time.strftime('%m-%d %H:%M:%S'), m), flush=True)


def get_embed(df, acct='zixen8v8'):
    if os.path.exists(EMB):
        e = pd.read_csv(EMB, encoding='utf-8-sig')
        if len(e) >= len(df):
            return e
    import ee
    VC.ensure_ctx(acct)
    pts = [ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]), {'point_id': r.point_id})
           for r in df.itertuples()]
    img = (ee.ImageCollection(AEF).filterDate('%d-01-01' % YEAR, '%d-01-01' % (YEAR + 1))
           .mosaic().select(VC.FEATS))
    got = img.sampleRegions(collection=ee.FeatureCollection(pts), scale=10, geometries=False,
                            tileScale=4).getInfo()['features']
    recs = [f['properties'] for f in got if all(k in f['properties'] for k in VC.FEATS)]
    e = pd.DataFrame(recs)[['point_id'] + VC.FEATS]
    e.to_csv(EMB, index=False, encoding='utf-8-sig')
    log('AEF 取样 %d/%d → %s' % (len(e), len(df), EMB))
    return e


def cv_eval(X, y, seed=7):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    p = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=1, max_features='sqrt',
                                     class_weight='balanced_subsample', n_jobs=15,
                                     random_state=seed).fit(X[tr], y[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y, p)
    best = max(((f1_score(y, (p >= t).astype(int)), t) for t in np.arange(0.2, 0.75, 0.05)))
    return dict(AUC=round(float(auc), 4), F1=round(float(best[0]), 4), thr=round(float(best[1]), 2),
                prec=round(float(precision_score(y, (p >= best[1]).astype(int), zero_division=0)), 4),
                rec=round(float(recall_score(y, (p >= best[1]).astype(int), zero_division=0)), 4)), p


def main():
    df = pd.read_csv(REF, encoding='utf-8-sig')
    df = df[df['Q1'] != '无法判读'].copy()
    df['y'] = (df['Q1'] == '是').astype(int)
    for c in STRUCT:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    emb = get_embed(df)
    df = df.merge(emb, on='point_id', how='left')
    miss = df[VC.FEATS].isna().any(axis=1).sum()
    log('样本 %d（灌丛 %d / 其它 %d）；特征缺失 %d 点' % (len(df), df.y.sum(), len(df) - df.y.sum(), miss))
    df = df[df[VC.FEATS].notna().all(axis=1)].copy()

    out = {'n': int(len(df)), 'n_shrub': int(df.y.sum()),
           'wc20_judged_yes': int(((df.wc == 20) & (df.y == 1)).sum()),
           'wc20_n': int((df.wc == 20).sum()), 'results': {}}
    y = df.y.to_numpy()
    sets = {
        'F1_AEF': df[VC.FEATS].to_numpy('float32'),
        'F2_AEF+结构': df[VC.FEATS + STRUCT].to_numpy('float32'),
        'F3_仅结构': df[STRUCT].to_numpy('float32'),
    }
    for name, X in sets.items():
        m, _ = cv_eval(np.nan_to_num(X, nan=-999), y)
        out['results'][name] = m
        log('%-12s AUC=%.3f  F1=%.3f (thr=%.2f, P=%.3f, R=%.3f)' % (
            name, m['AUC'], m['F1'], m['thr'], m['prec'], m['rec']))
    best = max(out['results'].values(), key=lambda m: m['AUC'])
    ok = best['AUC'] >= 0.85 and best['F1'] >= 0.55
    out['verdict'] = ('可建窄口径灌丛（AUC %.3f/F1 %.3f）' % (best['AUC'], best['F1'])) if ok else \
        ('不可分（最佳 AUC %.3f/F1 %.3f）→ 走灌草合并+辅助层路线' % (best['AUC'], best['F1']))
    log('判定：%s' % out['verdict'])
    log('旁证：WorldCover 灌丛层(20) 在本区 %d/%d 点为真灌丛 = %.0f%%' % (
        out['wc20_judged_yes'], out['wc20_n'], 100 * out['wc20_judged_yes'] / max(1, out['wc20_n'])))
    os.makedirs(os.path.join(VC.RES, 'd2'), exist_ok=True)
    VC.jsave(out, os.path.join(VC.RES, 'd2', 'shrub_probe.json'))


if __name__ == '__main__':
    main()
