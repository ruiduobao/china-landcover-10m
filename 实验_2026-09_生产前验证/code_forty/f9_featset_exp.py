# -*- coding: utf-8 -*-
"""f9_featset_exp.py — 小实验：生态特征能否提升困难类判别（决定是否改生产特征集）

* 输入：data/m3/{exp*_judged,refset_all,gsnx_ref_judged,gsnx_ref_s1,gsnx_s2dem,gap_s2dem,eco_features_ref}.csv
* 规则源（预注册，跑前定稿）：
  特征集（逐级叠加）：F1=AEF64 ｜ F2=+S1(4) ｜ F3=+S2(12) ｜ F4=+DEM(elev,slope) ｜
                      F5=+生态(bio01,bio12,occ_pt,occ_90) ｜ F6=AEF+经纬度(2)（用户 Q4 的地理先验）
  任务（二分类，5 折分层 CV，RF300 balanced_subsample；判读真值为正/负标签）：
    T1 灌丛 vs 其它（干湿合并；正=影像真灌丛）
    T2 草本沼泽 vs 草地（expC 主战场）
    T3 湖河滩地 vs 裸地+水体（用户 Q1）
    T4 水体 vs 草本沼泽（阳性对照，应高）
  判据：**F5 相对 F4 的 AUC 提升 ≥0.02（且方向一致）→ 采纳生态特征入生产；否则不加**。
        任一任务的最佳特征集若 AUC≥0.85 → 该类可分（可用于"救回"）
* 门槛：每个任务正负样本各 ≥15 才做（否则报"样本不足"）
* 输出：results/d2/featset_exp.json + reports/_log 表
* 用法：python f9_featset_exp.py
"""
import collections
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
sys.path.insert(0, os.path.join(WORK, 'code'))
import numpy as np
import pandas as pd
import v31_common as VC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score

M3 = os.path.join(WORK, 'data', 'm3')
OUT = os.path.join(WORK, 'results', 'd2', 'featset_exp.json')
SEED = 7
NAMES = VC.V31_NAMES()


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def load_all():
    frames = []
    # exp 系列（含 AEF+S1+S2+DEM）
    for f in ('expA_judged.csv', 'expA2_judged.csv', 'expB_judged.csv', 'expC_judged.csv',
              'expC2_judged.csv', 'expC3_judged.csv', 'expC4_judged.csv'):
        p = os.path.join(M3, f)
        if os.path.exists(p):
            d = pd.read_csv(p, encoding='utf-8-sig')
            keep = ['point_id', 'lon', 'lat', 'truth_code'] + \
                   [c for c in d.columns if c.startswith('A') and c[1:].isdigit()] + \
                   [c for c in ('s1_vv_w', 's1_vh_w', 's1_vv_s', 's1_vh_s', 'dem', 'slope',
                                'ndvi_djf', 'ndwi_djf', 'mndwi_djf', 'ndvi_mam', 'ndwi_mam', 'mndwi_mam',
                                'ndvi_jja', 'ndwi_jja', 'mndwi_jja', 'ndvi_son', 'ndwi_son', 'mndwi_son')
                    if c in d.columns]
            frames.append(d[keep])
    # gap 点（AEF? 无 → 用 eco + s2dem；AEF 需另取，先并入 s2dem 与 eco）
    gp = pd.read_csv(os.path.join(M3, 'gap_points.csv'), encoding='utf-8-sig')
    gj = []
    for i in range(1, 6):
        p = os.path.join(M3, 'gap_imagery', 'judge_%d.csv' % i)
        if os.path.exists(p):
            gj.append(pd.read_csv(p, encoding='utf-8-sig'))
    G = gp.merge(pd.concat(gj, ignore_index=True), on='point_id')
    G = G[G.Q1 != '无法判读'].copy()
    G['truth_code'] = G.Q2.map(lambda s: {v: int(k) for k, v in NAMES.items()}.get(str(s), 0))
    sd = os.path.join(M3, 'gap_s2dem.csv')
    if os.path.exists(sd):
        G = G.merge(pd.read_csv(sd, encoding='utf-8-sig'), on='point_id', how='left')
    frames.append(G[[c for c in ['point_id', 'lon', 'lat', 'truth_code'] + \
                     [x for x in ('s1_vv_w', 's1_vh_w', 's1_vv_s', 's1_vh_s', 'dem', 'slope',
                                  'ndvi_djf', 'ndwi_djf', 'mndwi_djf', 'ndvi_mam', 'ndwi_mam', 'mndwi_mam',
                                  'ndvi_jja', 'ndwi_jja', 'mndwi_jja', 'ndvi_son', 'ndwi_son', 'mndwi_son')
                      if x in G.columns] if c in G.columns]])
    # 甘宁（AEF + s1 + s2dem + Q1）
    gs = pd.read_csv(os.path.join(M3, 'gsnx_ref_judged.csv'), encoding='utf-8-sig')
    gs = gs[gs.Q1 != '无法判读'].copy()
    gs['truth_code'] = gs.apply(lambda r: 10 if r.Q1 == '是' else
                                {v: int(k) for k, v in NAMES.items()}.get(str(r.Q2), 0), axis=1)
    for f, cols in (('gsnx_ref_embed.csv', None), ('gsnx_ref_s1.csv', None), ('gsnx_s2dem.csv', None)):
        p = os.path.join(M3, f)
        if os.path.exists(p):
            gs = gs.merge(pd.read_csv(p, encoding='utf-8-sig'), on='point_id', how='left')
    frames.append(gs)
    # 生态特征（所有点）
    eco = pd.read_csv(os.path.join(M3, 'eco_features_ref.csv'), encoding='utf-8-sig')
    df = pd.concat(frames, ignore_index=True, sort=False).drop_duplicates('point_id')
    df = df.merge(eco, on='point_id', how='left')
    return df


def cv(X, y, seed=SEED):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    p = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        clf = RandomForestClassifier(n_estimators=300, class_weight='balanced_subsample',
                                     n_jobs=15, random_state=seed).fit(X[tr], y[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
    auc = float(roc_auc_score(y, p))
    best = max(f1_score(y, (p >= t).astype(int)) for t in np.arange(0.2, 0.75, 0.05))
    return round(auc, 4), round(float(best), 4)


def main():
    t0 = time.time()
    df = load_all()
    emit('样本 %d 点' % len(df))
    aef = [c for c in VC.FEATS if c in df.columns]
    s1 = [c for c in ('s1_vv_w', 's1_vh_w', 's1_vv_s', 's1_vh_s') if c in df.columns]
    s2 = [c for c in df.columns if c.startswith(('ndvi_', 'ndwi_', 'mndwi_'))]
    dmb = [c for c in ('dem', 'slope', 'elev') if c in df.columns]
    eco = [c for c in ('bio01', 'bio12', 'occ_pt', 'occ_90') if c in df.columns]
    sets = {
        'F1_AEF': aef,
        'F2_+S1': aef + s1,
        'F3_+S2': aef + s1 + s2,
        'F4_+DEM': aef + s1 + s2 + dmb,
        'F5_+生态': aef + s1 + s2 + dmb + eco,
        'F6_AEF+经纬': aef + ['lon', 'lat'],
    }
    emit('特征集：%s' % {k: len(v) for k, v in sets.items()})

    def task(mask_pos, mask_neg, name):
        sub = df[mask_pos | mask_neg].copy()
        y = mask_pos[mask_pos | mask_neg].to_numpy().astype(int)
        if y.sum() < 15 or (1 - y).sum() < 15:
            emit('  %-22s 样本不足（正 %d / 负 %d）' % (name, int(y.sum()), int((1 - y).sum())))
            return None
        res = {}
        for k, cols in sets.items():
            X = np.nan_to_num(sub[cols].to_numpy('float32'), nan=-999, posinf=-999, neginf=-999)
            auc, f1 = cv(X, y)
            res[k] = dict(AUC=auc, F1=f1, n_feat=len(cols))
        best = max(res.items(), key=lambda kv: kv[1]['AUC'])
        emit('  %-22s n=%3d(正%2d) ｜ %s' % (
            name, len(sub), int(y.sum()),
            ' ｜ '.join('%s %.3f' % (k.split('_')[0], v['AUC']) for k, v in res.items())))
        emit('       最佳 %s AUC=%.3f F1=%.3f' % (best[0], best[1]['AUC'], best[1]['F1']))
        return dict(n=int(len(sub)), n_pos=int(y.sum()), results=res, best=best[0])

    out = {'feature_sets': {k: len(v) for k, v in sets.items()}, 'tasks': {}}
    tc = df.truth_code.fillna(0).astype(int)
    # T1 灌丛 vs 其它
    out['tasks']['T1_灌丛vs其它'] = task(tc == 10, (tc > 0) & (tc != 10), 'T1 灌丛vs其它')
    # T2 草本沼泽 vs 草地
    out['tasks']['T2_草沼vs草地'] = task(tc == 15, tc == 11, 'T2 草沼vs草地')
    # T3 湖河滩地 vs 裸地+水体
    out['tasks']['T3_滩地vs裸水'] = task(tc == 16, (tc == 22) | (tc == 23), 'T3 滩地vs裸+水')
    # T4 水体 vs 草本沼泽（对照）
    out['tasks']['T4_水体vs草沼'] = task(tc == 23, tc == 15, 'T4 水体vs草沼')
    # T5 旱地 vs 灌溉耕地（耕地内部）
    out['tasks']['T5_旱地vs灌溉'] = task(tc == 1, tc == 3, 'T5 旱地vs灌溉')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    VC.jsave(out, OUT)
    emit('完成 %.1f min → %s' % ((time.time() - t0) / 60, OUT))


if __name__ == '__main__':
    main()
