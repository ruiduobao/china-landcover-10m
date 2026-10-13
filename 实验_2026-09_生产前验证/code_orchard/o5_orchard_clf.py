# -*- coding: utf-8 -*-
"""o5_orchard_clf.py — 乔灌园地：AEF64 判别器训练/校准/全国候选打分（本地）

* 输入：data/orchard/feats/feat_p*.parquet（候选特征）＋ data/orchard/judge_merged.csv（子代理判读合并）
* 规则源：Q1==是→1、Q1==否→0、无法判读剔除；RF(400, sqrt, balanced_subsample)，5 折分层 CV（seed 7）；
          全量 fit 后给所有候选打分 p_orchard；校准报告 θ∈{0.5,0.6,0.7,0.8} 的 P(是|score≥θ)（Wilson CI）
* 门槛：AUC 报实测；每框 precision@θ 一并报（框选择依据）
* 输出：data/orchard/scored.parquet（point_uid,lon,lat,frame,p_orchard）+ results/orchard/clf.json
* 用法：python o5_orchard_clf.py
"""
import glob
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

WORK = r'F:/lc_work/v31_exp'
ORD = os.path.join(WORK, 'data', 'orchard')
RES = os.path.join(WORK, 'results', 'orchard')
JUDGED = os.path.join(ORD, 'judge2_merged.csv')


def wilson(p, n, z=1.96):
    if n == 0:
        return (0, 0)
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


def main():
    os.makedirs(RES, exist_ok=True)
    feats = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(ORD, 'feats', 'feat_*_p*.parquet')))],
                      ignore_index=True)
    print('候选特征 %d 行；aef_ok %d' % (len(feats), int(feats.aef_ok.sum())))
    feats = feats[feats.aef_ok].copy()
    j = pd.read_csv(JUDGED, encoding='utf-8-sig')
    j = j[j.Q1.isin(['是', '否'])].copy()
    j['y'] = (j.Q1 == '是').astype(int)
    jm = j.merge(feats[['point_uid', 'lon', 'lat'] + VC.FEATS], left_on=['point_id'], right_on=['point_uid'],
                 how='inner') if 'point_uid' in feats.columns else None
    # 判读点与候选特征按坐标连接（判读点抽样自候选池）
    key = lambda d: d.round({'lon': 6, 'lat': 6})
    j2 = key(j[['point_id', 'lon', 'lat', 'frame', 'y']])
    f2 = key(feats[['lon', 'lat'] + VC.FEATS])
    m = j2.merge(f2, on=['lon', 'lat'], how='inner')
    print('判读点 %d，可连特征 %d' % (len(j2), len(m)))
    X = m[VC.FEATS].to_numpy('float32')
    y = m.y.to_numpy(int)
    out = {'n_judged': int(len(j2)), 'n_joined': int(len(m)), 'pos': int(y.sum()), 'neg': int((1 - y).sum()),
           'cv': {}, 'theta_prec': {}, 'by_frame': {}}
    # CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    p = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, max_features='sqrt',
                                     class_weight='balanced_subsample', n_jobs=14, random_state=7).fit(X[tr], y[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y, p)
    best = max(((f1_score(y, (p >= t).astype(int), zero_division=0), t) for t in np.arange(0.2, 0.8, 0.05)))
    out['cv'] = dict(AUC=round(float(auc), 4), F1=round(float(best[0]), 4), thr=round(float(best[1]), 2),
                     prec=round(float(precision_score(y, (p >= best[1]).astype(int), zero_division=0)), 4),
                     rec=round(float(recall_score(y, (p >= best[1]).astype(int), zero_division=0)), 4))
    print('CV: AUC=%.3f F1=%.3f @thr=%.2f' % (out['cv']['AUC'], out['cv']['F1'], out['cv']['thr']))
    for t in (0.5, 0.6, 0.7, 0.8):
        sel = p >= t
        if sel.sum():
            pr = float(y[sel].mean())
            out['theta_prec'][str(t)] = dict(n=int(sel.sum()), precision=round(pr, 4),
                                             wilson=wilson(pr, int(sel.sum())), recall=round(float(sel[y == 1].mean()), 4))
    print('θ校准:', json.dumps(out['theta_prec'], ensure_ascii=False))
    # 分框精度（CV 预测）
    m['p'] = p
    for fr, g in m.groupby('frame'):
        out['by_frame'][fr] = dict(n=int(len(g)), pos=int(g.y.sum()),
                                   rate=round(float(g.y.mean()), 3),
                                   prec_at_06=round(float(g[g.p >= 0.6].y.mean()), 3) if (g.p >= 0.6).any() else None,
                                   n_at_06=int((g.p >= 0.6).sum()))
    print('分框:', json.dumps(out['by_frame'], ensure_ascii=False, indent=1))
    # 全量 fit + 打分
    clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, max_features='sqrt',
                                 class_weight='balanced_subsample', n_jobs=14, random_state=7).fit(X, y)
    sc = feats[['point_uid', 'lon', 'lat', 'frame']].copy()
    sc['p_orchard'] = clf.predict_proba(feats[VC.FEATS].to_numpy('float32'))[:, 1]
    sc.to_parquet(os.path.join(ORD, 'scored.parquet'), index=False)
    VC.jsave(out, os.path.join(RES, 'clf.json'))
    print('打分 %d → scored.parquet' % len(sc))
    # 附：母库框（mk）单独报（现有样本的模型一致率）
    mk = sc[sc.frame == 'mk_orchard']
    print('母库框 %d 点 p≥0.6 占比 %.3f' % (len(mk), (mk.p_orchard >= 0.6).mean()))


if __name__ == '__main__':
    main()
