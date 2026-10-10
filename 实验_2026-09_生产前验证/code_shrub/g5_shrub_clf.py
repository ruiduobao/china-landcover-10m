# -*- coding: utf-8 -*-
"""g5_shrub_clf.py — 用 78 个判读确认点检验"灌丛判别器"可分性 + 若可用则扩增

* 输入：data/shrub/judged_all.csv（454 判读点：是 78 / 否 328 / 无法 48）
* 规则源（预注册）：
  T1 二分类「真灌丛 vs 非灌丛」：5 折分层 CV，特征集 F1=AEF64 ｜ F2=+S2物候 ｜ F3=+DEM/气候/水
  T2 若 T1 最佳 AUC ≥0.85 → 视为可用判别器，对省内候选池打分扩增（T2 层）；
     0.75–0.85 → 仅作"辅助排序"（T3 层，需人工复核）；<0.75 → 不扩增（只用判读确认的 78 点）
  T3 常绿/落叶二分类同样评估（仅 78 点中的 68 个有明确 Q2）
* 门槛：正例 78 / 负例 328；报告 AUC/F1 与混淆
* 输出：results/d2/shrub_clf.json
* 用法：python g5_shrub_clf.py
"""
import collections, json, os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
sys.path.insert(0, os.path.join(WORK, 'code'))
import numpy as np, pandas as pd
import v31_common as VC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score

IN = os.path.join(WORK, 'data', 'shrub', 'judged_all.csv')
OUT = os.path.join(WORK, 'results', 'd2', 'shrub_clf.json')
SEED = 7

def emit(m): print(time.strftime('[%m-%d %H:%M:%S] ')+str(m), flush=True)

def cv(X, y, seed=SEED):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    p = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        clf = RandomForestClassifier(n_estimators=300, class_weight='balanced_subsample',
                                     n_jobs=15, random_state=seed).fit(X[tr], y[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
    auc = float(roc_auc_score(y, p))
    f1 = max(f1_score(y, (p >= t).astype(int)) for t in np.arange(0.2, 0.8, 0.05))
    return round(auc, 4), round(float(f1), 4), p

def main():
    df = pd.read_csv(IN, encoding='utf-8-sig')
    # 回连候选表补 AEF（judge_points.csv 未含 A00-A63 —— 2026-10-10 修正）
    cand = pd.read_csv(os.path.join(WORK, 'data', 'shrub', 'cand_all.csv'), encoding='utf-8-sig')
    missing = [c for c in VC.FEATS if c not in df.columns]
    if missing:
        df = df.merge(cand[['point_id'] + missing], on='point_id', how='left')
        emit('回连候选表补 %d 个 AEF 列（命中 nan=%d）' % (len(missing), int(df[missing[0]].isna().sum())))
    aef = [c for c in VC.FEATS if c in df.columns]
    s2 = [c for c in ('ndvi_djf', 'ndvi_jja', 'ndvi_amp') if c in df.columns]
    aux = [c for c in ('dem', 'slope', 'bio01', 'bio12', 'occ') if c in df.columns]
    sets = {'F1_AEF': aef, 'F2_+S2物候': aef + s2, 'F3_+地形气候': aef + s2 + aux}
    out = {'n_judged': int(len(df)), 'feature_sets': {k: len(v) for k, v in sets.items()}, 'tasks': {}}

    # T1 灌丛 vs 非灌丛
    sub = df[df.Q1_shrub.isin(['是', '否'])].copy()
    y = (sub.Q1_shrub == '是').astype(int).to_numpy()
    emit('T1：正 %d / 负 %d' % (int(y.sum()), int((1 - y).sum())))
    res = {}
    for k, cols in sets.items():
        X = np.nan_to_num(sub[cols].to_numpy('float32'), nan=-999, posinf=-999, neginf=-999)
        auc, f1, p = cv(X, y)
        res[k] = dict(AUC=auc, F1=f1)
        emit('  %-12s AUC=%.3f F1=%.3f' % (k, auc, f1))
        out['tasks'].setdefault('T1_shrub', {})[k] = dict(AUC=auc, F1=f1)
        np.save(os.path.join(WORK, 'data', 'shrub', 'clf_pred_%s.npy' % k.split('_')[0]), p)
    best = max(out['tasks']['T1_shrub'].items(), key=lambda kv: kv[1]['AUC'])
    out['T1_best'] = best[0]
    out['T1_verdict'] = ('可用判别器（扩增 T2）' if best[1]['AUC'] >= 0.85 else
                         ('仅辅助排序（扩增 T3，需复核）' if best[1]['AUC'] >= 0.75 else '不扩增'))
    emit('T1 判定：%s（%s AUC=%.3f）' % (out['T1_verdict'], best[0], best[1]['AUC']))

    # T3 常绿 vs 落叶（仅确认灌丛点）
    yes = df[(df.Q1_shrub == '是') & (df.Q2_evergreen.isin(['常绿', '落叶']))].copy()
    if len(yes) >= 20 and yes.Q2_evergreen.nunique() == 2:
        y2 = (yes.Q2_evergreen == '常绿').astype(int).to_numpy()
        emit('T3：常绿 %d / 落叶 %d' % (int(y2.sum()), int((1 - y2).sum())))
        for k, cols in sets.items():
            X = np.nan_to_num(yes[cols].to_numpy('float32'), nan=-999, posinf=-999, neginf=-999)
            auc, f1, _ = cv(X, y2)
            out['tasks'].setdefault('T3_evergreen', {})[k] = dict(AUC=auc, F1=f1)
            emit('  %-12s AUC=%.3f F1=%.3f' % (k, auc, f1))
        # 仅用物候
        if 'ndvi_amp' in yes.columns:
            a = yes.ndvi_amp.astype(float).to_numpy()
            auc = float(roc_auc_score(y2, -np.nan_to_num(a, nan=0.2)))
            out['tasks'].setdefault('T3_evergreen', {})['仅ndvi_amp'] = dict(AUC=round(auc, 4), F1=None)
            emit('  仅 ndvi_amp（取负） AUC=%.3f' % auc)
    else:
        out['tasks']['T3_evergreen'] = '样本不足'
    VC.jsave(out, OUT)
    emit('→ %s' % OUT)

if __name__ == '__main__': main()
