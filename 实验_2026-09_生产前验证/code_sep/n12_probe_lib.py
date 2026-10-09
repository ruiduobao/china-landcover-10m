# -*- coding: utf-8 -*-
"""n12_probe_lib.py — 可分性探针公共库（5 折分层 CV，RF；AUC/最佳 F1/精确率/召回率）

被 n13/n14/n15 复用。判据不在此处——各实验把判据写在自己文件头（预注册）。
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score


def cv_eval(X, y, seed=7, n=300):
    """返回 (metrics, p)。y 为 0/1；样本 <10 或单一类别时返回 None。"""
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2 or len(y) < 10:
        return None, None
    n_splits = min(5, int(np.bincount(y).min()))
    if n_splits < 2:
        return None, None
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    p = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        clf = RandomForestClassifier(n_estimators=n, min_samples_leaf=1, max_features='sqrt',
                                     class_weight='balanced_subsample', n_jobs=15,
                                     random_state=seed).fit(X[tr], y[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y, p)
    best = max(((f1_score(y, (p >= t).astype(int), zero_division=0), t) for t in np.arange(0.2, 0.75, 0.05)))
    return dict(AUC=round(float(auc), 4), F1=round(float(best[0]), 4), thr=round(float(best[1]), 2),
                prec=round(float(precision_score(y, (p >= best[1]).astype(int), zero_division=0)), 4),
                rec=round(float(recall_score(y, (p >= best[1]).astype(int), zero_division=0)), 4),
                n=int(len(y)), pos=int(y.sum())), p


def pair_probe(df, feats_sets, pos_mask, neg_mask, seed=7):
    """对每个特征集跑 5 折 CV。feats_sets: {name: [cols]}；pos/neg: bool Series。"""
    sub = df[pos_mask | neg_mask].copy()
    y = pos_mask.loc[sub.index].astype(int).to_numpy()
    out = {}
    for name, cols in feats_sets.items():
        X = np.nan_to_num(sub[cols].to_numpy('float32'), nan=-999)
        m, _ = cv_eval(X, y, seed=seed)
        if m:
            out[name] = m
    return out, sub, y
