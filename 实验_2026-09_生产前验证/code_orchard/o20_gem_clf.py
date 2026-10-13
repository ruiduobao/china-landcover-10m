# -*- coding: utf-8 -*-
"""o20_gem_clf.py — GEM-Forest 全球树作物样本 → 中国判别器重训 + 全国候选打分（本地）

* 输入：F:/lc_work/外部样本/GEM_Forest/training_with_AEF.csv（47,308 点，A00-A63 + forest 标签 + .geo）
* 规则源（**预注册**）：
  正类 = forest∈{2,3}（2=热带树作物/3=欧洲果树，合计 4,465 ≈ 论文 ~4,400）；
  负类 = forest∈{0,1}（非森林/森林）抽样 2:1；
  特征 = AEF64；模型 = RF(400) 与 LinearSVM（论文称 SVM 更稳）双跑；
  中国点 = DataV 国界内含（复用 k19 边界逻辑）。
  测试：
    T1 全球 5 折 CV（AUC/F1）
    T2 **中国迁移**：用非中国点训练 → 测中国 352 个树作物点召回 + 中国非树作物点 FPR
  生产模型 = 全数据（含中国）训练 → 打分全部候选
* 输出：results/orchard/gem_clf.json；data/orchard/scored_gem.parquet（big pool + gem610 的 p_treecrop）
* 用法：python o20_gem_clf.py
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
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

WORK = r'F:/lc_work/v31_exp'
ORD = os.path.join(WORK, 'data', 'orchard')
RES = os.path.join(WORK, 'results', 'orchard')
GEM_TRAIN = r'F:/lc_work/外部样本/GEM_Forest/training_with_AEF.csv'
BND = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/边界/china_100000_full.json'


def emit(m):
    import time
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def cv_metrics(X, y, seed=7):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    p = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, max_features='sqrt',
                                     class_weight='balanced_subsample', n_jobs=14,
                                     random_state=seed).fit(X[tr], y[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y, p)
    best = max(((f1_score(y, (p >= t).astype(int), zero_division=0), t) for t in np.arange(0.2, 0.8, 0.05)))
    return dict(AUC=round(float(auc), 4), F1=round(float(best[0]), 4),
                prec=round(float(precision_score(y, (p >= best[1]).astype(int), zero_division=0)), 4),
                rec=round(float(recall_score(y, (p >= best[1]).astype(int), zero_division=0)), 4)), p


def main():
    os.makedirs(RES, exist_ok=True)
    from shapely.geometry import shape
    from shapely import make_valid, contains_xy
    from shapely.ops import unary_union
    emit('载入 GEM 训练集…')
    d = pd.read_csv(GEM_TRAIN, encoding='utf-8-sig')
    geo = d['.geo'].map(lambda s: json.loads(s)['coordinates'] if isinstance(s, str) else None)
    d['lon'] = [g[0] if g else np.nan for g in geo]
    d['lat'] = [g[1] if g else np.nan for g in geo]
    d = d[d.lon.notna()].reset_index(drop=True)
    gj = json.load(open(BND, encoding='utf-8'))
    geoms = []
    for ft in gj['features']:
        g = shape(ft['geometry'])
        if not g.is_valid:
            g = make_valid(g)
        geoms.append(g)
    china = unary_union(geoms)
    inb = (d.lon >= 73) & (d.lon <= 136) & (d.lat >= 18) & (d.lat <= 54)
    d['in_cn'] = False
    idx = np.where(inb)[0]
    if len(idx):
        c = contains_xy(china, d.lon.to_numpy()[idx], d.lat.to_numpy()[idx])
        d.loc[idx[c], 'in_cn'] = True
    d['y'] = d['forest'].isin([2, 3]).astype(int)
    X_all = d[VC.FEATS].to_numpy('float64')
    ok = ~np.isnan(X_all).any(axis=1)
    d, X_all = d[ok].reset_index(drop=True), X_all[ok]
    emit('总 %d（正 %d / 中国点 %d，中国正 %d）' % (len(d), int(d.y.sum()), int(d.in_cn.sum()),
                                              int((d.in_cn & (d.y == 1)).sum())))
    out = {'n': int(len(d)), 'pos': int(d.y.sum()), 'cn': int(d.in_cn.sum()),
           'cn_pos': int((d.in_cn & (d.y == 1)).sum())}
    # T2 中国迁移（先做，用非中国训练）
    tr = (~d.in_cn).to_numpy()
    te_pos = (d.in_cn & (d.y == 1)).to_numpy()
    te_neg = (d.in_cn & (d.y == 0)).to_numpy()
    clf_t = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, max_features='sqrt',
                                   class_weight='balanced_subsample', n_jobs=14, random_state=7)
    # 负类下采样 2:1
    neg_idx = d.index[tr & (d.y == 0)].to_numpy()
    pos_idx = d.index[tr & (d.y == 1)].to_numpy()
    rng = np.random.default_rng(7)
    neg_sel = rng.choice(neg_idx, size=min(len(neg_idx), 2 * len(pos_idx)), replace=False)
    trn = np.concatenate([pos_idx, neg_sel])
    clf_t.fit(X_all[trn], d.y.to_numpy()[trn])
    p_pos = clf_t.predict_proba(X_all[te_pos])[:, 1]
    p_neg = clf_t.predict_proba(X_all[te_neg])[:, 1]
    out['T2_cn_transfer'] = {'n_pos': int(te_pos.sum()), 'n_neg': int(te_neg.sum()),
                             'recall@0.5': round(float((p_pos >= 0.5).mean()), 3),
                             'recall@0.3': round(float((p_pos >= 0.3).mean()), 3),
                             'fpr@0.5': round(float((p_neg >= 0.5).mean()), 3),
                             'fpr@0.3': round(float((p_neg >= 0.3).mean()), 3),
                             'auc_cn': round(float(roc_auc_score(np.r_[np.ones(len(p_pos)), np.zeros(len(p_neg))],
                                                                 np.r_[p_pos, p_neg])), 4)}
    emit('T2 中国迁移: %s' % out['T2_cn_transfer'])
    # T1 全球 CV（正:负=1:2 采样）
    pos_all = d.index[d.y == 1].to_numpy()
    neg_all = d.index[d.y == 0].to_numpy()
    neg_sel2 = rng.choice(neg_all, size=min(len(neg_all), 2 * len(pos_all)), replace=False)
    sub = np.concatenate([pos_all, neg_sel2])
    m, _ = cv_metrics(X_all[sub], d.y.to_numpy()[sub])
    out['T1_global_cv'] = m
    emit('T1 全球 CV: %s' % m)
    # 生产模型（全数据 1:2）
    neg_sel3 = rng.choice(neg_all, size=min(len(neg_all), 2 * len(pos_all)), replace=False)
    sub3 = np.concatenate([pos_all, neg_sel3])
    clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, max_features='sqrt',
                                 class_weight='balanced_subsample', n_jobs=14, random_state=7)
    clf.fit(X_all[sub3], d.y.to_numpy()[sub3])
    # 打分：big pool（feats/*） + gem_cn
    feats = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(ORD, 'feats', 'feat_*_p*.parquet')))],
                      ignore_index=True)
    feats = feats[feats.aef_ok]
    feats['p_treecrop'] = clf.predict_proba(feats[VC.FEATS].to_numpy('float32'))[:, 1]
    gem = pd.read_parquet(os.path.join(ORD, 'gem_cn.parquet'))
    gf = pd.read_parquet(os.path.join(ORD, 'gem', 'feats_y2023.parquet'))
    gem['k'] = gem.lon.round(6).astype(str) + '_' + gem.lat.round(6).astype(str)
    gf['k'] = gf.lon.round(6).astype(str) + '_' + gf.lat.round(6).astype(str)
    gm = gem.merge(gf[['k', 'aef_ok'] + VC.FEATS], on='k', how='left')
    gm = gm[gm.aef_ok == True]
    gm['p_treecrop'] = clf.predict_proba(gm[VC.FEATS].to_numpy('float32'))[:, 1]
    sc = pd.concat([feats[['lon', 'lat', 'frame', 'p_treecrop']].assign(src='pool'),
                    gm[['lon', 'lat', 'p_treecrop']].assign(frame='gem_cn', src='gem')], ignore_index=True)
    sc.to_parquet(os.path.join(ORD, 'scored_gem.parquet'), index=False)
    out['score_dist'] = {'pool_p>=0.5': int((feats.p_treecrop >= 0.5).sum()),
                         'pool_p>=0.7': int((feats.p_treecrop >= 0.7).sum()),
                         'gem_p>=0.5': int((gm.p_treecrop >= 0.5).sum()),
                         'gem_p>=0.7': int((gm.p_treecrop >= 0.7).sum())}
    emit('打分: %s' % out['score_dist'])
    VC.jsave(out, os.path.join(RES, 'gem_clf.json'))


if __name__ == '__main__':
    main()
