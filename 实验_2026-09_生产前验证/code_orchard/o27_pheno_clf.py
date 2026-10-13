# -*- coding: utf-8 -*-
"""o27_pheno_clf.py — 物候判别器：橡胶 vs 天然林（S2 月度 NDVI + S1 冬夏）

* 输入：data/orchard/pheno/{rubber_p*, gemcn_p*}.parquet
        GEM 标签（gemcn 的 forest 列：2/3=树作物[中国区=橡胶]，0/1=其他）
        data/orchard/judge_r2/judge_r2_rubber_*.csv（155 点目视标签）
* 规则源（**预注册**）：
  F1 基础物候 = 6 个月度 NDVI；F2 = F1 + S1 4 波；
  T1 训练/测试 = GEM 中国点（正 352 vs 负 2913 采样），5 折 CV；
  T2 关键对比 = **海南岛内 橡胶点 vs 天然林负样本点**（同岛对照，最紧）；
  T3 外部验证 = 目视判读 155 点（AUC：物候分 vs 目视是否）；
  应用：30k 橡胶候选过筛 → v5 分层（A=产品∧物候∧[判是]；B=产品∧物候；C=其余）
* 输出：results/orchard/pheno_clf.json + data/orchard/rubber_pheno_scored.parquet
* 用法：python o27_pheno_clf.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

WORK = r'F:/lc_work/v31_exp'
ORD = os.path.join(WORK, 'data', 'orchard')
RES = os.path.join(WORK, 'results', 'orchard')
CB = ['nd_m01', 'nd_m03', 'nd_m05', 'nd_m07', 'nd_m09', 'nd_m11']
SB = ['s1_vv_w', 's1_vh_w', 's1_vv_s', 's1_vh_s']


def emit(m):
    import time
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def add_feats(d):
    d = d.copy()
    d['drop'] = d['nd_m07'] - d['nd_m01']          # 冬落幅（橡胶应大）
    d['amp'] = d[CB].max(axis=1) - d[CB].min(axis=1)
    d['spring'] = d['nd_m03'] - d['nd_m01']
    d['summer_fall'] = d['nd_m09'] - d['nd_m07']
    d['s1_ratio'] = d['s1_vv_s'] - d['s1_vv_w']     # 冬夏结构差
    return d


def cv_eval(X, y, seed=7):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    p = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features='sqrt',
                                     class_weight='balanced_subsample', n_jobs=14, random_state=seed)
        clf.fit(X[tr], y[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
    return round(float(roc_auc_score(y, p)), 4), p


def main():
    os.makedirs(RES, exist_ok=True)
    out = {}
    # ---- 数据 ----
    g = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(ORD, 'pheno', 'gemcn_p*.parquet')))],
                  ignore_index=True)
    if 'forest_x' in g.columns:
        g = g.rename(columns={'forest_x': 'forest'}).drop(columns=['forest_y'], errors='ignore')
    g = add_feats(g)
    feats_all = CB + SB + ['drop', 'amp', 'spring', 'summer_fall', 's1_ratio']
    g_ok = g[g[CB].notna().all(axis=1) & g[SB].notna().all(axis=1)].copy()
    feats_ok = feats_all
    g_ok['y'] = g_ok.forest.isin([2, 3]).astype(int)
    emit('GEM 中国点 %d（正 %d）' % (len(g_ok), int(g_ok.y.sum())))
    # T1 全局 GEM（1:2 负采样）
    rng = np.random.default_rng(7)
    pos = g_ok.index[g_ok.y == 1].to_numpy()
    neg = rng.choice(g_ok.index[g_ok.y == 0].to_numpy(), size=min(int((g_ok.y == 0).sum()), 2 * len(pos)),
                     replace=False)
    sub = pd.concat([g_ok.loc[pos], g_ok.loc[neg]])
    X = sub[feats_ok].to_numpy('float32')
    auc1, _ = cv_eval(X, sub.y.to_numpy())
    out['T1_GEM全中国'] = dict(AUC_PHENO_ONLY=round(float(roc_auc_score(
        sub.y, RandomForestClassifier(n_estimators=200, n_jobs=14, random_state=7).fit(
            sub[CB].to_numpy('float32'), sub.y).predict_proba(sub[CB].to_numpy('float32'))[:, 1])), 4),
        AUC_F2=auc1, n=int(len(sub)))
    emit('T1 GEM 全中国（物候→橡胶） AUC(F2)=%.3f' % auc1)
    # T2 海南岛内对照
    hn = g_ok[(g_ok.lon > 108.5) & (g_ok.lon < 111.2) & (g_ok.lat > 18.1) & (g_ok.lat < 20.2)]
    hn_p, hn_n = hn[hn.y == 1], hn[hn.y == 0]
    out['T2_海南对照'] = dict(n_pos=int(len(hn_p)), n_neg=int(len(hn_n)))
    if len(hn_p) >= 15 and len(hn_n) >= 15:
        sub2 = pd.concat([hn_p, hn_n])
        X2 = sub2[feats_ok].to_numpy('float32')
        auc2, p2 = cv_eval(X2, sub2.y.to_numpy())
        sub2 = sub2.assign(pheno_p=p2)
        out['T2_海南对照'].update(AUC=auc2,
                                  drop_pos_med=round(float(hn_p['drop'].median()), 3),
                                  drop_neg_med=round(float(hn_n['drop'].median()), 3))
        emit('T2 海南对照（%d vs %d） AUC=%.3f；冬落幅中位 橡胶 %.3f / 天然林 %.3f' % (
            len(hn_p), len(hn_n), auc2, hn_p['drop'].median(), hn_n['drop'].median()))
    else:
        out['T2_海南对照']['note'] = '样本不足，只报分布'
        emit('T2 样本不足：%d vs %d' % (len(hn_p), len(hn_n)))
    # 生产模型（全 GEM 中国点 1:2）
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features='sqrt',
                                 class_weight='balanced_subsample', n_jobs=14, random_state=7)
    clf.fit(X, sub.y.to_numpy())
    # T3 外部验证：判读点
    rub = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(ORD, 'pheno', 'rubber_p*.parquet')))],
                    ignore_index=True)
    rub = add_feats(rub)
    rub_ok = rub[rub[CB].notna().all(axis=1) & rub[SB].notna().all(axis=1)].copy()
    rub_ok['pheno_p'] = clf.predict_proba(rub_ok[feats_ok].to_numpy('float32'))[:, 1]
    j = pd.concat([pd.read_csv(os.path.join(ORD, 'judge_r2', 'judge_r2_rubber_0%d.csv' % k), encoding='utf-8-sig')
                   for k in range(5)], ignore_index=True)
    jp = pd.read_csv(os.path.join(ORD, 'judge_points_r2.csv'), encoding='utf-8-sig')
    j = j.merge(jp[['point_id', 'lon', 'lat']], on='point_id')
    jj = j[j.Q1.isin(['是', '否'])].assign(lon6=j.lon.round(6), lat6=j.lat.round(6))
    mm = jj.merge(rub_ok.assign(lon6=rub_ok.lon.round(6), lat6=rub_ok.lat.round(6))[
        ['lon6', 'lat6', 'pheno_p', 'drop']], on=['lon6', 'lat6'], how='inner')
    mm['y'] = (mm.Q1 == '是').astype(int)
    if len(mm) > 30 and mm.y.nunique() == 2:
        auc3 = round(float(roc_auc_score(mm.y, mm.pheno_p)), 4)
        out['T3_视读验证'] = dict(n=int(len(mm)), AUC=auc3,
                                 drop_yes_med=round(float(mm[mm.y == 1]['drop'].median()), 3),
                                 drop_no_med=round(float(mm[mm.y == 0]['drop'].median()), 3))
        emit('T3 判读点验证（n=%d） AUC=%.3f；判是点冬落幅中位 %.3f vs 判否 %.3f' % (
            len(mm), auc3, mm[mm.y == 1]['drop'].median(), mm[mm.y == 0]['drop'].median()))
    else:
        out['T3_视读验证'] = dict(n=int(len(mm)), note='样本期间无覆盖或单类')
    # 30k 打分
    rub_ok.to_parquet(os.path.join(ORD, 'rubber_pheno_scored.parquet'), index=False)
    out['rubber_scored'] = {'n': int(len(rub_ok)),
                           'pheno_p>=0.5': int((rub_ok.pheno_p >= 0.5).sum()),
                           'pheno_p>=0.7': int((rub_ok.pheno_p >= 0.7).sum())}
    emit('30k 物候分：p≥0.5 %d / p≥0.7 %d' % (out['rubber_scored']['pheno_p>=0.5'],
                                           out['rubber_scored']['pheno_p>=0.7']))
    VC.jsave(out, os.path.join(RES, 'pheno_clf.json'))


if __name__ == '__main__':
    main()
