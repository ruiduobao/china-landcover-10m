# -*- coding: utf-8 -*-
"""o24_strat_verify.py — 分层验证：判读结论 × 新模型 p（橡胶/茶园/GEM/池），输出验证表

* 输入：data/orchard/judge_r2/judge_r2_{rubber,tea}_*.csv（320 判读）
        data/orchard/{gem_judge_merged.csv, judge2_merged.csv}
        data/orchard/scored_{rubber,tea}.parquet、gem/feats_y2023、feats/*
* 规则源：把每批判读点接到 p_treecrop（新模型，2023 特征），按 p 分档算"判是率"
* 输出：results/orchard/strat_verify.json + 打印表
* 用法：python o24_strat_verify.py
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

WORK = r'F:/lc_work/v31_exp'
ORD = os.path.join(WORK, 'data', 'orchard')
RES = os.path.join(WORK, 'results', 'orchard')
GEM_TRAIN = r'F:/lc_work/外部样本/GEM_Forest/training_with_AEF.csv'


def train_model():
    d = pd.read_csv(GEM_TRAIN, encoding='utf-8-sig')
    d['y'] = d['forest'].isin([2, 3]).astype(int)
    X = d[VC.FEATS].to_numpy('float64')
    ok = ~np.isnan(X).any(axis=1)
    y = d.y.to_numpy()[ok]
    X = X[ok]
    rng = np.random.default_rng(7)
    pos = np.where(y == 1)[0]
    neg = rng.choice(np.where(y == 0)[0], size=2 * len(pos), replace=False)
    sub = np.concatenate([pos, neg])
    clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, max_features='sqrt',
                                 class_weight='balanced_subsample', n_jobs=14, random_state=7)
    clf.fit(X[sub], y[sub])
    return clf


def strat(j, tag, out):
    j = j[j.Q1.isin(['是', '否'])].copy()
    j['yes'] = (j.Q1 == '是').astype(int)
    bins = pd.cut(j.p_treecrop, [0, 0.3, 0.5, 0.7, 1.0])
    g = j.groupby(bins, observed=False).agg(n=('yes', 'size'), yes=('yes', 'sum'))
    g['rate'] = (g.yes / g.n).round(3)
    out[tag] = {str(k): {'n': int(r.n), 'yes': int(r.yes), 'rate': (None if pd.isna(r.rate) else float(r.rate))}
                for k, r in g.iterrows()}
    print('\n== %s（n=%d）判是率 × p ==' % (tag, len(j)))
    print(g.to_string())
    return out


def main():
    clf = train_model()
    out = {}
    # 判读集（带 p）
    rub_j = pd.concat([pd.read_csv(os.path.join(ORD, 'judge_r2', 'judge_r2_rubber_0%d.csv' % k),
                                   encoding='utf-8-sig') for k in range(5)], ignore_index=True)
    tea_j = pd.concat([pd.read_csv(os.path.join(ORD, 'judge_r2', 'judge_r2_tea_0%d.csv' % k),
                                   encoding='utf-8-sig') for k in range(5)], ignore_index=True)
    jp = pd.read_csv(os.path.join(ORD, 'judge_points_r2.csv'), encoding='utf-8-sig')
    for tag, j in [('rubber判读', rub_j), ('tea判读', tea_j)]:
        j = j.merge(jp[['point_id', 'lon', 'lat']], on='point_id', how='left')
        m = j.assign(lon6=j.lon.round(6), lat6=j.lat.round(6))
        # 接 p：用对应候选集
        src = 'rubber' if 'rubber' in tag else 'tea'
        sc = pd.read_parquet(os.path.join(ORD, 'scored_%s.parquet' % src))
        sc = sc.assign(lon6=sc.lon.round(6), lat6=sc.lat.round(6))
        mm = m.merge(sc[['lon6', 'lat6', 'p_treecrop']], on=['lon6', 'lat6'], how='inner')
        strat(mm, tag, out)
    # GEM 判读
    gj = pd.read_csv(os.path.join(ORD, 'gem_judge_merged.csv'), encoding='utf-8-sig')
    gf = pd.read_parquet(os.path.join(ORD, 'gem', 'feats_y2023.parquet'))
    gf = gf.assign(lon6=gf.lon.round(6), lat6=gf.lat.round(6))
    gf['p_treecrop'] = clf.predict_proba(gf[VC.FEATS].to_numpy('float32'))[:, 1]
    gm = gj.assign(lon6=gj.lon.round(6), lat6=gj.lat.round(6)).merge(gf[['lon6', 'lat6', 'p_treecrop']],
                                                                     on=['lon6', 'lat6'], how='inner')
    strat(gm, 'GEM判读(208)', out)
    # 上一轮 judge2
    j2 = pd.read_csv(os.path.join(ORD, 'judge2_merged.csv'), encoding='utf-8-sig')
    feats = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(ORD, 'feats', 'feat_*_p*.parquet')))],
                      ignore_index=True)
    feats = feats.assign(lon6=feats.lon.round(6), lat6=feats.lat.round(6))
    jm = j2.assign(lon6=j2.lon.round(6), lat6=j2.lat.round(6)).merge(feats[['lon6', 'lat6'] + VC.FEATS],
                                                                     on=['lon6', 'lat6'], how='inner')
    jm['p_treecrop'] = clf.predict_proba(jm[VC.FEATS].to_numpy('float32'))[:, 1]
    strat(jm, 'judge2(上一轮480)', out)
    VC.jsave(out, os.path.join(RES, 'strat_verify.json'))


if __name__ == '__main__':
    main()
