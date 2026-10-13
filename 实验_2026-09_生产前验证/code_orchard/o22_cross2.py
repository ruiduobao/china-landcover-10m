# -*- coding: utf-8 -*-
"""o22_cross2.py — 新判别器（GEM重训）交叉核验：历次判读点 × p_treecrop

* 输入：data/orchard/{judge2_merged.csv(480), gem_judge_merged.csv(208)} + feats/*.parquet（2023 AEF）
        GEM 训练集（重训模型配方同 o20）
* 规则源：新模型打分历次判读点；交叉表 = 判读结论 × p_treecrop（分 区带/来源）
* 输出：results/orchard/cross2.json + data/orchard/judge_all_scored.csv
* 用法：python o22_cross2.py
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


def main():
    clf = train_model()
    feats = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(ORD, 'feats', 'feat_*_p*.parquet')))],
                      ignore_index=True)
    f2 = feats[['lon', 'lat'] + VC.FEATS].round({'lon': 6, 'lat': 6})
    out = {}
    parts = []
    gemf = pd.read_parquet(os.path.join(ORD, 'gem', 'feats_y2023.parquet'))
    gf2 = gemf[['lon', 'lat'] + VC.FEATS].round({'lon': 6, 'lat': 6})
    for name, fp, zone_col, fsrc in [('judge2(两轮480)', 'judge2_merged.csv', None, f2),
                                     ('gem(208)', 'gem_judge_merged.csv', 'zone', gf2)]:
        j = pd.read_csv(os.path.join(ORD, fp), encoding='utf-8-sig')
        j = j[j.Q1.isin(['是', '否'])].copy()
        j['src'] = name
        jm = j.round({'lon': 6, 'lat': 6}).merge(fsrc, on=['lon', 'lat'], how='inner')
        jm['p_treecrop'] = clf.predict_proba(jm[VC.FEATS].to_numpy('float32'))[:, 1]
        parts.append(jm)
        out[name] = {'n': int(len(jm)),
                     'yes_p_med': round(float(jm[jm.Q1 == '是'].p_treecrop.median()), 3) if (jm.Q1 == '是').any() else None,
                     'no_p_med': round(float(jm[jm.Q1 == '否'].p_treecrop.median()), 3) if (jm.Q1 == '否').any() else None}
        if zone_col and zone_col in jm.columns:
            for z, g in jm.groupby(zone_col):
                out[name + '|' + str(z)] = {'n': int(len(g)), 'yes': int((g.Q1 == '是').sum()),
                                            'yes_p_med': round(float(g[g.Q1 == '是'].p_treecrop.median()), 3) if (g.Q1 == '是').any() else None,
                                            'no_p_med': round(float(g[g.Q1 == '否'].p_treecrop.median()), 3) if (g.Q1 == '否').any() else None}
    allj = pd.concat(parts, ignore_index=True)
    allj.to_csv(os.path.join(ORD, 'judge_all_scored.csv'), index=False, encoding='utf-8-sig')
    print(json.dumps(out, ensure_ascii=False, indent=1))
    # 关键交叉表：判读 × p 阈值
    for name in [p for p in out if '|' not in p]:
        jm = allj[allj.src == name]
        ct = pd.crosstab(jm.Q1, pd.cut(jm.p_treecrop, [0, 0.3, 0.5, 0.7, 1.0]))
        print('\n== %s 判读 × p_treecrop ==' % name)
        print(ct.to_string())
    VC.jsave(out, os.path.join(RES, 'cross2.json'))


if __name__ == '__main__':
    main()
