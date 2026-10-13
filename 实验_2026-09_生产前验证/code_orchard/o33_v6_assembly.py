# -*- coding: utf-8 -*-
"""o33_v6_assembly.py — v6 装配：苹果子集入列 + 儋州县级验证表 + 推荐训练集

* 输入：data/orchard/aomc_apple_points.parquet + aef/aomc_p*.parquet（30k AEF64）
        GEM 重训模型（配方同 o20）给苹果点打分（OOD 参考）
        儋州验证结果（本次内联计算）
* 输出：交付目录 v6_含苹果/：苹果样本 parquet/csv、推荐训练集 v6、儋州验证表、stats
* 用法：python o33_v6_assembly.py
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
D5 = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12/乔灌园地/v5_精化样本'
DEST = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12/乔灌园地/v6_含苹果'
GEM_TRAIN = r'F:/lc_work/外部样本/GEM_Forest/training_with_AEF.csv'


def main():
    os.makedirs(DEST, exist_ok=True)
    # 苹果样本
    d = pd.read_parquet(os.path.join(ORD, 'aomc_apple_points.parquet'))
    aef = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(ORD, 'aef', 'aomc_p*.parquet')))],
                    ignore_index=True)
    d['lon6'] = d.lon.round(6); d['lat6'] = d.lat.round(6)
    aef['lon6'] = aef.lon.round(6); aef['lat6'] = aef.lat.round(6)
    d = d.merge(aef[['lon6', 'lat6'] + VC.FEATS], on=['lon6', 'lat6'], how='left').drop(columns=['lon6', 'lat6'])
    # GEM 模型打分（OOD 参考）
    g = pd.read_csv(GEM_TRAIN, encoding='utf-8-sig')
    g['y'] = g['forest'].isin([2, 3]).astype(int)
    X = g[VC.FEATS].to_numpy('float64')
    ok = ~np.isnan(X).any(axis=1)
    clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, max_features='sqrt',
                                 class_weight='balanced_subsample', n_jobs=14, random_state=7)
    rng = np.random.default_rng(7)
    pos = np.where(g.y.to_numpy()[ok] == 1)[0]
    neg = rng.choice(np.where(g.y.to_numpy()[ok] == 0)[0], size=2 * len(pos), replace=False)
    sub = np.concatenate([pos, neg])
    clf.fit(X[ok][sub], g.y.to_numpy()[ok][sub])
    d['p_treecrop'] = clf.predict_proba(d[VC.FEATS].to_numpy('float32'))[:, 1]
    d['subtype'] = '苹果'
    d['grade'] = 'B_产品+面积校准'
    d.to_parquet(os.path.join(DEST, '乔灌园地v6_苹果样本.parquet'), index=False)
    d.drop(columns=VC.FEATS).to_csv(os.path.join(DEST, '乔灌园地v6_苹果样本_精简.csv'),
                                    index=False, encoding='utf-8-sig')
    print('苹果 %d（AEF 命中 %d；p≥0.5 %d）' % (len(d), int(d[VC.FEATS].notna().all(axis=1).sum()),
                                            int((d.p_treecrop >= 0.5).sum())))
    # 推荐训练集 v6 = v5 推荐 + 苹果
    rec = pd.read_csv(os.path.join(D5, '乔灌园地v5_推荐训练集.csv'), encoding='utf-8-sig')
    ap = d[['lon', 'lat', 'prov', 'subtype', 'grade']].assign(label='乔灌园地')
    allp = pd.concat([rec, ap], ignore_index=True).drop_duplicates(subset=['lon', 'lat'])
    allp.to_csv(os.path.join(DEST, '乔灌园地v6_推荐训练集.csv'), index=False, encoding='utf-8-sig')
    print('推荐训练集 v6:', len(allp), allp.grade.value_counts().to_dict())
    # 沿用 v5 其余文件
    import shutil
    for f in os.listdir(D5):
        if f.startswith('乔灌园地v5_') and '推荐训练集' not in f:
            shutil.copy(os.path.join(D5, f), os.path.join(DEST, f.replace('v5', 'v6')))
    VC.jsave({'apple': int(len(d)), 'train_v6': int(len(allp)),
              'train_grade': allp.grade.value_counts().to_dict()},
             os.path.join(DEST, 'v6_stats.json'))
    print('→', DEST)


if __name__ == '__main__':
    main()
