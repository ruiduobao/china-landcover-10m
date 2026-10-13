# -*- coding: utf-8 -*-
"""o34_v6_vs_mk.py — 对照实验：v6 样本 vs 旧 mk 样本，谁训练的判别器更像"人工判读的园地"

* 输入：判读并集（judge2 454 + gem 206 + rubber 155 + tea 157 + apple Bing 158，去重）
        v6 训练正样本 = v6_推荐训练集（橡胶A/B+茶+苹果+多源）去重、剔除全部判读点
        mk 训练正样本 = feats frame=mk_orchard，剔除判读点
        训练负样本 = 大池 feats 中非 mk、非 v6、非判读点的随机 20k
* 规则源（**预注册**）：
  AEF64 + RF(400, sqrt, balanced)；5 折 CV（按坐标分组由判读点天然离散）；
  指标：在判读并集上的 AUC（总）与分来源 AUC（rubber/tea/apple/gem/老轮）
* 输出：results/orchard/v6_vs_mk.json + 打印表
* 用法：python o34_v6_vs_mk.py
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
from sklearn.metrics import roc_auc_score

WORK = r'F:/lc_work/v31_exp'
ORD = os.path.join(WORK, 'data', 'orchard')
RES = os.path.join(WORK, 'results', 'orchard')


def load_feats(pattern, subdir='feats'):
    fs = sorted(glob.glob(os.path.join(ORD, subdir, pattern)))
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def r6(d):
    d = d.copy()
    d['lon6'] = d.lon.round(6)
    d['lat6'] = d.lat.round(6)
    return d


def main():
    # ---- 判读并集 ----
    parts = []
    j2 = pd.read_csv(os.path.join(ORD, 'judge2_merged.csv'), encoding='utf-8-sig')
    parts.append(j2[j2.Q1.isin(['是', '否'])][['point_id', 'lon', 'lat', 'Q1']].assign(src='judge2'))
    gj = pd.read_csv(os.path.join(ORD, 'gem_judge_merged.csv'), encoding='utf-8-sig')
    parts.append(gj[gj.Q1.isin(['是', '否'])][['point_id', 'lon', 'lat', 'Q1']].assign(src='gem'))
    for tag, pat in [('rubber', 'judge_r2_rubber_*.csv'), ('tea', 'judge_r2_tea_*.csv')]:
        j = pd.concat([pd.read_csv(f, encoding='utf-8-sig') for f in
                       sorted(glob.glob(os.path.join(ORD, 'judge_r2', pat)))], ignore_index=True)
        jp = pd.read_csv(os.path.join(ORD, 'judge_points_r2.csv'), encoding='utf-8-sig')
        j = j.merge(jp[['point_id', 'lon', 'lat']], on='point_id')
        parts.append(j[j.Q1.isin(['是', '否'])][['point_id', 'lon', 'lat', 'Q1']].assign(src=tag))
    ab = pd.concat([pd.read_csv(os.path.join(ORD, 'aa_judge', 'judge_aabing_0%d.csv' % k), encoding='utf-8-sig')
                    for k in range(5)], ignore_index=True)
    ap = pd.read_csv(os.path.join(ORD, 'aa_judge_points.csv'), encoding='utf-8-sig')
    ab = ab.merge(ap[['point_id', 'lon', 'lat']], on='point_id')
    parts.append(ab[ab.Q1.isin(['是', '否'])][['point_id', 'lon', 'lat', 'Q1']].assign(src='apple'))
    jud = pd.concat(parts, ignore_index=True).drop_duplicates(subset=['lon', 'lat'])
    jud = r6(jud)
    jud['y'] = (jud.Q1 == '是').astype(int)
    print('判读并集 %d（正 %d）；分来源 %s' % (len(jud), jud.y.sum(), jud.groupby('src').y.agg(['size', 'sum']).to_dict('index')))

    # ---- 特征：判读点 ----
    FE = VC.FEATS
    fsets = []
    fsets.append(load_feats('feat_*_p*.parquet')[['lon', 'lat'] + FE])
    fsets.append(load_feats('*_p*.parquet', 'aef')[['lon', 'lat'] + FE])
    gfs = pd.read_parquet(os.path.join(ORD, 'gem', 'feats_y2023.parquet'))[['lon', 'lat'] + FE]
    fsets.append(gfs)
    ALLFE = pd.concat(fsets, ignore_index=True)
    ALLFE = r6(ALLFE).drop_duplicates(subset=['lon6', 'lat6'])
    jud = jud.merge(ALLFE[['lon6', 'lat6'] + FE], on=['lon6', 'lat6'], how='left')
    jud = jud[jud[FE].notna().all(axis=1)].reset_index(drop=True)
    print('判读点可配特征 %d' % len(jud))

    # ---- v6 正样本 ----
    v6 = pd.read_csv(r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12/乔灌园地/v6_含苹果/乔灌园地v6_推荐训练集.csv',
                     encoding='utf-8-sig')
    v6 = r6(v6).drop_duplicates(subset=['lon6', 'lat6'])
    v6 = v6.merge(ALLFE[['lon6', 'lat6'] + FE], on=['lon6', 'lat6'], how='left')
    v6 = v6[v6[FE].notna().all(axis=1)]
    v6 = v6[~v6.set_index(['lon6', 'lat6']).index.isin(jud.set_index(['lon6', 'lat6']).index)]
    print('v6 正样本 %d' % len(v6))

    # ---- mk 正样本 ----
    pool = load_feats('feat_*_p*.parquet')
    pool = r6(pool)
    mk = pool[pool.frame == 'mk_orchard'].drop_duplicates(subset=['lon6', 'lat6'])
    mk = mk[~mk.set_index(['lon6', 'lat6']).index.isin(jud.set_index(['lon6', 'lat6']).index)]
    print('mk 正样本 %d' % len(mk))

    # ---- 负样本（硬负：池内非 mk、非 v6、非判读） ----
    rng = np.random.default_rng(7)
    neg = pool[(pool.frame != 'mk_orchard')].drop_duplicates(subset=['lon6', 'lat6'])
    neg = neg[~neg.set_index(['lon6', 'lat6']).index.isin(jud.set_index(['lon6', 'lat6']).index)]
    neg = neg[~neg.set_index(['lon6', 'lat6']).index.isin(v6.set_index(['lon6', 'lat6']).index)]
    neg = neg.sample(n=min(20000, len(neg)), random_state=7)
    print('负样本 %d' % len(neg))

    # ---- 训练与评估 ----
    out = {}
    for name, pos in [('M_mk(旧样本)', mk), ('M_v6(v6样本)', v6)]:
        X = np.vstack([pos[FE].to_numpy('float32'), neg[FE].to_numpy('float32')])
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg))].astype(int)
        clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, max_features='sqrt',
                                     class_weight='balanced_subsample', n_jobs=14, random_state=7)
        clf.fit(X, y)
        jud['p_%s' % name] = clf.predict_proba(jud[FE].to_numpy('float32'))[:, 1]
        auc_all = roc_auc_score(jud.y, jud['p_%s' % name])
        per = {}
        for s, g in jud.groupby('src'):
            if g.y.nunique() == 2 and len(g) >= 20:
                per[s] = dict(n=int(len(g)), pos=int(g.y.sum()),
                              AUC=round(float(roc_auc_score(g.y, g['p_%s' % name])), 4))
        out[name] = dict(train_pos=int(len(pos)), train_neg=int(len(neg)),
                         AUC_all=round(float(auc_all), 4), per=per)
        print('\n== %s == 训练 %d 正 / %d 负 | 判读并集 AUC=%.4f' % (name, len(pos), len(neg), auc_all))
        for s, v in per.items():
            print('   %-8s n=%3d pos=%3d AUC=%.3f' % (s, v['n'], v['pos'], v['AUC']))
    VC.jsave(out, os.path.join(RES, 'v6_vs_mk.json'))
    jud.to_csv(os.path.join(ORD, 'v6_vs_mk_scored.csv'), index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
