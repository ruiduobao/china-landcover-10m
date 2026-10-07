# -*- coding: utf-8 -*-
"""
sep_probe.py — 可分性探针（纯本地，不碰 GEE）
问：91/140/52/11/62 在现有 64 维 AEF 嵌入里到底"不可分"，还是"可分但被建模方式压住"？
做法：对每个失败类，取其与主要混淆伙伴的点，在 64 维嵌入上做 5 折交叉验证的二分类 RF，
      报 AUC / 召回率@0.5 / 最优阈值 F1。
判读：
  AUC>=0.90  → 可分，问题在建模（层级/权重/阈值）→ 用廉价方案即可修复
  AUC<=0.75  → 不可分，必须加新特征
输出: F:/lc_work/sep_probe.json
"""
import os, sys, json, time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, recall_score

SUB = r'Z:/地理所/论文/中国土地覆盖数据_2017-2024/数据/本地处理/全国清洗训练/年度子集_含稀有类/r7_train_2022.parquet'
OUT = r'F:/lc_work/sep_probe.json'
FEATS = [f'A{i:02d}' for i in range(64)]
NJ = 15
YEAR = 2022

# (失败类, 混淆伙伴, 说明)
TESTS = [
    (52, [51],            '疏闭常绿阔叶 vs 郁闭常绿阔叶'),
    (62, [61],            '疏闭落叶阔叶 vs 郁闭落叶阔叶'),
    (91, [71],            '郁闭针阔混交 vs 郁闭常绿针叶'),
    (91, [61],            '郁闭针阔混交 vs 郁闭落叶阔叶'),
    (92, [82],            '疏闭针阔混交 vs 疏闭落叶针叶'),
    (140, [150],          '地衣苔藓 vs 稀疏植被'),
    (140, [201],          '地衣苔藓 vs 裸地'),
    (11, [10],            '乔灌园地 vs 草本旱地'),
    (11, [121],           '乔灌园地 vs 落叶灌丛'),
    (180, [181],          '木本沼泽 vs 草本沼泽（对照，F1=0.64）'),
    (72, [71],            '疏闭常绿针叶 vs 郁闭常绿针叶（对照，F1=0.66）'),
    (51, [61],            '郁闭常绿阔叶 vs 郁闭落叶阔叶（对照，F1 高）'),
]


def probe(df, pos, neg):
    sub = df[df.class_new.isin([pos, neg])]
    X = sub[FEATS].to_numpy(np.float32)
    y = (sub.class_new.to_numpy() == pos).astype(np.int8)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos < 30 or n_neg < 30:
        return None
    oof = np.zeros(len(y), np.float32)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    for tr, te in skf.split(X, y):
        rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                    class_weight='balanced_subsample',
                                    n_jobs=NJ, random_state=13)
        rf.fit(X[tr], y[tr])
        oof[te] = rf.predict_proba(X[te])[:, 1]
    auc = float(roc_auc_score(y, oof))
    thr = float(np.quantile(oof, 1 - n_pos / len(y)))     # 使预测正例数=真实正例数
    r05 = float(recall_score(y, (oof >= 0.5).astype(int), zero_division=0))
    # 最优 F1 阈值
    best, bt = -1.0, 0.5
    for t in np.arange(0.05, 0.96, 0.05):
        f = f1_score(y, (oof >= t).astype(int), zero_division=0)
        if f > best:
            best, bt = f, float(t)
    return {'auc': round(auc, 4), 'n_pos': n_pos, 'n_neg': n_neg,
            'floor_recall': round(float(recall_score(y, (oof >= thr).astype(int), zero_division=0)), 4),
            'recall@0.5': round(r05, 4), 'bestF1': round(float(best), 4),
            'best_thr': round(bt, 2)}


def main():
    t0 = time.time()
    print('载入', os.path.basename(SUB), flush=True)
    df = pd.read_parquet(SUB, columns=['class_new'] + FEATS)
    print('  %d 行 × %d 特征' % (len(df), len(FEATS)), flush=True)
    rep = {'year': YEAR, 'file': SUB, 'tests': []}
    print(f"\n{'检验':<34}{'n+':>7}{'n-':>8}{'AUC':>8}{'等量召回':>9}{'R@0.5':>8}{'最优F1':>8}")
    for pos, negs, note in TESTS:
        for neg in negs:
            r = probe(df, pos, neg)
            if r is None:
                print(f'{note:<34} 点数不足，跳过'); continue
            r.update({'pos': pos, 'neg': neg, 'note': note})
            rep['tests'].append(r)
            print(f"{note:<34}{r['n_pos']:>7}{r['n_neg']:>8}{r['auc']:>8.3f}"
                  f"{r['floor_recall']:>9.3f}{r['recall@0.5']:>8.3f}{r['bestF1']:>8.3f}", flush=True)
    rep['elapsed_s'] = round(time.time() - t0, 1)
    json.dump(rep, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n判读：AUC≥0.90=可分(建模问题)；0.75–0.90=勉强；≤0.75=不可分(需加特征)')
    print('输出:', OUT)


if __name__ == '__main__':
    main()
