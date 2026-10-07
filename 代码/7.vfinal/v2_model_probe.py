# -*- coding: utf-8 -*-
"""
model_probe.py — 决策门实验（纯本地，一年 ~2 分钟）
假设：91/140/11 在嵌入空间 AUC 0.95–0.99（可分），但 30 类 argmax 把它们的票摊薄 → 建模问题。
做法：固定 2022 年、固定 holdout 划分，只改"类别权重策略"，看 5 个弱类能否被预测出来，
      同时监控总体 OA / macro-F1 / 精度是否崩溃。
输出: F:/lc_work/model_probe.json + 控制台表
"""
import os, sys, json, time
import numpy as np
import pandas as pd

PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
sys.path.insert(0, os.path.join(PROJ, '代码', '4.全国清洗训练'))
import e6_spatial_eval as E6
from sklearn.ensemble import RandomForestClassifier

OUT = r'F:/lc_work/model_probe.json'
YEAR = 2022
TREES = 150
WEAK = [52, 62, 91, 140, 11]


def scheme_weights(ytr, kind):
    """返回 class_weight 参数"""
    cls, cnt = np.unique(ytr, return_counts=True)
    if kind == 'balanced':
        return 'balanced_subsample'
    base = {int(c): float(len(ytr)) / (len(cls) * n) for c, n in zip(cls, cnt)}
    if kind == 'x3':
        d = dict(base)
        for c in WEAK:
            if c in d: d[c] *= 3.0
        return d
    if kind == 'x8':
        d = dict(base)
        for c in WEAK:
            if c in d: d[c] *= 8.0
        return d
    if kind == 'x8cap':
        d = {int(c): min(v, 12.0) for c, v in base.items()}     # 先封顶抑制极端权重
        for c in WEAK:
            if c in d: d[c] = min(d[c] * 8.0, 40.0)
        return d
    return 'balanced_subsample'


def run(kind, df, val, hold, trees=TREES):
    bk = E6.albers_bk(df.lon.to_numpy(), df.lat.to_numpy())
    tr = df[~np.isin(bk, hold)]
    X = tr[E6.FEATS].to_numpy(np.float32)
    y = tr.class_new.to_numpy(int)
    fin = np.isfinite(X).all(1)
    X, y, tr2 = X[fin], y[fin], tr[fin]
    w = np.nan_to_num(tr2.train_weight.to_numpy(np.float32), nan=1.0, posinf=1.0, neginf=1.0)
    cw = scheme_weights(y, kind)
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=trees, n_jobs=15, random_state=42,
                                min_samples_leaf=2, max_features='sqrt', class_weight=cw)
    rf.fit(X, y, sample_weight=w)
    fit_s = time.time() - t0

    v = val.merge(df[['row_id'] + E6.FEATS], on='row_id', how='inner')
    VX = v[E6.FEATS].to_numpy(np.float32)
    P = rf.predict(VX)
    labels = sorted(set(v.class_new.unique()) | set(np.unique(P)))
    ix = {c: i for i, c in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), np.int64)
    for t, p in zip(v.class_new.to_numpy(int), P):
        cm[ix[t], ix[p]] += 1
    met, per = E6.metrics_from_cm(cm, labels)
    row = {'scheme': kind, 'fit_s': round(fit_s, 1),
           'OA': round(met['OA'], 4), 'macro_F1': round(met['macro_F1'], 4),
           'bal_acc': round(met['balanced_accuracy'], 4)}
    for c in WEAK:
        r = per[per['class'] == c]
        row[f'c{c}'] = (int(r.n_pred.iloc[0]) if len(r) else -1,
                        round(float(r.F1.iloc[0]), 3) if len(r) else -1)
    return row, cm, labels, per


def main():
    print('载入年度子集 2022 …', flush=True)
    df = pd.read_parquet(os.path.join(E6.SUB_DIR, f'r7_train_{YEAR}.parquet'),
                         columns=['row_id', 'lon', 'lat', 'class_new', 'train_weight'] + E6.FEATS)
    print(f'  {len(df):,} 行', flush=True)
    hold = np.asarray(json.load(open(E6.HOLDOUT))['block_keys'], dtype=np.int64)
    val = E6.load_val_ids()
    rep = {'year': YEAR, 'runs': []}
    print(f"\n{'方案':<12}{'OA':>8}{'macroF1':>9}{'balAcc':>8}   " +
          '  '.join(f'{c}(pred,F1)' for c in WEAK))
    for kind in ['balanced', 'x3', 'x8', 'x8cap']:
        row, cm, labels, per = run(kind, df, val, hold)
        rep['runs'].append(row)
        desc = '  '.join(f"{row[f'c{c}'][0]:>5},{row[f'c{c}'][1]:.2f}" for c in WEAK)
        print(f"{kind:<12}{row['OA']:>8.4f}{row['macro_F1']:>9.4f}{row['bal_acc']:>8.4f}   {desc}", flush=True)
    json.dump(rep, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
    print('\n输出:', OUT)


if __name__ == '__main__':
    main()
