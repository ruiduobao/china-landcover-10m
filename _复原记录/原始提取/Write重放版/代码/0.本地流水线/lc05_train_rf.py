# -*- coding: utf-8 -*-
"""
lc05_train_rf.py — 本地训练随机森林（06文档 §2）
* 真实数据：样本 Parquet（A00..A63 + class），来自GEE sampleRegions导出后本地化。
* 自测(合成)：--synthetic 用合成多分类数据验证管线。
运行: python lc05_train_rf.py --parquet data/samples.parquet --out models/rf
      python lc05_train_rf.py --synthetic --out models/synth_rf
"""
import os, json, argparse, numpy as np, pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from lc_conf import TARGET_FEATURES, OUT

def make_synthetic(n=20000, seed=42):
    """合成数据：6类、64特征、不同均值高斯，模拟嵌入空间语义可分性"""
    rng = np.random.default_rng(seed)
    Xlist, ylist, tlist = [], [], []
    centers = rng.normal(0, 1, (6, 64))
    for c in range(6):
        mu = centers[c] * 2.0
        n_c = n // 6
        x = rng.normal(mu, 0.6, (n_c, 64))
        Xlist.append(x); ylist.append(np.full(n_c, c + 1)); tlist.append(np.full(n_c, c))
    X = np.vstack(Xlist); y = np.concatenate(ylist)
    # 伪 tile 分组（类内混合、粒度较细），保证分组切分后 train/val 都含全部类
    tiles = np.arange(len(y)) % 40
    return pd.DataFrame(X, columns=TARGET_FEATURES).assign(**{'class': y, 'tile_id': tiles})

def train(df, feats, n_trees=300, n_jobs=4, seed=42):
    X = df[feats].to_numpy(float); y = df['class'].to_numpy(int)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr, va = next(gss.split(X, y, groups=df['tile_id'].to_numpy()))
    Xtr, Xva, ytr, yva = X[tr], X[va], y[tr], y[va]
    cw = compute_class_weight('balanced', classes=np.unique(ytr), y=ytr)
    cw_map = dict(zip(np.unique(ytr), cw))
    rf = RandomForestClassifier(n_estimators=n_trees, max_features='sqrt',
                                min_samples_leaf=5, class_weight=cw_map,
                                oob_score=True, n_jobs=n_jobs, random_state=seed)
    rf.fit(Xtr, ytr)
    print('OOB Score:', round(rf.oob_score_, 4))
    print('独立验证集:')
    print(classification_report(yva, rf.predict(Xva), digits=3))
    return rf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet'); ap.add_argument('--out', default='models/rf')
    ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--n_trees', type=int, default=300)
    a = ap.parse_args()

    if a.synthetic:
        df = make_synthetic(); feats = TARGET_FEATURES; note = 'synthetic'
    else:
        df = pd.read_parquet(a.parquet); feats = TARGET_FEATURES; note = 'real'
    # 只保留30类内(或数据已有的)特征与类列
    df = df[[*feats, 'class', 'tile_id']].dropna()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    rf = train(df, feats, n_trees=a.n_trees)
    joblib.dump(rf, a.out + '.joblib')
    cfg = {'features': feats, 'classes': rf.classes_.tolist(), 'note': note}
    json.dump(cfg, open(a.out + '_cfg.json', 'w'))
    print('模型:', a.out + '.joblib')
    print('配置:', a.out + '_cfg.json')

if __name__ == '__main__':
    main()
