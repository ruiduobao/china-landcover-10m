# -*- coding: utf-8 -*-
"""
lc07_verify.py — 本地复现 GEE decisionTreeEnsemble 语义，校验 sklearn 与转换一致（06文档 §7）
* 用纯 Python 逐树遍历转换后的 dict，实现与 GEE 相同的判定：
      特征值 <= value -> 左子树，否则右子树；叶子 label 取多数树投票（argmax）。
* 关键：与 sklearn predict 对比不一致率应=0（浮点阈值边界允许<0.1%）。
运行: python lc07_verify.py --model models/rf.joblib  --cfg models/rf_cfg.json --trees models/rf_gee_trees.json --data data/verify_points.parquet
      （合成数据的points可现场生成）
"""
import os, json, argparse, numpy as np, pandas as pd, joblib
from collections import Counter

def predict_trees(trees, feats, X):
    """复现 GEE decisionTreeEnsemble 的大多数投票预测"""
    n = X.shape[0]; votes = [None] * len(trees)
    for ti, t in enumerate(trees):
        lab = np.empty(n, dtype=int)
        # 逐点遍历（用栈模拟递归，避免深度限制）
        for i in range(n):
            node = t
            while 'label' not in node:
                f = node['featureIndex']
                node = node['left'] if X[i, f] <= node['value'] else node['right']
            lab[i] = node['label']
        votes[ti] = lab
    votes_m = np.vstack(votes)
    pred = np.array([Counter(votes_m[:, i]).most_common(1)[0][0] for i in range(n)])
    return pred

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--cfg', required=True)
    ap.add_argument('--trees', required=True); ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--parquet'); ap.add_argument('--nsample', type=int, default=4000)
    a = ap.parse_args()
    rf = joblib.load(a.model); cfg = json.load(open(a.cfg))
    payload = json.load(open(a.trees)); trees = payload.get('trees_mirror_json') or payload['trees']
    feats = cfg['features']
    if a.synthetic:
        # 同 lc05 合成数据，前 nsample 条用于一致性校验
        import sys; sys.path.insert(0, os.path.dirname(__file__))
        from lc05_train_rf import make_synthetic
        df = make_synthetic().head(a.nsample)
    else:
        df = pd.read_parquet(a.parquet).head(a.nsample)
    X = df[feats].to_numpy(float)
    y_sk = rf.predict(X).astype(int)
    y_gee = predict_trees(trees, feats, X)
    mismatch = int(np.sum(y_sk != y_gee))
    rate = mismatch / len(y_sk)
    # 逐类不一致统计
    conf = Counter((int(s), int(g)) for s, g in zip(y_sk, y_gee) if s != g)
    print(f'样本数: {len(y_sk)}  不一致数: {mismatch}  不一致率: {rate:.5f}')
    if rate > 0.001:
        print('逐类不一致(真实,预测):', conf.most_common(10))
    print('[%s] %s' % ('PASS' if rate <= 0.001 else 'FAIL',
                       '转换后预测与sklearn一致(<0.1%边界容忍)' if rate <= 0.001
                       else '不一致率超标，检查特征顺序/类别映射/阈值精度'))

if __name__ == '__main__':
    main()
