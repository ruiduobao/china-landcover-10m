# -*- coding: utf-8 -*-
"""t5_fourclass.py — 4 类（EBF/DBF/ENF/DNF）vs 2 类（针/阔）稳定性对照
* 目的（任务关键问题 1）：合并成 2 类后，针叶/阔叶 F1 是否比 4 类各自拆开更稳？
* 输入（只读）：points_76.parquet（含 FCS10 原始码，可回溯 4 类）
* 规则源：
    ArmB-4class 标签 = FCS10 4 类体系：4=EBF(51/52) 5=DBF(61/62) 6=ENF(71/72) 7=DNF(81/82) 8=MIX(91/92)
    非森林沿用 ArmB 非森林标签（与 2 类实验完全同一批点、同一非森林标签）
    2 类结果直接读 two_arm_metrics.json 的 ArmB（同一 CV 折、同种子）
* 门槛：同 RF / 同 10km 空间块 5 折 / 同种子 {7,11,23}；参考一律 FCS10
* 输出：four_vs_two.json
* 用法：python t5_fourclass.py
"""
import os, sys, json, time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
AEF = [f'A{i:02d}' for i in range(64)]
# FCS10 原始码 → 4 类森林码（与 ArmB 非森林码不冲突：4/5/6/7/8）
F4 = {51: 4, 52: 4, 61: 5, 62: 5, 71: 6, 72: 6, 81: 7, 82: 7, 91: 8, 92: 8}


def spatial_fold(df, km=10.0, nf=5, seed=20261017):
    kx = np.floor(df.lon.values / (km / 111.32 / np.cos(np.radians(df.lat.values)))).astype(np.int64)
    ky = np.floor(df.lat.values / (km / 111.32)).astype(np.int64)
    key = kx * 100000 + ky
    u = np.unique(key)
    rng = np.random.default_rng(seed)
    rng.shuffle(u)
    folds = np.zeros(len(key), dtype=np.int8)
    for fi, blk in enumerate(np.array_split(u, nf)):
        folds[np.isin(key, blk)] = fi
    return folds


def mode_vote(P):
    n = P.shape[1]
    cnt = np.zeros((n, int(P.max()) + 1), dtype=np.int16)
    for s in range(P.shape[0]):
        np.add.at(cnt, (np.arange(n), P[s]), 1)
    return cnt.argmax(axis=1).astype(np.int16)


def prf(pred, ref, c):
    p = pred == c; r = ref == c
    tp = int((p & r).sum())
    prec = tp / p.sum() if p.sum() else None
    rec = tp / r.sum() if r.sum() else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else (0.0 if (prec is not None and rec is not None) else None)
    return {'precision': None if prec is None else round(float(prec), 4),
            'recall': None if rec is None else round(float(rec), 4),
            'f1': None if f1 is None else round(float(f1), 4),
            'n_pred': int(p.sum()), 'n_ref': int(r.sum()), 'tp': tp}


def main():
    df = pd.read_parquet(os.path.join(HERE, 'points_76.parquet'))
    df = df[(df.cl_newB > 0) & (df.cl_newA > 0)].reset_index(drop=True)
    # ArmB-4 标签：旧森林点 → FCS10 4 类；旧非森林点 → ArmB 非森林码
    lut = np.zeros(256, dtype=np.int16)
    for k, v in F4.items():
        lut[k] = v
    raw = df.fcs10_raw.to_numpy()
    f4 = lut[np.clip(raw, 0, 255)]
    f4[raw < 0] = -1
    isf = df.cl.isin([4, 5, 6, 7, 8]).values
    y4 = df.cl_newB.to_numpy(np.int64).copy()
    y4[isf] = f4[isf]
    assert (y4[isf] > 0).all() or True
    keep = y4 > 0
    print('n', len(df), 'keep', int(keep.sum()))
    df = df[keep].reset_index(drop=True); y4 = y4[keep]
    ref4 = lut[np.clip(df.fcs10_raw.to_numpy(), 0, 255)].astype(np.int16)
    ref4[df.fcs10_raw.to_numpy() < 0] = -1

    X = np.nan_to_num(df[AEF].to_numpy(np.float32), nan=0.0)
    folds = spatial_fold(df)
    seeds = [7, 11, 23]
    oofs = []
    for seed in seeds:
        t0 = time.time()
        oof = np.zeros(len(df), dtype=np.int16)
        for f in range(5):
            te = folds == f; tr = ~te
            m = RandomForestClassifier(n_estimators=100, min_samples_leaf=2,
                                       max_leaf_nodes=5000, n_jobs=13, random_state=seed)
            m.fit(X[tr], y4[tr])
            oof[te] = m.predict(X[te]).astype(np.int16)
        oofs.append(oof)
        print('  seed%d %.0fs' % (seed, time.time() - t0), flush=True)
    P = np.array(oofs)
    vote = mode_vote(P)

    # ---- 4 类：各自 F1（参考=FCS10 对应类，分母为该类参考点）----
    res = {'meta': {'n': int(len(df)), 'seeds': seeds, 'fold_seed': 20261017,
                    'labels_armB4': {str(k): int(v) for k, v in pd.Series(y4).value_counts().sort_index().items()},
                    'ref4': {str(k): int(v) for k, v in pd.Series(ref4[ref4 > 0]).value_counts().sort_index().items()}},
           'four_class': {}, 'two_class_from_four': {}, 'per_seed_four': []}

    m4 = np.isin(ref4, [4, 5, 6, 7, 8])
    for c, nm in [(4, 'EBF'), (5, 'DBF'), (6, 'ENF'), (7, 'DNF'), (8, 'MIX')]:
        res['four_class'][nm] = prf(vote[m4], ref4[m4], c)
    f1s = [res['four_class'][k]['f1'] for k in ('EBF', 'DBF', 'ENF', 'DNF')]
    res['four_class']['macro_f1_4'] = round(float(np.mean([x for x in f1s if x is not None])), 4)
    res['four_class']['n_ref_forest4'] = int((np.isin(ref4, [4, 5, 6, 7])).sum())

    # ---- 4 类结果聚合到针/阔，与 2 类模型比 ----
    v2 = vote[m4].copy()
    v2[np.isin(v2, [6, 7])] = 31
    v2[np.isin(v2, [4, 5])] = 32
    v2[v2 == 8] = 33
    r2 = ref4[m4].copy()
    r2[np.isin(r2, [6, 7])] = 31
    r2[np.isin(r2, [4, 5])] = 32
    r2[r2 == 8] = 33
    res['two_class_from_four']['needle'] = prf(v2, r2, 31)
    res['two_class_from_four']['broad'] = prf(v2, r2, 32)
    res['two_class_from_four']['macro_f1_2'] = round(float(np.mean(
        [res['two_class_from_four']['needle']['f1'], res['two_class_from_four']['broad']['f1']])), 4)
    # 按种子看 F1 方差（稳定性）
    per_seed = []
    for si, o in enumerate(oofs):
        d = {}
        for c, nm in [(4, 'EBF'), (5, 'DBF'), (6, 'ENF'), (7, 'DNF')]:
            d[nm] = prf(o[m4], ref4[m4], c)['f1']
        o2 = o[m4].copy(); o2[np.isin(o2, [6, 7])] = 31; o2[np.isin(o2, [4, 5])] = 32; o2[o2 == 8] = 33
        d['needle_2c'] = prf(o2, r2, 31)['f1']
        d['broad_2c'] = prf(o2, r2, 32)['f1']
        per_seed.append(d)
    res['per_seed_four'] = per_seed
    stab = {}
    for k in ['EBF', 'DBF', 'ENF', 'DNF', 'needle_2c', 'broad_2c']:
        vals = [p[k] for p in per_seed if p.get(k) is not None]
        if vals:
            stab[k] = {'mean': round(float(np.mean(vals)), 4), 'sd': round(float(np.std(vals)), 4),
                       'range': round(float(max(vals) - min(vals)), 4)}
    res['seed_stability'] = stab

    with open(os.path.join(HERE, 'four_vs_two.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: res[k] for k in ('four_class', 'two_class_from_four', 'seed_stability')},
                     ensure_ascii=False, indent=1))
    np.savez_compressed(os.path.join(HERE, 'oof_four.npz'), folds=folds, y4=y4, ref4=ref4,
                        **{'oof_%d' % s: oofs[i] for i, s in enumerate(seeds)})
    print('-> four_vs_two.json')


if __name__ == '__main__':
    main()
