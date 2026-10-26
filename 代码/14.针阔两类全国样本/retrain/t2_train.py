# -*- coding: utf-8 -*-
"""t2_train.py — 针阔两类重训：两臂 RF + 10km 空间块 OOF 评估
* 输入（只读）：exp/two_class_retrain/points_76.parquet（t1_prep 产出）
                 exp/two_class_retrain/needle_broad_map.json（冻结码表/判据）
* 规则源（判据先写死，见 needle_broad_map.json → evaluation_contract）：
    ArmA = 旧池 24 类按 from_ours24 合并（仅换类别体系；4/5→32、6/7→31、8→33）
    ArmB = 旧池非森林原样；旧池森林点的去向全部由 GLC_FCS10-2023 决定（→31/32/33 或非森林）
    参考一律 FCS10（严禁 FCS30D）
    RF：100 树 / minLeaf 2 / maxLeafNodes 5000 / seeds {7,11,23} / n_jobs 13
    CV：10 km 空间块 5 折，块划分种子 20261017（两臂同一折）
    坡向 OR：needle(31) vs broad(32)，阴=[315,360)∪[0,45]，阳=[135,225]，slope>5°
* 门槛：G2 |OR|∈[0.8,1.25]；G9 针叶 precision 相对 ArmA 提升且 OR 向 FCS10 靠拢
* 输出：two_arm_metrics.json（每臂×每区域×每瓦×各指标）、oof_preds.npz
* 用法：python t2_train.py [--tiles all|priority] [--seeds 7 11 23]
"""
import os, sys, json, time, argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
AEF = [f'A{i:02d}' for i in range(64)]
FOREST = (31, 32, 33)


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


def or_calc(a_sh, a_su, b_sh, b_su):
    if min(a_sh, a_su, b_sh, b_su) <= 0:
        return None
    return float((a_sh / a_su) / (b_sh / b_su))


def or_ci(a_sh, a_su, b_sh, b_su):
    if min(a_sh, a_su, b_sh, b_su) <= 0:
        return None
    lo = np.log((a_sh * b_su) / (a_su * b_sh))
    se = np.sqrt(1 / a_sh + 1 / a_su + 1 / b_sh + 1 / b_su)
    return [float(np.exp(lo - 1.96 * se)), float(np.exp(lo + 1.96 * se))]


def mode_vote(P):
    """多种子硬预测投票（向量化，避免逐点 bincount）"""
    n = P.shape[1]
    mx = int(P.max()) + 1
    cnt = np.zeros((n, mx), dtype=np.int16)
    for s in range(P.shape[0]):
        np.add.at(cnt, (np.arange(n), P[s]), 1)
    return cnt.argmax(axis=1).astype(np.int16)


def prf(pred, ref, c):
    """单类 precision/recall/F1（pred/ref 为等长类别数组）"""
    p = pred == c
    r = ref == c
    tp = int((p & r).sum())
    prec = tp / p.sum() if p.sum() else None
    rec = tp / r.sum() if r.sum() else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else (0.0 if (prec is not None and rec is not None) else None)
    return {'precision': None if prec is None else round(float(prec), 4),
            'recall': None if rec is None else round(float(rec), 4),
            'f1': None if f1 is None else round(float(f1), 4),
            'n_pred': int(p.sum()), 'n_ref': int(r.sum()), 'tp': tp}


def eval_slice(pred, ref, asp_cls=None, slope=None, tag=''):
    """在给定子集上算针阔指标 + OR"""
    out = {'n': int(len(pred)), 'tag': tag}
    # 参考有效（FCS10 映射成功）
    ok = ref > 0
    out['n_ref_valid'] = int(ok.sum())
    if ok.any():
        out['oa_all22'] = round(float((pred[ok] == ref[ok]).mean()), 4)
    # 纯林参考（31/32）上的针阔 P/R/F1
    m = np.isin(ref, [31, 32])
    out['n_ref_pure_forest'] = int(m.sum())
    if m.any():
        out['needle'] = prf(pred[m], ref[m], 31)
        out['broad'] = prf(pred[m], ref[m], 32)
        f1s = [out['needle']['f1'], out['broad']['f1']]
        out['macro_f1_2'] = round(float(np.mean([x for x in f1s if x is not None])), 4) if any(x is not None for x in f1s) else None
        # 森林三分类（含混交）macro-F1：参考 31/32/33
        m3 = np.isin(ref, [31, 32, 33])
        if m3.any():
            f3 = [prf(pred[m3], ref[m3], c)['f1'] for c in (31, 32, 33)]
            out['macro_f1_3_with_mixed'] = round(float(np.mean([x for x in f3 if x is not None])), 4)
        # 预测森林构成
        pf = np.isin(pred[m], [31, 32, 33])
        out['pred_forest_n'] = int(pf.sum())
        if pf.sum():
            out['pred_needle_share_of_forest'] = round(float((pred[m] == 31).sum() / pf.sum()), 4)
            out['pred_mixed_share_of_forest'] = round(float((pred[m] == 33).sum() / pf.sum()), 4)
        out['ref_needle_share_of_forest'] = round(float((ref[m] == 31).sum() / m.sum()), 4)
        # 混交泄漏：参考纯林的点上被预测为 33 的比例
        out['mixed_leak_on_pure_ref'] = round(float((pred[m] == 33).mean()), 4)
    # 阴阳坡 OR（预测 31 vs 32；slope>5 且坡向有效）
    if asp_cls is not None and slope is not None:
        steep = (slope > 5) & (asp_cls > 0)
        sh = steep & (asp_cls == 2)
        su = steep & (asp_cls == 1)
        cells = {
            'needle_shady': int(((pred == 31) & sh).sum()),
            'needle_sunny': int(((pred == 31) & su).sum()),
            'broad_shady': int(((pred == 32) & sh).sum()),
            'broad_sunny': int(((pred == 32) & su).sum()),
        }
        out['or_cells_pred'] = cells
        out['or_needle_broad'] = or_calc(cells['needle_shady'], cells['needle_sunny'],
                                         cells['broad_shady'], cells['broad_sunny'])
        out['or_ci95'] = or_ci(cells['needle_shady'], cells['needle_sunny'],
                               cells['broad_shady'], cells['broad_sunny'])
        if ref is not None:
            rc = {
                'needle_shady': int(((ref == 31) & sh).sum()),
                'needle_sunny': int(((ref == 31) & su).sum()),
                'broad_shady': int(((ref == 32) & sh).sum()),
                'broad_sunny': int(((ref == 32) & su).sum()),
            }
            out['or_cells_ref'] = rc
            out['or_ref'] = or_calc(rc['needle_shady'], rc['needle_sunny'],
                                    rc['broad_shady'], rc['broad_sunny'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tiles', default='all', help="all | priority | comma list")
    ap.add_argument('--seeds', type=int, nargs='+', default=[7, 11, 23])
    ap.add_argument('--trees', type=int, default=100)
    ap.add_argument('--nfolds', type=int, default=5)
    ap.add_argument('--out', default='two_arm_metrics.json')
    ap.add_argument('--tag', default='all76')
    a = ap.parse_args()

    df = pd.read_parquet(os.path.join(HERE, 'points_76.parquet'))
    if a.tiles == 'priority':
        df = df[df.is_priority == 1].reset_index(drop=True)
    elif a.tiles not in ('all',):
        keep = set(a.tiles.split(','))
        df = df[df.tile.isin(keep)].reset_index(drop=True)
    # 评估/训练共用点集：ArmB 标签必须有定义
    df = df[(df.cl_newA > 0) & (df.cl_newB > 0)].reset_index(drop=True)
    print('[%s] n=%d tiles=%d' % (a.tag, len(df), df.tile.nunique()))

    X = np.nan_to_num(df[AEF].to_numpy(np.float32), nan=0.0)
    folds = spatial_fold(df, km=10.0, nf=a.nfolds)
    df['fold'] = folds
    ref = df.cl_fcs10.to_numpy(np.int16)
    asp_cls = df.asp_cls.to_numpy()
    slope = df.slope.to_numpy(np.float32)
    res = {
        'meta': {'tag': a.tag, 'n': int(len(df)), 'tiles': int(df.tile.nunique()),
                 'seeds': a.seeds, 'nfolds': a.nfolds, 'block_km': 10.0,
                 'rf': {'n_estimators': a.trees, 'min_samples_leaf': 2, 'max_leaf_nodes': 5000},
                 'reference': 'GLC_FCS10-2023 (needle 71/72/81/82, broad 51/52/61/62, mixed 91/92)',
                 'fold_seed': 20261017},
        'label_dist': {
            'armA': {str(k): int(v) for k, v in df.cl_newA.value_counts().sort_index().items()},
            'armB': {str(k): int(v) for k, v in df.cl_newB.value_counts().sort_index().items()},
            'ref': {str(k): int(v) for k, v in pd.Series(ref).value_counts().sort_index().items() if k > 0}},
        'arms': {}
    }
    print('参考分布', res['label_dist']['ref'])

    oof_store = {}
    for arm, col in [('A', 'cl_newA'), ('B', 'cl_newB')]:
        y = df[col].to_numpy(np.int64)
        t_arm = time.time()
        per_seed = []
        oof_seeds = []
        for seed in a.seeds:
            t0 = time.time()
            oof = np.zeros(len(df), dtype=np.int16)
            for f in range(a.nfolds):
                te = folds == f
                tr = ~te
                m = RandomForestClassifier(n_estimators=a.trees, min_samples_leaf=2,
                                           max_leaf_nodes=5000, n_jobs=13,
                                           random_state=seed)
                m.fit(X[tr], y[tr])
                oof[te] = m.predict(X[te]).astype(np.int16)
            per_seed.append(eval_slice(oof, ref, asp_cls, slope, tag='seed%d' % seed))
            oof_seeds.append(oof)
            print('  arm%s seed%s %.0fs' % (arm, seed, time.time() - t0), flush=True)
        oof_store[arm] = np.array(oof_seeds)      # (n_seed, n)
        # 种子聚合（投票后重算，作为主口径）
        P = oof_store[arm]
        vote = mode_vote(P)
        agg = eval_slice(vote, ref, asp_cls, slope, tag='vote3')
        # 每种子指标的均值/标准差（稳定性）
        stab = {}
        for k in ['needle', 'broad']:
            for mtr in ['precision', 'recall', 'f1']:
                vals = [s[k][mtr] for s in per_seed if s.get(k) and s[k].get(mtr) is not None]
                if vals:
                    stab['%s_%s' % (k, mtr)] = {'mean': round(float(np.mean(vals)), 4),
                                                'sd': round(float(np.std(vals)), 4)}
        ors = [s['or_needle_broad'] for s in per_seed if s.get('or_needle_broad')]
        res['arms'][arm] = {
            'vote3': agg, 'per_seed': per_seed, 'seed_stability': stab,
            'or_per_seed': [round(x, 3) for x in ors],
            'or_median': round(float(np.median(ors)), 3) if ors else None,
            'secs': round(time.time() - t_arm, 0),
        }
        print('  arm%s vote3: needle P/R/F1=%s/%s/%s macroF1_2=%s OR=%s (median %s) OA=%s' % (
            arm, agg.get('needle', {}).get('precision'), agg.get('needle', {}).get('recall'),
            agg.get('needle', {}).get('f1'), agg.get('macro_f1_2'),
            agg.get('or_needle_broad'), res['arms'][arm]['or_median'], agg.get('oa_all22')), flush=True)

    # 分层：区域 / 省 / 瓦
    strata = {}
    for key, col in [('region', 'region'), ('province', 'province'), ('tile', 'tile')]:
        strata[key] = {}
        for g, sub in df.groupby(col, dropna=False):
            if not isinstance(g, str):
                g = 'NA'
            idx = sub.index.to_numpy()
            if len(idx) < 200:
                continue
            entry = {}
            for arm in ('A', 'B'):
                P = oof_store[arm]
                subP = P[:, idx]
                vote = mode_vote(subP)
                entry['arm%s' % arm] = eval_slice(
                    vote, ref[idx], asp_cls[idx], slope[idx], tag=str(g))
            strata[key][g] = entry
    res['strata'] = strata

    np.savez_compressed(os.path.join(HERE, 'oof_preds_%s.npz' % a.tag),
                        folds=folds, cl_newA=df.cl_newA.to_numpy(np.int16),
                        cl_newB=df.cl_newB.to_numpy(np.int16),
                        cl_fcs10=ref, lon=df.lon.to_numpy(np.float32),
                        lat=df.lat.to_numpy(np.float32),
                        **{'oofA_%d' % s: oof_store['A'][i] for i, s in enumerate(a.seeds)},
                        **{'oofB_%d' % s: oof_store['B'][i] for i, s in enumerate(a.seeds)})
    with open(os.path.join(HERE, a.out), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print('->', os.path.join(HERE, a.out))


if __name__ == '__main__':
    main()
