# -*- coding: utf-8 -*-
"""t3_south.py — 南方针叶表达方案实测（ArmB 标签下）
* 输入（只读）：exp/two_class_retrain/points_76.parquet
                  exp/two_class_retrain/needle_broad_map.json
* 方案（全部同一 OOF 折、同一 RF 参数、同一参考 FCS10）：
    S0 基线        多类 RF argmax（22 类）
    S1 类别权重    多类 RF，针叶(31)样本权重 w∈{2,5,10}
    S2 二分类专家  森林训练点(31/32，排除 33) 训练 P(needle|forest)，
                  在多类预测为森林的点上按阈值 t∈{.5,.3,.2,.1,.05} 覆盖针/阔
    S3 占比匹配上限（诊断）阈值取到预测针叶占比=参考针叶占比，测该点的 recall/precision
* 门槛：G10 南方针叶不得禁用 → 报告南方针叶 recall 与预测占比，不设人为下限；
        同时报阔叶 precision 损失与 OR，防止为保针叶而破坏阔叶/坡向
* 输出：south_variants.json
* 用法：python t3_south.py [--arm B]
"""
import os, sys, json, time, argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
AEF = [f'A{i:02d}' for i in range(64)]
SOUTH = ['T2002', 'T1902', 'T2204']
NORTH = ['T2813', 'T2712']
SW = ['T1505', 'T1406', 'T1606']


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


def or_calc(c):
    if min(c['needle_shady'], c['needle_sunny'], c['broad_shady'], c['broad_sunny']) <= 0:
        return None
    return float((c['needle_shady'] / c['needle_sunny']) / (c['broad_shady'] / c['broad_sunny']))


def metrics(pred, ref, asp_cls, slope):
    m2 = np.isin(ref, [31, 32])
    out = {'n_ref_pure': int(m2.sum())}
    if m2.any():
        for c, nm in [(31, 'needle'), (32, 'broad')]:
            p = pred[m2] == c; r = ref[m2] == c
            tp = int((p & r).sum())
            prec = tp / p.sum() if p.sum() else None
            rec = tp / r.sum() if r.sum() else None
            f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else 0.0
            out[nm] = {'precision': None if prec is None else round(float(prec), 4),
                       'recall': None if rec is None else round(float(rec), 4),
                       'f1': round(float(f1), 4), 'n_pred': int(p.sum()), 'n_ref': int(r.sum())}
        f = [out['needle']['f1'], out['broad']['f1']]
        out['macro_f1_2'] = round(float(np.mean(f)), 4)
        pf = np.isin(pred[m2], [31, 32, 33])
        out['pred_needle_share_of_forest'] = round(float((pred[m2] == 31).sum() / max(pf.sum(), 1)), 4)
        out['ref_needle_share_of_forest'] = round(float((ref[m2] == 31).sum() / m2.sum()), 4)
        out['pred_mixed_share_of_forest'] = round(float((pred[m2] == 33).sum() / max(pf.sum(), 1)), 4)
    steep = (slope > 5) & (asp_cls > 0)
    sh = steep & (asp_cls == 2); su = steep & (asp_cls == 1)
    c = {'needle_shady': int(((pred == 31) & sh).sum()), 'needle_sunny': int(((pred == 31) & su).sum()),
         'broad_shady': int(((pred == 32) & sh).sum()), 'broad_sunny': int(((pred == 32) & su).sum())}
    out['or_cells'] = c
    out['or_needle_broad'] = or_calc(c)
    return out


def slice_eval(pred, ref, asp, slp, df):
    out = {'all': metrics(pred, ref, asp, slp)}
    for nm, tiles in [('south', SOUTH), ('north', NORTH), ('southwest', SW)]:
        m = df.tile.isin(tiles).to_numpy()
        if m.sum() >= 200:
            mm = metrics(pred[m], ref[m], asp[m], slp[m])
            mm['n'] = int(m.sum())
            out[nm] = mm
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='B')
    ap.add_argument('--seeds', type=int, nargs='+', default=[7, 11, 23])
    ap.add_argument('--trees', type=int, default=100)
    a = ap.parse_args()
    df = pd.read_parquet(os.path.join(HERE, 'points_76.parquet'))
    df = df[(df.cl_newB > 0) & (df.cl_newA > 0)].reset_index(drop=True)
    y = df['cl_new%s' % a.arm].to_numpy(np.int64)
    ref = df.cl_fcs10.to_numpy(np.int16)
    asp = df.asp_cls.to_numpy(); slp = df.slope.to_numpy(np.float32)
    X = np.nan_to_num(df[AEF].to_numpy(np.float32), nan=0.0)
    folds = spatial_fold(df)
    print('n', len(df), 'arm', a.arm, 'ref needle share', round(float((ref == 31).sum() / np.isin(ref, [31, 32]).sum()), 4))

    classes = np.array(sorted(np.unique(y)))
    res = {'meta': {'n': int(len(df)), 'arm': a.arm, 'seeds': a.seeds,
                    'south_tiles': SOUTH, 'north_tiles': NORTH, 'sw_tiles': SW}, 'variants': {}}

    # ---------- S0 / S1：多类 RF（含权重） ----------
    for name, w_needle in [('S0_baseline', 1.0), ('S1_w2', 2.0), ('S1_w5', 5.0), ('S1_w10', 10.0)]:
        t0 = time.time()
        preds = []
        proba_needle = np.zeros(len(df), np.float64)
        for seed in a.seeds:
            oof = np.zeros(len(df), dtype=np.int16)
            pn = np.zeros(len(df), np.float64)
            for f in range(5):
                te = folds == f; tr = ~te
                sw = np.ones(int(tr.sum()), np.float64)
                if w_needle != 1.0:
                    yy = y[tr]
                    sw[yy == 31] = w_needle
                m = RandomForestClassifier(n_estimators=a.trees, min_samples_leaf=2,
                                           max_leaf_nodes=5000, n_jobs=13, random_state=seed)
                m.fit(X[tr], y[tr], sample_weight=sw)
                pr = m.predict(X[te]).astype(np.int16)
                oof[te] = pr
                cls = list(m.classes_)
                if 31 in cls:
                    pn[te] = m.predict_proba(X[te])[:, cls.index(31)]
            preds.append(oof)
            proba_needle += pn / len(a.seeds)
        P = np.array(preds)
        vote = np.zeros(len(df), dtype=np.int16)
        for i in range(len(df)):
            vote[i] = np.bincount(P[:, i]).argmax()
        res['variants'][name] = {'weight_needle': w_needle,
                                 'metrics': slice_eval(vote, ref, asp, slp, df),
                                 'secs': round(time.time() - t0, 0)}
        v = res['variants'][name]['metrics']
        print('%-12s all: needle P/R/F1=%s/%s/%s  south share=%s rec=%s  OR=%s (%.0fs)' % (
            name, v['all']['needle']['precision'], v['all']['needle']['recall'], v['all']['needle']['f1'],
            v.get('south', {}).get('pred_needle_share_of_forest'),
            v.get('south', {}).get('needle', {}).get('recall'),
            v['all']['or_needle_broad'], time.time() - t0), flush=True)
        if name == 'S0_baseline':
            base_pred = vote
            base_pn = proba_needle.copy()
            base_proba = proba_needle

    # ---------- S2：二分类专家（森林点） ----------
    t0 = time.time()
    isf_lab = np.isin(y, [31, 32])
    idx_f = np.where(isf_lab)[0]
    ybin = (y[idx_f] == 31).astype(np.int8)
    p_needle_bin = np.zeros(len(df), np.float64)
    for seed in a.seeds:
        pn = np.zeros(len(df), np.float64)
        for f in range(5):
            te_all = folds == f
            te_f = te_all[idx_f]
            tr_f = ~te_all[idx_f]
            tr_others = (~te_all) & (~isf_lab)
            # 训练集 = 该折森林训练点 + 非森林训练点（保持"森林 vs 非森林"判别不变，
            # 二分类器只在森林点上决定针/阔，非森林点不参与二元训练）
            m = RandomForestClassifier(n_estimators=a.trees, min_samples_leaf=2,
                                       max_leaf_nodes=5000, n_jobs=13, random_state=seed)
            m.fit(X[idx_f][tr_f], ybin[tr_f])
            if te_f.any():
                pn[idx_f[te_f]] = m.predict_proba(X[idx_f][te_f])[:, list(m.classes_).index(1)]
        p_needle_bin += pn / len(a.seeds)
    res['variants']['S2_binary'] = {'thresholds': {}, 'secs': round(time.time() - t0, 0)}
    for t in [0.5, 0.3, 0.2, 0.1, 0.05]:
        pred = base_pred.copy()
        fmask = np.isin(base_pred, [31, 32, 33])
        pred[fmask & (p_needle_bin >= t)] = 31
        pred[fmask & (p_needle_bin < t)] = 32
        res['variants']['S2_binary']['thresholds'][str(t)] = {
            'metrics': slice_eval(pred, ref, asp, slp, df),
            'n_flipped': int(((base_pred != pred) & fmask).sum())}
        v = res['variants']['S2_binary']['thresholds'][str(t)]['metrics']['all']
        vs = res['variants']['S2_binary']['thresholds'][str(t)]['metrics'].get('south', {})
        print('  S2 t=%.2f all needle P/R/F1=%s/%s/%s  south share=%s rec=%s  flipped=%d' % (
            t, v['needle']['precision'], v['needle']['recall'], v['needle']['f1'],
            vs.get('pred_needle_share_of_forest'), vs.get('needle', {}).get('recall'),
            res['variants']['S2_binary']['thresholds'][str(t)]['n_flipped']), flush=True)

    # ---------- S3：占比匹配上限（诊断） ----------
    for region, tiles in [('all', None), ('south', SOUTH)]:
        if tiles is None:
            m = np.ones(len(df), bool)
        else:
            m = df.tile.isin(tiles).to_numpy()
        target = float((ref[m] == 31).sum() / max(np.isin(ref[m], [31, 32]).sum(), 1))
        fm = np.isin(base_pred, [31, 32, 33]) & m
        if fm.sum() == 0:
            continue
        q = np.quantile(p_needle_bin[fm], 1 - target) if target > 0 else 1.1
        pred = base_pred.copy()
        pred[fm & (p_needle_bin >= q)] = 31
        pred[fm & (p_needle_bin < q)] = 32
        v = metrics(pred, ref, asp, slp)
        if 'S3_prevalence' not in res['variants']:
            res['variants']['S3_prevalence'] = {}
        res['variants']['S3_prevalence'][region] = {
            'target_share': round(target, 4), 'quantile_thr': round(float(q), 4),
            'metrics': v}
        print('  S3 %s target=%.4f thr=%.4f needle P/R/F1=%s/%s/%s share=%s' % (
            region, target, q, v['needle']['precision'], v['needle']['recall'],
            v['needle']['f1'], v.get('pred_needle_share_of_forest')), flush=True)

    np.savez_compressed(os.path.join(HERE, 't3_scores.npz'),
                        p_needle_bin=p_needle_bin, p_needle_multi=base_proba,
                        base_pred=base_pred, ref=ref, folds=folds,
                        tile=df.tile.to_numpy().astype('U8'))
    with open(os.path.join(HERE, 'south_variants.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print('-> south_variants.json')


if __name__ == '__main__':
    main()
