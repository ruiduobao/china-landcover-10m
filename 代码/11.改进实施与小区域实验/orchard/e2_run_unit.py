# -*- coding: utf-8 -*-
"""e2_run_unit.py — 园地先验实验·单元运行器（region × arm × 3 种子；限时 1200s）

* 输入：<WS>/cache/<region>_train_window.parquet（tile±2° 窗内池点，已剔本区 dev+conf 块）
        <WS>/cache/<region>_ref.parquet（本区参考点，含 split/block/t24/isf + AEF）
        <WS>/prereg/regions_frozen.json（冻结注册）
* 规则源：doc70 §4.3（基线=局部均匀随机 20k；O-CAP=类2 上限 1%）、§6.2（同点/来源分层）、§6.3（块 bootstrap）
* 门槛：单元 wall-clock ≤1200s（由 e3_driver 监督终止）；本地 sklearn RF ≠ GEE smileRF
* 输出：<WS>/results/<region>_<arm>_pred.csv、<WS>/results/<region>_<arm>.json
* 用法：PYTHONDONTWRITEBYTECODE=1 python e2_run_unit.py --region SC1 --arm baseline
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys, json, time, argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.dont_write_bytecode = True
sys.path.insert(0, r'F:/lc_work/v31_exp/code')
import v31_common as VC

FEATS = ['A%02d' % i for i in range(64)]
RF_PARAMS = dict(n_estimators=100, min_samples_leaf=2, max_leaf_nodes=5000, n_jobs=10)


def emit(m):
    print('[%s] %s' % (time.strftime('%m-%d %H:%M:%S'), m), flush=True)


def sample_arm(cl, seed, arm, n_limit, cap_frac=0.01):
    """基线：rng(seed).permutation 取前 n_limit。O-CAP：同 permutation，类2 上限 cap_frac*n_limit。"""
    n = len(cl)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    if arm == 'baseline':
        return perm[:n_limit].copy(), dict(n_train=int(min(n_limit, n)), n_class2=int((cl[perm[:n_limit]] == 2).sum()))
    cap2 = int(round(cap_frac * n_limit))          # 200
    chosen, n2 = [], 0
    for j in perm:
        if cl[j] == 2:
            if n2 >= cap2:
                continue
            n2 += 1
        chosen.append(j)
        if len(chosen) >= n_limit:
            break
    idx = np.asarray(chosen, dtype='int64')
    return idx, dict(n_train=int(len(idx)), n_class2=int(n2), cap2=cap2,
                     n_class2_window=int((cl == 2).sum()))


def metrics_pack(t, p, macro_fn, label):
    """单层指标：OA24、macro9 OA、macroF1（固定 labels）、逐类 F1/PA/UA。"""
    m = VC.metrics(t, p)
    m9 = VC.metrics(macro_fn(t), macro_fn(p))
    return dict(label=label, n=int(m['n']), OA24=round(m['OA'], 4),
                macro9_OA=round(m9['OA'], 4), macroF1_24=round(m['macroF1'], 4),
                macroF1_9=round(m9['macroF1'], 4),
                per={int(c): dict(n=d['n'], f1=round(d['f1'], 4), pa=round(d['pa'], 4),
                                  ua=round(d['ua'], 4)) for c, d in m['per'].items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--region', required=True)
    ap.add_argument('--arm', required=True, choices=['baseline', 'OCAP'])
    a = ap.parse_args()
    t_start = time.time()
    reg = json.load(open(os.path.join(WS, 'prereg', 'regions_frozen.json'), encoding='utf-8'))
    R = reg['regions'][a.region]
    seeds = reg['seeds']
    n_limit = reg['n_train_limit']

    win = pd.read_parquet(R['cache'])
    ref = pd.read_parquet(R['ref_cache'])
    cl = win['cl'].to_numpy()
    Xw = win[FEATS].to_numpy(np.float32)
    Xr = ref[FEATS].to_numpy(np.float32)
    t24 = ref['t24'].to_numpy()
    isf = ref['isf'].to_numpy()
    emit('%s/%s 训练窗 %d 点（类2 %d）｜参考 %d（dev %d/conf %d，非fcs10 %d）' % (
        a.region, a.arm, len(win), int((cl == 2).sum()), len(ref),
        int((ref.split == 'dev').sum()), int((ref.split == 'conf').sum()), int((~isf).sum())))

    from sklearn.ensemble import RandomForestClassifier
    macro_fn = VC.to_macro
    per_seed = []
    pred_frames = []
    for seed in seeds:
        idx, sinfo = sample_arm(cl, seed, a.arm, n_limit)
        rf = RandomForestClassifier(random_state=seed, **RF_PARAMS)
        t0 = time.time()
        rf.fit(Xw[idx], cl[idx])
        fit_s = time.time() - t0
        t0 = time.time()
        pred = rf.predict(Xr)
        pred_s = time.time() - t0
        per_seed.append(dict(seed=seed, **sinfo, fit_s=round(fit_s, 1), predict_s=round(pred_s, 1),
                             n_classes_train=int(len(np.unique(cl[idx]))),
                             train_class_counts={int(c): int(v) for c, v in
                                                 zip(*np.unique(cl[idx], return_counts=True))}))
        pred_frames.append(pd.DataFrame(dict(region=a.region, arm=a.arm, seed=seed,
                                             row_id=ref.row_id.to_numpy(), lon=ref.lon.to_numpy(),
                                             lat=ref.lat.to_numpy(), t24=t24, pred=pred,
                                             isf=isf, block=ref.block.to_numpy(),
                                             split=ref.split.to_numpy())))
        emit('  seed %d: train %d（类2 %d）fit %.1fs pred %.1fs' % (seed, len(idx), sinfo['n_class2'], fit_s, pred_s))

    preds = pd.concat(pred_frames, ignore_index=True)
    out_dir = os.path.join(WS, 'results')
    os.makedirs(out_dir, exist_ok=True)
    preds.to_csv(os.path.join(out_dir, '%s_%s_pred.csv' % (a.region, a.arm)), index=False, encoding='utf-8-sig')

    # 每 seed × 每层（全点/非fcs10/fcs10/dev/conf）指标
    per_seed_m = []
    for seed in seeds:
        ps = preds[preds.seed == seed]
        t, p = ps.t24.to_numpy(), ps.pred.to_numpy()
        f = ps.isf.to_numpy()
        sp = ps.split.to_numpy()
        blk = dict(
            all=metrics_pack(t, p, macro_fn, 'all'),
            nonfcs10=metrics_pack(t[~f], p[~f], macro_fn, 'nonfcs10') if (~f).sum() >= 10 else None,
            fcs10=metrics_pack(t[f], p[f], macro_fn, 'fcs10') if f.sum() >= 10 else None,
            dev=metrics_pack(t[sp == 'dev'], p[sp == 'dev'], macro_fn, 'dev'),
            conf=metrics_pack(t[sp == 'conf'], p[sp == 'conf'], macro_fn, 'conf'),
            conf_nonfcs10=metrics_pack(t[(sp == 'conf') & (~f)], p[(sp == 'conf') & (~f)],
                                       macro_fn, 'conf_nonfcs10') if ((sp == 'conf') & (~f)).sum() >= 10 else None,
        )
        per_seed_m.append(dict(seed=seed, **blk))

    unit = dict(region=a.region, arm=a.arm, rid_meta=dict(
        bbox=R['bbox'], tile=R['tile'], training_bbox=R['training_bbox'],
        refn=R['refn'], n_dev=R['n_dev_pts'], n_conf=R['n_conf_pts'],
        n_conf_cl4=R['n_conf_cl4'], n_window_after=R['n_window_after'],
        exclude_blocks_rule=R['exclude_blocks_rule']),
        wall_s=round(time.time() - t_start, 1), n_train_per_seed=per_seed,
        per_seed_metrics=per_seed_m)
    fp = os.path.join(out_dir, '%s_%s.json' % (a.region, a.arm))
    with open(fp, 'w', encoding='utf-8') as fh:
        json.dump(unit, fh, ensure_ascii=False, indent=1, default=str)
    emit('→ %s (wall %.1fs)' % (fp, unit['wall_s']))


if __name__ == '__main__':
    main()
