# -*- coding: utf-8 -*-
"""a2_recompute_metrics.py — 独立复算同点/分层/细类指标 + 配对块 bootstrap + McNemar

* 输入：
    --preds  ：一个或多个 CSV/Parquet，长表，列：
               arm, seed, ref_id, truth, pred, stratum, block
               （arm=臂名；stratum∈{all,fcs10,nonfcs10}；block=空间块 id）
    --config ：JSON，含
               class_codes  : 固定 labels 集合（如 1..24 中实际出现码）
               macro9_map   : {类码: 组名}（宏观九组归并；必须由实验方提供，不猜测）
               baseline_arm : 基线臂名
               可选 target_classes:[类码...]；bootstrap:10000；seed_level:True
* 规则源：doc70 §6.2/§6.3/§7（同点、分层、UA=precision/PA=recall、固定labels、macro9）
* 门槛（仅报告，不自动裁决）：目标类 ΔF1≥0.03；macro9 Δ点估≥−0.01 且块 CI 下界≥−0.02
* 输出：out/metrics_<ts>.json + per_class_<ts>.csv + paired_<ts>.csv
* 用法：python a2_recompute_metrics.py --preds a.csv b.csv --config cfg.json [--bootstrap 10000]
* 说明：只复算，不重训 RF、不改实验原结果；schema/映射缺失即报错。
"""
import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIT = os.path.dirname(HERE)
OUT = os.path.join(AUDIT, 'out')

REQ_COLS = ['arm', 'seed', 'ref_id', 'truth', 'pred', 'stratum', 'block']


def read_preds(paths):
    frames = []
    for p in paths:
        p = os.path.abspath(p)
        if not os.path.exists(p):
            sys.exit('[ERR] preds not found: %s' % p)
        if p.lower().endswith(('.parquet', '.pq')):
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p)
        df['__src'] = os.path.basename(p)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        sys.exit('[ERR] preds missing columns: %s' % missing)
    return df


def f1_table(df, class_codes):
    """返回 per-class UA/PA/F1（固定 labels 集合）。"""
    y, p = df['truth'].to_numpy(), df['pred'].to_numpy()
    rows = []
    for c in class_codes:
        tp = int(np.sum((y == c) & (p == c)))
        fp = int(np.sum((y != c) & (p == c)))
        fn = int(np.sum((y == c) & (p != c)))
        ua = tp / (tp + fp) if (tp + fp) else 0.0
        pa = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * ua * pa / (ua + pa)) if (ua + pa) else 0.0
        rows.append({'class': c, 'n_ref': int(np.sum(y == c)), 'n_pred': int(np.sum(p == c)),
                     'UA': round(ua, 6), 'PA': round(pa, 6), 'F1': round(f1, 6)})
    return rows


def oa(df):
    return float(np.mean(df['truth'].to_numpy() == df['pred'].to_numpy())) if len(df) else float('nan')


def macro9_oa(df, macro9_map):
    m = df[df['truth'].isin(macro9_map) & df['pred'].isin(macro9_map)]
    if not len(m):
        return float('nan'), 0
    t = m['truth'].map(macro9_map).to_numpy()
    p = m['pred'].map(macro9_map).to_numpy()
    return float(np.mean(t == p)), int(len(m))


def macro_f1(rows):
    v = [r['F1'] for r in rows if r['n_ref'] > 0]
    return float(np.mean(v)) if v else float('nan')


def summarize(df, class_codes, macro9_map):
    per = f1_table(df, class_codes)
    m9, n9 = macro9_oa(df, macro9_map)
    return {
        'n': int(len(df)),
        'OA24': round(oa(df), 6),
        'macro9_OA': round(m9, 6) if m9 == m9 else None,
        'macro9_n': n9,
        'macro_F1': round(macro_f1(per), 6),
        'per_class': per,
    }


def block_bootstrap_delta(dfb, dfc, key, blocks, class_codes, macro9_map, metric, n_boot, rng):
    """按空间块重采样，计算 metric 的配对 delta 分布。metric ∈ {OA24, macro9_OA, F1:<c>}。"""
    bmap = dfb.set_index(key)
    cmap = dfc.set_index(key)
    common = bmap.index.intersection(cmap.index)
    bmap, cmap = bmap.loc[common], cmap.loc[common]
    blk = bmap['block'].to_numpy()
    uniq = np.array(sorted(set(blocks) & set(blk)))
    if len(uniq) < 2:
        return None, int(len(uniq))

    idx_by_blk = {b: np.where(blk == b)[0] for b in uniq}

    def metric_of(d, idx):
        sub = d.iloc[idx]
        if metric == 'OA24':
            return oa(sub)
        if metric == 'macro9_OA':
            return macro9_oa(sub, macro9_map)[0]
        if metric.startswith('F1:'):
            c = int(metric.split(':')[1])
            for r in f1_table(sub, [c]):
                return r['F1']
        return float('nan')

    b_arr = bmap.reset_index(drop=True)
    c_arr = cmap.reset_index(drop=True)
    deltas = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_blk[b] for b in pick])
        d = metric_of(b_arr, idx) - metric_of(c_arr, idx)
        if d == d:
            deltas.append(d)
    if not deltas:
        return None, int(len(uniq))
    a = np.array(deltas)
    return {'median': float(np.median(a)), 'lo': float(np.percentile(a, 2.5)),
            'hi': float(np.percentile(a, 97.5)), 'n_boot': int(len(a))}, int(len(uniq))


def mcnemar_exact(y, pb, pc):
    """配对 McNemar 精确检验（baseline 正确 vs candidate 正确）。"""
    b_ok = (pb == y)
    c_ok = (pc == y)
    b01 = int(np.sum(b_ok & ~c_ok))
    b10 = int(np.sum(~b_ok & c_ok))
    n = b01 + b10
    if n == 0:
        return {'b01': 0, 'b10': 0, 'p_value': 1.0}
    try:
        from scipy.stats import binomtest
        p = binomtest(min(b01, b10), n, 0.5).pvalue
    except Exception:
        from math import comb
        k = min(b01, b10)
        p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))
    return {'b01': b01, 'b10': b10, 'p_value': float(p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds', nargs='+', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--bootstrap', type=int, default=None)
    args = ap.parse_args()

    cfg = json.load(open(args.config, encoding='utf-8'))
    for k in ['class_codes', 'macro9_map', 'baseline_arm']:
        if k not in cfg:
            sys.exit('[ERR] config missing key: %s' % k)
    class_codes = [int(c) for c in cfg['class_codes']]
    macro9_map = {int(k): v for k, v in cfg['macro9_map'].items()}
    baseline = cfg['baseline_arm']
    n_boot = args.bootstrap or int(cfg.get('bootstrap', 10000))
    rng = np.random.default_rng(cfg.get('bootstrap_seed', 20261015))

    df = read_preds(args.preds)

    out = {'generated_at': datetime.now().isoformat(timespec='seconds'),
           'baseline_arm': baseline, 'bootstrap': n_boot,
           'class_codes': class_codes, 'n_boot_note': 'block bootstrap; 报告有效块数',
           'summaries': {}, 'paired': {}, 'warnings': []}

    # 1) 各臂×种子×分层摘要
    per_class_rows = []
    for (arm, seed, stratum), sub in df.groupby(['arm', 'seed', 'stratum']):
        s = summarize(sub, class_codes, macro9_map)
        out['summaries']['%s|seed%s|%s' % (arm, seed, stratum)] = {
            'n': s['n'], 'OA24': s['OA24'], 'macro9_OA': s['macro9_OA'],
            'macro9_n': s['macro9_n'], 'macro_F1': s['macro_F1'],
        }
        for r in s['per_class']:
            per_class_rows.append({'arm': arm, 'seed': seed, 'stratum': stratum, **r})

    # 2) 配对（候选 vs 基线），同 ref_id、同 seed、同 stratum
    key = ['ref_id']
    metrics = ['OA24', 'macro9_OA'] + ['F1:%d' % c for c in cfg.get('target_classes', [])]
    for stratum in sorted(df['stratum'].unique()):
        base = df[(df['arm'] == baseline) & (df['stratum'] == stratum)]
        if not len(base):
            out['warnings'].append('no baseline rows for stratum=%s' % stratum)
            continue
        cand_arms = [a for a in sorted(df['arm'].unique()) if a != baseline]
        for arm in cand_arms:
            cand = df[(df['arm'] == arm) & (df['stratum'] == stratum)]
            if not len(cand):
                continue
            seed_deltas = {}
            for seed in sorted(set(base['seed']) & set(cand['seed'])):
                b = base[base['seed'] == seed]
                c = cand[cand['seed'] == seed]
                # 固定共同有效点集（同点）
                common = set(b['ref_id']) & set(c['ref_id'])
                b, c = b[b['ref_id'].isin(common)], c[c['ref_id'].isin(common)]
                if not len(b):
                    continue
                bc = b.set_index('ref_id').sort_index()
                cc = c.set_index('ref_id').sort_index()
                d = {'n_common': int(len(common))}
                d['d_OA24'] = round(oa(cc) - oa(bc), 6)
                d['d_macro9_OA'] = round(macro9_oa(cc, macro9_map)[0] - macro9_oa(bc, macro9_map)[0], 6)
                pb, pc = bc['pred'].to_numpy(), cc['pred'].to_numpy()
                d['mcnemar_OA'] = mcnemar_exact(bc['truth'].to_numpy(), pb, pc)
                seed_deltas[int(seed)] = d

                # 目标类 ΔF1（同点、固定 labels）
                for c_code in cfg.get('target_classes', []):
                    fb = f1_table(bc.reset_index(), [c_code])[0]
                    fc = f1_table(cc.reset_index(), [c_code])[0]
                    d['d_F1_%d' % c_code] = round(fc['F1'] - fb['F1'], 6)
                    d['n_ref_%d' % c_code] = fb['n_ref']

            # 跨种子均值 + 块 bootstrap（用第一种子做块结构代表，块来自基线）
            b0 = base[base['seed'] == sorted(set(base['seed']) & set(cand['seed']))[0]]
            c0 = cand[cand['seed'] == b0['seed'].iloc[0]]
            common0 = set(b0['ref_id']) & set(c0['ref_id'])
            b0, c0 = b0[b0['ref_id'].isin(common0)], c0[c0['ref_id'].isin(common0)]
            blocks = sorted(set(b0['block']))
            boots = {}
            for m in metrics:
                ci, nblk = block_bootstrap_delta(b0, c0, key, blocks, class_codes, macro9_map,
                                                 m, n_boot, rng)
                boots[m] = {'ci': ci, 'n_blocks': nblk}
            out['paired']['%s_vs_%s|%s' % (arm, baseline, stratum)] = {
                'per_seed': seed_deltas,
                'n_seeds': len(seed_deltas),
                'block_bootstrap': boots,
                'block_count_note': '块数<30 时 CI 仅作该框描述，不外推省域',
            }

    os.makedirs(OUT, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(os.path.join(OUT, 'metrics_%s.json' % ts), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    pd.DataFrame(per_class_rows).to_csv(os.path.join(OUT, 'per_class_%s.csv' % ts),
                                        index=False, encoding='utf-8-sig')
    print('[OK] metrics -> %s/metrics_%s.json ; per_class_%s.csv' % (OUT, ts, ts))
    for w in out['warnings']:
        print('  [WARN] %s' % w)


if __name__ == '__main__':
    main()
