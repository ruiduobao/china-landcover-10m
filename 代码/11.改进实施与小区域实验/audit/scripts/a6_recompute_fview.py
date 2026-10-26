# -*- coding: utf-8 -*-
"""a6_recompute_fview.py — F-VIEW(G2) 主层(conf∩nonFCS10)独立复算（不重训 RF）

* 输入：terrain/out/t4_fview_arms_G2_seed7.npz（eval_split/eval_is_fcs10/eval_block/eval_y/eval_lon/eval_lat）
        terrain/out/t4_pred_G2_<arm>_seed{k}.csv（pred 列；列名容错：pred/pred24/pred_v31）
        terrain/out/t4_metrics_G2.json（对照，可选）
* 规则源：doc70 §6.2/§6.3/§7；macro9 来自 v31_common.to_macro
* 门槛（仅报告）：主层 conf∩nonFCS10 macro9 Δ点估≥−0.01 且块bootstrap95%下界≥−0.02；目标类4/6 F1 Δ≥+0.03
* 输出：audit/out/fview_recompute_<ts>.json
* 用法：python a6_recompute_fview.py [--terrain ...] [--frame G2] [--arms baseline_common strict_common]
"""
import argparse
import glob
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIT = os.path.dirname(HERE)
OUT = os.path.join(AUDIT, 'out')
sys.path.insert(0, r'F:/lc_work/v31_exp/code')
import v31_common as VC  # noqa: E402

MACRO = VC.to_macro
NBOOT = 10000
BOOT_SEED = 20261015
SEEDS = [7, 11, 23]


def boot_ci(blocks, dpoint, nboot=NBOOT, seed=BOOT_SEED):
    ub, inv = np.unique(blocks, return_inverse=True)
    idx_by_b = [np.flatnonzero(inv == j) for j in range(len(ub))]
    rng = np.random.default_rng(seed)
    means = np.empty(nboot)
    for i in range(nboot):
        pick = rng.integers(0, len(ub), len(ub))
        sel = np.concatenate([idx_by_b[j] for j in pick])
        means[i] = dpoint[sel].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), int(len(ub))


def search_dirs(od, frame):
    ds = [od, os.path.join(od, 'fview_g2')]
    if frame != 'G2':
        ds.append(os.path.join(od, 'fview_%s' % frame.lower()))
    return [d for d in ds if os.path.isdir(d)]


def find_pred(od, frame, arm, seed):
    """兼容 t5_pred / t4_pred 及多种列名/顺序；优先 fview_g2 新目录。"""
    stems = ['t5_pred', 't5', 't4_pred', 't4']
    for d in search_dirs(od, frame):
        for st in stems:
            for cand in ['%s_%s_%s_seed%d.csv' % (st, frame, arm, seed),
                         '%s_%s_seed%d_%s.csv' % (st, frame, seed, arm),
                         '%s_%s_%s_s%d.csv' % (st, frame, arm, seed)]:
                fp = os.path.join(d, cand)
                if os.path.exists(fp):
                    return fp
        hits = glob.glob(os.path.join(d, '*pred*%s*%s*%d*.csv' % (frame, arm.split('_')[0], seed)))
        if hits:
            return sorted(hits)[0]
    return None


def find_eval_npz(od, frame):
    for d in search_dirs(od, frame):
        for pat in ['*arms*%s*.npz' % frame, '*arms*npz', '*eval*.npz', '*.npz']:
            hits = sorted(glob.glob(os.path.join(d, pat)))
            for h in hits:
                try:
                    z = np.load(h, allow_pickle=True)
                    if 'eval_y' in z and 'eval_split' in z:
                        return h
                except Exception:
                    continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--terrain', default=r'F:/lc_work/improve_2023_pilot_20261015/terrain')
    ap.add_argument('--frame', default='G2')
    ap.add_argument('--arms', nargs='+', default=['baseline_common', 'strict_common'])
    args = ap.parse_args()
    od = os.path.join(args.terrain, 'out')
    npz_path = find_eval_npz(od, args.frame)
    if not npz_path:
        print('[PENDING] 未找到含 eval_y/eval_split 的 npz')
        return
    out_npz = npz_path
    z = np.load(npz_path, allow_pickle=True)
    t24 = z['eval_y'].astype(int)
    split = z['eval_split'].astype(int)          # 0=conf, 1=dev（实测约定）
    isf = z['eval_is_fcs10'].astype(bool)
    block = z['eval_block']
    lon, lat = z['eval_lon'], z['eval_lat']

    preds, missing = {}, []
    for a in args.arms:
        for s in SEEDS:
            fp = find_pred(od, args.frame, a, s)
            if not fp:
                missing.append('%s/seed%d' % (a, s))
                continue
            p = pd.read_csv(fp)
            col = next((c for c in ['pred', 'pred24', 'pred_v31', 'pred24_v31'] if c in p.columns), None)
            if col is None:
                missing.append('%s/seed%d (no pred col: %s)' % (a, s, list(p.columns)))
                continue
            if len(p) != len(t24):
                missing.append('%s/seed%d len %d != %d' % (a, s, len(p), len(t24)))
                continue
            preds[(a, s)] = p[col].to_numpy().astype(int)

    out = {'generated_at': datetime.now().isoformat(timespec='seconds'), 'frame': args.frame,
           'npz_source': out_npz,
           'arms': args.arms, 'n_eval': int(len(t24)), 'n_conf': int((split == 0).sum()),
           'missing': missing, 'layers': {}, 'target_f1': {}}
    if missing:
        json.dump(out, open(os.path.join(OUT, 'fview_recompute_pending.json'), 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=2)
        print('[PENDING] 缺少预测文件: %s' % missing)
        return

    base, cand = args.arms[0], args.arms[1]
    masks = {'all': np.ones(len(t24), bool), 'nonfcs10': ~isf, 'fcs10': isf,
             'conf': split == 0, 'conf_nonfcs10': (split == 0) & (~isf), 'dev': split == 1}
    for name, m in masks.items():
        if m.sum() < 10:
            out['layers'][name] = None
            continue
        ok24b = np.mean([(preds[(base, s)][m] == t24[m]) for s in SEEDS], axis=0)
        ok24o = np.mean([(preds[(cand, s)][m] == t24[m]) for s in SEEDS], axis=0)
        ok9b = np.mean([(MACRO(preds[(base, s)][m]) == MACRO(t24[m])) for s in SEEDS], axis=0)
        ok9o = np.mean([(MACRO(preds[(cand, s)][m]) == MACRO(t24[m])) for s in SEEDS], axis=0)
        d24, d9 = ok24o - ok24b, ok9o - ok9b
        lo24, hi24, nb = boot_ci(block[m], d24)
        lo9, hi9, _ = boot_ci(block[m], d9)
        out['layers'][name] = dict(n=int(m.sum()), n_blocks=nb,
                                   OA24_b=round(float(ok24b.mean()), 4), OA24_o=round(float(ok24o.mean()), 4),
                                   dOA24=round(float(d24.mean()), 4), dOA24_ci=[round(lo24, 4), round(hi24, 4)],
                                   m9_b=round(float(ok9b.mean()), 4), m9_o=round(float(ok9o.mean()), 4),
                                   dm9=round(float(d9.mean()), 4), dm9_ci=[round(lo9, 4), round(hi9, 4)])

    m = (split == 0) & (~isf)
    tt = t24[m]
    for c in sorted(set(tt.tolist())):
        fb, fo = [], []
        for s in SEEDS:
            for a, acc in [(base, fb), (cand, fo)]:
                pv = preds[(a, s)][m]
                tp = int(((tt == c) & (pv == c)).sum()); fp = int(((tt != c) & (pv == c)).sum())
                fn = int(((tt == c) & (pv != c)).sum())
                pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
                acc.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
        out['target_f1'][int(c)] = dict(n=int((tt == c).sum()), f1_b=round(float(np.mean(fb)), 4),
                                        f1_o=round(float(np.mean(fo)), 4),
                                        dF1=round(float(np.mean(fo) - np.mean(fb)), 4))

    os.makedirs(OUT, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = os.path.join(OUT, 'fview_recompute_%s.json' % ts)
    json.dump(out, open(fp, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('[OK] -> %s' % fp)
    for nm, l in out['layers'].items():
        if l:
            print('  %-16s n=%d blocks=%d dOA24=%+.4f %s dm9=%+.4f %s' % (
                nm, l['n'], l['n_blocks'], l['dOA24'], l['dOA24_ci'], l['dm9'], l['dm9_ci']))
    print('  target:', {k: (v['n'], v['dF1']) for k, v in out['target_f1'].items()})


if __name__ == '__main__':
    main()
