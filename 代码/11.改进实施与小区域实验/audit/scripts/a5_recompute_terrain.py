# -*- coding: utf-8 -*-
"""a5_recompute_terrain.py — terrain T-SC 主层(conf∩nonFCS10)独立复算（不重训 RF）

* 输入：terrain/out/t1_arms_<frame>_seed7.npz（eval_split/eval_is_fcs10/eval_block/eval_y/eval_lon/eval_lat）
        terrain/out/t2_pred_<frame>_{baseline,T-STRATA}_seed{7,11,23}.csv（pred24 逐点）
        terrain/out/t2_metrics_<frame>.json（对照）
* 规则源：doc70 §6.2/§6.3/§7；macro9 来自 v31_common.to_macro
* 门槛（仅报告）：主层 conf∩nonFCS10 macro9 Δ点估≥−0.01 且块bootstrap95%下界≥−0.02；目标类 F1 Δ≥+0.03
* 输出：audit/out/terrain_recompute_<ts>.json
* 用法：python a5_recompute_terrain.py [--terrain ...]
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
sys.path.insert(0, r'F:/lc_work/v31_exp/code')
import v31_common as VC  # noqa: E402

MACRO = VC.to_macro
NBOOT = 10000
BOOT_SEED = 20261015
SEEDS = [7, 11, 23]
ARMS = ['baseline', 'T-STRATA']
FRAMES = ['F1', 'F2']


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


def layer(blocks, d24, d9, oa_b=None, oa_o=None, m9_b=None, m9_o=None):
    lo24, hi24, nb = boot_ci(blocks, d24)
    lo9, hi9, _ = boot_ci(blocks, d9)
    r = dict(n=int(len(d24)), n_blocks=nb,
             OA24_b=round(float(np.mean(oa_b)), 4) if oa_b is not None else None,
             OA24_o=round(float(np.mean(oa_o)), 4) if oa_o is not None else None,
             dOA24=round(float(d24.mean()), 4), dOA24_ci=[round(lo24, 4), round(hi24, 4)],
             m9_b=round(float(np.mean(m9_b)), 4) if m9_b is not None else None,
             m9_o=round(float(np.mean(m9_o)), 4) if m9_o is not None else None,
             dm9=round(float(d9.mean()), 4), dm9_ci=[round(lo9, 4), round(hi9, 4)])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--terrain', default=r'F:/lc_work/improve_2023_pilot_20261015/terrain')
    args = ap.parse_args()
    od = os.path.join(args.terrain, 'out')
    out = {'generated_at': datetime.now().isoformat(timespec='seconds'), 'frames': {}, 'discrepancies': []}

    for fr in FRAMES:
        z = np.load(os.path.join(od, 't1_arms_%s_seed7.npz' % fr), allow_pickle=True)
        t24 = z['eval_y'].astype(int)
        split = z['eval_split'].astype(int)          # 实测 1=dev,0=conf（与 t1_frames n_eval_conf/dev 对齐校验）
        isf = z['eval_is_fcs10'].astype(bool)
        block = z['eval_block']
        lon, lat = z['eval_lon'], z['eval_lat']

        # split 语义：0=conf, 1=dev（实测，与 t1_frames 对齐）
        n_conf = int((split == 0).sum())
        n_dev = int((split == 1).sum())
        preds = {}
        order_ok = True
        for arm in ARMS:
            for s in SEEDS:
                p = pd.read_csv(os.path.join(od, 't2_pred_%s_%s_seed%d.csv' % (fr, arm, s)))
                assert len(p) == len(t24), '%s %s %d rows != %d' % (fr, arm, s, len(p), len(t24))
                if not (np.allclose(p.lon.to_numpy(), lon, atol=1e-9) and np.allclose(p.lat.to_numpy(), lat, atol=1e-9)):
                    order_ok = False
                preds[(arm, s)] = p.pred24.to_numpy().astype(int)

        conf = split == 0
        masks = {
            'all': np.ones(len(t24), bool),
            'nonfcs10': ~isf,
            'fcs10': isf,
            'conf': conf,
            'conf_nonfcs10': conf & (~isf),
            'dev': ~conf,
        }
        frame_out = {'n_eval': int(len(t24)), 'n_conf': n_conf, 'n_dev': n_dev,
                     'n_conf_nonfcs10': int((conf & ~isf).sum()),
                     'pred_order_matches_npz': order_ok, 'layers': {}}

        for name, m in masks.items():
            if m.sum() < 10:
                frame_out['layers'][name] = None
                continue
            # 逐点跨种子平均正确率
            ok24 = {a: np.zeros(int(m.sum())) for a in ARMS}
            ok9 = {a: np.zeros(int(m.sum())) for a in ARMS}
            for a in ARMS:
                for s in SEEDS:
                    pv = preds[(a, s)][m]
                    ok24[a] += (pv == t24[m]) / len(SEEDS)
                    ok9[a] += (MACRO(pv) == MACRO(t24[m])) / len(SEEDS)
            lay = layer(block[m], ok24['T-STRATA'] - ok24['baseline'], ok9['T-STRATA'] - ok9['baseline'],
                        oa_b=ok24['baseline'], oa_o=ok24['T-STRATA'],
                        m9_b=ok9['baseline'], m9_o=ok9['T-STRATA'])
            frame_out['layers'][name] = lay

        # 目标类 F1（主层 conf∩nonFCS10）——T-STRATA vs baseline，逐 seed 平均
        m = conf & (~isf)
        tt = t24[m]
        classes = sorted(set(tt.tolist()))
        f1 = {}
        for c in classes:
            fb, fo = [], []
            for s in SEEDS:
                for a, acc in [('baseline', fb), ('T-STRATA', fo)]:
                    pv = preds[(a, s)][m]
                    tp = int(((tt == c) & (pv == c)).sum())
                    fp = int(((tt != c) & (pv == c)).sum())
                    fn = int(((tt == c) & (pv != c)).sum())
                    pr = tp / (tp + fp) if tp + fp else 0.0
                    rc = tp / (tp + fn) if tp + fn else 0.0
                    acc.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
            f1[int(c)] = dict(n=int((tt == c).sum()), f1_b=round(float(np.mean(fb)), 4),
                              f1_o=round(float(np.mean(fo)), 4),
                              dF1=round(float(np.mean(fo) - np.mean(fb)), 4))
        frame_out['target_f1_conf_nonfcs10'] = f1

        # 对照 t2_metrics summary
        mj = json.load(open(os.path.join(od, 't2_metrics_%s.json' % fr), encoding='utf-8'))
        sm = mj.get('summary', {})
        for key, mine in [('boot_nonfcs10_OA24_med', frame_out['layers'].get('nonfcs10', {}) or {}),
                          ('boot_conf_OA24_med', frame_out['layers'].get('conf', {}) or {}),
                          ('boot_conf_macro9_med', frame_out['layers'].get('conf', {}) or {})]:
            if key in sm and mine:
                rep = sm[key].get('mean')
                if key.endswith('macro9_med'):
                    ind = mine['dm9'] * 100
                else:
                    ind = mine['dOA24'] * 100
                if rep is not None and abs(rep - ind) > 0.05:
                    out['discrepancies'].append({'frame': fr, 'metric': key, 'reported_pp': rep,
                                                 'independent_pp': round(ind, 3)})
        out['frames'][fr] = frame_out

    os.makedirs(OUT, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = os.path.join(OUT, 'terrain_recompute_%s.json' % ts)
    json.dump(out, open(fp, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('[OK] -> %s' % fp)
    for fr, v in out['frames'].items():
        print('%s n_eval=%d n_conf=%d n_conf_nonfcs10=%d order_ok=%s' % (
            fr, v['n_eval'], v['n_conf'], v['n_conf_nonfcs10'], v['pred_order_matches_npz']))
        for nm, l in v['layers'].items():
            if l:
                print('   %-16s n=%d blocks=%d dOA24=%+.4f %s dm9=%+.4f %s' % (
                    nm, l['n'], l['n_blocks'], l['dOA24'], l['dOA24_ci'], l['dm9'], l['dm9_ci']))
    print('discrepancies:', out['discrepancies'])


if __name__ == '__main__':
    main()
