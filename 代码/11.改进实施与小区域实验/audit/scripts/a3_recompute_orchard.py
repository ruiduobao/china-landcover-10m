# -*- coding: utf-8 -*-
"""a3_recompute_orchard.py — 独立复算 orchard 园地先验实验主层指标（不重训 RF）

* 输入：<WS>/results/{rid}_{baseline,OCAP}_pred.csv（长表：region,arm,seed,row_id,t24,pred,isf,block,split）
        <WS>/results/summary.json（对照用）
* 规则源：doc70 §6.2/§6.3/§7；macro9 映射来自 v31_common.to_macro（与实验同源，避免口径漂移）
* 门槛（仅报告）：主门=非FCS10确认层 macro9 Δ点估≥−0.01 且块bootstrap95%下界≥−0.02
* 输出：audit/out/orchard_recompute_<ts>.json（含与 summary.json 的 n/CI 对照与差异标记）
* 用法：python a3_recompute_orchard.py [--ws F:/lc_work/improve_2023_pilot_20261015/orchard]
* 说明：关键审计点——summary 的 main_layer.n 是否等于**唯一点数**（去 seed 复制的行数）。
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
REGIONS = ['SC1', 'SC2', 'GD', 'NX']
TARGET = {'SC1': [4], 'SC2': [4], 'GD': [4, 5], 'NX': [22, 3]}


def block_bootstrap_ci(blocks, dpoint, nboot=NBOOT, seed=BOOT_SEED):
    ub, inv = np.unique(blocks, return_inverse=True)
    idx_by_b = [np.flatnonzero(inv == j) for j in range(len(ub))]
    rng = np.random.default_rng(seed)
    means = np.empty(nboot)
    for i in range(nboot):
        pick = rng.integers(0, len(ub), len(ub))
        sel = np.concatenate([idx_by_b[j] for j in pick])
        means[i] = dpoint[sel].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), int(len(ub))


def point_frame(m):
    """(seed,row_id) 长表 → 唯一点 frame（跨种子平均正确率）。"""
    m = m.copy()
    m['ok24_b'] = (m.pred_b == m.t24_b).astype(float)
    m['ok24_o'] = (m.pred_o == m.t24_b).astype(float)
    m['ok9_b'] = (MACRO(m.pred_b.to_numpy()) == MACRO(m.t24_b.to_numpy())).astype(float)
    m['ok9_o'] = (MACRO(m.pred_o.to_numpy()) == MACRO(m.t24_b.to_numpy())).astype(float)
    g = m.groupby('row_id', sort=False)
    return pd.DataFrame(dict(
        row_id=g.row_id.first(), t24=g.t24_b.first(), isf=g.isf_b.first(),
        block=g.block_b.first(), split=g.split_b.first(),
        ok24_b=g.ok24_b.mean(), ok24_o=g.ok24_o.mean(),
        ok9_b=g.ok9_b.mean(), ok9_o=g.ok9_o.mean(), nseed=g.size())).reset_index(drop=True)


def layer(pt, mask):
    sel = np.flatnonzero(mask)
    if len(sel) < 10:
        return None
    d9 = pt.ok9_o.to_numpy()[sel] - pt.ok9_b.to_numpy()[sel]
    d24 = pt.ok24_o.to_numpy()[sel] - pt.ok24_b.to_numpy()[sel]
    lo9, hi9, nblk = block_bootstrap_ci(pt.block.to_numpy()[sel], d9)
    lo24, hi24, _ = block_bootstrap_ci(pt.block.to_numpy()[sel], d24)
    return dict(n=int(len(sel)), n_blocks=nblk,
                dm9=round(float(d9.mean()), 4), dm9_ci=[round(lo9, 4), round(hi9, 4)],
                dOA24=round(float(d24.mean()), 4), dOA24_ci=[round(lo24, 4), round(hi24, 4)])


def f1_conf_nonfcs10(m, pt, seeds, target_classes):
    """目标类 F1（非FCS10确认层，逐 seed 计算后平均）。"""
    conf = pt[(pt.split == 'conf') & (~pt.isf)]
    rid_set = set(conf.row_id)
    out = {}
    for c in target_classes:
        nb = int((conf.t24 == c).sum())
        f1b, f1o = [], []
        for s in seeds:
            sub = m[m.seed == s]
            sub = sub[sub.row_id.isin(rid_set)]
            tt = sub.t24_b.to_numpy()
            for arm, pv, acc in [('b', sub.pred_b.to_numpy(), f1b), ('o', sub.pred_o.to_numpy(), f1o)]:
                tp = int(((tt == c) & (pv == c)).sum())
                fp = int(((tt != c) & (pv == c)).sum())
                fn = int(((tt == c) & (pv != c)).sum())
                pr = tp / (tp + fp) if tp + fp else 0.0
                rc = tp / (tp + fn) if tp + fn else 0.0
                acc.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
        out[int(c)] = dict(n=nb, f1_b=round(float(np.mean(f1b)), 4),
                           f1_o=round(float(np.mean(f1o)), 4),
                           dF1=round(float(np.mean(f1o) - np.mean(f1b)), 4))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ws', default=r'F:/lc_work/improve_2023_pilot_20261015/orchard')
    args = ap.parse_args()
    ws = args.ws
    res_dir = os.path.join(ws, 'results')
    summary = json.load(open(os.path.join(res_dir, 'summary.json'), encoding='utf-8'))

    out = {'generated_at': datetime.now().isoformat(timespec='seconds'),
           'workspace': ws, 'bootstrap': NBOOT, 'boot_seed': BOOT_SEED,
           'regions': {}, 'joint': {}, 'discrepancies': []}
    pts = {}
    ms = {}
    for rid in REGIONS:
        fb = os.path.join(res_dir, '%s_baseline_pred.csv' % rid)
        fo = os.path.join(res_dir, '%s_OCAP_pred.csv' % rid)
        if not (os.path.exists(fb) and os.path.exists(fo)):
            out['regions'][rid] = {'state': 'MISSING'}
            continue
        db, do = pd.read_csv(fb), pd.read_csv(fo)
        m = db.merge(do, on=['seed', 'row_id'], suffixes=('_b', '_o'))
        assert len(m) == len(db) == len(do), '%s arm point mismatch' % rid
        pt = point_frame(m)
        pts[rid] = pt
        ms[rid] = m
        f = pt.isf.to_numpy()
        sp = (pt.split.to_numpy() == 'conf')
        lay = layer(pt, sp & ~f)
        seeds = sorted(m.seed.unique())
        f1 = f1_conf_nonfcs10(m, pt, seeds, TARGET.get(rid, []))
        out['regions'][rid] = {'n_unique_conf_nonfcs10': lay['n'] if lay else None,
                               'main_layer': lay, 'target_f1_conf': f1}

        s = summary.get('regions', {}).get(rid, {})
        sm = s.get('main_layer') or {}
        if sm and lay:
            if sm.get('n') != lay['n']:
                out['discrepancies'].append({
                    'where': rid, 'field': 'main_layer.n',
                    'summary': sm.get('n'), 'independent_unique': lay['n'],
                    'ratio': round(sm.get('n') / lay['n'], 3) if lay['n'] else None})
            if abs(sm.get('dm9', 0) - lay['dm9']) > 1e-3:
                out['discrepancies'].append({'where': rid, 'field': 'dm9',
                                             'summary': sm.get('dm9'), 'independent': lay['dm9']})
            if sm.get('dm9_ci') and lay.get('dm9_ci'):
                if abs(sm['dm9_ci'][0] - lay['dm9_ci'][0]) > 1e-3 or \
                   abs(sm['dm9_ci'][1] - lay['dm9_ci'][1]) > 1e-3:
                    out['discrepancies'].append({'where': rid, 'field': 'dm9_ci',
                                                 'summary': sm['dm9_ci'], 'independent': lay['dm9_ci']})

    # 联合 SC1+SC2
    if 'SC1' in pts and 'SC2' in pts:
        pj = pd.concat([pts['SC1'], pts['SC2']], ignore_index=True)
        mj = pd.concat([ms['SC1'], ms['SC2']], ignore_index=True)
        fj = pj.isf.to_numpy()
        sj = (pj.split.to_numpy() == 'conf')
        jl = layer(pj, sj & ~fj)
        seeds = sorted(mj.seed.unique())
        jf1 = f1_conf_nonfcs10(mj, pj, seeds, [4])
        out['joint'] = {'n_unique_conf_nonfcs10': jl['n'] if jl else None,
                        'main_layer': jl, 'target_f1_conf': jf1,
                        'note': '描述性（doc70：联合仅在首预测前登记才可作门；否则只描述）'}
        sj_sum = summary.get('regions', {}).get('SC1+SC2_joint', {})
        if sj_sum.get('main_layer') and jl and sj_sum['main_layer'].get('n') != jl['n']:
            out['discrepancies'].append({'where': 'SC1+SC2_joint', 'field': 'main_layer.n',
                                         'summary': sj_sum['main_layer'].get('n'),
                                         'independent_unique': jl['n']})

    os.makedirs(OUT, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = os.path.join(OUT, 'orchard_recompute_%s.json' % ts)
    json.dump(out, open(fp, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('[OK] -> %s' % fp)
    for rid, v in out['regions'].items():
        ml = v.get('main_layer') or {}
        print('  %-5s n_unique=%s dm9=%s ci=%s' % (rid, v.get('n_unique_conf_nonfcs10'),
                                                   ml.get('dm9'), ml.get('dm9_ci')))
    print('  joint n_unique=%s dm9=%s ci=%s' % (out['joint'].get('n_unique_conf_nonfcs10'),
                                                (out['joint'].get('main_layer') or {}).get('dm9'),
                                                (out['joint'].get('main_layer') or {}).get('dm9_ci')))
    print('  discrepancies: %d' % len(out['discrepancies']))
    for d in out['discrepancies']:
        print('    ', d)


if __name__ == '__main__':
    main()
