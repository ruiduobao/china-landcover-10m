# -*- coding: utf-8 -*-
"""t3_aggregate.py — T-SC **仅重汇总**（不训练）：doc70 主门 = conf∩非FCS10 的 3-seed 表

* 输入（只读，来自已完成的 t2 运行，不再拟合任何模型）：
    out/t1_arms_<F>_seed<seed>.npz（评价点真值/地形/块/split/fcs10 标记）
    out/t2_pred_<F>_<arm>_seed<seed>.csv（逐点预测；行序=评价点序）
* 规则源：doc70 §6.2（同点/来源分层）、§6.3（配对块 bootstrap）、§7（门）；
    主门口径 = **conf ∩ 非FCS10** 的三种子 mean OA24/macro9/类4 F1 与配对块 CI；与全非FCS10 主表分列。
* 门槛：**只重汇总，不训练**；退化判读必须 CI 支持，不以"2/3 种子点估 <-0.05"自动判可确认退化。
* 输出：out/t3_primary_gate.csv、out/t3_primary_gate.md、out/t3_metrics.json
* 用法：python t3_aggregate.py [--frames F1,F2] [--seeds 7,11,23] [--boot-B 2000] [--boot-block 0.25]
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding='utf-8')
sys.dont_write_bytecode = True

OUTD = os.path.join(HERE, 'out')
import t2_rf_eval as T2  # 复用 metrics/OR/条件错误/配对块 bootstrap（只读）


def load_pred(frame, arm, seed):
    fp = os.path.join(OUTD, 't2_pred_%s_%s_seed%d.csv' % (frame, arm, seed))
    with open(fp, encoding='utf-8') as f:
        r = csv.DictReader(f)
        return np.array([int(row['pred24']) for row in r], dtype=np.int64)


def strat_pack(t, p, a3, isf, split, blocks, B, seed, boot_block):
    """单臂单分层指标（OA24/macro9/macroF1/森林F1/OR/条件错误/块数）。"""
    if len(t) == 0:
        return None
    m = T2.metrics24(t, p)
    m['m9'] = round(float((T2.to_macro9(t) == T2.to_macro9(p)).mean()), 4)
    m['OR_pred'] = T2.or_stat(p, a3)
    m['OR_pred_boot'] = T2.block_boot_or(p, a3, blocks, B=B, seed=seed)
    m['cond_err_forest'] = T2.cond_error(t, p, a3, [4, 5, 6, 7])
    m['cond_err_c6'] = T2.cond_error(t, p, a3, [6])
    m['cond_err_c4'] = T2.cond_error(t, p, a3, [4])
    m['forest_f1'] = T2.forest_f1(p, t, '')
    m['n_blocks'] = int(len(np.unique(blocks)))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', default='F1,F2')
    ap.add_argument('--seeds', default='7,11,23')
    ap.add_argument('--boot-B', type=int, default=2000)
    ap.add_argument('--boot-block', type=float, default=0.25)
    a = ap.parse_args()
    frames = [s.strip() for s in a.frames.split(',') if s.strip()]
    seeds = [int(s) for s in a.seeds.split(',') if s.strip()]

    out = {'experiment': 'T-SC', 'stage': 't3_aggregate (仅重汇总，未训练)',
           'primary_gate_definition': 'conf ∩ 非FCS10（doc70 §6.2/§6.3）；与全非FCS10 主表分列',
           'boot_block_deg': a.boot_block, 'boot_B': a.boot_B, 'frames': {}}

    # 分层：主门 conf∩非FCS10；对照 conf、非FCS10(全)、全点
    def strata(isf, split):
        return {
            'conf_nonfcs10': (split == 0) & (isf == 0),   # doc70 主门
            'conf': split == 0,
            'nonfcs10_all': isf == 0,
            'all': np.ones(len(isf), bool),
        }

    for fr in frames:
        fr_out = {'seeds': seeds, 'strata': {}}
        # 先收集每 seed 的逐点数据
        per = {}
        for sd in seeds:
            d = np.load(os.path.join(OUTD, 't1_arms_%s_seed%d.npz' % (fr, sd)), allow_pickle=False)
            t = d['eval_y'].astype(np.int64); a3 = d['eval_a3'].astype(np.int64)
            isf = d['eval_is_fcs10'].astype(np.int64); split = d['eval_split'].astype(np.int64)
            lon = d['eval_lon']; lat = d['eval_lat']
            blocks = (np.floor(lon / a.boot_block).astype(np.int64) * 100000 +
                      np.floor(lat / a.boot_block).astype(np.int64)).astype(np.int64)
            preds = {arm: load_pred(fr, arm, sd) for arm in ('baseline', 'T-STRATA')}
            assert all(len(p) == len(t) for p in preds.values()), '预测长度与真值不一致'
            per[sd] = dict(t=t, a3=a3, isf=isf, split=split, blocks=blocks, preds=preds)

        for sname in ('conf_nonfcs10', 'conf', 'nonfcs10_all', 'all'):
            rows = {'baseline': [], 'T-STRATA': []}
            deltas = []
            n_s, nb_s = None, None
            for sd in seeds:
                q = per[sd]
                msk = strata(q['isf'], q['split'])[sname]
                if msk.sum() == 0:
                    continue
                n_s = int(msk.sum()); nb_s = int(len(np.unique(q['blocks'][msk])))
                for arm in ('baseline', 'T-STRATA'):
                    rows[arm].append(strat_pack(q['t'][msk], q['preds'][arm][msk], q['a3'][msk],
                                                q['isf'][msk], q['split'][msk], q['blocks'][msk],
                                                a.boot_B, sd, a.boot_block))
                dOA = T2.paired_boot(q['preds']['baseline'], q['preds']['T-STRATA'],
                                     q['t'], q['blocks'], msk, a.boot_B, sd, macro=False)
                dm9 = T2.paired_boot(q['preds']['baseline'], q['preds']['T-STRATA'],
                                     q['t'], q['blocks'], msk, a.boot_B, sd, macro=True)
                deltas.append(dict(seed=sd, n=int(msk.sum()), blocks=nb_s, OA24=dOA, macro9=dm9,
                                   dc4F1=[_v(rows['baseline'][-1], 'c4'), _v(rows['T-STRATA'][-1], 'c4')],
                                   dc6F1=[_v(rows['baseline'][-1], 'c6'), _v(rows['T-STRATA'][-1], 'c6')]))

            def mean_of(arm, key):
                vs = [_v(r, key) for r in rows[arm]]
                vs = [v for v in vs if v is not None]
                return dict(mean=round(float(np.mean(vs)), 4) if vs else None,
                            per_seed=[None if v is None else round(float(v), 4) for v in [_v(r, key) for r in rows[arm]]])
            summ = {}
            for key in ('OA', 'm9', 'macroF1'):
                summ[key] = {arm: mean_of(arm, key) for arm in ('baseline', 'T-STRATA')}
            summ['c4F1'] = {arm: mean_of(arm, 'c4') for arm in ('baseline', 'T-STRATA')}
            summ['c6F1'] = {arm: mean_of(arm, 'c6') for arm in ('baseline', 'T-STRATA')}

            def raw(arm, key):
                return [v for v in (_v(r, key) for r in rows[arm]) if v is not None]

            def dm(key, scale=100.0):
                b, tt = raw('baseline', key), raw('T-STRATA', key)
                return round((float(np.mean(tt)) - float(np.mean(b))) * scale, 4) if b and tt else None

            # Δ = 3-seed 均值之差（与表内均值列同源，避免与 bootstrap 中位混用）
            delta_OA = dm('OA'); delta_m9 = dm('m9'); delta_mf1 = dm('macroF1')
            delta_c4 = dm('c4', 1.0); delta_c6 = dm('c6', 1.0)
            # bootstrap 中位（逐种子，供 CI 参考，单列不与上面混算）
            dOA_med = [d['OA24']['med'] for d in deltas if d['OA24']['med'] is not None]
            dm9_med = [d['macro9']['med'] for d in deltas if d['macro9']['med'] is not None]
            fr_out['strata'][sname] = dict(
                n=n_s, n_blocks=nb_s, seeds_used=len(deltas),
                per_arm=summ, delta_per_seed=deltas,
                delta_OA24_mean_pp=delta_OA, delta_macro9_mean_pp=delta_m9,
                delta_macroF1_mean_pp=delta_mf1,
                delta_c4F1_mean=delta_c4, delta_c6F1_mean=delta_c6,
                boot_med_OA24_mean_pp=round(float(np.mean(dOA_med)), 2) if dOA_med else None,
                boot_med_macro9_mean_pp=round(float(np.mean(dm9_med)), 2) if dm9_med else None,
                delta_c4F1_per_seed=[None if None in d['dc4F1'] else round(d['dc4F1'][1] - d['dc4F1'][0], 4)
                                     for d in deltas])
        out['frames'][fr] = fr_out

    with open(os.path.join(OUTD, 't3_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    build_tables(out, os.path.join(OUTD, 't3_primary_gate.md'), os.path.join(OUTD, 't3_primary_gate.csv'))
    print('→ out/t3_metrics.json + t3_primary_gate.md/.csv')


def _v(r, key):
    if r is None:
        return None
    if key in ('c4', 'c6'):
        return r['forest_f1'][key]['f1']
    return r.get(key)


def build_tables(out, fp_md, fp_csv):
    rows_csv = []
    L = ['# T-SC 主门表（conf ∩ 非FCS10）与对照分层（仅重汇总，未训练）', '',
         '> 主门口径：**conf ∩ 非FCS10**（doc70 §6.2/§6.3），三种子；与"全非FCS10"主表分列。',
         '> Δ = T-STRATA − baseline；配对块 bootstrap（%.2f° 块，B=%d）。' % (
             out['boot_block_deg'], out['boot_B']),
         '> **退化判读须 CI 支持**；"多数种子点估 <-0.05"不自动等于可确认退化。', '',
         '## 表A 主门：conf ∩ 非FCS10', '',
         '| 框 | n | 块 | 指标 | baseline(3seed 均值) | T-STRATA(3seed 均值) | Δ均值 | Δ逐种子配对块CI(中位[95%]) |',
         '|---|---|---|---|---|---|---|---|']
    for fr, fv in out['frames'].items():
        s = fv['strata']['conf_nonfcs10']
        for key, lab, dk in (('OA', 'OA24', 'delta_OA24_mean_pp'), ('m9', 'macro9', 'delta_macro9_mean_pp')):
            b = s['per_arm'][key]['baseline']; t = s['per_arm'][key]['T-STRATA']
            cis = ' ; '.join('s%d %s%s' % (d['seed'], d['OA24']['med'] if key == 'OA' else d['macro9']['med'],
                                           d['OA24']['CI'] if key == 'OA' else d['macro9']['CI'])
                             for d in s['delta_per_seed'])
            L.append('| %s | %d | %d | %s | %s | %s | %s | %s |' % (
                fr, s['n'], s['n_blocks'], lab, b['mean'], t['mean'], s[dk], cis))
            rows_csv.append(dict(frame=fr, stratum='conf_nonfcs10', metric=lab, n=s['n'],
                                 n_blocks=s['n_blocks'], baseline_mean=b['mean'], TSTRATA_mean=t['mean'],
                                 delta_mean_pp=s[dk], delta_per_seed_CI=cis))
        # 类4 F1
        b4 = s['per_arm']['c4F1']['baseline']; t4 = s['per_arm']['c4F1']['T-STRATA']
        L.append('| %s | %d | %d | 类4 F1 | %s | %s | %s | 逐种子Δ=%s |' % (
            fr, s['n'], s['n_blocks'], b4['mean'], t4['mean'], s['delta_c4F1_mean'], s['delta_c4F1_per_seed']))
        rows_csv.append(dict(frame=fr, stratum='conf_nonfcs10', metric='c4F1', n=s['n'],
                             n_blocks=s['n_blocks'], baseline_mean=b4['mean'], TSTRATA_mean=t4['mean'],
                             delta_mean_pp=s['delta_c4F1_mean'], delta_per_seed_CI=s['delta_c4F1_per_seed']))
        b6 = s['per_arm']['c6F1']['baseline']; t6 = s['per_arm']['c6F1']['T-STRATA']
        L.append('| %s | %d | %d | 类6 F1 | %s | %s | — | 逐种子Δ=%s |' % (
            fr, s['n'], s['n_blocks'], b6['mean'], t6['mean'],
            [None if None in d['dc6F1'] else round(d['dc6F1'][1] - d['dc6F1'][0], 4) for d in s['delta_per_seed']]))
    L += ['', '## 表B 对照分层（全非FCS10 / conf / 全点；Δ 与主门分列，勿混）', '',
          '| 框 | 分层 | n | 块 | ΔOA24 均值(pp) | Δmacro9 均值(pp) | Δ类4F1 均值 |', '|---|---|---|---|---|---|---|']
    for fr, fv in out['frames'].items():
        for sname in ('nonfcs10_all', 'conf', 'all'):
            s = fv['strata'][sname]
            L.append('| %s | %s | %d | %d | %s | %s | %s |' % (
                fr, sname, s['n'], s['n_blocks'], s['delta_OA24_mean_pp'],
                s['delta_macro9_mean_pp'], s['delta_c4F1_mean']))
            rows_csv.append(dict(frame=fr, stratum=sname, metric='delta_summary', n=s['n'],
                                 n_blocks=s['n_blocks'], baseline_mean=None, TSTRATA_mean=None,
                                 delta_mean_pp=s['delta_OA24_mean_pp'],
                                 delta_per_seed_CI='Δm9=%s Δc4F1=%s' % (s['delta_macro9_mean_pp'], s['delta_c4F1_mean'])))
    open(fp_md, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    with open(fp_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys())); w.writeheader(); w.writerows(rows_csv)


if __name__ == '__main__':
    main()
