# -*- coding: utf-8 -*-
"""t5_fview_run.py — F-VIEW 独立 common-support pair：baseline_common vs strict_common（3 seed）

* 输入：out/t4_fview_arms_<F>_seed<seed>.npz（t4_fview_prep.py；bc_X/sc_X + 评价点）
* 规则源：doc70 §4.3 F-VIEW / §6.2 / §6.3 / §7；主门 = conf ∩ 非FCS10，目标类 4 F1
* 门槛（预注册，写死再跑）：3 配对种子 {7,11,23}；RF=100/leaf2/nodes5000/64维；非森林两臂同点；
    class4 目标在 conf∩非FCS10 上判读，**n<30 → 支持不足**（不得改看全点层挑涨分）。
* 输出：out/t5_metrics_<F>.json、out/t5_report_<F>.md、out/t5_pred_<F>_<arm>_seed<seed>.csv
* 用法：python t5_fview_run.py --frame G2 [--n-jobs 10] [--deadline 1200]
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
sys.stdout.reconfigure(encoding='utf-8')
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUTD = os.path.join(HERE, 'out')
import t2_rf_eval as T2


def emit(m):
    print('[%s] %s' % (time.strftime('%m-%d %H:%M:%S'), m), flush=True)


def pack(t, p, a3, B, seed, blocks):
    m = T2.metrics24(t, p)
    m['m9'] = round(float((T2.to_macro9(t) == T2.to_macro9(p)).mean()), 4)
    m['OR_pred'] = T2.or_stat(p, a3)
    m['cond_err_forest'] = T2.cond_error(t, p, a3, [4, 5, 6, 7])
    m['forest_f1'] = T2.forest_f1(p, t, '')
    m['n_blocks'] = int(len(np.unique(blocks)))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frame', default='G2')
    ap.add_argument('--seeds', default='7,11,23')
    ap.add_argument('--n-jobs', type=int, default=10)
    ap.add_argument('--trees', type=int, default=100)
    ap.add_argument('--deadline', type=int, default=1200)
    ap.add_argument('--boot-B', type=int, default=2000)
    ap.add_argument('--boot-block', type=float, default=0.25)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(',') if s.strip()]
    t_start = time.time()
    from sklearn.ensemble import RandomForestClassifier

    runs = []
    for seed in seeds:
        fp = os.path.join(OUTD, 't4_fview_arms_%s_seed%d.npz' % (a.frame, seed))
        if not os.path.exists(fp):
            emit('缺 %s — 先跑 t4_fview_prep.py' % fp); sys.exit(2)
        d = np.load(fp, allow_pickle=False)
        Xe = d['eval_X']; t = d['eval_y'].astype(np.int64); a3 = d['eval_a3'].astype(np.int64)
        isf = d['eval_is_fcs10'].astype(np.int64); split = d['eval_split'].astype(np.int64)
        lon, lat = d['eval_lon'], d['eval_lat']
        blocks = (np.floor(lon / a.boot_block).astype(np.int64) * 100000 +
                  np.floor(lat / a.boot_block).astype(np.int64)).astype(np.int64)
        pairs = [('baseline_common', 'bc_X', 'bc_y'), ('strict_common', 'sc_X', 'sc_y')]
        run = dict(seed=seed, frame=a.frame, n_eval=int(len(t)), arms={})
        preds = {}
        for name, kx, ky in pairs:
            if time.time() - t_start > a.deadline:
                run['TIMEOUT'] = True; emit('TIMEOUT 训练前检查'); break
            X = d[kx]; y = d[ky].astype(np.int64)
            rf = RandomForestClassifier(n_estimators=a.trees, min_samples_leaf=2,
                                        max_leaf_nodes=5000, random_state=seed, n_jobs=a.n_jobs)
            t0 = time.time(); rf.fit(X, y); fs = time.time() - t0
            pred = rf.predict(Xe); preds[name] = pred
            run['arms'][name] = dict(n_train=int(len(y)), n_classes=int(len(np.unique(y))),
                                     fit_seconds=round(fs, 1))
            emit('seed %d ｜ %-16s 训练 %d 点（%d 类）%.1fs' % (seed, name, len(y), len(np.unique(y)), fs))
            with open(os.path.join(OUTD, 't5_pred_%s_%s_seed%d.csv' % (a.frame, name, seed)),
                      'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f); w.writerow(['lon', 'lat', 'true24', 'pred24', 'is_fcs10', 'split'])
                for i in range(len(t)):
                    w.writerow(['%.6f' % lon[i], '%.6f' % lat[i], int(t[i]), int(pred[i]),
                                int(isf[i]), int(split[i])])
        strata = {'conf_nonfcs10': (split == 0) & (isf == 0), 'conf': split == 0,
                  'nonfcs10_all': isf == 0, 'all': np.ones(len(t), bool), 'dev': split == 1}
        run['strata'] = {}
        for sname, msk in strata.items():
            if msk.sum() == 0:
                continue
            blk = {}
            for name in preds:
                blk[name] = pack(t[msk], preds[name][msk], a3[msk], a.boot_B, seed, blocks[msk])
            dOA = T2.paired_boot(preds['baseline_common'], preds['strict_common'], t, blocks, msk,
                                 a.boot_B, seed, macro=False) if len(preds) == 2 else None
            dm9 = T2.paired_boot(preds['baseline_common'], preds['strict_common'], t, blocks, msk,
                                 a.boot_B, seed, macro=True) if len(preds) == 2 else None
            run['strata'][sname] = dict(n=int(msk.sum()), n_blocks=int(blk['baseline_common']['n_blocks']),
                                        arms=blk, dOA24=dOA, dmacro9=dm9)
        runs.append(run)

    # 汇总：主门 conf∩非FCS10 的 3-seed 均值 + Δ
    def ms(sname, arm, key):
        vs = []
        for r in runs:
            s = r['strata'].get(sname)
            if not s:
                continue
            v = s['arms'][arm]['forest_f1'][key]['f1'] if key in ('c4', 'c6') else s['arms'][arm][key]
            if v is not None:
                vs.append(v)
        return vs

    out = dict(experiment='F-VIEW', frame=a.frame, seeds=seeds,
               generated=time.strftime('%Y-%m-%d %H:%M:%S'),
               deadline_s=a.deadline, wall_seconds=round(time.time() - t_start, 1),
               prereg={'pair': 'baseline_common vs strict_common（共同支持 count；非森林同点）',
                       'primary': 'conf ∩ 非FCS10，目标类 4 F1（n<30 支持不足）',
                       'seeds': '配对 7/11/23', 'no_cherry_pick': '不看其它层挑涨分'},
               runs=runs,
               honesty=['w2∩川 开发参考（FCS10 系教师标签）非独立真值；无独立针/阔真值',
                        '本地 sklearn RF ≠ GEE smileRF', 'strict 视图为旧教师标签质检子集，非新真值'])
    # 主门表
    gate = {}
    for sname in ('conf_nonfcs10', 'conf', 'nonfcs10_all', 'all'):
        g = {}
        for key, lab in (('OA', 'OA24'), ('m9', 'macro9'), ('c4', 'c4F1'), ('c6', 'c6F1')):
            b = ms(sname, 'baseline_common', key); st = ms(sname, 'strict_common', key)
            g[lab] = dict(baseline_mean=round(float(np.mean(b)), 4) if b else None,
                          strict_mean=round(float(np.mean(st)), 4) if st else None,
                          delta=round(float(np.mean(st)) - float(np.mean(b)), 4) if (b and st) else None,
                          n_seeds=len(b))
        # CI 来自主门 (取该层 逐种子 bootstrap)
        cis = []
        for r in runs:
            s = r['strata'].get(sname)
            if s and s.get('dOA24'):
                cis.append(dict(seed=r['seed'], OA24=s['dOA24'], macro9=s['dmacro9']))
        g['boot_CI'] = cis
        gate[sname] = g
    out['gate_tables'] = gate
    out['class4_conf_nonfcs10_n'] = int(((a_cache := None) or 0))
    for r in runs:
        s = r['strata'].get('conf_nonfcs10')
        if s:
            out['class4_conf_nonfcs10_n'] = int(s['arms']['baseline_common']['forest_f1']['c4']['n'])
            break
    with open(os.path.join(OUTD, 't5_metrics_%s.json' % a.frame), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    emit('→ out/t5_metrics_%s.json（wall %.1fs）' % (a.frame, out['wall_seconds']))
    build_md(out, os.path.join(OUTD, 't5_report_%s.md' % a.frame))
    emit('→ out/t5_report_%s.md' % a.frame)


def build_md(out, fp):
    L = ['# F-VIEW 独立 common-support pair · %s（开发口径）' % out['frame'], '',
         '> 生成 %s ｜ 种子 %s ｜ wall %.1fs / deadline %ds' % (
             out['generated'], out['seeds'], out['wall_seconds'], out['deadline_s']),
         '> **开发诊断**：参考非独立真值；无独立针/阔真值；不以任何层挑涨分。', '',
         '## 逐种子（n_train / 类数 / fit s / OA24 / macro9 / c4 F1 / c6 F1，全点口径）', '',
         '| seed | 臂 | n_train | 类 | fit(s) | OA24 | macro9 | c4 F1 | c6 F1 |', '|---|---|---|---|---|---|---|---|---|']
    for r in out['runs']:
        for name, v in r['arms'].items():
            s = r['strata']['all']['arms'][name]
            L.append('| %d | %s | %d | %d | %.1f | %.4f | %.4f | %s | %s |' % (
                r['seed'], name, v['n_train'], v['n_classes'], v['fit_seconds'], s['OA'], s['m9'],
                _f(s['forest_f1']['c4']['f1']), _f(s['forest_f1']['c6']['f1'])))
    L += ['', '## 主门与对照（Δ = strict_common − baseline_common，3 seed 均值）', '',
          '| 分层 | 指标 | baseline_common 均值 | strict_common 均值 | Δ |', '|---|---|---|---|---|']
    for sname, g in out['gate_tables'].items():
        for lab in ('OA24', 'macro9', 'c4F1', 'c6F1'):
            v = g[lab]
            L.append('| %s | %s | %s | %s | %s |' % (sname, lab, v['baseline_mean'], v['strict_mean'], v['delta']))
    L += ['', '> **主门 = conf ∩ 非FCS10**，目标类 4 F1；n<30 → 支持不足。', '',
          '## 配对块 bootstrap（0.25° 块，逐种子）', '',
          '| 分层 | seed | ΔOA24 中位[95%CI] | 块 | Δmacro9 中位[95%CI] |', '|---|---|---|---|---|']
    for sname, g in out['gate_tables'].items():
        for c in g['boot_CI']:
            L.append('| %s | %d | %s %s | %s | %s %s |' % (
                sname, c['seed'], c['OA24']['med'], c['OA24']['CI'], c['OA24']['n_blocks'],
                c['macro9']['med'], c['macro9']['CI']))
    L += ['', '> conf∩非FCS10 目标类4 有效 n = %s（<30 即支持不足，不作类级达标声明）' % out['class4_conf_nonfcs10_n'],
          '', '## 诚实性声明', '']
    for s in out['honesty']:
        L.append('- %s' % s)
    open(fp, 'w', encoding='utf-8').write('\n'.join(L) + '\n')


def _f(x, nd=3):
    return '—' if x is None else ('%.*f' % (nd, x))


if __name__ == '__main__':
    main()
