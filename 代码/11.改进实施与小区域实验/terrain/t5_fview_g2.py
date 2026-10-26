# -*- coding: utf-8 -*-
"""t5_fview_g2.py — F-VIEW G2 独立 common-support pair：baseline_common vs strict_common

* 输入：out/fview_g2/t4_fview_arms_G2_seed<seed>.npz（t4b_fview_prep_g2.py，修复版）
* 规则源：doc70 §4.3 F-VIEW / §6.2 / §6.3 / §7；主门 = conf ∩ 非FCS10，目标类 4 F1
* 预注册门槛（写死再跑，不看其它层挑涨分）：
    - 3 配对种子 {7,11,23}；RF=100 树 / leaf2 / nodes5000 / 64 维 AEF / n_jobs=10；
    - 非森林两臂同点（prep v2 已断言）；每类 count 两臂相同；
    - 目标：conf∩非FCS10 上类 4 F1 平均增量 ≥ +0.03 且配对块 bootstrap CI 支持正向；
      n<30 → 支持不足，不得改看全点层；
    - 组级非劣：同层 macro9 点估 ≥ −0.01 且块 bootstrap 95% 下界 ≥ −0.02；
    - 新退化：同层参考 n≥30 的类 ΔF1 ≤ −0.05 → 拒绝；
    - 全点 / FCS10 / 非FCS10 完整表同时报告；一切为开发口径、非独立真值。
* 三阶段（供 control/run_experiment.py 硬截止串联，窗口 = train 开始 → eval 结束 ≤1200s）：
    --stage train  6 次 fit（每 seed 两臂）→ out/fview_g2/models/*.joblib + train_manifest.json
    --stage infer  逐点预测 → out/fview_g2/pred_G2_<arm>_seed<s>.csv + preds_G2.npz
    --stage eval   同点评价 / 分层表 / 配对块 bootstrap / 采纳门 → metrics + report
* 用法：python t5_fview_g2.py --stage train|infer|eval [--n-jobs 10]
"""
import argparse
import csv
import hashlib
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
import t2_rf_eval as T2   # 复用 metrics24 / to_macro9 / forest_f1 / or_stat / cond_error（只读）

FOREST = (4, 5, 6, 7)
GATES = dict(target_dF1=0.03, group_d_pt=-0.01, group_ci=-0.02, new_regression_dF1=-0.05)
MIN_N_CLASS = 30


def emit(m):
    print('[%s] %s' % (time.strftime('%m-%d %H:%M:%S'), m), flush=True)


def sha16(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def blocks_of(lon, lat, bs):
    return (np.floor(lon / bs).astype(np.int64) * 100000 + np.floor(lat / bs).astype(np.int64)).astype(np.int64)


def f1_of(t, p, cls):
    return float(T2._f1(t, p, cls))


def paired_boot_f1(predA, predB, t, cls, blocks, mask, B=2000, seed=7):
    """配对块 bootstrap 的 ΔF1(cls) = F1(B) − F1(A)（块重采样；仅返回分布中位与 95%CI）。"""
    ub = np.unique(blocks[mask])
    if len(ub) < 4:
        return dict(med=None, CI=[None, None], n_blocks=int(len(ub)), note='块<4')
    rng = np.random.RandomState(seed)
    idxb = {b: np.flatnonzero(mask & (blocks == b)) for b in ub}
    vals = []
    for _ in range(B):
        pick = rng.randint(0, len(ub), len(ub))
        sel = np.concatenate([idxb[ub[p]] for p in pick])
        if len(sel) == 0:
            continue
        vals.append((f1_of(t[sel], predB[sel], cls) - f1_of(t[sel], predA[sel], cls)) * 100)
    if not vals:
        return dict(med=None, CI=[None, None], n_blocks=int(len(ub)))
    v = np.array(vals)
    return dict(med=round(float(np.median(v)), 2),
                CI=[round(float(np.percentile(v, 2.5)), 2), round(float(np.percentile(v, 97.5)), 2)],
                n_blocks=int(len(ub)), B=int(len(v)))


# --------------------------------------------------------------------------- #
def stage_train(a, indir):
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    mdir = os.path.join(indir, 'models')
    os.makedirs(mdir, exist_ok=True)
    man = dict(stage='train', frame=a.frame, seeds=[], n_jobs=a.n_jobs,
               rf_params=dict(n_estimators=a.trees, min_samples_leaf=2, max_leaf_nodes=5000),
               generated=time.strftime('%Y-%m-%d %H:%M:%S'), fits=[])
    t_all = time.time()
    for seed in [int(s) for s in a.seeds.split(',') if s.strip()]:
        fp = os.path.join(indir, 't4_fview_arms_%s_seed%d.npz' % (a.frame, seed))
        d = np.load(fp, allow_pickle=False)
        rec = dict(seed=seed, npz=os.path.basename(fp),
                   npz_sha16=hashlib.sha256(open(fp, 'rb').read()).hexdigest()[:16], arms={})
        for name, kx, ky, ki in (('baseline_common', 'bc_X', 'bc_y', 'bc_idx'),
                                 ('strict_common', 'sc_X', 'sc_y', 'sc_idx')):
            X = d[kx]; y = d[ky].astype(np.int64)
            rf = RandomForestClassifier(n_estimators=a.trees, min_samples_leaf=2,
                                        max_leaf_nodes=5000, random_state=seed, n_jobs=a.n_jobs)
            t0 = time.time(); rf.fit(X, y); fs = time.time() - t0
            mp = os.path.join(mdir, 'rf_%s_%s_seed%d.joblib' % (a.frame, name, seed))
            joblib.dump(rf, mp)
            cnt = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
            idx = d[ki].astype(np.int64) if ki in d.files else None
            rec['arms'][name] = dict(n_train=int(len(y)), shape=list(X.shape), n_classes=len(cnt),
                                     fit_seconds=round(fs, 3), model=os.path.basename(mp),
                                     train_counts=cnt,
                                     train_ids_sha16=sha16(np.sort(idx)) if idx is not None else None,
                                     train_ids_n=int(len(idx)) if idx is not None else None,
                                     train_Xy_sha16=sha16(X.astype(np.float32)) + '|' + sha16(y))
            man['fits'].append(dict(seed=seed, arm=name, fit_seconds=round(fs, 3),
                                    n_train=int(len(y)), shape=list(X.shape), n_classes=len(cnt)))
            emit('fit seed %d %-16s n=%d shape=%s 类=%d %.2fs' % (seed, name, len(y), X.shape, len(cnt), fs))
        man['seeds'].append(rec)
    man['wall_seconds'] = round(time.time() - t_all, 1)
    with open(os.path.join(indir, 'train_manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(man, f, ensure_ascii=False, indent=1, default=float)
    emit('→ %s（train 阶段 wall %.1fs）' % (os.path.join(indir, 'train_manifest.json'), man['wall_seconds']))


def stage_infer(a, indir):
    import joblib
    mdir = os.path.join(indir, 'models')
    out = {}
    t_all = time.time()
    meta = {}
    for seed in [int(s) for s in a.seeds.split(',') if s.strip()]:
        d = np.load(os.path.join(indir, 't4_fview_arms_%s_seed%d.npz' % (a.frame, seed)), allow_pickle=False)
        Xe = d['eval_X']
        meta[seed] = dict(eval_y=d['eval_y'].astype(np.int64), eval_a3=d['eval_a3'].astype(np.int64),
                          eval_split=d['eval_split'].astype(np.int64),
                          eval_is_fcs10=d['eval_is_fcs10'].astype(np.int64),
                          eval_lon=d['eval_lon'], eval_lat=d['eval_lat'],
                          eval_block=d['eval_block'].astype(str),
                          eval_row_id=d['eval_row_id'].astype(np.int64))
        for name in ('baseline_common', 'strict_common'):
            rf = joblib.load(os.path.join(mdir, 'rf_%s_%s_seed%d.joblib' % (a.frame, name, seed)))
            t0 = time.time(); pred = rf.predict(Xe).astype(np.int64); inf = time.time() - t0
            out['pred_%s_seed%d' % (name, seed)] = pred
            m = meta[seed]
            with open(os.path.join(indir, 'pred_%s_%s_seed%d.csv' % (a.frame, name, seed)),
                      'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['row_id', 'lon', 'lat', 'true24', 'pred24', 'is_fcs10', 'split', 'block'])
                for i in range(len(pred)):
                    w.writerow([int(m['eval_row_id'][i]), '%.6f' % m['eval_lon'][i], '%.6f' % m['eval_lat'][i],
                                int(m['eval_y'][i]), int(pred[i]), int(m['eval_is_fcs10'][i]),
                                'conf' if m['eval_split'][i] == 0 else 'dev', m['eval_block'][i]])
            emit('infer seed %d %-16s n=%d %.2fs  pred_sha16=%s' % (
                seed, name, len(pred), inf, sha16(pred)))
    # 评价元数据 + 预测统一存 npz（供 eval 阶段精确复用）
    save = dict(out)
    for seed, m in meta.items():
        for k, v in m.items():
            save['%s_seed%d' % (k, seed)] = v
    np.savez_compressed(os.path.join(indir, 'preds_%s.npz' % a.frame), **save)
    emit('→ %s（infer 阶段 wall %.1fs）' % (os.path.join(indir, 'preds_%s.npz' % a.frame),
                                          round(time.time() - t_all, 1)))


def _stratum_block(t, preds, a3, isf, split, blocks, mask, bootB, seed):
    blk = {}
    for name, p in preds.items():
        m = T2.metrics24(t[mask], p[mask])
        m['m9'] = round(float((T2.to_macro9(t[mask]) == T2.to_macro9(p[mask])).mean()), 4)
        m['OR_pred'] = T2.or_stat(p[mask], a3[mask])
        m['cond_err_forest'] = T2.cond_error(t[mask], p[mask], a3[mask], list(FOREST))
        m['forest_f1'] = T2.forest_f1(p[mask], t[mask], '')
        blk[name] = m
    d = dict(n=int(mask.sum()), n_blocks=int(len(np.unique(blocks[mask]))), arms=blk)
    for key, cls in (('OA24', None), ('macro9', 'm9')):
        d['d' + key] = T2.paired_boot(preds['baseline_common'], preds['strict_common'], t, blocks, mask,
                                      bootB, seed, macro=(cls is not None))
    for lab, cls in (('c4', 4), ('c6', 6), ('c4_6', [4, 6])):
        d['dF1_' + lab] = paired_boot_f1(preds['baseline_common'], preds['strict_common'], t, cls,
                                         blocks, mask, bootB, seed)
    return d


def stage_eval(a, indir):
    t_all = time.time()
    z = np.load(os.path.join(indir, 'preds_%s.npz' % a.frame), allow_pickle=False)
    seeds = [int(s) for s in a.seeds.split(',') if s.strip()]
    runs = []
    for seed in seeds:
        t = z['eval_y_seed%d' % seed]; a3 = z['eval_a3_seed%d' % seed]
        isf = z['eval_is_fcs10_seed%d' % seed]; split = z['eval_split_seed%d' % seed]
        lon = z['eval_lon_seed%d' % seed]; lat = z['eval_lat_seed%d' % seed]
        rowid = z['eval_row_id_seed%d' % seed]
        blocks = blocks_of(lon, lat, a.boot_block)
        preds = {'baseline_common': z['pred_baseline_common_seed%d' % seed],
                 'strict_common': z['pred_strict_common_seed%d' % seed]}
        strata = {'conf_nonfcs10': (split == 0) & (isf == 0), 'conf': split == 0,
                  'dev': split == 1, 'nonfcs10_all': isf == 0, 'fcs10': isf == 1,
                  'all': np.ones(len(t), bool)}
        run = dict(seed=seed, n_eval=int(len(t)), n_conf=int((split == 0).sum()),
                   n_conf_nonfcs10=int(((split == 0) & (isf == 0)).sum()),
                   eval_rowid_sha16=sha16(np.sort(rowid)), strata={})
        for sname, msk in strata.items():
            if msk.sum() == 0:
                continue
            run['strata'][sname] = _stratum_block(t, preds, a3, isf, split, blocks, msk, a.boot_B, seed)
        runs.append(run)

    def agg(sname, arm, path):
        vs = []
        for r in runs:
            s = r['strata'].get(sname)
            if not s:
                continue
            v = s['arms'][arm]
            for k in path:
                v = v[k]
            vs.append(v)
        return vs

    def mean(x):
        vv = [x for x in x if x is not None]
        return round(float(np.mean(vv)), 4) if vv else None

    def per_agg(sname, arm):
        """该层逐类 n/ua/pa/f1 的 3-seed 均值（缺失类不补零）。"""
        bag = {}
        for r in runs:
            s = r['strata'].get(sname)
            if not s:
                continue
            for c, v in s['arms'][arm]['per'].items():
                bag.setdefault(int(c), []).append(v)
        return {c: dict(n=round(float(np.mean([v['n'] for v in lst])), 1),
                        f1=round(float(np.mean([v['f1'] for v in lst])), 4),
                        pa=round(float(np.mean([v['pa'] for v in lst])), 4),
                        ua=round(float(np.mean([v['ua'] for v in lst])), 4),
                        n_seeds=len(lst)) for c, lst in bag.items()}

    gate_tables = {}
    for sname in ('conf_nonfcs10', 'conf', 'dev', 'nonfcs10_all', 'fcs10', 'all'):
        if sname not in runs[0]['strata']:
            continue
        g = {'n': runs[0]['strata'][sname]['n'], 'n_blocks': runs[0]['strata'][sname]['n_blocks']}
        for lab, path in (('OA24', ['OA']), ('macro9', ['m9']), ('macroF1', ['macroF1']),
                          ('c4F1', ['forest_f1', 'c4', 'f1']), ('c4n', ['forest_f1', 'c4', 'n']),
                          ('c6F1', ['forest_f1', 'c6', 'f1']), ('c6n', ['forest_f1', 'c6', 'n']),
                          ('c4_6F1', ['forest_f1', 'c4_6', 'f1']),
                          ('c4_5_6_7F1', ['forest_f1', 'c4_5_6_7', 'f1'])):
            b = agg(sname, 'baseline_common', path); st = agg(sname, 'strict_common', path)
            g[lab] = dict(baseline_mean=mean(b), strict_mean=mean(st),
                          delta=(round(mean(st) - mean(b), 4) if (mean(b) is not None and mean(st) is not None) else None),
                          per_seed_baseline=b, per_seed_strict=st)
        g['boot'] = [dict(seed=r['seed'], dOA24=r['strata'][sname]['dOA24'],
                          dmacro9=r['strata'][sname]['dmacro9'],
                          dF1_c4=r['strata'][sname]['dF1_c4'],
                          dF1_c6=r['strata'][sname]['dF1_c6'],
                          dF1_c4_6=r['strata'][sname]['dF1_c4_6']) for r in runs if sname in r['strata']]
        g['per_baseline'] = per_agg(sname, 'baseline_common')
        g['per_strict'] = per_agg(sname, 'strict_common')
        gate_tables[sname] = g

    # ---- 采纳门（主层 = conf ∩ 非FCS10）----
    main = gate_tables['conf_nonfcs10']
    n4 = main['c4n']['baseline_mean']
    d4 = main['c4F1']['delta']
    boot_main = main['boot'][0] if main['boot'] else {}
    ci4 = boot_main.get('dF1_c4', {})
    d_m9 = main['macro9']['delta']
    ci_m9 = boot_main.get('dmacro9', {})
    target_status = ('INSUFFICIENT_SUPPORT' if (n4 is None or n4 < MIN_N_CLASS) else
                     ('TARGET_IMPROVED' if (d4 is not None and d4 >= GATES['target_dF1'] and
                                            (ci4.get('CI') or [None, None])[0] is not None and
                                            ci4['CI'][0] > 0) else 'NOT_IMPROVED'))
    group_ok = bool(d_m9 is not None and d_m9 >= GATES['group_d_pt'] and
                    (ci_m9.get('CI') or [None, None])[0] is not None and
                    ci_m9['CI'][0] >= GATES['group_ci'])
    regressions = []
    # 逐类新退化（主层，n≥30；3-seed 均值 F1）
    base_per = main['per_baseline']; st_per = main['per_strict']
    for c in sorted(set(base_per) | set(st_per)):
        nb = base_per.get(c, {}).get('n', 0); ns = st_per.get(c, {}).get('n', 0)
        if max(nb, ns) < MIN_N_CLASS:
            continue
        fb = base_per.get(c, {}).get('f1', 0.0); fs = st_per.get(c, {}).get('f1', 0.0)
        if round(fs - fb, 4) <= GATES['new_regression_dF1']:
            regressions.append(dict(cls=int(c), n=max(nb, ns), f1_baseline=fb, f1_strict=fs,
                                    delta=round(fs - fb, 4)))
    verdict = dict(
        main_stratum='conf ∩ 非FCS10', n_conf_nonfcs10=int(main['n']),
        target_class=4, target_n=int(n4) if n4 is not None else None,
        target_dF1=d4, target_ci=ci4, target_status=target_status,
        group_macro9_delta=d_m9, group_macro9_ci=ci_m9, group_noninferior=group_ok,
        new_regressions=regressions,
        adoption=('ACCEPT_EXPERIMENTAL' if (target_status == 'TARGET_IMPROVED' and group_ok and not regressions)
                  else 'REJECT'),
        note='开发口径（w2∩川 教师标签），非独立真值；不得外推省域；n<30 不作类级达标声明')

    out = dict(experiment='F-VIEW', frame=a.frame, seeds=seeds,
               generated=time.strftime('%Y-%m-%d %H:%M:%S'), wall_seconds=round(time.time() - t_all, 1),
               boot=dict(B=a.boot_B, block_deg=a.boot_block),
               gates=GATES, prereg=dict(
                   pair='baseline_common vs strict_common（共同支持 count；非森林同点，prep v2 断言）',
                   primary='conf ∩ 非FCS10，目标类 4 F1（n<30 支持不足）',
                   seeds='配对 7/11/23', no_cherry_pick='不看其它层挑涨分'),
               runs=runs, gate_tables=gate_tables, verdict=verdict,
               honesty=['w2∩川 五窗开发参考（FCS10 系教师标签），非独立人工真值',
                        '本地 sklearn RF ≠ GEE smileRF；非森林两臂同点、森林取 strict 公共支持',
                        'strict 视图为旧教师标签质检子集（doc65），非新真值；无独立针/阔真值',
                        'strict_short 为空（strict 支持充足），未做补数'])
    # 逐层逐类完整表（含 n/UA/PA/F1）落 CSV：逐种子 + 分层
    rows = []
    for r in runs:
        for sname, s in r['strata'].items():
            for arm in ('baseline_common', 'strict_common'):
                for c, v in sorted(s['arms'][arm]['per'].items(), key=lambda kv: int(kv[0])):
                    rows.append(dict(stratum=sname, seed=r['seed'], arm=arm, cls=int(c), n=v.get('n'),
                                     ua=v.get('ua'), pa=v.get('pa'), f1=v.get('f1')))
    with open(os.path.join(indir, 't5_tables_%s.csv' % a.frame), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['stratum', 'seed', 'arm', 'cls', 'n', 'ua', 'pa', 'f1'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(os.path.join(indir, 't5_metrics_%s.json' % a.frame), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    build_md(out, os.path.join(indir, 't5_report_%s.md' % a.frame))
    emit('→ t5_metrics_%s.json + t5_report_%s.md + t5_tables_%s.csv（eval 阶段 wall %.1fs）' % (
        a.frame, a.frame, a.frame, out['wall_seconds']))
    emit('VERDICT adoption=%s target=%s (n=%s, dF1=%s, CI=%s) | group_macro9 Δ=%s CI=%s ok=%s | 新退化 %d' % (
        verdict['adoption'], target_status, verdict['target_n'], d4, ci4.get('CI'),
        d_m9, ci_m9.get('CI'), group_ok, len(regressions)))


def build_md(out, fp):
    v = out['verdict']
    L = ['# F-VIEW 独立 common-support pair · %s（开发口径）' % out['frame'], '',
         '> 生成 %s ｜ 种子 %s ｜ 评价 wall %.1fs' % (out['generated'], out['seeds'], out['wall_seconds']),
         '> **开发诊断**：参考为 w2∩川 教师标签，非独立真值；不以任何层挑涨分。', '',
         '## 预注册', '']
    for k, s in out['prereg'].items():
        L.append('- **%s**：%s' % (k, s))
    L += ['', '## 采纳门（主层 = conf ∩ 非FCS10）', '',
          '| 项 | 值 |', '|---|---|',
          '| 主层 n（点/块） | %s / %s |' % (out['gate_tables']['conf_nonfcs10']['n'],
                                            out['gate_tables']['conf_nonfcs10']['n_blocks']),
          '| 目标类 4 参考 n | %s |' % v['target_n'],
          '| ΔF1(类4) | %s |' % v['target_dF1'],
          '| ΔF1(类4) 块 bootstrap 中位[95%%CI] | %s %s |' % (v['target_ci'].get('med'), v['target_ci'].get('CI')),
          '| 目标状态（门 ≥ +%.2f 且 CI 支持正向） | **%s** |' % (out['gates']['target_dF1'], v['target_status']),
          '| Δmacro9（点估） | %s |' % v['group_macro9_delta'],
          '| Δmacro9 块 bootstrap 中位[95%%CI] | %s %s |' % (v['group_macro9_ci'].get('med'),
                                                            v['group_macro9_ci'].get('CI')),
          '| 组级非劣（点估 ≥ %.2f 且下界 ≥ %.2f） | %s |' % (out['gates']['group_d_pt'],
                                                            out['gates']['group_ci'], v['group_noninferior']),
          '| 新退化类（n≥30 且 ΔF1 ≤ %.2f） | %s |' % (out['gates']['new_regression_dF1'],
                                                     v['new_regressions'] or '无'),
          '| **实验级采纳** | **%s** |' % v['adoption'], '',
          '> %s' % v['note'], '',
          '## 分层总表（3 seed 均值；Δ = strict_common − baseline_common）', '',
          '| 分层 | n | 块 | OA24 b/s/Δ | macro9 b/s/Δ | c4F1 b/s/Δ (n) | c6F1 b/s/Δ (n) |',
          '|---|---|---|---|---|---|---|']
    for sname, g in out['gate_tables'].items():
        L.append('| %s | %d | %d | %s / %s / %s | %s / %s / %s | %s / %s / %s (%s) | %s / %s / %s (%s) |' % (
            sname, g['n'], g['n_blocks'],
            g['OA24']['baseline_mean'], g['OA24']['strict_mean'], g['OA24']['delta'],
            g['macro9']['baseline_mean'], g['macro9']['strict_mean'], g['macro9']['delta'],
            g['c4F1']['baseline_mean'], g['c4F1']['strict_mean'], g['c4F1']['delta'], g['c4n']['baseline_mean'],
            g['c6F1']['baseline_mean'], g['c6F1']['strict_mean'], g['c6F1']['delta'], g['c6n']['baseline_mean']))
    L += ['', '## 配对块 bootstrap（0.25° 块，逐种子；Δ 单位 pp）', '',
          '| 分层 | seed | ΔOA24 中位[CI] | Δmacro9 中位[CI] | ΔF1(类4) 中位[CI] | ΔF1(类6) 中位[CI] |',
          '|---|---|---|---|---|---|']
    for sname, g in out['gate_tables'].items():
        for b in g['boot']:
            L.append('| %s | %d | %s %s | %s %s | %s %s | %s %s |' % (
                sname, b['seed'], b['dOA24'].get('med'), b['dOA24'].get('CI'),
                b['dmacro9'].get('med'), b['dmacro9'].get('CI'),
                b['dF1_c4'].get('med'), b['dF1_c4'].get('CI'),
                b['dF1_c6'].get('med'), b['dF1_c6'].get('CI')))
    L += ['', '## 逐种子主层（conf ∩ 非FCS10）', '',
          '| seed | 臂 | n | OA24 | macro9 | c4 F1(n) | c6 F1(n) |', '|---|---|---|---|---|---|---|']
    for r in out['runs']:
        s = r['strata']['conf_nonfcs10']
        for arm in ('baseline_common', 'strict_common'):
            m = s['arms'][arm]
            L.append('| %d | %s | %d | %.4f | %.4f | %s (%d) | %s (%d) |' % (
                r['seed'], arm, m['n'], m['OA'], m['m9'],
                m['forest_f1']['c4']['f1'], m['forest_f1']['c4']['n'],
                m['forest_f1']['c6']['f1'], m['forest_f1']['c6']['n']))
    L += ['', '## 诚实性声明', '']
    for s in out['honesty']:
        L.append('- %s' % s)
    open(fp, 'w', encoding='utf-8').write('\n'.join(L) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['train', 'infer', 'eval'])
    ap.add_argument('--frame', default='G2')
    ap.add_argument('--seeds', default='7,11,23')
    ap.add_argument('--n-jobs', type=int, default=10)
    ap.add_argument('--trees', type=int, default=100)
    ap.add_argument('--boot-B', type=int, default=2000)
    ap.add_argument('--boot-block', type=float, default=0.25)
    ap.add_argument('--indir', default=os.path.join(HERE, 'out', 'fview_g2'))
    a = ap.parse_args()
    os.makedirs(a.indir, exist_ok=True)
    {'train': stage_train, 'infer': stage_infer, 'eval': stage_eval}[a.stage](a, a.indir)


if __name__ == '__main__':
    main()
