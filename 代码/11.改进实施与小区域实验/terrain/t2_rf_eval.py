# -*- coding: utf-8 -*-
"""t2_rf_eval.py — T-SC 四川地形分层受控分类：本地 RF 训练+预测+评价（**等主 agent slot**）

* 输入：out/t1_arms_<frame>_seed<seed>.npz（t1_prep.py 产出；含各臂训练点、评价点 AEF 与地形）
* 规则源：doc70 v1.0 §3.2（≤1200s，训练到评价落盘）、§4.3（同每类 count）、§6.2/§6.3（同点、分层、
    块 bootstrap、多种子）；AGENTS 硬约束 12（<2pp 单种子不可信）、4/5（留出/分层）
* 门槛（预注册，写死再跑）：
    - 主对：baseline vs T-STRATA（各≈20k，同每类 count，仅选点机制不同）；
    - 独立对：baseline_common vs F-VIEW（森林每类降到 strict 公共上限，非森林不变；总 N≤20k）；
    - 三个配对种子 {7,11,23}；RF=100/leaf2/nodes5000/64维 AEF；
    - **不以 OR 下降当精度提升**：同时报开发参考条件错误（阳/阴坡）与 F1；
    - 无独立针/阔真值 → 全部结论标"开发诊断"，不判伪影已修；
    - 每实验（该框×该对×三种子）wall-clock ≤1200s，超时终止自己并记 TIMEOUT。
* 输出：out/t2_metrics_<frame>.json、out/t2_pred_<frame>_seed<seed>.csv（逐点预测，非伪连续图）、
    out/t2_report_<frame>.md
* 用法：python t2_rf_eval.py --frame F1 [--seeds 7,11,23] [--n-jobs 6] [--deadline 1200]
    python t2_rf_eval.py --frame F1 --arms main      # 仅主对（12 fits 首轮）
    python t2_rf_eval.py --frame F1 --arms all       # 主对 + F-VIEW 对
"""
import argparse
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

WORK = r'F:/lc_work/improve_2023_pilot_20261015/terrain'
OUTD = os.path.join(WORK, 'out')
FOREST = (4, 5, 6, 7, 8)
SUNNY, SHADY = 0, 1
BNAME = {4: '常绿阔叶林', 5: '落叶阔叶林', 6: '常绿针叶林', 7: '落叶针叶林', 8: '针阔混交林',
         1: '草本旱地', 2: '乔灌园地', 3: '灌溉耕地', 9: '常绿灌丛', 10: '落叶灌丛', 11: '草地',
         12: '地衣苔藓', 13: '稀疏植被', 14: '木本沼泽', 15: '草本沼泽', 16: '湖河滩地',
         17: '盐渍湿地', 18: '红树林', 19: '海岸盐沼', 20: '潮滩', 21: '不透水面', 22: '裸地',
         23: '水体', 24: '永久冰雪'}


def emit(m):
    print('[%s] %s' % (time.strftime('%m-%d %H:%M:%S'), m), flush=True)


def metrics24(t, p):
    """OA、macroF1、逐类 {f1,pa,ua,n}（对 t∪p 出现的类）。"""
    t = np.asarray(t, int); p = np.asarray(p, int)
    n = len(t)
    oa = float((t == p).mean()) if n else 0.0
    per, f1s = {}, []
    for c in sorted(set(t.tolist()) | set(p.tolist())):
        tp = int(((t == c) & (p == c)).sum())
        fp = int(((t != c) & (p == c)).sum())
        fn = int(((t == c) & (p != c)).sum())
        if tp + fp + fn == 0:
            continue
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        per[int(c)] = dict(f1=round(f1, 4), pa=round(rc, 4), ua=round(pr, 4), n=tp + fn)
        f1s.append(f1)
    return dict(OA=round(oa, 4), macroF1=round(float(np.mean(f1s)) if f1s else 0.0, 4), n=n, per=per)


def to_macro9(t):
    """24 码 → 9 大类（macro9.json 口径，与 v31_common.to_macro 一致）。"""
    M = {'cropland': [1, 2, 3], 'forest': [4, 5, 6, 7, 8], 'shrubland': [9, 10],
         'grass_lowveg': [11, 12, 13], 'wetland': [14, 15, 16, 17, 18, 19, 20],
         'impervious': [21], 'bare': [22], 'water': [23], 'ice_snow': [24]}
    out = np.zeros(len(t), int)
    for i, (_, lst) in enumerate(sorted(M.items()), 1):
        out[np.isin(t, lst)] = i
    return out


def or_stat(lab, a3):
    """OR = [n(类6&阴)/n(类6&阳)] / [n(类4&阴)/n(类4&阳)]（slope>5 的阳/阴坡，口径同 terrain REPORT）。"""
    a = (lab == 6); b = (lab == 4)
    a_sh = int((a & (a3 == SHADY)).sum()); a_su = int((a & (a3 == SUNNY)).sum())
    b_sh = int((b & (a3 == SHADY)).sum()); b_su = int((b & (a3 == SUNNY)).sum())
    if min(a_sh, a_su, b_sh, b_su) == 0:
        return dict(OR=None, n6_sh=a_sh, n6_su=a_su, n4_sh=b_sh, n4_su=b_su)
    return dict(OR=round((a_sh * b_su) / (a_su * b_sh), 3),
                n6_sh=a_sh, n6_su=a_su, n4_sh=b_sh, n4_su=b_su)


def block_boot_or(lab, a3, blocks, B=2000, seed=7):
    """0.5°块 bootstrap 的 OR 分布 → 中位与 95% 区间（只重采样块计数）。"""
    ub = np.unique(blocks)
    if len(ub) < 4:
        return dict(OR_med=None, CI=[None, None], n_blocks=int(len(ub)), note='块<4，CI 不可用')
    rng = np.random.RandomState(seed)
    # 每块各类的 (6阳,6阴,4阳,4阴)
    tab = {}
    for b in ub:
        m = blocks == b
        tab[b] = (int(((lab == 6) & (a3 == SHADY) & m).sum()), int(((lab == 6) & (a3 == SUNNY) & m).sum()),
                  int(((lab == 4) & (a3 == SHADY) & m).sum()), int(((lab == 4) & (a3 == SUNNY) & m).sum()))
    arr = np.array([tab[b] for b in ub], float)
    vals = []
    for _ in range(B):
        pick = rng.randint(0, len(ub), len(ub))
        s = arr[pick].sum(0)
        if min(s[0], s[1], s[2], s[3]) == 0:
            continue
        vals.append((s[0] * s[3]) / (s[1] * s[2]))
    if not vals:
        return dict(OR_med=None, CI=[None, None], n_blocks=int(len(ub)))
    vals = np.array(vals)
    return dict(OR_med=round(float(np.median(vals)), 3),
                CI=[round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)],
                n_blocks=int(len(ub)), B=int(len(vals)))


def cond_error(t, p, a3, cls):
    """指定真值类 cls 的阳/阴坡条件错误率（真值∈cls 中预测≠cls 的比例）。"""
    m = np.isin(t, cls)
    out = {}
    for name, code in (('sunny', SUNNY), ('shady', SHADY)):
        mm = m & (a3 == code)
        out[name] = dict(n=int(mm.sum()),
                         err=round(float((p[mm] != t[mm]).mean()), 4) if mm.sum() else None)
    if out['sunny']['err'] is not None and out['shady']['err'] is not None:
        out['diff_shady_minus_sunny'] = round(out['shady']['err'] - out['sunny']['err'], 4)
    return out


def forest_f1(pred_pack, t, name):
    """森林 4/5/6/7 的 F1 与 4/6 合并、4/5/6/7 合并的 F1。"""
    out = {}
    for c in (4, 5, 6, 7):
        m = t == c
        out['c%d' % c] = dict(n=int(m.sum()),
                              f1=round(float(_f1(t, pred_pack, c)), 4) if m.sum() else None)
    out['c4_6'] = dict(n=int((np.isin(t, [4, 6])).sum()), f1=round(float(_f1(t, pred_pack, [4, 6])), 4))
    out['c4_5_6_7'] = dict(n=int((np.isin(t, [4, 5, 6, 7])).sum()),
                           f1=round(float(_f1(t, pred_pack, [4, 5, 6, 7])), 4))
    return out


def _f1(t, p, cls):
    cls = np.atleast_1d(cls)
    tt = np.isin(t, cls); pp = np.isin(p, cls)
    tp = int((tt & pp).sum()); fp = int((~tt & pp).sum()); fn = int((tt & ~pp).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return 2 * pr * rc / (pr + rc) if pr + rc else 0.0


def paired_boot(predA, predB, t, blocks, mask, B=2000, seed=7, macro=False):
    """配对块 bootstrap：Δ=acc(B)−acc(A)（pp），按 0.25° 块重采样；macro=True 用 9 大类。"""
    if macro:
        t = to_macro9(t); predA = to_macro9(predA); predB = to_macro9(predB)
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
        a = float((predA[sel] == t[sel]).mean())
        b = float((predB[sel] == t[sel]).mean())
        vals.append((b - a) * 100)
    if not vals:
        return dict(med=None, CI=[None, None], n_blocks=int(len(ub)))
    v = np.array(vals)
    return dict(med=round(float(np.median(v)), 2),
                CI=[round(float(np.percentile(v, 2.5)), 2), round(float(np.percentile(v, 97.5)), 2)],
                n_blocks=int(len(ub)), B=int(len(v)))


def eval_arm(pred, t, a3, isf, blocks, split):
    """一个臂在一组评价点上的全套指标（全点 + fcs10/非fcs10 分层 + OR + 条件错误）。"""
    def pack(m):
        if m.sum() == 0:
            return None
        mm = metrics24(t[m], pred[m])
        mm['m9'] = round(float((to_macro9(t[m]) == to_macro9(pred[m])).mean()), 4)
        mm['OR_pred'] = or_stat(pred[m], a3[m])
        mm['OR_pred_boot'] = block_boot_or(pred[m], a3[m], blocks[m])
        mm['cond_err_forest'] = cond_error(t[m], pred[m], a3[m], [4, 5, 6, 7])
        mm['cond_err_c6'] = cond_error(t[m], pred[m], a3[m], [6])
        mm['cond_err_c4'] = cond_error(t[m], pred[m], a3[m], [4])
        mm['forest_f1'] = forest_f1(pred[m], t[m], '')
        return mm
    out = {'all': pack(np.ones(len(t), bool))}
    if isf is not None:
        out['fcs10'] = pack(isf == 1)
        out['nonfcs10'] = pack(isf == 0)
    if split is not None:
        out['dev'] = pack(split == 1)
        out['conf'] = pack(split == 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frame', default='F1')
    ap.add_argument('--seeds', default='7,11,23')
    ap.add_argument('--n-jobs', type=int, default=6)
    ap.add_argument('--trees', type=int, default=100)
    ap.add_argument('--arms', default='main', choices=['main', 'all'],
                    help='main=baseline vs T-STRATA（首轮 12 fits）；all=另加 F-VIEW 独立对')
    ap.add_argument('--deadline', type=int, default=1200)
    ap.add_argument('--boot-B', type=int, default=2000)
    ap.add_argument('--boot-block', type=float, default=0.25,
                    help='bootstrap 块尺度（默认=注册的 0.25°，与 dev/conf 拆分同尺度）')
    ap.add_argument('--dry-run', action='store_true',
                    help='不训练 RF：用多数类常量预测跑通评价/报表管路（验证脚本，非结论）')
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(',') if s.strip()]
    t_start = time.time()

    from sklearn.ensemble import RandomForestClassifier

    runs = []
    for seed in seeds:
        fp = os.path.join(OUTD, 't1_arms_%s_seed%d.npz' % (a.frame, seed))
        if not os.path.exists(fp):
            emit('缺 %s — 先跑 t1_prep.py' % fp); sys.exit(2)
        d = np.load(fp, allow_pickle=False)
        Xe = d['eval_X']; t = d['eval_y'].astype(np.int64)
        a3 = d['eval_a3'].astype(np.int64)
        isf = d['eval_is_fcs10'].astype(np.int64)
        split = d['eval_split'].astype(np.int64)
        lon, lat = d['eval_lon'], d['eval_lat']
        # 空间块（bootstrap；尺度=注册的 0.25°，与 dev/conf 拆分同尺度，doc70 §6.1）
        blocks = (np.floor(lon / a.boot_block).astype(np.int64) * 100000 +
                  np.floor(lat / a.boot_block).astype(np.int64)).astype(np.int64)
        pairs = [('baseline', 'base_X', 'base_y'), ('T-STRATA', 'terr_X', 'terr_y')]
        if a.arms == 'all':
            pairs += [('baseline_common', 'bc_X', 'bc_y'), ('F-VIEW', 'fv_X', 'fv_y')]
        run = dict(seed=seed, frame=a.frame, n_eval=int(len(t)), arms={})
        preds = {}
        for name, kx, ky in pairs:
            if time.time() - t_start > a.deadline:
                run['TIMEOUT'] = True
                emit('TIMEOUT：超 deadline %ds（训练前检查），中止本轮' % a.deadline)
                break
            X = d[kx]; y = d[ky].astype(np.int64)
            if a.dry_run:
                pred = np.full(len(t), int(np.bincount(y).argmax()), np.int64)
                run['arms'][name] = dict(n_train=int(len(y)), n_classes=int(len(np.unique(y))),
                                         fit_seconds=0.0, dry_run=True)
                preds[name] = pred
                emit('seed %d ｜ %-15s DRY-RUN（常量预测，不训练）' % (seed, name))
                continue
            rf = RandomForestClassifier(n_estimators=a.trees, min_samples_leaf=2,
                                        max_leaf_nodes=5000, random_state=seed, n_jobs=a.n_jobs)
            t0 = time.time(); rf.fit(X, y); fit_s = time.time() - t0
            pred = rf.predict(Xe)
            preds[name] = pred
            run['arms'][name] = dict(n_train=int(len(y)), n_classes=int(len(np.unique(y))),
                                     fit_seconds=round(fit_s, 1))
            emit('seed %d ｜ %-15s 训练 %d 点（%d 类）%.1fs' % (seed, name, len(y), len(np.unique(y)), fit_s))
            # 逐点预测 CSV（非伪连续图）
            if name in ('baseline', 'T-STRATA', 'F-VIEW'):
                import csv
                with open(os.path.join(OUTD, 't2_pred_%s_%s_seed%d.csv' % (a.frame, name, seed)),
                          'w', newline='', encoding='utf-8') as f:
                    w = csv.writer(f); w.writerow(['lon', 'lat', 'a3', 'true24', 'pred24', 'is_fcs10'])
                    for i in range(len(t)):
                        w.writerow(['%.6f' % lon[i], '%.6f' % lat[i], int(a3[i]),
                                    int(t[i]), int(pred[i]), int(isf[i])])
            if time.time() - t_start > a.deadline:
                run['TIMEOUT'] = True
                emit('TIMEOUT：超过 deadline %ds，终止本轮' % a.deadline)
                break
        # 评价
        for name in preds:
            run['arms'][name]['metrics'] = eval_arm(preds[name], t, a3, isf, blocks, split)
        # 训练标签自身的坡向结构（机制量，doc69 §5.2 M1）
        run['train_label_aspect'] = {}
        for name, kx, ky in pairs:
            y = d[ky].astype(np.int64)
            aa = d[name.replace('baseline_common', 'bc').replace('baseline', 'base')
                   .replace('T-STRATA', 'terr').replace('F-VIEW', 'fv') + '_a3'].astype(np.int64)
            run['train_label_aspect'][name] = or_stat(y, aa) if len(aa) == len(y) else None
        # 配对 Δ（baseline vs T-STRATA；baseline_common vs F-VIEW）
        def delta(p1, p2, k='all'):
            m1 = run['arms'][p1]['metrics'][k]; m2 = run['arms'][p2]['metrics'][k]
            if m1 is None or m2 is None:
                return None
            return dict(OA_pp=round((m2['OA'] - m1['OA']) * 100, 2),
                        m9_pp=round((m2['m9'] - m1['m9']) * 100, 2),
                        macroF1_pp=round((m2['macroF1'] - m1['macroF1']) * 100, 2),
                        OR_pred=[m1['OR_pred']['OR'], m2['OR_pred']['OR']],
                        f1_c4=[m1['forest_f1']['c4']['f1'], m2['forest_f1']['c4']['f1']],
                        f1_c6=[m1['forest_f1']['c6']['f1'], m2['forest_f1']['c6']['f1']])
        run['delta_main'] = delta('baseline', 'T-STRATA')
        # 主对配对块 bootstrap（各分层 ΔOA24 / Δmacro9，doc70 §6.3/§7）
        if 'baseline' in preds and 'T-STRATA' in preds:
            strata_def = {'all': np.ones(len(t), bool), 'fcs10': isf == 1, 'nonfcs10': isf == 0,
                          'dev': split == 1, 'conf': split == 0}
            run['delta_main_boot'] = {}
            for sname, msk in strata_def.items():
                if msk.sum() == 0:
                    continue
                run['delta_main_boot'][sname] = dict(
                    n=int(msk.sum()),
                    OA24=paired_boot(preds['baseline'], preds['T-STRATA'], t, blocks, msk,
                                     a.boot_B, seed, macro=False),
                    macro9=paired_boot(preds['baseline'], preds['T-STRATA'], t, blocks, msk,
                                       a.boot_B, seed, macro=True))
        if a.arms == 'all':
            run['delta_fview'] = delta('baseline_common', 'F-VIEW')
        runs.append(run)
        if run.get('TIMEOUT'):
            break

    # 汇总（种子均值/极差）
    def agg(keypath):
        vals = []
        for r in runs:
            try:
                v = r
                for k in keypath:
                    v = v[k]
                vals.append(v)
            except Exception:
                vals.append(None)
        vv = [x for x in vals if x is not None]
        if not vv:
            return dict(mean=None, range=None, per_seed=vals)
        return dict(mean=round(float(np.mean(vv)), 3), range=round(float(np.max(vv) - np.min(vv)), 3),
                    per_seed=[None if x is None else round(float(x), 3) for x in vals])

    summary = {
        'OA_pp_main': agg(['delta_main', 'OA_pp']),
        'm9_pp_main': agg(['delta_main', 'm9_pp']),
        'OA_pp_fview': agg(['delta_fview', 'OA_pp']),
        'm9_pp_fview': agg(['delta_fview', 'm9_pp']),
    }
    for sname in ('nonfcs10', 'conf', 'fcs10', 'dev'):
        summary['boot_%s_OA24_med' % sname] = agg(['delta_main_boot', sname, 'OA24', 'med'])
        summary['boot_%s_macro9_med' % sname] = agg(['delta_main_boot', sname, 'macro9', 'med'])
    out = dict(experiment='T-SC', frame=a.frame, seeds=seeds, arms_mode=a.arms,
               generated=time.strftime('%Y-%m-%d %H:%M:%S'),
               deadline_s=a.deadline, wall_seconds=round(time.time() - t_start, 1),
               prereg={'main_pair': 'baseline vs T-STRATA（各≈20k，同每类 count，仅选点机制不同）',
                       'fview_pair': 'baseline_common vs F-VIEW（森林每类降到 strict 公共上限，非森林不变）',
                       'rule': '不以 OR 下降当精度提升；同时报条件错误与 F1；无独立针/阔真值 → 开发诊断',
                       'seeds': '配对 7/11/23'},
               runs=runs, summary=summary,
               honesty=['w2∩川 开发参考为 FCS10 系教师标签，非独立真值；全点 OA 受 fcs10 层机械主导',
                        '本地 sklearn RF ≠ GEE smileRF；20k 抽样为统计复现',
                        '无独立针/阔真值 → 不判伪影已修；OR 与条件错误仅开发口径',
                        'OR_pred 为评价点上预测标签的坡向比；非全域密集图（无 64band 栅格）'])
    fpj = os.path.join(OUTD, 't2_metrics_%s.json' % a.frame)
    with open(fpj, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    emit('→ %s（wall %.1fs）' % (fpj, out['wall_seconds']))
    build_md(out, os.path.join(OUTD, 't2_report_%s.md' % a.frame))
    emit('→ out/t2_report_%s.md' % a.frame)


def build_md(out, fp):
    L = ['# T-SC 四川地形分层受控分类 · %s（开发口径）' % out['frame'], '',
         '> 生成 %s ｜ 种子 %s ｜ 臂 %s ｜ wall %.1fs / deadline %ds' % (
             out['generated'], out['seeds'], out['arms_mode'], out['wall_seconds'], out['deadline_s']),
         '> **开发诊断**：参考为 w2∩川 五窗开发点（FCS10 系教师标签），非独立真值；',
         '> 无独立针/阔真值 → 不判伪影已修；**不以 OR 下降当精度提升**。', '',
         '## 预注册', '']
    for k, v in out['prereg'].items():
        L.append('- **%s**：%s' % (k, v))
    L += ['', '## 逐种子结果', '',
          '| seed | 臂 | n_train | 类数 | fit(s) | OA24 | macro9 | macroF1 | OR_pred | c4 F1 | c6 F1 |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in out['runs']:
        for name, v in r['arms'].items():
            m = v.get('metrics', {}).get('all')
            if m is None:
                L.append('| %d | %s | %d | %d | %.1f | — | — | — | — | — | — |' % (
                    r['seed'], name, v['n_train'], v['n_classes'], v['fit_seconds']))
                continue
            L.append('| %d | %s | %d | %d | %.1f | %.4f | %.4f | %.4f | %s | %s | %s |' % (
                r['seed'], name, v['n_train'], v['n_classes'], v['fit_seconds'],
                m['OA'], m['m9'], m['macroF1'],
                m['OR_pred']['OR'] if m['OR_pred']['OR'] is not None else '—',
                _f(m['forest_f1']['c4']['f1']), _f(m['forest_f1']['c6']['f1'])))
    L += ['', '## 配对 Δ（候选 − 基线）', '',
          '| seed | 对 | ΔOA24(pp) | Δmacro9(pp) | ΔmacroF1(pp) | OR 基线→候选 | c4 F1 基→候 | c6 F1 基→候 |',
          '|---|---|---|---|---|---|---|---|']
    for r in out['runs']:
        for lab, key in (('主对 baseline→T-STRATA', 'delta_main'),
                         ('独立对 baseline_common→F-VIEW', 'delta_fview')):
            dl = r.get(key)
            if not dl:
                continue
            L.append('| %d | %s | %+.2f | %+.2f | %+.2f | %s→%s | %s→%s | %s→%s |' % (
                r['seed'], lab, dl['OA_pp'], dl['m9_pp'], dl['macroF1_pp'],
                dl['OR_pred'][0], dl['OR_pred'][1],
                _f(dl['f1_c4'][0]), _f(dl['f1_c4'][1]), _f(dl['f1_c6'][0]), _f(dl['f1_c6'][1])))
    L += ['', '## 主对配对块 bootstrap（Δ = T-STRATA − baseline，pp；0.25° 块重采样）', '',
          '| seed | 分层 | n | ΔOA24 med [95%CI] | 块 | Δmacro9 med [95%CI] |', '|---|---|---|---|---|---|']
    for r in out['runs']:
        for sname, v in r.get('delta_main_boot', {}).items():
            o, m = v['OA24'], v['macro9']
            L.append('| %d | %s | %d | %s %s | %s | %s %s |' % (
                r['seed'], sname, v['n'], o['med'], o['CI'], o['n_blocks'], m['med'], m['CI']))
    L += ['', '## 阴阳坡条件错误（真值森林 4/5/6/7 中预测错的比例）', '',
          '| seed | 臂 | 阳坡 err | 阴坡 err | 阴−阳 | OR_pred block-CI | 有效块 |', '|---|---|---|---|---|---|---|']
    for r in out['runs']:
        for name, v in r['arms'].items():
            m = v.get('metrics', {}).get('all')
            if m is None:
                continue
            ce = m['cond_err_forest']
            b = m['OR_pred_boot']
            L.append('| %d | %s | %s | %s | %s | %s | %s |' % (
                r['seed'], name, _f(ce['sunny']['err']), _f(ce['shady']['err']),
                _f(ce.get('diff_shady_minus_sunny')),
                '%s %s' % (b.get('OR_med'), b.get('CI')), b.get('n_blocks')))
    L += ['', '## 训练标签自身坡向结构（机制量；OR=[c6阴/c6阳]/[c4阴/c4阳]）', '',
          '| seed | 臂 | OR(训练标签) | c6阴/c6阳 | c4阴/c4阳 |', '|---|---|---|---|---|']
    for r in out['runs']:
        for name, v in r.get('train_label_aspect', {}).items():
            if not v:
                continue
            L.append('| %d | %s | %s | %d/%d | %d/%d |' % (
                r['seed'], name, v['OR'], v['n6_sh'], v['n6_su'], v['n4_sh'], v['n4_su']))
    L += ['', '## 多种子汇总', '']
    for k, v in out['summary'].items():
        L.append('- **%s**：均值 %s ｜极差 %s ｜逐种子 %s' % (k, v['mean'], v['range'], v['per_seed']))
    L += ['', '## 诚实性声明', '']
    for s in out['honesty']:
        L.append('- %s' % s)
    open(fp, 'w', encoding='utf-8').write('\n'.join(L) + '\n')


def _f(x, nd=3):
    return '—' if x is None else ('%.*f' % (nd, x))


if __name__ == '__main__':
    main()
