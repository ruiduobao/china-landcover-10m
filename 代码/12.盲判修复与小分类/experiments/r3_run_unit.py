# -*- coding: utf-8 -*-
"""r3_run_unit.py — 标签修复实验·受门控 RF 运行器（train→infer→eval ≤1200s）

* 输入：<WS>/cache/view_<region>_<size>.npz（r2_build_view.py 产出）
* 规则源：prereg/label_repair_prereg.json（冻结）+ doc70 §3.2（1200s 硬截止）
* 门槛（硬门，缺一即拒绝启动）：
    1) **必须显式 --gate**（主 agent 下达『盲判及QC过门、修复规则冻结、开始实验』后才可传）
    2) verify_frozen().ok == True（prereg 正文未被改动）
    3) real20k 只允许一次：results 下已存在 real20k 预测即拒绝
    4) 只写 pred，不写真值；不做真值指标（解盲归 audit）
* 输出：results/pred_<region>_<size>_<arm>_s<seed>.csv（schema=region,arm,seed,vid,lon,lat,pred）
        results/run_<region>_<size>.json（计时/身份断言/selectedIDs 统计）
        logs/r3_<region>_<size>.log
* 用法：
    python r3_run_unit.py --region SC_F --size probe --gate
    python r3_run_unit.py --region SC_O --size probe --gate --arms baseline.v2
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import lr_common as LC  # noqa: E402

ARMS = ['baseline.v2', 'view-repair']
GATE_PHRASE = '盲判及QC过门、修复规则冻结、开始实验'


def emit(m, logf=None):
    s = '[%s] %s' % (time.strftime('%m-%d %H:%M:%S'), m)
    print(s, flush=True)
    if logf:
        logf.write(s + '\n'); logf.flush()


def identity_assertions(d, size, region):
    """X 行 ↔ selectedIDs 身份断言（不读真值）。"""
    chk = {}
    chk['n_base'] = int(d['base_X'].shape[0])
    chk['n_cand'] = int(d['cand_X'].shape[0])
    chk['n_base_ids'] = int(len(d['base_ids']))
    chk['n_cand_ids'] = int(len(d['cand_ids']))
    chk['rows_match_ids'] = bool(d['base_X'].shape[0] == len(d['base_ids']) and
                                 d['cand_X'].shape[0] == len(d['cand_ids']))
    chk['n_feats'] = int(d['base_X'].shape[1])
    chk['feats_ok'] = bool(d['base_X'].shape[1] == 64 and d['cand_X'].shape[1] == 64)
    chk['same_count'] = bool(d['base_X'].shape[0] == d['cand_X'].shape[0])
    # 非变更点（C）特征在两臂逐值一致：base 前 nC 行 == cand 前 nC 行
    nC = len(d['C_ids'])
    chk['C_feature_identical'] = bool(nC > 0 and np.array_equal(d['base_X'][:nC], d['cand_X'][:nC]))
    chk['n_C'] = int(nC)
    chk['n_backfill'] = int(len(d['back_ids']))
    chk['n_R'] = int(len(d['R_ids']))
    # 唯一性
    chk['unique_base_ids'] = bool(len(set(d['base_ids'].tolist())) == len(d['base_ids']))
    chk['unique_cand_ids'] = bool(len(set(d['cand_ids'].tolist())) == len(d['cand_ids']))
    chk['no_true_labels_in_pred'] = True   # 由写表 schema 保证
    hard = ['rows_match_ids', 'feats_ok', 'same_count', 'C_feature_identical',
            'unique_base_ids', 'unique_cand_ids']
    chk['hard_pass'] = all(bool(chk[k]) for k in hard)
    return chk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--region', required=True)
    ap.add_argument('--size', choices=['probe', 'real20k'], default='probe')
    ap.add_argument('--arms', default=','.join(ARMS))
    ap.add_argument('--gate', action='store_true',
                    help='确认主 agent 已下达：%s' % GATE_PHRASE)
    a = ap.parse_args()

    LC.ensure_dirs()
    logf = open(os.path.join(LC.LOGS, 'r3_%s_%s.log' % (a.region, a.size)), 'a', encoding='utf-8')

    # ---- 硬门 1/2 ----
    if not a.gate:
        emit('拒绝：缺 --gate（须主 agent 下达「%s」后才可训练）' % GATE_PHRASE, logf)
        return 3
    vf = LC.verify_frozen()
    if not vf['ok']:
        emit('拒绝：%s' % vf.get('reason'), logf)
        emit(json.dumps(vf, ensure_ascii=False), logf)
        return 3
    cfg = LC.load_prereg()
    size_n = cfg['design']['n_total_sizes'][a.size]

    # ---- 硬门 3：目标已冻结（R QC 后、看 V 真值前写死） ----
    tgt = LC.verify_targets_frozen()
    if not tgt['ok']:
        emit('拒绝：%s' % tgt.get('reason'), logf)
        emit('先跑 r0_prereg.py --freeze-targets（R QC 完成后）', logf)
        return 3

    # ---- 硬门 4：real20k 唯一一次 ----
    if a.size == 'real20k':
        prior = [f for f in os.listdir(LC.EXP) if f.startswith('pred_') and '_real20k_' in f]
        if prior:
            emit('拒绝：real20k 已运行过（%d 个预测件），只允许一次' % len(prior), logf)
            return 3

    npz = os.path.join(LC.CACHE, 'view_%s_%s.npz' % (a.region, a.size))
    if not os.path.exists(npz):
        emit('缺视图 %s —— 先跑 r2_build_view.py' % npz, logf)
        return 2
    d = np.load(npz, allow_pickle=False)

    idc = identity_assertions(d, a.size, a.region)
    if not idc['hard_pass']:
        emit('拒绝：身份断言未通过 %s' % json.dumps(idc, ensure_ascii=False), logf)
        return 4
    emit('身份断言通过：base %d / cand %d / C %d / R %d / backfill %d' % (
        idc['n_base'], idc['n_cand'], idc['n_C'], idc['n_R'], idc['n_backfill']), logf)

    DL, _ = LC.import_control()
    from sklearn.ensemble import RandomForestClassifier
    rf_p = dict(n_estimators=cfg['model']['n_estimators'],
                min_samples_leaf=cfg['model']['min_samples_leaf'],
                max_leaf_nodes=cfg['model']['max_leaf_nodes'],
                n_jobs=cfg['model']['n_jobs'])

    budget = DL.DeadlineBudget(limit_s=cfg['resources']['limit_s'])
    budget.start_window()                     # 训练调用开始 = 窗口零点
    runs, timed_out = [], False
    for seed in cfg['seeds']:
        for arm in [x for x in a.arms.split(',') if x.strip()]:
            if budget.remaining() <= 0:
                timed_out = True; break
            Xk = 'base_X' if arm == 'baseline.v2' else 'cand_X'
            yk = 'base_y' if arm == 'baseline.v2' else 'cand_y'
            X, y = d[Xk], d[yk].astype(np.int64)
            t0 = time.monotonic()
            rf = RandomForestClassifier(random_state=seed, **rf_p)
            rf.fit(X, y)
            fit_s = time.monotonic() - t0
            t1 = time.monotonic()
            pred = rf.predict(d['eval_X'])
            pred_s = time.monotonic() - t1
            # 预测表：**只写 pred，不附真值**
            csv = os.path.join(LC.EXP, 'pred_%s_%s_%s_s%d.csv' % (
                a.region, a.size, arm.replace('.', ''), seed))
            import csv as _csv
            with open(csv, 'w', newline='', encoding='utf-8') as f:
                w = _csv.writer(f)
                w.writerow(['region', 'arm', 'seed', 'vid', 'lon', 'lat', 'pred'])
                for i in range(len(pred)):
                    w.writerow([a.region, arm, seed, str(d['eval_vid'][i]),
                                '%.6f' % d['eval_lon'][i], '%.6f' % d['eval_lat'][i],
                                int(pred[i])])
            runs.append(dict(seed=seed, arm=arm, n_train=int(len(y)),
                             n_classes=int(len(np.unique(y))),
                             fit_s=round(fit_s, 2), predict_s=round(pred_s, 2),
                             pred_csv=csv, pred_csv_md5=LC.md5_file(csv),
                             train_class_counts={int(c): int(v) for c, v in
                                                 zip(*np.unique(y, return_counts=True))}))
            emit('seed %d ｜ %-12s 训练 %d 点(%d 类) fit %.1fs pred %.1fs → %s' % (
                seed, arm, len(y), len(np.unique(y)), fit_s, pred_s, os.path.basename(csv)), logf)
        if timed_out:
            break
    budget.mark_eval_end()
    win_s = budget.window_s()
    res = dict(region=a.region, size=a.size, n_total=size_n, arms=a.arms.split(','),
               gate=True, prereg=vf, seeds=cfg['seeds'], rf=rf_p,
               identity=idc, runs=runs, timed_out=bool(timed_out),
               deadline_window_s=round(win_s, 1) if win_s is not None else None,
               limit_s=cfg['resources']['limit_s'],
               budget_ok=(win_s is not None and win_s <= cfg['resources']['limit_s']),
               generated=time.strftime('%Y-%m-%d %H:%M:%S'),
               honesty=['pred 表不含真值；指标须由 audit 一次解盲后计算',
                        '本地 sklearn RF ≠ GEE smileRF',
                        'probe 非原生产 20k 结论' if a.size == 'probe' else 'real20k 仅一次受控旁证'])
    fp = os.path.join(LC.EXP, 'run_%s_%s.json' % (a.region, a.size))
    DL.atomic_write_json(fp, res)
    emit('→ %s（窗口 %.1fs / 上限 %ds；TIMEOUT=%s）' % (
        fp, win_s or -1, cfg['resources']['limit_s'], timed_out), logf)
    logf.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
