# -*- coding: utf-8 -*-
"""s4_eval.py — 逐任务精度 + 臂级聚合（同口径双评：doc37 §五.1）
  A 臂：30 类预测 → 归并 24 类评估（同时报原生 30 类口径）
  B/D 臂：24 类原生评估
  大类（macro9）两套都算。D 臂 = 组内 5 成员本地众数投票（并列取小码）。
产物：results/metrics/<job>.json、agg.json、perclass.csv
"""
import os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v31_common as VC
import numpy as np
import pandas as pd

FOREST_COMP = [4, 5, 6, 7, 8]


def vote(preds):
    """ preds: list of ndarray（等长）→ 众数，并列取小码。"""
    P = np.vstack(preds)
    out = np.zeros(P.shape[1], dtype=int)
    for i in range(P.shape[1]):
        vals, cnts = np.unique(P[:, i], return_counts=True)
        out[i] = vals[cnts.argmax()]  # unique 已排序，argmax 取首个最大=小码
    return out


def eval_job(j):
    fp = j.get('result_csv')
    if not fp or not os.path.exists(fp):
        return None
    w = j['window']
    ev = pd.read_parquet(eval_fp_local(w))
    pred = pd.read_csv(fp)
    m = ev[['row_id', 'class_new']].merge(
        pred[['rid', 'classification']], left_on='row_id', right_on='rid', how='inner')
    if m['classification'].isna().any():
        m = m.dropna(subset=['classification'])
    t30 = m['class_new'].to_numpy(int)
    p = m['classification'].to_numpy(int)
    t24 = VC.to_v31(t30)
    p24 = VC.to_v31(p) if j['arm'] == 'A' else p
    res = dict(n=len(m), n_dropped=int(len(ev) - len(m)))
    res['v24'] = VC.metrics(t24, p24)
    tm, pm = VC.to_macro(t24), VC.to_macro(p24)
    res['m9'] = VC.metrics(tm, pm)
    if j['arm'] == 'A':
        res['raw30'] = VC.metrics(t30, p)
    res['pred'] = p24.tolist()          # D 投票/后续复用
    res['rids'] = m['row_id'].tolist()
    return res


def eval_fp_local(w):
    fp = os.path.join(VC.DATA, 'eval_%s.parquet' % w)
    return fp if os.path.exists(fp) else os.path.join(VC.EVAL_DIR_GEE, 'eval_%s.parquet' % w)


def main(only_missing=True):
    jobs = VC.jload(VC.JOBS_FP, [])
    if not jobs:
        VC.emit('无任务'); return
    n_new = 0
    for j in jobs:
        out_fp = os.path.join(VC.MET, j['job_id'] + '.json')
        if only_missing and os.path.exists(out_fp):
            continue
        if not j.get('done') or not j.get('result_csv'):
            continue
        try:
            r = eval_job(j)
        except Exception as e:
            VC.emit('%s 评估失败: %s' % (j['job_id'], str(e)[:110]))
            continue
        if r is None:
            continue
        VC.jsave(r, out_fp)
        n_new += 1
        VC.emit('评估 %s: OA24=%.4f mF1=%.4f macroOA=%.4f (n=%d)' % (
            j['job_id'], r['v24']['OA'], r['v24']['macroF1'], r['m9']['OA'], r['n']))
    VC.emit('s4: 新评估 %d' % n_new)


def aggregate():
    """臂级聚合 + 配对比较 → agg.json / perclass.csv"""
    jobs = VC.jload(VC.JOBS_FP, [])
    names = VC.V31_NAMES()
    agg = {}
    rows = []
    for j in jobs:
        fp = os.path.join(VC.MET, j['job_id'] + '.json')
        if not os.path.exists(fp):
            continue
        r = VC.jload(fp)
        w, arm = j['window'], j['arm']
        a = agg.setdefault(w, {}).setdefault(arm, {})
        if arm in ('A', 'B', 'C'):
            a.setdefault('seeds', {})[str(j['seed'])] = dict(
                OA=r['v24']['OA'], macroF1=r['v24']['macroF1'],
                macroOA=r['m9']['OA'], raw30=r.get('raw30', {}).get('OA'),
                raw30_mF1=r.get('raw30', {}).get('macroF1'))
        else:
            a.setdefault('members', {})[str(j['member'])] = dict(
                OA=r['v24']['OA'], macroF1=r['v24']['macroF1'], macroOA=r['m9']['OA'])
        for c, d in r['v24']['per'].items():
            rows.append(dict(window=w, arm=arm, cls=c, name=names.get(str(c), '?'),
                             f1=d['f1'], ua=d['ua'], pa=d['pa'], n=d['n']))
        agg[w].setdefault('_n_eval', r['n'])
    # D 组投票（需要各成员预测向量对齐同一 rids）
    votes = {}
    for j in jobs:
        if j['arm'] != 'D':
            continue
        fp = os.path.join(VC.MET, j['job_id'] + '.json')
        if not os.path.exists(fp):
            continue
        r = VC.jload(fp)
        key = (j['window'], j['group'])
        votes.setdefault(key, {})[j['member']] = (np.array(r['rids']), np.array(r['pred']))
    for (w, g), mem in votes.items():
        if len(mem) < 2:
            continue
        rids0 = mem[min(mem)][0]
        if any(not np.array_equal(v[0], rids0) for v in mem.values()):
            VC.emit('%s D_g%d rids 不齐，跳过投票' % (w, g))
            continue
        pv = vote([p for _k, (_r, p) in sorted(mem.items())])
        ev = pd.read_parquet(eval_fp_local(w))
        mm = ev[['row_id', 'class_new']].merge(
            pd.DataFrame(dict(row_id=rids0)), on='row_id', how='inner')
        t24 = VC.to_v31(mm['class_new'].to_numpy(int))
        res = VC.metrics(t24, pv)
        agg[w].setdefault('D', {}).setdefault('vote', {})[str(g)] = dict(
            OA=res['OA'], macroF1=res['macroF1'], macroOA=VC.metrics(
                VC.to_macro(t24), VC.to_macro(pv))['OA'], k=len(mem))
        for c, d in res['per'].items():
            rows.append(dict(window=w, arm='Dvote', cls=c, name=names.get(str(c), '?'),
                             f1=d['f1'], ua=d['ua'], pa=d['pa'], n=d['n']))
    VC.jsave(dict(agg=agg, gen=time.strftime('%Y-%m-%d %H:%M')), os.path.join(VC.MET, 'agg.json'))
    pd.DataFrame(rows).to_csv(os.path.join(VC.MET, 'perclass.csv'), index=False, encoding='utf-8-sig')
    VC.emit('s4 聚合完成 → agg.json / perclass.csv')
    return agg


if __name__ == '__main__':
    main(only_missing='--agg' not in sys.argv)
    if '--agg' in sys.argv:
        aggregate()
