# -*- coding: utf-8 -*-
"""e0_eval_inventory.py — 园地先验实验·只读探索：五窗 eval 全类计数 + 候选小框清单

* 输入：F:/lc_work/v31_exp/data/yearly/eval_2023_w{2,3,4}.parquet（留出窗，含 A00..A63）
        省界 DataV GCJ-02（经 p5common 只读加载）
* 规则源：doc69 §8（小区域受控实验）、c7_local_ab.py（排除/归省/分层口径逐条复用）、
        AGENTS 硬约束 4/5/12
* 门槛：本脚本**零训练、零预测**——只做输入盘点与候选框枚举（不看预测结果选框）
* 输出：F:/lc_work/improve_2023_pilot_20261015/orchard/explore/
        e0_eval_class_counts.csv / e0_eval_class_counts.md / e0_candidate_boxes.csv /
        e0_inventory.json
* 用法：PYTHONDONTWRITEBYTECODE=1 python e0_eval_inventory.py
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys, json, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, r'F:/地理所/论文/中国土地覆盖数据_2017-2024/代码/9.第二次分类川粤宁')
sys.path.insert(0, r'F:/lc_work/prod5p_2023/mosaic')
sys.path.insert(0, r'F:/lc_work/v31_exp/code')
import c4_eval_second as C4
import p5common as PC
import v31_common as VC

EVAL_DIR = r'F:/lc_work/v31_exp/data/yearly'
OUTDIR = HERE
FEATS = ['A%02d' % i for i in range(64)]
WINDOWS = ['w2', 'w3', 'w4']
YEAR = 2023
PROV2 = [('川', '四川省'), ('粤', '广东省'), ('宁', '宁夏回族自治区')]


def emit(m):
    print('[%s] %s' % (time.strftime('%m-%d %H:%M:%S'), m), flush=True)


def load_holdout(year, windows):
    frames = []
    for w in windows:
        fp = os.path.join(EVAL_DIR, 'eval_%d_%s.parquet' % (year, w))
        d = pd.read_parquet(fp, columns=['row_id', 'lon', 'lat', 'class_new', 'src'] + FEATS)
        d['win'] = w
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def main():
    names = VC.V31_NAMES()
    ev_raw = load_holdout(YEAR, WINDOWS)
    n0 = len(ev_raw)
    keep, xd = C4.build_exclusions(ev_raw)
    ev = ev_raw[keep].reset_index(drop=True)
    del ev_raw
    emit('留出点 %d → 保留 %d（排除明细 %s）' % (n0, len(ev), xd))

    prov_full, bdist = C4.assign_prov(ev.lon.to_numpy(), ev.lat.to_numpy(), list(PC.PROV_SHORT))
    ev['prov_full'] = prov_full
    ev['near_boundary'] = bdist < 0.005
    t24 = VC.to_v31(ev.class_new.to_numpy().astype(np.int64))
    ev['t24'] = t24
    isf = PC.is_fcs10(ev.src).to_numpy()
    ev['fcs10'] = isf
    assert int((~np.isfinite(ev[FEATS].to_numpy(np.float32))).sum()) == 0

    # ---- 全类计数：窗×省×类（含 fcs10/非fcs10 分层） ----
    rows = []
    for w in WINDOWS:
        mw = ev.win.to_numpy() == w
        for tag, full in PROV2:
            mp = mw & (ev.prov_full.to_numpy() == full)
            sub = ev[mp]
            for c in sorted(set(sub.t24.tolist())):
                sc = sub[sub.t24 == c]
                rows.append(dict(win=w, prov=tag, cl=int(c), name=names.get(str(int(c)), '?'),
                                 n=int(len(sc)), n_fcs10=int(sc.fcs10.sum()),
                                 n_nonfcs10=int((~sc.fcs10).sum())))
    cc = pd.DataFrame(rows).sort_values(['win', 'prov', 'cl'])
    cc.to_csv(os.path.join(OUTDIR, 'e0_eval_class_counts.csv'), index=False, encoding='utf-8-sig')
    with open(os.path.join(OUTDIR, 'e0_eval_class_counts.md'), 'w', encoding='utf-8') as f:
        f.write('# 五窗 eval 全类计数（2023，已排除双池+错位；prov 按 DataV 归省）\n\n')
        f.write('| 窗 | 省 | 类码 | 类名 | n | fcs10 | 非fcs10 |\n|---|---|---|---|---|---|---|\n')
        for r in cc.itertuples():
            f.write('| %s | %s | %d | %s | %d | %d | %d |\n' % (
                r.win, r.prov, r.cl, r.name, r.n, r.n_fcs10, r.n_nonfcs10))
    emit('全类计数 %d 行 → e0_eval_class_counts.csv' % len(cc))

    # ---- 省×窗 汇总 ----
    summ = []
    for w in WINDOWS:
        mw = ev.win.to_numpy() == w
        for tag, full in PROV2:
            mp = mw & (ev.prov_full.to_numpy() == full)
            summ.append(dict(win=w, prov=tag, n=int(mp.sum()),
                             n_nonfcs10=int((mp & ~isf).sum()),
                             n_classes=int(len(set(t24[mp].tolist())))))
    pd.DataFrame(summ).to_csv(os.path.join(OUTDIR, 'e0_window_prov_summary.csv'),
                              index=False, encoding='utf-8-sig')

    # ---- 候选小框：0.5° 与 1.0° 网格，只按标签/地理统计（不看预测） ----
    cand = []
    for step in (0.5, 1.0):
        for w in WINDOWS:
            mw = ev.win.to_numpy() == w
            sub = ev[mw]
            if len(sub) == 0:
                continue
            lon0 = np.floor(sub.lon.to_numpy() / step) * step
            lat0 = np.floor(sub.lat.to_numpy() / step) * step
            key = np.stack([lon0, lat0], 1)
            for (bx, by), idx in pd.DataFrame(key, columns=['x', 'y']).groupby(['x', 'y']).indices.items():
                ii = np.asarray(idx)
                s = sub.iloc[ii]
                if len(s) < 100:
                    continue
                # 框内省归属（多数省）
                pv = s.prov_full.value_counts()
                pv_tag = {v: k for k, v in PROV2}
                top_full = pv.index[0]
                top_tag = pv_tag.get(top_full, '')
                n_top = int(pv.iloc[0])
                cand.append(dict(
                    step=step, win=w, x0=round(float(bx), 3), y0=round(float(by), 3),
                    x1=round(float(bx) + step, 3), y1=round(float(by) + step, 3),
                    n=int(len(s)), n_nonfcs10=int((~s.fcs10).sum()),
                    top_prov=top_tag, top_prov_full=top_full, n_top_prov=n_top,
                    n_classes=int(s.t24.nunique()),
                    n_cl4=int((s.t24 == 4).sum()), n_cl2=int((s.t24 == 2).sum()),
                    n_cl6=int((s.t24 == 6).sum()), n_cl5=int((s.t24 == 5).sum()),
                    n_cl1=int((s.t24 == 1).sum()), n_cl3=int((s.t24 == 3).sum()),
                    n_cl10=int((s.t24 == 10).sum()), n_cl9=int((s.t24 == 9).sum()),
                    n_cl11=int((s.t24 == 11).sum()), n_cl22=int((s.t24 == 22).sum()),
                ))
    cb = pd.DataFrame(cand)
    if len(cb):
        cb = cb.sort_values(['win', 'n'], ascending=[True, False])
    cb.to_csv(os.path.join(OUTDIR, 'e0_candidate_boxes.csv'), index=False, encoding='utf-8-sig')
    emit('候选框（≥100 点）%d 行 → e0_candidate_boxes.csv' % len(cb))

    inv = dict(generated=time.strftime('%Y-%m-%d %H:%M:%S'), year=YEAR, windows=WINDOWS,
               n_raw=int(n0), n_keep=int(len(ev)), excl_detail=xd,
               n_by_win={w: int((ev.win.to_numpy() == w).sum()) for w in WINDOWS},
               n_candidate_boxes=int(len(cb)))
    with open(os.path.join(OUTDIR, 'e0_inventory.json'), 'w', encoding='utf-8') as f:
        json.dump(inv, f, ensure_ascii=False, indent=1, default=str)
    emit('完成')


if __name__ == '__main__':
    main()
