# -*- coding: utf-8 -*-
"""y1_prep.py — 年度匹配检查 · 本地建表（doc38 §4.2）
协议（4 年同协议，含 2023 作为同协议参照）：
  · 年：2017 / 2020 / 2023 / 2024；窗：w1–w5（沿用 bbox）
  · 训练池：年子集∩窗内，剔除 ①全部固定留出 rid ②Baseline sqrt 训练 rid；按 row_id 哈希取前 20,000（确定性）
  · 评估集：固定留出 rid（2023 窗留出）∩ 该年窗内点 → 逐年 n 会不同；另算 4 年交集作主口径
  · 内存纪律：ParquetFile.iter_batches 流式过滤，单批 20 万行，峰值 <1.5G
产物：data/yearly/{train,eval}_{year}_{w}.parquet + data/yearly/plan.json
"""
import os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v31_common as VC
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

YEARS = [2017, 2020, 2023, 2024]
WIN = VC.cfg('windows.json')['windows']
OUTD = os.path.join(VC.DATA, 'yearly')
N_TRAIN = 20000
BATCH = 200_000


def eval_rids_of(w):
    fp = os.path.join(VC.DATA, 'eval_%s.parquet' % w)
    if not os.path.exists(fp):
        fp = os.path.join(VC.EVAL_DIR_GEE, 'eval_%s.parquet' % w)
    return set(pd.read_parquet(fp, columns=['row_id'])['row_id'].astype('int64'))


def stream_window_points(year, boxes):
    """流式读年子集，保留落在任一小窗盒内的点（只保留需要的列）。"""
    fp = r'F:/lc_work/年度子集_含稀有类/r7_train_%d.parquet' % year
    cols = ['row_id', 'lon', 'lat', 'class_new', 'src', 'tier', 'qc_status', 'train_weight'] + VC.FEATS
    keep = []
    pf = pq.ParquetFile(fp)
    for b in pf.iter_batches(batch_size=BATCH, columns=cols):
        d = b.to_pandas()
        m = np.zeros(len(d), dtype=bool)
        for w, bx in boxes.items():
            m |= ((d.lon >= bx[0]) & (d.lon <= bx[2]) & (d.lat >= bx[1]) & (d.lat <= bx[3])).to_numpy()
        if m.any():
            keep.append(d[m])
    if not keep:
        return pd.DataFrame(columns=cols)
    out = pd.concat(keep, ignore_index=True)
    out['row_id'] = out['row_id'].astype('int64')
    return out


def main():
    os.makedirs(OUTD, exist_ok=True)
    plan_fp = os.path.join(OUTD, 'plan.json')
    hashes = {w: None for w in WIN}
    EV = {w: eval_rids_of(w) for w in WIN}
    EV_ALL = set().union(*EV.values())
    VC.emit('固定留出 rid：%s（并集 %d）' % ({w: len(s) for w, s in EV.items()}, len(EV_ALL)))
    nat = pd.read_csv(VC.SQRT_CSV, usecols=['row_id'])
    NAT = set(nat.row_id.astype('int64'))
    del nat
    VC.emit('Baseline 训练 rid %d（训练集将剔除）' % len(NAT))
    boxes = {w: WIN[w]['bbox'] for w in WIN}
    plan = dict(years=YEARS, windows=list(WIN), n_train_cap=N_TRAIN, per={})
    eval_sets = {}
    for year in YEARS:
        t0 = time.time()
        d = stream_window_points(year, boxes)
        VC.emit('【%d】窗内合计 %d 点（%.1f 分钟）' % (year, len(d), (time.time() - t0) / 60))
        plan['per'][str(year)] = {}
        for w in WIN:
            bx = boxes[w]
            sub = d[(d.lon >= bx[0]) & (d.lon <= bx[2]) & (d.lat >= bx[1]) & (d.lat <= bx[3])]
            rid = sub.row_id.to_numpy()
            ev_mask = np.isin(rid, list(EV[w]))
            ev = sub[ev_mask]
            tr = sub[~ev_mask & ~np.isin(rid, list(NAT))]
            # 确定性取前 N：哈希排序
            h = (tr.row_id.to_numpy() * 2654435761) % (2 ** 31)
            tr = tr.iloc[np.argsort(h, kind='stable')[:N_TRAIN]]
            tr.to_parquet(os.path.join(OUTD, 'train_%d_%s.parquet' % (year, w)), index=False)
            ev.to_parquet(os.path.join(OUTD, 'eval_%d_%s.parquet' % (year, w)), index=False)
            eval_sets.setdefault(w, {})[year] = set(ev.row_id.tolist())
            plan['per'][str(year)][w] = dict(n_win=int(len(sub)), n_train=int(len(tr)),
                                             n_eval=int(len(ev)))
            VC.emit('  %s: 窗内 %6d → 训练 %5d / 评估 %5d' % (w, len(sub), len(tr), len(ev)))
        del d
    # 4 年共同评估集（主口径）
    for w in WIN:
        common = set.intersection(*[eval_sets[w][y] for y in YEARS])
        plan.setdefault('common_eval', {})[w] = dict(n=len(common),
                                                     rids=sorted(int(x) for x in common)[:200000])
        VC.emit('共同评估集 %s：%d 点' % (w, len(common)))
    VC.jsave(plan, plan_fp)
    VC.emit('建表完成 → %s' % plan_fp)


if __name__ == '__main__':
    main()
