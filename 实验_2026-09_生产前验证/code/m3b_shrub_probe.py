# -*- coding: utf-8 -*-
"""m3b_shrub_probe.py — M3-B 灌丛标签仲裁点（D2 决定性探针 · 本地抽样，零 GEE 成本）

问题：2023/2024 训练池里 24–59% 的点来自 `glc_fcs10_2023_shrub`，且五个窗的灌丛训练点**全部**来自它；
      底座（FCS30D-2020 系）几乎不给这些位置灌丛标签。到底谁对？——只有独立判读能回答。

设计（两臂，层抽样）：
  A 臂：从 `src == glc_fcs10_2023_shrub` 的点层抽样 600 点（5 窗按点数比例，最小 60/窗）
        → 估计"FCS10 灌丛层精度" P(真灌丛 | FCS10 说灌丛)
  B 臂：从底座点（`src` 前缀 v2_glc_fcs30d）中，抽取位于 A 臂点 3 km 邻域内的 300 点（按底座类分层）
        → 同邻域对照，估计"底座漏判灌丛"程度
判据（预注册）：
  · FCS10 精度 ≥0.70 → 保留 FCS10 灌丛点（R0/R2/R3 方向）
  · FCS10 精度 ≤0.40 → 全筛（R1 方向）
  · 0.40–0.70 → 折中（保留但降采样），并按区域分别决策
产物：data/m3/m3b_points.csv + m3b_points.kml + m3b_判读表单.csv
"""
import os, sys, time, csv, json
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v31_common as VC
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SRC = r'F:/lc_work/年度子集_含稀有类/r7_train_2023.parquet'
OUTD = os.path.join(VC.DATA, 'm3')
WIN = VC.cfg('windows.json')['windows']
N_A, N_B = 600, 300
SEED = 20260927
R_KM = 6371.0088


def xyz(lon, lat):
    a = np.radians(np.asarray(lat, dtype='float64')); o = np.radians(np.asarray(lon, dtype='float64'))
    return np.stack([np.cos(a) * np.cos(o), np.cos(a) * np.sin(o), np.sin(a)], axis=1)


def load(only_cols=('row_id', 'lon', 'lat', 'class_new', 'src')):
    """流式读取 2023 子集，返回 FCS10灌丛 与 底座 两类点（只留列）。"""
    f10, base = [], []
    pf = pq.ParquetFile(SRC)
    for b in pf.iter_batches(batch_size=200_000, columns=list(only_cols)):
        d = b.to_pandas()
        s = d['src'].fillna('').astype(str)
        m1 = s == 'glc_fcs10_2023_shrub'
        m2 = s.str.startswith('v2_glc_fcs30d')
        if m1.any():
            f10.append(d[m1])
        if m2.any():
            base.append(d[m2])
    return pd.concat(f10, ignore_index=True), pd.concat(base, ignore_index=True)


def sample_arm(f10, base):
    rng = np.random.default_rng(SEED)
    who = np.full(len(f10), '', dtype='U4')
    for w, meta in WIN.items():
        bx = meta['bbox']
        m = ((f10.lon >= bx[0]) & (f10.lon <= bx[2]) & (f10.lat >= bx[1]) & (f10.lat <= bx[3])).to_numpy()
        who[m] = w
    f10 = f10.assign(win=who)
    cnt = f10[f10.win != ''].win.value_counts()
    VC.emit('FCS10 灌丛点窗内分布：%s' % dict(cnt))
    tot = int(cnt.sum())
    rows = []
    for w, n in cnt.items():
        k = max(60, int(round(N_A * n / tot)))
        sub = f10[f10.win == w]
        idx = rng.choice(len(sub), size=min(k, len(sub)), replace=False)
        rows.append(sub.iloc[idx])
    A = pd.concat(rows, ignore_index=True)
    A['arm'] = 'A_fcs10_shrub'
    A['w'] = A.win
    # B 臂：A 点 3km 邻域内的底座点（按底座类分层）
    from scipy.spatial import cKDTree
    tA = cKDTree(xyz(A.lon.to_numpy(), A.lat.to_numpy()))
    xb = xyz(base.lon.to_numpy(), base.lat.to_numpy())
    dist, _ = tA.query(xb, k=1)
    dkm = 2 * R_KM * np.arcsin(np.clip(dist / 2, 0, 1))
    nb = base[dkm < 3.0].copy()
    nb['near_shrub_km'] = dkm[dkm < 3.0].round(2)
    nb['v31'] = VC.to_v31(nb['class_new'].to_numpy(int))
    VC.emit('A 臂 3km 邻域内底座点 %d（按 v31 类分层抽 %d）' % (len(nb), N_B))
    names = VC.V31_NAMES()
    rows = []
    per_cls = max(20, N_B // max(1, nb.v31.nunique()))
    for c, g in nb.groupby('v31'):
        k = min(per_cls, len(g))
        rows.append(g.iloc[rng.choice(len(g), size=k, replace=False)])
    B = pd.concat(rows, ignore_index=True)
    B['arm'] = 'B_control_base'
    who = np.full(len(B), '', dtype='U4')
    for w, meta in WIN.items():
        bx = meta['bbox']
        m = ((B.lon >= bx[0]) & (B.lon <= bx[2]) & (B.lat >= bx[1]) & (B.lat <= bx[3])).to_numpy()
        who[m] = w
    B['w'] = who
    for df in (A, B):
        df['v31'] = VC.to_v31(df['class_new'].to_numpy(int))
    return A, B, dict(names)


def main():
    os.makedirs(OUTD, exist_ok=True)
    VC.emit('读取 2023 年度子集（流式）…')
    f10, base = load()
    VC.emit('FCS10 灌丛点 %d ／ 底座点 %d' % (len(f10), len(base)))
    A, B, names = sample_arm(f10, base)
    A['point_id'] = ['M3B-A%03d' % i for i in range(len(A))]
    B['point_id'] = ['M3B-B%03d' % i for i in range(len(B))]
    allp = pd.concat([A[['point_id', 'lon', 'lat', 'w', 'arm', 'v31']],
                      B[['point_id', 'lon', 'lat', 'w', 'arm', 'v31']]], ignore_index=True)
    fp = os.path.join(OUTD, 'm3b_points.csv')
    allp.to_csv(fp, index=False, encoding='utf-8-sig')
    VC.emit('点位 → %s（A=%d B=%d）' % (fp, len(A), len(B)))
    kml = ['<?xml version="1.0" encoding="UTF-8"?>', '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
           '<name>M3B 灌丛仲裁点</name>']
    col = {'A_fcs10_shrub': 'ff0000ff', 'B_control_base': 'ff00ff00'}
    for _, p in allp.iterrows():
        kml.append('<Placemark><name>%s</name><description>臂=%s 窗=%s 教师类=%s</description>'
                   '<Style><IconStyle><color>%s</color></IconStyle></Style>'
                   '<Point><coordinates>%.6f,%.6f,0</coordinates></Point></Placemark>' % (
                       p.point_id, p.arm, p.w, names.get(str(p.v31), p.v31), col.get(p.arm, ''), p.lon, p.lat))
    kml.append('</Document></kml>')
    open(os.path.join(OUTD, 'm3b_points.kml'), 'w', encoding='utf-8').write('\n'.join(kml))
    form = os.path.join(OUTD, 'm3b_判读表单.csv')
    with open(form, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['point_id', 'arm', 'w', 'lon', 'lat', '教师类(v31)',
                    'Q1_是否灌丛(是/否/无法判读)', 'Q2_若否主要地物(24类名)',
                    'Q3_证据(影像/日期)', '判读员', '复核员', '备注'])
        for _, p in allp.iterrows():
            w.writerow([p.point_id, p.arm, p.w, p.lon, p.lat, names.get(str(p.v31), p.v31),
                        '', '', '', '', '', ''])
    VC.emit('表单 → %s' % form)
    # 层权（用于精度估计）：A 臂各窗 抽样比
    wts = {}
    for w_, meta in WIN.items():
        tot_w = int((A.w == w_).sum())
        wts[w_] = tot_w
    VC.jsave(dict(n_A=len(A), n_B=len(B), per_win_A=wts, seed=SEED), os.path.join(OUTD, 'm3b_plan.json'))


if __name__ == '__main__':
    main()
