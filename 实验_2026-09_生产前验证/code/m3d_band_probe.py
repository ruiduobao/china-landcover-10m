# -*- coding: utf-8 -*-
"""m3d_band_probe.py — M3-C/D：`glc_fcs10_2023_band130` 来源仲裁点（本地抽样，零 GEE 成本）

背景（doc42 §3.1）：三江（w5）窗内 61% 的训练点来自 `glc_fcs10_2023_band130`（混类抽样），
筛除它可让该窗共同集 OA +6.4~7.8pp —— 但该来源含 181 草本沼泽/182 湖河滩地 等**湿地类**，
而湿地正是三江的弱点类。到底该不该筛？——与 D2 同样的"标签惯例之争"，需独立判读。

设计：
  C 臂 200 点：从 `src == glc_fcs10_2023_band130` 点中按 v31 类分层抽样（全体全国范围）
  D 臂 120 点：C 臂点 3 km 邻域内的底座点（v2_glc_fcs30d*）按类分层抽样
判读问题：Q1 是否有湿地证据（持续/季节性水淹）→ 是/否/无法判读；Q2 主要地物（24 类名）；Q3 证据
产物：data/m3/m3cd_points.csv + .kml + m3cd_判读表单.csv
"""
import os, sys, csv
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v31_common as VC
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SRC = r'F:/lc_work/年度子集_含稀有类/r7_train_2023.parquet'
OUTD = os.path.join(VC.DATA, 'm3')
BAND = 'glc_fcs10_2023_band130'
N_C, N_D = 200, 120
SEED = 20260928
R_KM = 6371.0088


def xyz(lon, lat):
    a = np.radians(np.asarray(lat, dtype='float64')); o = np.radians(np.asarray(lon, dtype='float64'))
    return np.stack([np.cos(a) * np.cos(o), np.cos(a) * np.sin(o), np.sin(a)], axis=1)


def main():
    os.makedirs(OUTD, exist_ok=True)
    band, base = [], []
    pf = pq.ParquetFile(SRC)
    for b in pf.iter_batches(batch_size=200_000, columns=['row_id', 'lon', 'lat', 'class_new', 'src']):
        d = b.to_pandas()
        s = d['src'].fillna('').astype(str)
        m1 = s == BAND
        m2 = s.str.startswith('v2_glc_fcs30d')
        if m1.any():
            band.append(d[m1])
        if m2.any():
            base.append(d[m2])
    band = pd.concat(band, ignore_index=True); base = pd.concat(base, ignore_index=True)
    band['v31'] = VC.to_v31(band['class_new'].to_numpy(int))
    VC.emit('band130 点 %d ／ 底座点 %d' % (len(band), len(base)))
    rng = np.random.default_rng(SEED)
    per = max(20, N_C // max(1, band.v31.nunique()))
    rows = []
    for c, g in band.groupby('v31'):
        k = min(per, len(g))
        rows.append(g.iloc[rng.choice(len(g), size=k, replace=False)])
    C = pd.concat(rows, ignore_index=True)
    C['arm'] = 'C_band130'
    # D 臂：C 点 3km 邻域内底座点
    from scipy.spatial import cKDTree
    tC = cKDTree(xyz(C.lon.to_numpy(), C.lat.to_numpy()))
    xb = xyz(base.lon.to_numpy(), base.lat.to_numpy())
    dist, _ = tC.query(xb, k=1)
    dkm = 2 * R_KM * np.arcsin(np.clip(dist / 2, 0, 1))
    nb = base[dkm < 3.0].copy()
    nb['v31'] = VC.to_v31(nb['class_new'].to_numpy(int))
    per2 = max(15, N_D // max(1, nb.v31.nunique()))
    rows = []
    for c, g in nb.groupby('v31'):
        k = min(per2, len(g))
        rows.append(g.iloc[rng.choice(len(g), size=k, replace=False)])
    D = pd.concat(rows, ignore_index=True)
    D['arm'] = 'D_control_base'
    C['point_id'] = ['M3C-%03d' % i for i in range(len(C))]
    D['point_id'] = ['M3D-%03d' % i for i in range(len(D))]
    allp = pd.concat([C[['point_id', 'lon', 'lat', 'arm', 'v31']],
                      D[['point_id', 'lon', 'lat', 'arm', 'v31']]], ignore_index=True)
    allp.to_csv(os.path.join(OUTD, 'm3cd_points.csv'), index=False, encoding='utf-8-sig')
    names = VC.V31_NAMES()
    kml = ['<?xml version="1.0" encoding="UTF-8"?>', '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
           '<name>M3C/D band130 仲裁点</name>']
    col = {'C_band130': 'ff0000ff', 'D_control_base': 'ff00ff00'}
    for _, p in allp.iterrows():
        kml.append('<Placemark><name>%s</name><description>臂=%s 教师类=%s</description>'
                   '<Style><IconStyle><color>%s</color></IconStyle></Style>'
                   '<Point><coordinates>%.6f,%.6f,0</coordinates></Point></Placemark>' % (
                       p.point_id, p.arm, names.get(str(p.v31), p.v31), col[p.arm], p.lon, p.lat))
    kml.append('</Document></kml>')
    open(os.path.join(OUTD, 'm3cd_points.kml'), 'w', encoding='utf-8').write('\n'.join(kml))
    with open(os.path.join(OUTD, 'm3cd_判读表单.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['point_id', 'arm', 'lon', 'lat', '教师类(v31)',
                    'Q1_是否有湿地证据(是/否/无法判读)', 'Q2_主要地物(24类名)', 'Q3_证据(影像/日期)',
                    '判读员', '复核员', '备注'])
        for _, p in allp.iterrows():
            w.writerow([p.point_id, p.arm, p.lon, p.lat, names.get(str(p.v31), p.v31), '', '', '', '', '', ''])
    from collections import Counter
    VC.emit('C 臂 %d 点：%s' % (len(C), dict(Counter(names.get(str(k), k) for k in C.v31))))
    VC.emit('D 臂 %d 点：%s' % (len(D), dict(Counter(names.get(str(k), k) for k in D.v31))))
    VC.emit('→ data/m3/m3cd_points.csv/.kml/判读表单.csv')


if __name__ == '__main__':
    main()
