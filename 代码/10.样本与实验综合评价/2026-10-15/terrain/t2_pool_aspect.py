# -*- coding: utf-8 -*-
"""t2_pool_aspect.py — 训练池（v2 1.76M / v1 2.06M）川域样点逐类坡向分布（本地 60m DEM 采样）

* 输入（只读）：F:/lc_work/prod2nd_2023/data/train2023_v2.parquet（cl 24类紧凑码 + lon/lat + src）
    F:/lc_work/prod5p_2023/data/train2023_clean.parquet（class_new 30类母库码 + lon/lat + src，v1 五省批池）
    本地 SRTM 60m dem_T*.tif（川 24 瓦，只读复用）
* 规则源：v31_map.json merges（4=[51,52] 5=[61,62] 6=[71,72] 7=[81,82] 8=[91,92]）
* 门槛：无（诊断）；输出 v2/v1 各类阴坡占比、ENF/EBF OR、按 src 拆分与两池对照
* 输出：terrain/tables_pool_aspect.json + CSV
* 用法：python t2_pool_aspect.py
"""
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
OUTD = r'F:/lc_work/prod2nd_2023/reports/quality_review_2026-10-15/terrain'
DEMD = r'F:/lc_work/v31_exp/data/eco_gate/_dem'
V2 = r'F:/lc_work/prod2nd_2023/data/train2023_v2.parquet'
V1 = r'F:/lc_work/prod5p_2023/data/train2023_clean.parquet'
NAME = {4: '常绿阔叶', 5: '落叶阔叶', 6: '常绿针叶', 7: '落叶针叶', 8: '针阔混交',
        1: '草本旱地', 9: '常绿灌丛', 10: '落叶灌丛', 11: '草地', 21: '不透水面', 22: '裸地'}
MERGE = {51: 4, 52: 4, 61: 5, 62: 5, 71: 6, 72: 6, 81: 7, 82: 7, 91: 8, 92: 8}


def emit(m):
    print(m, flush=True)


def load_dem_index():
    """返回 {tile: (b, t, dlat, dlon, W, H)} 以及按 tile 惰性读取高程。"""
    import rasterio
    idx = {}
    for f in sorted(os.listdir(DEMD)):
        if f.startswith('dem_T') and f.endswith('.tif'):
            with rasterio.open(os.path.join(DEMD, f)) as s:
                idx[f[4:-4]] = (s.bounds, s.transform, s.width, s.height)
    return idx


def sample_dem(lon, lat, idx):
    """最近 DEM cell 取样（连续量按最近格，仅供样点地形属性，不涉类别重采样）。"""
    import rasterio
    out = np.full(lon.shape, np.nan, np.float32)
    sl_out = np.full(lon.shape, np.nan, np.float32)
    for t, (b, tfm, W, H) in idx.items():
        m = (lon >= b.left) & (lon < b.right) & (lat > b.bottom) & (lat <= b.top)
        if not m.any():
            continue
        with rasterio.open(os.path.join(DEMD, 'dem_%s.tif' % t)) as s:
            els = s.read(1)
            dlat, dlon = -tfm.e, tfm.a
            col = ((lon[m] - tfm.c) / dlon).astype(np.int64).clip(1, W - 2)
            row = ((tfm.f - lat[m]) / dlat).astype(np.int64).clip(1, H - 2)
            val = els[row, col].astype(np.float32)
            # 局部坡度（该点 3×3 内，米制）
            la = np.deg2rad(lat[m])
            m_lat = 111132.92 - 559.82 * np.cos(2 * la) + 1.175 * np.cos(4 * la)
            m_lon = 111412.84 * np.cos(la) - 93.5 * np.cos(3 * la)
            dzdx = (els[row, col + 1] - els[row, col - 1]).astype(np.float32) / 2.0 / (dlon * m_lon)
            dzdy = -(els[row + 1, col] - els[row - 1, col]).astype(np.float32) / 2.0 / (dlat * m_lat)
            out[m] = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
            asp = (np.degrees(np.arctan2(-dzdx, -dzdy)) + 360.0) % 360.0
            sl_out[m] = asp
    return out, sl_out


def stats_table(cls, asp, slope, tag, out):
    sunny = (asp >= 135) & (asp <= 225)
    shady = (asp <= 45) | (asp >= 315)
    rows = []
    for c in [4, 5, 6, 7, 8, 1, 9, 10, 11, 21, 22]:
        m = (cls == c) & np.isfinite(asp) & (slope > 5)
        k = int(m.sum())
        if k < 100:
            continue
        su, sh = int((m & sunny).sum()), int((m & shady).sum())
        rows.append({'pool': tag, 'class': int(c), 'name': NAME.get(int(c)), 'n_slope5': k,
                     'sunny_pct': round(100 * su / k, 1), 'shady_pct': round(100 * sh / k, 1),
                     'mean_slope': round(float(slope[m].mean()), 1)})
    out['per_class_%s' % tag] = rows
    emit('== 池%s 逐类（slope>5°）==' % tag)
    for r in rows:
        emit('  cls%2d %s n=%d sunny=%.1f%% shady=%.1f%% slope~%.1f' %
             (r['class'], r['name'], r['n_slope5'], r['sunny_pct'], r['shady_pct'], r['mean_slope']))
    # ENF/EBF OR
    for cA, cB, lab in ((6, 4, 'ENF6_vs_EBF4'), (6, 5, 'ENF6_vs_DBF5')):
        A = (cls == cA) & np.isfinite(asp) & (slope > 5)
        B = (cls == cB) & np.isfinite(asp) & (slope > 5)
        a_sh, a_su = int((A & shady).sum()), int((A & sunny).sum())
        b_sh, b_su = int((B & shady).sum()), int((B & sunny).sum())
        if min(a_sh, a_su, b_sh, b_su) == 0:
            continue
        orr = (a_sh * b_su) / (a_su * b_sh)
        out.setdefault('OR_%s' % lab, {})[tag] = round(orr, 3)
        emit('  OR %s [%s] = %.3f (A shady %.1f%% n=%d | B shady %.1f%% n=%d)' %
             (lab, tag, orr, 100 * a_sh / A.sum(), int(A.sum()), 100 * b_sh / B.sum(), int(B.sum())))


def main():
    import pyarrow.parquet as pq
    idx = load_dem_index()
    out = {'dem_tiles': sorted(idx.keys())}
    # ---- v2 ----
    t = pq.read_table(V2, columns=['lon', 'lat', 'cl', 'src'])
    lon = np.asarray(t['lon']); lat = np.asarray(t['lat'])
    cl = np.asarray(t['cl']); src = np.asarray(t['src'].to_pylist())
    m川 = (lon >= 97.0) & (lon <= 109.0) & (lat >= 26.0) & (lat <= 34.5)
    emit('v2 总 %d，川域 %d' % (cl.size, m川.sum()))
    slope, asp = sample_dem(lon[m川], lat[m川], idx)
    stats_table(cl[m川], asp, slope, 'v2', out)
    # 按 src 拆分（仅森林类 4/5/6）
    from collections import Counter
    cnt = Counter(src[m川])
    out['v2_src_top'] = dict(cnt.most_common(10))
    fc = np.isin(cl[m川], (4, 5, 6))
    shady2 = (asp <= 45) | (asp >= 315)
    sub = {}
    for s in [k for k, _ in cnt.most_common(12)]:
        sel = fc & (src[m川] == s)
        if sel.sum() < 300:
            continue
        v = {'n_forest': int(sel.sum())}
        for c in (4, 6):
            m2 = sel & (cl[m川] == c)
            if m2.sum() >= 100:
                v['shady_pct_c%d' % c] = round(100 * float((m2 & shady2).sum()) / m2.sum(), 1)
        sub[s] = v
    out['v2_by_src_forest'] = sub
    emit('v2 按 src（森林类阴坡%）：')
    for k, v in sub.items():
        emit('  %-32s %s' % (k, v))
    # ---- v1 ----
    t1 = pq.read_table(V1, columns=['lon', 'lat', 'class_new', 'src', 'qc_status'])
    lon1 = np.asarray(t1['lon']); lat1 = np.asarray(t1['lat'])
    c30 = np.asarray(t1['class_new'])
    cl1 = np.array([MERGE.get(int(c), int(c)) for c in c30], dtype=np.int64)
    m川1 = (lon1 >= 97.0) & (lon1 <= 109.0) & (lat1 >= 26.0) & (lat1 <= 34.5)
    emit('v1 总 %d，川域 %d（class_new 已并 24 类紧凑码）' % (cl1.size, m川1.sum()))
    slope1, asp1 = sample_dem(lon1[m川1], lat1[m川1], idx)
    stats_table(cl1[m川1], asp1, slope1, 'v1', out)
    json.dump(out, open(os.path.join(OUTD, 'tables_pool_aspect.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1, default=float)
    emit('已写 tables_pool_aspect.json')


if __name__ == '__main__':
    main()
