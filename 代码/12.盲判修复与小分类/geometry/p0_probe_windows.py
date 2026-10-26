# -*- coding: utf-8 -*-
"""p0_probe_windows.py — 三省南缘小幅几何复验：窗口选址探针（纯本地，零 GEE）

* 职责
    在**真实生产瓦片 affine** 上，为川/粤/宁各选 1 条南缘小窗；列范围限定在该省
    **实际省界内**（DataV GCJ-02 省界，像元中心归属），行范围跨瓦片南边界（含瓦外对照行）。
* 规则源
    - doc70 §4.4 / §5 G-30N / §7 几何覆盖门；doc69 §6.4（边缘确定性扫描 ≥128 行/列带）
    - 瓦片格网：x0 = 2*ix + 73, y0 = 2*iy + 18（prod2nd/plan5 tiles_secondrun.csv 反推并核验）
    - 生产 affine 取瓦片 GeoTIFF 表头（px=py=8.983152841195215e-05°）
* 门槛
    - 小窗宽 256–600 px、高 80–128 px（原生 10 m）；南边界两侧均含（瓦外对照行另单列，不计 EVM 分母）；
    - 列范围须落在该省**省界内且瓦片有真实分类值**（不用 land_frac 替代）；
    - 同一瓦片同一省；不跨省选窗。
* 输出
    - F:/lc_work/label_repair_2023_20261015/geometry/probe/window_pick.json
    - 控制台逐窗打印（tile / bbox / rc / 省界内列覆盖率 / 瓦内非零率）
* 用法
    python p0_probe_windows.py
"""
import json
import os
import sys

import numpy as np
import rasterio
from shapely.geometry import shape, box as shpbox

sys.stdout.reconfigure(encoding='utf-8')

PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
RAST = r'F:/lc_work/prod2nd_2023/rasters_batch'
BOUND = os.path.join(PROJ, '数据', '边界', 'china_100000_full.json')
OUTD = r'F:/lc_work/label_repair_2023_20261015/geometry/probe'

PX = 8.983152841195215e-05
NCOL = 557          # 与已验收 G-30N 同宽（256–600 内）
NROW = 93           # 与已验收 G-30N 同高（80–128 内）：上 72 行瓦内 + 下 21 行瓦外对照
UP_IN_TILE = 72     # 南边界以上瓦内行数

# (region_key, prov_name, tile, edge_lat, tile_box)
REGIONS = [
    ('SC_32N', '四川省',                 'T1207', 32.0, [97.0, 32.0, 99.0, 34.0]),
    ('GD_24N', '广东省',                 'T1903', 24.0, [111.0, 24.0, 113.0, 26.0]),
    ('NX_38N', '宁夏回族自治区',          'T1610', 38.0, [105.0, 38.0, 107.0, 40.0]),
]


def prov_geom(name):
    d = json.load(open(BOUND, encoding='utf-8'))
    for f in d['features']:
        if f['properties'].get('name') == name:
            return shape(f['geometry'])
    raise KeyError(name)


def tile_affine(tile):
    with rasterio.open(os.path.join(RAST, '%s_10m.tif' % tile)) as ds:
        t = ds.transform
        return dict(px=float(t.a), py=float(-t.e), x0=float(t.c), ytop=float(t.f),
                    W=ds.width, H=ds.height)


def edge_row(g, edge_lat):
    """瓦片南边界 latitude 对应的行号（0-index，像元中心；可落在瓦片最后一行附近）。"""
    return (g['ytop'] - edge_lat) / g['py'] - 0.5


def province_lon_runs(pg, lat, xlo, xhi, step=1.0 / 3600):
    """在纬线 lat 上，返回省界内的经度区间（度），限定在 [xlo,xhi]。"""
    xs = np.arange(xlo, xhi, step)
    inside = np.array([pg.contains(__import__('shapely').geometry.Point(float(x), float(lat)))
                       for x in xs])
    runs, s = [], None
    for i, v in enumerate(inside):
        if v and s is None:
            s = i
        if not v and s is not None:
            runs.append((xs[s], xs[i - 1]))
            s = None
    if s is not None:
        runs.append((xs[s], xs[-1]))
    return runs


def tile_nonzero_colfrac(tile, g, rows):
    """瓦片在给定行带上的逐列非零率（用于确认'实际有分类值'的列范围）。"""
    r0, r1 = max(0, rows[0]), min(g['H'], rows[1])
    with rasterio.open(os.path.join(RAST, '%s_10m.tif' % tile)) as ds:
        a = ds.read(1, window=((r0, r1), (0, g['W'])))
    return (a > 0).mean(axis=0), (r0, r1)


def pick_window(rk, prov, tile, edge_lat, tbox):
    g = tile_affine(tile)
    er = edge_row(g, edge_lat)
    r0 = int(round(er)) - UP_IN_TILE
    pg = prov_geom(prov)
    # 省界内经度区间（在 edge_lat 上；边界 GCJ-02，窗选内差 <5e-3° 不影响选址）
    runs = province_lon_runs(pg, edge_lat, tbox[0], tbox[2])
    # 瓦片真实非零列带：取南边界以上 40..130 行（避开楔形空带）
    nzfrac, rr = tile_nonzero_colfrac(tile, g, (int(round(er)) - 130, int(round(er)) - 40))
    from shapely.geometry import Point
    rows = np.arange(r0, r0 + NROW)
    lats = g['ytop'] - (rows + 0.5) * g['py']
    # 候选窗：省界内区间长度≥NCOL，窗口**居中**放在区间中点（避免贴省界，保真度最稳）
    best = None
    for a, b in runs:
        ca = (a - g['x0']) / g['px']
        cb = (b - g['x0']) / g['px']
        lo, hi = int(np.ceil(ca)), int(np.floor(cb))
        if hi - lo < NCOL:
            continue
        c0 = lo + (hi - lo - NCOL) // 2
        # 整窗（所有行）省界内覆盖率：逐行取窗内像元中心做包含判定
        cc = np.arange(c0, c0 + NCOL)
        lon = g['x0'] + (cc + 0.5) * g['px']
        pts = [Point(float(x), float(y)) for y in lats for x in lon]
        frac = float(np.mean([pg.contains(p) for p in pts]))
        nz = float(np.mean(nzfrac[c0:c0 + NCOL]))
        score = (frac, nz)
        if best is None or score > best[0]:
            best = (score, c0, a, b, frac, nz)
    if best is None:
        raise RuntimeError('%s 无满足宽度的省界内窗口' % rk)
    _, c0, a, b, frac, nz = best
    xl = g['x0'] + c0 * g['px']
    xr = g['x0'] + (c0 + NCOL) * g['px']
    yt = g['ytop'] - r0 * g['py']
    yb = g['ytop'] - (r0 + NROW) * g['py']
    return dict(
        region=rk, prov=prov, tile=tile, tile_box=tbox, edge_lat=edge_lat,
        grid=dict(px=g['px'], py=g['py'], x0=g['x0'], y_top=g['ytop'], W=g['W'], H=g['H']),
        edge_row=round(er, 3), r0=r0, c0=c0, ncol=NCOL, nrow=NROW,
        rows_in_tile=max(0, min(g['H'], r0 + NROW) - r0),
        rows_outside_tile=max(0, r0 + NROW - g['H']),
        window_bbox_deg=[round(xl, 8), round(yb, 8), round(xr, 8), round(yt, 8)],
        province_lon_runs=[[round(float(a), 4), round(float(b), 4)] for a, b in runs],
        tile_nonzero_colfrac_mean=round(nz, 4),
        tile_nonzero_rows_used=rr,
        in_province_window_frac=round(frac, 4),
        in_province_lat_at_edge=round(frac, 4),
    )


def main():
    os.makedirs(OUTD, exist_ok=True)
    out = []
    for rk, prov, tile, elat, tbox in REGIONS:
        rec = pick_window(rk, prov, tile, elat, tbox)
        out.append(rec)
        print('== %-7s %-6s %s edge=%.1fN edge_row=%.2f' % (rk, tile, prov, elat, rec['edge_row']))
        print('   c0=%d r0=%d ncol=%d nrow=%d（瓦内 %d 行 / 瓦外 %d 行）'
              % (rec['c0'], rec['r0'], rec['ncol'], rec['nrow'], rec['rows_in_tile'], rec['rows_outside_tile']))
        print('   bbox=%s' % (rec['window_bbox_deg'],))
        print('   省界内列占比=%.3f  瓦内非零列率=%.3f  省界经度区间=%s'
              % (rec["in_province_window_frac"], rec['tile_nonzero_colfrac_mean'], rec['province_lon_runs']))
    json.dump(dict(schema='tri_window_pick/v1', px=PX, ncol=NCOL, nrow=NROW,
                   up_in_tile=UP_IN_TILE, regions=out,
                   note='窗口列范围限定省界内且瓦片有真实分类值；行跨瓦片南边界，瓦外对照行不计 EVM 分母'),
              open(os.path.join(OUTD, 'window_pick.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('\n→ %s' % os.path.join(OUTD, 'window_pick.json'))


if __name__ == '__main__':
    main()
