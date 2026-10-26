# -*- coding: utf-8 -*-
"""t1_sichuan_terrain.py — 川 2023 第二次分类 × 本地 SRTM60m 地形：坡向/坡度/高程关联与两版对照

* 输入（全部只读，跑前记录 path/size/mtime/md5）：
    二版类码 F:/lc_work/prod2nd_2023/delivery/成品/分省/四川_2023_10m.tif（0=nodata）
    一版类码 D:/生产_2023五省/成品/分省/四川_2023_10m.tif（与二版同 shape/transform）、
    地形     F:/lc_work/v31_exp/data/eco_gate/_dem/dem_T*.tif（SRTMGL1_003 60m，单波段 int16 高程，只读复用）
* 规则源：v31_map.json 24 类码（4=常绿阔叶 5=落叶阔叶 6=常绿针叶 7=落叶针叶 8=针阔混交）；
    坡度/坡向本地由 60m 高程计算，**经纬格距按纬度换算为米**（避免经纬单位造成坡向/坡度假象）；
    DEM 连续量 bilinear 对齐到 10m 原生类码格；类码一律同格直读（nearest 语义，不做任何重采样）。
* 门槛：无（诊断统计脚本）；抽样=每个 DEM 瓦(2°×2°) 内 10 个 480×480 类码窗（种子固定）
* 输出：reports/quality_review_2026-10-15/terrain/ 下 file_identity.json / records_川.npz /
    tables_川.json / tables_川_*.csv
* 用法：python t1_sichuan_terrain.py [--tiles T1508,T1505] [--wins 10] [--max-tiles N]
"""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback

import numpy as np

os.environ.setdefault('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
sys.stdout.reconfigure(encoding='utf-8')

PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
OUTD = r'F:/lc_work/prod2nd_2023/reports/quality_review_2026-10-15/terrain'
CLS2 = r'F:/lc_work/prod2nd_2023/delivery/成品/分省/四川_2023_10m.tif'
CLS1 = r'D:/生产_2023五省/成品/分省/四川_2023_10m.tif'
DEMD = r'F:/lc_work/v31_exp/data/eco_gate/_dem'
SEED = 20261015
FOREST = (4, 5, 6, 7, 8)
REF_CLS = (1, 9, 10, 11, 21, 22, 23, 24)
WIN = 480                     # 类码窗边长 px（=80 DEM cell × 6px/cell）
ELEV_BANDS = [0, 500, 1000, 1500, 2500, 3500, 4500, 9999]
SLOPE_BANDS = [0, 5, 15, 25, 90]


def emit(m):
    print(time.strftime('[%H:%M:%S] ') + str(m), flush=True)


def md5_of(fp):
    h = hashlib.md5()
    with open(fp, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def identity(fp):
    st = os.stat(fp)
    return {'path': fp, 'size': st.st_size,
            'mtime': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_mtime)),
            'md5': md5_of(fp)}


# ---------- 地形计算（米制换算） ----------
def m_per_deg(lat_deg):
    """WGS84 球面近似：每度纬度/经度对应的米数。"""
    lat = np.deg2rad(lat_deg)
    m_lat = 111132.92 - 559.82 * np.cos(2 * lat) + 1.175 * np.cos(4 * lat) \
        - 0.0023 * np.cos(6 * lat)
    m_lon = 111412.84 * np.cos(lat) - 93.5 * np.cos(3 * lat) + 0.118 * np.cos(5 * lat)
    return m_lat, m_lon


def terrain_arrays(dem_fp, unit_test_deg=False):
    """读 DEM 瓦 → elev/slope(deg)/aspect(deg 0=N 顺时针)/sinA/cosA。经纬按米换算。"""
    import rasterio
    with rasterio.open(dem_fp) as s:
        elev = s.read(1).astype(np.float32)
        tfm = s.transform
        H, W = elev.shape
        dlon, dlat = tfm.a, -tfm.e
        lats = tfm.f - dlat * (np.arange(H) + 0.5)      # doc55 §5 修正后的行纬度
    m_lat, m_lon = m_per_deg(lats)
    if unit_test_deg:                                    # 反例：不做米换算（deg 直接当米）
        sx = dlon
        sy = dlat
    else:
        sx = dlon * m_lon[:, None]                       # 米/px（经向逐行）
        sy = dlat * m_lat[:, None]
    gE = np.gradient(elev, axis=1) / sx                  # dz/d(east)  m/m
    gN = -np.gradient(elev, axis=0) / sy                 # dz/d(north) m/m
    slope = np.degrees(np.arctan(np.sqrt(gE * gE + gN * gN))).astype(np.float32)
    aspect = (np.degrees(np.arctan2(-gE, -gN)) + 360.0) % 360.0
    aspect = aspect.astype(np.float32)
    sinA = np.sin(np.radians(aspect)).astype(np.float32)
    cosA = np.cos(np.radians(aspect)).astype(np.float32)
    return elev, slope, aspect, sinA, cosA


def hillshade(aspect, slope, sun_az, sun_el):
    a = np.radians(aspect)
    z = np.radians(90.0 - sun_el)
    s = np.radians(slope)
    return (np.cos(z) * np.cos(s) + np.sin(z) * np.sin(s) * np.cos(np.radians(sun_az) - a))


# ---------- 采样 ----------
def win_centers(tile_box, n, rng, dem_shape, dem_tfm):
    """在 DEM 瓦内取 n 个 480px 窗中心（DEM cell 索引奇数格）。"""
    W, H = dem_shape
    cells = WIN // 6
    ni = (W - cells) // cells
    nj = (H - cells) // cells
    out = []
    used = set()
    for _ in range(n * 40):
        if len(out) >= n:
            break
        i = int(rng.integers(0, ni)) * cells
        j = int(rng.integers(0, nj)) * cells
        if (i // cells, j // cells) in used:
            continue
        used.add((i // cells, j // cells))
        out.append((i, j))
    return out


def run(tiles=None, wins=10, max_tiles=None):
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.warp import reproject, Resampling

    os.makedirs(OUTD, exist_ok=True)
    ident = {'inputs': {}, 'dem': {}}
    for name, fp in (('cls2_川', CLS2), ('cls1_川', CLS1)):
        ident['inputs'][name] = identity(fp)
        emit('%s md5=%s' % (name, ident['inputs'][name]['md5']))
    with rasterio.open(CLS2) as s2, rasterio.open(CLS1) as s1:
        assert s2.shape == s1.shape and s2.transform == s1.transform, '两版网格不一致'
        shape, tfm = s2.shape, s2.transform

    dem_list = sorted(f for f in os.listdir(DEMD)
                      if f.startswith('dem_T') and f.endswith('.tif'))
    if tiles:
        keep = set(tiles.split(','))
        dem_list = [f for f in dem_list if f[4:-4] in keep]
    if max_tiles:
        dem_list = dem_list[:max_tiles]
    emit('DEM 瓦 %d: %s' % (len(dem_list), ' '.join(f[4:-4] for f in dem_list)))

    rng = np.random.default_rng(SEED)
    recs = []          # (cls2, cls1, elev, slope, asp, tile_i, lon, lat)
    for ti, fname in enumerate(dem_list):
        fp = os.path.join(DEMD, fname)
        ident['dem'][fname[4:-4]] = identity(fp)
        elev, slope, asp, sinA, cosA = terrain_arrays(fp)
        DEM_H, DEM_W = elev.shape
        with rasterio.open(fp) as s:
            dtfm = s.transform
        # DEM 瓦 bbox（含小外扩）
        pad = 0.02
        b = (dtfm.c - pad, dtfm.f - dtfm.e * DEM_H - pad, dtfm.c + dtfm.a * DEM_W + pad, dtfm.f + pad)
        centers = win_centers(b, wins * 8, rng, (DEM_W, DEM_H), dtfm)
        acc = 0
        for (i, j) in centers:
            if acc >= wins:
                break
            x0 = dtfm.c + i * dtfm.a
            x1 = dtfm.c + (i + WIN // 6) * dtfm.a
            y1 = dtfm.f - j * (-dtfm.e)
            y0 = dtfm.f - (j + WIN // 6) * (-dtfm.e)
            with rasterio.open(CLS2) as s2, rasterio.open(CLS1) as s1:
                win = from_bounds(x0, y0, x1, y1, transform=tfm).round_offsets().round_lengths()
                a2 = s2.read(1, window=win, boundless=True, fill_value=0)
                a1 = s1.read(1, window=win, boundless=True, fill_value=0)
                wtfm = rasterio.windows.transform(win, tfm)
            h, w = a2.shape
            if (a2 > 0).mean() < 0.25:
                continue
            acc += 1
            # DEM 连续量 → 类码窗网格（bilinear）；sin/cos 分量分别重采样
            def rp(src):
                dst = np.empty((h, w), dtype=np.float32)
                reproject(src, dst, src_transform=dtfm, src_crs='EPSG:4326',
                          dst_transform=wtfm, dst_crs='EPSG:4326',
                          resampling=Resampling.bilinear)
                return dst
            row0 = wtfm.f
            lats = row0 - wtfm.e * (np.arange(h) + 0.5)
            el = rp(elev)
            sl = rp(slope)
            sa = rp(sinA)
            ca = rp(cosA)
            aspw = (np.degrees(np.arctan2(sa, ca)) + 360.0) % 360.0
            lon = wtfm.c + wtfm.a * (np.arange(w) + 0.5)
            LON = np.repeat(lon[None, :], h, axis=0)
            LAT = np.repeat(lats[:, None], w, axis=1)
            # 子采样（森林类 40%，参照类 6%）
            m_f = np.isin(a2, FOREST)
            m_r = np.isin(a2, REF_CLS)
            keep = ((m_f & (rng.random(a2.shape) < 0.40)) |
                    (m_r & (rng.random(a2.shape) < 0.06)))
            if not keep.any():
                continue
            recs.append(np.stack([
                a2[keep].astype(np.float32), a1[keep].astype(np.float32),
                el[keep], sl[keep], aspw[keep], np.full(keep.sum(), ti, np.float32),
                LON[keep], LAT[keep]], axis=1).astype(np.float32))
        emit('  %s 完成 (接受窗 %d/%d, 累计行 %.2fM)' % (fname[4:-4], acc, wins,
                                              sum(r.shape[0] for r in recs) / 1e6))
    R = np.concatenate(recs, axis=0)
    emit('记录行数 %.2fM' % (R.shape[0] / 1e6))
    np.savez_compressed(os.path.join(OUTD, 'records_川.npz'),
                        cls2=R[:, 0].astype(np.uint8), cls1=R[:, 1].astype(np.uint8),
                        elev=R[:, 2].astype(np.float32), slope=R[:, 3].astype(np.float32),
                        asp=R[:, 4].astype(np.float32), tile=R[:, 5].astype(np.uint8),
                        lon=R[:, 6], lat=R[:, 7])
    json.dump(ident, open(os.path.join(OUTD, 'file_identity.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    emit('已写 records_川.npz / file_identity.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tiles', default=None)
    ap.add_argument('--wins', type=int, default=10)
    ap.add_argument('--max-tiles', type=int, default=None)
    a = ap.parse_args()
    try:
        run(a.tiles, a.wins, a.max_tiles)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
