# -*- coding: utf-8 -*-
"""o17_rubber_sea_extract_cn.py — Rubber_SEA_2020 中国区橡胶候选点提取 + 境内橡胶像元面积统计（本地，只读 tif）

* 输入：F:/lc_work/外部样本/rare_class_dl/Rubber_SEA_2020/china_tiles/ 下 6 个 tif
        （任务指定 5 瓦 × 4°×4° @10m；补入 rubber_SEA_2020_96.17E_21.65N.tif —— 覆盖德宏/保山/
         临沧西部云南橡胶区，任务清单遗漏；瓦片名 = 左下角坐标；1=橡胶 / 0=其他林 / -99=nodata）
* 规则源：国界与省归属 = 数据/边界/china_100000_full.json（DataV 省界；20 个 invalid 几何
        buffer(0) 修复后栅格化，JD 九段线 name='' 剔除）；瓦片网格带 1 像元共享边界行列
        （已实测逐像元一致）→ 由北/东邻瓦片持有，本瓦片跳过 row0/col0（去重）
* 门槛：抽样 stride=20 像元（~200 m 网格）；每瓦片 ≤15,000 点、全国 ≤30,000 点
        （seed=20261013 两级随机抽稀）；面积 = 像元数×100 m²（任务口径）+ cos(φ) 校正口径
        （1/3 弧秒网格真实像元面积 100·cosφ m²）
* 输出：F:/lc_work/v31_exp/data/orchard/rubber_sea_cn_points.parquet（lon,lat,tile）
        F:/lc_work/v31_exp/results/orchard/rubber_sea_cn_stats.json
* 用法：F:/anaconda/envs/geeMAP_env/python.exe o17_rubber_sea_extract_cn.py
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely.geometry import box, shape
from shapely.ops import unary_union
from shapely.strtree import STRtree

sys.stdout.reconfigure(encoding='utf-8')
CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
import k19_provattrib as PA  # noqa: E402

TILEDIR = r'F:/lc_work/外部样本/rare_class_dl/Rubber_SEA_2020/china_tiles'
TILES = [
    'rubber_SEA_2020_100.17E_17.65N.tif',
    'rubber_SEA_2020_100.17E_21.65N.tif',
    'rubber_SEA_2020_104.17E_17.65N.tif',
    'rubber_SEA_2020_104.17E_21.65N.tif',
    'rubber_SEA_2020_108.17E_17.65N.tif',
    'rubber_SEA_2020_96.17E_21.65N.tif',   # 任务清单外补入：云南西部（德宏/保山/临沧西）
]
OUT_PTS = r'F:/lc_work/v31_exp/data/orchard/rubber_sea_cn_points.parquet'
OUT_JSON = r'F:/lc_work/v31_exp/results/orchard/rubber_sea_cn_stats.json'
BND = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/边界/china_100000_full.json'
STRIDE = 20            # 20 像元 = ~200 m 抽样
WIN = 4000             # 窗口尺寸（必须被 STRIDE 整除）
CAP_TILE, CAP_ALL = 15000, 30000
SEED = 20261013


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def fix_geom(g):
    """invalid 几何：buffer(0) 修复；非面类型时抽多边形部件再 union。"""
    if g.is_valid:
        return g
    g2 = g.buffer(0)
    if g2.is_empty:
        return g
    if g2.geom_type in ('Polygon', 'MultiPolygon'):
        return g2
    parts = [p for p in shapely.get_parts(g2) if p.geom_type == 'Polygon']
    u = unary_union(parts) if parts else g
    return u if not u.is_empty else g


def load_provs():
    gj = json.load(open(BND, encoding='utf-8'))
    out, ninv = [], 0
    for f in gj['features']:
        nm = (f['properties'].get('name') or '').strip()
        g = f.get('geometry')
        if not nm or not g:            # JD 九段线（name=''）
            continue
        gg = shape(g)
        if not gg.is_valid:
            ninv += 1
            gg = fix_geom(gg)
        out.append((nm, gg))
    emit('省界载入 %d 省 / invalid 修复 %d' % (len(out), ninv))
    return out


def tile_info():
    info = {}
    for tf in TILES:
        with rasterio.open(os.path.join(TILEDIR, tf)) as ds:
            info[tf] = dict(W=ds.width, H=ds.height, res=ds.res[0],
                            l=ds.bounds.left, r=ds.bounds.right,
                            b=ds.bounds.bottom, t=ds.bounds.top,
                            crs=str(ds.crs), dt=ds.dtypes[0], nodata=ds.nodata)
    return info


def skip_flags(info):
    """共享边界行列归属：本瓦片 row0/col0 与某北/南邻瓦片末行/末列中心重合 → 本瓦片跳过。"""
    flags = {}
    for tf, a in info.items():
        r = a['res']
        skip_c = any(o['b'] == a['b'] and abs((o['r'] - 0.5 * r) - (a['l'] + 0.5 * r)) < 0.01 * r
                     for tf2, o in info.items() if tf2 != tf)
        skip_r = any(o['l'] == a['l'] and abs((o['b'] + 0.5 * r) - (a['t'] - 0.5 * r)) < 0.01 * r
                     for tf2, o in info.items() if tf2 != tf)
        flags[tf] = (skip_r, skip_c)   # (跳 row0, 跳 col0)
    return flags


def main():
    t0 = time.time()
    provs = load_provs()
    pnames = [nm for nm, _ in provs]
    geoms = [g for _, g in provs]
    tree = STRtree(geoms)
    info = tile_info()
    flags = skip_flags(info)

    tile_stats, prov_px, prov_cos = {}, np.zeros(len(provs)), np.zeros(len(provs))
    prov_pts_pre = np.zeros(len(provs), dtype=int)
    all_lon, all_lat, all_tile, all_prov = [], [], [], []

    for ti, tf in enumerate(TILES):
        a = info[tf]
        W, H, res = a['W'], a['H'], a['res']
        skip_r, skip_c = flags[tf]
        st = dict(bounds=[a['l'], a['b'], a['r'], a['t']], size=[W, H], res_deg=res,
                  crs=a['crs'], nodata=a['nodata'], skip_row0=bool(skip_r), skip_col0=bool(skip_c),
                  rub_px_tile_total=0, rub_px_cn=0)
        plon, plat, pprov = [], [], []
        with rasterio.open(os.path.join(TILEDIR, tf)) as ds:
            left, top = ds.bounds.left, ds.bounds.top
            for r0 in range(0, H, WIN):
                for c0 in range(0, W, WIN):
                    h = min(WIN, H - r0)
                    w = min(WIN, W - c0)
                    win = Window(c0, r0, w, h)
                    arr = ds.read(1, window=win)
                    ones = arr == 1
                    n1 = int(ones.sum())
                    st['rub_px_tile_total'] += n1
                    if n1 == 0:
                        continue
                    wt = ds.window_transform(win)
                    wl, wb, wr, wtt = rasterio.transform.array_bounds(h, w, wt)
                    cand = tree.query(box(wl, wb, wr, wtt))
                    if len(cand) == 0:
                        continue
                    label = np.zeros((h, w), np.int16)
                    for ci in cand:
                        tmp = rasterize([(geoms[int(ci)], 1)], out_shape=(h, w), transform=wt,
                                        fill=0, dtype='uint8')
                        if tmp.any():
                            label[(tmp == 1) & (label == 0)] = int(ci) + 1
                    valid = np.ones((h, w), bool)
                    if skip_c and c0 == 0:
                        valid[:, 0] = False
                    if skip_r and r0 == 0:
                        valid[0, :] = False
                    m = ones & valid & (label > 0)
                    if m.any():
                        st['rub_px_cn'] += int(m.sum())
                        cw = np.cos(np.radians(top - (np.arange(r0, r0 + h) + 0.5) * res))
                        for ci in np.unique(label[m]):
                            mm = m & (label == ci)
                            prov_px[int(ci) - 1] += int(mm.sum())
                            prov_cos[int(ci) - 1] += float((mm.sum(axis=1) * cw).sum())
                        # 抽样点（与面积口径同一 in-China mask；网格自窗口起点每 20 像元）
                        sm = m[::STRIDE, ::STRIDE]
                        if sm.any():
                            ii, jj = np.nonzero(sm)
                            rg = r0 + ii * STRIDE
                            cg = c0 + jj * STRIDE
                            plon.append(left + (cg + 0.5) * res)
                            plat.append(top - (rg + 0.5) * res)
                            pprov.append(label[::STRIDE, ::STRIDE][ii, jj] - 1)
        if plon:
            plon = np.concatenate(plon)
            plat = np.concatenate(plat)
            pprov = np.concatenate(pprov)
        else:
            plon = np.array([]); plat = np.array([]); pprov = np.array([], dtype=int)
        st['points_pre_cap'] = int(len(plon))
        for ci in np.unique(pprov) if len(pprov) else np.array([], dtype=int):
            prov_pts_pre[int(ci)] += int((pprov == ci).sum())
        # 瓦片级上限 15k（固定种子）
        if len(plon) > CAP_TILE:
            idx = np.random.default_rng(SEED + ti).choice(len(plon), CAP_TILE, replace=False)
            plon, plat, pprov = plon[idx], plat[idx], pprov[idx]
        st['points_after_tile_cap'] = int(len(plon))
        emit('%s: 橡胶像元 全瓦 %d / 境内 %d（100m²口径 %.1f km²）点 %d→%d' % (
            tf, st['rub_px_tile_total'], st['rub_px_cn'], st['rub_px_cn'] * 100 / 1e6,
            st['points_pre_cap'], st['points_after_tile_cap']))
        tile_stats[tf] = st
        all_lon.append(plon); all_lat.append(plat); all_prov.append(pprov)
        all_tile.append(np.array([os.path.splitext(tf)[0]] * len(plon)))

    lon = np.concatenate(all_lon); lat = np.concatenate(all_lat)
    prov_idx = np.concatenate(all_prov); tile = np.concatenate(all_tile)
    n_after_tile_cap = len(lon)
    # 全国级上限 30k
    if n_after_tile_cap > CAP_ALL:
        idx = np.random.default_rng(SEED).choice(n_after_tile_cap, CAP_ALL, replace=False)
        idx.sort()
        lon, lat, prov_idx, tile = lon[idx], lat[idx], prov_idx[idx], tile[idx]
    df = pd.DataFrame({'lon': lon, 'lat': lat, 'tile': tile})
    os.makedirs(os.path.dirname(OUT_PTS), exist_ok=True)
    df.to_parquet(OUT_PTS, index=False)
    md5 = hashlib.md5(open(OUT_PTS, 'rb').read()).hexdigest()
    for tf in TILES:
        tile_stats[tf]['points_final'] = int((df['tile'] == os.path.splitext(tf)[0]).sum())
    emit('交付 抽样候选 %d → 瓦片限后 %d → 全国限后 %d → %s (md5 %s)' % (
        sum(v['points_pre_cap'] for v in tile_stats.values()), n_after_tile_cap, len(df), OUT_PTS, md5))

    # 抽样点间距核查（交付点随机 3000 的最近邻，米制）
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(SEED)
    sidx = rng.choice(len(df), min(3000, len(df)), replace=False)
    s = df.iloc[sidx]
    xy = np.c_[s['lon'].values * 111320 * np.cos(np.radians(s['lat'].values)),
               s['lat'].values * 111320]
    dd, _ = cKDTree(xy).query(xy, k=2)
    nn = dd[:, 1]
    spacing = dict(n_sample=len(sidx), nn_min_m=float(nn.min()),
                   nn_p05_m=float(np.percentile(nn, 5)), nn_median_m=float(np.median(nn)),
                   nn_p95_m=float(np.percentile(nn, 95)),
                   note='名义 200 m 网格；经度向实距 = 20×10×cosφ ≈ 180–191 m（瓦片纬度 17.7–25.65°N），纬度向 200 m')

    # k19 attribute() 独立复核省归属（含 invalid 原始几何 contains）
    df2 = pd.DataFrame({'lon': df['lon'].values, 'lat': df['lat'].values})
    pa = PA.attribute(df2)
    mismatch = int((pa != np.array([pnames[int(i)] if i >= 0 else '' for i in prov_idx])).sum())

    # 分省统计
    prov_stat = {}
    for i, nm in enumerate(pnames):
        if prov_px[i] == 0 and prov_pts_pre[i] == 0:
            continue
        prov_stat[nm] = dict(rub_px=int(prov_px[i]),
                             area_km2_100m2=float(prov_px[i] * 100 / 1e6),
                             area_km2_cos=float(prov_cos[i] * 100 / 1e6),
                             points_pre_cap=int(prov_pts_pre[i]),
                             points_delivered_k19=int((pa == nm).sum()))
    total_px = int(prov_px.sum())
    total = dict(rub_px_cn=total_px,
                 area_km2_100m2=float(total_px * 100 / 1e6),
                 area_km2_cos=float(prov_cos.sum() * 100 / 1e6),
                 points_pre_cap=int(sum(v['points_pre_cap'] for v in tile_stats.values())),
                 points_after_tile_cap=int(n_after_tile_cap),
                 points_final=int(len(df)))
    # 瓦片逐像素数一致性
    chk = sum(v['rub_px_cn'] for v in tile_stats.values())
    out = dict(date='2026-10-13', script='o17_rubber_sea_extract_cn.py',
               tiles=TILES, tile_stats=tile_stats, province=prov_stat, total=total,
               caps=dict(per_tile=CAP_TILE, all=CAP_ALL, seed=SEED, stride=STRIDE),
               pixel_sum_check=dict(tile_sum=chk, prov_sum=total_px),
               k19_mismatch=mismatch, spacing_check=spacing,
               parquet=OUT_PTS, parquet_md5=md5,
               runtime_s=round(time.time() - t0, 1))
    json.dump(out, open(OUT_JSON, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    emit('k19 复核不一致 %d；省数 %d；总 %d px（%.1f km²·100m² / %.1f km²·cos）；%.1fs' % (
        mismatch, len(prov_stat), total_px, total['area_km2_100m2'], total['area_km2_cos'],
        time.time() - t0))
    print(json.dumps(prov_stat, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
