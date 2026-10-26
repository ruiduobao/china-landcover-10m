# -*- coding: utf-8 -*-
"""t0_terrain_instrument_check.py — 地形仪器检查：本地 60m 坡向/坡度 vs 既有 aspect_check(GEE 30m) 窗口对照

* 输入（只读）：F:/lc_work/v31_exp/data/eco_gate/_dem/dem_T1508.tif（本地 SRTM 60m 高程）
    F:/lc_work/v31_exp/data/aspect_check/a_crop_dem/aspect.tif（2026-10-09 另一会话产物，GEE USGS/SRTMGL1_003
    aspect 10m scale → nearest 对齐到 10m 网格；仅只读复用，不改动）
* 规则：本地坡度/坡向由 60m 高程计算，经纬格距按纬度换算为米（WGS84 球面近似）；
    同时给出"不换算（度当米）"的反例，量化经纬单位不换算造成的坡向偏差。
* 门槛：无（仪器检查）；输出 agreement 统计
* 输出：terrain/t0_instrument_check.json（+ 控制台）
* 用法：python t0_terrain_instrument_check.py
"""
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
OUTD = r'F:/lc_work/prod2nd_2023/reports/quality_review_2026-10-15/terrain'
DEM = r'F:/lc_work/v31_exp/data/eco_gate/_dem/dem_T1508.tif'
ASP_REF = r'F:/lc_work/v31_exp/data/aspect_check/a_crop_dem/aspect.tif'
WIN = (103.8232, 34.146, 103.9132, 34.236)


def m_per_deg(lat_deg):
    lat = np.deg2rad(lat_deg)
    m_lat = 111132.92 - 559.82 * np.cos(2 * lat) + 1.175 * np.cos(4 * lat) - 0.0023 * np.cos(6 * lat)
    m_lon = 111412.84 * np.cos(lat) - 93.5 * np.cos(3 * lat) + 0.118 * np.cos(5 * lat)
    return m_lat, m_lon


def ang_diff(a, b):
    d = np.abs((a - b + 180.0) % 360.0 - 180.0)
    return d


def main():
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.windows import from_bounds
    with rasterio.open(DEM) as s:
        tfm = s.transform
        win = from_bounds(*WIN, transform=tfm)
        win = win.round_offsets().round_lengths()
        elev = s.read(1, window=win).astype(np.float32)
        wtfm = rasterio.windows.transform(win, tfm)
    H, W = elev.shape
    lats = wtfm.f - (-wtfm.e) * (np.arange(H) + 0.5)
    m_lat, m_lon = m_per_deg(lats)
    out = {'dem': DEM, 'dem_shape_win': [int(H), int(W)], 'win': list(WIN),
           'dem_res_deg': float(wtfm.a), 'lat_range': [float(lats.min()), float(lats.max())]}

    to_rad = np.deg2rad(np.nan)

    def slope_aspect(sx_scale, sy_scale):
        gE = np.gradient(elev, axis=1) / sx_scale
        gN = -np.gradient(elev, axis=0) / sy_scale
        slope = np.degrees(np.arctan(np.sqrt(gE * gE + gN * gN)))
        aspect = (np.degrees(np.arctan2(-gE, -gN)) + 360.0) % 360.0
        return slope, aspect

    slope_m, asp_m = slope_aspect(wtfm.a * m_lon[:, None], (-wtfm.e) * m_lat[:, None])   # 米制（正）
    slope_d, asp_d = slope_aspect(wtfm.a, -wtfm.e)                                       # 反例（deg 当 m）

    # 参考：既有 GEE aspect（10m 网格）
    with rasterio.open(ASP_REF) as s:
        ref = s.read(1).astype(np.float32)
        ref_tfm, ref_shape = s.transform, s.shape
    mine_on_ref = np.empty(ref_shape, dtype=np.float32)
    reproject(asp_m, mine_on_ref, src_transform=wtfm, src_crs='EPSG:4326',
              dst_transform=ref_tfm, dst_crs='EPSG:4326', resampling=Resampling.nearest)
    valid = np.isfinite(ref) & (ref > 0) & (ref < 360)

    def sunny_shady(a, m=None):
        m = np.ones_like(a, bool) if m is None else m
        sunny = ((a >= 135) & (a <= 225) & m).sum()
        shady = (((a >= 0) & (a <= 45)) | ((a >= 315) & (a <= 360)) & m).sum()
        return int(sunny), int(shady)

    su_m, sh_m = sunny_shady(asp_m)
    su_d, sh_d = sunny_shady(asp_d)
    su_r, sh_r = sunny_shady(ref, valid)
    out['sunny_shady_meter_correct'] = {'sunny': su_m, 'shady': sh_m,
                                        'ratio_sunny_shady': round(su_m / sh_m, 4)}
    out['sunny_shady_unit_unconverted'] = {'sunny': su_d, 'shady': sh_d,
                                           'ratio_sunny_shady': round(su_d / sh_d, 4)}
    out['sunny_shady_ref_gee30m'] = {'sunny': su_r, 'shady': sh_r,
                                     'ratio_sunny_shady': round(su_r / sh_r, 4)}
    d = ang_diff(mine_on_ref, ref)
    dv = d[valid]
    out['vs_gee_ref_10m'] = {
        'n': int(valid.sum()),
        'median_abs_deg_diff': float(np.median(dv)),
        'frac_within_22.5deg': float((dv <= 22.5).mean()),
        'frac_within_45deg': float((dv <= 45).mean()),
        'mean_slope_deg_mine': float(np.nanmean(slope_m)),
        'mean_slope_deg_unconverted': float(np.nanmean(slope_d)),
        'note': '参考为 GEE 30m aspect 重采样10m（nearest）；本机为 60m 高程本地计算（米制换算），'
                '两者分辨率不同，一致性为量级校验而非逐像元等价'
    }
    # 米制 vs 未换算的坡向差（同网格，直接可比）
    dv2 = ang_diff(asp_m, asp_d).ravel()
    out['meter_vs_unconverted_same_grid'] = {
        'median_abs_deg_diff': float(np.median(dv2)),
        'frac_within_22.5deg': float((dv2 <= 22.5).mean()),
        'note': '同一高程下，是否按纬度米制换算经纬格距对坡向角的敏感性'
    }
    json.dump(out, open(os.path.join(OUTD, 't0_instrument_check.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
