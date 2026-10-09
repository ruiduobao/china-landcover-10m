# -*- coding: utf-8 -*-
"""n10_gsnx_stats.py — 甘肃+宁夏 最终统计：面积表（分省/合计）+ 304 点独立判读验证 + 口径对比

* 输入：rasters_r4/<tile>_10m.tif（R4 新品；甘肃 26 瓦 + 宁夏 6 瓦，部分瓦跨省）
        rasters_batch/<tile>_10m.tif（宁夏旧交付品，仅用于对比）
        data/m3/gsnx_ref_judged.csv（304 点判读真值，含省/坐标/Q1/Q2）
* 规则源：面积＝10m 原生网格本地统计（纬度加权，逐瓦累加，按省裁剪用 DataV 省界 buffer -0.005°）；
        验证＝点所在瓦的像元值 vs 判读真值（24 类 OA / 9 大类 OA / 灌草合并 OA）
* 门槛：面积对账两口径（分省 / 合计）；验证只统计成品非 0 像元
* 输出：delivery_r4_甘肃宁夏/（面积 CSV + 验证报告 md + MD5.txt）
* 用法：python n10_gsnx_stats.py [--prov 甘肃|宁夏]
"""
import csv
import hashlib
import json
import math
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import rasterio
import v31_common as VC
import n2_nx_exp as N2

ROOT = r'F:/lc_work/prod5p_2023'
RDIR = os.path.join(ROOT, 'rasters_r4')
DEL = os.path.join(ROOT, 'delivery_r4_甘肃宁夏')
M3D = r'F:/lc_work/v31_exp/data/m3'
REF = os.path.join(M3D, 'gsnx_ref_judged.csv')
BOUND = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/边界/china_100000_full.json'
NX6 = ['T1509', 'T1608', 'T1609', 'T1610', 'T1709', 'T1710']
GS_EXTRA = ['T0910', 'T0911', 'T1010', 'T1011', 'T1110', 'T1111', 'T1112', 'T1210', 'T1211', 'T1212',
            'T1307', 'T1308', 'T1309', 'T1310', 'T1311', 'T1407', 'T1408', 'T1409', 'T1410',
            'T1507', 'T1508', 'T1510', 'T1607', 'T1708']
ALL = sorted(set(NX6 + GS_EXTRA))
SHRUBG = (9, 10, 11)


def log(m):
    print('[%s] %s' % (time.strftime('%m-%d %H:%M:%S'), m), flush=True)


def provinces():
    from shapely.geometry import shape
    from shapely import make_valid
    gj = json.load(open(BOUND, encoding='utf-8'))
    out = {}
    for ft in gj['features']:
        nm = ft.get('properties', {}).get('name', '')
        if nm[:2] in ('甘肃', '宁夏'):
            out[nm[:2]] = make_valid(shape(ft['geometry'])).buffer(-0.005)
    return out


def area_by_prov(polys):
    """逐瓦读像元：全分辨率精确合计 + 1/16 抽样按省归属（shapely contains_xy 向量化）。"""
    import shapely
    names = VC.V31_NAMES()
    acc = {p: {} for p in polys}
    acc['跨省合计'] = {}
    keys = list(polys)
    geoms = [polys[k] for k in keys]
    STEP = 16
    for t in ALL:
        fp = os.path.join(RDIR, '%s_10m.tif' % t)
        if not os.path.exists(fp):
            continue
        with rasterio.open(fp) as ds:
            res = ds.res
            for r0 in range(0, ds.height, 2048):
                hh = min(2048, ds.height - r0)
                blk = ds.read(1, window=rasterio.windows.Window(0, r0, ds.width, hh))
                lat0 = ds.xy(r0, 0, offset='ul')[1]
                lat1 = ds.xy(r0 + hh - 1, 0, offset='ul')[1]
                cell = (res[0] * 111.32 * math.cos(math.radians((lat0 + lat1) / 2))) * (abs(res[1]) * 110.57)
                v, n = np.unique(blk, return_counts=True)
                for a, b in zip(v.tolist(), n.tolist()):
                    if a > 0:
                        acc['跨省合计'][a] = acc['跨省合计'].get(a, 0.0) + b * cell
                # 分省：1/16 抽样（抽样像元代表 STEP² 个像元）
                sub = blk[::STEP, ::STEP]
                rr, cc = np.mgrid[0:sub.shape[0], 0:sub.shape[1]]
                yy = r0 + rr.ravel() * STEP
                xx = cc.ravel() * STEP
                xs, ys = ds.xy(yy, xx)
                cl = sub.ravel()
                ok = cl > 0
                if not ok.any():
                    continue
                lab = np.full(ok.sum(), -1, dtype=np.int8)
                xs2, ys2, cl2 = np.asarray(xs)[ok], np.asarray(ys)[ok], cl[ok]
                for gi, g in enumerate(geoms):
                    m2 = shapely.contains_xy(g, xs2, ys2)
                    lab[(lab < 0) & m2] = gi
                area1 = cell * STEP * STEP
                for gi, k in enumerate(keys):
                    sel = cl2[lab == gi]
                    if sel.size == 0:
                        continue
                    vv, nn = np.unique(sel, return_counts=True)
                    d = acc[k]
                    for a, b in zip(vv.tolist(), nn.tolist()):
                        d[a] = d.get(a, 0.0) + b * area1
    return acc, names


def point_validation(polys):
    ref = pd.read_csv(REF, encoding='utf-8-sig')
    ref = ref[ref.Q1 != '无法判读'].copy()
    ref['truth'] = ref['Q2'].map(lambda s: N2.code_of(str(s) or ''))
    ref = ref[ref.truth.notna()]
    # 载入 24 类名 → 码
    rows = []
    for _, r in ref.iterrows():
        got = {}
        for t in ALL:
            fp = os.path.join(RDIR, '%s_10m.tif' % t)
            if not os.path.exists(fp):
                continue
            with rasterio.open(fp) as ds:
                b = ds.bounds
                if not (b.left <= r.lon <= b.right and b.bottom <= r.lat <= b.top):
                    continue
                rr, cc = ds.index(r.lon, r.lat)
                if 0 <= rr < ds.height and 0 <= cc < ds.width:
                    got[t] = int(ds.read(1, window=rasterio.windows.Window(cc, rr, 1, 1))[0, 0])
        prov = next((k for k, g in polys.items() if g.contains(__import__('shapely.geometry', fromlist=['Point']).Point(r.lon, r.lat))), None)
        rows.append(dict(point_id=r.point_id, prov=prov, lon=r.lon, lat=r.lat, truth=int(r.truth),
                         pred=int(got.get(next((t for t in got), ''), 0)) if got else 0))
    df = pd.DataFrame(rows)
    df = df[df.pred > 0]
    out = {}
    for grp, sub in list(df.groupby('prov')) + [('两省合计', df)]:
        t, p = sub.truth.to_numpy(int), sub.pred.to_numpy(int)
        tm, pm = VC.to_macro(t), VC.to_macro(p)
        m = np.where(np.isin(t, SHRUBG), -1, t), np.where(np.isin(p, SHRUBG), -1, p)
        out[str(grp)] = dict(n=int(len(t)), OA24=round(float((t == p).mean()), 4),
                             OA9=round(float((tm == pm).mean()), 4),
                             OA_sg=round(float((m[0] == m[1]).mean()), 4),
                             shrub_true=int((t == 10).sum() + (t == 9).sum()),
                             shrub_pred=int((p == 10).sum() + (p == 9).sum()))
    return out


def main():
    os.makedirs(DEL, exist_ok=True)
    polys = provinces()
    log('省界就绪：%s' % list(polys))
    acc, names = area_by_prov(polys)
    # 面积表
    for p in list(polys) + ['跨省合计']:
        tot = sum(acc[p].values())
        fp = os.path.join(DEL, '面积_%s.csv' % p)
        with open(fp, 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.writer(f)
            w.writerow(['code', 'name', 'area_km2', 'share_pct'])
            for k in sorted(acc[p], key=lambda x: -acc[p][x]):
                w.writerow([k, names.get(str(k), ''), round(acc[p][k], 3), round(100 * acc[p][k] / max(1e-9, tot), 4)])
        log('%s 合计 %.0f km²；灌丛类 %.1f km²' % (p, tot, acc[p].get(10, 0.0) + acc[p].get(9, 0.0)))
    val = point_validation(polys)
    for k, v in val.items():
        log('验证 %-6s n=%3d OA24=%.3f OA9=%.3f 灌草合并=%.3f（真值灌丛 %d / 预测 %d）' % (
            k, v['n'], v['OA24'], v['OA9'], v['OA_sg'], v['shrub_true'], v['shrub_pred']))
    VC.jsave({'area': {p: {str(k): round(v, 2) for k, v in acc[p].items()} for p in acc},
              'validation': val}, os.path.join(VC.RES, 'd2', 'gsnx_stats.json'))
    with open(os.path.join(DEL, 'MD5.txt'), 'w', encoding='utf-8') as f:
        for t in ALL:
            fp = os.path.join(RDIR, '%s_10m.tif' % t)
            if os.path.exists(fp):
                f.write('%s  %s_10m.tif  %d bytes\n' % (
                    hashlib.md5(open(fp, 'rb').read()).hexdigest(), t, os.path.getsize(fp)))
    log('交付目录：%s' % DEL)


if __name__ == '__main__':
    main()
