# -*- coding: utf-8 -*-
"""n6_nx_check.py — 宁夏 R4 新品 vs R0 交付品：在 133 个影像判读点上的直接校验

* 输入：F:/lc_work/prod5p_2023/rasters_batch/<t>_10m.tif（R0 交付）
        F:/lc_work/prod5p_2023/rasters_r4/<t>_10m.tif（R4 新品）
        F:/lc_work/v31_exp/data/m3/nx_判读.csv + nx_neutral_points.csv（判读真值）
* 规则源：类名→24 类码映射与 n2_nx_exp.py 同一套 RULES；9 大类用 VC.to_macro；灌草合并 {9,10,11}
* 门槛：仅统计两品都有值的点；逐组（A/S1/S2）+ 合计
* 输出：results/d2/nx_raster_check.json + 打印表
* 用法：python n6_nx_check.py
"""
import csv
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import rasterio
import v31_common as VC
import n2_nx_exp as N2

ROOT = r'F:/lc_work/prod5p_2023'
SHRUBG = (9, 10, 11)


def read_at(fp, lon, lat):
    with rasterio.open(fp) as ds:
        b = ds.bounds
        if not (b.left <= lon <= b.right and b.bottom <= lat <= b.top):
            return None
        r, c = ds.index(lon, lat)
        if not (0 <= r < ds.height and 0 <= c < ds.width):
            return None
        return int(ds.read(1, window=rasterio.windows.Window(c, r, 1, 1))[0, 0])


def pack(truth, pred, tag):
    t, p = np.asarray(truth), np.asarray(pred)
    tm, pm = VC.to_macro(t), VC.to_macro(p)
    mg = np.where(np.isin(t, SHRUBG), -1, t), np.where(np.isin(p, SHRUBG), -1, p)
    return dict(tag=tag, n=int(len(t)), OA24=round(float((t == p).mean()), 4),
                OA9=round(float((tm == pm).mean()), 4),
                OA_sg=round(float((mg[0] == mg[1]).mean()), 4),
                pred_shrub=int((p == 10).sum()), true_shrub=int((t == 10).sum()))


def main():
    T = N2.load_truth()
    rows = []
    for _, r in T.iterrows():
        rec = dict(point_id=r['point_id'], grp=r['grp'], tile=r['tile'], truth=int(r['truth']))
        for tag, d in (('R0', 'rasters_batch'), ('R4', 'rasters_r4')):
            fp = os.path.join(ROOT, d, '%s_10m.tif' % r['tile'])
            rec[tag] = read_at(fp, r['lon'], r['lat']) if os.path.exists(fp) else None
        rows.append(rec)
    have = [r for r in rows if r['R0'] and r['R4']]
    print('判读点 %d；两品均有值 %d（缺 R4 %d）' % (
        len(rows), len(have), sum(1 for r in rows if r['R0'] and not r['R4'])))
    out = {'n_truth': len(rows), 'n_both': len(have)}
    for gname in ('ALL', 'A', 'S1', 'S2'):
        sub = have if gname == 'ALL' else [r for r in have if r['grp'] == gname]
        if not sub:
            continue
        truth = [r['truth'] for r in sub]
        for tag in ('R0', 'R4'):
            out['%s|%s' % (gname, tag)] = pack(truth, [r[tag] for r in sub], '%s|%s' % (gname, tag))
        print('%-4s n=%3d | R0 OA24 %.3f OA9 %.3f 灌草合并 %.3f 预测灌丛 %d | R4 OA24 %.3f OA9 %.3f 灌草合并 %.3f 预测灌丛 %d' % (
            gname, len(sub),
            out['%s|R0' % gname]['OA24'], out['%s|R0' % gname]['OA9'], out['%s|R0' % gname]['OA_sg'], out['%s|R0' % gname]['pred_shrub'],
            out['%s|R4' % gname]['OA24'], out['%s|R4' % gname]['OA9'], out['%s|R4' % gname]['OA_sg'], out['%s|R4' % gname]['pred_shrub']))
    VC.jsave(out, os.path.join(VC.RES, 'd2', 'nx_raster_check.json'))


if __name__ == '__main__':
    main()
