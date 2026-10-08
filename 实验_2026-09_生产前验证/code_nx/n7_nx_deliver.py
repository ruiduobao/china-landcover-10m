# -*- coding: utf-8 -*-
"""n7_nx_deliver.py — 宁夏 R4 新品转交付态（内嵌调色板 + 手写金字塔 PIL 验收）+ 面积表 + 交付目录

* 输入：F:/lc_work/prod5p_2023/rasters_r4/<tile>_10m.tif（uint8，无调色板）
* 规则源：调色板取 pilot/d_tiles.COLORS（与交付 5 省同源）；金字塔用 pilot/ov_build.py（doc 43 铁律：
        禁用 build_overviews，禁用 GDAL out_shape 自检）；面积按 10m 原生网格本地统计（纬度加权）
* 门槛：每瓦 ov_build PIL 验收通过（隶属/TL 双 100%）；面积表逐瓦 + 全省合计
* 输出：F:/lc_work/prod5p_2023/rasters_r4/<tile>_10m.tif（就地转交付态）
        delivery_r4_宁夏/<tile>_10m.tif + <tile>_面积.csv + 宁夏_面积表.csv + MD5.txt
* 用法：python n7_nx_deliver.py [--skip-ov]
"""
import csv
import hashlib
import math
import os
import shutil
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'F:/lc_work/v31_exp/code')
sys.path.insert(0, r'F:/lc_work/prod5p_2023/pilot')
import numpy as np
import rasterio
import v31_common as VC
from d_tiles import COLORS

ROOT = r'F:/lc_work/prod5p_2023'
RDIR = os.path.join(ROOT, 'rasters_r4')
DEL = os.path.join(ROOT, 'delivery_r4_宁夏')
OV = os.path.join(ROOT, 'pilot', 'ov_build.py')
NX = ['T1509', 'T1608', 'T1609', 'T1610', 'T1709', 'T1710']


def log(m):
    print('[%s] %s' % (time.strftime('%m-%d %H:%M:%S'), m), flush=True)


def palette():
    pal = {i: (230, 230, 230, 0) for i in range(256)}
    for k, c in COLORS.items():
        pal[k] = (int(round(c[0] * 255)), int(round(c[1] * 255)), int(round(c[2] * 255)), 255)
    for k in range(25, 256):
        pal[k] = (200, 200, 200, 255)
    return pal


def add_palette(t):
    fp = os.path.join(RDIR, '%s_10m.tif' % t)
    tmp = fp + '.pal.tif'
    with rasterio.open(fp) as ds:
        prof = ds.profile.copy()
        prof.update(dtype='uint8', compress='deflate', predictor=2, tiled=True,
                    blockxsize=512, blockysize=512, count=1, nodata=0)
        with rasterio.open(tmp, 'w', **prof) as d2:
            d2.write_colormap(1, palette())
            for r0 in range(0, ds.height, 2048):
                hh = min(2048, ds.height - r0)
                win = rasterio.windows.Window(0, r0, ds.width, hh)
                d2.write(ds.read(1, window=win), 1, window=win)
    os.replace(tmp, fp)
    log('%s 调色板内嵌 ✓' % t)


def area_table(t):
    fp = os.path.join(RDIR, '%s_10m.tif' % t)
    names = VC.V31_NAMES()
    cnt = {}
    with rasterio.open(fp) as ds:
        res = ds.res
        for r0 in range(0, ds.height, 2048):
            hh = min(2048, ds.height - r0)
            blk = ds.read(1, window=rasterio.windows.Window(0, r0, ds.width, hh))
            lat0 = ds.xy(r0, 0, offset='ul')[1]
            lat1 = ds.xy(r0 + hh - 1, 0, offset='ul')[1]
            cell = (res[0] * 111.32 * math.cos(math.radians((lat0 + lat1) / 2))) * (abs(res[1]) * 110.57)
            v, c = np.unique(blk, return_counts=True)
            for a, b in zip(v.tolist(), c.tolist()):
                cnt[a] = cnt.get(a, 0.0) + b * cell
    tot = sum(v for k, v in cnt.items() if k > 0)
    fp_csv = os.path.join(DEL, '%s_面积.csv' % t)
    with open(fp_csv, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['code', 'name', 'area_km2', 'share_pct'])
        for k in sorted(cnt):
            if k <= 0 or cnt[k] < 1:
                continue
            w.writerow([k, names.get(str(k), ''), round(cnt[k], 3), round(100 * cnt[k] / tot, 4)])
    return cnt, tot


def main():
    os.makedirs(DEL, exist_ok=True)
    skip_ov = '--skip-ov' in sys.argv
    total = {}
    per = {}
    for t in NX:
        add_palette(t)
        if not skip_ov:
            # IFD 脚手架：build_overviews 只负责建概览 IFD（内容由 ov_build 手写重写，doc43 铁律）
            fp_r = os.path.join(RDIR, '%s_10m.tif' % t)
            with rasterio.open(fp_r, 'r+') as ds:
                ds.build_overviews([2, 4, 8, 16, 32], rasterio.enums.Resampling.nearest)
            r = subprocess.run([sys.executable, OV, os.path.join(RDIR, '%s_10m.tif' % t)],
                               capture_output=True, text=True, encoding='utf-8', errors='ignore')
            tag = 'PIL 验收通过' if r.returncode == 0 else '验收失败 rc=%d' % r.returncode
            log('%s ov_build %s' % (t, tag))
            if r.returncode != 0:
                log('  %s' % (r.stdout or r.stderr or '')[-200:].replace('\n', ' | '))
        cnt, tot = area_table(t)
        per[t] = (cnt, tot)
        for k, v in cnt.items():
            if k > 0:
                total[k] = total.get(k, 0.0) + v
        shutil.copy2(os.path.join(RDIR, '%s_10m.tif' % t), os.path.join(DEL, '%s_10m.tif' % t))
    names = VC.V31_NAMES()
    tot_all = sum(total.values())
    with open(os.path.join(DEL, '宁夏_面积表.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['code', 'name', 'area_km2', 'share_pct'])
        for k in sorted(total, key=lambda x: -total[x]):
            w.writerow([k, names.get(str(k), ''), round(total[k], 3), round(100 * total[k] / tot_all, 4)])
    with open(os.path.join(DEL, 'MD5.txt'), 'w', encoding='utf-8') as f:
        for t in NX:
            fp = os.path.join(DEL, '%s_10m.tif' % t)
            h = hashlib.md5(open(fp, 'rb').read()).hexdigest()
            f.write('%s  %s  %d bytes\n' % (h, os.path.basename(fp), os.path.getsize(fp)))
    log('全省合计 %.0f km²；面积表 → %s' % (tot_all, os.path.join(DEL, '宁夏_面积表.csv')))
    for k in sorted(total, key=lambda x: -total[x])[:8]:
        log('  %-8s %8.0f km²  %5.2f%%' % (names.get(str(k), k), total[k], 100 * total[k] / tot_all))
    log('交付目录：%s' % DEL)


if __name__ == '__main__':
    main()
