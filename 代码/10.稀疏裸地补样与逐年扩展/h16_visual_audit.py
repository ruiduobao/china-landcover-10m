# -*- coding: utf-8 -*-
"""h16_visual_audit.py — 逐年补样点的 S2 真彩色人工目视抽查（接触表）

* 目的：给"精度"增加一路独立目视证据（不是规则、不是产品投票）
* 抽样：每年 稀疏 12 + 裸地 12（固定种子），默认 2018 / 2021 / 2024
* 影像：当年 6–9 月 S2 无云中值合成，10 m，每点 256 px（≈2.56 km 视场）
* 输出：reports/visual_audit/sheet_<year>.png（4×6 接触表，带序号与类别标签）+ points.csv
* 用法：python h16_visual_audit.py --acct <acct> --pid <pid> [--years 2018 2021 2024]
"""
import argparse
import io
import os
import sys

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw

sys.stdout.reconfigure(encoding='utf-8')
BASE = r'F:/lc_work/bare_sparse_v3'
sys.path.insert(0, r'F:/地理所/论文/中国土地覆盖数据_2017-2024/代码/8.全国生产')
OUT = os.path.join(BASE, 'reports', 'visual_audit')
PROXY = {'http': 'socks5h://127.0.0.1:7890', 'https': 'socks5h://127.0.0.1:7890'}
SIZE = 320
HALF = 0.0045   # ≈500 m 半宽 → 1 km 视场
PER_CLASS = 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acct', required=True)
    ap.add_argument('--pid', required=True)
    ap.add_argument('--years', nargs='*', type=int, default=[2018, 2021, 2024])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    from prod_conf import apply_account_env
    apply_account_env(a.acct)
    import ee
    ee.Initialize(project=a.pid)

    metadata = []
    for y in a.years:
        fp = os.path.join(BASE, 'data', 'yearly', 'topup_%d.parquet' % y)
        if not os.path.isfile(fp):
            print('%d 缺层' % y); continue
        d = pd.read_parquet(fp, columns=['pid', 'lon', 'lat', 'cl'])
        parts = []
        for cl in (13, 22):
            g = d[d.cl == cl]
            parts.append(g.sample(min(PER_CLASS, len(g)), random_state=23))
        s = pd.concat(parts, ignore_index=True).reset_index(drop=True)
        col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
               .filterDate('%d-04-01' % y, '%d-11-01' % y)
               .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
               .map(lambda im: im.updateMask(im.select('SCL').neq(3)
                                             .And(im.select('SCL').neq(8))
                                             .And(im.select('SCL').neq(9)))).median())
        rgb = col.select(['B4', 'B3', 'B2']).multiply(0.00025)   # 约 0-0.4 拉伸
        cells = []
        for i, r in s.iterrows():
            half = HALF
            region = ee.Geometry.Rectangle([r.lon - half, r.lat - half, r.lon + half, r.lat + half])
            url = rgb.getThumbURL({'region': region, 'dimensions': SIZE, 'format': 'png',
                                   'min': 0.005, 'max': 0.40, 'gamma': 1.3})
            try:
                resp = requests.get(url, proxies=PROXY, timeout=90)
                resp.raise_for_status()
                im = Image.open(io.BytesIO(resp.content)).convert('RGB').resize((SIZE, SIZE))
            except Exception as e:
                print('  点 %d 取图失败：%s' % (i, str(e)[:80]))
                im = Image.new('RGB', (SIZE, SIZE), (30, 30, 30))
            cells.append(im)
            metadata.append({'year': y, 'idx': i, 'pid': r.pid, 'cl': int(r.cl), 'lon': r.lon, 'lat': r.lat})
        # 每类一张表：4 列 × 3 行，320 px/格，中心十字准星
        for fam, tag in [('sparse', 'S'), ('bare', 'B')]:
            idxs = [i for i in range(len(s)) if (s.iloc[i].cl == 13) == (fam == 'sparse')]
            cols, rows = 4, 3
            sheet = Image.new('RGB', (cols * SIZE, rows * SIZE), (0, 0, 0))
            dr = ImageDraw.Draw(sheet)
            for j, i in enumerate(idxs[:cols * rows]):
                im = cells[i]
                cx, cy = (j % cols) * SIZE, (j // cols) * SIZE
                sheet.paste(im, (cx, cy))
                mx, my = cx + SIZE // 2, cy + SIZE // 2
                dr.line([mx - 14, my, mx + 14, my], fill=(255, 0, 0), width=2)
                dr.line([mx, my - 14, mx, my + 14], fill=(255, 0, 0), width=2)
                lab = '%d %s' % (j, tag)
                dr.rectangle([cx, cy, cx + 46, cy + 18], fill=(0, 0, 0))
                dr.text((cx + 3, cy + 3), lab, fill=(255, 255, 0))
            fp = os.path.join(OUT, 'sheet_%d_%s.png' % (y, fam))
            sheet.save(fp)
            print('%d %s → %s' % (y, fam, fp), flush=True)
    pd.DataFrame(metadata).to_csv(os.path.join(OUT, 'points.csv'), index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
