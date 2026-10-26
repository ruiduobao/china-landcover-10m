# -*- coding: utf-8 -*-
"""b7_validate_bare_south.py — 南方多源裸地层的核验（来源一致性/双路/目视接触表）

* 输入：data/final/bare_south_cand.parquet（过滤后候选）、data/yearly/topup_bare_south_<year>.parquet
* 检查：
    1) 2023 层点数、AEF 完备
    2) 多源来源一致性：CLCD=7 / WC=60 / DW bare≥0.4 / MODIS=16 的占比
    3) 双路证据：100 点独立最小构建（DW bare/water + S2 NDVI）与存档值 max|Δ|
    4) 目视接触表：60 点（2023），S2 真彩色 1 km + 十字
* 输出：data/yearly/bare_south_verify.json、reports/bare_south_verify.md、reports/visual_audit/bare_south_2023.png
* 用法：python b7_validate_bare_south.py --acct <acct> --pid <pid>
"""
import argparse
import glob
import io
import json
import os
import sys

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw

sys.stdout.reconfigure(encoding='utf-8')
BASE = r'F:/lc_work/bare_sparse_v3'
PROXY = {'http': 'socks5h://127.0.0.1:7890', 'https': 'socks5h://127.0.0.1:7890'}
FEATS = ['A%02d' % i for i in range(64)]
SIZE, HALF = 320, 0.0045


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acct', required=True)
    ap.add_argument('--pid', required=True)
    ap.add_argument('--n-vis', type=int, default=60)
    a = ap.parse_args()
    sys.path.insert(0, r'F:/地理所/论文/中国土地覆盖数据_2017-2024/代码/8.全国生产')
    from prod_conf import apply_account_env
    apply_account_env(a.acct)
    import ee
    ee.Initialize(project=a.pid)
    res = {}
    cand = pd.read_parquet(os.path.join(BASE, 'data', 'bare_cand', 'bare_cand_filtered.parquet'))
    res['candidates'] = int(len(cand))
    y23 = os.path.join(BASE, 'data', 'yearly', 'topup_bare_south_2023.parquet')
    if os.path.isfile(y23):
        d = pd.read_parquet(y23)
        res['year2023'] = {'n': int(len(d)),
                           'aef_missing': int(d[FEATS].isna().all(axis=1).sum()) if set(FEATS) <= set(d.columns) else None}
    # 来源一致性
    res['source_shares'] = {
        'clcd7_pct': round(float((cand.clcd == 7).mean() * 100), 1),
        'wc60_pct': round(float((cand.wc == 60).mean() * 100), 1),
        'dw_bare_ge04_pct': round(float((cand.bare >= 0.4).mean() * 100), 1),
        'modis16_pct': round(float((cand.modis_lc == 16).mean() * 100), 1),
        'votes_mean': round(float(cand.votes.mean()), 2)}
    res['by_prov'] = {k: int(v) for k, v in cand.ct_prov.value_counts().head(12).items() if isinstance(k, str)}
    # 双路
    if os.path.isfile(y23):
        d = pd.read_parquet(y23)
        ev = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(
            BASE, 'data', 'evidence_yearly_bare', '2023', '*.parquet')))], ignore_index=True).drop_duplicates('pid')
        s = d.sample(min(100, len(d)), random_state=5).merge(ev, on='pid', suffixes=('', '_st'), how='left')
        dw = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1').filterDate('2023-01-01', '2024-01-01')
              .select(['bare', 'water']).mean().unmask(-1))
        s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterDate('2023-06-01', '2023-09-01')
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))
              .map(lambda im: im.updateMask(im.select('SCL').neq(3).And(im.select('SCL').neq(8))
                                            .And(im.select('SCL').neq(9)))).median())
        nd = s2.normalizedDifference(['B8', 'B4']).multiply(1000).rename('s2_ndvi_xia').toFloat().unmask(-9999)
        img = ee.Image.cat([dw, nd]).toFloat()
        feats = [ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]), {'pid': r.pid})
                 for r in s[['pid', 'lon', 'lat']].itertuples(index=False)]
        fc = img.sampleRegions(collection=ee.FeatureCollection(feats), scale=30, tileScale=4, geometries=False)
        got = pd.DataFrame([f['properties'] for f in fc.getInfo()['features']])
        m = s.merge(got, on='pid', suffixes=('', '_new'), how='left')
        da = {}
        for b in ['bare', 'water', 's2_ndvi_xia']:
            if b in m.columns and b + '_new' in m.columns:
                x = pd.to_numeric(m[b], errors='coerce'); y = pd.to_numeric(m[b + '_new'], errors='coerce')
                ok = x.notna() & y.notna() & (x > -9000) & (y > -9000)
                dd = (x[ok] - y[ok]).abs()
                da[b] = {'n': int(ok.sum()), 'max_abs_diff': float(dd.max()) if len(dd) else None}
        res['double_arm'] = da
    # 目视接触表
    if os.path.isfile(y23):
        d = pd.read_parquet(y23)
        s = d.sample(min(a.n_vis, len(d)), random_state=11)
        col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
               .filterDate('2023-06-01', '2023-09-01')
               .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
               .map(lambda im: im.updateMask(im.select('SCL').neq(3).And(im.select('SCL').neq(8))
                                             .And(im.select('SCL').neq(9)))).median())
        rgb = col.select(['B4', 'B3', 'B2']).multiply(0.00025)
        cells = []
        for i, r in s.reset_index(drop=True).iterrows():
            reg = ee.Geometry.Rectangle([r.lon - HALF, r.lat - HALF, r.lon + HALF, r.lat + HALF])
            url = rgb.getThumbURL({'region': reg, 'dimensions': SIZE, 'format': 'png',
                                   'min': 0.005, 'max': 0.40, 'gamma': 1.3})
            try:
                im = Image.open(io.BytesIO(requests.get(url, proxies=PROXY, timeout=90).content)).convert('RGB').resize((SIZE, SIZE))
            except Exception:
                im = Image.new('RGB', (SIZE, SIZE), (30, 30, 30))
            cells.append(im)
        cols, rows = 6, 10
        sheet = Image.new('RGB', (cols * SIZE, rows * SIZE), (0, 0, 0))
        dr = ImageDraw.Draw(sheet)
        for i, im in enumerate(cells[:cols * rows]):
            cx, cy = (i % cols) * SIZE, (i // cols) * SIZE
            sheet.paste(im, (cx, cy))
            mx, my = cx + SIZE // 2, cy + SIZE // 2
            dr.line([mx - 14, my, mx + 14, my], fill=(255, 0, 0), width=2)
            dr.line([mx, my - 14, mx, my + 14], fill=(255, 0, 0), width=2)
            dr.rectangle([cx, cy, cx + 40, cy + 16], fill=(0, 0, 0))
            dr.text((cx + 3, cy + 2), str(i), fill=(255, 255, 0))
        os.makedirs(os.path.join(BASE, 'reports', 'visual_audit'), exist_ok=True)
        fp = os.path.join(BASE, 'reports', 'visual_audit', 'bare_south_2023.png')
        sheet.save(fp)
        res['visual_sheet'] = fp
    os.makedirs(os.path.join(BASE, 'reports'), exist_ok=True)
    json.dump(res, open(os.path.join(BASE, 'data', 'yearly', 'bare_south_verify.json'), 'w',
                        encoding='utf-8'), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1)[:1500])


if __name__ == '__main__':
    main()
