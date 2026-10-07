# -*- coding: utf-8 -*-
"""
n2_orchard_samples.py — 园地(11)专项样本（全本地）
* 数据源（本地）:
  1) 全球油棕 OP-YoP: E:\地理所\论文\博士论文\数据\数据\全球油棕分布数据集\GlobalOilPalm_OP-YoP\
     (30m, 值=种植年 1989-2021, 0=非) —— 中国分布: 海南/滇南
  2) Du 全球人工林: ...\全球人工林产品数据\TIF原始数据\global_plantedForest_30m_*.tif (30m, {0,1,2})
     —— 类码语义需实证（timber plantation ≠ 园地！）
  3) 中国逐年人工林 PF_YYYY (10°分幅)
* 步骤:
  verify: 三个已知点(海南油棕/大兴安岭天然林/广西桉树人工林)对比 OP-YoP、Du、CLCD → 决定类码映射
  gen   : 生成园地候选点（油棕区 + Du×CLCD过渡），写 samples_orchard_candidates.parquet
"""
import sys, os, glob
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from pyproj import Transformer

ROOT = r'E:\地理所\论文\博士论文\数据\数据'
OP_DIR = os.path.join(ROOT, '全球油棕分布数据集', 'GlobalOilPalm_OP-YoP')
DU_DIR = os.path.join(ROOT, '全球人工林产品数据', 'TIF原始数据')
OUT = r'F:\地理所\论文\中国土地覆盖数据_2017-2024\数据\外部样本源'
os.makedirs(OUT, exist_ok=True)

KNOWN = {
    '海南油棕区': (109.5, 19.2),
    '大兴安岭天然林': (122.5, 51.0),
    '广西桉树人工林': (109.4, 22.8),
}

def open_tile(directory, lon, lat):
    """在分幅目录中找覆盖 (lon,lat) 的 tif 并打开"""
    for fp in glob.glob(os.path.join(directory, '*.tif')):
        try:
            with rasterio.open(fp) as s:
                b = s.bounds
                if b.left <= lon <= b.right and b.bottom <= lat <= b.top:
                    return fp, s.crs
        except Exception:
            continue
    return None, None

def probe(fp, lon, lat, win=25):
    with rasterio.open(fp) as s:
        if s.crs and s.crs.to_epsg() != 4326:
            tr = Transformer.from_crs(4326, s.crs, always_xy=True)
            x, y = tr.transform(lon, lat)
        else:
            x, y = lon, lat
        r, c = s.index(x, y)
        rr0, cc0 = max(0, r-win), max(0, c-win)
        a = s.read(1, window=Window(cc0, rr0, 2*win, 2*win))
        v, cnt = np.unique(a, return_counts=True)
        return dict(zip(v.tolist(), (cnt/ cnt.sum()).round(3)))

def verify():
    print('=== 类码语义实证 ===')
    for name, (lon, lat) in KNOWN.items():
        print(f'-- {name} ({lon},{lat})')
        fp, _ = open_tile(OP_DIR, lon, lat)
        if fp:
            print('   OP-YoP:', probe(fp, lon, lat))
        else:
            print('   OP-YoP: 无覆盖瓦片')
        fp, _ = open_tile(DU_DIR, lon, lat)
        if fp:
            print('   Du 30m:', probe(fp, lon, lat))
        else:
            print('   Du 30m: 无覆盖瓦片')

def gen_orchard(n_target=8000):
    """海南/滇南: OP-YoP>0 的像元 → 园地候选点(30m中心)"""
    import random
    random.seed(7)
    pts = []
    tiles = [fp for fp in glob.glob(os.path.join(OP_DIR, '*.tif'))]
    # 只处理中国附近: 简单按文件名/范围过滤太贵，直接逐幅读 header
    for fp in tiles:
        try:
            with rasterio.open(fp) as s:
                b = s.bounds
                # 中国域: lon 97-122, lat 17-31（海南/云南/广东广西/福建）
                if b.right < 97 or b.left > 122 or b.top < 17 or b.bottom > 31:
                    continue
                a = s.read(1)
                ys, xs = np.where(a > 0)
                if len(ys) == 0:
                    continue
                take = min(n_target // max(len(tiles)//3, 1), len(ys))
                pick = random.sample(range(len(ys)), take)
                for k in pick:
                    r, c = ys[k], xs[k]
                    x, y = s.transform * (c + 0.5, r + 0.5)
                    pts.append({'lon': x, 'lat': y,
                                'op_year': int(a[r, c]),
                                'src': 'oilpalm_yop'})
        except Exception as e:
            print('ERR', fp, str(e)[:80])
    df = pd.DataFrame(pts)
    fp_out = os.path.join(OUT, 'samples_orchard_candidates.parquet')
    df.to_parquet(fp_out, index=False)
    print('园地候选(油棕):', len(df), '->', fp_out)
    return df

if __name__ == '__main__':
    step = sys.argv[1] if len(sys.argv) > 1 else 'verify'
    if step == 'verify':
        verify()
    elif step == 'gen':
        gen_orchard()
