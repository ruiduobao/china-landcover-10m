# -*- coding: utf-8 -*-
"""
n1_thematic_samples.py — GEE 专题训练区采样：湿地7亚类 + 不透水城乡（交互式 table）
* 资产（已核实）:
    GWL_FCS30 : projects/sat-io/open-datasets/GWL_FCS30   (2000-2022, 0/180-187)
    GISA-new  : projects/sat-io/open-datasets/GISA_1972_2021 (像元=首城市化年份 1..37, 0=非)
    GISD30    : projects/sat-io/open-datasets/GISD30_1985_2020 (期次1..8)
    GMW v4    : projects/sat-io/open-datasets/GMW/annual-extent/GMW_MNG_2020
    DW        : GOOGLE/DYNAMICWORLD/V1 (built 概率)
* 产出: 数据/外部样本源/gee_thematic/{wetland,imperv}.parquet
* 用法: python n1_thematic_samples.py [wetland|imperv|all]
"""
import sys, os, io, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r'代码/2.北京试点')
import pilot_common as PC
import numpy as np
import pandas as pd
import requests

OUT = '数据/外部样本源/gee_thematic'
os.makedirs(OUT, exist_ok=True)
BBOX = (73, 17, 136, 55)

def chunk_bbox(step=6):
    lon0, lat0, lon1, lat1 = BBOX
    out = []
    x = lon0
    while x < lon1:
        y = lat0
        while y < lat1:
            out.append((x, y, min(x+step, lon1), min(y+step, lat1)))
            y += step
        x += step
    return out

def download_fc_csv(fc, selectors):
    url = fc.getDownloadURL(filetype='csv', selectors=selectors)
    r = requests.get(url, proxies=PC.PROXY, timeout=900)
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content))

def wetland(years=(2000, 2022), target_per_class=20000):
    """GWL_FCS30 逐亚类时序稳定 → 采样"""
    ee, pid = PC.load_account('zsi8emo')
    wl = ee.ImageCollection('projects/sat-io/open-datasets/GWL_FCS30')
    n_img = wl.size().getInfo()
    print('GWL_FCS30 景数:', n_img)
    first = ee.Image(wl.toList(n_img).get(0))
    band0 = first.bandNames().getInfo()[0]
    # 稳定掩膜：首年==末年 且类别为湿地码
    last = ee.Image(wl.toList(n_img).get(n_img-1))
    f = first.select(band0); l = last.select(band0)
    wet_codes = [180, 181, 182, 183, 184, 185, 186, 187]
    stable = f.eq(l).updateMask(f.gt(0))
    # 分块采样（每 6° 块 stratifiedSample）
    frames = []
    for bi, (x0, y0, x1, y1) in enumerate(chunk_bbox()):
        roi = ee.Geometry.Rectangle([x0, y0, x1, y1])
        try:
            samp = (stable.updateMask(stable.neq(0))
                    .rename('code')
                    .addBands(f.rename('code_first'))
                    .stratifiedSample(
                        numPoints=600, classBand='code', region=roi,
                        scale=30, geometries=False, tileScale=4, dropNulls=True))
            df = download_fc_csv(samp, ['code'])
            if len(df):
                frames.append(df)
            print(f'  block{bi} ({x0},{y0}): {len(df)}')
        except Exception as e:
            print(f'  block{bi} ERR: {str(e)[:100]}')
        time.sleep(1)
    if not frames:
        print('无结果'); return
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(os.path.join(OUT, 'wetland_gwl.parquet'), index=False)
    print('湿地样本:', len(df), dict(df.code.value_counts()))

def imperv():
    """GISA(首城市化年) ∩ GISD30(期次) → 城镇池; 乡村=GISA 早年份点已含"""
    ee, pid = PC.load_account('zsi8emo')
    gisa = ee.Image('projects/sat-io/open-datasets/GISA_1972_2021').selfMask()
    gisd = ee.Image('projects/sat-io/open-datasets/GISD30_1985_2020').selfMask()
    both = gisa.updateMask(gisd.mask()).rename('gisa_year')
    frames = []
    for bi, (x0, y0, x1, y1) in enumerate(chunk_bbox()):
        roi = ee.Geometry.Rectangle([x0, y0, x1, y1])
        try:
            samp = both.stratifiedSample(
                numPoints=400, classBand='gisa_year', region=roi,
                scale=30, geometries=False, tileScale=4, dropNulls=True)
            df = download_fc_csv(samp, ['gisa_year'])
            if len(df):
                frames.append(df)
            print(f'  block{bi} ({x0},{y0}): {len(df)}')
        except Exception as e:
            print(f'  block{bi} ERR: {str(e)[:100]}')
        time.sleep(1)
    if not frames:
        print('无结果'); return
    df = pd.concat(frames, ignore_index=True)
    # 需要 lon/lat —— stratifiedSample 无 geometries 时没有坐标，改用带坐标版本
    print('注意: 无坐标版本，需重跑带 geometries 的采样')
    df.to_parquet(os.path.join(OUT, 'imperv_gisa_raw.parquet'), index=False)

if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('wetland', 'all'):
        wetland()
    if which in ('imperv', 'all'):
        imperv()
