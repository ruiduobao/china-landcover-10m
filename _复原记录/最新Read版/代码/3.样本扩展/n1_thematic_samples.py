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
P = lambda *a: print(*a, flush=True)

OUT = '数据/外部样本源/gee_thematic'
os.makedirs(OUT, exist_ok=True)
BBOX = (73, 17, 136, 55)

def chunk_bbox(step=12):
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

def add_lonlat(df):
    """从 .geo 列(形如 {'type':'Point','coordinates':[lon,lat]}) 解析经纬度"""
    import json as _json
    def _p(s):
        try:
            c = _json.loads(s.replace("'", '"'))['coordinates']
            return pd.Series({'lon': c[0], 'lat': c[1]})
        except Exception:
            return pd.Series({'lon': np.nan, 'lat': np.nan})
    geo = df['.geo'].apply(_p)
    return pd.concat([df.drop(columns=['.geo']), geo], axis=1)

def wetland(years=(2000, 2022), target_per_class=20000):
    """GWL_FCS30 逐亚类时序稳定 → 采样"""
    ee, pid = PC.load_account('zsi8emo')
    wl = ee.ImageCollection('projects/sat-io/open-datasets/GWL_FCS30')
    n_img = wl.size().getInfo()
    P('GWL_FCS30 景数:', n_img)
    first = ee.Image(wl.toList(n_img).get(0))
    band0 = first.bandNames().getInfo()[0]
    last = ee.Image(wl.toList(n_img).get(n_img-1))
    f = first.select(band0); l = last.select(band0)
    wet_codes = [180, 181, 182, 183, 184, 185, 186, 187]
    stable = f.eq(l).updateMask(f.gt(0))
    # 分块采样（每 6° 块 stratifiedSample）
    frames = []
    for bi, (x0, y0, x1, y1) in enumerate(chunk_bbox()):
        roi = ee.Geometry.Rectangle([x0, y0, x1, y1])
        try:
            wl_r = wl.filterBounds(roi)
            ni = wl_r.size().getInfo()
            fi = ee.Image(wl_r.toList(max(ni, 1)).get(0))
            la = ee.Image(wl_r.toList(max(ni, 1)).get(ni - 1))
            b_f = fi.bandNames().getInfo()[0]
            b_l = la.bandNames().getInfo()[0]
            f_img = fi.select(b_f)
            l_img = la.select(b_l)
            img = (f_img.updateMask(f_img.eq(l_img))
                        .updateMask(f_img.gt(0))
                        .rename('code'))
            samp = (img.stratifiedSample(
                        numPoints=600, classBand='code', region=roi,
                        scale=30, geometries=True, tileScale=4, dropNulls=True))
            df = download_fc_csv(samp, ['code', '.geo'])
            if len(df):
                df = add_lonlat(df)
                frames.append(df)
            P(f'  block{bi} ({x0},{y0}): {len(df)}')
        except Exception as e:
            P(f'  block{bi} ERR: {str(e)[:100]}')
        time.sleep(1)
    if not frames:
        P('无结果'); return
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(os.path.join(OUT, 'wetland_gwl.parquet'), index=False)
    P('湿地样本:', len(df), dict(df.code.value_counts()))

def imperv():
    """GISA(首城市化年) ∩ GISD30(期次) → 城镇池; 乡村=GISA 早年份点已含"""
    ee, pid = PC.load_account('zsi8emo')
    gisa_img = ee.Image('projects/sat-io/open-datasets/GISA_1972_2021').selfMask()
    gisd_img = ee.Image('projects/sat-io/open-datasets/GISD30_1985_2020').selfMask()
    frames = []
    for bi, (x0, y0, x1, y1) in enumerate(chunk_bbox()):
        roi = ee.Geometry.Rectangle([x0, y0, x1, y1])
        try:
            both = gisa_img.updateMask(gisd_img.mask()).rename('gisa_year')
            samp = both.stratifiedSample(
                numPoints=400, classBand='gisa_year', region=roi,
                scale=30, geometries=True, tileScale=4, dropNulls=True)
            df = download_fc_csv(samp, ['gisa_year', '.geo'])
            if len(df):
                df = add_lonlat(df)
                frames.append(df)
            P(f'  block{bi} ({x0},{y0}): {len(df)}')
        except Exception as e:
            P(f'  block{bi} ERR: {str(e)[:100]}')
        time.sleep(1)
    if not frames:
        P('无结果'); return
    df = pd.concat(frames, ignore_index=True)
    # 需要 lon/lat —— stratifiedSample 无 geometries 时没有坐标，改用带坐标版本
    df.to_parquet(os.path.join(OUT, 'imperv_gisa_raw.parquet'), index=False)

def glc12_orchard(year=2020, n_per_block=150):
    """GLC_FCS30D 单年 class 12(乔灌园地) 采样——不做稳定掩膜（园地被稳定掩膜洗掉的教训）"""
    ee, pid = PC.load_account('zsi8emo')
    glc = ee.ImageCollection('projects/sat-io/open-datasets/GLC-FCS30D/annual')
    band = f'b{year-2000+1}'
    frames = []
    for bi, (x0, y0, x1, y1) in enumerate(chunk_bbox(step=12)):
        roi = ee.Geometry.Rectangle([x0, y0, x1, y1])
        try:
            img = glc.filterBounds(roi).mosaic().select(band)
            orchard = img.updateMask(img.eq(12)).rename('code')
            samp = orchard.stratifiedSample(
                numPoints=n_per_block, classBand='code', region=roi,
                scale=30, geometries=True, tileScale=4, dropNulls=True)
            df = download_fc_csv(samp, ['code', '.geo'])
            if len(df):
                df = add_lonlat(df)
                df['src'] = 'glc12_2020'
                frames.append(df)
            P(f'  glc12 block{bi} ({x0},{y0}): {len(df)}')
        except Exception as e:
            P(f'  glc12 block{bi} ERR: {str(e)[:100]}')
        time.sleep(1)
    if frames:
        df = pd.concat(frames, ignore_index=True)
        df.to_parquet(os.path.join(OUT, 'orchard_glc12.parquet'), index=False)
        print('园地候选(glc12):', len(df))

if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('wetland', 'all'):
        wetland()
    if which in ('imperv', 'all'):
        imperv()
    if which in ('glc12', 'all'):
        glc12_orchard()

