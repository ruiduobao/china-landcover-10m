# -*- coding: utf-8 -*-
"""
p2_sample_embeddings.py — 北京样本点 2020 年 64 维嵌入提取（交互式 getDownloadURL）
* gold+silver → 训练集（8.5k 点）；bronze → 留出验证集（10.9k 点）
* 输出: 数据/本地处理/北京试点/samples_bj_train.parquet / samples_bj_holdout.parquet
"""
import sys, os, io, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot_common as PC
import numpy as np
import pandas as pd
import requests

ee, pid = PC.load_account('zsi8emo')

BASE = pd.read_parquet('数据/本地处理/样本底座/cn_samples_v2.parquet')
lo, la0, hi, la1 = PC.BBOX
bj = BASE[(BASE.lon >= lo) & (BASE.lon <= hi) & (BASE.lat >= la0) & (BASE.lat <= la1)].copy()
bj['row_id'] = np.arange(len(bj))
print('北京样本:', len(bj), dict(bj.tier.value_counts()))

def extract(year, df, tag):
    fc = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]),
                   {'row_id': int(r.row_id), 'cls': int(r.class_new),
                    'tier': r.tier})
        for r in df.itertuples()])
    emb = (ee.ImageCollection(PC.EMB_COL)
           .filterDate(f'{year}-01-01', f'{year+1}-01-01')
           .filterBounds(fc))
    print(f'  {tag}: 年度集合景数 =', emb.size().getInfo())
    emb = emb.mosaic()
    samp = emb.sampleRegions(collection=fc, properties=['row_id', 'cls', 'tier'],
                             scale=10, tileScale=2)
    url = samp.getDownloadURL(filetype='csv', selectors=['row_id', 'cls', 'tier'] + PC.FEATS_ALL)
    print(f'  {tag}: 请求下载 ({time.strftime("%H:%M:%S")})')
    r = requests.get(url, proxies=PC.PROXY, timeout=600)
    r.raise_for_status()
    out = pd.read_csv(io.BytesIO(r.content))
    print(f'  {tag}: {len(out)} 行, {out.shape[1]} 列')
    return out

tr = extract(2020, bj[bj.tier.isin(['gold', 'silver'])], 'train(gold+silver)')
ho = extract(2020, bj[bj.tier == 'bronze'], 'holdout(bronze)')

os.makedirs('数据/本地处理/北京试点', exist_ok=True)
tr.to_parquet('数据/本地处理/北京试点/samples_bj_train.parquet', index=False)
ho.to_parquet('数据/本地处理/北京试点/samples_bj_holdout.parquet', index=False)
print('训练集:', tr.shape, ' 留出集:', ho.shape)
print('训练集类别分布:', dict(tr.cls.value_counts().sort_index()))

