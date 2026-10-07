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

def extract(year, df, tag):
    fc = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]),
                   {'row_id': int(r.row_id), 'cls': int(r.cls),
                    'tier': str(r.tier), 'lon': float(r.lon), 'lat': float(r.lat)})
        for r in df.itertuples()])
    emb = (ee.ImageCollection(PC.EMB_COL)
           .filterDate(f'{year}-01-01', f'{year+1}-01-01')
           .filterBounds(fc))
    print(f'  {tag}: 年度集合景数 =', emb.size().getInfo())
    emb = emb.mosaic()
    samp = emb.sampleRegions(collection=fc, properties=['row_id', 'cls', 'tier', 'lon', 'lat'],
                             scale=10, tileScale=2)
    url = samp.getDownloadURL(filetype='csv',
                              selectors=['row_id', 'cls', 'tier', 'lon', 'lat'] + PC.FEATS_ALL)
    print(f'  {tag}: 请求下载 ({time.strftime("%H:%M:%S")})')
    r = requests.get(url, proxies=PC.PROXY, timeout=900)
    r.raise_for_status()
    out = pd.read_csv(io.BytesIO(r.content))
    print(f'  {tag}: {len(out)} 行, {out.shape[1]} 列')
    return out

USE_CFG = bool(PC.CFG.get('train_file')) and os.path.exists(PC.TRAIN_FILE)
os.makedirs(PC.OUT_DIR, exist_ok=True)

if USE_CFG:
    # ---- 生产配置路径：直接读配置样本文件（r1 等，含 lon/lat/class_new/tier） ----
    for srcf, tag in [(PC.TRAIN_FILE, 'train'), (PC.HOLDOUT_FILE, 'holdout')]:
        s = pd.read_parquet(srcf).reset_index(drop=True)
        s['row_id'] = np.arange(len(s))
        s['cls'] = s['class_new'].astype(int)
        out = extract(PC.ANCHOR_YEAR, s, f'{tag}({len(s)})')
        dst = os.path.join(PC.OUT_DIR, f'samples_bj_{tag}.parquet')
        out.to_parquet(dst, index=False)
        print(f'  → {dst}  类别覆盖: {out.cls.nunique()} 类')
        print('  类别分布:', dict(out.cls.value_counts().sort_index()))
    print(f'[{PC.AREA_NAME}] 嵌入提取完成 → {PC.OUT_DIR}')
else:
    # ---- 旧版路径（v2 底座 + tier 分割），行为不变 ----
    BASE = pd.read_parquet('数据/本地处理/样本底座/cn_samples_v2.parquet')
    lo, la0, hi, la1 = PC.BBOX
    bj = BASE[(BASE.lon >= lo) & (BASE.lon <= hi) & (BASE.lat >= la0) & (BASE.lat <= la1)].copy()
    bj['row_id'] = np.arange(len(bj))
    bj['cls'] = bj['class_new'].astype(int)
    print('北京样本:', len(bj), dict(bj.tier.value_counts()))
    tr = extract(PC.ANCHOR_YEAR, bj[bj.tier.isin(['gold', 'silver'])], 'train(gold+silver)')
    ho = extract(PC.ANCHOR_YEAR, bj[bj.tier == 'bronze'], 'holdout(bronze)')
    tr.to_parquet('数据/本地处理/北京试点/samples_bj_train.parquet', index=False)
    ho.to_parquet('数据/本地处理/北京试点/samples_bj_holdout.parquet', index=False)
    print('训练集:', tr.shape, ' 留出集:', ho.shape)
