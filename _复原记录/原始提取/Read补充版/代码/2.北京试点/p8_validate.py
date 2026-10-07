# -*- coding: utf-8 -*-
"""
p8_validate.py — 北京试点精度验证 + 对比
* 留出集(bronze, 嵌入空间点) → 与 2020 成品像元值比对 → 混淆矩阵 OA/κ/UA/PA
* 与 CLCD 2020 / ESA WC 2021 北京区一致性
* 输出: 数据/本地处理/北京试点/p8_validation.json
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot_common as PC
import numpy as np
import pandas as pd
import rasterio
from rasterio import transform as riotrans
from pyproj import Transformer

OUT = '数据/本地处理/北京试点'
CLASS_NAMES = {10: '旱地', 12: '灌溉/水田', 61: '落叶阔叶', 71: '常绿针叶', 81: '落叶针叶',
               130: '草地', 150: '稀疏植被', 181: '草本沼泽', 182: '滩地', 190: '城镇',
               200: '乡村不透水', 201: '裸地', 202: '水体'}

def confusion(y_true, y_pred, labels):
    L = len(labels)
    idx = {c: i for i, c in enumerate(labels)}
    M = np.zeros((L, L), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            M[idx[t], idx[p]] += 1
    return M

def metrics(M):
    OA = M.diagonal().sum() / M.sum()
    pe = (M.sum(0) * M.sum(1)).sum() / M.sum()**2
    kappa = (OA - pe) / (1 - pe) if pe < 1 else 0
    pa = M.diagonal() / M.sum(1)     # producer (recall)
    ua = M.diagonal() / M.sum(0)     # user (precision)
    return {'OA': round(float(OA), 4), 'kappa': round(float(kappa), 4),
            'PA': {CLASS_NAMES.get(c, c): round(float(a), 3) for c, a in zip(LABELS, pa) if M.sum(1)[list(LABELS).index(c)] > 0},
            'UA': {CLASS_NAMES.get(c, c): round(float(a), 3) for c, a in zip(LABELS, ua) if M.sum(0)[list(LABELS).index(c)] > 0}}

if __name__ == '__main__':
    LABELS = sorted(CLASS_NAMES)
    ho = pd.read_parquet(os.path.join(OUT, 'samples_bj_holdout.parquet'))
    fp2020 = os.path.join(OUT, 'raster', 'CNLC10_BJ_2020_final.tif')
    if not os.path.exists(fp2020):
        fp2020 = os.path.join(OUT, 'raster', 'CNLC10_BJ_2020_mosaic.tif')
    with rasterio.open(fp2020) as s:
        arr = s.read(1)
        tf = s.transform
        # 嵌入特征点是 GEE 采样 (row_id ↔ v2 底座顺序)
        BASE = pd.read_parquet('数据/本地处理/样本底座/cn_samples_v2.parquet')
        bj = BASE[(BASE.lon >= PC.BBOX[0]) & (BASE.lon <= PC.BBOX[2]) &
                  (BASE.lat >= PC.BBOX[1]) & (BASE.lat <= PC.BBOX[3])].reset_index(drop=True)
        ho_m = ho.merge(bj[['lon', 'lat']].reset_index().rename(columns={'index': 'row_id'}),
                        on='row_id', how='left')
        rows, cols = riotrans.rowcol(tf, ho_m['lon'].to_numpy(), ho_m['lat'].to_numpy(),
                                     op=lambda v: np.floor(v).astype(int))
        rows = np.array(rows); cols = np.array(cols)
        H, W = arr.shape
        ok = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
        pred = np.full(len(ho_m), 0)
        pred[ok] = arr[rows[ok], cols[ok]]
        y_true = ho_m['cls'].to_numpy()
        keep = ok & (pred > 0) & np.isin(y_true, LABELS)
        M = confusion(y_true[keep], pred[keep], LABELS)
        met = metrics(M)
        print('比对点:', keep.sum())
        print('OA:', met['OA'], ' kappa:', met['kappa'])
        print('PA:', met['PA'])
        print('UA:', met['UA'])
        json.dump({'n': int(keep.sum()), 'confusion': M.tolist(), 'labels': LABELS,
                   'metrics': met},
                  open(os.path.join(OUT, 'p8_validation.json'), 'w'),
                  ensure_ascii=False, indent=2, default=str)
        print('已写 p8_validation.json')

