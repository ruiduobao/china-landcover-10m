# -*- coding: utf-8 -*-
"""f4_candidates.py — ForTy 2020 单次扫描：正候选（类5纯块）+ 负候选块（森林类/其他土地），250 m 块级

* 输入：D:/lcwork_forty/ForTy_china/<tile>/*.tif（f2 工作副本，中国相关文件）
        F:/lc_work/forty_exp/map/forty_file_index.csv
* 规则源（**预注册，先写死再跑**）：
  A. 250 m 块（25×25 像元）：统计每块"有效像元数、类5数、各类数"；
     - 正候选块：c5_frac ≥ 0.50 且 valid_ratio ≥ 0.90（地块级纯园，沿用"园地=地块格局类"判据）；
     - 难负块（hard）：块众数类 ∈ {1原始林, 2天然更新林, 3人工林} 且 该类推数/有效 ≥ 0.70；
     - 背景负块（bg）：块众数类 = 6 其他土地 且 纯度 ≥ 0.80，**按坐标哈希 2% 抽样**（控制体量）。
  B. 每块记录：中心经纬度、c5_frac、dom 类、dom 纯度、kind。
  说明：本脚本只**产出候选块**；去重/隔离/配额/省归属在 f5 完成（f5 不再读栅格）。
* 门槛：中国文件扫描 100% 成功；正候选块数 > 全局目标 1.5 倍（不足则降级并记录）
* 输出：F:/lc_work/forty_exp/cand/cand_blocks.parquet（lon,lat,frac5,dom,pur,kind,valid_ratio）
        F:/lc_work/forty_exp/cand/scan_summary.json
* 用法：python f4_candidates.py [--workers 6]
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
from rasterio.warp import transform as rtransform

WORK = r'F:/lc_work/forty_exp'
MAP = os.path.join(WORK, 'map')
DCOPY = r'D:/lcwork_forty/ForTy_china'
OUTD = os.path.join(WORK, 'cand')
BLK = 25
BG_P = 0.02
SEED = 20261026


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def hash_keep(lon, lat, p):
    """确定性哈希抽样（坐标→[0,1)）"""
    x = np.floor(lon * 1e5).astype('int64')
    y = np.floor(lat * 1e5).astype('int64')
    h = (x * 73856093) ^ (y * 19349663)
    h = (h ^ (h >> 13)) * 1274126177
    u = ((h >> 8) & 0xFFFFFF).astype('float64') / float(0xFFFFFF)
    return u < p


def scan_one(rel):
    fp = os.path.join(DCOPY, rel.replace('/', os.sep))
    try:
        with rasterio.open(fp) as s:
            a = s.read(1)
            t = s.transform
            crs = s.crs
    except Exception as e:
        return ([], str(e)[:80])
    h, w = a.shape
    H, W = (h + BLK - 1) // BLK, (w + BLK - 1) // BLK
    pad = np.full((H * BLK, W * BLK), 255, dtype=np.uint8)
    pad[:h, :w] = a
    del a
    blk = pad.reshape(H, BLK, W, BLK)
    valid = (blk != 255)
    nv = valid.sum(axis=(1, 3))
    ratio = nv / (BLK * BLK)
    # 各类计数 (1..6)
    counts = np.zeros((6, H, W), dtype='int16')
    for cls in range(1, 7):
        counts[cls - 1] = (blk == cls).sum(axis=(1, 3))
    del blk, valid
    safe = np.maximum(nv, 1)
    frac = counts / safe[None, :, :]
    dom = frac.argmax(axis=0) + 1
    pur = frac.max(axis=0)
    c5 = frac[4]
    okv = (nv > 0)
    # 三类候选
    m_pos = (c5 >= 0.50) & (ratio >= 0.90) & okv
    m_hard = (np.isin(dom, [1, 2, 3])) & (pur >= 0.70) & okv
    m_bg = (dom == 6) & (pur >= 0.80) & okv
    out = []
    for m, kind in ((m_pos, 'pos'), (m_hard, 'hard'), (m_bg, 'bg')):
        br, bc = np.where(m)
        if len(br) == 0:
            continue
        xs = t.c + (bc * BLK + BLK / 2.0 + 0.5) * t.a
        ys = t.f + (br * BLK + BLK / 2.0 + 0.5) * t.e
        lo, la = rtransform(crs, 'EPSG:4326', xs.tolist(), ys.tolist())
        lo, la = np.asarray(lo), np.asarray(la)
        if kind == 'bg':
            keep = hash_keep(lo, la, BG_P)
            lo, la = lo[keep], la[keep]
            br, bc = br[keep], bc[keep]
        if len(lo) == 0:
            continue
        kcode = {'pos': 0, 'hard': 1, 'bg': 2}[kind]
        out.append(np.column_stack([lo, la, c5[br, bc], dom[br, bc], pur[br, bc],
                                    np.full(len(lo), kcode), ratio[br, bc]]))
    if not out:
        return ([], None)
    arr = np.vstack(out)
    kindmap = np.array(['pos', 'hard', 'bg'], dtype=object)
    # 注意：arr[:,5] 已由下方 scan_one 写入 kind 编码（0=pos,1=hard,2=bg）
    df = pd.DataFrame({'lon': arr[:, 0].astype('float32'), 'lat': arr[:, 1].astype('float32'),
                       'frac5': arr[:, 2].astype('float32'), 'dom': arr[:, 3].astype('int8'),
                       'pur': arr[:, 4].astype('float32'),
                       'kind': kindmap[arr[:, 5].astype('int8')],
                       'valid_ratio': arr[:, 6].astype('float32')})
    return (df, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    os.makedirs(OUTD, exist_ok=True)
    idx = pd.read_csv(os.path.join(MAP, 'forty_file_index.csv'), encoding='utf-8-sig')
    idx = idx[idx.in_china == True].reset_index(drop=True)
    emit('中国文件 %d 个，开始扫描（%d 进程，250 m 块）' % (len(idx), a.workers))
    t0, done, errs, n_out = time.time(), 0, [], 0
    writer = None
    for rel, (df, err) in zip(idx.relpath.tolist(),
                              _imap(scan_one, idx.relpath.tolist(), a.workers)):
        done += 1
        if err:
            errs.append((rel, err))
        if df is not None and len(df):
            tbl = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(os.path.join(OUTD, 'cand_blocks.parquet'), tbl.schema)
            writer.write_table(tbl)
            n_out += len(df)
        if done % 500 == 0:
            emit('  扫描 %d/%d，块 %d（%.1f 分钟）' % (done, len(idx), n_out, (time.time() - t0) / 60))
    if writer is not None:
        writer.close()
    emit('扫描完成：%d 文件（失败 %d），候选块 %d（%.1f 分钟）' % (
        len(idx), len(errs), n_out, (time.time() - t0) / 60))
    d = pd.read_parquet(os.path.join(OUTD, 'cand_blocks.parquet'))
    summ = {'n_files': int(len(idx)), 'n_err': len(errs),
            'err_sample': errs[:5], 'n_blocks': int(len(d)),
            'by_kind': d.kind.value_counts().to_dict(),
            'pos_frac5_med': round(float(d.loc[d.kind == 'pos', 'frac5'].median()), 3)
            if (d.kind == 'pos').any() else None,
            'pos_area_km2_est': round(float((d.kind == 'pos').sum() * 0.0625), 1)}
    json.dump(summ, open(os.path.join(OUTD, 'scan_summary.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    emit(json.dumps(summ['by_kind'], ensure_ascii=False))
    emit('正候选块面积估计 ≈ %.0f km²' % summ['pos_area_km2_est'])


def _imap(fn, items, workers):
    """按文件流式产出，边算边写（避免一次性攒 135k 结果）"""
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(fn, items, chunksize=4):
            yield r


if __name__ == '__main__':
    main()
