# -*- coding: utf-8 -*-
"""
e1_extract_embeddings.py — 全国样本嵌入提取（v4 除 reject 外 822 万点，年匹配）
* 块索引: 数据/本地处理/全国清洗训练/chunks_index.parquet（709 块 × ~11.6k 点）
* 每 1° 格窗口 mosaic + sampleRegions(scale=10, tileScale=4) → CSV → 分片 parquet
* 多进程: 每子进程绑定一个账号（Windows 下 ee 全局态进程隔离）
* 断点续跑: 已存在的 chunk_*.parquet 跳过
* 用法: python e1_extract_embeddings.py [每账号进程数, 默认1]
"""
import sys, os, io, time, glob, queue, threading
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r'F:\地理所\论文\中国土地覆盖数据_2017-2024\代码\2.北京试点')

OUT_DIR = '数据/本地处理/全国清洗训练/emb_parts'
IDX = '数据/本地处理/全国清洗训练/chunks_index.parquet'
ACCOUNTS = ['zsi8emo', 's4ezbd', 'w2qe4hiu']
FEATS = [f'A{i:02d}' for i in range(64)]
os.makedirs(OUT_DIR, exist_ok=True)

def load_chunk(idx_df, cid):
    g = idx_df[idx_df.chunk_id == cid]
    return g[['row_id', 'lon', 'lat', 'class_new']].to_dict('records'), int(g.year.iloc[0])

def worker_proc(acct, q):
    """单账号子进程：绑定账号 → 逐块提取"""
    sys.path.insert(0, r'F:\地理所\论文\中国土地覆盖数据_2017-2024\代码\2.北京试点')
    import pilot_common as PC
    import ee
    ee, pid = PC.load_account(acct)
    idx_df = pd.read_parquet(IDX)
    col_cache = {}
    ok_n = 0
    while True:
        try:
            cid = q.get_nowait()
        except queue.Empty:
            break
        fp = os.path.join(OUT_DIR, f'chunk_{cid:04d}.parquet')
        if os.path.exists(fp):
            continue
        rows, year = load_chunk(idx_df, cid)
        done = False
        for attempt in range(3):
            try:
                fc = ee.FeatureCollection([
                    ee.Feature(ee.Geometry.Point([float(r['lon']), float(r['lat'])]),
                               {'row_id': int(r['row_id'])})
                    for r in rows])
                if year not in col_cache:
                    col_cache[year] = (ee.ImageCollection(PC.EMB_COL)
                                       .filterDate(f'{year}-01-01', f'{year+1}-01-01'))
                ic = col_cache[year].filterBounds(fc)
                emb = ic.mosaic().select(FEATS)
                samp = emb.sampleRegions(collection=fc, properties=['row_id'],
                                         scale=10, tileScale=4)
                url = samp.getDownloadURL(filetype='csv', selectors=['row_id'] + FEATS)
                resp = requests.get(url, proxies=PC.PROXY, timeout=1800)
                resp.raise_for_status()
                out = pd.read_csv(io.BytesIO(resp.content))
                out['emb_year'] = year
                out['chunk_id'] = cid
                out.to_parquet(fp, index=False)
                ok_n += 1
                print(f'[{acct}] chunk{cid:04d} y{year}: {len(out)} 行 '
                      f'({time.strftime("%H:%M:%S")})', flush=True)
                done = True
                break
            except Exception as e:
                print(f'[{acct}] chunk{cid} attempt{attempt+1} ERR: {str(e)[:110]}', flush=True)
                time.sleep(20 + attempt * 30)
        if not done:
            print(f'[{acct}] chunk{cid} FINAL-FAIL', flush=True)
    print(f'[{acct}] 进程结束, 成功 {ok_n} 块', flush=True)

def main():
    per_acct_procs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    import pyarrow.parquet as pq
    idx_df = pq.read_table(IDX, columns=['chunk_id']).to_pandas()
    all_cids = sorted(idx_df.chunk_id.unique().tolist())
    done = {int(os.path.basename(f).split('_')[1].split('.')[0])
            for f in glob.glob(os.path.join(OUT_DIR, 'chunk_*.parquet'))}
    todo = [c for c in all_cids if c not in done]
    print(f'总块 {len(all_cids)}, 已完成 {len(done)}, 待跑 {len(todo)}', flush=True)
    from multiprocessing import Process
    qs = {a: queue.Queue() for a in ACCOUNTS}
    for w, cid in enumerate(todo):
        qs[ACCOUNTS[w % len(ACCOUNTS)]].put(cid)
    procs = []
    for acct in ACCOUNTS:
        if qs[acct].empty():
            continue
        for k in range(per_acct_procs):
            pr = Process(target=worker_proc, args=(acct, qs[acct]))
            pr.start(); procs.append(pr)
            time.sleep(2)
    for pr in procs:
        pr.join()
    remain = len(all_cids) - len(glob.glob(os.path.join(OUT_DIR, 'chunk_*.parquet')))
    print(f'\n提取结束: 剩余未完成 {remain} 块', flush=True)

if __name__ == '__main__':
    main()
