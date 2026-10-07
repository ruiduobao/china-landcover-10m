# -*- coding: utf-8 -*-
"""
e1_worker.py — 单账号嵌入提取工作器（顺序处理分给该账号的块）
* 分配规则: chunk_id % 3 == 账号序号（静态确定）
* 用法: python e1_worker.py <zsi8emo|s4ezbd|w2qe4hiu>
* 断点续跑: 已存在的 chunk_*.parquet 跳过
"""
import sys, os, io, time, glob, queue
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r'F:\地理所\论文\中国土地覆盖数据_2017-2024\代码\2.北京试点')

OUT_DIR = '数据/本地处理/全国清洗训练/emb_parts'
IDX = '数据/本地处理/全国清洗训练/chunks_index.parquet'
ACCOUNTS = ['zsi8emo', 's4ezbd', 'w2qe4hiu']
FEATS = [f'A{i:02d}' for i in range(64)]

def main():
    acct = sys.argv[1]
    reverse = len(sys.argv) > 2 and sys.argv[2] == 'rev'
    ai = ACCOUNTS.index(acct)
    sys.path.insert(0, r'F:\地理所\论文\中国土地覆盖数据_2017-2024\代码\2.北京试点')
    import pilot_common as PC
    import ee
    ee, pid = PC.load_account(acct)
    idx_df = pd.read_parquet(IDX)
    all_cids = sorted(idx_df.chunk_id.unique().tolist())
    my_cids = [c for c in all_cids if c % 3 == ai]
    done = {int(os.path.basename(f).split('_')[1].split('.')[0])
            for f in glob.glob(os.path.join(OUT_DIR, 'chunk_*.parquet'))}
    todo = [c for c in my_cids if c not in done]
    if reverse:
        todo = todo[::-1]
    print(f'[{acct}] 分配 {len(my_cids)} 块, 已完成 {len(my_cids)-len(todo)}, 待跑 {len(todo)}', flush=True)
    col_cache = {}
    ok_n = fail_n = 0
    t0 = time.time()
    for cid in todo:
        fp = os.path.join(OUT_DIR, f'chunk_{cid:04d}.parquet')
        g = idx_df[idx_df.chunk_id == cid]
        rows = g[['row_id', 'lon', 'lat', 'class_new']].to_dict('records')
        year = int(g.year.iloc[0])
        done_f = False
        for attempt in range(3):
            try:
                fc = ee.FeatureCollection([
                    ee.Feature(ee.Geometry.Point([float(r['lon']), float(r['lat'])]),
                               {'row_id': int(r['row_id'])})
                    for r in rows])
                if year not in col_cache:
                    col_cache[year] = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
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
                el = time.time() - t0
                print(f'[{acct}] chunk{cid:04d} y{year}: {len(out)} 行 '
                      f'({ok_n}/{len(todo)}, {el/60:.0f}min)', flush=True)
                done_f = True
                break
            except Exception as e:
                print(f'[{acct}] chunk{cid} attempt{attempt+1} ERR: {str(e)[:110]}', flush=True)
                time.sleep(20 + attempt * 30)
        if not done_f:
            fail_n += 1
            print(f'[{acct}] chunk{cid} FINAL-FAIL', flush=True)
    print(f'[{acct}] 全部完成: 成功 {ok_n}, 失败 {fail_n}, '
          f'用时 {(time.time()-t0)/3600:.1f}h', flush=True)

if __name__ == '__main__':
    main()

