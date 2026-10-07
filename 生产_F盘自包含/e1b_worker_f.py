# -*- coding: utf-8 -*-
"""
e1b_worker_f.py — r7 年度嵌入提取工作器（F 盘自包含版，2026-09-09 磁盘故障迁移）
* 与 Z 盘零依赖：账号凭证(F:\地理所\...\gee_accounts)、索引、输出全在 F 盘
* 用法: F:\\anaconda\\python.exe F:\\r7_prod\\e1b_worker_f.py <账号名>
  （12 账号名单见 FLEET；断点续跑：F:\\r7_prod\\emb_parts_r7 已有块自动跳过）
"""
import sys, os, io, time, glob, random
import numpy as np
import pandas as pd
import requests

OUT_DIR = r'Z:\地理所\论文\中国土地覆盖数据_2017-2024\数据\本地处理\全国清洗训练\emb_parts_r7'
IDX = r'F:\r7_prod\chunks_index_r7.parquet'
GEE_ACC = r'F:\地理所\论文\博士论文\中期\实验区域\02_中间数据\gee_accounts'
PROXY = {'http': 'socks5h://127.0.0.1:7890', 'https': 'socks5h://127.0.0.1:7890'}
FEATS = [f'A{i:02d}' for i in range(64)]
os.makedirs(OUT_DIR, exist_ok=True)

FLEET = {
    'zsi8emo':         ('inductive-seat-507709-s5', dict(prop='row_id',    ts=4, rev=False, sleep=(2, 4), nretry=4)),
    's4ezbd':          ('swift-yew-507709-e7',      dict(prop='rid',       ts=2, rev=False, sleep=(1, 3), nretry=3)),
    'w2qe4hiu':        ('pacific-torus-507710-v5',  dict(prop='ptid',      ts=4, rev=False, sleep=(2, 5), nretry=4)),
    'kitmyfaceplease2':('swift-library-507508-n0',  dict(prop='sample_id', ts=3, rev=True,  sleep=(1, 4), nretry=3)),
    'berk95733':       ('vocal-byte-507310-e0',     dict(prop='fid_r7',    ts=2, rev=False, sleep=(2, 4), nretry=4)),
    'chengruiduobao':  ('pelagic-plexus-507507-t6', dict(prop='pt_idx',    ts=4, rev=True,  sleep=(1, 3), nretry=3)),
    'ughwvm7968.med':  ('fast-flight-507509-m9',    dict(prop='sid',       ts=2, rev=False, sleep=(2, 5), nretry=4)),
    'zhnagningdan1':   ('involuted-fold-507511-k7', dict(prop='obs_id',    ts=3, rev=True,  sleep=(1, 4), nretry=3)),
    '5vqm9g':          ('pivotal-biplane-507607-p8',dict(prop='key_id',    ts=4, rev=False, sleep=(2, 4), nretry=4)),
    'save456jr':       ('ultra-water-507608-q5',    dict(prop='rownum',    ts=2, rev=True,  sleep=(1, 3), nretry=3)),
    'e0p36771':        ('lucky-rookery-507608-v1',  dict(prop='sp_id',     ts=3, rev=False, sleep=(2, 5), nretry=4)),
    '7lus57it':        ('rational-photon-507609-q9',dict(prop='cell_id',   ts=4, rev=True,  sleep=(1, 4), nretry=3)),
}
NACC = len(FLEET)

def load_account(acct, pid):
    """独立实现（pilot_common 在 Z 盘不可用）：HOME 指向凭证目录 + ee.Initialize"""
    d = os.path.join(GEE_ACC, acct)
    os.environ['HOME'] = d
    os.environ['USERPROFILE'] = d
    os.environ.setdefault('HTTP_PROXY', 'socks5h://127.0.0.1:7890')
    os.environ.setdefault('HTTPS_PROXY', 'socks5h://127.0.0.1:7890')
    import ee
    for a in range(3):
        try:
            ee.Initialize(project=pid)
            break
        except Exception as e:
            if a == 2:
                raise
            print(f'  ee.Initialize 重试({a+1}): {str(e)[:80]}', flush=True)
            time.sleep(20)
    print(f'[ee] {acct} @ {pid} OK', flush=True)
    return ee

def main():
    acct = sys.argv[1]
    pid, P = FLEET[acct]
    ai = list(FLEET).index(acct)
    ee = load_account(acct, pid)
    idx_df = pd.read_parquet(IDX)
    all_cids = sorted(idx_df.chunk_id.unique().tolist())
    my_cids = [c for c in all_cids if c % NACC == ai]
    done = {int(os.path.basename(f).split('_')[1].split('.')[0])
            for f in glob.glob(os.path.join(OUT_DIR, 'chunkr7_*.parquet'))}
    todo = [c for c in my_cids if c not in done]
    if P['rev']:
        todo = todo[::-1]
    print(f'[{acct}] 指纹 prop={P["prop"]} ts={P["ts"]} rev={P["rev"]} '
          f'| 分配 {len(my_cids)} 块, 已完成 {len(my_cids)-len(todo)}, 待跑 {len(todo)}', flush=True)
    col_cache = {}
    ok_n = fail_n = 0
    t0 = time.time()
    for cid in todo:
        fp = os.path.join(OUT_DIR, f'chunkr7_{cid:04d}.parquet')
        g = idx_df[idx_df.chunk_id == cid]
        rows = g[['row_id', 'lon', 'lat']].to_dict('records')
        year = int(g.year.iloc[0])
        done_f = False
        for attempt in range(3):
            try:
                fc = ee.FeatureCollection([
                    ee.Feature(ee.Geometry.Point([float(r['lon']), float(r['lat'])]),
                               {P['prop']: int(r['row_id'])})
                    for r in rows])
                if year not in col_cache:
                    col_cache[year] = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
                                       .filterDate(f'{year}-01-01', f'{year+1}-01-01'))
                ic = col_cache[year].filterBounds(fc)
                emb = ic.mosaic().select(FEATS)
                samp = emb.sampleRegions(collection=fc, properties=[P['prop']],
                                         scale=10, tileScale=P['ts'])
                url = samp.getDownloadURL(filetype='csv', selectors=[P['prop']] + FEATS)
                out = None
                for dtry in range(P['nretry']):
                    try:
                        resp = requests.get(url, proxies=PROXY, timeout=1200)
                        resp.raise_for_status()
                        out = pd.read_csv(io.BytesIO(resp.content))
                        break
                    except Exception as de:
                        print(f'[{acct}] chunk{cid} dl-retry{dtry+1} ERR: '
                              f'{str(de)[:90]}', flush=True)
                        time.sleep(20 + 10 * dtry)
                if out is None:
                    raise RuntimeError('download retries exhausted')
                out = out.rename(columns={P['prop']: 'row_id'})
                out['emb_year'] = year
                out['chunk_id'] = cid
                out.to_parquet(fp, index=False)
                ok_n += 1
                print(f'[{acct}] chunk{cid:04d} y{year}: {len(out)} 行 '
                      f'({ok_n}/{len(todo)}, {(time.time()-t0)/3600:.1f}h)', flush=True)
                done_f = True
                break
            except Exception as e:
                print(f'[{acct}] chunk{cid} attempt{attempt+1} ERR: {str(e)[:110]}', flush=True)
                time.sleep(20 + attempt * 30)
        if not done_f:
            fail_n += 1
            print(f'[{acct}] chunk{cid} FINAL-FAIL', flush=True)
        time.sleep(random.uniform(*P['sleep']))
    print(f'[{acct}] 全部完成: 成功 {ok_n}, 失败 {fail_n}, '
          f'用时 {(time.time()-t0)/3600:.1f}h', flush=True)

if __name__ == '__main__':
    main()
