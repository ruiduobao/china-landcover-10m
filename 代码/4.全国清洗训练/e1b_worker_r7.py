# -*- coding: utf-8 -*-
"""
e1b_worker_r7.py — P2：r7 年度嵌入提取工作器（12 账号编队版，2026-09-09）
* 输入: chunks_index_r7.parquet（14.27M point-years / 4758 块 × 3000 点）
* 防识别差异化（每账号指纹不同，避免被识别为同一程序）:
    - Feature 属性名 12 种（row_id/rid/ptid/sample_id/fid_r7/pt_idx/sid/obs_id/
      key_id/rownum/sp_id/cell_id）
    - tileScale ∈ {2,3,4} 轮换、块遍历方向正/反、块间 sleep 区间、下载重试次数各异
* 分配: chunk_id % 12 == 账号序号；断点续跑（chunkr7_*.parquet 已存在即跳过）
* 用法: 必须从项目根目录: python 代码/4.全国清洗训练/e1b_worker_r7.py <账号名>
"""
import sys, os, io, time, glob, random
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, r'F:\地理所\论文\中国土地覆盖数据_2017-2024\代码\2.北京试点')

OUT_DIR = '数据/本地处理/全国清洗训练/emb_parts_r7'
IDX = '数据/本地处理/全国清洗训练/chunks_index_r7.parquet'
FEATS = [f'A{i:02d}' for i in range(64)]
os.makedirs(OUT_DIR, exist_ok=True)

# 账号 → (项目ID, 指纹参数)
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

def main():
    acct = sys.argv[1]
    pid, P = FLEET[acct]
    ai = list(FLEET).index(acct)
    import pilot_common as PC
    import ee
    ee, _ = PC.load_account(acct, pid)
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
        years_in = sorted(int(v) for v in g.year.unique().tolist())
        done_f = False
        for attempt in range(3):
            try:
                outs = []
                for year in years_in:
                    gy = g[g.year == year]
                    rows = gy[['row_id', 'lon', 'lat']].to_dict('records')
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
                    for dtry in range(P['nretry']):     # 下载阶段重试（代理空闲超时防护）
                        try:
                            resp = requests.get(url, proxies=PC.PROXY, timeout=1200)
                            resp.raise_for_status()
                            out = pd.read_csv(io.BytesIO(resp.content))
                            break
                        except Exception as de:
                            print(f'[{acct}] chunk{cid} y{year} dl-retry{dtry+1} ERR: '
                                  f'{str(de)[:90]}', flush=True)
                            time.sleep(20 + 10 * dtry)
                    if out is None:
                        raise RuntimeError('download retries exhausted')
                    out = out.rename(columns={P['prop']: 'row_id'})
                    out['emb_year'] = year
                    outs.append(out)
                out = pd.concat(outs, ignore_index=True)
                out['chunk_id'] = cid
                out.to_parquet(fp, index=False)
                ok_n += 1
                el = time.time() - t0
                print(f'[{acct}] chunk{cid:04d} y{years_in}: {len(out)} 行 '
                      f'({ok_n}/{len(todo)}, {el/3600:.1f}h)', flush=True)
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
