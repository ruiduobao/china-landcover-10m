# -*- coding: utf-8 -*-
"""679i9zo111222 专属嵌入提取器 · 下一代补样 · 生态区候选点构建（多源一致性）
由 ng_gen.py 生成（模板 ng_emb_tmpl.tpl），勿手改。
读 ng6_build_samples.py 产出的年度索引，为下一代样本点提取 AlphaEarth 64 维嵌入。
用法: python nge_679i9zo111222.py [--limit N] [--dry]
"""
import os
import sys
import io
import json
import time
import random
import argparse

import numpy as np
import pandas as pd
import requests

oak_net = {'http': 'socks5h://127.0.0.1:7890',
             'https': 'socks5h://127.0.0.1:7890'}
oak_cred_home = 'F:/地理所/论文/博士论文/中期/实验区域/02_中间数据/gee_accounts'
oak_who = '679i9zo111222'
oak_home_proj = 'silicon-webbing-508500-t4'
oak_tile = 4
oak_rest = (1, 6)

_oak_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_oak_dir))
import ng0_paths as oak_paths

oak_bands = [f'A{i:02d}' for i in range(64)]
oak_embcol = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
oak_nacc = 15


def oak_emit(msg):
    print(time.strftime('%H:%M:%S | ') + str(msg), flush=True)


def oak_bring_up(account, project_id):
    cred_dir = os.path.join(oak_cred_home, account)
    os.environ['HOME'] = cred_dir
    os.environ['USERPROFILE'] = cred_dir
    os.environ.setdefault('HTTP_PROXY', oak_net['http'])
    os.environ.setdefault('HTTPS_PROXY', oak_net['https'])
    import ee
    for i_oak in range(5):
        try:
            ee.Initialize(project=project_id)
            break
        except Exception as x_oak_init:
            if i_oak == 5 - 1:
                raise
            oak_emit('initialize retry %d: %s' % (i_oak + 1, str(x_oak_init)[:96]))
            time.sleep(40)
    oak_emit('[ee] %s @ %s ready' % (account, project_id))
    return ee


def oak_pull_csv(target_url, tries, timeout_s):
    err_oak_last = None
    for j_oak in range(tries):
        try:
            resp = requests.get(target_url, proxies=oak_net, timeout=timeout_s)
            resp.raise_for_status()
            return pd.read_csv(io.BytesIO(resp.content))
        except Exception as x_oak_pull:
            err_oak_last = x_oak_pull
            oak_emit('download retry %d/%d: %s' % (j_oak + 1, tries, str(x_oak_pull)[:96]))
            time.sleep(25 + 20 * j_oak)
    raise RuntimeError('download exhausted: %s' % str(err_oak_last)[:120])


def oak_grab(ee, idx_df, cache, cid, prop):
    g = idx_df[idx_df.chunk_id == cid]
    outs = []
    for year in sorted(int(v) for v in g.year.unique().tolist()):
        gy = g[g.year == year]
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([float(r['lon']), float(r['lat'])]),
                       {prop: int(r['row_id'])}) for r in gy[['row_id', 'lon', 'lat']].to_dict('records')])
        key = '%s_%d' % (prop, year)
        if key not in cache:
            cache[key] = (ee.ImageCollection(oak_embcol)
                          .filterDate('%d-01-01' % year, '%d-01-01' % (year + 1)))
        emb = cache[key].filterBounds(fc).mosaic().select(oak_bands)
        samp = emb.sampleRegions(collection=fc, properties=[prop],
                                 scale=20, tileScale=oak_tile)
        url = samp.getDownloadURL(filetype='csv', selectors=[prop] + oak_bands)
        out = oak_pull_csv(url, 5, 840)
        out = out.rename(columns={prop: 'oak_rid'})
        out['oak_embyr'] = year
        outs.append(out)
    frame = pd.concat(outs, ignore_index=True)
    frame['oak_cid'] = int(cid)
    return frame


def oak_roll():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    oak_paths.ensure_all()
    idx_fp = os.path.join(oak_paths.NG, 'ng_chunks_index.parquet')
    if not os.path.isfile(idx_fp):
        raise SystemExit('缺少年度索引: %s' % idx_fp)
    idx_df = pd.read_parquet(idx_fp)
    out_dir = os.path.join(oak_paths.NG, 'ng_emb')
    os.makedirs(out_dir, exist_ok=True)
    all_cids = sorted(int(v) for v in idx_df.chunk_id.unique().tolist())
    my_cids = [c for c in all_cids if c % oak_nacc == 14]
    done = {int(f.split('_')[1].split('.')[0])
            for f in os.listdir(out_dir) if f.startswith('oakemb_') and f.endswith('.parquet')}
    todo = [c for c in my_cids if c not in done]
    if False:
        todo = todo[::-1]
    if a.limit:
        todo = todo[:a.limit]
    oak_emit('[%s] 索引 %d 块, 分配 %d, 待跑 %d' % (oak_who, len(all_cids), len(my_cids), len(todo)))
    if a.dry or not todo:
        return 0
    ee = oak_bring_up(oak_who, oak_home_proj)
    cache = {}
    ok = bad = 0
    for cid in todo:
        fp = os.path.join(out_dir, '%s%d.parquet' % ('oakemb_', cid))
        for try_oak in range(2):
            try:
                frame = oak_grab(ee, idx_df, cache, cid, 'oak_embid')
                frame.to_parquet(fp, index=False)
                ok += 1
                oak_emit('chunk %04d saved %d rows (%d/%d)' % (cid, len(frame), ok, len(todo)))
                break
            except Exception as x_oak_shard:
                oak_emit('chunk %04d attempt %d ERR %s' % (cid, try_oak + 1, str(x_oak_shard)[:110]))
                time.sleep(40 + try_oak * 25)
        else:
            bad += 1
            oak_emit('chunk %04d FINAL-FAIL' % cid)
        time.sleep(random.uniform(*oak_rest))
    oak_emit('[%s] done ok=%d fail=%d' % (oak_who, ok, bad))
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    raise SystemExit(oak_roll())
