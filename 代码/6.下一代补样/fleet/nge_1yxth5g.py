# -*- coding: utf-8 -*-
"""1yxth5g 专属嵌入提取器 · 下一代补样 · 潮汐湿地候选点筛选（过渡频次判据）
由 ng_gen.py 生成（模板 ng_emb_tmpl.tpl），勿手改。
读 ng6_build_samples.py 产出的年度索引，为下一代样本点提取 AlphaEarth 64 维嵌入。
用法: python nge_1yxth5g.py [--limit N] [--dry]
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

maple_net = {'http': 'socks5h://127.0.0.1:7890',
               'https': 'socks5h://127.0.0.1:7890'}
maple_cred_home = 'F:/地理所/论文/博士论文/中期/实验区域/02_中间数据/gee_accounts'
maple_who = '1yxth5g'
maple_home_proj = 'elated-chassis-508309-u6'
maple_tile = 2
maple_rest = (3, 4)

_maple_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_maple_dir))
import ng0_paths as maple_paths

maple_bands = [f'A{i:02d}' for i in range(64)]
maple_embcol = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
maple_nacc = 15


def maple_emit(msg):
    print(time.strftime('[%H:%M:%S] ') + str(msg), flush=True)


def maple_bring_up(account, project_id):
    cred_dir = os.path.join(maple_cred_home, account)
    os.environ['HOME'] = cred_dir
    os.environ['USERPROFILE'] = cred_dir
    os.environ.setdefault('HTTP_PROXY', maple_net['http'])
    os.environ.setdefault('HTTPS_PROXY', maple_net['https'])
    import ee
    for i_maple in range(5):
        try:
            ee.Initialize(project=project_id)
            break
        except Exception as x_maple_init:
            if i_maple == 5 - 1:
                raise
            maple_emit('initialize retry %d: %s' % (i_maple + 1, str(x_maple_init)[:96]))
            time.sleep(20)
    maple_emit('[ee] %s @ %s ready' % (account, project_id))
    return ee


def maple_pull_csv(target_url, tries, timeout_s):
    err_maple_last = None
    for j_maple in range(tries):
        try:
            resp = requests.get(target_url, proxies=maple_net, timeout=timeout_s)
            resp.raise_for_status()
            return pd.read_csv(io.BytesIO(resp.content))
        except Exception as x_maple_pull:
            err_maple_last = x_maple_pull
            maple_emit('download retry %d/%d: %s' % (j_maple + 1, tries, str(x_maple_pull)[:96]))
            time.sleep(15 + 10 * j_maple)
    raise RuntimeError('download exhausted: %s' % str(err_maple_last)[:120])


def maple_grab(ee, idx_df, cache, cid, prop):
    g = idx_df[idx_df.chunk_id == cid]
    outs = []
    for year in sorted(int(v) for v in g.year.unique().tolist()):
        gy = g[g.year == year]
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([float(r['lon']), float(r['lat'])]),
                       {prop: int(r['row_id'])}) for r in gy[['row_id', 'lon', 'lat']].to_dict('records')])
        key = '%s_%d' % (prop, year)
        if key not in cache:
            cache[key] = (ee.ImageCollection(maple_embcol)
                          .filterDate('%d-01-01' % year, '%d-01-01' % (year + 1)))
        emb = cache[key].filterBounds(fc).mosaic().select(maple_bands)
        samp = emb.sampleRegions(collection=fc, properties=[prop],
                                 scale=10, tileScale=maple_tile)
        url = samp.getDownloadURL(filetype='csv', selectors=[prop] + maple_bands)
        out = maple_pull_csv(url, 3, 600)
        out = out.rename(columns={prop: 'maple_rid'})
        out['maple_embyr'] = year
        outs.append(out)
    frame = pd.concat(outs, ignore_index=True)
    frame['maple_cid'] = int(cid)
    return frame


def maple_roll():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    maple_paths.ensure_all()
    idx_fp = os.path.join(maple_paths.NG, 'ng_chunks_index.parquet')
    if not os.path.isfile(idx_fp):
        raise SystemExit('缺少年度索引: %s' % idx_fp)
    idx_df = pd.read_parquet(idx_fp)
    out_dir = os.path.join(maple_paths.NG, 'ng_emb')
    os.makedirs(out_dir, exist_ok=True)
    all_cids = sorted(int(v) for v in idx_df.chunk_id.unique().tolist())
    my_cids = [c for c in all_cids if c % maple_nacc == 12]
    done = {int(f.split('_')[1].split('.')[0])
            for f in os.listdir(out_dir) if f.startswith('mapleemb_') and f.endswith('.parquet')}
    todo = [c for c in my_cids if c not in done]
    if False:
        todo = todo[::-1]
    if a.limit:
        todo = todo[:a.limit]
    maple_emit('[%s] 索引 %d 块, 分配 %d, 待跑 %d' % (maple_who, len(all_cids), len(my_cids), len(todo)))
    if a.dry or not todo:
        return 0
    ee = maple_bring_up(maple_who, maple_home_proj)
    cache = {}
    ok = bad = 0
    for cid in todo:
        fp = os.path.join(out_dir, '%s%d.parquet' % ('mapleemb_', cid))
        for try_maple in range(2):
            try:
                frame = maple_grab(ee, idx_df, cache, cid, 'maple_embid')
                frame.to_parquet(fp, index=False)
                ok += 1
                maple_emit('chunk %04d saved %d rows (%d/%d)' % (cid, len(frame), ok, len(todo)))
                break
            except Exception as x_maple_shard:
                maple_emit('chunk %04d attempt %d ERR %s' % (cid, try_maple + 1, str(x_maple_shard)[:110]))
                time.sleep(20 + try_maple * 15)
        else:
            bad += 1
            maple_emit('chunk %04d FINAL-FAIL' % cid)
        time.sleep(random.uniform(*maple_rest))
    maple_emit('[%s] done ok=%d fail=%d' % (maple_who, ok, bad))
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    raise SystemExit(maple_roll())
