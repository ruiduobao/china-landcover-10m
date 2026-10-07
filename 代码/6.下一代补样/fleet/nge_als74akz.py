# -*- coding: utf-8 -*-
"""als74akz 专属嵌入提取器 · 下一代补样 · 水稻淹水特征候选点筛选（SAR 时序）
由 ng_gen.py 生成（模板 ng_emb_tmpl.tpl），勿手改。
读 ng6_build_samples.py 产出的年度索引，为下一代样本点提取 AlphaEarth 64 维嵌入。
用法: python nge_als74akz.py [--limit N] [--dry]
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

dogwood_net = {'http': 'socks5h://127.0.0.1:7890',
                 'https': 'socks5h://127.0.0.1:7890'}
dogwood_cred_home = 'F:/地理所/论文/博士论文/中期/实验区域/02_中间数据/gee_accounts'
dogwood_who = 'als74akz'
dogwood_home_proj = 'the-method-508211-s4'
dogwood_tile = 2
dogwood_rest = (2, 6)

_dogwood_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_dogwood_dir))
import ng0_paths as dogwood_paths

dogwood_bands = [f'A{i:02d}' for i in range(64)]
dogwood_embcol = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
dogwood_nacc = 15


def dogwood_emit(msg):
    print(time.strftime('(%H:%M:%S) ') + str(msg), flush=True)


def dogwood_bring_up(account, project_id):
    cred_dir = os.path.join(dogwood_cred_home, account)
    os.environ['HOME'] = cred_dir
    os.environ['USERPROFILE'] = cred_dir
    os.environ.setdefault('HTTP_PROXY', dogwood_net['http'])
    os.environ.setdefault('HTTPS_PROXY', dogwood_net['https'])
    import ee
    for i_dogwood in range(5):
        try:
            ee.Initialize(project=project_id)
            break
        except Exception as x_dogwood_init:
            if i_dogwood == 5 - 1:
                raise
            dogwood_emit('initialize retry %d: %s' % (i_dogwood + 1, str(x_dogwood_init)[:96]))
            time.sleep(20)
    dogwood_emit('[ee] %s @ %s ready' % (account, project_id))
    return ee


def dogwood_pull_csv(target_url, tries, timeout_s):
    err_dogwood_last = None
    for j_dogwood in range(tries):
        try:
            resp = requests.get(target_url, proxies=dogwood_net, timeout=timeout_s)
            resp.raise_for_status()
            return pd.read_csv(io.BytesIO(resp.content))
        except Exception as x_dogwood_pull:
            err_dogwood_last = x_dogwood_pull
            dogwood_emit('download retry %d/%d: %s' % (j_dogwood + 1, tries, str(x_dogwood_pull)[:96]))
            time.sleep(30 + 10 * j_dogwood)
    raise RuntimeError('download exhausted: %s' % str(err_dogwood_last)[:120])


def dogwood_grab(ee, idx_df, cache, cid, prop):
    g = idx_df[idx_df.chunk_id == cid]
    outs = []
    for year in sorted(int(v) for v in g.year.unique().tolist()):
        gy = g[g.year == year]
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([float(r['lon']), float(r['lat'])]),
                       {prop: int(r['row_id'])}) for r in gy[['row_id', 'lon', 'lat']].to_dict('records')])
        key = '%s_%d' % (prop, year)
        if key not in cache:
            cache[key] = (ee.ImageCollection(dogwood_embcol)
                          .filterDate('%d-01-01' % year, '%d-01-01' % (year + 1)))
        emb = cache[key].filterBounds(fc).mosaic().select(dogwood_bands)
        samp = emb.sampleRegions(collection=fc, properties=[prop],
                                 scale=10, tileScale=dogwood_tile)
        url = samp.getDownloadURL(filetype='csv', selectors=[prop] + dogwood_bands)
        out = dogwood_pull_csv(url, 3, 960)
        out = out.rename(columns={prop: 'dogwood_rid'})
        out['dogwood_embyr'] = year
        outs.append(out)
    frame = pd.concat(outs, ignore_index=True)
    frame['dogwood_cid'] = int(cid)
    return frame


def dogwood_roll():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    dogwood_paths.ensure_all()
    idx_fp = os.path.join(dogwood_paths.NG, 'ng_chunks_index.parquet')
    if not os.path.isfile(idx_fp):
        raise SystemExit('缺少年度索引: %s' % idx_fp)
    idx_df = pd.read_parquet(idx_fp)
    out_dir = os.path.join(dogwood_paths.NG, 'ng_emb')
    os.makedirs(out_dir, exist_ok=True)
    all_cids = sorted(int(v) for v in idx_df.chunk_id.unique().tolist())
    my_cids = [c for c in all_cids if c % dogwood_nacc == 3]
    done = {int(f.split('_')[1].split('.')[0])
            for f in os.listdir(out_dir) if f.startswith('dogwoodemb_') and f.endswith('.parquet')}
    todo = [c for c in my_cids if c not in done]
    if True:
        todo = todo[::-1]
    if a.limit:
        todo = todo[:a.limit]
    dogwood_emit('[%s] 索引 %d 块, 分配 %d, 待跑 %d' % (dogwood_who, len(all_cids), len(my_cids), len(todo)))
    if a.dry or not todo:
        return 0
    ee = dogwood_bring_up(dogwood_who, dogwood_home_proj)
    cache = {}
    ok = bad = 0
    for cid in todo:
        fp = os.path.join(out_dir, '%s%d.parquet' % ('dogwoodemb_', cid))
        for try_dogwood in range(3):
            try:
                frame = dogwood_grab(ee, idx_df, cache, cid, 'dogwood_embid')
                frame.to_parquet(fp, index=False)
                ok += 1
                dogwood_emit('chunk %04d saved %d rows (%d/%d)' % (cid, len(frame), ok, len(todo)))
                break
            except Exception as x_dogwood_shard:
                dogwood_emit('chunk %04d attempt %d ERR %s' % (cid, try_dogwood + 1, str(x_dogwood_shard)[:110]))
                time.sleep(20 + try_dogwood * 15)
        else:
            bad += 1
            dogwood_emit('chunk %04d FINAL-FAIL' % cid)
        time.sleep(random.uniform(*dogwood_rest))
    dogwood_emit('[%s] done ok=%d fail=%d' % (dogwood_who, ok, bad))
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    raise SystemExit(dogwood_roll())
