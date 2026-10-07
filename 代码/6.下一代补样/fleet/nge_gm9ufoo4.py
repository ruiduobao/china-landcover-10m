# -*- coding: utf-8 -*-
"""gm9ufoo4 专属嵌入提取器 · 下一代补样 · 目标类核心点提取
由 ng_gen.py 生成（模板 ng_emb_tmpl.tpl），勿手改。
读 ng6_build_samples.py 产出的年度索引，为下一代样本点提取 AlphaEarth 64 维嵌入。
用法: python nge_gm9ufoo4.py [--limit N] [--dry]
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

iris_net = {'http': 'socks5h://127.0.0.1:7890',
              'https': 'socks5h://127.0.0.1:7890'}
iris_cred_home = 'F:/地理所/论文/博士论文/中期/实验区域/02_中间数据/gee_accounts'
iris_who = 'gm9ufoo4'
iris_home_proj = 'xenon-momentum-508404-k3'
iris_tile = 4
iris_rest = (2, 3)

_iris_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_iris_dir))
import ng0_paths as iris_paths

iris_bands = [f'A{i:02d}' for i in range(64)]
iris_embcol = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
iris_nacc = 15


def iris_emit(msg):
    print(time.strftime('%H:%M:%S | ') + str(msg), flush=True)


def iris_bring_up(account, project_id):
    cred_dir = os.path.join(iris_cred_home, account)
    os.environ['HOME'] = cred_dir
    os.environ['USERPROFILE'] = cred_dir
    os.environ.setdefault('HTTP_PROXY', iris_net['http'])
    os.environ.setdefault('HTTPS_PROXY', iris_net['https'])
    import ee
    for i_iris in range(5):
        try:
            ee.Initialize(project=project_id)
            break
        except Exception as x_iris_init:
            if i_iris == 5 - 1:
                raise
            iris_emit('initialize retry %d: %s' % (i_iris + 1, str(x_iris_init)[:96]))
            time.sleep(40)
    iris_emit('[ee] %s @ %s ready' % (account, project_id))
    return ee


def iris_pull_csv(target_url, tries, timeout_s):
    err_iris_last = None
    for j_iris in range(tries):
        try:
            resp = requests.get(target_url, proxies=iris_net, timeout=timeout_s)
            resp.raise_for_status()
            return pd.read_csv(io.BytesIO(resp.content))
        except Exception as x_iris_pull:
            err_iris_last = x_iris_pull
            iris_emit('download retry %d/%d: %s' % (j_iris + 1, tries, str(x_iris_pull)[:96]))
            time.sleep(15 + 20 * j_iris)
    raise RuntimeError('download exhausted: %s' % str(err_iris_last)[:120])


def iris_grab(ee, idx_df, cache, cid, prop):
    g = idx_df[idx_df.chunk_id == cid]
    outs = []
    for year in sorted(int(v) for v in g.year.unique().tolist()):
        gy = g[g.year == year]
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([float(r['lon']), float(r['lat'])]),
                       {prop: int(r['row_id'])}) for r in gy[['row_id', 'lon', 'lat']].to_dict('records')])
        key = '%s_%d' % (prop, year)
        if key not in cache:
            cache[key] = (ee.ImageCollection(iris_embcol)
                          .filterDate('%d-01-01' % year, '%d-01-01' % (year + 1)))
        emb = cache[key].filterBounds(fc).mosaic().select(iris_bands)
        samp = emb.sampleRegions(collection=fc, properties=[prop],
                                 scale=20, tileScale=iris_tile)
        url = samp.getDownloadURL(filetype='csv', selectors=[prop] + iris_bands)
        out = iris_pull_csv(url, 5, 600)
        out = out.rename(columns={prop: 'iris_rid'})
        out['iris_embyr'] = year
        outs.append(out)
    frame = pd.concat(outs, ignore_index=True)
    frame['iris_cid'] = int(cid)
    return frame


def iris_roll():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    iris_paths.ensure_all()
    idx_fp = os.path.join(iris_paths.NG, 'ng_chunks_index.parquet')
    if not os.path.isfile(idx_fp):
        raise SystemExit('缺少年度索引: %s' % idx_fp)
    idx_df = pd.read_parquet(idx_fp)
    out_dir = os.path.join(iris_paths.NG, 'ng_emb')
    os.makedirs(out_dir, exist_ok=True)
    all_cids = sorted(int(v) for v in idx_df.chunk_id.unique().tolist())
    my_cids = [c for c in all_cids if c % iris_nacc == 8]
    done = {int(f.split('_')[1].split('.')[0])
            for f in os.listdir(out_dir) if f.startswith('irisemb_') and f.endswith('.parquet')}
    todo = [c for c in my_cids if c not in done]
    if False:
        todo = todo[::-1]
    if a.limit:
        todo = todo[:a.limit]
    iris_emit('[%s] 索引 %d 块, 分配 %d, 待跑 %d' % (iris_who, len(all_cids), len(my_cids), len(todo)))
    if a.dry or not todo:
        return 0
    ee = iris_bring_up(iris_who, iris_home_proj)
    cache = {}
    ok = bad = 0
    for cid in todo:
        fp = os.path.join(out_dir, '%s%d.parquet' % ('irisemb_', cid))
        for try_iris in range(2):
            try:
                frame = iris_grab(ee, idx_df, cache, cid, 'iris_embid')
                frame.to_parquet(fp, index=False)
                ok += 1
                iris_emit('chunk %04d saved %d rows (%d/%d)' % (cid, len(frame), ok, len(todo)))
                break
            except Exception as x_iris_shard:
                iris_emit('chunk %04d attempt %d ERR %s' % (cid, try_iris + 1, str(x_iris_shard)[:110]))
                time.sleep(40 + try_iris * 25)
        else:
            bad += 1
            iris_emit('chunk %04d FINAL-FAIL' % cid)
        time.sleep(random.uniform(*iris_rest))
    iris_emit('[%s] done ok=%d fail=%d' % (iris_who, ok, bad))
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    raise SystemExit(iris_roll())
