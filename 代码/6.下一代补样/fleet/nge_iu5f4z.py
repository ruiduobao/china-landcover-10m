# -*- coding: utf-8 -*-
"""iu5f4z 专属嵌入提取器 · 下一代补样 · 生态区候选点构建（多源一致性）
由 ng_gen.py 生成（模板 ng_emb_tmpl.tpl），勿手改。
读 ng6_build_samples.py 产出的年度索引，为下一代样本点提取 AlphaEarth 64 维嵌入。
用法: python nge_iu5f4z.py [--limit N] [--dry]
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

elm_net = {'http': 'socks5h://127.0.0.1:7890',
             'https': 'socks5h://127.0.0.1:7890'}
elm_cred_home = 'F:/地理所/论文/博士论文/中期/实验区域/02_中间数据/gee_accounts'
elm_who = 'iu5f4z'
elm_home_proj = 'ringed-sentinel-508412-d8'
elm_tile = 3
elm_rest = (1, 6)

_elm_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_elm_dir))
import ng0_paths as elm_paths

elm_bands = [f'A{i:02d}' for i in range(64)]
elm_embcol = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
elm_nacc = 15


def elm_emit(msg):
    print(time.strftime('[%H:%M] ') + str(msg), flush=True)


def elm_bring_up(account, project_id):
    cred_dir = os.path.join(elm_cred_home, account)
    os.environ['HOME'] = cred_dir
    os.environ['USERPROFILE'] = cred_dir
    os.environ.setdefault('HTTP_PROXY', elm_net['http'])
    os.environ.setdefault('HTTPS_PROXY', elm_net['https'])
    import ee
    for i_elm in range(5):
        try:
            ee.Initialize(project=project_id)
            break
        except Exception as x_elm_init:
            if i_elm == 5 - 1:
                raise
            elm_emit('initialize retry %d: %s' % (i_elm + 1, str(x_elm_init)[:96]))
            time.sleep(30)
    elm_emit('[ee] %s @ %s ready' % (account, project_id))
    return ee


def elm_pull_csv(target_url, tries, timeout_s):
    err_elm_last = None
    for j_elm in range(tries):
        try:
            resp = requests.get(target_url, proxies=elm_net, timeout=timeout_s)
            resp.raise_for_status()
            return pd.read_csv(io.BytesIO(resp.content))
        except Exception as x_elm_pull:
            err_elm_last = x_elm_pull
            elm_emit('download retry %d/%d: %s' % (j_elm + 1, tries, str(x_elm_pull)[:96]))
            time.sleep(15 + 15 * j_elm)
    raise RuntimeError('download exhausted: %s' % str(err_elm_last)[:120])


def elm_grab(ee, idx_df, cache, cid, prop):
    g = idx_df[idx_df.chunk_id == cid]
    outs = []
    for year in sorted(int(v) for v in g.year.unique().tolist()):
        gy = g[g.year == year]
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([float(r['lon']), float(r['lat'])]),
                       {prop: int(r['row_id'])}) for r in gy[['row_id', 'lon', 'lat']].to_dict('records')])
        key = '%s_%d' % (prop, year)
        if key not in cache:
            cache[key] = (ee.ImageCollection(elm_embcol)
                          .filterDate('%d-01-01' % year, '%d-01-01' % (year + 1)))
        emb = cache[key].filterBounds(fc).mosaic().select(elm_bands)
        samp = emb.sampleRegions(collection=fc, properties=[prop],
                                 scale=10, tileScale=elm_tile)
        url = samp.getDownloadURL(filetype='csv', selectors=[prop] + elm_bands)
        out = elm_pull_csv(url, 4, 600)
        out = out.rename(columns={prop: 'elm_rid'})
        out['elm_embyr'] = year
        outs.append(out)
    frame = pd.concat(outs, ignore_index=True)
    frame['elm_cid'] = int(cid)
    return frame


def elm_roll():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    elm_paths.ensure_all()
    idx_fp = os.path.join(elm_paths.NG, 'ng_chunks_index.parquet')
    if not os.path.isfile(idx_fp):
        raise SystemExit('缺少年度索引: %s' % idx_fp)
    idx_df = pd.read_parquet(idx_fp)
    out_dir = os.path.join(elm_paths.NG, 'ng_emb')
    os.makedirs(out_dir, exist_ok=True)
    all_cids = sorted(int(v) for v in idx_df.chunk_id.unique().tolist())
    my_cids = [c for c in all_cids if c % elm_nacc == 4]
    done = {int(f.split('_')[1].split('.')[0])
            for f in os.listdir(out_dir) if f.startswith('elmemb_') and f.endswith('.parquet')}
    todo = [c for c in my_cids if c not in done]
    if False:
        todo = todo[::-1]
    if a.limit:
        todo = todo[:a.limit]
    elm_emit('[%s] 索引 %d 块, 分配 %d, 待跑 %d' % (elm_who, len(all_cids), len(my_cids), len(todo)))
    if a.dry or not todo:
        return 0
    ee = elm_bring_up(elm_who, elm_home_proj)
    cache = {}
    ok = bad = 0
    for cid in todo:
        fp = os.path.join(out_dir, '%s%d.parquet' % ('elmemb_', cid))
        for try_elm in range(2):
            try:
                frame = elm_grab(ee, idx_df, cache, cid, 'elm_embid')
                frame.to_parquet(fp, index=False)
                ok += 1
                elm_emit('chunk %04d saved %d rows (%d/%d)' % (cid, len(frame), ok, len(todo)))
                break
            except Exception as x_elm_shard:
                elm_emit('chunk %04d attempt %d ERR %s' % (cid, try_elm + 1, str(x_elm_shard)[:110]))
                time.sleep(30 + try_elm * 20)
        else:
            bad += 1
            elm_emit('chunk %04d FINAL-FAIL' % cid)
        time.sleep(random.uniform(*elm_rest))
    elm_emit('[%s] done ok=%d fail=%d' % (elm_who, ok, bad))
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    raise SystemExit(elm_roll())
