# -*- coding: utf-8 -*-
"""y30b63ye 专属嵌入提取器 · 下一代补样 · 海岸带候选点交叉筛选（专题源与全球产品双证）
由 ng_gen.py 生成（模板 ng_emb_tmpl.tpl），勿手改。
读 ng6_build_samples.py 产出的年度索引，为下一代样本点提取 AlphaEarth 64 维嵌入。
用法: python nge_y30b63ye.py [--limit N] [--dry]
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

birch_net = {'http': 'socks5h://127.0.0.1:7890',
               'https': 'socks5h://127.0.0.1:7890'}
birch_cred_home = 'F:/地理所/论文/博士论文/中期/实验区域/02_中间数据/gee_accounts'
birch_who = 'y30b63ye'
birch_home_proj = 'quick-cache-508211-s9'
birch_tile = 3
birch_rest = (3, 4)

_birch_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_birch_dir))
import ng0_paths as birch_paths

birch_bands = [f'A{i:02d}' for i in range(64)]
birch_embcol = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
birch_nacc = 15


def birch_emit(msg):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(msg), flush=True)


def birch_bring_up(account, project_id):
    cred_dir = os.path.join(birch_cred_home, account)
    os.environ['HOME'] = cred_dir
    os.environ['USERPROFILE'] = cred_dir
    os.environ.setdefault('HTTP_PROXY', birch_net['http'])
    os.environ.setdefault('HTTPS_PROXY', birch_net['https'])
    import ee
    for i_birch in range(5):
        try:
            ee.Initialize(project=project_id)
            break
        except Exception as x_birch_init:
            if i_birch == 5 - 1:
                raise
            birch_emit('initialize retry %d: %s' % (i_birch + 1, str(x_birch_init)[:96]))
            time.sleep(30)
    birch_emit('[ee] %s @ %s ready' % (account, project_id))
    return ee


def birch_pull_csv(target_url, tries, timeout_s):
    err_birch_last = None
    for j_birch in range(tries):
        try:
            resp = requests.get(target_url, proxies=birch_net, timeout=timeout_s)
            resp.raise_for_status()
            return pd.read_csv(io.BytesIO(resp.content))
        except Exception as x_birch_pull:
            err_birch_last = x_birch_pull
            birch_emit('download retry %d/%d: %s' % (j_birch + 1, tries, str(x_birch_pull)[:96]))
            time.sleep(20 + 15 * j_birch)
    raise RuntimeError('download exhausted: %s' % str(err_birch_last)[:120])


def birch_grab(ee, idx_df, cache, cid, prop):
    g = idx_df[idx_df.chunk_id == cid]
    outs = []
    for year in sorted(int(v) for v in g.year.unique().tolist()):
        gy = g[g.year == year]
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([float(r['lon']), float(r['lat'])]),
                       {prop: int(r['row_id'])}) for r in gy[['row_id', 'lon', 'lat']].to_dict('records')])
        key = '%s_%d' % (prop, year)
        if key not in cache:
            cache[key] = (ee.ImageCollection(birch_embcol)
                          .filterDate('%d-01-01' % year, '%d-01-01' % (year + 1)))
        emb = cache[key].filterBounds(fc).mosaic().select(birch_bands)
        samp = emb.sampleRegions(collection=fc, properties=[prop],
                                 scale=10, tileScale=birch_tile)
        url = samp.getDownloadURL(filetype='csv', selectors=[prop] + birch_bands)
        out = birch_pull_csv(url, 4, 720)
        out = out.rename(columns={prop: 'birch_rid'})
        out['birch_embyr'] = year
        outs.append(out)
    frame = pd.concat(outs, ignore_index=True)
    frame['birch_cid'] = int(cid)
    return frame


def birch_roll():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    birch_paths.ensure_all()
    idx_fp = os.path.join(birch_paths.NG, 'ng_chunks_index.parquet')
    if not os.path.isfile(idx_fp):
        raise SystemExit('缺少年度索引: %s' % idx_fp)
    idx_df = pd.read_parquet(idx_fp)
    out_dir = os.path.join(birch_paths.NG, 'ng_emb')
    os.makedirs(out_dir, exist_ok=True)
    all_cids = sorted(int(v) for v in idx_df.chunk_id.unique().tolist())
    my_cids = [c for c in all_cids if c % birch_nacc == 1]
    done = {int(f.split('_')[1].split('.')[0])
            for f in os.listdir(out_dir) if f.startswith('birchemb_') and f.endswith('.parquet')}
    todo = [c for c in my_cids if c not in done]
    if True:
        todo = todo[::-1]
    if a.limit:
        todo = todo[:a.limit]
    birch_emit('[%s] 索引 %d 块, 分配 %d, 待跑 %d' % (birch_who, len(all_cids), len(my_cids), len(todo)))
    if a.dry or not todo:
        return 0
    ee = birch_bring_up(birch_who, birch_home_proj)
    cache = {}
    ok = bad = 0
    for cid in todo:
        fp = os.path.join(out_dir, '%s%d.parquet' % ('birchemb_', cid))
        for try_birch in range(3):
            try:
                frame = birch_grab(ee, idx_df, cache, cid, 'birch_embid')
                frame.to_parquet(fp, index=False)
                ok += 1
                birch_emit('chunk %04d saved %d rows (%d/%d)' % (cid, len(frame), ok, len(todo)))
                break
            except Exception as x_birch_shard:
                birch_emit('chunk %04d attempt %d ERR %s' % (cid, try_birch + 1, str(x_birch_shard)[:110]))
                time.sleep(30 + try_birch * 20)
        else:
            bad += 1
            birch_emit('chunk %04d FINAL-FAIL' % cid)
        time.sleep(random.uniform(*birch_rest))
    birch_emit('[%s] done ok=%d fail=%d' % (birch_who, ok, bad))
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    raise SystemExit(birch_roll())
