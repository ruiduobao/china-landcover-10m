# -*- coding: utf-8 -*-
"""zitwwufh 专属嵌入提取器 · 下一代补样 · 潮汐湿地候选点筛选（过渡频次判据）
由 ng_gen.py 生成（模板 ng_emb_tmpl.tpl），勿手改。
读 ng6_build_samples.py 产出的年度索引，为下一代样本点提取 AlphaEarth 64 维嵌入。
用法: python nge_zitwwufh.py [--limit N] [--dry]
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

cedar_net = {'http': 'socks5h://127.0.0.1:7890',
               'https': 'socks5h://127.0.0.1:7890'}
cedar_cred_home = 'F:/地理所/论文/博士论文/中期/实验区域/02_中间数据/gee_accounts'
cedar_who = 'zitwwufh'
cedar_home_proj = 'fast-drake-508210-f7'
cedar_tile = 4
cedar_rest = (3, 4)

_cedar_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_cedar_dir))
import ng0_paths as cedar_paths

cedar_bands = [f'A{i:02d}' for i in range(64)]
cedar_embcol = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
cedar_nacc = 15


def cedar_emit(msg):
    print(time.strftime('%H:%M:%S | ') + str(msg), flush=True)


def cedar_bring_up(account, project_id):
    cred_dir = os.path.join(cedar_cred_home, account)
    os.environ['HOME'] = cred_dir
    os.environ['USERPROFILE'] = cred_dir
    os.environ.setdefault('HTTP_PROXY', cedar_net['http'])
    os.environ.setdefault('HTTPS_PROXY', cedar_net['https'])
    import ee
    for i_cedar in range(5):
        try:
            ee.Initialize(project=project_id)
            break
        except Exception as x_cedar_init:
            if i_cedar == 5 - 1:
                raise
            cedar_emit('initialize retry %d: %s' % (i_cedar + 1, str(x_cedar_init)[:96]))
            time.sleep(40)
    cedar_emit('[ee] %s @ %s ready' % (account, project_id))
    return ee


def cedar_pull_csv(target_url, tries, timeout_s):
    err_cedar_last = None
    for j_cedar in range(tries):
        try:
            resp = requests.get(target_url, proxies=cedar_net, timeout=timeout_s)
            resp.raise_for_status()
            return pd.read_csv(io.BytesIO(resp.content))
        except Exception as x_cedar_pull:
            err_cedar_last = x_cedar_pull
            cedar_emit('download retry %d/%d: %s' % (j_cedar + 1, tries, str(x_cedar_pull)[:96]))
            time.sleep(25 + 20 * j_cedar)
    raise RuntimeError('download exhausted: %s' % str(err_cedar_last)[:120])


def cedar_grab(ee, idx_df, cache, cid, prop):
    g = idx_df[idx_df.chunk_id == cid]
    outs = []
    for year in sorted(int(v) for v in g.year.unique().tolist()):
        gy = g[g.year == year]
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([float(r['lon']), float(r['lat'])]),
                       {prop: int(r['row_id'])}) for r in gy[['row_id', 'lon', 'lat']].to_dict('records')])
        key = '%s_%d' % (prop, year)
        if key not in cache:
            cache[key] = (ee.ImageCollection(cedar_embcol)
                          .filterDate('%d-01-01' % year, '%d-01-01' % (year + 1)))
        emb = cache[key].filterBounds(fc).mosaic().select(cedar_bands)
        samp = emb.sampleRegions(collection=fc, properties=[prop],
                                 scale=20, tileScale=cedar_tile)
        url = samp.getDownloadURL(filetype='csv', selectors=[prop] + cedar_bands)
        out = cedar_pull_csv(url, 5, 840)
        out = out.rename(columns={prop: 'cedar_rid'})
        out['cedar_embyr'] = year
        outs.append(out)
    frame = pd.concat(outs, ignore_index=True)
    frame['cedar_cid'] = int(cid)
    return frame


def cedar_roll():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    cedar_paths.ensure_all()
    idx_fp = os.path.join(cedar_paths.NG, 'ng_chunks_index.parquet')
    if not os.path.isfile(idx_fp):
        raise SystemExit('缺少年度索引: %s' % idx_fp)
    idx_df = pd.read_parquet(idx_fp)
    out_dir = os.path.join(cedar_paths.NG, 'ng_emb')
    os.makedirs(out_dir, exist_ok=True)
    all_cids = sorted(int(v) for v in idx_df.chunk_id.unique().tolist())
    my_cids = [c for c in all_cids if c % cedar_nacc == 2]
    done = {int(f.split('_')[1].split('.')[0])
            for f in os.listdir(out_dir) if f.startswith('cedaremb_') and f.endswith('.parquet')}
    todo = [c for c in my_cids if c not in done]
    if False:
        todo = todo[::-1]
    if a.limit:
        todo = todo[:a.limit]
    cedar_emit('[%s] 索引 %d 块, 分配 %d, 待跑 %d' % (cedar_who, len(all_cids), len(my_cids), len(todo)))
    if a.dry or not todo:
        return 0
    ee = cedar_bring_up(cedar_who, cedar_home_proj)
    cache = {}
    ok = bad = 0
    for cid in todo:
        fp = os.path.join(out_dir, '%s%d.parquet' % ('cedaremb_', cid))
        for try_cedar in range(2):
            try:
                frame = cedar_grab(ee, idx_df, cache, cid, 'cedar_embid')
                frame.to_parquet(fp, index=False)
                ok += 1
                cedar_emit('chunk %04d saved %d rows (%d/%d)' % (cid, len(frame), ok, len(todo)))
                break
            except Exception as x_cedar_shard:
                cedar_emit('chunk %04d attempt %d ERR %s' % (cid, try_cedar + 1, str(x_cedar_shard)[:110]))
                time.sleep(40 + try_cedar * 25)
        else:
            bad += 1
            cedar_emit('chunk %04d FINAL-FAIL' % cid)
        time.sleep(random.uniform(*cedar_rest))
    cedar_emit('[%s] done ok=%d fail=%d' % (cedar_who, ok, bad))
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    raise SystemExit(cedar_roll())
