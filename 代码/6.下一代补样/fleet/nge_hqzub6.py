# -*- coding: utf-8 -*-
"""hqzub6 专属嵌入提取器 · 下一代补样 · 海岸带候选点交叉筛选（专题源与全球产品双证）
由 ng_gen.py 生成（模板 ng_emb_tmpl.tpl），勿手改。
读 ng6_build_samples.py 产出的年度索引，为下一代样本点提取 AlphaEarth 64 维嵌入。
用法: python nge_hqzub6.py [--limit N] [--dry]
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

larch_net = {'http': 'socks5h://127.0.0.1:7890',
               'https': 'socks5h://127.0.0.1:7890'}
larch_cred_home = 'F:/地理所/论文/博士论文/中期/实验区域/02_中间数据/gee_accounts'
larch_who = 'hqzub6'
larch_home_proj = 'copper-bot-508501-c4'
larch_tile = 4
larch_rest = (3, 4)

_larch_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_larch_dir))
import ng0_paths as larch_paths

larch_bands = [f'A{i:02d}' for i in range(64)]
larch_embcol = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
larch_nacc = 15


def larch_emit(msg):
    print(time.strftime('<%H:%M:%S> ') + str(msg), flush=True)


def larch_bring_up(account, project_id):
    cred_dir = os.path.join(larch_cred_home, account)
    os.environ['HOME'] = cred_dir
    os.environ['USERPROFILE'] = cred_dir
    os.environ.setdefault('HTTP_PROXY', larch_net['http'])
    os.environ.setdefault('HTTPS_PROXY', larch_net['https'])
    import ee
    for i_larch in range(5):
        try:
            ee.Initialize(project=project_id)
            break
        except Exception as x_larch_init:
            if i_larch == 5 - 1:
                raise
            larch_emit('initialize retry %d: %s' % (i_larch + 1, str(x_larch_init)[:96]))
            time.sleep(40)
    larch_emit('[ee] %s @ %s ready' % (account, project_id))
    return ee


def larch_pull_csv(target_url, tries, timeout_s):
    err_larch_last = None
    for j_larch in range(tries):
        try:
            resp = requests.get(target_url, proxies=larch_net, timeout=timeout_s)
            resp.raise_for_status()
            return pd.read_csv(io.BytesIO(resp.content))
        except Exception as x_larch_pull:
            err_larch_last = x_larch_pull
            larch_emit('download retry %d/%d: %s' % (j_larch + 1, tries, str(x_larch_pull)[:96]))
            time.sleep(30 + 20 * j_larch)
    raise RuntimeError('download exhausted: %s' % str(err_larch_last)[:120])


def larch_grab(ee, idx_df, cache, cid, prop):
    g = idx_df[idx_df.chunk_id == cid]
    outs = []
    for year in sorted(int(v) for v in g.year.unique().tolist()):
        gy = g[g.year == year]
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([float(r['lon']), float(r['lat'])]),
                       {prop: int(r['row_id'])}) for r in gy[['row_id', 'lon', 'lat']].to_dict('records')])
        key = '%s_%d' % (prop, year)
        if key not in cache:
            cache[key] = (ee.ImageCollection(larch_embcol)
                          .filterDate('%d-01-01' % year, '%d-01-01' % (year + 1)))
        emb = cache[key].filterBounds(fc).mosaic().select(larch_bands)
        samp = emb.sampleRegions(collection=fc, properties=[prop],
                                 scale=20, tileScale=larch_tile)
        url = samp.getDownloadURL(filetype='csv', selectors=[prop] + larch_bands)
        out = larch_pull_csv(url, 5, 960)
        out = out.rename(columns={prop: 'larch_rid'})
        out['larch_embyr'] = year
        outs.append(out)
    frame = pd.concat(outs, ignore_index=True)
    frame['larch_cid'] = int(cid)
    return frame


def larch_roll():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    larch_paths.ensure_all()
    idx_fp = os.path.join(larch_paths.NG, 'ng_chunks_index.parquet')
    if not os.path.isfile(idx_fp):
        raise SystemExit('缺少年度索引: %s' % idx_fp)
    idx_df = pd.read_parquet(idx_fp)
    out_dir = os.path.join(larch_paths.NG, 'ng_emb')
    os.makedirs(out_dir, exist_ok=True)
    all_cids = sorted(int(v) for v in idx_df.chunk_id.unique().tolist())
    my_cids = [c for c in all_cids if c % larch_nacc == 11]
    done = {int(f.split('_')[1].split('.')[0])
            for f in os.listdir(out_dir) if f.startswith('larchemb_') and f.endswith('.parquet')}
    todo = [c for c in my_cids if c not in done]
    if True:
        todo = todo[::-1]
    if a.limit:
        todo = todo[:a.limit]
    larch_emit('[%s] 索引 %d 块, 分配 %d, 待跑 %d' % (larch_who, len(all_cids), len(my_cids), len(todo)))
    if a.dry or not todo:
        return 0
    ee = larch_bring_up(larch_who, larch_home_proj)
    cache = {}
    ok = bad = 0
    for cid in todo:
        fp = os.path.join(out_dir, '%s%d.parquet' % ('larchemb_', cid))
        for try_larch in range(3):
            try:
                frame = larch_grab(ee, idx_df, cache, cid, 'larch_embid')
                frame.to_parquet(fp, index=False)
                ok += 1
                larch_emit('chunk %04d saved %d rows (%d/%d)' % (cid, len(frame), ok, len(todo)))
                break
            except Exception as x_larch_shard:
                larch_emit('chunk %04d attempt %d ERR %s' % (cid, try_larch + 1, str(x_larch_shard)[:110]))
                time.sleep(40 + try_larch * 25)
        else:
            bad += 1
            larch_emit('chunk %04d FINAL-FAIL' % cid)
        time.sleep(random.uniform(*larch_rest))
    larch_emit('[%s] done ok=%d fail=%d' % (larch_who, ok, bad))
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    raise SystemExit(larch_roll())
