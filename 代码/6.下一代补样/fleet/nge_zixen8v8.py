# -*- coding: utf-8 -*-
"""zixen8v8 专属嵌入提取器 · 下一代补样 · 高寒候选点交叉筛选（双年 WorldCover 一致）
由 ng_gen.py 生成（模板 ng_emb_tmpl.tpl），勿手改。
读 ng6_build_samples.py 产出的年度索引，为下一代样本点提取 AlphaEarth 64 维嵌入。
用法: python nge_zixen8v8.py [--limit N] [--dry]
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

alder_net = {'http': 'socks5h://127.0.0.1:7890',
               'https': 'socks5h://127.0.0.1:7890'}
alder_cred_home = 'F:/地理所/论文/博士论文/中期/实验区域/02_中间数据/gee_accounts'
alder_who = 'zixen8v8'
alder_home_proj = 'braided-horizon-508210-a5'
alder_tile = 2
alder_rest = (3, 4)

_alder_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_alder_dir))
import ng0_paths as alder_paths

alder_bands = [f'A{i:02d}' for i in range(64)]
alder_embcol = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
alder_nacc = 15


def alder_emit(msg):
    print(time.strftime('[%H:%M:%S] ') + str(msg), flush=True)


def alder_bring_up(account, project_id):
    cred_dir = os.path.join(alder_cred_home, account)
    os.environ['HOME'] = cred_dir
    os.environ['USERPROFILE'] = cred_dir
    os.environ.setdefault('HTTP_PROXY', alder_net['http'])
    os.environ.setdefault('HTTPS_PROXY', alder_net['https'])
    import ee
    for i_alder in range(5):
        try:
            ee.Initialize(project=project_id)
            break
        except Exception as x_alder_init:
            if i_alder == 5 - 1:
                raise
            alder_emit('initialize retry %d: %s' % (i_alder + 1, str(x_alder_init)[:96]))
            time.sleep(20)
    alder_emit('[ee] %s @ %s ready' % (account, project_id))
    return ee


def alder_pull_csv(target_url, tries, timeout_s):
    err_alder_last = None
    for j_alder in range(tries):
        try:
            resp = requests.get(target_url, proxies=alder_net, timeout=timeout_s)
            resp.raise_for_status()
            return pd.read_csv(io.BytesIO(resp.content))
        except Exception as x_alder_pull:
            err_alder_last = x_alder_pull
            alder_emit('download retry %d/%d: %s' % (j_alder + 1, tries, str(x_alder_pull)[:96]))
            time.sleep(15 + 10 * j_alder)
    raise RuntimeError('download exhausted: %s' % str(err_alder_last)[:120])


def alder_grab(ee, idx_df, cache, cid, prop):
    g = idx_df[idx_df.chunk_id == cid]
    outs = []
    for year in sorted(int(v) for v in g.year.unique().tolist()):
        gy = g[g.year == year]
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([float(r['lon']), float(r['lat'])]),
                       {prop: int(r['row_id'])}) for r in gy[['row_id', 'lon', 'lat']].to_dict('records')])
        key = '%s_%d' % (prop, year)
        if key not in cache:
            cache[key] = (ee.ImageCollection(alder_embcol)
                          .filterDate('%d-01-01' % year, '%d-01-01' % (year + 1)))
        emb = cache[key].filterBounds(fc).mosaic().select(alder_bands)
        samp = emb.sampleRegions(collection=fc, properties=[prop],
                                 scale=10, tileScale=alder_tile)
        url = samp.getDownloadURL(filetype='csv', selectors=[prop] + alder_bands)
        out = alder_pull_csv(url, 3, 600)
        out = out.rename(columns={prop: 'alder_rid'})
        out['alder_embyr'] = year
        outs.append(out)
    frame = pd.concat(outs, ignore_index=True)
    frame['alder_cid'] = int(cid)
    return frame


def alder_roll():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    alder_paths.ensure_all()
    idx_fp = os.path.join(alder_paths.NG, 'ng_chunks_index.parquet')
    if not os.path.isfile(idx_fp):
        raise SystemExit('缺少年度索引: %s' % idx_fp)
    idx_df = pd.read_parquet(idx_fp)
    out_dir = os.path.join(alder_paths.NG, 'ng_emb')
    os.makedirs(out_dir, exist_ok=True)
    all_cids = sorted(int(v) for v in idx_df.chunk_id.unique().tolist())
    my_cids = [c for c in all_cids if c % alder_nacc == 0]
    done = {int(f.split('_')[1].split('.')[0])
            for f in os.listdir(out_dir) if f.startswith('alderemb_') and f.endswith('.parquet')}
    todo = [c for c in my_cids if c not in done]
    if False:
        todo = todo[::-1]
    if a.limit:
        todo = todo[:a.limit]
    alder_emit('[%s] 索引 %d 块, 分配 %d, 待跑 %d' % (alder_who, len(all_cids), len(my_cids), len(todo)))
    if a.dry or not todo:
        return 0
    ee = alder_bring_up(alder_who, alder_home_proj)
    cache = {}
    ok = bad = 0
    for cid in todo:
        fp = os.path.join(out_dir, '%s%d.parquet' % ('alderemb_', cid))
        for try_alder in range(2):
            try:
                frame = alder_grab(ee, idx_df, cache, cid, 'alder_embid')
                frame.to_parquet(fp, index=False)
                ok += 1
                alder_emit('chunk %04d saved %d rows (%d/%d)' % (cid, len(frame), ok, len(todo)))
                break
            except Exception as x_alder_shard:
                alder_emit('chunk %04d attempt %d ERR %s' % (cid, try_alder + 1, str(x_alder_shard)[:110]))
                time.sleep(20 + try_alder * 15)
        else:
            bad += 1
            alder_emit('chunk %04d FINAL-FAIL' % cid)
        time.sleep(random.uniform(*alder_rest))
    alder_emit('[%s] done ok=%d fail=%d' % (alder_who, ok, bad))
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    raise SystemExit(alder_roll())
