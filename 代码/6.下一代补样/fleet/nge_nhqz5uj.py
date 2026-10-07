# -*- coding: utf-8 -*-
"""nhqz5uj 专属嵌入提取器 · 下一代补样 · 带状类别分层配额采点
由 ng_gen.py 生成（模板 ng_emb_tmpl.tpl），勿手改。
读 ng6_build_samples.py 产出的年度索引，为下一代样本点提取 AlphaEarth 64 维嵌入。
用法: python nge_nhqz5uj.py [--limit N] [--dry]
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

fir_net = {'http': 'socks5h://127.0.0.1:7890',
             'https': 'socks5h://127.0.0.1:7890'}
fir_cred_home = 'F:/地理所/论文/博士论文/中期/实验区域/02_中间数据/gee_accounts'
fir_who = 'nhqz5uj'
fir_home_proj = 'sunlit-center-508411-g3'
fir_tile = 4
fir_rest = (1, 6)

_fir_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_fir_dir))
import ng0_paths as fir_paths

fir_bands = [f'A{i:02d}' for i in range(64)]
fir_embcol = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
fir_nacc = 15


def fir_emit(msg):
    print(time.strftime('<%H:%M:%S> ') + str(msg), flush=True)


def fir_bring_up(account, project_id):
    cred_dir = os.path.join(fir_cred_home, account)
    os.environ['HOME'] = cred_dir
    os.environ['USERPROFILE'] = cred_dir
    os.environ.setdefault('HTTP_PROXY', fir_net['http'])
    os.environ.setdefault('HTTPS_PROXY', fir_net['https'])
    import ee
    for i_fir in range(5):
        try:
            ee.Initialize(project=project_id)
            break
        except Exception as x_fir_init:
            if i_fir == 5 - 1:
                raise
            fir_emit('initialize retry %d: %s' % (i_fir + 1, str(x_fir_init)[:96]))
            time.sleep(40)
    fir_emit('[ee] %s @ %s ready' % (account, project_id))
    return ee


def fir_pull_csv(target_url, tries, timeout_s):
    err_fir_last = None
    for j_fir in range(tries):
        try:
            resp = requests.get(target_url, proxies=fir_net, timeout=timeout_s)
            resp.raise_for_status()
            return pd.read_csv(io.BytesIO(resp.content))
        except Exception as x_fir_pull:
            err_fir_last = x_fir_pull
            fir_emit('download retry %d/%d: %s' % (j_fir + 1, tries, str(x_fir_pull)[:96]))
            time.sleep(20 + 20 * j_fir)
    raise RuntimeError('download exhausted: %s' % str(err_fir_last)[:120])


def fir_grab(ee, idx_df, cache, cid, prop):
    g = idx_df[idx_df.chunk_id == cid]
    outs = []
    for year in sorted(int(v) for v in g.year.unique().tolist()):
        gy = g[g.year == year]
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([float(r['lon']), float(r['lat'])]),
                       {prop: int(r['row_id'])}) for r in gy[['row_id', 'lon', 'lat']].to_dict('records')])
        key = '%s_%d' % (prop, year)
        if key not in cache:
            cache[key] = (ee.ImageCollection(fir_embcol)
                          .filterDate('%d-01-01' % year, '%d-01-01' % (year + 1)))
        emb = cache[key].filterBounds(fc).mosaic().select(fir_bands)
        samp = emb.sampleRegions(collection=fc, properties=[prop],
                                 scale=20, tileScale=fir_tile)
        url = samp.getDownloadURL(filetype='csv', selectors=[prop] + fir_bands)
        out = fir_pull_csv(url, 5, 720)
        out = out.rename(columns={prop: 'fir_rid'})
        out['fir_embyr'] = year
        outs.append(out)
    frame = pd.concat(outs, ignore_index=True)
    frame['fir_cid'] = int(cid)
    return frame


def fir_roll():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    fir_paths.ensure_all()
    idx_fp = os.path.join(fir_paths.NG, 'ng_chunks_index.parquet')
    if not os.path.isfile(idx_fp):
        raise SystemExit('缺少年度索引: %s' % idx_fp)
    idx_df = pd.read_parquet(idx_fp)
    out_dir = os.path.join(fir_paths.NG, 'ng_emb')
    os.makedirs(out_dir, exist_ok=True)
    all_cids = sorted(int(v) for v in idx_df.chunk_id.unique().tolist())
    my_cids = [c for c in all_cids if c % fir_nacc == 5]
    done = {int(f.split('_')[1].split('.')[0])
            for f in os.listdir(out_dir) if f.startswith('firemb_') and f.endswith('.parquet')}
    todo = [c for c in my_cids if c not in done]
    if True:
        todo = todo[::-1]
    if a.limit:
        todo = todo[:a.limit]
    fir_emit('[%s] 索引 %d 块, 分配 %d, 待跑 %d' % (fir_who, len(all_cids), len(my_cids), len(todo)))
    if a.dry or not todo:
        return 0
    ee = fir_bring_up(fir_who, fir_home_proj)
    cache = {}
    ok = bad = 0
    for cid in todo:
        fp = os.path.join(out_dir, '%s%d.parquet' % ('firemb_', cid))
        for try_fir in range(3):
            try:
                frame = fir_grab(ee, idx_df, cache, cid, 'fir_embid')
                frame.to_parquet(fp, index=False)
                ok += 1
                fir_emit('chunk %04d saved %d rows (%d/%d)' % (cid, len(frame), ok, len(todo)))
                break
            except Exception as x_fir_shard:
                fir_emit('chunk %04d attempt %d ERR %s' % (cid, try_fir + 1, str(x_fir_shard)[:110]))
                time.sleep(40 + try_fir * 25)
        else:
            bad += 1
            fir_emit('chunk %04d FINAL-FAIL' % cid)
        time.sleep(random.uniform(*fir_rest))
    fir_emit('[%s] done ok=%d fail=%d' % (fir_who, ok, bad))
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    raise SystemExit(fir_roll())
