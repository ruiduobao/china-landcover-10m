# -*- coding: utf-8 -*-
"""5rqs3xw2 专属嵌入提取器 · 下一代补样 · 高寒候选点交叉筛选（双年 WorldCover 一致）
由 ng_gen.py 生成（模板 ng_emb_tmpl.tpl），勿手改。
读 ng6_build_samples.py 产出的年度索引，为下一代样本点提取 AlphaEarth 64 维嵌入。
用法: python nge_5rqs3xw2.py [--limit N] [--dry]
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

kudzu_net = {'http': 'socks5h://127.0.0.1:7890',
               'https': 'socks5h://127.0.0.1:7890'}
kudzu_cred_home = 'F:/地理所/论文/博士论文/中期/实验区域/02_中间数据/gee_accounts'
kudzu_who = '5rqs3xw2'
kudzu_home_proj = 'tokyo-kingdom-508501-t0'
kudzu_tile = 3
kudzu_rest = (3, 4)

_kudzu_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_kudzu_dir))
import ng0_paths as kudzu_paths

kudzu_bands = [f'A{i:02d}' for i in range(64)]
kudzu_embcol = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
kudzu_nacc = 15


def kudzu_emit(msg):
    print(time.strftime('[%H:%M] ') + str(msg), flush=True)


def kudzu_bring_up(account, project_id):
    cred_dir = os.path.join(kudzu_cred_home, account)
    os.environ['HOME'] = cred_dir
    os.environ['USERPROFILE'] = cred_dir
    os.environ.setdefault('HTTP_PROXY', kudzu_net['http'])
    os.environ.setdefault('HTTPS_PROXY', kudzu_net['https'])
    import ee
    for i_kudzu in range(5):
        try:
            ee.Initialize(project=project_id)
            break
        except Exception as x_kudzu_init:
            if i_kudzu == 5 - 1:
                raise
            kudzu_emit('initialize retry %d: %s' % (i_kudzu + 1, str(x_kudzu_init)[:96]))
            time.sleep(30)
    kudzu_emit('[ee] %s @ %s ready' % (account, project_id))
    return ee


def kudzu_pull_csv(target_url, tries, timeout_s):
    err_kudzu_last = None
    for j_kudzu in range(tries):
        try:
            resp = requests.get(target_url, proxies=kudzu_net, timeout=timeout_s)
            resp.raise_for_status()
            return pd.read_csv(io.BytesIO(resp.content))
        except Exception as x_kudzu_pull:
            err_kudzu_last = x_kudzu_pull
            kudzu_emit('download retry %d/%d: %s' % (j_kudzu + 1, tries, str(x_kudzu_pull)[:96]))
            time.sleep(25 + 15 * j_kudzu)
    raise RuntimeError('download exhausted: %s' % str(err_kudzu_last)[:120])


def kudzu_grab(ee, idx_df, cache, cid, prop):
    g = idx_df[idx_df.chunk_id == cid]
    outs = []
    for year in sorted(int(v) for v in g.year.unique().tolist()):
        gy = g[g.year == year]
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([float(r['lon']), float(r['lat'])]),
                       {prop: int(r['row_id'])}) for r in gy[['row_id', 'lon', 'lat']].to_dict('records')])
        key = '%s_%d' % (prop, year)
        if key not in cache:
            cache[key] = (ee.ImageCollection(kudzu_embcol)
                          .filterDate('%d-01-01' % year, '%d-01-01' % (year + 1)))
        emb = cache[key].filterBounds(fc).mosaic().select(kudzu_bands)
        samp = emb.sampleRegions(collection=fc, properties=[prop],
                                 scale=10, tileScale=kudzu_tile)
        url = samp.getDownloadURL(filetype='csv', selectors=[prop] + kudzu_bands)
        out = kudzu_pull_csv(url, 4, 840)
        out = out.rename(columns={prop: 'kudzu_rid'})
        out['kudzu_embyr'] = year
        outs.append(out)
    frame = pd.concat(outs, ignore_index=True)
    frame['kudzu_cid'] = int(cid)
    return frame


def kudzu_roll():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    kudzu_paths.ensure_all()
    idx_fp = os.path.join(kudzu_paths.NG, 'ng_chunks_index.parquet')
    if not os.path.isfile(idx_fp):
        raise SystemExit('缺少年度索引: %s' % idx_fp)
    idx_df = pd.read_parquet(idx_fp)
    out_dir = os.path.join(kudzu_paths.NG, 'ng_emb')
    os.makedirs(out_dir, exist_ok=True)
    all_cids = sorted(int(v) for v in idx_df.chunk_id.unique().tolist())
    my_cids = [c for c in all_cids if c % kudzu_nacc == 10]
    done = {int(f.split('_')[1].split('.')[0])
            for f in os.listdir(out_dir) if f.startswith('kudzuemb_') and f.endswith('.parquet')}
    todo = [c for c in my_cids if c not in done]
    if False:
        todo = todo[::-1]
    if a.limit:
        todo = todo[:a.limit]
    kudzu_emit('[%s] 索引 %d 块, 分配 %d, 待跑 %d' % (kudzu_who, len(all_cids), len(my_cids), len(todo)))
    if a.dry or not todo:
        return 0
    ee = kudzu_bring_up(kudzu_who, kudzu_home_proj)
    cache = {}
    ok = bad = 0
    for cid in todo:
        fp = os.path.join(out_dir, '%s%d.parquet' % ('kudzuemb_', cid))
        for try_kudzu in range(2):
            try:
                frame = kudzu_grab(ee, idx_df, cache, cid, 'kudzu_embid')
                frame.to_parquet(fp, index=False)
                ok += 1
                kudzu_emit('chunk %04d saved %d rows (%d/%d)' % (cid, len(frame), ok, len(todo)))
                break
            except Exception as x_kudzu_shard:
                kudzu_emit('chunk %04d attempt %d ERR %s' % (cid, try_kudzu + 1, str(x_kudzu_shard)[:110]))
                time.sleep(30 + try_kudzu * 20)
        else:
            bad += 1
            kudzu_emit('chunk %04d FINAL-FAIL' % cid)
        time.sleep(random.uniform(*kudzu_rest))
    kudzu_emit('[%s] done ok=%d fail=%d' % (kudzu_who, ok, bad))
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    raise SystemExit(kudzu_roll())
