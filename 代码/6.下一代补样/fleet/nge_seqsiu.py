# -*- coding: utf-8 -*-
"""seqsiu 专属嵌入提取器 · 下一代补样 · 水稻淹水特征候选点筛选（SAR 时序）
由 ng_gen.py 生成（模板 ng_emb_tmpl.tpl），勿手改。
读 ng6_build_samples.py 产出的年度索引，为下一代样本点提取 AlphaEarth 64 维嵌入。
用法: python nge_seqsiu.py [--limit N] [--dry]
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

nutmeg_net = {'http': 'socks5h://127.0.0.1:7890',
                'https': 'socks5h://127.0.0.1:7890'}
nutmeg_cred_home = 'F:/地理所/论文/博士论文/中期/实验区域/02_中间数据/gee_accounts'
nutmeg_who = 'seqsiu'
nutmeg_home_proj = 'electric-orbit-508407-g4'
nutmeg_tile = 3
nutmeg_rest = (1, 5)

_nutmeg_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_nutmeg_dir))
import ng0_paths as nutmeg_paths

nutmeg_bands = [f'A{i:02d}' for i in range(64)]
nutmeg_embcol = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
nutmeg_nacc = 15


def nutmeg_emit(msg):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(msg), flush=True)


def nutmeg_bring_up(account, project_id):
    cred_dir = os.path.join(nutmeg_cred_home, account)
    os.environ['HOME'] = cred_dir
    os.environ['USERPROFILE'] = cred_dir
    os.environ.setdefault('HTTP_PROXY', nutmeg_net['http'])
    os.environ.setdefault('HTTPS_PROXY', nutmeg_net['https'])
    import ee
    for i_nutmeg in range(5):
        try:
            ee.Initialize(project=project_id)
            break
        except Exception as x_nutmeg_init:
            if i_nutmeg == 5 - 1:
                raise
            nutmeg_emit('initialize retry %d: %s' % (i_nutmeg + 1, str(x_nutmeg_init)[:96]))
            time.sleep(30)
    nutmeg_emit('[ee] %s @ %s ready' % (account, project_id))
    return ee


def nutmeg_pull_csv(target_url, tries, timeout_s):
    err_nutmeg_last = None
    for j_nutmeg in range(tries):
        try:
            resp = requests.get(target_url, proxies=nutmeg_net, timeout=timeout_s)
            resp.raise_for_status()
            return pd.read_csv(io.BytesIO(resp.content))
        except Exception as x_nutmeg_pull:
            err_nutmeg_last = x_nutmeg_pull
            nutmeg_emit('download retry %d/%d: %s' % (j_nutmeg + 1, tries, str(x_nutmeg_pull)[:96]))
            time.sleep(20 + 15 * j_nutmeg)
    raise RuntimeError('download exhausted: %s' % str(err_nutmeg_last)[:120])


def nutmeg_grab(ee, idx_df, cache, cid, prop):
    g = idx_df[idx_df.chunk_id == cid]
    outs = []
    for year in sorted(int(v) for v in g.year.unique().tolist()):
        gy = g[g.year == year]
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([float(r['lon']), float(r['lat'])]),
                       {prop: int(r['row_id'])}) for r in gy[['row_id', 'lon', 'lat']].to_dict('records')])
        key = '%s_%d' % (prop, year)
        if key not in cache:
            cache[key] = (ee.ImageCollection(nutmeg_embcol)
                          .filterDate('%d-01-01' % year, '%d-01-01' % (year + 1)))
        emb = cache[key].filterBounds(fc).mosaic().select(nutmeg_bands)
        samp = emb.sampleRegions(collection=fc, properties=[prop],
                                 scale=10, tileScale=nutmeg_tile)
        url = samp.getDownloadURL(filetype='csv', selectors=[prop] + nutmeg_bands)
        out = nutmeg_pull_csv(url, 4, 720)
        out = out.rename(columns={prop: 'nutmeg_rid'})
        out['nutmeg_embyr'] = year
        outs.append(out)
    frame = pd.concat(outs, ignore_index=True)
    frame['nutmeg_cid'] = int(cid)
    return frame


def nutmeg_roll():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    nutmeg_paths.ensure_all()
    idx_fp = os.path.join(nutmeg_paths.NG, 'ng_chunks_index.parquet')
    if not os.path.isfile(idx_fp):
        raise SystemExit('缺少年度索引: %s' % idx_fp)
    idx_df = pd.read_parquet(idx_fp)
    out_dir = os.path.join(nutmeg_paths.NG, 'ng_emb')
    os.makedirs(out_dir, exist_ok=True)
    all_cids = sorted(int(v) for v in idx_df.chunk_id.unique().tolist())
    my_cids = [c for c in all_cids if c % nutmeg_nacc == 13]
    done = {int(f.split('_')[1].split('.')[0])
            for f in os.listdir(out_dir) if f.startswith('nutmegemb_') and f.endswith('.parquet')}
    todo = [c for c in my_cids if c not in done]
    if True:
        todo = todo[::-1]
    if a.limit:
        todo = todo[:a.limit]
    nutmeg_emit('[%s] 索引 %d 块, 分配 %d, 待跑 %d' % (nutmeg_who, len(all_cids), len(my_cids), len(todo)))
    if a.dry or not todo:
        return 0
    ee = nutmeg_bring_up(nutmeg_who, nutmeg_home_proj)
    cache = {}
    ok = bad = 0
    for cid in todo:
        fp = os.path.join(out_dir, '%s%d.parquet' % ('nutmegemb_', cid))
        for try_nutmeg in range(3):
            try:
                frame = nutmeg_grab(ee, idx_df, cache, cid, 'nutmeg_embid')
                frame.to_parquet(fp, index=False)
                ok += 1
                nutmeg_emit('chunk %04d saved %d rows (%d/%d)' % (cid, len(frame), ok, len(todo)))
                break
            except Exception as x_nutmeg_shard:
                nutmeg_emit('chunk %04d attempt %d ERR %s' % (cid, try_nutmeg + 1, str(x_nutmeg_shard)[:110]))
                time.sleep(30 + try_nutmeg * 20)
        else:
            bad += 1
            nutmeg_emit('chunk %04d FINAL-FAIL' % cid)
        time.sleep(random.uniform(*nutmeg_rest))
    nutmeg_emit('[%s] done ok=%d fail=%d' % (nutmeg_who, ok, bad))
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    raise SystemExit(nutmeg_roll())
