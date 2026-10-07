# -*- coding: utf-8 -*-
"""@@ACCT@@ 专属嵌入提取器 · 下一代补样 · @@WORDING@@
由 ng_gen.py 生成（模板 ng_emb_tmpl.tpl），勿手改。
读 ng6_build_samples.py 产出的年度索引，为下一代样本点提取 AlphaEarth 64 维嵌入。
用法: python @@EMBFILE@@ [--limit N] [--dry]
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

@@CONST_BLOCK@@

@@V_HERE@@ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(@@V_HERE@@))
import ng0_paths as @@M_PATHS@@

@@V_FEATS@@ = [f'A{i:02d}' for i in range(64)]
@@V_EMBIC@@ = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
@@V_NACC@@ = @@N_NACC@@


def @@FN_LOG@@(msg):
    print(time.strftime(@@TSFMT@@) + str(msg), flush=True)


def @@FN_BOOT@@(account, project_id):
    cred_dir = os.path.join(@@V_ACCROOT@@, account)
    os.environ['HOME'] = cred_dir
    os.environ['USERPROFILE'] = cred_dir
    os.environ.setdefault('HTTP_PROXY', @@V_PROXY@@[@@K_HTTP@@])
    os.environ.setdefault('HTTPS_PROXY', @@V_PROXY@@[@@K_HTTPS@@])
    import ee
    for @@L_I@@ in range(@@N_INIT_TRIES@@):
        try:
            ee.Initialize(project=project_id)
            break
        except Exception as @@L_EXC@@:
            if @@L_I@@ == @@N_INIT_TRIES@@ - 1:
                raise
            @@FN_LOG@@('initialize retry %d: %s' % (@@L_I@@ + 1, str(@@L_EXC@@)[:96]))
            time.sleep(@@N_INIT_WAIT@@)
    @@FN_LOG@@('[ee] %s @ %s ready' % (account, project_id))
    return ee


def @@FN_FETCH@@(target_url, tries, timeout_s):
    @@L_LAST@@ = None
    for @@L_J@@ in range(tries):
        try:
            resp = requests.get(target_url, proxies=@@V_PROXY@@, timeout=timeout_s)
            resp.raise_for_status()
            return pd.read_csv(io.BytesIO(resp.content))
        except Exception as @@L_EXC2@@:
            @@L_LAST@@ = @@L_EXC2@@
            @@FN_LOG@@('download retry %d/%d: %s' % (@@L_J@@ + 1, tries, str(@@L_EXC2@@)[:96]))
            time.sleep(@@N_DL_WAIT@@ + @@N_DL_STEP@@ * @@L_J@@)
    raise RuntimeError('download exhausted: %s' % str(@@L_LAST@@)[:120])


def @@FN_CHUNK@@(ee, idx_df, cache, cid, prop):
    g = idx_df[idx_df.chunk_id == cid]
    outs = []
    for year in sorted(int(v) for v in g.year.unique().tolist()):
        gy = g[g.year == year]
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([float(r['lon']), float(r['lat'])]),
                       {prop: int(r['row_id'])}) for r in gy[['row_id', 'lon', 'lat']].to_dict('records')])
        key = '%s_%d' % (prop, year)
        if key not in cache:
            cache[key] = (ee.ImageCollection(@@V_EMBIC@@)
                          .filterDate('%d-01-01' % year, '%d-01-01' % (year + 1)))
        emb = cache[key].filterBounds(fc).mosaic().select(@@V_FEATS@@)
        samp = emb.sampleRegions(collection=fc, properties=[prop],
                                 scale=@@N_EMBSCALE@@, tileScale=@@V_TS@@)
        url = samp.getDownloadURL(filetype=@@K_CSV@@, selectors=[prop] + @@V_FEATS@@)
        out = @@FN_FETCH@@(url, @@N_NRETRY@@, @@N_TIMEOUT@@)
        out = out.rename(columns={prop: @@K_RID@@})
        out[@@K_EMBYR@@] = year
        outs.append(out)
    frame = pd.concat(outs, ignore_index=True)
    frame[@@K_CID@@] = int(cid)
    return frame


def @@FN_MAIN@@():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    @@M_PATHS@@.ensure_all()
    idx_fp = os.path.join(@@M_PATHS@@.NG, @@K_IDXFN@@)
    if not os.path.isfile(idx_fp):
        raise SystemExit('缺少年度索引: %s' % idx_fp)
    idx_df = pd.read_parquet(idx_fp)
    out_dir = os.path.join(@@M_PATHS@@.NG, @@K_OUTDN@@)
    os.makedirs(out_dir, exist_ok=True)
    all_cids = sorted(int(v) for v in idx_df.chunk_id.unique().tolist())
    my_cids = [c for c in all_cids if c % @@V_NACC@@ == @@N_AIDX@@]
    done = {int(f.split('_')[1].split('.')[0])
            for f in os.listdir(out_dir) if f.startswith(@@K_PREFIX@@) and f.endswith('.parquet')}
    todo = [c for c in my_cids if c not in done]
    if @@E_REV@@:
        todo = todo[::-1]
    if a.limit:
        todo = todo[:a.limit]
    @@FN_LOG@@('[%s] 索引 %d 块, 分配 %d, 待跑 %d' % (@@V_TAG@@, len(all_cids), len(my_cids), len(todo)))
    if a.dry or not todo:
        return 0
    ee = @@FN_BOOT@@(@@V_TAG@@, @@V_ANCHOR@@)
    cache = {}
    ok = bad = 0
    for cid in todo:
        fp = os.path.join(out_dir, '%s%d.parquet' % (@@K_PREFIX@@, cid))
        for @@L_ATT@@ in range(@@N_SHARD_TRIES@@):
            try:
                frame = @@FN_CHUNK@@(ee, idx_df, cache, cid, @@K_PROP@@)
                frame.to_parquet(fp, index=False)
                ok += 1
                @@FN_LOG@@('chunk %04d saved %d rows (%d/%d)' % (cid, len(frame), ok, len(todo)))
                break
            except Exception as @@L_EXC3@@:
                @@FN_LOG@@('chunk %04d attempt %d ERR %s' % (cid, @@L_ATT@@ + 1, str(@@L_EXC3@@)[:110]))
                time.sleep(@@N_SHARD_WAIT@@ + @@L_ATT@@ * @@N_SHARD_STEP@@)
        else:
            bad += 1
            @@FN_LOG@@('chunk %04d FINAL-FAIL' % cid)
        time.sleep(random.uniform(*@@V_SLEEP@@))
    @@FN_LOG@@('[%s] done ok=%d fail=%d' % (@@V_TAG@@, ok, bad))
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    raise SystemExit(@@FN_MAIN@@())
