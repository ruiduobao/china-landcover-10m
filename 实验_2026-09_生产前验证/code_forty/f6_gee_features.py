# -*- coding: utf-8 -*-
"""f6_gee_features.py — GEE 侧特征/校验数据补齐（三种模式，分片可续跑，多账号并行）

* 输入：点表 parquet（至少含 id 列与 lon/lat；id 列名 --idcol，默认 pid）
* 规则源：
  A. mode=forty：ForTy v1 概率 5 波段（projects/nature-trace/assets/forest_typology/
     forest_typology_2020_v1_0_collection 的 mosaic），scale=10，tileScale=4；
     派生 p_other = 250 − Σ(p1..p5)（ForTy 官方口径）；输出 6 个概率 + forty_ok。
  B. mode=terrain：SRTM 30 m 高程 + 坡度（ee.Terrain.slope），scale=30。
  C. mode=aef：AEF 64 维年度嵌入（GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL，scale=10），
     按 --years 逐年抓取，输出长表 (id, year, A00–A63, aef_ok)。
* 门槛：单点命中率 <0.95 告警；单块失败重试 3 次（退避）后置 aef_ok=0/forty_ok=0
* 输出：F:/lc_work/forty_exp/feat/<mode>/<name>_p<part>.parquet（--nparts 分片，幂等）
* 用法：
  python f6_gee_features.py --mode forty   --in cand/pos_candidates.parquet --name pos  --part 0 --nparts 4 --acct 1yxth5g
  python f6_gee_features.py --mode terrain --in cand/pos_candidates.parquet --name pos  --part 0 --nparts 2 --acct 5631jn8
  python f6_gee_features.py --mode aef     --in cand/pos_candidates.parquet --name pos  --part 0 --nparts 8 --acct 744s27z
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import v31_common as VC

WORK = r'F:/lc_work/forty_exp'
FEAT = os.path.join(WORK, 'feat')
FORTY = 'projects/nature-trace/assets/forest_typology/forest_typology_2020_v1_0_collection'
AEF = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
FB = ['PrimaryForest', 'NaturallyRegeneratingForest', 'PlantedForest',
      'PlantationForest', 'TreeCropsAndAgroforestry']
CHUNK = 4000


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def fc_of(df, idcol):
    import ee
    import geopandas as gpd  # noqa
    return ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([float(r[1]), float(r[2])]), {idcol: int(r[0])})
        for r in df[[idcol, 'lon', 'lat']].to_numpy()])


def sample(img, fc, idcol, scale, tile_scale=4, tries=3):
    got, last = None, ''
    for i in range(tries):
        try:
            got = img.sampleRegions(collection=fc, scale=scale, geometries=False,
                                    tileScale=tile_scale).getInfo()
            break
        except Exception as e:
            last = str(e)[:140]
            emit('  重试 %d：%s' % (i + 1, last))
            time.sleep(8 + 6 * i)
    if got is None:
        return None, last
    return [f['properties'] for f in got.get('features', [])], ''


def run_forty(df, idcol, out, acct):
    import ee
    VC.ensure_ctx(acct)
    img = ee.ImageCollection(FORTY).mosaic()
    recs = []
    for k in range(0, len(df), CHUNK):
        sub = df.iloc[k:k + CHUNK]
        props, err = sample(img, fc_of(sub, idcol), idcol, 10)
        if props is None:
            emit('  块 %d 失败：%s' % (k // CHUNK, err)); continue
        for p in props:
            r = {idcol: p.get(idcol)}
            ok = all(b in p for b in FB)
            r['forty_ok'] = bool(ok)
            for b in FB:
                r[b] = p.get(b, np.nan)
            recs.append(r)
        emit('  块 %d/%d 完成（%d 点）' % (k // CHUNK + 1, (len(df) + CHUNK - 1) // CHUNK, len(props)))
    d = pd.DataFrame(recs)
    if len(d):
        for b in FB:
            d[b] = pd.to_numeric(d[b], errors='coerce')
        d['p_other'] = 250.0 - d[FB].sum(axis=1, min_count=5)
        d['p_treecrop'] = d['TreeCropsAndAgroforestry']
        d['p_crop_argmax'] = d[FB + ['p_other']].idxmax(axis=1)
    d.to_parquet(out, index=False)
    hit = float(d.forty_ok.mean()) if len(d) else 0
    emit('forty：%d/%d 命中（%.4f）→ %s' % (int(d.forty_ok.sum()) if len(d) else 0, len(df), hit, out))


def run_terrain(df, idcol, out, acct):
    import ee
    VC.ensure_ctx(acct)
    dem = ee.Image('USGS/SRTMGL1_003')
    img = ee.Image.cat([dem.rename('elev'), ee.Terrain.slope(dem).rename('slope')])
    recs = []
    for k in range(0, len(df), CHUNK):
        sub = df.iloc[k:k + CHUNK]
        props, err = sample(img, fc_of(sub, idcol), idcol, 30)
        if props is None:
            emit('  块 %d 失败：%s' % (k // CHUNK, err)); continue
        recs += [{'%s' % idcol: p.get(idcol), 'elev': p.get('elev'), 'slope': p.get('slope')}
                 for p in props]
    d = pd.DataFrame(recs)
    d.to_parquet(out, index=False)
    emit('terrain：%d/%d → %s' % (len(d), len(df), out))


def run_aef(df, idcol, out, acct, years):
    import ee
    VC.ensure_ctx(acct)
    recs = []
    nchunks = (len(df) + CHUNK - 1) // CHUNK
    for y in years:
        img = (ee.ImageCollection(AEF).filterDate('%d-01-01' % y, '%d-01-01' % (y + 1))
               .mosaic().select(VC.FEATS))
        for k in range(0, len(df), CHUNK):
            sub = df.iloc[k:k + CHUNK]
            props, err = sample(img, fc_of(sub, idcol), idcol, 10)
            if props is None:
                emit('  %d 块 %d 失败：%s' % (y, k // CHUNK, err)); continue
            for p in props:
                r = {idcol: p.get(idcol), 'year': y, 'aef_ok': int(all(f in p for f in VC.FEATS))}
                for f in VC.FEATS:
                    r[f] = p.get(f, np.nan)
                recs.append(r)
            emit('  %d 块 %d/%d（累计 %d 行）' % (y, k // CHUNK + 1, nchunks, len(recs)))
    d = pd.DataFrame(recs)
    d.to_parquet(out, index=False)
    emit('aef：%d 行（%d 点 × %d 年），命中 %.4f → %s' % (
        len(d), len(df), len(years), float(d.aef_ok.mean()) if len(d) else 0, out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['forty', 'terrain', 'aef'])
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--idcol', default='pid')
    ap.add_argument('--part', type=int, default=0)
    ap.add_argument('--nparts', type=int, default=1)
    ap.add_argument('--acct', default='1yxth5g')
    ap.add_argument('--years', default='2017,2018,2019,2020,2021,2022,2023,2024')
    a = ap.parse_args()
    od = os.path.join(FEAT, a.mode)
    os.makedirs(od, exist_ok=True)
    out = os.path.join(od, '%s_p%d.parquet' % (a.name, a.part))
    if os.path.exists(out) and os.path.getsize(out) > 0:
        emit('已存在，跳过 %s' % out)
        return
    df = pd.read_parquet(a.inp)
    if a.nparts > 1:
        df = df.iloc[a.part::a.nparts].reset_index(drop=True)
    emit('%s：%d 点（part %d/%d，账号 %s）' % (a.mode, len(df), a.part, a.nparts, a.acct))
    if a.mode == 'forty':
        run_forty(df, a.idcol, out, a.acct)
    elif a.mode == 'terrain':
        run_terrain(df, a.idcol, out, a.acct)
    else:
        years = [int(y) for y in a.years.split(',')]
        run_aef(df, a.idcol, out, a.acct, years)


if __name__ == '__main__':
    main()
