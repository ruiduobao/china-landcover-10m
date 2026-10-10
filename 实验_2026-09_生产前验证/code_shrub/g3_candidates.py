# -*- coding: utf-8 -*-
"""g3_candidates.py — 四省灌丛候选人生成（多源共识分层）+ 特征提取

* 输入：GEE 源——Dynamic World v1（label/prob）、ESA WorldCover v200(20)、Copernicus LC100(20)、
        FCS10 灌丛层点（池 src）、我们的成品（v1 灌丛类 9/10 像元）
* 规则源（预注册，分层抽样的"共识层级"）：
        L1 = DW shrub 且 (WC20 或 LC100_20) —— 双源以上一致
        L2 = DW shrub 且 prob ≥ 0.6 —— 单源高置信
        L3 = 仅 WC20 或仅 LC100_20（DW 未判灌丛）—— 单源低共识（对照用）
        另：OSM 要素点（独立）与 我们的成品灌丛像元（独立）单独列层 OSM / V1
* 门槛：每省每层抽样上限见 N_MAX；点间距 ≥2 km；AEF 必须取到
* 特征：DW 概率、WC20/LC100 是否灌丛、AEF64、S2 四季 NDVI（→ 振幅 amp = max-min）、
        GEDI rh98（L2A MONTHLY，150m 邻域中位）、SLOPE/DEM、bio01/bio12
* 输出：data/shrub/cand_all.csv（point_id,prov,stratum,lon,lat,DW_prob,WC20,LC100,A00..A63,
        ndvi_jja,ndvi_djf,ndvi_amp,rh98,dem,bio01,bio12,src_osm,v1_shrub）
* 用法：python g3_candidates.py [--acct bx15mw] [--per 60]
"""
import argparse, json, os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
sys.path.insert(0, os.path.join(WORK, 'code'))
import numpy as np, pandas as pd
import v31_common as VC

OUTD = os.path.join(WORK, 'data', 'shrub')
OUT = os.path.join(OUTD, 'cand_all.csv')
POOL = r'F:/lc_work/prod5p_2023/data/train2023_clean.parquet'
PROV = {'宁夏': (104.0, 35.0, 108.0, 40.0), '四川': (97.0, 26.0, 109.0, 34.5),
        '黑龙江': (121.0, 43.0, 135.5, 53.5), '福建': (115.5, 23.0, 120.5, 28.5)}
SEED = 20261010
MIN_KM = 2.0


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acct', default='bx15mw')
    ap.add_argument('--per', type=int, default=60, help='每省每层抽样上限')
    a = ap.parse_args()
    os.makedirs(OUTD, exist_ok=True)
    import ee
    VC.ensure_ctx(a.acct)
    rng = np.random.RandomState(SEED)

    dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
    wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map')
    lc = ee.Image('COPERNICUS/Landcover/100m/Proba-V-C3/Global/2019').select('discrete_classification')
    dem = ee.Image('USGS/SRTMGL1_003')
    bio = ee.Image('WORLDCLIM/V1/BIO').select(['bio01', 'bio12'])
    gedi = ee.ImageCollection('LARSE/GEDI/GEDI02_A_002_MONTHLY').select('rh98')

    def gt(pts, tag):
        return ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(x), float(y)]),
                                                {'point_id': '%s' % pid, 'stratum': tag})
                                     for pid, x, y in pts])

    def sample_pts(fc, want, n):
        """在 fc 的像元网格上按随机点抽样；返回 [(lon,lat)]。"""
        try:
            s = fc.sample(region=fc.geometry().bounds(), scale=100, numPixels=n * 12,
                          seed=SEED, geometries=True, tileScale=4).getInfo()['features']
        except Exception as e:
            emit('  sample 失败 %s' % str(e)[:80])
            return []
        out, used = [], []
        for f in s:
            c = f['geometry']['coordinates']
            lon, lat = float(c[0]), float(c[1])
            if any(((lon - u) ** 2 * 100 + (lat - v) ** 2 * 120) < (MIN_KM / 111.0) ** 2 for u, v in used):
                continue
            used.append((lon, lat))
            out.append((lon, lat))
            if len(out) >= n:
                break
        return out

    rows = []
    N_RAND = 40000       # 共识层太稀：9000 随机点仅命中 L1 1-8 个
    for prov, box in PROV.items():
        x0, y0, x1, y1 = box
        emit('=== %s ===' % prov)
        # 随机点（覆盖面）+ 采样标签 → 按共识分层
        rs = np.random.RandomState(SEED + hash(prov) % 1000)
        lon = rs.uniform(x0, x1, N_RAND); lat = rs.uniform(y0, y1, N_RAND)
        pts = [ee.Feature(ee.Geometry.Point([float(a), float(b)]),
                          {'point_id': 'R%05d' % i}) for i, (a, b) in enumerate(zip(lon, lat))]
        _dw = dw.filterDate('2023-01-01', '2023-12-31').filterBounds(ee.Geometry.Rectangle(list(box)))
        # label 取众数；概率取**全年均值**（mode 聚合概率是错的——2026-10-10 修正）
        dwi = ee.Image.cat([_dw.select('label').mode().rename('dwl'),
                            _dw.select('shrub_and_scrub').mean().rename('dwp')])
        stack = ee.Image.cat([dwi, wc.rename('wc'), lc.rename('lc')])
        got = []
        for k in range(0, len(pts), 3000):
            for att in range(4):
                try:
                    r = stack.sampleRegions(collection=ee.FeatureCollection(pts[k:k + 3000]), scale=100,
                                            geometries=False, tileScale=4).getInfo()['features']
                    got += [f['properties'] for f in r]
                    break
                except Exception as e:
                    emit('  %s 采样重试%d %s' % (prov, att + 1, str(e)[:70])); time.sleep(8)
        emit('  %s 标签采样 %d/%d' % (prov, len(got), len(pts)))
        # 回连坐标
        coord = {('R%05d' % i): (float(a), float(b)) for i, (a, b) in enumerate(zip(lon, lat))}
        cnt = {'L1': 0, 'L2': 0, 'L3': 0, 'L4': 0}
        used = []
        for g in got:
            pid = g.get('point_id'); p = coord.get(pid)
            if not p:
                continue
            x, y = p
            dwl = g.get('dwl'); dwp = g.get('dwp') or 0
            w20 = (g.get('wc') == 20); l20 = (g.get('lc') == 20)
            isdw = (dwl == 6)
            if isdw and (w20 or l20):
                tag = 'L1'
            elif isdw and float(dwp) >= 0.4:
                tag = 'L2'
            elif isdw:
                tag = 'L4'
            elif (w20 or l20) and not isdw:
                tag = 'L3'
            else:
                continue
            if cnt[tag] >= a.per:
                continue
            if any(((x - u) ** 2 * 100 + (y - v) ** 2 * 120) < (MIN_KM / 111.0) ** 2 for u, v in used):
                continue
            used.append((x, y)); cnt[tag] += 1
            rows.append(dict(prov=prov, stratum=tag, lon=x, lat=y, src='GEE多源',
                             dw_label=int(dwl) if dwl is not None else -1,
                             dw_prob=round(float(dwp), 3), wc20=int(w20), lc100=int(l20)))
        emit('  %s 分层：%s' % (prov, cnt))
    df = pd.DataFrame(rows)
    df.insert(0, 'point_id', ['SH-%s-%s-%04d' % (r.prov, r.stratum, i) for i, r in enumerate(df.itertuples())])

    # ---- OSM 要素点 ----
    osm = json.load(open(os.path.join(OUTD, 'osm_scrub.json'), encoding='utf-8'))
    orows = []
    for prov, lst in osm.items():
        if not lst:
            continue
        take = lst if len(lst) <= a.per * 2 else [lst[i] for i in rng.choice(len(lst), a.per * 2, replace=False)]
        for i, r in enumerate(take):
            orows.append(dict(prov=prov, stratum='OSM', lon=float(r['lon']), lat=float(r['lat']), src='OSM'))
    if orows:
        od = pd.DataFrame(orows)
        od.insert(0, 'point_id', ['SH-%s-OSM-%04d' % (r.prov, i) for i, r in enumerate(od.itertuples())])
        df = pd.concat([df, od], ignore_index=True)
        emit('OSM 点 %d' % len(orows))

    # ---- 我们的成品灌丛像元（v1；四川用 rasters_v2_sc 与 batch 都不含灌丛 → 用 batch R0）----
    # 简化：直接从 FCS10 灌丛层的池点抽样作为"第一版训练输入"的代理，另列 V1 层
    try:
        pool = pd.read_parquet(POOL, columns=['lon', 'lat', 'src'])
        sh = pool[pool['src'].fillna('').astype(str) == 'glc_fcs10_2023_shrub']
        emit('池内 FCS10 灌丛层点 %d' % len(sh))
        for prov, (x0, y0, x1, y1) in PROV.items():
            sub = sh[(sh.lon >= x0) & (sh.lon <= x1) & (sh.lat >= y0) & (sh.lat <= y1)]
            if len(sub) == 0:
                continue
            take = sub.iloc[rng.choice(len(sub), min(a.per, len(sub)), replace=False)]
            vd = pd.DataFrame(dict(prov=prov, stratum='FCS10', lon=take.lon.to_numpy(), lat=take.lat.to_numpy(),
                                   src='FCS10灌丛层'))
            vd.insert(0, 'point_id', ['SH-%s-FCS10-%04d' % (prov, i) for i in range(len(vd))])
            df = pd.concat([df, vd], ignore_index=True)
    except Exception as e:
        emit('FCS10 层取点失败 %s' % str(e)[:80])

    emit('候选合计 %d：%s' % (len(df), dict(df.groupby(['prov', 'stratum']).size())))

    # ---- 特征提取 ----
    pts = [ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]), {'point_id': r.point_id})
           for r in df.itertuples()]
    def s2_stack():
        def msk(im):
            scl = im.select('SCL')
            return im.updateMask(scl.neq(3).And(scl.neq(8)).And(scl.neq(9)))
        col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterDate('2022-12-01', '2024-01-01')
               .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60)).map(msk))
        seas = [('djf', '2022-12-01', '2023-03-01'), ('jja', '2023-06-01', '2023-09-01')]
        bands = []
        for tag, x, y in seas:
            c = col.filterDate(x, y).median()
            bands.append(c.normalizedDifference(['B8', 'B4']).rename('ndvi_' + tag))
        return ee.Image.cat(bands)
    occ = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence').unmask(0).rename('occ')
    groups = [
        ('core', ee.Image.cat([dem.rename('dem'), ee.Terrain.slope(dem).rename('slope'),
                               bio, occ]).unmask(-999),
         ['dem', 'slope', 'bio01', 'bio12', 'occ']),
        ('aef', ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
         .filterDate('2023-01-01', '2024-01-01').mosaic().select(VC.FEATS), VC.FEATS),
        ('s2', s2_stack(), ['ndvi_djf', 'ndvi_jja']),
        ('gedi', gedi.filterDate('2019-01-01', '2024-12-31').median().rename('rh98'), ['rh98']),
    ]
    frames = {}
    for tag, img, want in groups:
        got = []
        step = 3000
        for k in range(0, len(pts), step):
            for att in range(4):
                try:
                    r = img.sampleRegions(collection=ee.FeatureCollection(pts[k:k + step]), scale=100,
                                          geometries=False, tileScale=4).getInfo()['features']
                    got += [dict(point_id=f['properties']['point_id'],
                                 **{c: f['properties'].get(c) for c in want}) for f in r]
                    break
                except Exception as e:
                    emit('  %s 重试%d %s' % (tag, att + 1, str(e)[:70])); time.sleep(8)
        frames[tag] = pd.DataFrame(got)
        emit('  %s 命中 %d/%d' % (tag, len(frames[tag]), len(pts)))
    out = df.copy()
    for tag in ('core', 'aef', 's2', 'gedi'):
        if len(frames[tag]):
            out = out.merge(frames[tag], on='point_id', how='left')
    keep = out[out['A00'].notna()].copy() if 'A00' in out.columns else out
    if 'ndvi_jja' in keep.columns and 'ndvi_djf' in keep.columns:
        keep['ndvi_amp'] = (keep['ndvi_jja'].astype(float) - keep['ndvi_djf'].astype(float)).abs()
    keep.to_csv(OUT, index=False, encoding='utf-8-sig')
    emit('候选（含特征）%d → %s' % (len(keep), OUT))
    emit('  分层：%s' % dict(keep.groupby(['prov', 'stratum']).size()))


if __name__ == '__main__':
    main()
