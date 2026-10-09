# -*- coding: utf-8 -*-
"""n11_exp_features.py — 三个可分性实验统一特征取样（AEF 64 + S1 冬夏 + S2 四季指数 + DEM/坡度）

* 输入：data/m3/exp{A,B,C}_*.csv（point_id/lon/lat）
* 规则源（与生产/既往实验一致）：
  AEF 2023 = GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL 2023 mosaic，64 波段 A00–A63；
  S1 = COPERNICUS/S1_GRD IW VV+VH，冬(DJF: 2023-01-01~03-01)/夏(JJA: 06-01~09-01)中位 → s1_vv_w/vh_w/vv_s/vh_s；
  S2 = COPERNICUS/S2_SR_HARMONIZED，**SCL∉{3,8,9} 掩膜 + CLOUDY_PIXEL_PERCENTAGE<60**（与 m5a_materials 同口径；
  实测 QA60 掩膜/无云量过滤的中位合成在本机取样为空），四季(DJF=2022-12-01~2023-03-01/MAM/JJA/SON)中位 →
       ndvi/ndwi(B3,B8)/mndwi(B3,B11) ×4 = 12 波段；
  DEM = COPERNICUS/DEM/GLO30（ImageCollection→mosaic）高程 + 坡度。
* 门槛：**两次 sampleRegions**（AEF 单独；S1+S2+DEM 一组），本地按 point_id 合并；逐点须有 AEF 全 64 波段才算成功；
  S1/S2/DEM 缺失记 NaN（不阻断）。
* 输出：data/m3/exp{A,B,C}_features.csv（point_id,lon,lat + 64 + 4 + 12 + 2）
* 用法：python n11_exp_features.py --set A [--acct zixen8v8] [--force]
幂等：输出存在且行数一致则跳过。
"""
import csv
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import v31_common as VC

M3D = r'F:/lc_work/v31_exp/data/m3'
SETS = {'A': 'expA_forest_points.csv', 'B': 'expB_humid_shrub_points.csv', 'C': 'expC_wetland_points.csv',
        'A2': 'expA2_points.csv', 'C2': 'expC2_points.csv', 'C3': 'expC3_points.csv'}
AEF = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
YEAR = 2023
S1B = ['s1_vv_w', 's1_vh_w', 's1_vv_s', 's1_vh_s']
SEAS = [('djf', '2022-12-01', '2023-03-01'), ('mam', '2023-03-01', '2023-06-01'),
        ('jja', '2023-06-01', '2023-09-01'), ('son', '2023-09-01', '2023-12-01')]
S2B = ['%s_%s' % (k, s) for s, _, _ in SEAS for k in ('ndvi', 'ndwi', 'mndwi')]
AUXB = S1B + S2B + ['dem', 'slope']


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def s2_stack():
    import ee

    def msk(im):
        scl = im.select('SCL')
        return im.updateMask(scl.neq(3).And(scl.neq(8)).And(scl.neq(9)))
    col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterDate('2022-12-01', '2024-01-01')
           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60)).map(msk))
    bands = []
    for tag, a, b in SEAS:
        c = col.filterDate(a, b).median()
        bands += [c.normalizedDifference(['B8', 'B4']).rename('ndvi_' + tag),
                  c.normalizedDifference(['B3', 'B8']).rename('ndwi_' + tag),
                  c.normalizedDifference(['B3', 'B11']).rename('mndwi_' + tag)]
    return ee.Image.cat(bands)


def aef_image():
    import ee
    return (ee.ImageCollection(AEF).filterDate('%d-01-01' % YEAR, '%d-01-01' % (YEAR + 1))
            .mosaic().select(VC.FEATS))


def s1_image():
    import ee
    col = (ee.ImageCollection('COPERNICUS/S1_GRD')
           .filter(ee.Filter.eq('instrumentMode', 'IW'))
           .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
           .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
           .filterDate('2023-01-01', '2024-01-01').select(['VV', 'VH']))
    return ee.Image.cat([
        col.filterDate('2023-01-01', '2023-03-01').median().rename(['s1_vv_w', 's1_vh_w']),
        col.filterDate('2023-06-01', '2023-09-01').median().rename(['s1_vv_s', 's1_vh_s'])])


def s2dem_image():
    import ee
    dem = ee.ImageCollection('COPERNICUS/DEM/GLO30').select('DEM').mosaic().rename('dem')
    slope = ee.Terrain.slope(dem).rename('slope')
    return ee.Image.cat([s2_stack(), dem, slope])


def sample(ee, img, pts, want, tag):
    """sampleRegions + 重试；返回 list[dict]。"""
    for attempt in range(4):
        try:
            got = img.sampleRegions(collection=ee.FeatureCollection(pts), scale=10, geometries=False,
                                    tileScale=4).getInfo()['features']
            recs = [{**{k: f['properties'].get(k) for k in want}, 'point_id': f['properties']['point_id']}
                    for f in got]
            emit('  %s 取样 %d/%d' % (tag, len(recs), len(pts)))
            return recs
        except Exception as e:
            emit('  %s 重试 %d: %s' % (tag, attempt, str(e)[:150]))
            time.sleep(10)
    return []


def main():
    a = sys.argv[1:]
    key = a[a.index('--set') + 1].upper()
    acct = a[a.index('--acct') + 1] if '--acct' in a else 'zixen8v8'
    src = os.path.join(M3D, SETS[key])
    out = os.path.join(M3D, 'exp%s_features.csv' % key)
    rows = list(csv.DictReader(open(src, encoding='utf-8-sig')))
    if os.path.exists(out) and '--force' not in a:
        n = sum(1 for _ in csv.DictReader(open(out, encoding='utf-8-sig')))
        if n == len(rows):
            emit('%s 已存在（%d 行）→ 跳过' % (out, n))
            return
    import ee
    for attempt in range(3):
        try:
            VC.ensure_ctx(acct)
            break
        except Exception as e:
            emit('ctx 重试 %d: %s' % (attempt, str(e)[:120]))
            time.sleep(8)
    emit('账号 %s / 项目 %s；点 %d' % (acct, VC.pid_of(acct), len(rows)))
    pts = [ee.Feature(ee.Geometry.Point([float(r['lon']), float(r['lat'])]), {'point_id': r['point_id']})
           for r in rows]
    recs = sample(ee, aef_image(), pts, VC.FEATS, 'AEF')
    r1 = sample(ee, s1_image(), pts, S1B, 'S1')
    r2 = sample(ee, s2dem_image(), pts, S2B + ['dem', 'slope'], 'S2+DEM')
    da = pd.DataFrame(recs)
    db = pd.DataFrame(r1)
    dc = pd.DataFrame(r2)
    m = pd.DataFrame(rows)[['point_id', 'lon', 'lat']]
    df = m.merge(da, on='point_id', how='left').merge(db, on='point_id', how='left').merge(dc, on='point_id', how='left')
    df.to_csv(out, index=False, encoding='utf-8-sig')
    ok = df[VC.FEATS].notna().all(axis=1).sum()
    emit('AEF 成功 %d/%d；S1 完整 %d；S2 完整 %d；DEM %d → %s' % (
        ok, len(rows), df[S1B].notna().all(axis=1).sum(), df[S2B].notna().all(axis=1).sum(),
        df['dem'].notna().sum(), out))


if __name__ == '__main__':
    main()
