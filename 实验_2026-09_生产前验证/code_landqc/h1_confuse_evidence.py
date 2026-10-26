# -*- coding: utf-8 -*-
"""h1_confuse_evidence.py — 混淆对定向修复的证据抓取（S1 雷达 + DW + S2；按类组）

* 目标混淆对（来自大区可分离度报告）与所需"新"证据：
    P1 草本沼泽 ↔ 湖河滩地      ：植被有无（S2 NDVI + DW grass/trees/shrub/bare）——湖河滩地已有 2023 证据，只需补沼泽
    P2 草本沼泽 ↔ 木本沼泽      ：**SAR 结构**（S1 VV/VH 夏/冬中位）+ DW trees/shrub
    P3 灌木 ↔ 草地 / 落叶阔叶林  ：DW shrub/trees/grass + SAR VH（木质结构）
    P4 乔灌园地 ↔ 耕地/常绿阔叶林：DW crops/trees/bare + S2 NDVI（地块格局类只能删极端）
    其余对（裸地↔稀疏植被、灌溉↔旱地、水体↔湖河滩地）用既有 2023 证据，不另抓。
* 波段（14）：s1_vv_jja/s1_vh_jja/s1_vv_djf/s1_vh_djf（dB，median 合成）、
    dw_water/dw_trees/dw_grass/dw_crops/dw_shrub/dw_bare（2023 均值）、dw_trees_150（150 m 邻域）、
    s2_ndvi_xia/s2_ndvi_dong/s2_ndwi_xia（×1000）
* 输入：--group swamp|shrub|orchard（对应工作区点表或交付 GPKG）
* 分片：--part/--nparts；块 2000（子请求 1000）；幂等
* 用法：python h1_confuse_evidence.py --group swamp --part 0 --nparts 4 --acct <acct> --outdir ../data/confuse_fix/feats/swamp
"""
import argparse, glob, json, os, sys, time
import numpy as np
import pandas as pd
import geopandas as gpd

sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
ROOT = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12'
BASE = os.path.join(WORK, 'data', 'confuse_fix')
SENT = -9999.0
COLS = ['s1_vv_jja', 's1_vh_jja', 's1_vv_djf', 's1_vh_djf',
        'dw_water', 'dw_trees', 'dw_grass', 'dw_crops', 'dw_shrub', 'dw_bare',
        'dw_trees_150', 's2_ndvi_xia', 's2_ndvi_dong', 's2_ndwi_xia']
SRC = {
    'swamp': [('木本沼泽', '木本沼泽/2023/木本沼泽_2023_WGS84.gpkg', 'pid'),
              ('草本沼泽', '草本沼泽/2023/草本沼泽_2023_WGS84.gpkg', 'pid')],
    'shrub': [('灌木_常绿', '常绿灌木/2023/常绿灌木_2023_WGS84.gpkg', 'point_id'),
              ('灌木_落叶', '落叶灌木/2023/落叶灌木_2023_WGS84.gpkg', 'point_id')],
    'orchard': [('乔灌园地', '乔灌园地/2023/乔灌园地v7_2023_WGS84.gpkg', 'uid')],
}


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + 'P%d %s' % (os.getpid(), str(m)), flush=True)


def build(ee):
    s1 = ee.ImageCollection('COPERNICUS/S1_GRD').filter(ee.Filter.eq('instrumentMode', 'IW'))
    def win(d0, d1):
        c = (s1.filterDate(d0, d1)
             .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
             .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')))
        return c.select(['VV', 'VH']).median()
    jja = win('2023-06-01', '2023-09-01').rename(['s1_vv_jja', 's1_vh_jja'])
    djf = win('2022-12-01', '2023-03-01').rename(['s1_vv_djf', 's1_vh_djf'])
    dw = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1').filterDate('2023-01-01', '2024-01-01')
          .select(['water', 'trees', 'grass', 'crops', 'shrub_and_scrub', 'bare']).mean()
          .rename(['dw_water', 'dw_trees', 'dw_grass', 'dw_crops', 'dw_shrub', 'dw_bare']))
    dw150 = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1').filterDate('2023-01-01', '2024-01-01')
             .select('trees').mean().unmask(0).focal_mean(radius=15, kernelType='circle').rename('dw_trees_150'))
    col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    def msk(im):
        s = im.select('SCL')
        return im.updateMask(s.neq(3).And(s.neq(8)).And(s.neq(9)))
    xia = (col.filterDate('2023-06-01', '2023-09-01').filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))
           .map(msk).median())
    dong = (col.filterDate('2022-12-01', '2023-02-01').filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))
            .map(msk).median())
    s2 = ee.Image.cat([xia.normalizedDifference(['B8', 'B4']).multiply(1000).unmask(SENT).rename('s2_ndvi_xia'),
                       dong.normalizedDifference(['B8', 'B4']).multiply(1000).unmask(SENT).rename('s2_ndvi_dong'),
                       xia.normalizedDifference(['B3', 'B8']).multiply(1000).unmask(SENT).rename('s2_ndwi_xia')]).toFloat()
    img = ee.Image.cat([jja.unmask(SENT), djf.unmask(SENT), dw.unmask(-1), dw150, s2]).toFloat()
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', required=True, choices=list(SRC))
    ap.add_argument('--part', type=int, required=True)
    ap.add_argument('--nparts', type=int, default=4)
    ap.add_argument('--acct', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--step', type=int, default=2000)
    ap.add_argument('--sub', type=int, default=1000)
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(WORK, 'code'))
    import ee
    import v31_common as VC
    VC.ensure_ctx(a.acct)
    pts = []
    for cls, path, idcol in SRC[a.group]:
        g = gpd.read_file(os.path.join(ROOT, path), columns=[idcol], engine='pyogrio')
        pts.append(pd.DataFrame(dict(point_id=g[idcol].astype(str), cls=cls,
                                     lon=g.geometry.x, lat=g.geometry.y)))
    d = pd.concat(pts, ignore_index=True)
    mine = d.iloc[a.part::a.nparts].reset_index(drop=True)
    os.makedirs(a.outdir, exist_ok=True)
    img = build(ee)
    emit('%s：本片 %d 点（%d 块）' % (a.group, len(mine), (len(mine) + a.step - 1) // a.step))
    for k in range(0, len(mine), a.step):
        fp = os.path.join(a.outdir, '%s_p%02d_c%04d.parquet' % (a.group, a.part, k // a.step))
        if os.path.exists(fp):
            continue
        blk = mine.iloc[k:k + a.step]
        rows, failed = [], False
        for s0 in range(0, len(blk), a.sub):
            sub = blk.iloc[s0:s0 + a.sub]
            fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(x), float(y)]), {'point_id': p})
                                       for p, x, y in zip(sub.point_id, sub.lon, sub.lat)])
            got, err = None, ''
            for att in range(6):
                try:
                    r = img.sampleRegions(collection=fc, scale=20, geometries=False,
                                          tileScale=4).getInfo()['features']
                    got = {f['properties']['point_id']: f['properties'] for f in r}; break
                except Exception as e:
                    err = str(e)[:110]
                    emit('  块%d 子%d 重试%d %s' % (k // a.step, s0 // a.sub, att + 1, err))
                    time.sleep(8 * (att + 1))
            if got is None:
                with open(os.path.join(a.outdir, '_fail_%s_p%02d.json' % (a.group, a.part)), 'a',
                          encoding='utf-8') as fh:
                    fh.write(json.dumps(dict(chunk=k // a.step, err=err), ensure_ascii=False) + '\n')
                failed = True; break
            rows += [dict(point_id=p, cls=c2, **{c: got.get(p, {}).get(c) for c in COLS})
                     for p, c2 in zip(sub.point_id, sub.cls)]
        if failed:
            continue
        o = pd.DataFrame(rows)
        for c in COLS:
            o[c] = pd.to_numeric(o[c], errors='coerce')
        for c in ('s1_vv_jja', 's1_vh_jja', 's1_vv_djf', 's1_vh_djf', 's2_ndvi_xia', 's2_ndvi_dong', 's2_ndwi_xia'):
            o.loc[o[c] <= -9000, c] = np.nan
        o.to_parquet(fp, index=False)
        emit('  %d 块完成' % (k // a.step + 1))
    emit('完成 %s p%02d' % (a.group, a.part))


if __name__ == '__main__':
    main()
