# -*- coding: utf-8 -*-
"""m1_mangrove_scope.py — 红树林扩充 第1步：产品范围 → 500 m 粗矢量 → 候选瓦片清单

* 源：ESA/WorldCover/v200 Map==95（红树林）∪ LANDSAT/MANGROVE_FORESTS==1（Giri 2000）
      **逐源 unmask 后再 Or**（联合掩膜陷阱）。
* 产出：data/mangrove/scope/polys_500m.geojson（粗范围多边形）+ tiles_0p25.json（覆盖瓦片清单）
* 用法：python m1_mangrove_scope.py --acct <acct>
"""
import argparse, json, os, sys, time
import ee

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'F:/lc_work/v31_exp/code')
WORK = r'F:/lc_work/v31_exp/data/mangrove'
BOX = [106.0, 17.0, 125.5, 31.5]          # 中国沿海框（含台湾/浙闽）


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acct', required=True)
    a = ap.parse_args()
    import v31_common as VC
    VC.ensure_ctx(a.acct)
    os.makedirs(os.path.join(WORK, 'scope'), exist_ok=True)
    wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map')
    giri = ee.ImageCollection('LANDSAT/MANGROVE_FORESTS').first().select('1')
    mask = wc.eq(95).unmask(0).Or(giri.eq(1).unmask(0)).toByte().rename('mang')
    box = ee.Geometry.Rectangle(BOX)
    emit('AOI %s' % BOX)
    n_cells = mask.reduceRegion(ee.Reducer.sum(), box, 500, maxPixels=1e10, bestEffort=True).getInfo()
    emit('500m 像元计数 %s' % n_cells)
    m500 = mask.reduceResolution(ee.Reducer.max(), bestEffort=True).reproject('EPSG:4326', None, 500)
    polys = m500.selfMask().reduceToVectors(geometry=box, scale=500, geometryType='polygon',
                                            eightConnected=False, labelProperty='mang',
                                            maxPixels=1e10, bestEffort=True, tileScale=4)
    n = polys.size().getInfo()
    emit('粗多边形 %d 个' % n)
    # 下载（<5000 上限；否则提高聚合）
    gj = polys.getInfo()
    out = os.path.join(WORK, 'scope', 'polys_500m.geojson')
    json.dump(gj, open(out, 'w', encoding='utf-8'))
    emit('→ %s（%d 要素）' % (out, len(gj['features'])))
    # 瓦片清单（0.25°；对每个多边形 bbox 打格）
    tiles = set()
    for f in gj['features']:
        g = f['geometry']
        xs, ys = [], []
        rings = [g['coordinates']] if g['type'] == 'Polygon' else g['coordinates']
        for poly in rings:
            for ring in poly:
                for x, y in ring:
                    xs.append(x); ys.append(y)
        if not xs:
            continue
        for x in range(int(min(xs) // 0.25) - 1, int(max(xs) // 0.25) + 2):
            for y in range(int(ys[0] // 0.25) - 1, int(max(ys) // 0.25) + 2):
                tiles.add((round(x * 0.25, 2), round(y * 0.25, 2)))
    tl = sorted(tiles)
    json.dump(dict(box=BOX, tile_deg=0.25, tiles=[[x, y] for x, y in tl],
                   n_polys=len(gj['features']), n_tiles=len(tl)),
              open(os.path.join(WORK, 'scope', 'tiles_0p25.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    emit('瓦片 %d 个 → scope/tiles_0p25.json' % len(tl))


if __name__ == '__main__':
    main()
