# -*- coding: utf-8 -*-
"""f14_sc_v2_stats.py — 四川 v2 成品统计（面积/构成 + 与 R0 交付品对比 + 灌丛消失量化）

* 输入：rasters_v2_sc/<tile>_10m.tif（v2 新品）、rasters_batch/<tile>_10m.tif（R0 交付品）
        PROJ/数据/边界/china_100000_full.json（省界 buffer -0.005°）
* 规则源：面积按 10 m 原生网格 + 纬度改正；nearest 不重采样
* 输出：data/eco_gate/sc_v2/{sc_v2_area.csv, sc_v2_vs_r0.csv, sc_v2_stats.json}
* 用法：python f14_sc_v2_stats.py
"""
import csv, json, math, os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
sys.path.insert(0, os.path.join(WORK, 'code'))
import numpy as np, rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
import v31_common as VC
from e2_sichuan_v_mask import load_tiles

V2 = r'F:/lc_work/prod5p_2023/rasters_v2_sc'
R0 = r'F:/lc_work/prod5p_2023/rasters_batch'
OUTD = os.path.join(WORK, 'data', 'eco_gate', 'sc_v2')
BOUND = os.path.join(PROJ, '数据', '边界', 'china_100000_full.json')
BLK = 4096
NAMES = VC.V31_NAMES()

def emit(m): print(time.strftime('[%m-%d %H:%M:%S] ')+str(m), flush=True)

def geom():
    from shapely.geometry import shape
    from shapely import make_valid
    gj = json.load(open(BOUND, encoding='utf-8'))
    for ft in gj['features']:
        if ft.get('properties', {}).get('name', '').startswith('四川'):
            return make_valid(shape(ft['geometry'])).buffer(-0.005)

def main():
    os.makedirs(OUTD, exist_ok=True)
    G = geom()
    tiles = load_tiles()
    acc = {'v2': {}, 'r0': {}}
    for t in tiles:
        for tag, d in (('v2', V2), ('r0', R0)):
            fp = os.path.join(d, '%s_10m.tif' % t)
            if not os.path.exists(fp):
                emit('%s %s 缺文件' % (tag, t)); continue
            with rasterio.open(fp) as ds:
                h, w, res, tfm = ds.height, ds.width, ds.res, ds.transform
                inside = rasterize([(G, 1)], out_shape=(h, w), transform=tfm, fill=0, dtype='uint8')
                for r0 in range(0, h, BLK):
                    hh = min(BLK, h - r0)
                    cb = ds.read(1, window=Window(0, r0, w, hh))
                    ib = inside[r0:r0 + hh]
                    lat0 = ds.xy(r0, 0, offset='ul')[1]
                    lat1 = ds.xy(r0 + hh - 1, 0, offset='ul')[1]
                    cell = (res[0]*111.32*math.cos(math.radians((lat0+lat1)/2)))*(abs(res[1])*110.57)
                    sel = (cb > 0) & (ib == 1)
                    v, n = np.unique(cb[sel], return_counts=True)
                    for c, k in zip(v.tolist(), n.tolist()):
                        acc[tag][c] = acc[tag].get(c, 0.0) + k*cell
        emit('%s 完成' % t)
    rows = []
    for tag in ('v2', 'r0'):
        tot = sum(acc[tag].values())
        for c in sorted(acc[tag], key=lambda x: -acc[tag][x]):
            rows.append(dict(strategy=tag, code=c, name=NAMES.get(str(c), ''), km2=round(acc[tag][c], 1),
                             pct=round(100*acc[tag][c]/max(tot,1e-9), 4)))
    with open(os.path.join(OUTD, 'sc_v2_area.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['strategy','code','name','km2','pct']); w.writeheader(); w.writerows(rows)
    codes = sorted(set(list(acc['v2'])+list(acc['r0'])), key=lambda c: -(acc['v2'].get(c,0)+acc['r0'].get(c,0)))
    cmp_rows = [dict(code=c, name=NAMES.get(str(c),''), r0_km2=round(acc['r0'].get(c,0),1),
                     v2_km2=round(acc['v2'].get(c,0),1),
                     delta_km2=round(acc['v2'].get(c,0)-acc['r0'].get(c,0),1)) for c in codes]
    with open(os.path.join(OUTD, 'sc_v2_vs_r0.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['code','name','r0_km2','v2_km2','delta_km2']); w.writeheader(); w.writerows(cmp_rows)
    tot_v2, tot_r0 = sum(acc['v2'].values()), sum(acc['r0'].values())
    out = dict(area_v2_km2=round(tot_v2,1), area_r0_km2=round(tot_r0,1),
               shrub_v2=round(acc['v2'].get(10,0)+acc['v2'].get(9,0),1),
               shrub_r0=round(acc['r0'].get(10,0)+acc['r0'].get(9,0),1),
               wetland_v2=round(sum(acc['v2'].get(c,0) for c in (14,15,16,17)),1),
               wetland_r0=round(sum(acc['r0'].get(c,0) for c in (14,15,16,17)),1),
               per_class=cmp_rows)
    VC.jsave(out, os.path.join(OUTD, 'sc_v2_stats.json'))
    emit('四川（省界内）v2 %.0f km² ｜ R0 %.0f km²' % (tot_v2, tot_r0))
    emit('  灌丛: R0 %.1f → v2 %.1f km² ｜ 湿地四类: %.1f → %.1f' % (
        out['shrub_r0'], out['shrub_v2'], out['wetland_r0'], out['wetland_v2']))
    for r in cmp_rows[:10]:
        emit('  %-8s R0 %8.1f → v2 %8.1f (%+.1f km²)' % (r['name'], r['r0_km2'], r['v2_km2'], r['delta_km2']))
    emit('→ %s' % OUTD)

if __name__ == '__main__': main()
