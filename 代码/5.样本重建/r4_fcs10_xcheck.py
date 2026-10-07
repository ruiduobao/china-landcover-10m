# -*- coding: utf-8 -*-
"""
r4_fcs10_xcheck.py — 盲区类点位的 GLC_FCS10 2023 交叉验证（10m 同源第二教师）
* 目的：解决 120/121 等的空间截断（产品覆盖不均 + 盲区保底规则造成）
* 方法：对盲区类（120/121/91/92/140/180-186）池内全部点，读取 FCS10 2023 raw 码，
        按类映射到产品码后与本点标签比对 → 一致/不一致/nodata 三态
* 输出: r4_fcs10_xcheck.parquet（row_id, fcs10_raw, fcs10_cls, match）
"""
import os, re, sys, glob, json, time
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_conf as C

BLIND = {120, 121, 180, 181, 182, 183, 184, 185, 186, 140, 91, 92}
OUT = os.path.join(C.WORK, 'r4_fcs10_xcheck.parquet')

FMAP = np.full(256, -1, dtype=np.int16)
for a, b in C.CODE_MAP.items():
    if 0 <= a < 256:
        FMAP[a] = b
for a, b in C.FCS10_EXTRA.items():
    FMAP[a] = b

def china_tiles():
    out = []
    for fp in glob.glob(os.path.join(C.FCS10_TILE_ROOT, 'GLC_FCS10maps_*', '*.tif')):
        m = re.search(r'_E(\d+)N(\d+)\.tif$', os.path.basename(fp))
        if not m:
            continue
        lon0, lat0 = int(m.group(1)), int(m.group(2))
        b = C.BBOX
        if lon0 + 5 > b[0] and lon0 < b[2] and lat0 > b[1] and lat0 - 5 < b[3]:
            out.append((fp, lon0, lat0))
    return out

def main():
    t0 = time.time()
    pool = pd.read_parquet(os.path.join(C.WORK, 'r2_backup', 'r1_pool_备份_20260908.parquet'),
                           columns=['lon', 'lat', 'class_new'])
    m = pool.class_new.isin(BLIND).to_numpy()
    pts = pool[m].reset_index(drop=True)
    pts['pid'] = np.arange(len(pts))
    print(f'盲区类池内点: {len(pts):,}', flush=True)
    lon = pts.lon.to_numpy(); lat = pts.lat.to_numpy()
    # 1° 格分组
    ck = np.floor(lon).astype(np.int64) * 1000 + np.floor(lat).astype(np.int64)
    order = np.argsort(ck, kind='stable')
    sk = ck[order]
    b = np.flatnonzero(np.r_[True, sk[1:] != sk[:-1], True])
    cells = {int(sk[b[i]]): order[b[i]:b[i + 1]] for i in range(len(b) - 1)}
    print(f'涉及 1° 格: {len(cells):,}', flush=True)

    raw = np.full(len(pts), 255, dtype=np.uint8)
    tiles = china_tiles()
    print(f'FCS10 瓦片: {len(tiles)}', flush=True)
    n_done = 0
    for fp, lon0, lat0 in tiles:
        try:
            with rasterio.open(fp) as s:
                inv = ~s.transform
                for cx in range(max(int(np.floor(C.BBOX[0])), lon0), min(int(np.ceil(C.BBOX[2])), lon0 + 5)):
                    for cy in range(max(int(np.floor(C.BBOX[1])), lat0 - 5), min(int(np.ceil(C.BBOX[3])), lat0)):
                        dk = cx * 1000 + cy
                        idx = cells.get(dk)
                        if idx is None:
                            continue
                        cc0, rr0 = [int(v) for v in inv * (float(cx), float(cy + 1))]
                        cc1, rr1 = [int(v) for v in inv * (float(cx + 1), float(cy))]
                        cc0 = max(0, cc0); rr0 = max(0, rr0)
                        cc1 = min(s.width, cc1); rr1 = min(s.height, rr1)
                        if cc1 <= cc0 or rr1 <= rr0:
                            continue
                        a = s.read(1, window=Window(cc0, rr0, cc1 - cc0, rr1 - rr0))
                        if a.ndim == 3:
                            a = a[0]
                        px, py = ~s.transform * (lon[idx], lat[idx])
                        rr = np.clip(py.astype(int) - rr0, 0, a.shape[0] - 1)
                        cc = np.clip(px.astype(int) - cc0, 0, a.shape[1] - 1)
                        raw[idx] = a[rr, cc].astype(np.uint8)
                        n_done += len(idx)
        except Exception as e:
            print('ERR', os.path.basename(fp), str(e)[:80], flush=True)
        if (tiles.index((fp, lon0, lat0)) + 1) % 10 == 0:
            print(f'  瓦片 {tiles.index((fp,lon0,lat0))+1}/{len(tiles)} 已回读点 {n_done:,}', flush=True)

    pts['fcs10_raw'] = raw
    pts['fcs10_cls'] = FMAP[raw]
    pts['match'] = np.where(raw == 255, 'nodata',
                            np.where(pts.fcs10_cls == pts.class_new, 'same', 'diff'))
    pts.to_parquet(OUT, index=False)
    print('\n=== FCS10 交叉验证结果 ===', flush=True)
    for c in sorted(BLIND):
        s = pts[pts.class_new == c]
        if len(s) == 0:
            continue
        vc = s.match.value_counts()
        print(f'类{c}: 池 {len(s):>7,} | same {vc.get("same",0):>7,} '
              f'({vc.get("same",0)/len(s):.1%}) | diff {vc.get("diff",0):>7,} | nodata {vc.get("nodata",0):>7,}', flush=True)
    print(f'({time.time()-t0:.0f}s) → {OUT}', flush=True)

if __name__ == '__main__':
    main()
