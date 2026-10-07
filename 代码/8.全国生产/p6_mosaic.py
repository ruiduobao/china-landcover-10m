# -*- coding: utf-8 -*-
"""
p6_mosaic.py — 分区拼接与年度质检（全国 10m 成果按"瓦片集合"交付，另出概览与逐类面积）
设计取舍：
  * 10 m 全国单文件 = 272 Gpx（≈272 GB uint8），单文件既不可行也不实用；
    GLC_FCS10 本身也是**按瓦片分发**。因此成品 = 逐瓦片 COG + 瓦片索引（STAC 风格 JSON）。
  * 逐类面积**不用概览图统计**，而是直接对 10 m 瓦片逐块精确计数（100 m²/像元），避免小类被稀释。
  * 另出全国概览 COG（默认 0.005°≈500 m）供目视与 QA。
输出（Z:/生产输出/）：
  栅格/<year>/<tile>.tif（COG 化拷贝）、栅格/<year>/mosaic_index.json、
  栅格/<year>/overview_<year>.tif、元数据/area_<year>.csv、元数据/qa_<year>.json
用法：python p6_mosaic.py --year 2022 [--overview-deg 0.005] [--no-copy]
"""
import os, sys, io, json, glob, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '0.本地流水线'))
import prod_conf as C
from lc_conf import CLASSES

sys.stdout.reconfigure(encoding='utf-8')


def emit(m):
    print(time.strftime('[%H:%M:%S] ') + str(m), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--year', type=int, required=True)
    ap.add_argument('--overview-deg', type=float, default=0.005)
    ap.add_argument('--no-copy', action='store_true')
    ap.add_argument('--stage', default=C.STAGE)
    a = ap.parse_args()
    import rasterio
    from rasterio.shutil import copy as rio_copy

    src_dir = os.path.join(a.stage, str(a.year))
    files = [f for f in sorted(glob.glob(os.path.join(src_dir, '*_%d.tif' % a.year)))
             if not os.path.basename(f).startswith('probe_')]
    if not files:
        raise SystemExit('暂存目录没有 %d 年的瓦片: %s' % (a.year, src_dir))
    out_dir = os.path.join(C.RASTER, str(a.year))
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(C.LEDGER, exist_ok=True)
    emit('年 %d：%d 个瓦片待拼接' % (a.year, len(files)))

    x0, y0, x1, y1 = C.BBOX
    d = a.overview_deg
    W = int(round((x1 - x0) / d)); H = int(round((y1 - y0) / d))
    ov = np.zeros((H, W), np.uint8)
    counts = np.zeros(256, np.int64)
    area_km2_by_class = np.zeros(256, np.float64)   # 按真实像元面积累加（不假设 10m）
    index = []
    for fp in files:
        with rasterio.open(fp) as s:
            arr = s.read(1)
            tr = s.transform
            counts += np.bincount(arr.ravel(), minlength=256)
            # 像元面积按该瓦片的 transform 实算（度 → 米，经向随纬度收缩）
            lat_mid = tr.f + s.height * tr.e / 2.0
            px_m = abs(tr.a) * 111320.0 * np.cos(np.radians(lat_mid))
            py_m = abs(tr.e) * 110540.0
            px_km2 = px_m * py_m / 1e6
            area_km2_by_class += np.bincount(arr.ravel(), minlength=256) * px_km2
            # 概览：按整像素映射（nearest），保证类别码不被插值破坏
            c0 = int(round((tr.c - x0) / d)); r0 = int(round((y1 - (tr.f + s.height * tr.e)) / d))
            hh = max(1, int(round(s.height * abs(tr.e) / d)))
            ww = max(1, int(round(s.width * abs(tr.a) / d)))
            k = max(1, int(round(1.0 / (abs(tr.e) / d))))
            small = arr[::k, ::k]
            if r0 < 0 or c0 < 0 or r0 + small.shape[0] > H or c0 + small.shape[1] > W:
                emit('  %s 越界，裁剪写入' % os.path.basename(fp))
                r1 = min(H, r0 + small.shape[0]); c1 = min(W, c0 + small.shape[1])
                rr0 = max(0, r0); cc0 = max(0, c0)
                ov[rr0:r1, cc0:c1] = small[rr0 - r0:r1 - r0, cc0 - c0:c1 - c0]
            else:
                ov[r0:r0 + small.shape[0], c0:c0 + small.shape[1]] = small
            index.append({'tile': os.path.basename(fp).split('_')[0],
                          'file': os.path.basename(fp),
                          'bounds': [tr.c, tr.f + s.height * tr.e, tr.c + s.width * tr.a, tr.f],
                          'size': [s.width, s.height], 'crs': str(s.crs)})
            if not a.no_copy:
                dst = os.path.join(out_dir, os.path.basename(fp))
                if not os.path.isfile(dst):
                    try:
                        rio_copy(fp, dst, driver='COG', COMPRESS='DEFLATE',
                                 NUM_THREADS='ALL_CPUS')
                    except Exception:
                        rio_copy(fp, dst)
        emit('  + %s 概览写入完成' % os.path.basename(fp))

    # 概览落盘
    ov_fp = os.path.join(out_dir, 'overview_%d.tif' % a.year)
    from rasterio.transform import from_origin
    with rasterio.open(ov_fp, 'w', driver='GTiff', height=H, width=W, count=1,
                       dtype='uint8', crs='EPSG:4326',
                       transform=from_origin(x0, y1, d, d), compress='deflate') as dst:
        dst.write(ov, 1)
    land = int((ov > 0).sum())
    emit('概览 %s（%.1f Mpx，陆域填充 %.1f%%）' % (ov_fp, H * W / 1e6, 100.0 * land / ov.size))

    # 逐类面积：10 m 像元 100 m² → km²
    rows = []
    tot = int(counts[1:].sum())
    for c in sorted(CLASSES):
        n = int(counts[c])
        if n:
            rows.append({'class': c, 'name': CLASSES[c][1], 'n_px': n,
                         'area_km2': round(float(area_km2_by_class[c]), 3),
                         'pct': round(100.0 * n / tot, 4)})
    import pandas as pd
    df = pd.DataFrame(rows).sort_values('area_km2', ascending=False)
    df.to_csv(os.path.join(C.LEDGER, 'area_%d.csv' % a.year), index=False,
              encoding='utf-8-sig')
    qa = {'year': a.year, 'n_tiles': len(files), 'n_classes_nonzero': len(rows),
          'pixel_area_source': 'raster transform（非硬编码 10m）',
          'total_km2': round(float(df.area_km2.sum()), 1),
          'missing_classes': [c for c in sorted(CLASSES) if c not in set(df['class'])],
          'min_class_km2': float(df.area_km2.min()) if len(df) else None,
          'overview': ov_fp, 'overview_deg': d,
          'time': time.strftime('%Y-%m-%d %H:%M:%S')}
    json.dump({'year': a.year, 'bbox': C.BBOX, 'tile_deg': C.TILE_DEG,
               'n_tiles': len(index), 'tiles': index},
              open(os.path.join(out_dir, 'mosaic_index.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    json.dump(qa, open(os.path.join(C.LEDGER, 'qa_%d.json' % a.year), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    emit('逐类面积 %d 类非零 / 共 %d 类；缺失类 %s' % (len(rows), len(CLASSES),
                                                qa['missing_classes'] or '无'))
    print(df.head(12).to_string(index=False))
    print('输出:', out_dir)


if __name__ == '__main__':
    main()
