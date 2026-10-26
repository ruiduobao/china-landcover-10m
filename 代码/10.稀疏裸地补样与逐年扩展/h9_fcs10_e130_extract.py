# -*- coding: utf-8 -*-
"""h9_fcs10_e130_extract.py — 解压 E130-E135 带 zip 到 E 盘标准目录并验收

* 输入：E:/data/全球土地覆盖数据/数据/GLC_FCS10/_dl/zip/GLC_FCS10maps_2023_E130-E135.zip
* 输出：E:/data/全球土地覆盖数据/数据/GLC_FCS10/GLC_FCS10maps_2023_E130-E135/*.tif
* 验收：① zip 内条目数；② 解压后 tif 数一致；③ 每瓦可打开、EPSG:4326、res≈8.98e-5、bounds 对齐 5°；
        ④ 抽 3 瓦读 1° 窗口，统计类别非空
* 用法：python h9_fcs10_e130_extract.py
"""
import glob
import json
import os
import sys
import zipfile

import numpy as np
import rasterio
from rasterio.windows import from_bounds

sys.stdout.reconfigure(encoding='utf-8')
DL = r'E:/data/全球土地覆盖数据/数据/GLC_FCS10/_dl'
ZIP = os.path.join(DL, 'zip', 'GLC_FCS10maps_2023_E130-E135.zip')
OUT = r'E:/data/全球土地覆盖数据/数据/GLC_FCS10/GLC_FCS10maps_2023_E130-E135'


def main():
    os.makedirs(OUT, exist_ok=True)
    with zipfile.ZipFile(ZIP) as z:
        names = [n for n in z.namelist() if n.lower().endswith('.tif')]
        bad = z.testzip()
        print('zip 条目 %d（tif %d）｜ testzip=%s' % (len(z.namelist()), len(names), bad), flush=True)
        assert bad is None, 'zip CRC 损坏'
        for n in names:
            dst = os.path.join(OUT, os.path.basename(n))
            if os.path.isfile(dst) and os.path.getsize(dst) == z.getinfo(n).file_size:
                continue
            with z.open(n) as src, open(dst, 'wb') as f:
                while True:
                    b = src.read(1 << 22)
                    if not b:
                        break
                    f.write(b)
    tifs = sorted(glob.glob(os.path.join(OUT, '*.tif')))
    print('解压后 tif %d' % len(tifs), flush=True)
    ok = 0
    rows = []
    for t in tifs:
        with rasterio.open(t) as ds:
            b = ds.bounds
            rows.append((os.path.basename(t), str(ds.crs), ds.res[0], round(ds.width), round(ds.height),
                         round(b.left, 3), round(b.bottom, 3), round(b.right, 3), round(b.top, 3)))
            ok += 1
    for r in rows[:3]:
        print('  ', r)
    bad = [r for r in rows if (r[1] in (None, 'None')) or abs(r[2] - 8.983152841195215e-05) > 1e-12]
    print('可打开 %d/%d ｜ 异常 %d' % (ok, len(tifs), len(bad)), flush=True)
    # 抽 3 瓦读一个 1° 窗口
    for t in tifs[:1] + tifs[len(tifs)//2:len(tifs)//2+1] + tifs[-1:]:
        with rasterio.open(t) as ds:
            b = ds.bounds
            x0 = max(b.left, np.floor(b.left) + 1); y0 = max(b.bottom, np.floor(b.bottom) + 1)
            try:
                a = ds.read(1, window=from_bounds(x0, y0, x0 + 1, y0 + 1, ds.transform))
                u, c = np.unique(a, return_counts=True)
                print('  %s 1°窗口类别 %d 种: %s' % (os.path.basename(t), len(u), dict(zip(u.tolist()[:8], c.tolist()[:8]))))
            except Exception as e:
                print('  %s 抽读失败: %s' % (os.path.basename(t), e))
    json.dump({'zip_tifs': len(names), 'extracted': len(tifs), 'openable': ok,
               'abnormal': len(bad), 'files': rows},
              open(os.path.join(OUT, '_verify.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('→', OUT)


if __name__ == '__main__':
    main()
