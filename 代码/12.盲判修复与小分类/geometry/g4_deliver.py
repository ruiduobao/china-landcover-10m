# -*- coding: utf-8 -*-
"""g4_deliver.py — 三省几何复验单波段交付件：photo=PALETTE 修正 + 存储字节验收（纯本地，无 GEE）

* 问题：单波段分类图若 `PhotometricInterpretation=1`（MinIsBlack），部分读取器/QGIS 报告“无调色板”。
* 规则源：doc72 §8「新单band photo=PALETTE，诊断多band不palette」；doc70 §4.4；doc43（概览纪律）。
* 做法：**只读源文件** → 复制到 `delivery/` 新文件名，写入 `photometric='PALETTE'` + 原 colormap；
  逐像元校验数据一致（numpy）+ **PIL 忠实解码**（mode=P、palette 存在）+ tifffile 读回 photo=3；
  **不覆盖任何原件、不改其 hash**。
* 输出：`delivery/TRI_<tag>_<arm>_10m_palette.tif` + `delivery/MANIFEST_MD5.txt`（源与交付双 hash）
* 用法：python g4_deliver.py
"""
import glob
import hashlib
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
WS = r'F:/lc_work/label_repair_2023_20261015/geometry'
OUTD = os.path.join(WS, 'out')
DEL = os.path.join(WS, 'delivery')


def md5(fp):
    h = hashlib.md5()
    with open(fp, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 20), b''):
            h.update(ch)
    return h.hexdigest()


def main():
    import numpy as np
    import rasterio
    import tifffile
    from PIL import Image
    os.makedirs(DEL, exist_ok=True)
    rows, checks = [], []
    srcs = sorted(glob.glob(os.path.join(OUTD, 'TRI_*_legacy_10m.tif'))) + \
        sorted(glob.glob(os.path.join(OUTD, 'TRI_*_planar_10m.tif')))
    for src in srcs:
        base = os.path.basename(src)
        dst = os.path.join(DEL, base.replace('_10m.tif', '_10m_palette.tif'))
        with rasterio.open(src) as ds:
            assert ds.count == 1 and ds.dtypes[0] == 'uint8', '%s 非单波段 uint8' % base
            arr = ds.read()
            prof = ds.profile.copy()
            cm = ds.colormap(1)
            src_photo = int(tifffile.TiffFile(src).pages[0].tags['PhotometricInterpretation'].value)
        prof.update(photometric='PALETTE')
        with rasterio.open(dst, 'w', **prof) as ds:
            ds.write(arr)
            ds.write_colormap(1, cm)
        with rasterio.open(dst) as ds2:
            same = bool((ds2.read() == arr).all())
            cm2 = ds2.colormap(1)
        dst_photo = int(tifffile.TiffFile(dst).pages[0].tags['PhotometricInterpretation'].value)
        im = Image.open(dst)
        pil_ok = bool(im.mode == 'P' and im.palette)
        pil_same = bool((np.asarray(im) == arr[0]).all())
        rows.append((base, md5(src), os.path.basename(dst), md5(dst),
                     'src_photo=%d' % src_photo, 'dst_photo=%d' % dst_photo,
                     'data_same=%s' % same, 'cm=%d' % len(cm2),
                     'pil_mode=%s' % im.mode, 'pil_palette=%s' % bool(im.palette),
                     'pil_pixels_same=%s' % pil_same))
        checks.append(dict(src=base, dst=os.path.basename(dst), src_photo=src_photo,
                           dst_photo=dst_photo, data_same=same, cm_len=len(cm2),
                           pil_mode=im.mode, pil_palette=bool(im.palette), pil_pixels_same=pil_same,
                           src_md5=md5(src), dst_md5=md5(dst)))
        print('%-34s photo %d->%d data_same=%s cm=%d PIL=%s/%s pixels_same=%s'
              % (base, src_photo, dst_photo, same, len(cm2), im.mode, bool(im.palette), pil_same))
    with open(os.path.join(DEL, 'MANIFEST_MD5.txt'), 'w', encoding='utf-8') as f:
        f.write('# 三省几何复验单波段交付（photometric=PALETTE）；源文件未改动\n')
        f.write('# src_name\tsrc_md5\tdeliver_name\tdeliver_md5\tnotes\n')
        for r in rows:
            f.write('\t'.join(r) + '\n')
    ok = all(c['dst_photo'] == 3 and c['data_same'] and c['pil_mode'] == 'P' and
             c['pil_palette'] and c['pil_pixels_same'] for c in checks)
    json.dump(dict(schema='tri_geom_delivery/v1', n=len(checks), all_pass=bool(ok), files=checks),
              open(os.path.join(DEL, 'DELIVERY_CHECK.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('\n交付 %d 个 → %s ｜ ALL_PASS=%s' % (len(rows), DEL, ok))
    print('清单 → %s' % os.path.join(DEL, 'MANIFEST_MD5.txt'))


if __name__ == '__main__':
    main()
