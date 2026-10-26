# -*- coding: utf-8 -*-
"""g2_deliver_palette.py — 单波段分类图「调色板可被读取」交付修正（纯本地小图加工，无 GEE）

* 问题（audit 发现）：`G30N_*_{legacy,planar}_10m.tif` 实际**含 ColorMap 标签（256 项）**，
  但 `PhotometricInterpretation=1`（MinIsBlack）而非 `3`（Palette）→ 部分读取器/QGIS 报告“无调色板”。
* 规则源：doc70 §4.4「类别栅格用最近邻、uint8、nodata=0、正确码表」；doc43（概览只走 ov_build+PIL）。
* 做法：**只读源文件** → 复制到 `delivery/` 新文件名，写入时 `photometric='PALETTE'` + 原 colormap；
  逐像元校验数据一致 + 校验 Photometric=3 + colormap 256 项；**不覆盖任何原件、不改其 hash**。
* 输出：`delivery/G30N_<tag>_<arm>_10m_palette.tif` + `delivery/MANIFEST_MD5.txt`（含源与交付双 hash）
* 用法：python g2_deliver_palette.py
"""
import glob
import hashlib
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, 'out')
DEL = os.path.join(HERE, 'delivery')


def md5(fp):
    h = hashlib.md5()
    with open(fp, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 20), b''):
            h.update(ch)
    return h.hexdigest()


def main():
    import rasterio
    import tifffile
    os.makedirs(DEL, exist_ok=True)
    rows = []
    srcs = sorted(glob.glob(os.path.join(OUTD, 'G30N_*_legacy_10m.tif'))) + \
        sorted(glob.glob(os.path.join(OUTD, 'G30N_*_planar_10m.tif')))
    for src in srcs:
        base = os.path.basename(src)
        dst = os.path.join(DEL, base.replace('_10m.tif', '_10m_palette.tif'))
        with rasterio.open(src) as ds:
            assert ds.count == 1 and ds.dtypes[0] == 'uint8', '%s 非单波段 uint8' % base
            arr = ds.read()
            prof = ds.profile.copy()
            cm = ds.colormap(1)
            src_photo = tifffile.TiffFile(src).pages[0].tags['PhotometricInterpretation'].value
        prof.update(photometric='PALETTE')
        with rasterio.open(dst, 'w', **prof) as ds:
            ds.write(arr)
            ds.write_colormap(1, cm)
        with rasterio.open(dst) as ds2:
            same = bool((ds2.read() == arr).all())
            cm2 = ds2.colormap(1)
        photo = tifffile.TiffFile(dst).pages[0].tags['PhotometricInterpretation'].value
        rows.append((base, md5(src), os.path.basename(dst), md5(dst),
                     'src_photo=%d' % src_photo, 'dst_photo=%d' % photo,
                     'data_same=%s' % same, 'cm=%d' % len(cm2)))
        print('%-34s photo %d->%d  data_same=%s  cm=%d' % (base, src_photo, photo, same, len(cm2)))
    with open(os.path.join(DEL, 'MANIFEST_MD5.txt'), 'w', encoding='utf-8') as f:
        f.write('# G30N 单波段分类图交付（photometric=PALETTE 修正版）；源文件未改动\n')
        f.write('# src_name\tsrc_md5\tdeliver_name\tdeliver_md5\tnotes\n')
        for r in rows:
            f.write('\t'.join(r) + '\n')
    print('\n交付 %d 个 → %s' % (len(rows), DEL))
    print('清单 → %s' % os.path.join(DEL, 'MANIFEST_MD5.txt'))


if __name__ == '__main__':
    main()
