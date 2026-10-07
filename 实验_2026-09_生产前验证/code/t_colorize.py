# -*- coding: utf-8 -*-
"""t_colorize.py — 给 10m 分类 GeoTIFF 写入颜色表 + 金字塔，并输出编码↔颜色↔地类对照文档
产物（每瓦片）：
  <tile>_10m.tif       uint8 + 内置调色板（colormap）+ 概览层（2/4/8/16/32）
  <tile>_10m.tif.aux.xml   （GDAL 自动元数据，若需要）
  <tile>.clr           QGIS/ArcGIS 通用色标文件（value R G B）
全局：
  地类颜色表.csv / 地类颜色对照.md   编码-名称-颜色(hex/RGB) 对照
"""
import os, sys, csv
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v31_common as VC
import numpy as np
import rasterio

SRC = os.path.join(VC.RES, 'tiles')                    # 工作副本（uint16 原件）
DST = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/试点成果_2023瓦片'
TILES = ['qinling', 'qinling_e', 'songnen', 'sanjiang']
sys.path.insert(0, os.path.join(VC.ROOT, 'code'))
from d_tiles import COLORS

OVR = [2, 4, 8, 16, 32]


def palette():
    """256 项 RGBA 字典（rasterio 要求 dict）：0=透明；1–24=地类色；其余=浅灰"""
    pal = {i: (230, 230, 230, 0) for i in range(256)}
    for k, c in COLORS.items():
        pal[k] = (int(round(c[0] * 255)), int(round(c[1] * 255)), int(round(c[2] * 255)), 255)
    for k in range(25, 256):
        pal[k] = (200, 200, 200, 255)
    return pal


def convert(t):
    src_fp = os.path.join(SRC, '%s_10m.tif' % t)
    dst_fp = os.path.join(DST, '%s_10m.tif' % t)
    tmp_fp = os.path.join(DST, '%s_10m.tmp.tif' % t)
    if not os.path.exists(src_fp):
        VC.emit('%s 缺源文件' % t); return
    with rasterio.open(src_fp) as ds:
        prof = ds.profile.copy()
        prof.update(dtype='uint8', compress='deflate', predictor=2, tiled=True,
                    blockxsize=512, blockysize=512, count=1, nodata=None)
        with rasterio.open(tmp_fp, 'w', **prof) as dst:
            for r0 in range(0, ds.height, 1024):
                hh = min(1024, ds.height - r0)
                blk = ds.read(1, window=rasterio.windows.Window(0, r0, ds.width, hh))
                blk = np.where(blk > 250, 0, blk).astype('uint8')
                dst.write(blk, 1, window=rasterio.windows.Window(0, r0, ds.width, hh))
            dst.write_colormap(1, palette())
            dst.build_overviews(OVR, rasterio.enums.Resampling.nearest)
            dst.update_tags(ns='rio_overview', resampling='nearest')
    os.replace(tmp_fp, dst_fp)
    # QGIS/ArcGIS 色标文件
    with open(os.path.join(DST, '%s.clr' % t), 'w', encoding='utf-8') as f:
        f.write('# QGIS/ArcGIS color file: value R G B (0-255)\n')
        for k, c in sorted(COLORS.items()):
            f.write('%d %d %d %d\n' % (k, round(c[0] * 255), round(c[1] * 255), round(c[2] * 255)))
    sz = os.path.getsize(dst_fp) / 1e6
    with rasterio.open(dst_fp) as chk:
        cm = chk.colormap(1)
        ovr = chk.overviews(1)
    VC.emit('%s ✅ %.0f MB | 调色板 %d 项 | 概览层 %s' % (t, sz, len(cm), ovr))


def docs():
    names = VC.V31_NAMES()
    rows = []
    for k in sorted(COLORS):
        c = COLORS[k]
        r, g, b = [round(x * 255) for x in c]
        rows.append(dict(code=k, name=names.get(str(k), '?'), hex='#%02X%02X%02X' % (r, g, b),
                         R=r, G=g, B=b))
    csv_fp = os.path.join(DST, '地类颜色表.csv')
    with open(csv_fp, 'w', newline='', encoding='utf-8-sig') as f:
        wr = csv.DictWriter(f, fieldnames=['code', 'name', 'hex', 'R', 'G', 'B'])
        wr.writeheader(); wr.writerows(rows)
    with open(os.path.join(DST, '地类颜色表.clr'), 'w', encoding='utf-8') as f:
        f.write('# QGIS/ArcGIS color file: value R G B (0-255)\n')
        for r_ in rows:
            f.write('%d %d %d %d\n' % (r_['code'], r_['R'], r_['G'], r_['B']))
    L = ['# 试点瓦片地类编码 · 颜色 · 名称对照表', '',
         '> 适用文件：`试点成果_2023瓦片\\{tile}_10m.tif`（uint8，内置调色板 + 金字塔概览层 2/4/8/16/32）',
         '> 像元值 = 下表 `编码`；**0 = 无数据（调色板中设为透明）**；25–255 未使用（显示为浅灰）', '',
         '| 编码 | 地类名称 | 颜色 | HEX | RGB(0-255) |', '|---|---|---|---|---|']
    for r_ in rows:
        L.append('| %d | %s | ██ | `%s` | %d, %d, %d |' % (r_['code'], r_['name'], r_['hex'], r_['R'], r_['G'], r_['B']))
    L += ['', '## 怎么用', '',
          '- **QGIS**：直接拖入 `xxx_10m.tif` → 自动按内置调色板显示为彩色；若想手动套色，加载同目录 `xxx.clr`（图层属性 → 符号化 → 调色板/唯一值 → 导入色标）。',
          '- **ArcGIS Pro**：拖入后若未自动识别彩色，图层属性 → 符号系统 → 色彩映射表(Colormap) → 导入 `地类颜色表.clr`。',
          '- **Python/rasterio**：`ds.read(1)` 得到的就是上表编码；`ds.colormap(1)` 可读出调色板。',
          '- **金字塔**：文件已内置 2/4/8/16/32 倍概览层（最近邻生成，不改变类码）→ 缩放到全国视图时不卡、且不会出现"凭空造类"的重采样伪影。',
          '- 注意：**不要对类别栅格使用双线性/立方重采样**（会产生不存在的类），需要重采样时一律选"最近邻"。']
    fp = os.path.join(DST, '地类颜色对照.md')
    open(fp, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    VC.emit('对照文档 → %s / 地类颜色表.csv / 地类颜色表.clr' % fp)


def main():
    os.makedirs(DST, exist_ok=True)
    for t in TILES:
        try:
            convert(t)
        except Exception as e:
            VC.emit('%s 转换失败: %s' % (t, str(e)[:120]))
    docs()


if __name__ == '__main__':
    main()
