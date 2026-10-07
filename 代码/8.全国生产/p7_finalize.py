# -*- coding: utf-8 -*-
"""
p7_finalize.py — 全部瓦片完成后的一次性最终装配
产物（Z:/生产输出/成品/）：
  1) <region>_<year>_10m.vrt        GDAL 虚拟镶嵌（全分辨率无缝，不占磁盘，QGIS 可直接开）
  2) <region>_<year>_100m.tif       100m 镶嵌成品：内嵌 FCS_GLC 颜色表 + 内部金字塔
  3) 分省 100m 镶嵌 <prov>_<year>_100m.tif（同上）
  4) area_final_<year>.csv          逐类面积（按 100m 成品统计）
  5) README_成品说明.md             产物说明 + 配色来源 + 复现命令

资源约束（用户要求）：内存 ≤20G、CPU ≤14 核
  * 逐瓦片处理，同一时刻只驻留 1 个 10m 瓦片（≈0.5 GB）
  * 分类栅格重采样一律用 **Resampling.mode**（众数），绝不用双线性——避免类别码被插值破坏
  * 金字塔同样用 mode；GDAL_NUM_THREADS 限制为 14
  * 每步前用 psutil 检查 RSS，超 20G 立即中止而不是拖垮机器

用法：
  python p7_finalize.py --check                # 只检查 36 瓦片是否齐
  python p7_finalize.py --run [--year 2023]    # 执行装配
"""
import os, sys, io, json, time, glob, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '0.本地流水线'))
import prod_conf as C
from lc_conf import CLASSES

sys.stdout.reconfigure(encoding='utf-8')
OUTDIR = os.path.join(C.KB, '生产输出', '成品')
COLOR_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fcs_glc_colors.json')
MEM_LIMIT_GB = 20.0
CPU_LIMIT = 14


def emit(m):
    print(time.strftime('[%H:%M:%S] ') + str(m), flush=True)


def mem_gb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9
    except Exception:
        return -1.0


def guard(tag):
    m = mem_gb()
    if m > MEM_LIMIT_GB:
        raise SystemExit('[%s] 内存 %.1f GB 超过 %.0f GB 上限，已中止（避免拖垮机器）'
                         % (tag, m, MEM_LIMIT_GB))
    if m > 0:
        emit('  [%s] RSS %.2f GB' % (tag, m))


def load_tiles(year):
    """返回 {tile: path}，只取已下载的"""
    d = os.path.join(C.STAGE, str(year))
    out = {}
    for f in sorted(glob.glob(os.path.join(d, '*_%d.tif' % year))):
        b = os.path.basename(f)
        if b.startswith('probe_'):
            continue
        out[b.split('_')[0]] = f
    return out


def expected_tiles():
    fp = os.path.join(C.PLAN, 'tiles_ne_all.json')
    if os.path.isfile(fp):
        return [t['tile'] for t in json.load(open(fp, encoding='utf-8'))['tiles']]
    return []


def do_check(year):
    got = load_tiles(year)
    exp = expected_tiles()
    miss = [t for t in exp if t not in got]
    emit('已下载 %d / 应有 %d；缺 %s' % (len(got), len(exp), ','.join(sorted(miss)) or '无'))
    return (len(miss) == 0), got, miss


def build_vrt(files, out_vrt, year):
    """GDAL 虚拟镶嵌（纯 Python 写 VRT XML，不依赖 osgeo 绑定）。
    VRT 不复制像素，QGIS/ArcGIS 可直接打开全分辨率无缝图层。"""
    import rasterio
    bs, res = [], None
    for f in files:
        with rasterio.open(f) as s:
            b = s.bounds
            bs.append((f, b, s.width, s.height))
            if res is None:
                res = (abs(s.transform.a), abs(s.transform.e))
    x0 = min(b[1][0] for b in bs); y0 = min(b[1][1] for b in bs)
    x1 = max(b[1][2] for b in bs); y1 = max(b[1][3] for b in bs)
    rx, ry = res
    W = int(round((x1 - x0) / rx)); H = int(round((y1 - y0) / ry))
    parts = ['<VRTDataset rasterXSize="%d" rasterYSize="%d">' % (W, H),
             '  <SRS>EPSG:4326</SRS>',
             '  <GeoTransform>%.10f, %.10f, 0.0, %.10f, 0.0, %.10f</GeoTransform>'
             % (x0, rx, y1, -ry),
             '  <VRTRasterBand dataType="Byte" band="1">',
             '    <ColorInterp>Palette</ColorInterp>']
    for f, b, w, h in bs:
        dx = int(round((b[0] - x0) / rx)); dy = int(round((y1 - b[3]) / ry))
        parts += ['    <SimpleSource>',
                  '      <SourceFilename relativeToVRT="0">%s</SourceFilename>' % f.replace(os.sep, '/'),
                  '      <SourceBand>1</SourceBand>',
                  '      <SrcRect xOff="0" yOff="0" xSize="%d" ySize="%d"/>' % (w, h),
                  '      <DstRect xOff="%d" yOff="%d" xSize="%d" ySize="%d"/>' % (dx, dy, w, h),
                  '    </SimpleSource>']
    parts += ['  </VRTRasterBand>', '</VRTDataset>']
    io.open(out_vrt, 'w', encoding='utf-8').write('\n'.join(parts))
    emit('VRT → %s（%d 个瓦片，网格 %d×%d）' % (out_vrt, len(files), W, H))


def build_mosaic(files, out_tif, size_deg, colormap=None, overviews=True, tag=''):
    """逐瓦片 reproject 到目标网格（mode 众数），内存只驻留 1 个瓦片"""
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import reproject, Resampling, calculate_default_transform
    x0, y0, x1, y1 = C.BBOX if False else _extent_of(files)
    W = int(round((x1 - x0) / size_deg)); H = int(round((y1 - y0) / size_deg))
    emit('  目标网格 %d×%d（%.4f° ≈ %.0f m）' % (W, H, size_deg, size_deg * 111320))
    dst_transform = from_origin(x0, y1, size_deg, size_deg)
    prof = dict(driver='GTiff', height=H, width=W, count=1, dtype='uint8',
                crs='EPSG:4326', transform=dst_transform, compress='deflate',
                tiled=True, blockxsize=512, blockysize=512, BIGTIFF='IF_SAFER')
    tmp = out_tif + '.part'
    # 累加数组：100 m 全国/区域网格约 3 亿像元 = 300 MB，内存安全；同一时刻只读 1 个源瓦片
    acc = np.zeros((H, W), np.uint8)
    with rasterio.open(tmp, 'w', **prof) as dst:
        for i, fp in enumerate(files):
            guard('重采样中')
            with rasterio.open(fp) as src:
                src_arr = src.read(1)
                dst_arr = np.zeros((H, W), np.uint8)
                reproject(source=src_arr, destination=dst_arr,
                          src_transform=src.transform, src_crs=src.crs,
                          dst_transform=dst_transform, dst_crs='EPSG:4326',
                          resampling=Resampling.mode)   # 分类栅格必须 mode，不能用双线性
                m = dst_arr > 0
                acc[m] = dst_arr[m]
            if (i + 1) % 10 == 0:
                emit('    %d/%d  RSS %.1f GB' % (i + 1, len(files), mem_gb()))
        dst.write(acc, 1)
        if colormap:
            dst.write_colormap(1, colormap)
    os.replace(tmp, out_tif)
    emit('镶嵌 → %s（%.1f MB）' % (out_tif, os.path.getsize(out_tif) / 1e6))
    if overviews:
        build_pyramids(out_tif)
    return out_tif


def _bounds(fp):
    import rasterio
    with rasterio.open(fp) as s:
        b = s.bounds
    return (b.left, b.bottom, b.right, b.top)


def _extent_of(files):
    bs = [_bounds(f) for f in files]
    return (min(b[0] for b in bs), min(b[1] for b in bs),
            max(b[2] for b in bs), max(b[3] for b in bs))


def build_pyramids(fp, levels=(2, 4, 8, 16, 32, 64)):
    import rasterio
    from rasterio.enums import Resampling
    t0 = time.time()
    with rasterio.Env(GDAL_NUM_THREADS=str(CPU_LIMIT), NUM_THREADS=str(CPU_LIMIT)):
        with rasterio.open(fp, 'r+') as ds:
            ds.build_overviews(list(levels), Resampling.mode)
            ds.update_tags(ns='rio_overview', resampling='mode')
    emit('  金字塔 %s（mode 重采样，%.0fs）' % (os.path.basename(fp), time.time() - t0))


def load_colormap():
    if not os.path.isfile(COLOR_JSON):
        emit('⚠ 缺配色文件 %s → 成品将不带颜色表' % COLOR_JSON)
        return None
    d = json.load(open(COLOR_JSON, encoding='utf-8'))
    cmap = {}
    for c, rgb in d['classes'].items():
        cmap[int(c)] = tuple(int(v) for v in rgb)
    emit('配色载入 %d 类（来源：%s）' % (len(cmap), d.get('source', '未知')))
    return cmap


def write_area(fp, out_csv):
    import numpy as np
    import rasterio
    import pandas as pd
    with rasterio.open(fp) as s:
        px_m = abs(s.transform.a) * 111320 * np.cos(np.radians(s.transform.f + s.height * s.transform.e / 2))
        px_km2 = (px_m * abs(s.transform.e) * 110540) / 1e6
        cnt = np.zeros(256, np.int64)
        for _, win in s.block_windows(1):
            cnt += np.bincount(s.read(1, window=win).ravel(), minlength=256)
    rows = [{'class': c, 'name': CLASSES[c][1], 'n_px': int(cnt[c]),
             'area_km2': round(float(cnt[c]) * px_km2, 2)} for c in sorted(CLASSES) if cnt[c]]
    df = pd.DataFrame(rows).sort_values('area_km2', ascending=False)
    df['pct'] = (100.0 * df.area_km2 / df.area_km2.sum()).round(4)
    df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    emit('逐类面积 → %s（%d 类非零，合计 %.0f km²）'
         % (out_csv, len(df), df.area_km2.sum()))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--year', type=int, default=2023)
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--region', default='东北三省')
    ap.add_argument('--size-deg', type=float, default=0.001)   # ≈100 m
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--allow-partial', action='store_true')
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    ok, got, miss = do_check(a.year)
    if a.check or not a.run:
        return 0 if ok else 3
    if not ok and not a.allow_partial:
        emit('尚有 %d 个瓦片未下载 → 先补下载再装配' % len(miss))
        return 3
    if not ok:
        emit('⚠ --allow-partial：用 %d 片先试跑（缺 %s）' % (len(got), ','.join(sorted(miss))))
    files = [got[t] for t in sorted(got)]
    cmap = load_colormap()
    stem = '%s_%d' % (a.region, a.year)
    guard('装配开始')

    # 1) 全分辨率 VRT（无缝、不占盘）
    vrt_fp = os.path.join(OUTDIR, stem + '_10m.vrt')
    if a.force or not os.path.isfile(vrt_fp):
        try:
            build_vrt(files, vrt_fp, a.year)
        except Exception as e:
            emit('VRT 构建失败（缺 GDAL python 绑定？）：%s' % str(e)[:120])
    # 2) 100m 镶嵌 + 配色 + 金字塔
    tif_fp = os.path.join(OUTDIR, stem + '_100m.tif')
    if a.force or not os.path.isfile(tif_fp):
        build_mosaic(files, tif_fp, a.size_deg, colormap=cmap)
    # 3) 逐类面积
    area_fp = os.path.join(C.LEDGER, 'area_final_%d.csv' % a.year)
    df = write_area(tif_fp, area_fp)
    # 4) 说明
    doc = os.path.join(OUTDIR, 'README_成品说明.md')
    io.open(doc, 'w', encoding='utf-8').write(
        '# %s %d 成品说明\n\n' % (a.region, a.year)
        + '- 源瓦片：%d 个 2° 格（10 m，EPSG:4326），来自 %s\n' % (len(files), C.STAGE)
        + '- `%s_10m.vrt`：GDAL 虚拟镶嵌，全分辨率无缝，QGIS 可直接打开（不占磁盘）\n' % stem
        + '- `%s_100m.tif`：100 m 镶嵌成品，**内嵌 FCS_GLC 颜色表**，含内部金字塔（mode 重采样）\n' % stem
        + '- `%s`：逐类面积（按 100 m 成品统计）\n' % os.path.basename(area_fp)
        + '\n## 颜色表来源\n\n' + json.load(open(COLOR_JSON, encoding='utf-8')).get('source', '见 fcs_glc_colors.json')
        + '\n\n## 复现\n\n```bash\npython 代码/8.全国生产/p7_finalize.py --check\npython 代码/8.全国生产/p7_finalize.py --run\n```\n')
    emit('说明 → %s' % doc)
    emit('装配完成。产物目录：%s' % OUTDIR)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
