# -*- coding: utf-8 -*-
"""c5_mosaic_compare.py — 第二次分类（川/粤/宁 40 瓦）成品拼接 + 与第一次成品的栅格级对比

* 输入：
    第二次瓦片：F:/lc_work/prod2nd_2023/rasters_batch/{tile}_10m.tif（清单 plan/tiles_secondrun.csv：
                tile,ix,iy,x0,y0,x1,y1,land_frac,prov,prov_area_km2；瓦片=uint8/10m/EPSG:4326/nodata=0）
    第一次参照（**只读**）：D:/生产_2023五省/成品/分省/{四川,广东,宁夏}_2023_10m.tif（省口径成品，权威）
                F:/lc_work/prod5p_2023/mosaic/out/ 与 F:/lc_work/prod5p_2023/rasters_batch + pilot/delivery（瓦片口径）
                D:/生产_2023五省/成品/面积表/*.csv（第一次面积基准）
* 规则源：第一次 F:/lc_work/prod5p_2023/mosaic/{mosaic.py,clip_prov.py,p5common.py}
    —— 统一 10m 方度格 px=py=8.983152841195215e-05°、全局锚 (0,0) 同格零重采样直拷；
       重叠带规则=**先布放者胜**（后布放瓦片只在目标像元为 0 处填充，与 mosaic.py 逐行一致）；
       省口径=DataV 省界（GCJ-02）**原样**（第一次实测口径：不做 gcj2wgs 反演，--gcj 可切换）
       + buffer -0.005° 内缩（第一次交付口径）；像元中心点在多边形内→保留，否则 nodata=0
* 门槛：类别栅格只允许 nearest/同格直拷（禁止非最近邻重采样——60m 造类已证伪）；
       类码 >24 立即中止；过渡相位非整 → 显式 nearest 归格并告警（--strict-grid 可改判中止）；
       内存分块 ≤192MB、GDAL_CACHEMAX=512MB、串行；概览纪律（doc43）——默认不写内建概览，
       交付级概览必须 ov_build.py 重建 + PIL 全平面验收（脚本会打印命令）
* 输出：F:/lc_work/prod2nd_2023/mosaic2/     —— {tag}_瓦片合幅_10m.tif（总幅）、{交付短名}_{year}_10m.tif（省口径，
              命名与第一次交付一致：四川/广东/宁夏）
              + .clr + _plan.json + _qa.json + _preview.png（可选 {交付短名}_{year}_瓦片合幅_10m.tif，--prov-mosaics）
        F:/lc_work/prod2nd_2023/reports/mosaic2/ —— trans_{省}_24x24.csv、compare_{省}_class.csv、
              compare_{省}_run1_run2.json/.md、compare_all.md、qa_{省}_{label}.md/.csv/.json
* 用法：
    python 代码/9.第二次分类川粤宁/c5_mosaic_compare.py mosaic  --prov all
    python 代码/9.第二次分类川粤宁/c5_mosaic_compare.py mosaic  --tiles T1304,T1404 \
        --tiles-dir F:/lc_work/prod5p_2023/rasters_batch,F:/lc_work/prod5p_2023/pilot/delivery \
        --tag smoke_2t          # 冒烟（用第一次瓦片，只读）
    python 代码/9.第二次分类川粤宁/c5_mosaic_compare.py compare --prov all
    python 代码/9.第二次分类川粤宁/c5_mosaic_compare.py compare --prov 川 --second-file <tif> --first-file <tif>
    python 代码/9.第二次分类川粤宁/c5_mosaic_compare.py qa      --prov all
"""
import os
import sys
import json
import time
import math
import csv
import random
import argparse
import subprocess
import warnings

sys.stdout.reconfigure(encoding='utf-8')
sys.dont_write_bytecode = True            # 只读纪律：不往 prod5p/v31_exp 写 __pycache__
os.environ.setdefault('GDAL_CACHEMAX', '512')   # 内存纪律：GDAL 块缓存 512MB
# rasterio 读窗在本环境触发 numpy 2.5 DeprecationWarning（无害，第一次流水线同样存在）；仅静音该条
warnings.filterwarnings('ignore', message='Setting the shape on a NumPy array')

# ---------------- 路径常量（raw string 绝对路径，单点维护） ----------------
PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
ROOT1 = r'F:/lc_work/prod5p_2023'                     # 第一次（只读）
ROOT2 = r'F:/lc_work/prod2nd_2023'                    # 第二次
P5COMMON_DIR = os.path.join(ROOT1, 'mosaic')
V31_CODE = r'F:/lc_work/v31_exp/code'
sys.path.insert(0, P5COMMON_DIR)
sys.path.insert(0, V31_CODE)
import numpy as np                                        # noqa: E402
import rasterio                                           # noqa: E402
from rasterio.enums import Resampling                     # noqa: E402
from rasterio.transform import Affine                     # noqa: E402
from rasterio.windows import Window                       # noqa: E402
import p5common as PC                                     # noqa: E402

PLAN2 = os.path.join(ROOT2, 'plan', 'tiles_secondrun.csv')
TILES2_DEFAULT = os.path.join(ROOT2, 'rasters_batch')
MOSAIC2 = os.path.join(ROOT2, 'mosaic2')
REPORTS2 = os.path.join(ROOT2, 'reports', 'mosaic2')
OV_BUILD = os.path.join(ROOT2, 'run', 'ov_build.py')
# 第一次瓦片（冒烟/交叉核对用；T2102 只在 pilot/delivery）
FIRST_TILES_DEFAULT = os.path.join(ROOT1, 'rasters_batch') + ',' + \
    os.path.join(ROOT1, 'pilot', 'delivery')
# 第一次省口径成品候选目录（只读）
FIRST_PROD_DIRS = [r'D:/生产_2023五省/成品/分省',
                   os.path.join(ROOT1, 'mosaic', 'out')]
FIRST_AREA_DIR = r'D:/生产_2023五省/成品/面积表'
YEAR = 2023
PROV2 = ['川', '粤', '宁']           # 第二次 40 瓦覆盖的省（plan 口径）
# DataV 全名 → 交付短名（与第一次交付/D:/生产_2023五省/成品 的文件命名一致：四川/广东/宁夏）
DELIV_NAME = {'黑龙江省': '黑龙江', '四川省': '四川', '广东省': '广东',
              '宁夏回族自治区': '宁夏', '福建省': '福建'}
BUFFER_DEG = -0.005                  # 第一次交付口径：DataV 省界内缩 500m
CHUNK_BYTES = 192 * 1024 * 1024      # 单块缓冲上限（双栅格流式读，比单栅格 mosaic 减半）


def deliv_name(full):
    """DataV 全名 → 交付短名（命名与第一次交付一致，如 四川省→四川）。"""
    return DELIV_NAME.get(full, full)


def emit(m):
    print('[%s] %s' % (time.strftime('%m-%d %H:%M:%S'), m), flush=True)


def v31_names():
    """24 类码 → 类名（v31_exp 只读加载；失败退化为类码字符串）。"""
    try:
        import v31_common as VC
        return VC.V31_NAMES()
    except Exception:
        return {str(i): str(i) for i in range(1, 25)}


def jar_ok(v):
    """整数判断（相位）。"""
    return abs(v - round(v)) < 1e-4


# ======================================================================
# 一、计划表 / 瓦片校验 / 布放
# ======================================================================
def read_plan2(path=PLAN2):
    rows = []
    with open(path, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append(dict(tile=r['tile'], ix=int(r['ix']), iy=int(r['iy']),
                             x0=float(r['x0']), y0=float(r['y0']),
                             x1=float(r['x1']), y1=float(r['y1']),
                             land_frac=float(r['land_frac']), prov=r['prov'],
                             prov_area_km2=float(r.get('prov_area_km2', 0) or 0)))
    return rows


def find_tiles(dirs):
    """目录内 T####_10m.tif 严格匹配；同 tile 名先列目录优先（与 mosaic.py 同规则）。"""
    import glob
    import re
    pat = re.compile(r'^T\d{4}_10m\.tif$')
    out = {}
    for d in dirs.split(','):
        if not os.path.isdir(d.strip()):
            continue
        for fp in sorted(glob.glob(os.path.join(d.strip(), 'T*_10m.tif'))):
            if pat.match(os.path.basename(fp)):
                out.setdefault(os.path.basename(fp).split('_')[0], fp)
    return out


def validate_tile(fp, pinfo, strict_grid=False):
    """瓦片与计划/全局 10m 方度格一致性校验，返回实测网格参数（含相位）。"""
    with rasterio.open(fp) as ds:
        assert ds.crs and ds.crs.to_epsg() == 4326, '%s 非 EPSG:4326' % fp
        assert ds.dtypes[0] == 'uint8' and ds.count == 1, '%s 非 uint8 单波段' % fp
        assert ds.nodata == 0, '%s nodata!=0' % fp
        t = ds.transform
        px, py = t.a, -t.e
        assert abs(px - PC.EQ_DEG) / PC.EQ_DEG < 1e-9, '%s px=%.10e 偏离统一方度格' % (fp, px)
        assert abs(py - PC.EQ_DEG) / PC.EQ_DEG < 1e-9, '%s py=%.10e 偏离统一方度格' % (fp, py)
        span = ds.width * px
        assert 1.99 <= span <= 2.01, '%s 覆盖跨度 %.6f° 异常' % (fp, span)
        assert 22260 <= ds.width <= 22270 and 22260 <= ds.height <= 22270, \
            '%s 尺寸 %d×%d 异常（非 2°×2° 瓦片）' % (fp, ds.width, ds.height)
        # 全局格网相位：锚点 (0,0)、步长 EQ_DEG；非整相位不中止（退化为显式 nearest 归格）
        phx = t.c / PC.EQ_DEG - round(t.c / PC.EQ_DEG)
        phy = t.f / PC.EQ_DEG - round(t.f / PC.EQ_DEG)
        non_exact = not (abs(phx) < 1e-4 and abs(phy) < 1e-4)
        if non_exact:
            msg = ('%s 相位非整（x%+.6f / y%+.6f px）→ 该瓦片退化为显式 nearest 归格'
                   % (pinfo['tile'], phx, phy))
            if strict_grid:
                raise AssertionError('[strict-grid] ' + msg)
            emit('[警告] ' + msg)
        return dict(fp=fp, tile=pinfo['tile'], prov=pinfo['prov'],
                    x0=t.c, y1=t.f, px=px, py=py, width=ds.width, height=ds.height,
                    non_exact=non_exact, phase_x=round(phx, 6), phase_y=round(phy, 6))


def build_placements(tiles_meta, frame):
    """布放：框架已吸附全局格 ⇒ 同格瓦片偏移=整数像元，零重采样直拷；
    非整相位瓦片保留真实相位，读取时走显式 nearest（read_tile_into）。"""
    out = []
    for m in tiles_meta:
        fx = (m['x0'] - frame['x_left']) / frame['px']
        fy = (frame['y_top'] - m['y1']) / frame['py']
        # 与第一次 mosaic.py 同式：round 取整（实测瓦片相对框架的残差 ~1e-7 px，
        # floor+eps 会整体错 1 px——dry-frame 冒烟实测教训）
        row0 = int(round(fy))
        col0 = int(round(fx))
        exact = (not m['non_exact']) and jar_ok(fx) and jar_ok(fy)
        if (not exact) and (not m['non_exact']):
            # 罕见：瓦片自身在全局格上但相对框架相位非整（框架吸附异常）
            emit('[警告] %s 相对框架相位非整（x%.4f/y%.4f px）→ nearest 归格' % (m['tile'], fx, fy))
        out.append(dict(m, row0=row0, col0=col0, exact=exact,
                        phase_x=round(fx - col0, 6), phase_y=round(fy - row0, 6)))
    return out


def read_tile_into(src, p, frame, buf, r0, hh, c0, w):
    """把瓦片 p 在框架窗口 [r0,r0+hh) × [c0,c0+w) 的内容按归格映射读入 buf（就地）。

    - exact 瓦片：同格直拷（窗口直读，零重采样）；
    - 非整相位瓦片：像元中心 → 源行列显式 nearest 索引（唯一合法重采样）。
    写入规则与第一次 mosaic.py 完全一致：**先布放者胜**（仅在 buf==0 且源值>0 处写入）。
    返回本次写入的像元数。
    """
    ov_r0 = max(r0, p['row0'])
    ov_r1 = min(r0 + hh, p['row0'] + p['height'])
    ov_c0 = max(c0, p['col0'])
    ov_c1 = min(c0 + w, p['col0'] + p['width'])
    if ov_r1 <= ov_r0 or ov_c1 <= ov_c0:
        return 0
    if p['exact']:
        blk = src.read(1, window=Window(ov_c0 - p['col0'], ov_r0 - p['row0'],
                                        ov_c1 - ov_c0, ov_r1 - ov_r0))
    else:
        ar = np.arange(ov_r0, ov_r1, dtype=np.float64)
        ac = np.arange(ov_c0, ov_c1, dtype=np.float64)
        lat = frame['y_top'] - (ar + 0.5) * frame['py']
        lon = frame['x_left'] + (ac + 0.5) * frame['px']
        sr = np.floor((p['y1'] - lat) / p['py'] + 1e-7).astype(np.int64)
        sc = np.floor((lon - p['x0']) / p['px'] + 1e-7).astype(np.int64)
        ok_r = (sr >= 0) & (sr < p['height'])
        ok_c = (sc >= 0) & (sc < p['width'])
        if not ok_r.any() or not ok_c.any():
            return 0
        srr = np.clip(sr, 0, p['height'] - 1)
        scc = np.clip(sc, 0, p['width'] - 1)
        blk0 = src.read(1, window=Window(int(scc.min()), int(srr.min()),
                                         int(scc.max()) - int(scc.min()) + 1,
                                         int(srr.max()) - int(srr.min()) + 1))
        blk = blk0[np.ix_(srr - int(srr.min()), scc - int(scc.min()))]
        blk[~(ok_r[:, None] & ok_c[None, :])] = 0
    seg = buf[ov_r0 - r0:ov_r1 - r0, ov_c0 - c0:ov_c1 - c0]
    m = (seg == 0) & (blk > 0)
    n = int(m.sum())
    np.copyto(seg, blk, where=m)
    return n


# ======================================================================
# 二、合并（总幅 / 省瓦片合幅）
# ======================================================================
def merge_tiles(frame, placements, out_fp, ref_cmap, chunk_rows=0, ovr_mode='none'):
    """瓦片 → 合幅（10m 原生网格分块直拷；稀疏跳写空块；调色板/nodata 继承）。"""
    t0 = time.time()
    W, H = frame['ncols'], frame['nrows']
    if chunk_rows <= 0:
        chunk_rows = max(512, (CHUNK_BYTES // max(1, W)) // 512 * 512)
    prof = dict(driver='GTiff', dtype='uint8', count=1, nodata=0, crs='EPSG:4326',
                width=W, height=H,
                transform=Affine(frame['px'], 0, frame['x_left'],
                                 0, -frame['py'], frame['y_top']),
                compress='deflate', predictor=2, tiled=True,
                blockxsize=512, blockysize=512, BIGTIFF='IF_SAFER')
    class_px, class_area = {}, {}
    tile_px = {p['tile']: 0 for p in placements}
    tot_data, skipped = 0, 0
    handles = {p['tile']: rasterio.open(p['fp']) for p in placements}
    try:
        with rasterio.open(out_fp, 'w', **prof) as dst:
            for r0 in range(0, H, chunk_rows):
                r1 = min(H, r0 + chunk_rows)
                hh = r1 - r0
                buf = np.zeros((hh, W), dtype=np.uint8)
                touched = False
                for p in placements:
                    if p['row0'] >= r1 or p['row0'] + p['height'] <= r0:
                        continue
                    touched = True
                    tile_px[p['tile']] += read_tile_into(handles[p['tile']], p, frame,
                                                         buf, r0, hh, 0, W)
                if not touched:
                    skipped += 1
                    continue
                nz = buf > 0
                mx = int(buf.max())
                assert mx <= 24, '第 %d 行块出现非法类码 %d（>24，网格/读回异常）' % (r0, mx)
                tot_data += int(nz.sum())
                v, c = np.unique(buf[nz], return_counts=True)
                lat_mid = frame['y_top'] - (r0 + hh / 2.0) * frame['py']
                cell = PC.cell_m2(frame['px'], frame['py'], lat_mid)
                for aa, bb in zip(v.tolist(), c.tolist()):
                    class_px[aa] = class_px.get(aa, 0) + int(bb)
                    class_area[aa] = class_area.get(aa, 0.0) + int(bb) * cell
                dst.write(buf, 1, window=Window(0, r0, W, hh))
            if ref_cmap:
                dst.write_colormap(1, ref_cmap)
            if ovr_mode in ('scaffold', 'ovbuild'):
                # scaffold：先写内建概览页作脚手架（本环境 GDAL 概览页内容不可信，doc43；
                # 'ovbuild' 会随后由 ov_build.py 整页重建 + PIL 全平面验收）
                dst.build_overviews(PC.OVR_LEVELS, Resampling.nearest)
                dst.update_tags(ns='rio_overview', resampling='nearest')
    finally:
        for hd in handles.values():
            hd.close()
    return dict(class_px=class_px, class_area_m2=class_area, tile_px=tile_px,
                tot_data=tot_data, skipped_chunks=skipped,
                minutes=round((time.time() - t0) / 60, 2), chunk_rows=chunk_rows)


def qa_identity(out_fp, placements, frame, n_win=8, seed=7):
    """回读一致性抽检：随机窗口内重建"先布放者胜"期望值（含重叠带与非整相位瓦片），
    与成品逐像元比对——比第一次 mosaic.py 的逐瓦片窗口比对更严格（覆盖了瓦片重叠带）。"""
    rng = random.Random(seed)
    handles = {p['tile']: rasterio.open(p['fp']) for p in placements}
    res = []
    try:
        with rasterio.open(out_fp) as ds:
            for p in placements:
                wh = min(256, p['height'])
                ww = min(256, p['width'])
                bad = 0
                checked = 0
                for _ in range(n_win):
                    rr = p['row0'] + rng.randint(0, p['height'] - wh)
                    cc = p['col0'] + rng.randint(0, p['width'] - ww)
                    rr = max(0, min(frame['nrows'] - wh, rr))
                    cc = max(0, min(frame['ncols'] - ww, cc))
                    exp = np.zeros((wh, ww), dtype=np.uint8)
                    for q in placements:
                        read_tile_into(handles[q['tile']], q, frame, exp, rr, wh, cc, ww)
                    got = ds.read(1, window=Window(cc, rr, ww, wh))
                    bad += int((got != exp).sum())
                    checked += int(got.size)
                res.append(dict(tile=p['tile'], mode='exact' if p['exact'] else 'nearest',
                                windows=n_win, px_checked=checked, bad_px=bad, pass_=bad == 0))
    finally:
        for hd in handles.values():
            hd.close()
    return res


# ======================================================================
# 三、省口径裁剪（DataV 省界 GCJ-02 原样 + buffer -0.005°，与第一次 clip_prov.py 同式）
# ======================================================================
def _geom_to_wgs(g):
    """GCJ-02 多边形 → WGS84（逐个环 gcj2wgs 反演；--gcj gcj2wgs 时启用）。"""
    from shapely.geometry import Polygon, MultiPolygon
    def ring(cs):
        return [PC.gcj2wgs(x, y) for x, y in cs]
    if g.geom_type == 'Polygon':
        return Polygon(ring(g.exterior.coords), [ring(r.coords) for r in g.interiors])
    return MultiPolygon([_geom_to_wgs(p) for p in g.geoms])


def clip_prov(src_fp, prov_full, buf_deg, out_fp, gcj='raw', chunk_bytes=CHUNK_BYTES,
              ovr_mode='none'):
    """省界裁剪：像元中心点在（可选缓冲后的）省界多边形内 → 保留，否则 nodata=0；零重采样。
    与第一次 clip_prov.py 同式（含 bbox 窗口 -2/+3 取整口径），仅增加 --gcj 开关。"""
    from shapely.ops import unary_union
    from shapely import contains_xy, prepare
    geom = PC.prov_geom(prov_full)
    if gcj == 'gcj2wgs':
        geom = _geom_to_wgs(geom)
    if buf_deg:
        geom = geom.buffer(buf_deg)
        geom = unary_union([g for g in (geom.geoms if hasattr(geom, 'geoms') else [geom])
                            if g.geom_type in ('Polygon', 'MultiPolygon')])
        if not geom.is_valid:
            from shapely.validation import make_valid
            geom = make_valid(geom)
    prepare(geom)
    names = v31_names()
    short = PC.PROV_SHORT.get(prov_full, prov_full)
    t0 = time.time()
    with rasterio.open(src_fp) as ds:
        assert ds.dtypes[0] == 'uint8' and ds.nodata == 0, '输入须 uint8/nodata=0'
        assert ds.crs.to_epsg() == 4326, '输入须 EPSG:4326'
        cmap = ds.colormap(1)
        assert cmap, '输入缺内置调色板'
        minx, miny, maxx, maxy = geom.bounds
        if (ds.bounds.right < minx - 0.01 or ds.bounds.left > maxx + 0.01 or
                ds.bounds.top < miny - 0.01 or ds.bounds.bottom > maxy + 0.01):
            emit('%s 与 %s 无空间相交，跳过' % (os.path.basename(src_fp), prov_full))
            return None
        c0 = max(0, int((minx - ds.bounds.left) / ds.transform.a) - 2)
        c1 = min(ds.width, int((maxx - ds.bounds.left) / ds.transform.a) + 3)
        r0 = max(0, int((ds.bounds.top - maxy) / (-ds.transform.e)) - 2)
        r1 = min(ds.height, int((ds.bounds.top - miny) / (-ds.transform.e)) + 3)
        W, H = c1 - c0, r1 - r0
        emit('%s × %s：省界 bbox 内窗口 %d×%d（全幅 %d×%d）' % (
            os.path.basename(src_fp), prov_full, W, H, ds.width, ds.height))
        if W <= 0 or H <= 0:
            return None
        chunk_rows = max(1, min(H, chunk_bytes // max(1, W)))
        tr = ds.transform
        out_tr = Affine(tr.a, tr.b, tr.c + c0 * tr.a, tr.d, tr.e, tr.f + r0 * tr.e)
        prof = ds.profile.copy()
        prof.update(width=W, height=H, transform=out_tr, tiled=True,
                    blockxsize=512, blockysize=512, BIGTIFF='IF_SAFER')
        class_px, class_area = {}, {}
        tot_in = 0
        with rasterio.open(out_fp, 'w', **prof) as dst:
            for rr0 in range(r0, r1, chunk_rows):
                rr1 = min(r1, rr0 + chunk_rows)
                blk = ds.read(1, window=Window(c0, rr0, W, rr1 - rr0))
                rows = np.arange(rr0, rr1)
                cols = np.arange(c0, c1)
                lons = ds.bounds.left + (cols + 0.5) * tr.a
                lats = ds.bounds.top - (rows + 0.5) * (-tr.e)
                mask = contains_xy(geom, lons[None, :], lats[:, None])
                out = np.where(mask, blk, 0).astype(np.uint8)
                mx = int(out.max())
                assert mx <= 24, '出现非法类码 %d（>24）' % mx
                nz = out > 0
                lat_mid = (lats[0] + lats[-1]) / 2.0
                cell = PC.cell_m2(abs(tr.a), abs(tr.e), lat_mid)
                v, cc = np.unique(out[nz], return_counts=True)
                for a2, b2 in zip(v.tolist(), cc.tolist()):
                    class_px[a2] = class_px.get(a2, 0) + int(b2)
                    class_area[a2] = class_area.get(a2, 0.0) + int(b2) * cell
                tot_in += int(nz.sum())
                dst.write(out, 1, window=Window(0, rr0 - r0, W, rr1 - rr0))
            dst.write_colormap(1, cmap)
            if ovr_mode in ('scaffold', 'ovbuild'):
                dst.build_overviews(PC.OVR_LEVELS, Resampling.nearest)
                dst.update_tags(ns='rio_overview', resampling='nearest')
    a_sum = sum(class_area.values())
    qa = dict(prov=prov_full, src=src_fp, buffer_deg=buf_deg, gcj=gcj, out=out_fp,
              width=W, height=H, keep_px=tot_in,
              minutes=round((time.time() - t0) / 60, 2),
              total_km2=round(a_sum / 1e6, 2),
              classes=[dict(code=k, name=names.get(str(k), '?'), px=v,
                            area_km2=round(class_area[k] / 1e6, 3))
                       for k, v in sorted(class_px.items(), key=lambda x: -x[1])],
              note='像元中心点在省界多边形（DataV GCJ-02 %s%s）内→保留；面积=块心纬度加权 cell_m2；'
                   '类码 0=nodata 不计' % (gcj, '±%.3f°缓冲' % buf_deg if buf_deg else '原边界'),
              generated=time.strftime('%Y-%m-%d %H:%M:%S'))
    stem = os.path.splitext(out_fp)[0]
    PC.write_clr(stem + '.clr')
    PC.jsave(qa, stem + '_qa.json')
    emit('裁剪 %s 完成：保留 %.0f km²，%d 类，用时 %.1f 分钟 → %s' % (
        short, a_sum / 1e6, len(class_px), (time.time() - t0) / 60, out_fp))
    return qa


def ov_hint(*fps):
    """概览纪律（doc43）：交付级概览必须 ov_build.py 重建 + PIL 全平面验收。"""
    emit('概览纪律提示：本脚本默认不写内建概览；如需交付级概览，请执行')
    for fp in fps:
        if fp:
            emit('    python %s %s' % (OV_BUILD, fp))


def run_ovbuild(*fps):
    """--ovr ovbuild：逐件调用 ov_build.py（重建存储页 + PIL 全平面验收，rc=0 为权威）。
    注：大合幅（数十亿像元）概览重建耗时可达小时级——默认 none，仅交付件显式启用。"""
    for fp in fps:
        if not fp or not os.path.exists(fp):
            continue
        emit('ov_build 重建概览 → %s' % fp)
        r = subprocess.run([sys.executable, OV_BUILD, fp], capture_output=True, text=True,
                           encoding='utf-8', errors='ignore')
        for ln in (r.stdout or '').strip().splitlines()[-2:]:
            emit('  ov_build| %s' % ln)
        if r.returncode != 0:
            raise SystemExit('ov_build 验收未通过（rc=%d）：%s' % (r.returncode, fp))


# ======================================================================
# 四、mosaic 子命令
# ======================================================================
def cmd_mosaic(a):
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(a.reports_dir, exist_ok=True)
    plan = read_plan2()
    by_tile = {r['tile']: r for r in plan}
    provs = [PC.PROV_ALIAS.get(s.strip(), s.strip())
             for s in a.prov.split(',') if s.strip() and s.strip() != 'all'] or None
    want = [s.strip() for s in a.tiles.split(',') if s.strip()]
    if want:
        bad = [t for t in want if t not in by_tile]
        assert not bad, '瓦片不在 plan/tiles_secondrun.csv：%s' % bad
    else:
        want = [r['tile'] for r in plan
                if not provs or PC.PROV_ALIAS.get(r['prov'], r['prov']) in provs]
    frame_set = [t for t in want]
    fset = [by_tile[t] for t in frame_set]
    if not fset:
        raise SystemExit('未选中任何瓦片（--prov/--tiles 过滤为空）')
    avail = find_tiles(a.tiles_dir)
    emit('输入瓦片目录：%s（命中 %d 个 T#### 瓦片）' % (a.tiles_dir, len(avail)))
    tiles_meta, missing = [], []
    for t in frame_set:
        if t in avail:
            tiles_meta.append(validate_tile(avail[t], by_tile[t], a.strict_grid))
        else:
            missing.append(t)
    if not tiles_meta:
        raise SystemExit('第二次瓦片未就绪：选中 %d 瓦，目录内 0 个可用（%s）。'
                         '第二次产物就绪后重跑；冒烟可 --tiles-dir 指向第一次瓦片目录'
                         '（%s，注意 T2102 在 pilot/delivery）。'
                         % (len(frame_set), a.tiles_dir, FIRST_TILES_DEFAULT))
    frame = PC.grid_frame(max(r['y1'] for r in fset), min(r['y0'] for r in fset),
                          min(r['x0'] for r in fset), max(r['x1'] for r in fset))
    emit('框架：%.4fE-%.4fE × %.4fN-%.4fN  %d×%d px  px=py=%.6e°（统一 10m 方度格）' % (
        frame['x_left'], frame['x_right'],
        frame['y_top'] - frame['nrows'] * frame['py'], frame['y_top'],
        frame['ncols'], frame['nrows'], frame['px']))
    if missing:
        emit('[警告] 选中 %d 瓦中 %d 瓦未就绪：%s（框架按已就绪瓦片范围收敛）'
             % (len(frame_set), len(missing), ','.join(missing)))
    placements = build_placements(tiles_meta, frame)
    for p in placements:
        emit('  %s（%s）row0=%d col0=%d 相位(x%+.4f,y%+.4f)px %s' % (
            p['tile'], p['prov'], p['row0'], p['col0'], p['phase_x'], p['phase_y'],
            '同格直拷' if p['exact'] else 'nearest 归格'))
    tag = a.tag or ('2nd_%d' % a.year)
    plan_fp = os.path.join(a.out_dir, '%s_plan.json' % tag)
    PC.jsave(dict(tag=tag, frame=frame, year=a.year, provs=provs, tiles=frame_set,
                  missing=missing, tiles_dir=a.tiles_dir,
                  placements=[{k: (round(v, 12) if isinstance(v, float) else v)
                               for k, v in p.items() if k != 'm'} for p in placements]),
             plan_fp)
    emit('框架/布放 → %s' % plan_fp)
    if a.dry_frame:
        emit('dry-frame 完成（未写栅格）')
        return 0

    ref_cmap = None
    for m in tiles_meta:
        with rasterio.open(m['fp']) as ds:
            cm = ds.colormap(1)
        if cm is None:
            continue
        if ref_cmap is None:
            ref_cmap = cm
        else:
            assert all((cm.get(i) or (0, 0, 0, 0)) == (ref_cmap.get(i) or (0, 0, 0, 0))
                       for i in range(256)), '瓦片调色板不一致：%s' % m['fp']
    assert ref_cmap, '输入瓦片缺内置调色板'

    # ---- 总幅合并 ----
    out_fp = a.out or os.path.join(a.out_dir, '%s_瓦片合幅_10m.tif' % tag)
    if os.path.exists(out_fp) and not a.force:
        emit('已存在，跳过合并（--force 覆盖）：%s' % out_fp)
        mr = None
    else:
        emit('开始合并 %d 瓦 → %s' % (len(tiles_meta), out_fp))
        mr = merge_tiles(frame, placements, out_fp, ref_cmap, a.chunk_rows, a.ovr)
        emit('合并完成：数据像元 %.2f 亿，跳写空块 %d，用时 %.1f 分钟' % (
            mr['tot_data'] / 1e8, mr['skipped_chunks'], mr['minutes']))
    PC.write_clr(os.path.splitext(out_fp)[0] + '.clr')
    if a.preview:
        PC.preview_png(out_fp, os.path.splitext(out_fp)[0] + '_preview.png')
    ident = qa_identity(out_fp, placements, frame, a.qa_identity)
    for r in ident:
        emit('  一致性 %s（%s，%d 窗 %d px）：%s' % (
            r['tile'], r['mode'], r['windows'], r['px_checked'],
            'PASS' if r['pass_'] else 'FAIL bad=%d' % r['bad_px']))
    if mr:
        names = v31_names()
        a_sum = sum(mr['class_area_m2'].values())
        qa = dict(kind='merge', tag=tag, out=out_fp, frame=frame, year=a.year,
                  n_tiles=len(tiles_meta), tiles=frame_set, missing=missing,
                  data_px=mr['tot_data'],
                  coverage_of_frame_pct=round(100 * mr['tot_data'] /
                                              (frame['nrows'] * frame['ncols']), 4),
                  total_km2=round(a_sum / 1e6, 2),
                  classes=[dict(code=k, name=names.get(str(k), '?'), px=v,
                                area_km2=round(mr['class_area_m2'][k] / 1e6, 3))
                           for k, v in sorted(mr['class_px'].items(), key=lambda x: -x[1])],
                  tile_contrib_px=mr['tile_px'], identity_check=ident,
                  ovr_mode=a.ovr,
                  note='面积=分块纬度加权 cell_m2 累计；类码 0=nodata 不计；重叠带=先布放者胜',
                  generated=time.strftime('%Y-%m-%d %H:%M:%S'))
        PC.jsave(qa, os.path.splitext(out_fp)[0] + '_qa.json')
        if any(not r['pass_'] for r in ident):
            raise SystemExit('合并回读一致性抽检 FAIL：%s' % out_fp)
    # ---- 逐省省口径裁剪 ----
    if not a.no_clip:
        seen = set()
        for sr in fset:
            short = sr['prov']
            if short in seen or (provs and PC.PROV_ALIAS.get(short, short) not in provs):
                continue
            seen.add(short)
            full = PC.PROV_ALIAS.get(short, short)
            dn = deliv_name(full)
            clipped = os.path.join(a.out_dir, '%s%s_%d_10m.tif' % (a.prefix, dn, a.year))
            all_p = [r['tile'] for r in plan if r['prov'] == short]
            sel_p = [t for t in frame_set if by_tile[t]['prov'] == short]
            if len(sel_p) < len(all_p) or any(t in missing for t in all_p):
                emit('[警告] %s 省口径产物为部分覆盖：本省计划 %d 瓦，本次并入 %d 瓦'
                     '（未并入 %s）——面积/构成不可当交付值' % (
                         full, len(all_p), len(sel_p),
                         ','.join(sorted(set(all_p) - set(sel_p))) or '无'))
            if os.path.exists(clipped) and not a.force:
                emit('已存在，跳过裁剪（--force 覆盖）：%s' % clipped)
                continue
            clip_prov(out_fp, full, a.buffer, clipped, a.gcj, ovr_mode=a.ovr)
            if a.ovr == 'ovbuild':
                run_ovbuild(clipped)
    # ---- 可选：逐省瓦片口径合幅（诊断用，另一次全量合并） ----
    if a.prov_mosaics:
        seen = set()
        for sr in fset:
            short = sr['prov']
            if short in seen or (provs and PC.PROV_ALIAS.get(short, short) not in provs):
                continue
            seen.add(short)
            sub = [p for p in placements if p['prov'] == short]
            if not sub:
                continue
            f2 = PC.grid_frame(max(p['y1'] for p in sub), min(p['y0'] for p in sub),
                               min(p['x0'] for p in sub), max(p['x1'] for p in sub))
            fp2 = os.path.join(a.out_dir, '%s%s_%d_瓦片合幅_10m.tif'
                               % (a.prefix, deliv_name(PC.PROV_ALIAS.get(short, short)), a.year))
            if os.path.exists(fp2) and not a.force:
                emit('已存在，跳过省瓦片合幅：%s' % fp2)
                continue
            emit('省瓦片口径合并 %s（%d 瓦）→ %s' % (short, len(sub), fp2))
            mr2 = merge_tiles(f2, sub, fp2, ref_cmap, a.chunk_rows, a.ovr)
            PC.write_clr(os.path.splitext(fp2)[0] + '.clr')
            PC.jsave(dict(kind='merge_prov', out=fp2, frame=f2, prov=short,
                          data_px=mr2['tot_data'], minutes=mr2['minutes'],
                          tile_contrib_px=mr2['tile_px']),
                     os.path.splitext(fp2)[0] + '_qa.json')
            if a.ovr == 'ovbuild':
                run_ovbuild(fp2)
    if a.ovr == 'ovbuild':
        run_ovbuild(out_fp)
    ov_hint(out_fp)
    emit('=== mosaic 完成（tag=%s） ===' % tag)
    return 0


# ======================================================================
# 五、栅格对齐 + compare 子命令
# ======================================================================
def align_pair(ds_a, ds_b):
    """A=第一次、B=第二次：同像素尺寸轴对齐 → 整数像元偏移（零重采样）；
    非整相位给 exact=False（读取时按像元中心显式 nearest 归格并告警）。"""
    ta, tb = ds_a.transform, ds_b.transform
    assert abs(ta.a - tb.a) < 1e-9 * abs(ta.a), '两幅像素宽不一致：%.12e vs %.12e' % (ta.a, tb.a)
    assert abs(abs(ta.e) - abs(tb.e)) < 1e-9 * abs(ta.e), '两幅像素高不一致'
    c_off = (tb.c - ta.c) / ta.a
    r_off = (ta.f - tb.f) / (-ta.e)
    exact = jar_ok(c_off) and jar_ok(r_off)
    return dict(c_off=c_off, r_off=r_off, c0=int(round(c_off)), r0=int(round(r_off)),
                exact=exact, px=ta.a, py=-ta.e)


def read_a_fill(ds_a, r0, hh, c0, w):
    """A 栅格在自身坐标窗口读（域外补 0）。"""
    out = np.zeros((hh, w), dtype=np.uint8)
    ir0 = max(r0, 0) - r0
    ir1 = min(r0 + hh, ds_a.height) - r0
    ic0 = max(c0, 0) - c0
    ic1 = min(c0 + w, ds_a.width) - c0
    if ir1 <= ir0 or ic1 <= ic0:
        return out
    out[ir0:ir1, ic0:ic1] = ds_a.read(1, window=Window(c0 + ic0, r0 + ir0,
                                                       ic1 - ic0, ir1 - ir0))
    return out


def read_b_mapped(ds_b, al, r0, hh, c0, w):
    """B 栅格映射到 A 坐标窗口：整数偏移→窗口直读语义（零重采样）；非整→显式 nearest 索引。

    映射式（与 align_pair 同一坐标系，交叉冒烟实测定标）：
      A 行 r → B 行 floor(r − r_off + 0.5)；A 列 c → B 列 floor(c − c_off + 0.5)
    （r_off=(A.top−B.top)/py、c_off=(B.left−A.left)/px，均为"B 原点相对 A 原点"的像元偏移）
    """
    ar = np.arange(r0, r0 + hh, dtype=np.float64)
    ac = np.arange(c0, c0 + w, dtype=np.float64)
    br = np.floor(ar - al['r_off'] + 0.5 + 1e-7).astype(np.int64)
    bc = np.floor(ac - al['c_off'] + 0.5 + 1e-7).astype(np.int64)
    ok_r = (br >= 0) & (br < ds_b.height)
    ok_c = (bc >= 0) & (bc < ds_b.width)
    if not ok_r.any() or not ok_c.any():
        return np.zeros((hh, w), dtype=np.uint8)
    brc = np.clip(br, 0, ds_b.height - 1)
    bcc = np.clip(bc, 0, ds_b.width - 1)
    r_lo, c_lo = int(brc.min()), int(bcc.min())
    blk = ds_b.read(1, window=Window(c_lo, r_lo, int(bcc.max()) - c_lo + 1,
                                     int(brc.max()) - r_lo + 1))
    out = blk[np.ix_(brc - r_lo, bcc - c_lo)]
    out[~(ok_r[:, None] & ok_c[None, :])] = 0
    return out


def _seam_index(plan_rows, ds_a, band=2):
    """标称瓦片边界（plan x0/x1/y0/y1，格网吸附后实际缝位可能偏 1 px）±band 像元的
    行/列查找表（A 坐标）——用于把"缝带变化"与"域内真实变化"分开报告。"""
    tr = ds_a.transform
    W, H = ds_a.width, ds_a.height
    col_lut = np.zeros(W + 4 * band + 8, dtype=bool)
    row_lut = np.zeros(H + 4 * band + 8, dtype=bool)
    for r in plan_rows:
        for xv in (r['x0'], r['x1']):
            c = int(round((xv - tr.c) / tr.a))
            for d in range(-band, band + 1):
                k = c + d + 2 * band
                if 0 <= k < len(col_lut):
                    col_lut[k] = True
        for yv in (r['y0'], r['y1']):
            rr = int(round((tr.f - yv) / (-tr.e)))
            for d in range(-band, band + 1):
                k = rr + d + 2 * band
                if 0 <= k < len(row_lut):
                    row_lut[k] = True
    return row_lut, col_lut, 2 * band


def compare_prov(short, full, second_fp, first_fp, reports, n_sample=200000,
                 seed=7, sample_only=False, chunk_rows=0):
    """第一次（A）vs 第二次（B）省口径成品栅格级对比：
    ① 逐类像元/面积（两侧全域 + Δ）；② 共同域 24×24 转移矩阵（像元计数）；③ 采样一致率。"""
    names = v31_names()
    plan_rows = [r for r in read_plan2() if r['prov'] == short]
    result = dict(prov=short, prov_full=full, year=YEAR,
                  first=first_fp, second=second_fp, sample_n_target=n_sample,
                  generated=time.strftime('%Y-%m-%d %H:%M:%S'))
    if not os.path.exists(first_fp):
        emit('%s 第一次参照缺失：%s' % (short, first_fp))
        return None
    if not os.path.exists(second_fp):
        emit('%s 第二次成品缺失（先跑 mosaic）：%s' % (short, second_fp))
        return None
    with rasterio.open(first_fp) as d1, rasterio.open(second_fp) as d2:
        for nm, ds in (('第一次', d1), ('第二次', d2)):
            assert ds.dtypes[0] == 'uint8' and ds.nodata == 0, '%s 须 uint8/nodata=0' % nm
            assert ds.crs.to_epsg() == 4326, '%s 须 EPSG:4326' % nm
        al = align_pair(d1, d2)
        emit('%s 对齐：Δrow=%d Δcol=%d %s（A=%s %dx%d，B=%s %dx%d）' % (
            short, al['r0'], al['c0'], '整数偏移零重采样' if al['exact'] else '[警告] 非整相位→显式 nearest',
            os.path.basename(first_fp), d1.width, d1.height,
            os.path.basename(second_fp), d2.width, d2.height))
        if not al['exact']:
            emit('[警告] 两幅非同格整数偏移（%.4f/%.4f px）——逐像元对比按最近邻语义进行，'
                 '网格错位本身可能贡献伪变化' % (al['c_off'] - al['c0'], al['r_off'] - al['r0']))
        # 并集范围（A 坐标）：A 全域 ∪ B 映射域（B 占 A 行 [r_off, r_off+H2)，列 [c_off, c_off+W2)）
        ur0 = min(0, int(math.floor(al['r_off'])) - 1)
        ur1 = max(d1.height, int(math.ceil(al['r_off'] + d2.height)) + 1)
        uc0 = min(0, int(math.floor(al['c_off'])) - 1)
        uc1 = max(d1.width, int(math.ceil(al['c_off'] + d2.width)) + 1)
        if chunk_rows <= 0:
            # 峰值 ≈ 8 B/px（a/b 两幅 + 三掩膜 + 取值副本 + 缝带掩膜）
            chunk_rows = max(256, CHUNK_BYTES // max(1, uc1 - uc0) // 8)
        row_lut, col_lut, band = _seam_index(plan_rows, d1)
        rng = np.random.RandomState(seed)
        px_a = np.zeros(25, dtype=np.int64)
        px_b = np.zeros(25, dtype=np.int64)
        area_a = np.zeros(25, dtype=np.float64)
        area_b = np.zeros(25, dtype=np.float64)
        trans = np.zeros((25, 25), dtype=np.int64)
        common = 0
        only_a = 0
        only_b = 0
        seam_px = 0
        seam_chg = 0
        chg = 0
        samp_a, samp_b = [], []
        samp_n = 0
        samp_cap = max(4 * n_sample, 1)
        # 采样预算按"A/B 行交集的块数"估计（按并集估计会在域外空块上稀释预算）
        ov_rows = max(1, min(ur1, int(math.ceil(al['r_off'] + d2.height)))
                      - max(ur0, int(math.floor(al['r_off']))))
        n_est_chunks = max(1, ov_rows // chunk_rows)
        per_chunk = max(64, int(math.ceil(n_sample / n_est_chunks))) if n_sample else 0
        t0 = time.time()
        r_lo, r_hi = (ur0, ur0) if sample_only else (ur0, ur1)   # sample-only：不读全图
        for r0 in range(r_lo, r_hi, chunk_rows):
            hh = min(ur1, r0 + chunk_rows) - r0
            a_arr = read_a_fill(d1, r0, hh, uc0, uc1 - uc0)
            b_arr = read_b_mapped(d2, al, r0, hh, uc0, uc1 - uc0)
            va = a_arr > 0
            vb = b_arr > 0
            both = va & vb
            mx = int(max(a_arr.max(), b_arr.max()))
            assert mx <= 24, '第 %d 行块出现非法类码 %d（>24；只允许 0=nodata 与 1..24 类）' % (r0, mx)
            na, nb, nboth = int(va.sum()), int(vb.sum()), int(both.sum())
            common += nboth
            only_a += na - nboth
            only_b += nb - nboth
            px_a += np.bincount(a_arr[va], minlength=25)
            px_b += np.bincount(b_arr[vb], minlength=25)
            if nboth:
                av = a_arr[both]
                bv = b_arr[both]
                chg += int((av != bv).sum())
                trans += np.bincount(av.astype(np.int64) * 25 + bv, minlength=625).reshape(25, 25)
                # 缝带诊断（标称瓦片边界 ±band px）：查表切片须做域内钳制（并集窗口可越界）
                rs = np.zeros(hh, dtype=bool)
                cs = np.zeros(uc1 - uc0, dtype=bool)
                i0, i1 = max(0, r0), min(d1.height, r0 + hh)
                if i1 > i0:
                    rs[i0 - r0: i1 - r0] = row_lut[i0 + 2 * band: i1 + 2 * band]
                j0, j1 = max(0, uc0), min(d1.width, uc1)
                if j1 > j0:
                    cs[j0 - uc0: j1 - uc0] = col_lut[j0 + 2 * band: j1 + 2 * band]
                sm_both = (rs[:, None] | cs[None, :]) & both
                seam_px += int(sm_both.sum())
                seam_chg += int((sm_both & (a_arr != b_arr)).sum())
                if n_sample and samp_n < samp_cap:
                    keep = min(1.0, per_chunk / max(1, nboth))
                    sel = rng.random_sample(nboth) < keep
                    if sel.any():
                        samp_a.append(av[sel])
                        samp_b.append(bv[sel])
                        samp_n += int(sel.sum())
            lat_mid = d1.transform.f - (r0 + hh / 2.0) * (-d1.transform.e)
            cell = PC.cell_m2(abs(d1.transform.a), -d1.transform.e, lat_mid)
            area_a += px_of_row(a_arr, va, cell)
            area_b += px_of_row(b_arr, vb, cell)
        minutes = round((time.time() - t0) / 60, 2)
    # ---- 汇总 ----
    so = bool(sample_only)
    result.update(align=dict(dr=al['r0'], dc=al['c0'], exact=bool(al['exact'])),
                  minutes=minutes,
                  common_px=None if so else common,
                  only_first_px=None if so else only_a,
                  only_second_px=None if so else only_b,
                  changed_px=None if so else chg,
                  change_pct_of_common=None if so else
                  (round(100 * chg / common, 4) if common else None),
                  agree_fullpass=None if so else
                  (round(1 - chg / common, 6) if common else None),
                  seam_band_px=None if so else seam_px,
                  seam_changed_px=None if so else seam_chg,
                  seam_change_pct=None if so else
                  (round(100 * seam_chg / seam_px, 4) if seam_px else None),
                  interior_change_pct=None if so else
                  round(100 * (chg - seam_chg) / max(1, common - seam_px), 4),
                  first_px=None if so else int(px_a[1:].sum()),
                  second_px=None if so else int(px_b[1:].sum()),
                  first_km2=None if so else round(area_a[1:].sum() / 1e6, 3),
                  second_km2=None if so else round(area_b[1:].sum() / 1e6, 3),
                  first_area_km2=None if so else
                  {int(k): round(area_a[k] / 1e6, 4) for k in range(1, 25)},
                  second_area_km2=None if so else
                  {int(k): round(area_b[k] / 1e6, 4) for k in range(1, 25)},
                  first_px_by_class=None if so else
                  {int(k): int(px_a[k]) for k in range(1, 25)},
                  second_px_by_class=None if so else
                  {int(k): int(px_b[k]) for k in range(1, 25)},
                  transition_matrix=None if so else
                  [[int(trans[i, j]) for j in range(1, 25)] for i in range(1, 25)],
                  classes=[], sample=None)
    if not so:
        for k in range(1, 25):
            rs = int(trans[k, :].sum())
            result['classes'].append(dict(
                code=k, name=names.get(str(k), '?'),
                first_px=int(px_a[k]), second_px=int(px_b[k]),
                first_km2=round(area_a[k] / 1e6, 4), second_km2=round(area_b[k] / 1e6, 4),
                d_px=int(px_b[k] - px_a[k]),
                d_km2=round((area_b[k] - area_a[k]) / 1e6, 4),
                d_pct=(round(100 * (area_b[k] - area_a[k]) / area_a[k], 3) if px_a[k] else None),
                keep_rate_matrix=(round(trans[k, k] / rs, 5) if rs else None)))
    # 采样一致率（全图随机抽点，流式累积；与全通过一致率互为独立校核）
    if n_sample and not sample_only and samp_a:
        sa = np.concatenate(samp_a)
        sb = np.concatenate(samp_b)
        agree = float((sa == sb).mean())
        per = {}
        for k in range(1, 25):
            m = sa == k
            per[k] = (round(float((sb[m] == k).mean()), 5) if int(m.sum()) else None,
                      int(m.sum()))
        result['sample'] = dict(n=int(sa.size), agree=round(agree, 6),
                                per_class={str(k): v for k, v in per.items()},
                                note='流式随机抽样（共同域内），与全通过 agreement 互校')
    # ---- sample-only 快路径：geo 坐标随机点直查（不读全图） ----
    if sample_only:
        result['sample'] = sample_points(first_fp, second_fp, n_sample, seed)
        result['sample_only'] = True
        result['note'] = 'sample-only 快路径：未做全图扫描——逐类面积/转移矩阵缺失，以全通过口径为准'
    # ---- 落盘 ----
    os.makedirs(reports, exist_ok=True)
    stem = 'compare_%s_run1_run2' % short
    if not sample_only:
        trans_csv = os.path.join(reports, 'trans_%s_24x24.csv' % short)
        write_trans_csv(trans_csv, trans, names, result)
        class_csv = os.path.join(reports, '%s_class.csv' % stem)
        write_class_csv(class_csv, result, names)
    with open(os.path.join(reports, '%s.json' % stem), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    md = render_compare_md(result, names)
    with open(os.path.join(reports, '%s.md' % stem), 'w', encoding='utf-8') as f:
        f.write(md)
    smp = result.get('sample') or {}

    def _n(v, fmt='%.4f'):
        return '未算' if v is None else (fmt % v)

    emit('%s：共同域 %s 亿 px，一致率 %s（全通过）/ %s（采样 n=%s），变化 %s%%；'
         '缝带变化 %s%% vs 域内 %s%%；用时 %.1f 分钟' % (
             short, _n((result['common_px'] or 0) / 1e8, '%.2f') if not so else '未扫描',
             _n(result['agree_fullpass'], '%.5f'),
             smp.get('agree', '未采样'), smp.get('n', 0),
             _n(result['change_pct_of_common']),
             _n(result['seam_change_pct']),
             _n(result['interior_change_pct']), minutes))
    files = os.path.join(reports, '%s.md' % stem)
    if not so:
        files += ' / trans.csv / class.csv'
    emit('  → %s / .json' % files)
    return result


def px_of_row(arr, mask, cell):
    """单块逐类像元数 × 块心 cell_m2 → 面积向量（25 类，0=nodata）。"""
    b = np.bincount(arr[mask], minlength=25).astype(np.float64)
    return b * cell


def sample_points(first_fp, second_fp, n, seed):
    """快路径采样：两幅范围交集内均匀随机取点，按 geo 坐标直查（不读全图，双幅自动对齐）。
    仅 --sample-only 使用；面积/转移矩阵口径以全通过（流式）为准。"""
    rng = np.random.RandomState(seed)
    with rasterio.open(first_fp) as d1, rasterio.open(second_fp) as d2:
        left = max(d1.bounds.left, d2.bounds.left)
        right = min(d1.bounds.right, d2.bounds.right)
        bottom = max(d1.bounds.bottom, d2.bounds.bottom)
        top = min(d1.bounds.top, d2.bounds.top)
        if right <= left or top <= bottom:
            return dict(n=0, agree=None, note='两幅范围无交集')
        sa_l, sb_l = [], []
        rnd = 0
        while len(sa_l) < n and rnd < 5:      # 顶部：共同域占比低时补采（≤5 轮）
            rnd += 1
            m = int((n - len(sa_l)) * 1.2) + 64
            pts = list(zip(rng.uniform(left, right, m).tolist(),
                           rng.uniform(bottom, top, m).tolist()))
            v1 = [int(x[0]) for x in d1.sample(pts)]
            v2 = [int(x[0]) for x in d2.sample(pts)]
            for x, y in zip(v1, v2):
                if x > 0 and y > 0:
                    sa_l.append(x)
                    sb_l.append(y)
        if not sa_l:
            return dict(n=0, agree=None, note='采样点全落 nodata/域外（两幅共同域为空？）')
        sa = np.array(sa_l, dtype=np.uint8)
        sb = np.array(sb_l, dtype=np.uint8)
        per = {}
        for k in range(1, 25):
            mk = sa == k
            per[str(k)] = (round(float((sb[mk] == k).mean()), 5) if int(mk.sum()) else None,
                           int(mk.sum()))
        return dict(n=int(sa.size), agree=round(float((sa == sb).mean()), 6),
                    per_class=per, rounds=rnd,
                    note='geo 坐标直查（--sample-only 快路径；范围交集内均匀随机）')


def write_trans_csv(fp, trans, names, result):
    """24×24 转移矩阵（行=第一次类码，列=第二次类码，像元计数）+ 域账。"""
    with open(fp, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['第一次\\第二次（共同域，像元计数）']
                   + ['%d %s' % (k, names.get(str(k), '')) for k in range(1, 25)] + ['行合计'])
        for i in range(1, 25):
            w.writerow(['%d %s' % (i, names.get(str(i), ''))]
                       + [int(trans[i, j]) for j in range(1, 25)]
                       + [int(trans[i, 1:].sum())])
        w.writerow(['列合计'] + [int(trans[1:, j].sum()) for j in range(1, 25)] + [int(trans[1:, 1:].sum())])
        w.writerow([])
        w.writerow(['# 域账（A=第一次 坐标并集窗口内）'])
        w.writerow(['# 共同域（两次均>0）', result['common_px']])
        w.writerow(['# 仅第一次有数据', result['only_first_px']])
        w.writerow(['# 仅第二次有数据', result['only_second_px']])
        w.writerow(['# 变化像元（共同域内 a!=b）', result['changed_px'],
                    '%.4f%%' % (result['change_pct_of_common'] or 0)])


def write_class_csv(fp, result, names):
    with open(fp, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['类码', '类名', '一次_像元数', '一次_面积km2', '二次_像元数', '二次_面积km2',
                    'Δ_像元数', 'Δ_面积km2', 'Δ_面积%', '保持率_矩阵', '一致率_采样', '采样n'])
        smp = (result.get('sample') or {}).get('per_class') or {}
        for c in result['classes']:
            s = smp.get(str(c['code'])) or (None, None)
            w.writerow([c['code'], c['name'], c['first_px'], c['first_km2'],
                        c['second_px'], c['second_km2'], c['d_px'], c['d_km2'],
                        '' if c['d_pct'] is None else c['d_pct'],
                        '' if c['keep_rate_matrix'] is None else c['keep_rate_matrix'],
                        '' if s[0] is None else s[0], '' if s[1] is None else s[1]])
        w.writerow(['合计', '', result['first_px'], result['first_km2'],
                    result['second_px'], result['second_km2'],
                    result['second_px'] - result['first_px'],
                    round(result['second_km2'] - result['first_km2'], 3), '', '', '', ''])


def render_compare_md(r, names):
    smp = r.get('sample') or {}
    so = bool(r.get('sample_only'))
    n_com = r.get('common_px')
    L = ['# 第二次 × 第一次 栅格级对比 · %s（%d）' % (r['prov_full'], r['year']), '',
         '- 第一次：`%s`' % r['first'],
         '- 第二次：`%s`' % r['second'],
         '- 网格对齐：Δrow=%d Δcol=%d（%s）' % (r['align']['dr'], r['align']['dc'],
                                            '整数像元偏移，零重采样' if r['align']['exact']
                                            else '非整相位：按像元中心显式 nearest'),
         '- 共同域：%s（仅一次有数据 %s / 仅二次 %s）' % (
             '未扫描（sample-only）' if n_com is None else '**%.4f 亿像元**' % (n_com / 1e8),
             '—' if so else '%.4f 亿' % (r['only_first_px'] / 1e8),
             '—' if so else '%.4f 亿' % (r['only_second_px'] / 1e8)),
         '- 全通过一致率：%s；变化像元 %s' % (
             '未算' if r['agree_fullpass'] is None else '**%s**' % r['agree_fullpass'],
             '未算' if r['changed_px'] is None else '%.4f%%（%.4f 亿）' % (
                 r['change_pct_of_common'], r['changed_px'] / 1e8)),
         '- 采样一致率：%s（n=%s）%s' % (
             smp.get('agree', '未采样'), smp.get('n', 0),
             '' if not smp.get('note') else '｜' + smp['note']),
         '- 缝带诊断（标称瓦片边界 ±%d px）：%s' % (
             2, '未算（sample-only）' if so else
             '缝带 %.4f%% vs 域内 %.4f%%（缝带占共同域 %.4f%%）' % (
                 r['seam_change_pct'] or 0, r['interior_change_pct'] or 0,
                 100 * (r['seam_band_px'] or 0) / max(1, n_com or 1))),
         '- 阅读注意：A=第一次、B=第二次；一致率高 = 两次分类结果在共同域上逐像元相等比例高；'
         '变化可能来自新样本池（v2）本身，也可能来自瓦片拼接缝带（见上）+ 省界裁剪边缘。',
         '', '## 逐类像元/面积（两次全域 + Δ）', '',
         '| 类码 | 类名 | 一次 px | 一次 km² | 二次 px | 二次 km² | Δ px | Δ km² | Δ% | 保持率(矩阵) | 一致率(采样) | 采样 n |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|']
    smp_pc = smp.get('per_class') or {}
    if so:
        L.append('（sample-only 快路径未做全图扫描；逐类面积/保持率见采样列）')
    for c in r['classes']:
        s = smp_pc.get(str(c['code'])) or (None, None)
        L.append('| %d | %s | %d | %.2f | %d | %.2f | %+d | %+.2f | %s | %s | %s | %s |' % (
            c['code'], c['name'], c['first_px'], c['first_km2'], c['second_px'], c['second_km2'],
            c['d_px'], c['d_km2'],
            '' if c['d_pct'] is None else '%+.2f%%' % c['d_pct'],
            '' if c['keep_rate_matrix'] is None else '%.4f' % c['keep_rate_matrix'],
            '' if s[0] is None else '%.4f' % s[0], '' if s[1] is None else s[1]))
    if so:
        L.append('| **合计** | — | — | — | — | — | — | — | — | — | — | — |')
    else:
        L.append('| **合计** | — | **%d** | **%.1f** | **%d** | **%.1f** | **%+d** | **%+.1f** | — | — | — | — |'
                 % (r['first_px'], r['first_km2'], r['second_px'], r['second_km2'],
                    r['second_px'] - r['first_px'], r['second_km2'] - r['first_km2']))
    L += ['', '## 主要变化对（Top 10）', '']
    tm = r.get('transition_matrix')
    if tm:
        trans = np.zeros((25, 25), dtype=np.int64)
        for i in range(24):
            for j in range(24):
                trans[i + 1, j + 1] = tm[i][j]
        L += top_pairs_md(trans, names, r['common_px'], 10)
    else:
        L.append('（sample-only 快路径未生成转移矩阵；去掉 --sample-only 重跑可得）')
    L += ['', '> 口径注记：A=第一次交付（DataV 省界 GCJ-02 + buffer %.3f° 裁剪），B=第二次同口径产物；'
          '共同域=两次均有数据像元；转移矩阵按像元计数（未做面积加权）；面积=块心纬度加权 cell_m2。'
          % BUFFER_DEG]
    return '\n'.join(L) + '\n'


def top_pairs_md(trans, names, common, k=10):
    rows = []
    for i in range(1, 25):
        tot = int(trans[i, 1:].sum())
        for j in range(1, 25):
            if i == j or trans[i, j] == 0:
                continue
            rows.append((int(trans[i, j]), i, j, tot))
    rows.sort(reverse=True)
    L = ['| 第一次类 → 第二次类 | 像元 | 占共同域% | 占该源类% |', '|---|---|---|---|']
    for n, i, j, tot in rows[:k]:
        L.append('| %d %s → %d %s | %d | %.4f%% | %s |' % (
            i, names.get(str(i), '?'), j, names.get(str(j), '?'), n,
            100 * n / max(1, common), '' if not tot else '%.3f%%' % (100 * n / tot)))
    return L


# ======================================================================
# 六、qa 子命令（第二次成品面积/覆盖 vs 第一次面积表 + plan + DataV）
# ======================================================================
def stream_area(fp, chunk_rows=0):
    """10m 原生网格分块直读：像元数 + 块心纬度加权面积（含 0 类，供双口径覆盖）。"""
    t0 = time.time()
    names = v31_names()
    with rasterio.open(fp) as ds:
        assert ds.dtypes[0] == 'uint8' and ds.nodata == 0, '须 uint8/nodata=0：%s' % fp
        assert ds.crs.to_epsg() == 4326, '须 EPSG:4326：%s' % fp
        W, H = ds.width, ds.height
        tr = ds.transform
        if chunk_rows <= 0:
            chunk_rows = max(256, CHUNK_BYTES // max(1, W))
        px = np.zeros(26, dtype=np.int64)
        area = np.zeros(26, dtype=np.float64)
        for r0 in range(0, H, chunk_rows):
            hh = min(chunk_rows, H - r0)
            blk = ds.read(1, window=Window(0, r0, W, hh))
            mx = int(blk.max())
            assert mx <= 24, '非法类码 %d @row%d（%s）' % (mx, r0, fp)
            px += np.bincount(blk.ravel(), minlength=26)
            lat_mid = tr.f - (r0 + hh / 2.0) * (-tr.e)
            cell = PC.cell_m2(abs(tr.a), abs(tr.e), lat_mid)
            nb = np.bincount(blk.ravel(), minlength=26).astype(np.float64)
            area += nb * cell
    data_px = int(px[1:25].sum())
    data_km2 = float(area[1:25].sum()) / 1e6
    zero_km2 = float(area[0]) / 1e6
    return dict(W=W, H=H, px_by_class={int(k): int(px[k]) for k in range(1, 25)},
                area_km2={int(k): round(area[k] / 1e6, 4) for k in range(1, 25)},
                data_px=data_px, data_km2=round(data_km2, 3), zero_km2=round(zero_km2, 3),
                coverage_px_pct=round(100 * data_px / (W * H), 4),
                coverage_km2_pct=round(100 * data_km2 / max(1e-9, data_km2 + zero_km2), 4),
                n_classes=sum(1 for k in range(1, 25) if px[k] > 0),
                minutes=round((time.time() - t0) / 60, 2))


def read_ref_area_csv(fp):
    """第一次面积表（utf-8-sig：类码,类名,像元数,面积_km2,占比%）→ {code: (px, km2)}。"""
    out = {}
    if not os.path.exists(fp):
        return None
    with open(fp, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            try:
                out[int(r['类码'])] = (int(r['像元数']), float(r['面积_km2']))
            except (KeyError, ValueError):
                continue
    return out


def datav_area_km2(prov_full):
    """DataV 省界（GCJ-02）球面面积（geodesic，权威几何口径，与第一次 qa_build 同式）。"""
    from pyproj import Geod
    geod = Geod(ellps='WGS84')
    g = PC.prov_geom(prov_full)
    tot = 0.0
    for p in (g.geoms if g.geom_type == 'MultiPolygon' else [g]):
        lons = [c[0] for c in p.exterior.coords]
        lats = [c[1] for c in p.exterior.coords]
        a, _ = geod.polygon_area_perimeter(lons, lats)
        tot += abs(a)
        for ring in p.interiors:
            lons = [c[0] for c in ring.coords]
            lats = [c[1] for c in ring.coords]
            ai, _ = geod.polygon_area_perimeter(lons, lats)
            tot -= abs(ai)
    return abs(tot) / 1e6


def qa_one(short, full, src_fp, ref_area_dir, reports, label, rows=0):
    if not os.path.exists(src_fp):
        emit('%s 成品缺失：%s' % (short, src_fp))
        return None
    names = v31_names()
    emit('%s 面积统计（10m 原生网格）%s' % (short, src_fp))
    st = stream_area(src_fp, rows)
    ref_fp = os.path.join(ref_area_dir, '面积表_%s_%d_10m.csv' % (deliv_name(full), YEAR))
    ref = read_ref_area_csv(ref_fp)
    plan_pa = sum(r['prov_area_km2'] for r in read_plan2() if r['prov'] == short)
    try:
        dv = datav_area_km2(full)
    except Exception as e:
        dv = None
        emit('DataV 球面面积失败：%s' % str(e)[:80])
    L = ['# 第二次成品 QA · %s（%s）' % (full, label), '',
         '- 路径：`%s`' % src_fp,
         '- 尺寸：%d × %d px；数据像元 %d（像元口径覆盖 %.4f%%；面积口径覆盖 %.4f%%，'
         '双口径分开报，防混单位虚报）' % (st['W'], st['H'], st['data_px'],
                                    st['coverage_px_pct'], st['coverage_km2_pct']),
         '- 非零类数：%d；成品面积合计 **%.1f km²**（nodata 面积 %.1f km²）' % (
             st['n_classes'], st['data_km2'], st['zero_km2']),
         '- 统计口径：10m 原生网格分块直读（零重采样）；面积=块心纬度加权 `p5common.cell_m2`'
         '（与第一次 mosaic/clip/qa_build 同式）；统计用时 %.1f 分钟' % st['minutes'], '']
    # 对账表
    rows_tbl = ['| 口径 | 面积 km² | 与成品差 |', '|---|---|---|']
    if dv:
        rows_tbl.append('| DataV 省界球面面积（GCJ-02，geodesic） | %.1f | %+.2f%% |'
                        % (dv, 100 * (st['data_km2'] - dv) / dv))
    if plan_pa:
        rows_tbl.append('| plan/tiles_secondrun.csv 按瓦片主省合计 | %.1f | %+.2f%% |'
                        % (plan_pa, 100 * (st['data_km2'] - plan_pa) / plan_pa))
    rows_tbl.append('| 本成品（连续 10m 网格 + buffer %.3f°） | %.1f | — |'
                    % (BUFFER_DEG, st['data_km2']))
    L += ['## 省面积对账', ''] + rows_tbl + ['',
          '> 与 DataV 差含全域边界 %.3f° 内缩（≈500 m 边界带）与像元中心/平面弧长口径差；'
          '与第一次交付对账差额 −0.75%%~−1.90%% 属预期（doc43/交付说明）。' % BUFFER_DEG, '']
    # 与第一次面积表逐类对比
    delta_rows = []
    if ref:
        L += ['## 与第一次面积表对比（%s）' % os.path.basename(ref_fp), '',
              '| 类码 | 类名 | 一次 px | 一次 km² | 二次 px | 二次 km² | Δ px | Δ km² | Δ% |',
              '|---|---|---|---|---|---|---|---|---|']
        for k in range(1, 25):
            p1, a1 = ref.get(k, (0, 0.0))
            p2, a2 = st['px_by_class'][k], st['area_km2'][k]
            d = a2 - a1
            dp = (100 * d / a1) if a1 else None
            if p1 == 0 and p2 == 0:
                continue
            L.append('| %d | %s | %d | %.2f | %d | %.2f | %+d | %+.2f | %s |' % (
                k, names.get(str(k), '?'), p1, a1, p2, a2, p2 - p1, d,
                '' if dp is None else '%+.2f%%' % dp))
            delta_rows.append([k, names.get(str(k), '?'), p1, a1, p2, a2, p2 - p1,
                               round(d, 4), '' if dp is None else round(dp, 3)])
        d_tot = st['data_km2'] - sum(v[1] for v in ref.values())
        L.append('| **合计** | — | **%d** | **%.1f** | **%d** | **%.1f** | **%+d** | **%+.1f** | %s |'
                 % (sum(v[0] for v in ref.values()), sum(v[1] for v in ref.values()),
                    st['data_px'], st['data_km2'],
                    st['data_px'] - sum(v[0] for v in ref.values()), d_tot,
                    '' if not sum(v[1] for v in ref.values()) else
                    '%+.2f%%' % (100 * d_tot / sum(v[1] for v in ref.values()))))
        emit('  对第一次面积表：成品 %.1f km² vs 一次 %.1f km²（%+.2f%%)' % (
            st['data_km2'], sum(v[1] for v in ref.values()),
            100 * d_tot / max(1e-9, sum(v[1] for v in ref.values()))))
    else:
        L += ['## 与第一次面积表对比', '', '第一次面积表缺失（%s），跳过逐类对比。' % ref_fp, '']
        emit('  第一次面积表缺失，跳过逐类对比：%s' % ref_fp)
    L += ['', '## 逐类面积表', '',
          '| 类码 | 类名 | 像元数 | 面积 km² | 占比% |', '|---|---|---|---|---|']
    for k in sorted(range(1, 25), key=lambda x: -st['area_km2'][x]):
        if st['px_by_class'][k] == 0:
            continue
        L.append('| %d | %s | %d | %.3f | %.4f |' % (
            k, names.get(str(k), '?'), st['px_by_class'][k], st['area_km2'][k],
            100 * st['area_km2'][k] / max(1e-9, st['data_km2'])))
    L += ['', '> 说明：与点留出评估（evalprep/c4 的 OA/F1）是两套口径——本表是**栅格像元面积**；'
          '第一次 evalprep 报告为点评估 JSON（无面积表），故面积对账基线=交付面积表 + plan + DataV。']
    os.makedirs(reports, exist_ok=True)
    md_fp = os.path.join(reports, 'qa_%s_%s.md' % (short, label))
    with open(md_fp, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    csv_fp = os.path.join(reports, 'qa_%s_%s_class.csv' % (short, label))
    with open(csv_fp, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['类码', '类名', '一次_像元数', '一次_面积km2', '二次_像元数', '二次_面积km2',
                    'Δ_像元数', 'Δ_面积km2', 'Δ_面积%'])
        w.writerows(delta_rows)
    js_fp = os.path.join(reports, 'qa_%s_%s.json' % (short, label))
    with open(js_fp, 'w', encoding='utf-8') as f:
        json.dump(dict(prov=short, prov_full=full, label=label, src=src_fp,
                       width=st['W'], height=st['H'], data_px=st['data_px'],
                       data_km2=st['data_km2'], zero_km2=st['zero_km2'],
                       coverage_px_pct=st['coverage_px_pct'],
                       coverage_km2_pct=st['coverage_km2_pct'],
                       n_classes=st['n_classes'],
                       px_by_class=st['px_by_class'], area_km2=st['area_km2'],
                       datav_km2=None if dv is None else round(dv, 2),
                       plan_km2=round(plan_pa, 2), ref_area_csv=ref_fp,
                       generated=time.strftime('%Y-%m-%d %H:%M:%S')), f,
                  ensure_ascii=False, indent=1)
    emit('  → %s（%.1f 亿 px，%d 类，%.1f km²）' % (md_fp, st['data_px'] / 1e8,
                                              st['n_classes'], st['data_km2']))
    return st


# ======================================================================
# 七、CLI
# ======================================================================
def resolve_provs(arg):
    if arg.strip().lower() == 'all':
        return list(PROV2)
    out = []
    for s in arg.split(','):
        s = s.strip()
        if not s:
            continue
        full = PC.PROV_ALIAS.get(s, s)
        short = PC.PROV_SHORT.get(full, s)
        if short not in PROV2:
            raise SystemExit('省 %s 不在第二次 40 瓦范围（%s）' % (s, ','.join(PROV2)))
        if short not in out:
            out.append(short)
    return out


def find_first_product(full, year, override=''):
    """第一次省口径成品默认探测（D:/生产_2023五省/成品/分省 优先，其次第一次 mosaic/out）。
    命名与第一次交付一致：<交付短名>_<year>_10m.tif（如 四川_2023_10m.tif）。"""
    if override:
        return override
    dn = deliv_name(full)
    for d in FIRST_PROD_DIRS:
        for nm in ('%s_%d_10m.tif' % (dn, year), '%s_%d_10m.tif' % (full, year),
                   '2nd_%s_%d_10m.tif' % (dn, year)):
            fp = os.path.join(d, nm)
            if os.path.exists(fp):
                return fp
    return os.path.join(FIRST_PROD_DIRS[0], '%s_%d_10m.tif' % (dn, year))


def name_of(tpl, full, short, year):
    """成品名模板：{full}=交付短名（四川）、{dv}=DataV 全名（四川省）、{short}=简称（川）。"""
    return tpl.format(full=deliv_name(full), dv=full, short=short, year=year)


def add_common(ap):
    ap.add_argument('--prov', default='all', help='省简称/全名（川,粤,宁）；all=三省')
    ap.add_argument('--year', type=int, default=YEAR)
    ap.add_argument('--rows', type=int, default=0, help='行块高（0=按 192MB 自适应）')


def main():
    ap = argparse.ArgumentParser(
        prog='c5_mosaic_compare.py',
        description='第二次分类（川/粤/宁 40 瓦）成品拼接 + 与第一次成品栅格级对比（nearest-only）')
    sub = ap.add_subparsers(dest='cmd')

    m = sub.add_parser('mosaic', help='40 瓦 → 总幅 + 逐省省口径裁剪')
    add_common(m)
    m.add_argument('--tiles', default='', help='只并入这些瓦片（逗号分隔；默认按 --prov 全选）')
    m.add_argument('--tiles-dir', default=TILES2_DEFAULT,
                   help='瓦片目录（逗号分隔可多个；默认 %s）' % TILES2_DEFAULT)
    m.add_argument('--out-dir', default=MOSAIC2)
    m.add_argument('--reports-dir', default=REPORTS2)
    m.add_argument('--tag', default='', help='总幅标签（默认 2nd_<year>）')
    m.add_argument('--prefix', default='', help='省口径成品文件名前缀（默认空）')
    m.add_argument('--out', default='', help='总幅输出路径（默认 <out-dir>/<tag>_瓦片合幅_10m.tif）')
    m.add_argument('--buffer', type=float, default=BUFFER_DEG, help='省界缓冲（度，默认 -0.005）')
    m.add_argument('--gcj', choices=['raw', 'gcj2wgs'], default='raw',
                   help='省界用法：raw=GCJ-02 多边形原样（第一次实测口径，默认）；'
                        'gcj2wgs=先反演到 WGS84（会与第一次交付足迹不同）')
    m.add_argument('--chunk-rows', type=int, default=0)
    m.add_argument('--qa-identity', type=int, default=8, help='每瓦片回读一致性抽检窗口数')
    m.add_argument('--dry-frame', action='store_true', help='只算框架/布放，不写栅格')
    m.add_argument('--no-clip', action='store_true', help='只出总幅，不做省口径裁剪')
    m.add_argument('--prov-mosaics', action='store_true',
                   help='额外输出逐省瓦片口径合幅（诊断用，全量合并一次，耗时翻倍）')
    m.add_argument('--no-preview', dest='preview', action='store_false', default=True)
    m.add_argument('--ovr', choices=['none', 'scaffold', 'ovbuild'], default='none',
                   help='概览：none=不写（默认，doc43 纪律）；scaffold=GDAL build_overviews'
                        '（非交付级！）；ovbuild=写完调用 ov_build.py 重建+PIL 验收')
    m.add_argument('--strict-grid', action='store_true', help='瓦片相位非整即中止（默认 nearest 归格+告警）')
    m.add_argument('--force', action='store_true', help='覆盖已存在产物')

    c = sub.add_parser('compare', help='第二次省口径成品 vs 第一次省口径成品（栅格级）')
    add_common(c)
    c.add_argument('--second-dir', default=MOSAIC2, help='第二次省口径成品目录')
    c.add_argument('--first-dir', default='', help='第一次省口径成品目录（默认自动探测 D:/生产_2023五省/成品/分省）')
    c.add_argument('--second-file', default='', help='（单省）直接指定第二次文件')
    c.add_argument('--first-file', default='', help='（单省）直接指定第一次文件')
    c.add_argument('--name-tpl', default='{full}_{year}_10m.tif', help='省口径成品文件名模板')
    c.add_argument('--reports-dir', default=REPORTS2)
    c.add_argument('--sample', type=int, default=200000, help='采样点数（0=不采样）')
    c.add_argument('--no-sample', dest='sample', action='store_const', const=0)
    c.add_argument('--sample-only', action='store_true',
                   help='只做 geo 坐标随机采样一致率快路径（不读全图，不出转移矩阵）')
    c.add_argument('--seed', type=int, default=7)

    q = sub.add_parser('qa', help='第二次成品逐省面积表（vs 第一次面积表 + plan + DataV）')
    add_common(q)
    q.add_argument('--src-dir', default=MOSAIC2, help='待统计成品目录（默认第二次 mosaic2；'
                                                      '冒烟可指向第一次目录做路径自检）')
    q.add_argument('--src-file', default='', help='（单省）直接指定文件')
    q.add_argument('--name-tpl', default='{full}_{year}_10m.tif')
    q.add_argument('--ref-area-dir', default=FIRST_AREA_DIR, help='第一次面积表目录')
    q.add_argument('--label', default='second', help='输出文件名标签（qa_<省>_<label>.md）')
    q.add_argument('--reports-dir', default=REPORTS2)

    a = ap.parse_args()
    if a.cmd == 'mosaic':
        return cmd_mosaic(a)
    if a.cmd == 'compare':
        provs = resolve_provs(a.prov)
        if a.sample_only:
            emit('sample-only 快路径：仅采样一致率（不出转移矩阵/逐类面积）')
        results = {}
        for short in provs:
            full = PC.PROV_ALIAS.get(short, short)
            second = a.second_file or os.path.join(a.second_dir,
                                                   name_of(a.name_tpl, full, short, a.year))
            if a.first_file:
                first = a.first_file
            elif a.first_dir:
                first = os.path.join(a.first_dir, name_of(a.name_tpl, full, short, a.year))
            else:
                first = find_first_product(full, a.year)
            r = compare_prov(short, full, second, first, a.reports_dir,
                             n_sample=a.sample, seed=a.seed,
                             sample_only=a.sample_only, chunk_rows=a.rows)
            if r:
                results[short] = r
        if not results:
            emit('无任何省完成对比（第二次成品未就绪？先跑 mosaic）')
            return 2
        if len(results) > 1:
            fp = os.path.join(a.reports_dir, 'compare_all.md')
            with open(fp, 'w', encoding='utf-8') as f:
                f.write('# 第二次 × 第一次 栅格级对比 · 汇总（川/粤/宁）\n\n')
                f.write('| 省 | 共同域(亿px) | 一致率(全通过) | 一致率(采样) | 变化% | 缝带变化% | 域内变化% | 一次km² | 二次km² | Δkm² |\n')
                f.write('|---|---|---|---|---|---|---|---|---|---|\n')
                for s, r in results.items():
                    smp = (r.get('sample') or {}).get('agree', '')
                    f.write('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n' % (
                        s,
                        '未扫描' if r['common_px'] is None else '%.4f' % (r['common_px'] / 1e8),
                        '未算' if r['agree_fullpass'] is None else r['agree_fullpass'],
                        smp,
                        '未算' if r['change_pct_of_common'] is None else '%.4f%%' % r['change_pct_of_common'],
                        '未算' if r['seam_change_pct'] is None else '%.4f%%' % r['seam_change_pct'],
                        '未算' if r['interior_change_pct'] is None else '%.4f%%' % r['interior_change_pct'],
                        '未算' if r['first_km2'] is None else '%.1f' % r['first_km2'],
                        '未算' if r['second_km2'] is None else '%.1f' % r['second_km2'],
                        '未算' if r['second_km2'] is None else '%+.1f' % (r['second_km2'] - r['first_km2'])))
            emit('汇总 → %s' % fp)
        return 0
    if a.cmd == 'qa':
        provs = resolve_provs(a.prov)
        done = 0
        for short in provs:
            full = PC.PROV_ALIAS.get(short, short)
            src = a.src_file or os.path.join(a.src_dir,
                                             name_of(a.name_tpl, full, short, a.year))
            if qa_one(short, full, src, a.ref_area_dir, a.reports_dir, a.label, a.rows):
                done += 1
        if not done:
            emit('无任何省完成统计（成品未就绪？）')
            return 2
        return 0
    ap.print_help()
    return 1


if __name__ == '__main__':
    sys.exit(main())
