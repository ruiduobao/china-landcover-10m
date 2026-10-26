# -*- coding: utf-8 -*-
"""c6_deliver.py — 第二次分类（川/粤/宁 40 瓦 + 补瓦 T2202 = 41 瓦 · 2023）交付打包（交付树对齐第一次五省批）

* 输入：
    第二次省口径成品（c5 产物）：F:/lc_work/prod2nd_2023/mosaic2/{四川,广东,宁夏}_2023_10m.tif
                （uint8/单波段/nodata=0/EPSG:4326/10m 原生方度格/内嵌 256 调色板/主文件单页）
    栅格 QA/对比（c5）：F:/lc_work/prod2nd_2023/reports/mosaic2/qa_{省}_{label}.json（面积首选，
                有则复用免重扫）、compare_{省}_run1_run2.json、trans_{省}_24x24.csv（软引用）
    点评估（c4）：F:/lc_work/prod2nd_2023/reports/eval2/eval2_{省}_2023.json/.md +
                compare_run1_run2_{省}.md + {省}_实跑小结.md + 三省_实跑汇总.md（有则收）
    c5 报告：reports/mosaic2/拼接对比实跑报告.md（有则收为 成品/评估/拼接对比实跑报告_c5.md）
    逐瓦 QA（c3）：F:/lc_work/prod2nd_2023/qa/{tile}_qa.json；瓦片 rasters_batch/{tile}_10m.tif
    计划/预算/池统计：plan/tiles_secondrun.csv、config.json、reports/pool_v2_stats.json
    EECU 台账：F:/lc_work/prod2nd_2023/ledger/*.csv（只读汇总，**不复制原始 CSV**）
    色卡权威源：F:/lc_work/v31_exp/code/d_tiles.py 的 COLORS + 第一次 D:/生产_2023五省/成品/色卡/
    第一次交付（**只读**；冒烟自检输入）：D:/生产_2023五省/成品/分省/*_2023_10m.tif、面积表/
* 规则源：第一次交付流水线 F:/lc_work/prod5p_2023/delivery_pkg/（pipeline_after_mosaic/make_readme/
    qa_build/ov_ext_big）——本脚本为第二次适配版：交付树对齐、口径逐条继承、避免重写算法
* 门槛：类别栅格全链 nearest（禁止非最近邻重采样）；主 tif 单 IFD 页（无内建概览，doc43），
    金字塔=外置 .ovr（复制路线做逐级随机瓦片 zlib 读回抽检；无源则 ov_ext 流式写出+全瓦验收）；
    面积=10m 原生网格块心纬度加权 PC.cell_m2（与第一/二次 mosaic/clip 同式）；覆盖率双口径分报；
    QA/面积统计来源显式标注（c5 复用 or 本脚本重扫），交付 tif 与源逐字节 MD5 一致才放行
* 输出：--dest 根（默认 F:/lc_work/prod2nd_2023/delivery/）：
    交付包_README.md ｜ MANIFEST_MD5.txt ｜ 台账/生产台账摘要.md
    成品/分省/{省}_2023_10m.tif(.ovr)(.clr) ｜ 成品/预览/{省}_预览10m.png
    成品/面积表/{省}_面积.csv + 汇总.csv ｜ 成品/色卡/{地类颜色对照.md,地类颜色表.csv,颜色表.clr}
    成品/QA/{省}_qa.json + qa汇总.md（+ c5 详表副本，有则收）
    成品/评估/{省}_eval2.json/.md + compare_run1_run2.md（有则收，无则清单占位）
              + {省}_实跑小结.md + 三省_实跑汇总.md + 拼接对比实跑报告_c5.md（有则收）
    冒烟（--smoke）：全部落 <dest>/_smoke/，输入可直接用第一次交付件（只读）
* 用法：
    python 代码/9.第二次分类川粤宁/c6_deliver.py                    # 正式打包（第二次产物就绪后）
    python 代码/9.第二次分类川粤宁/c6_deliver.py --prov 川,粤,宁 --ovr auto
    python 代码/9.第二次分类川粤宁/c6_deliver.py --smoke --prov 宁 \
        --src-dir D:/生产_2023五省/成品/分省 --ref-area-dir D:/生产_2023五省/成品/面积表
"""
import os
import sys
import csv
import time
import glob
import random
import shutil
import struct
import zlib
import hashlib
import argparse
import warnings

sys.stdout.reconfigure(encoding='utf-8')
sys.dont_write_bytecode = True            # 只读纪律：不往 prod5p/v31_exp 写 __pycache__
os.environ.setdefault('GDAL_CACHEMAX', '512')   # 内存纪律：GDAL 块缓存 512MB（串行）
# rasterio 读窗在本环境触发 numpy 2.5 DeprecationWarning（无害，第一次流水线同样存在）；仅静音该条
warnings.filterwarnings('ignore', message='Setting the shape on a NumPy array')

# ---------------- 路径常量（raw string 绝对路径，单点维护） ----------------
PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
ROOT1 = r'F:/lc_work/prod5p_2023'                       # 第一次（只读）
ROOT2 = r'F:/lc_work/prod2nd_2023'                      # 第二次
P5COMMON_DIR = os.path.join(ROOT1, 'mosaic')
DELIV_PKG1 = os.path.join(ROOT1, 'delivery_pkg')        # ov_ext_big（只读复用）
V31_CODE = r'F:/lc_work/v31_exp/code'
MOSAIC2 = os.path.join(ROOT2, 'mosaic2')                # c5 省口径成品
REPORTS2 = os.path.join(ROOT2, 'reports', 'mosaic2')    # c5 面积 QA / 栅格对比
EVAL2_DIR = os.path.join(ROOT2, 'reports', 'eval2')     # c4 点评估
QADIR2 = os.path.join(ROOT2, 'qa')                      # c3 逐瓦 QA
TILES2 = os.path.join(ROOT2, 'rasters_batch')           # c3 逐瓦栅格
LEDGER_DIR = os.path.join(ROOT2, 'ledger')
PLAN2 = os.path.join(ROOT2, 'plan', 'tiles_secondrun.csv')
CONFIG2 = os.path.join(ROOT2, 'config.json')
POOL_STATS = os.path.join(ROOT2, 'reports', 'pool_v2_stats.json')
FIRST_AREA_DIR = r'D:/生产_2023五省/成品/面积表'          # 只读（冒烟对账基线）
FIRST_PROV_DIR = r'D:/生产_2023五省/成品/分省'            # 只读（冒烟输入候选）
FIRST_COLOR_CSV = r'D:/生产_2023五省/成品/色卡/地类颜色表.csv'  # 只读（色卡兜底）
DEST_DEFAULT = os.path.join(ROOT2, 'delivery')
YEAR = 2023
PROV2 = ['川', '粤', '宁']                 # 第二次交付覆盖省（plan 口径：川24/粤10/宁6=40 瓦；
#                                           另有补瓦 T2202 主省=闽 1 瓦，仅入合幅/栅格对比，不单独分省交付）
DELIV_NAME = {'四川省': '四川', '广东省': '广东', '宁夏回族自治区': '宁夏'}
BUFFER_DEG = -0.005                        # 第一次交付口径：DataV 省界内缩 500m
C5_QA_LABEL = 'second'                     # c5 qa 子命令的默认标签
MAX_DIM = 2400                             # 预览缩略图长边上限（px）
CHUNK_BYTES = 192 * 1024 * 1024            # 统计分块缓冲上限

sys.path.insert(0, P5COMMON_DIR)
sys.path.insert(0, DELIV_PKG1)
sys.path.insert(0, V31_CODE)
import numpy as np                                        # noqa: E402
import rasterio                                           # noqa: E402
from rasterio.windows import Window                       # noqa: E402
import p5common as PC                                     # noqa: E402


def emit(m):
    print('[%s] %s' % (time.strftime('%m-%d %H:%M:%S'), m), flush=True)


def deliv_of(short):
    """省简称 → 交付短名（四川/广东/宁夏，与第一次交付文件命名一致）。"""
    return DELIV_NAME.get(PC.PROV_ALIAS.get(short, short), short)


def v31_names():
    """24 类码 → 类名（v31_exp 只读加载；失败退化为类码字符串）。"""
    try:
        import v31_common as VC
        return VC.V31_NAMES()
    except Exception:
        return {str(i): str(i) for i in range(1, 25)}


def md5_file(fp, buf=1 << 22):
    h = hashlib.md5()
    with open(fp, 'rb') as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_plan2():
    """plan/tiles_secondrun.csv → list[dict]（保持原序）。"""
    rows = []
    with open(PLAN2, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append(dict(tile=r['tile'], prov=r['prov'],
                             x0=float(r['x0']), y0=float(r['y0']),
                             x1=float(r['x1']), y1=float(r['y1']),
                             land_frac=float(r['land_frac']),
                             prov_area_km2=float(r.get('prov_area_km2', 0) or 0)))
    return rows


# ======================================================================
# 一、成品门禁与统计（10m 原生网格，双口径覆盖）
# ======================================================================
def gate_product(fp):
    """成品门禁：uint8/单波段/nodata=0/EPSG:4326/无旋转/10m 原生方度格/内嵌 256 调色板/
    主文件单 IFD 页（无内建概览，doc43）。返回元信息 dict（失败即 AssertionError）。"""
    import tifffile
    with rasterio.open(fp) as ds:
        assert ds.count == 1 and ds.dtypes[0] == 'uint8', '%s 非 uint8 单波段' % fp
        assert ds.nodata == 0, '%s nodata!=0' % fp
        assert ds.crs is not None and ds.crs.to_epsg() == 4326, '%s 非 EPSG:4326' % fp
        t = ds.transform
        msg = ''
        if abs(t.b) > 1e-12 or abs(t.d) > 1e-12:
            msg = '存在旋转项 b/d'
        if abs(t.a - PC.EQ_DEG) / PC.EQ_DEG > 1e-9 or abs(-t.e - PC.EQ_DEG) / PC.EQ_DEG > 1e-9:
            msg = msg or '非 10m 原生方度格（px=%.12e）' % t.a
        assert not msg, '%s：%s' % (fp, msg)
        cmap = ds.colormap(1)
        n_cmap = len(cmap) if cmap else 0
        assert n_cmap >= 256, '%s 调色板 %d 项（应 256）' % (fp, n_cmap)
        meta = dict(width=ds.width, height=ds.height, dtype=ds.dtypes[0], nodata=ds.nodata,
                    crs=str(ds.crs), px=t.a, py=-t.e, left=t.c, top=t.f,
                    n_cmap=n_cmap, size_gb=os.path.getsize(fp) / 2 ** 30)
    with tifffile.TiffFile(fp) as tf:
        meta['tif_pages'] = len(tf.pages)
    meta['internal_ov'] = meta['tif_pages'] > 1   # 主文件内建概览页=违反交付纪律（doc43）
    return meta


def scan_stats(fp, chunk_bytes=CHUNK_BYTES):
    """单遍流读：类像元数 + 类面积（块心纬度加权 cell_m2，与 c5/p5common 完全同式）+ 双口径覆盖。"""
    t0 = time.time()
    with rasterio.open(fp) as ds:
        W, H = ds.width, ds.height
        tr = ds.transform
        chunk_rows = max(256, chunk_bytes // max(1, W))
        px = np.zeros(26, dtype=np.int64)
        area = np.zeros(26, dtype=np.float64)
        for r0 in range(0, H, chunk_rows):
            hh = min(chunk_rows, H - r0)
            blk = ds.read(1, window=Window(0, r0, W, hh))
            mx = int(blk.max())
            assert mx <= 24, '非法类码 %d @row%d（%s）' % (mx, r0, fp)
            nb = np.bincount(blk.ravel(), minlength=26)
            px += nb
            lat_mid = tr.f - (r0 + hh / 2.0) * (-tr.e)
            area += nb.astype(np.float64) * PC.cell_m2(abs(tr.a), abs(tr.e), lat_mid)
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


def stats_from_c5_qa(fp):
    """c5 qa 子命令产物 → 与 scan_stats 同结构（复用免重扫；字段缺失即 None）。"""
    d = PC.jload(fp)
    if not d or 'px_by_class' not in d:
        return None
    need = ('width', 'height', 'data_px', 'data_km2', 'zero_km2',
            'coverage_px_pct', 'coverage_km2_pct', 'n_classes')
    if any(k not in d for k in need):
        return None
    return dict(W=d['width'], H=d['height'],
                px_by_class={int(k): int(v) for k, v in d['px_by_class'].items()},
                area_km2={int(k): float(v) for k, v in d['area_km2'].items()},
                data_px=int(d['data_px']), data_km2=float(d['data_km2']),
                zero_km2=float(d['zero_km2']),
                coverage_px_pct=float(d['coverage_px_pct']),
                coverage_km2_pct=float(d['coverage_km2_pct']),
                n_classes=int(d['n_classes']), minutes=None,
                c5_datav_km2=d.get('datav_km2'), c5_plan_km2=d.get('plan_km2'),
                c5_src=d.get('src'))


def datav_area_km2(short):
    """DataV 省界（GCJ-02）球面面积（geodesic，与 qa_build/c5 同式）。"""
    from pyproj import Geod
    geod = Geod(ellps='WGS84')
    g = PC.prov_geom(PC.PROV_ALIAS.get(short, short))
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


# ======================================================================
# 二、外置金字塔 .ovr（复制抽检 or ov_ext 流式重建；主 tif 始终单页）
# ======================================================================
def ovr_verify_spot(fp, n_tiles=2, seed=7):
    """有界读回抽检：每级随机 n_tiles 个 512×512 瓦片（自解码 .ovr 存储字节）与底图 TL
    重算严格比对（越界补零，与写出侧同式）+ GDAL 挂载确认。返回 dict(ok, levels, mount)。"""
    import ov_ext_big as OE
    TILE = OE.TILE
    ovr_fp = fp + '.ovr'
    rng = random.Random(seed)
    levels, ok = {}, True
    with rasterio.open(fp) as ds:
        H, W = ds.height, ds.width
        with open(ovr_fp, 'rb') as f:
            f.seek(4)
            ifd_off = struct.unpack('<I', f.read(4))[0]
            for lev in PC.OVR_LEVELS:
                pw, ph, offs, cnts = OE.read_page_ifd(f, ifd_off)
                exp_pw, exp_ph = (W + lev - 1) // lev, (H + lev - 1) // lev
                if (pw, ph) != (exp_pw, exp_ph):
                    levels[lev] = dict(checked=0, bad=-1, note='页尺寸 %s != %s' % ((pw, ph), (exp_pw, exp_ph)))
                    ok = False
                else:
                    ntw = (pw + TILE - 1) // TILE
                    nth = (ph + TILE - 1) // TILE
                    bad, checked = 0, 0
                    for _ in range(n_tiles):
                        tr, tc = rng.randrange(nth), rng.randrange(ntw)
                        f.seek(offs[tr * ntw + tc])
                        got = np.frombuffer(zlib.decompress(f.read(cnts[tr * ntw + tc])),
                                            'uint8').reshape(TILE, TILE)
                        r0p, c0p = tr * TILE * lev, tc * TILE * lev
                        R = max(0, min(TILE * lev, H - r0p))
                        C = max(0, min(TILE * lev, W - c0p))
                        exp = np.zeros((TILE, TILE), 'uint8')
                        if R > 0 and C > 0:
                            blk = ds.read(1, window=Window(c0p, r0p, C, R))
                            sub = blk[::lev, ::lev]        # TL：块左上
                            exp[:sub.shape[0], :sub.shape[1]] = sub
                        bad += int((got != exp).sum())
                        checked += int(got.size)
                    levels[lev] = dict(checked=checked, bad=bad)
                    ok = ok and bad == 0
                ifd_off += 2 + 12 * 14 + 4                 # 写出侧固定布局（见 ov_ext_big）
        mount = ds.overviews(1)
    if list(mount) != list(PC.OVR_LEVELS):
        ok = False
    return dict(ok=bool(ok), levels=levels, mount=list(mount))


def attach_ovr(src_tif, dst_tif, mode):
    """交付副本挂外置 .ovr：auto=源有则复制+抽检（失败改重建）/无则流式重建+全瓦验收。"""
    src_ovr, dst_ovr = src_tif + '.ovr', dst_tif + '.ovr'
    if mode == 'auto':
        mode = 'copy' if os.path.exists(src_ovr) else 'build'
    if mode == 'none':
        return dict(mode='none', ok=None, detail='未生成（--ovr none）')
    if mode == 'copy':
        if not os.path.exists(src_ovr):
            raise SystemExit('--ovr copy 但源缺 .ovr：%s' % src_ovr)
        if os.path.exists(dst_ovr):
            os.remove(dst_ovr)
        shutil.copy2(src_ovr, dst_ovr)
        d = ovr_verify_spot(dst_tif)
        if d['ok']:
            emit('  .ovr 复制+逐级随机瓦片读回抽检 PASS（%.1f MB）' % (os.path.getsize(dst_ovr) / 1e6))
            return dict(mode='copy', src=src_ovr, ok=True,
                        detail='抽检=%s；mount=%s' % (d['levels'], d['mount']))
        emit('  [警告] 复制 .ovr 抽检未过（%s）→ 改走 ov_ext 重建' % d['levels'])
        os.remove(dst_ovr)
        mode = 'build'
    if mode == 'build':
        import ov_ext_big as OE
        emit('  外置金字塔流式重建（ov_ext_big，写出+全瓦读回验收）→ %s' % dst_ovr)
        OE.write_ovr_stream(dst_tif)
        ok = OE.verify_stream(dst_tif)
        if not ok:
            raise SystemExit('❌ %s.ovr 全瓦验收未通过' % dst_tif)
        return dict(mode='build', src=None, ok=True,
                    detail='ov_ext 流式写出 + 全瓦读回验收（隶属/TL 双 100%）')
    raise SystemExit('未知 --ovr 模式：%s' % mode)


# ======================================================================
# 三、面积表 / 色卡 / 预览
# ======================================================================
def write_area_csv(fp, stats, names):
    """逐类面积表：Run-1 同式表头（类码,类名,像元数,面积_km2,占比%），按面积降序。"""
    total = stats['data_km2']
    classes = sorted(((k, v) for k, v in stats['px_by_class'].items() if v > 0),
                     key=lambda x: -stats['area_km2'][x[0]])
    with open(fp, 'w', encoding='utf-8-sig', newline='') as f:
        f.write('类码,类名,像元数,面积_km2,占比%\n')
        for code, px in classes:
            f.write('%d,%s,%d,%.3f,%.4f\n' % (
                code, names.get(str(code), '?'), px, stats['area_km2'][code],
                100 * stats['area_km2'][code] / max(1e-9, total)))
    return classes


def write_area_summary(fp, per_prov, names):
    """三省汇总（长表：省,类码,类名,像元数,面积_km2,占比% + 各省合计行 + 总计行）。"""
    with open(fp, 'w', encoding='utf-8-sig', newline='') as f:
        f.write('省,类码,类名,像元数,面积_km2,占比%\n')
        for short in PROV2:
            e = per_prov.get(short)
            if not e:
                continue
            st = e['stats']
            for code, px in sorted(e['classes'], key=lambda x: -st['area_km2'][x[0]]):
                f.write('%s,%d,%s,%d,%.3f,%.4f\n' % (
                    e['name'], code, names.get(str(code), '?'), px, st['area_km2'][code],
                    100 * st['area_km2'][code] / max(1e-9, st['data_km2'])))
            f.write('%s,合计,,%d,%.3f,100\n' % (e['name'], st['data_px'], st['data_km2']))
        t_px = sum(e['stats']['data_px'] for e in per_prov.values())
        t_km2 = sum(e['stats']['data_km2'] for e in per_prov.values())
        f.write('总计,,,%d,%.3f,\n' % (t_px, t_km2))


def load_colors():
    """色卡权威源级联：① d_tiles.COLORS（与交付调色板同源）→ ② 第一次交付色卡 CSV → ③ 报错。"""
    try:
        from d_tiles import COLORS
        return {int(k): tuple(int(round(c * 255)) for c in v) for k, v in COLORS.items()}, 'd_tiles.COLORS'
    except Exception as e:
        emit('  [提示] d_tiles.COLORS 不可用（%s），改用第一次交付色卡 CSV' % str(e)[:60])
    if os.path.exists(FIRST_COLOR_CSV):
        out = {}
        with open(FIRST_COLOR_CSV, encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                out[int(r['code'])] = (int(r['R']), int(r['G']), int(r['B']))
        return out, '第一次交付 地类颜色表.csv'
    raise SystemExit('色卡源不可用：d_tiles 导入失败且 %s 不存在' % FIRST_COLOR_CSV)


def write_color_card(out_dir, colors, names, present_codes):
    """色卡三件套（+ run-1 别名 .clr）：地类颜色对照.md / 地类颜色表.csv / 颜色表.clr。"""
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for k in sorted(colors):
        r, g, b = colors[k]
        rows.append((k, names.get(str(k), '?'), '#%02X%02X%02X' % (r, g, b), r, g, b))
    with open(os.path.join(out_dir, '地类颜色表.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        f.write('code,name,hex,R,G,B\n')
        for k, nm, hx, r, g, b in rows:
            f.write('%d,%s,%s,%d,%d,%d\n' % (k, nm, hx, r, g, b))
    clr_body = ''.join('%d %d %d %d\n' % (k, r, g, b) for k, nm, hx, r, g, b in rows)
    for fn in ('颜色表.clr', '地类颜色表.clr'):      # 后者=第一次交付同名别名，字节一致
        with open(os.path.join(out_dir, fn), 'w', encoding='utf-8') as f:
            f.write('# QGIS/ArcGIS color file: value R G B (0-255)\n' + clr_body)
    L = ['# 第二次分类交付 · 地类编码 · 颜色 · 名称对照表（川/粤/宁 2023）', '',
         '> 色源与第一次五省交付同源（`d_tiles.COLORS`），像元值=下表编码；**0=nodata（透明）**；25–255 未使用。',
         '> 交付 GeoTIFF 已内嵌同一调色板，QGIS 拖入即彩色；ArcGIS 可导入 `颜色表.clr`（`value R G B`）。', '',
         '| 编码 | 地类名称 | 颜色 | HEX | RGB(0-255) | 本次交付（川/粤/宁） |',
         '|---|---|---|---|---|---|']
    for k, nm, hx, r, g, b in rows:
        L.append('| %d | %s | ██ | `%s` | %d, %d, %d | %s |' % (
            k, nm, hx, r, g, b, '出现' if k in present_codes else '未出现'))
    L += ['', '## 怎么用', '',
          '- **QGIS**：拖入 `成品/分省/<省>_2023_10m.tif` 即按内置调色板显示；或图层属性 → 符号化 →',
          '  调色板/唯一值 → 导入 `成品/色卡/颜色表.clr`。',
          '- **ArcGIS**：符号系统 → 色彩映射表(Colormap) → 导入 `颜色表.clr`。',
          '- 含本表的 `地类颜色表.clr` 为第一次交付同名别名（内容一致），二选一即可。', '']
    with open(os.path.join(out_dir, '地类颜色对照.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    return rows


def make_preview(tif_fp, png_fp):
    """调色板渲染缩略图（长边 ≤MAX_DIM，nearest 抽稀；.ovr 就绪后 GDAL 抽稀读路径自动利用）。"""
    PC.preview_png(tif_fp, png_fp, max_dim=MAX_DIM)
    return png_fp


# ======================================================================
# 四、第一次面积表对账（冒烟自检主证据；正式运行可选）
# ======================================================================
def ref_area_check(stats, ref_dir, dn, year=YEAR):
    """交付统计 vs 第一次面积表（面积表_<交付短名>_<year>_10m.csv）逐类对账。"""
    fp = os.path.join(ref_dir, '面积表_%s_%d_10m.csv' % (dn, year))
    if not os.path.exists(fp):
        return None
    ref = {}
    with open(fp, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            try:
                ref[int(r['类码'])] = (int(r['像元数']), float(r['面积_km2']))
            except (KeyError, ValueError):
                continue
    max_dpx, tot_dpx, diff_classes = 0, 0, 0
    for k in range(1, 25):
        p1 = ref.get(k, (0, 0.0))[0]
        p2 = stats['px_by_class'].get(k, 0)
        max_dpx = max(max_dpx, abs(p2 - p1))
        tot_dpx += p2 - p1
        if p2 != p1:
            diff_classes += 1
    ref_n = sum(v[0] for v in ref.values())
    ref_km2 = sum(v[1] for v in ref.values())
    return dict(ref_csv=fp, ref_px=ref_n, ref_km2=round(ref_km2, 3),
                second_px=stats['data_px'], second_km2=stats['data_km2'],
                d_px=stats['data_px'] - ref_n, max_abs_dpx_class=max_dpx,
                diff_classes=diff_classes,
                d_pct=(100 * (stats['data_km2'] - ref_km2) / ref_km2) if ref_km2 else None)


# ======================================================================
# 五、评估（c4 产物收集 + compare 合并；无则清单占位）
# ======================================================================
def eval_caliber(ev):
    """评估分类图口径识别（瓦片口径 / 分省成品口径 / 合幅口径）。"""
    if not ev:
        return '（缺 eval2 结果）'
    ll = ev.get('local_list') or []
    s = ','.join(ll) if ll else str(ev.get('local') or '')
    if not s:
        return '（未记录分类图来源）'
    if 'rasters_batch' in s:
        return '瓦片口径（--local 逐瓦栅格）'
    if 'mosaic2' in s or '成品' in s:
        return '分省成品口径'
    if '五省合幅' in s:
        return '第一次合幅口径'
    return '其它：%s' % os.path.basename(s.split(',')[0])


def collect_eval(eval_dir, provs, out_dir, year=YEAR):
    """复制 eval2_{省}_{year}.json/.md → 评估/{交付短名}_eval2.{json,md}；
    合并 compare_run1_run2_{省}.md → 评估/compare_run1_run2.md（缺者清单占位）；
    附 c5 栅格级对比一览（读 json，只引用路径不复制）。返回 {short: eval_json_dict}。"""
    os.makedirs(out_dir, exist_ok=True)
    evals, copied, missing = {}, [], []
    for short in PROV2:
        if short not in provs:
            continue
        dn = deliv_of(short)
        js = os.path.join(eval_dir, 'eval2_%s_%d.json' % (short, year))
        md = os.path.join(eval_dir, 'eval2_%s_%d.md' % (short, year))
        ev = PC.jload(js)
        if ev is not None:
            evals[short] = ev
            shutil.copy2(js, os.path.join(out_dir, '%s_eval2.json' % dn))
            copied.append('%s_eval2.json' % dn)
        else:
            missing.append(js)
        if os.path.exists(md):
            shutil.copy2(md, os.path.join(out_dir, '%s_eval2.md' % dn))
            copied.append('%s_eval2.md' % dn)
    # ---- 实跑小结 / 三省汇总 / c5 拼接对比报告（有则收；README/本文件核心结论的可追溯源） ----
    extras = [os.path.join(eval_dir, '三省_实跑汇总.md')]
    for short in PROV2:
        if short in provs:
            extras.append(os.path.join(eval_dir, '%s_实跑小结.md' % deliv_of(short)))
    extras.append(os.path.join(REPORTS2, '拼接对比实跑报告.md'))
    for sfp in extras:
        if os.path.exists(sfp):
            dst_name = os.path.basename(sfp)
            if dst_name == '拼接对比实跑报告.md':
                dst_name = '拼接对比实跑报告_c5.md'      # 与 c4 层件区分
            shutil.copy2(sfp, os.path.join(out_dir, dst_name))
            copied.append(dst_name)
    # ---- compare_run1_run2.md 合并 ----
    L = ['# 第二次 × 第一次 · 评估层对比（川/粤/宁 · %d）' % year, '',
         '> 来源：c4_eval_second.py --compare 产出（`reports/eval2/compare_run1_run2_<省简称>.md`），',
         '> 本文件为合并副本；口径：同窗同口径（仅分类图与训练池排除不同），Δ = 第二次 − 第一次。',
         '> 判据带注记：单次种子噪声 ±1.5 pp，**<2 pp 的单次差异不可信**（AGENTS 硬约束 12）；',
         '> 来源分层（全点 / fcs10 / 非fcs10）为强制口径。', '']
    got = 0
    for short in PROV2:
        if short not in provs:
            continue
        cmpf = os.path.join(eval_dir, 'compare_run1_run2_%s.md' % short)
        dn = deliv_of(short)
        if os.path.exists(cmpf):
            got += 1
            L += ['', '---', '', '## %s（%s）' % (dn, short), '',
                  '> 原件：`%s`' % cmpf, '']
            with open(cmpf, encoding='utf-8') as f:
                L += f.read().rstrip('\n').splitlines()
        else:
            L += ['', '---', '', '## %s（%s）' % (dn, short), '',
                  '- [占位] 对比文件缺失：`%s`' % cmpf,
                  '- 就绪后运行：`python 代码/9.第二次分类川粤宁/c4_eval_second.py --compare-only --prov %s`，'
                  '并重跑 c6 纳入。' % short]
    # 附：栅格级对比一览（c5，只引用）
    L += ['', '---', '', '## 附：栅格级对比（c5_mosaic_compare，只引用不复制）', '']
    for short in PROV2:
        if short not in provs:
            continue
        cj = os.path.join(REPORTS2, 'compare_%s_run1_run2.json' % short)
        d = PC.jload(cj)
        dn = deliv_of(short)
        if d and d.get('agree_fullpass') is not None:
            L.append('- **%s**：共同域 %.4f 亿 px，全通过一致率 **%s**，变化 %.4f%%；'
                     '缝带 %.4f%% vs 域内 %.4f%%；%.1f→%.1f km²（见 `%s`）' % (
                         dn, (d.get('common_px') or 0) / 1e8, d['agree_fullpass'],
                         d.get('change_pct_of_common') or 0, d.get('seam_change_pct') or 0,
                         d.get('interior_change_pct') or 0,
                         d.get('first_km2') or 0, d.get('second_km2') or 0, cj))
        else:
            L.append('- **%s**：c5 栅格对比未就绪（`%s`）' % (dn, cj))
    L.append('')
    with open(os.path.join(out_dir, 'compare_run1_run2.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    return evals, copied, missing, got


# ======================================================================
# 六、EECU 台账摘要（读 ledger，不复制原始 CSV）
# ======================================================================
def _fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return 0.0


def read_ledger_rows(path):
    """兼容两种台账行式：c1 5 列（time,acct,task_id,desc,eecu_h）与 c3 8 列
    （time,acct,tile,task_id,desc,state,eecu_h,note）——按列数解析，混排不崩。"""
    out = []
    with open(path, encoding='utf-8', errors='replace', newline='') as f:
        for row in csv.reader(f):
            if not row:
                continue
            if 'eecu_h' in row:
                continue                              # 表头
            if len(row) >= 8:
                out.append(dict(time=row[0], acct=row[1], tile=row[2], task_id=row[3],
                                desc=row[4], state=row[5], eecu_h=_fnum(row[6]),
                                note=row[7] if len(row) > 7 else ''))
            elif len(row) == 5:
                out.append(dict(time=row[0], acct=row[1], tile='', task_id=row[2],
                                desc=row[3], state='', eecu_h=_fnum(row[4]), note=''))
            elif len(row) == 7:      # 兜底：无 note 的 7 列变体
                out.append(dict(time=row[0], acct=row[1], tile='', task_id=row[2],
                                desc=row[3], state=row[4], eecu_h=_fnum(row[5]),
                                note=row[6]))
    return out


def build_ledger_summary(out_fp, smoke=False):
    """生产台账摘要：EECU 汇总（按 desc 取最终值防多状态行重复）+ 覆盖面核查。原文路径只引用。"""
    ledgers = sorted(glob.glob(os.path.join(LEDGER_DIR, '*.csv')))
    # [2026-10-15 正式运行修正] 排除逐瓦状态板（board*.csv）：其 eecu 列与 eecu_ledger 的
    # p5t_* 行同源（逐瓦终值），混入 by_desc 汇总会重复计——41 瓦批实测虚增 ~180 EECU·h
    # （860.08 → 1040.22）；board 仅在下方"原始台账路径"按路径引用。
    ledgers = [fp for fp in ledgers if 'board' not in os.path.basename(fp).lower()]
    cfg = PC.jload(CONFIG2) or {}
    budget = cfg.get('budget_eecu')
    plan = read_plan2()
    by_prov = {}
    for r in plan:
        by_prov[r['prov']] = by_prov.get(r['prov'], 0) + 1
    rows_all, per_file = [], {}
    for fp in ledgers:
        rows = read_ledger_rows(fp)
        per_file[fp] = len(rows)
        rows_all += rows
    # 按 desc 取最大（任务最终值）——同任务多状态行/两路赛跑不重复计
    by_desc, by_tile, states = {}, {}, {}
    for r in rows_all:
        key = r['desc'] or r['task_id'] or '?'
        by_desc[key] = max(by_desc.get(key, 0.0), r['eecu_h'])
        if r['tile']:
            by_tile[r['tile']] = max(by_tile.get(r['tile'], 0.0), r['eecu_h'])
        if r['state']:
            states[r['state']] = states.get(r['state'], 0) + 1
    # 账号级：按 (acct, desc) 去重后求和
    acct_desc = {}
    for r in rows_all:
        if not r['acct']:
            continue
        k = (r['acct'], r['desc'] or r['task_id'])
        acct_desc[k] = max(acct_desc.get(k, 0.0), r['eecu_h'])
    by_acct = {}
    for (a, _), v in acct_desc.items():
        by_acct[a] = by_acct.get(a, 0.0) + v
    total = sum(by_desc.values())
    t0 = min((r['time'] for r in rows_all), default='—')
    t1 = max((r['time'] for r in rows_all), default='—')
    batch_tiles = [r['tile'] for r in rows_all if r['tile']]
    tiles_batch_dir = sorted(glob.glob(os.path.join(TILES2, 'T*_10m.tif')))
    n_qa_tile = len(glob.glob(os.path.join(QADIR2, 'T*_qa.json')))
    mosaic2_present = [os.path.basename(p) for p in
                       sorted(glob.glob(os.path.join(MOSAIC2, '*_2023_10m.tif')))]
    L = ['# 生产台账摘要（第二次分类 · 川/粤/宁 2023）', '']
    if smoke:
        L += ['> ⚠️ 冒烟自检件（非交付）：输入=第一次交付件，台账为当前真实台账快照。', '']
    L += ['> 生成：c6_deliver.py ｜ %s ｜ **原始逐行 CSV 不复制入交付包**，权威文件见文末路径。' %
          time.strftime('%Y-%m-%d %H:%M:%S'), '']
    CN = '一二三四五六七八九十'
    si = [0]

    def sec(title):
        si[0] += 1
        L.append('## %s、%s' % (CN[si[0] - 1], title))
        L.append('')

    sec('范围与预算')
    L += ['- 本次生产：**%d 瓦（%s）**，%d 年；计划 `%s`' % (
              len(plan), ' / '.join('%s %d' % (k, v) for k, v in by_prov.items()), YEAR, PLAN2),
          '  （含补瓦 T2202：v2 修复 v1 框架东缘截断的粤东 117.0–117.24°E 条带后按 41 瓦重拼重裁）',
          '- 预算：`config.json budget_eecu = %s EECU·h`（含批 1；项目级硬上限口径）' % budget,
          '- 样本池：`train2023_v2`（1,764,020 点 / 23 类；缺 8 针阔混交林，按用户指令弃用）', '']
    sec('EECU 台账汇总')
    L += ['> 口径：按 `desc`（任务）取**最终值**（最大 eecu_h）后求和，消除多状态行/重试行重复；',
          '> 该口径与"任务实际计费"一致（同一任务多行状态只计最终 EECU）。', '',
          '| 指标 | 值 |', '|---|---|',
          '| 合计 EECU·h | **%.2f**%s |' % (
              total, ('（预算 %s 的 %.1f%%）' % (budget, 100 * total / budget)) if budget else ''),
          '| 台账文件数 | %d |' % len(ledgers),
          '| 台账行数（含中间状态） | %d |' % len(rows_all),
          '| 时间跨度 | %s ~ %s |' % (t0, t1),
          '| 涉及账号数 | %d |' % len([a for a in by_acct if a]),
          '| 生产瓦片（去重） | %d |' % len(set(batch_tiles)), '']
    if by_acct:
        sec('逐账号（ECU·h）')
        L += ['| 账号 | EECU·h |', '|---|---|']
        for a, v in sorted(by_acct.items(), key=lambda x: -x[1]):
            L.append('| %s | %.2f |' % (a, v))
        L.append('')
    if by_tile:
        sec('逐瓦（ECU·h，取该瓦任务最终值）')
        L += ['| 瓦片 | EECU·h |', '|---|---|']
        for t, v in sorted(by_tile.items(), key=lambda x: -x[1]):
            L.append('| %s | %.2f |' % (t, v))
        L.append('')
    if states:
        sec('状态计数（行口径）')
        L += ['| state | 行数 |', '|---|---|']
        for s, n in sorted(states.items(), key=lambda x: -x[1]):
            L.append('| %s | %d |' % (s, n))
        L.append('')
    sec('生产覆盖面核查')
    n_plan = len(plan)
    exp_txt = '%d（%s）' % (n_plan, '，'.join('%s%d' % (k, v) for k, v in by_prov.items()))
    L += ['| 项 | 现状 | 期望 |', '|---|---|---|',
          '| plan 瓦片数 | %d（%s） | %s |' % (
              len(plan), '，'.join('%s%d' % (k, v) for k, v in by_prov.items()), exp_txt),
          '| rasters_batch 命中 | %d | %d |' % (len(tiles_batch_dir), n_plan),
          '| 逐瓦 QA jsons | %d | %d |' % (n_qa_tile, n_plan),
          '| mosaic2 省口径成品 | %d（%s） | 3（四川/广东/宁夏） |' % (
              len(mosaic2_present), '，'.join(mosaic2_present) or '无'), '']
    sec('原始台账路径（不复制）')
    for fp in ledgers:
        L.append('- `%s`（%d 行）' % (fp, per_file[fp]))
    if not ledgers:
        L.append('- （ledger 目录暂无 CSV：%s）' % LEDGER_DIR)
    for fp in sorted(glob.glob(os.path.join(LEDGER_DIR, 'board*.csv'))):
        L.append('- `%s`（逐瓦状态板，**不计入上方 EECU 汇总**：逐瓦 eecu 已由 eecu_ledger 的 '
                 'p5t_* 行覆盖，重复计会虚增）' % fp)
    L += ['- 补充：逐瓦 QA `%s/{tile}_qa.json`；生产驱动 `c3_driver_prod2.py`（工作区 run/prod2_driver.py）。'
          % QADIR2, '']
    with open(out_fp, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    return dict(total_eecu=total, n_rows=len(rows_all), n_files=len(ledgers),
                n_acct=len(by_acct), n_tiles=len(set(batch_tiles)))


# ======================================================================
# 七、README（对齐 run-1 make_readme 模板；数字全部取自 QA/评估产物）
# ======================================================================
def build_readme(dest, ctx):
    smoke = ctx['smoke']
    provs = ctx['order']                       # [(short, name, qa_entry)]
    L = []
    A = L.append
    scope = '/'.join(nm for _, nm, _ in provs) + ('·冒烟' if smoke else '')
    A('# 交付包 · 中国 10 m 土地覆盖 2023 · 第二次分类（%s）' % scope)
    A('')
    if smoke:
        A('> ⚠️ **冒烟自检件（非交付）**：输入=第一次五省交付成品（只读），用于验证 c6 打包链路；')
        A('> 正式交付请待第二次 41 瓦成品就绪后按 `代码/9.第二次分类川粤宁/README_c6.md` 重跑。')
        A('')
    A('> 生产链：AlphaEarth Foundations 年度嵌入（64 维/10 m）+ GEE 服务端随机森林，逐瓦生产 →')
    n_plan = ctx.get('n_plan') or len(read_plan2())
    A('> %d/%d 瓦（冻结计划 `plan/tiles_secondrun.csv`：川 24 + 粤 10 + 宁 6 + 补瓦 T2202）逐瓦 QA 后按 10 m'
      % (n_plan, n_plan))
    A('> 全局同锚方度格零重采样镶嵌。类别栅格全链 nearest 语义（同格直拷/像元中心包含裁剪），')
    A('> 未引入任何非最近邻重采样。')
    A('> 交付日期：%s ｜ 组装：c6_deliver.py（c5 拼接 → c6 交付打包）' % time.strftime('%Y-%m-%d'))
    A('')
    # 一、本次范围与样本版本
    pool = ctx['pool']
    A('## 一、本次范围与样本版本')
    A('')
    A('- **本次 = 第二次分类**：川 24 / 粤 10 / 宁 6 + **补瓦 T2202**（闽；v2 修复 v1 框架东缘'
      '截断的粤东 117.0–117.24°E 条带）= **41 瓦**，2023 年（计划表逐瓦 land_frac>0）。')
    A('- **训练池 v2 = %s 点 / %d 类**（缺 8 针阔混交林，按用户指令弃用）；来源=2023 修复后新池'
      % ('{:,}'.format(pool['n']), 24 - len(pool.get('missing_codes') or [])))
    A('  `train2023_v2`（第一次旧池 `smp2023_merged` 2,060,577 点的清洗修复版）。')
    A('- 其余生产配置（AEF 嵌入/随机森林/分区/镶嵌/裁剪/概览纪律）**全部冻结继承第一次五省批**。')
    A('')
    # 二、差异声明
    A('## 二、与第一次的差异声明')
    A('')
    A('| 项 | 第一次（五省批） | 第二次（本次） |')
    A('|---|---|---|')
    A('| 样本池 | `smp2023_merged` 2,060,577 点 / 24 类 | `train2023_v2` **%s 点 / 23 类**（修复后新池；缺 8） |'
      % '{:,}'.format(pool['n']))
    A('| 模型/特征/超参 | GEE 服务端 RF（冻结） | **冻结继承**（未改） |')
    A('| 网格/镶嵌/裁剪口径 | 10m 原生方度格、nearest、DataV buffer -0.005° | **冻结继承**（未改） |')
    A('| 概览纪律 | 主 tif 单页 + 外置 .ovr | **冻结继承**（c6 外置生成/复制） |')
    A('| 评估 | evalprep（同窗同口径） | c4 同窗同口径，仅分类图与训练池排除不同（见 评估/compare_run1_run2.md） |')
    A('')
    A('- 差异仅此两处（样本 + 分期覆盖省）；**任何单次 <2 pp 的评估差异不可归因于方法变化**'
      '（单次种子噪声 ±1.5 pp，AGENTS 硬约束 12）。')
    A('')
    A('### 核心评估结论（照抄 `reports/eval2/三省_实跑汇总.md` §4.1；原件已收入 `成品/评估/`）')
    A('')
    A('- **全点口径不得作发布指标**：全点 OA24 三省 −14.9 ~ −21.8pp；fcs10 层 96.4–98.2% 为同源')
    A('  灌丛评估点（`src=glc_fcs10_2023_shrub`）——第二次训练池撤掉 FCS10 灌丛惯例点后，模型在')
    A('  灌丛样区不再外推为灌丛（召回 0.3–19.9%），**全点被 fcs10 灌丛同源点机械拉低**')
    A('  （E13 预注册结论的生产端再现）。**在 M3 人工判读裁决灌丛口径之前，全点 OA 不得作为发布指标。**')
    A('- **真正可比的非fcs10 净窗口三省分岔**：宁夏 **+5.5pp**（真改善，超噪声带）；'
      '广东 +1.3pp（噪声带内，轻微中性偏正、结构有损）；四川 **−3.8pp**（净退化，超噪声带）。')
    A('- 发布建议：三省均以非fcs10 净窗口 + M3 判读结论联合定稿；四川列入改进省。')
    A('')
    A('- 口径可比性（三省汇总 §2/§4.2）：两轮同留出池/同归省（DataV）/同 miss 口径，**非fcs10 层')
    A('  形态效应实测 0.0000pp** → 分层 Δ 属模型（池）变化；全点/fcs10 层形态成分 <1.2pp，方向对')
    A('  第二次有利，量级不足以解释任何大降。')
    A('')
    # 三、目录树
    A('## 三、目录树')
    A('')
    A('```')
    A('%s/' % dest.replace('\\', '/'))
    A('├── 交付包_README.md            本文件')
    A('├── MANIFEST_MD5.txt            全交付文件 MD5（相对路径+MD5+字节）')
    A('├── 成品/')
    A('│   ├── 分省/      {%s}_2023_10m.tif + .ovr + .clr （省界 buffer -0.005°）'
      % ','.join(nm for _, nm, _ in provs))
    A('│   ├── 预览/      {省}_预览10m.png（内置调色板渲染，长边 ≤%d px）' % MAX_DIM)
    A('│   ├── 面积表/    {省}_面积.csv + 汇总.csv（10m 原生格纬度加权）')
    A('│   ├── 色卡/      地类颜色对照.md + 地类颜色表.csv + 颜色表.clr（+地类颜色表.clr 别名）')
    A('│   ├── QA/        {省}_qa.json + qa汇总.md（+ c5 面积详表副本，有则收）')
    A('│   └── 评估/      {省}_eval2.json/.md + compare_run1_run2.md（合并）+ {省}_实跑小结.md'
      ' + 三省_实跑汇总.md（有则收）')
    A('│                 （+ 拼接对比实跑报告_c5.md，有则收）')
    A('└── 台账/          生产台账摘要.md（引用 ledger，不复制原始 CSV）')
    A('```')
    A('')
    # 四、成品清单
    A('## 四、成品清单（含 MD5）')
    A('')
    A('| 成品 | 尺寸 px | 文件 | 数据像元 | 覆盖%(像元口径) | 覆盖%(面积口径) | 类数 | MD5 (tif) | .ovr |')
    A('|---|---|---|---|---|---|---|---|---|')
    for short, nm, e in provs:
        st, meta = e['stats'], e['meta']
        ovr_sz = ('%.1f MB' % (os.path.getsize(e['tif'] + '.ovr') / 1e6)
                  if os.path.exists(e['tif'] + '.ovr') else '—')
        A('| %s（buffer -0.005°） | %d×%d | %.2f GB | %d | %.2f | %.2f | %d | `%s` | %s |' % (
            nm, meta['width'], meta['height'], meta['size_gb'], st['data_px'],
            st['coverage_px_pct'], st['coverage_km2_pct'], st['n_classes'], e['md5'], ovr_sz))
    A('')
    A('- 每件成品：uint8 单波段 + 内置 256 项调色板 + nodata=0 + EPSG:4326 + 10 m 原生方度格')
    A('  （px=py=%.15e°）；**主 tif 无内建概览**（单 IFD 页），金字塔=外置 `.ovr`。'
      % provs[0][2]['meta']['px'])
    A('- **覆盖率双口径分报**（覆盖%按全幅像元 / 按有值域面积），防混单位虚报（AGENTS 硬约束 14）。')
    if not smoke and any(e['meta']['internal_ov'] for _, _, e in provs):
        A('- ⚠️ 检出主 tif 含内建概览页（%s）——与交付纪律不符，请核查 c5 是否误用 --ovr scaffold。'
          % '，'.join(nm for _, nm, e in provs if e['meta']['internal_ov']))
    A('')
    # 五、快速上手
    A('## 五、快速上手（QGIS / ArcGIS）')
    A('')
    A('1. **QGIS**：直接把 `成品/分省/*.tif` 拖入地图（调色板已内置）；外部色表用 `成品/色卡/颜色表.clr`。')
    A('2. **ArcGIS**：符号系统 → 色彩映射表(Colormap) → 导入 `成品/色卡/颜色表.clr`（`value R G B`）。')
    A('3. **统计/派生**：一律 10m 原生网格直读或显式 nearest；**禁止非最近邻重采样**'
      '（60 m 重采样会凭空造类，实测 10 m 逐点仲裁 96.7% vs 60 m 76.0%）。')
    A('')
    # 六、省面积对账
    A('## 六、省面积对账（10 m 原生格纬度加权 vs DataV 全域球面）')
    A('')
    A('| 省 | DataV 全域球面 km² | 计划表合计（主省归属）km² | 本成品 km² | 成品 vs DataV |')
    A('|---|---|---|---|---|')
    tot_d = tot_p = tot_c = 0.0
    for short, nm, e in provs:
        st = e['stats']
        dv = e.get('datav_km2') or 0
        pa = e.get('plan_km2') or 0
        tot_d += dv
        tot_p += pa
        tot_c += st['data_km2']
        A('| %s | %.1f | %.1f | %.1f | %s |' % (
            nm, dv, pa, st['data_km2'],
            '%+.2f%%' % (100 * (st['data_km2'] - dv) / dv) if dv else '—'))
    if len(provs) > 1:
        A('| 合计 | %.1f | %.1f | %.1f | %s |' % (
            tot_d, tot_p, tot_c,
            '%+.2f%%' % (100 * (tot_c - tot_d) / tot_d) if tot_d else '—'))
    A('')
    A('> 成品低于 DataV 全域的构成：① 边界 **buffer %.3f°（≈500 m）内缩**（周长×0.5 km 边界带）；'
      '② 像元中心包含 + 平面弧长加权 vs 球面多边形口径差（<±0.1%%）。第一次同口径实测 −0.75%%~−1.90%%。'
      % BUFFER_DEG)
    A('')
    # 七、分省评估
    A('## 七、分省评估（c4 · 同窗同口径，开发集）')
    A('')
    A('> 判据（沿用第一次 G7 口径，供参考判定）：**24 类 OA ≥ 0.80 且 9 大类 OA ≥ 0.88**；')
    A('> 双口径=全点 / 非fcs10 分层（强制）。n<30 的类只作合并级参考。')
    if ctx['evals']:
        A('')
        A('| 省 | n(计入) | 覆盖% | 24类OA 全点 | 9大类OA 全点 | 24类OA 非fcs10 | 9大类OA 非fcs10 | 判定 |')
        A('|---|---|---|---|---|---|---|---|')
        for short, nm, e in provs:
            ev = ctx['evals'].get(short)
            if not ev:
                A('| %s | — | — | — | — | — | — | （eval2 缺失，见 评估/compare_run1_run2.md 占位） |' % nm)
                continue
            st_all = (ev.get('strata') or {}).get('all') or {}
            st_nf = (ev.get('strata') or {}).get('nonfcs10') or {}
            oa, m9 = st_all.get('OA'), st_all.get('m9')
            nf_oa, nf_m9 = st_nf.get('OA'), st_nf.get('m9')
            ok_a = oa is not None and m9 is not None and oa >= 0.80 and m9 >= 0.88
            ok_n = nf_oa is not None and nf_m9 is not None and nf_oa >= 0.80 and nf_m9 >= 0.88
            concl = '双口径达标' if (ok_a and ok_n) else (
                '未达标' if not ok_a and not ok_n else
                '部分达标（全点%s / 非fcs10%s）' % ('✓' if ok_a else '✗', '✓' if ok_n else '✗'))
            A('| %s | %s | %s | %s | %s | %s | %s | %s |' % (
                nm, ev.get('n_scored'), ev.get('coverage_pct'),
                '%.4f' % oa if oa is not None else '—', '%.4f' % m9 if m9 is not None else '—',
                '%.4f' % nf_oa if nf_oa is not None else '—',
                '%.4f' % nf_m9 if nf_m9 is not None else '—', concl))
        A('')
        cals = {eval_caliber(ctx['evals'].get(s)) for s, _, _ in provs if ctx['evals'].get(s)}
        A('> **口径注记**：本轮 c4 评估分类图=%s。与交付分省成品同形口径可复跑：'
          '`python 代码/9.第二次分类川粤宁/c4_eval_second.py --prov all --local <交付tif逗号分隔>`。'
          % ('；'.join(sorted(cals)) if cals else '未知'))
        A('> 逐类 F1 与对比详表：`评估/compare_run1_run2.md`、`评估/<省>_eval2.md`、'
          '`评估/三省_实跑汇总.md`。')
        A('> **发布口径提示**：上表「全点」列按 `三省_实跑汇总.md` §4.1 不得作发布口径'
          '（fcs10 灌丛同源点主导）；发布以**非fcs10 净窗口 + M3 判读结论**联合定稿。')
    else:
        A('')
        A('- [占位] 本轮未收集到 eval2 结果：预期路径 `%s/eval2_<省>_2023.json`。' % EVAL2_DIR)
    A('')
    # 八、口径声明
    A('## 八、口径声明（引用前必读）')
    A('')
    A('1. **10 m 原生网格 / nearest 全链**：成品为 EPSG:4326 原生 10 m 方度格；一切派生/统计直读'
      '原生格或显式 nearest，未做任何非最近邻重采样（禁 GDAL 默认双线性/立方）。')
    A('2. **省界与 buffer**：DataV 省界（GCJ-02）原样使用 + `buffer %.3f°`（≈500 m 内缩），'
      '像元中心点在多边形内→保留（与第一次交付、c5 裁剪同式；不做 gcj2wgs 反演）。' % BUFFER_DEG)
    A('3. **近岸海面默认保留**（政策 A）：瓦片矩形内近岸水域参与分类（水体/潮滩类值），本次并入省域；'
      '如改政策 B（按海岸线置 nodata）需重裁并重出本册。')
    A('4. **覆盖率双口径**：像元口径（占全幅）与面积口径（占有值域）分开报，见第四节。')
    A('5. **评估噪声**：单次种子噪声 ±1.5 pp，<2 pp 的单次差异不可信；来源分层（fcs10/非fcs10）'
      '为强制口径；全点 OA 受 FCS10 标注惯例主导。')
    A('6. **瓦片口径 ≠ 合幅口径**：同一瓦片集与合幅成品可差 ±0.4 pp（c4/c5 实测）；'
      '引用评估 Δ 时看第七节口径注记。')
    A('')
    # 九、逐文件清单
    A('## 九、逐文件清单')
    A('')
    files = ctx['files']                       # [(rel, md5, bytes)]（不含 README/MANIFEST）
    A('| 相对路径 | 字节 | MD5 |')
    A('|---|---|---|')
    for rel, md5, n in files:
        A('| `%s` | %d | `%s` |' % (rel, n, md5))
    A('| `交付包_README.md` | 本文件 | 见 MANIFEST_MD5.txt |')
    A('| `MANIFEST_MD5.txt` | 全文件 MD5 清单（不自我哈希） | — |')
    A('')
    A('> 合计 %d 件（含本文件与 MANIFEST，共 %d 件）；MD5 为整文件字节 MD5（1<<22 分块）。'
      % (len(files), len(files) + 2))
    A('')
    # 十、复现与工具
    A('## 十、复现与工具')
    A('')
    A('- 拼接/裁剪（第二次，v2 41 瓦重拼）：`python 代码/9.第二次分类川粤宁/c5_mosaic_compare.py '
      'mosaic --prov 川,粤,宁 --tiles <41 瓦·末位 T2202> --force --ovr none`')
    A('- 交付打包（本册）：`代码/9.第二次分类川粤宁/c6_deliver.py`（结构对齐第一次 `delivery_pkg/pipeline_after_mosaic.py`）')
    A('- 外置金字塔：`F:/lc_work/prod5p_2023/delivery_pkg/ov_ext_big.py`（流式写出+全瓦读回验收；'
      '复制路线做逐级随机瓦片读回抽检）')
    A('- 点评估：`代码/9.第二次分类川粤宁/c4_eval_second.py --prov all --compare`')
    A('- 逐瓦生产：`代码/9.第二次分类川粤宁/c3_driver_prod2.py`（台账 `prod2nd_2023/ledger/`）')
    A('')
    # 十一、blockers 与移交项
    A('## 十一、blockers 与移交项')
    A('')
    A('- 无阻塞性 blocker。')
    A('- 移交项：① M3 独立人工验证点判读（≥3,000 点）待用户侧；② 全国 313 瓦 × 8 年放量待下令；')
    A('  ③ 宁夏 9 大类 OA 未达标（第一次遗留）→ 本批复评见第七节；④ 近岸海面政策 A/B 待定；')
    A('  ⑤ 灌丛/森林/湿地图例瘦身决策（doc 54）待用户定夺。')
    if smoke:
        A('- 冒烟输入件：' + '；'.join('%s ← `%s`' % (nm, e['src']) for _, nm, e in provs) + '（只读复用）。')
    A('')
    out_fp = os.path.join(dest, '交付包_README.md')
    with open(out_fp, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    return out_fp


# ======================================================================
# 八、MANIFEST
# ======================================================================
def list_files(dest):
    out = []
    for root, dirs, fs in os.walk(dest):
        dirs.sort()
        # [2026-10-15 正式运行修正] 交付清单排除下划线前缀草稿件（<dest>/_smoke 冒烟自检目录、
        # 运行日志草稿等与正式包同根共存；不排除会把它们写进正式 MANIFEST/README 清单）。
        dirs[:] = [d for d in dirs if not d.startswith('_')]
        for fn in sorted(fs):
            if fn.startswith('_'):
                continue
            fp = os.path.join(root, fn)
            rel = os.path.relpath(fp, dest).replace('\\', '/')
            if rel == 'MANIFEST_MD5.txt':
                continue
            out.append((rel, md5_file(fp), os.path.getsize(fp)))
    return out


def write_manifest(dest):
    files = list_files(dest)
    fp = os.path.join(dest, 'MANIFEST_MD5.txt')
    with open(fp, 'w', encoding='utf-8') as f:
        f.write('# MANIFEST_MD5.txt — 第二次分类交付包全文件清单（相对交付根目录）\n')
        f.write('# 生成：c6_deliver.py ｜ %s ｜ 列：MD5<TAB>字节<TAB>相对路径\n'
                % time.strftime('%Y-%m-%d %H:%M:%S'))
        f.write('# 校验（Git Bash）：cd <交付根> && awk -F"\\t" \'{print $1"  "$3}\' '
                'MANIFEST_MD5.txt | grep -v "^#" | md5sum -c -\n')
        f.write('# %d 件（不含本文件）\n' % len(files))
        for rel, md5, n in files:
            f.write('%s\t%d\t%s\n' % (md5, n, rel))
    return fp, files


# ======================================================================
# 九、主流程
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
            raise SystemExit('省 %s 不在第二次交付范围（%s）' % (s, ','.join(PROV2)))
        if short not in out:
            out.append(short)
    return out


def build_one(short, a, dest, names):
    """单省打包：门禁 → 复制 tif/clr → 挂 .ovr → MD5 → 统计 → 面积表 → 预览 → QA json。"""
    full = PC.PROV_ALIAS.get(short, short)
    nm = deliv_of(short)
    src = os.path.join(a.src_dir, '%s_%d_10m.tif' % (nm, a.year))
    if not os.path.exists(src):
        emit('[警告] %s 成品缺失（待生产）：%s' % (nm, src))
        return None
    emit('== %s（%s）== 源：%s' % (nm, short, src))
    meta = gate_product(src)
    if meta['internal_ov']:
        raise SystemExit('%s 主 tif 含内建概览页（%d 页）——违反交付纪律（doc43）。'
                         '请确认 c5 mosaic 未误用 --ovr scaffold；交付件须单页+外置 .ovr。'
                         % (src, meta['tif_pages']))
    # 1) 复制 tif + clr
    prov_dir = os.path.join(dest, '成品', '分省')
    os.makedirs(prov_dir, exist_ok=True)
    dst = os.path.join(prov_dir, os.path.basename(src))
    shutil.copy2(src, dst)
    md5 = md5_file(dst)
    md5_src = md5_file(src)
    assert md5 == md5_src, '复制后 MD5 不一致（%s）' % src
    emit('  复制 → %s（MD5 %s 与源一致 ✓）' % (os.path.basename(dst), md5[:12]))
    clr_src = os.path.splitext(src)[0] + '.clr'
    clr_dst = os.path.splitext(dst)[0] + '.clr'
    if os.path.exists(clr_src):
        shutil.copy2(clr_src, clr_dst)
    else:
        PC.write_clr(clr_dst)                    # 兜底：与内嵌调色板同源生成
    # 2) 外置 .ovr
    ovr = attach_ovr(src, dst, a.ovr)
    ovr['exists'] = os.path.exists(dst + '.ovr')
    ovr['size_mb'] = round(os.path.getsize(dst + '.ovr') / 1e6, 2) if ovr['exists'] else None
    # 3) 统计（首选 c5 面积 QA 复用；否则 10m 原生网格本脚本重扫）
    st, stats_src = None, ''
    c5q = os.path.join(a.c5_reports_dir, 'qa_%s_%s.json' % (short, a.c5_label))
    if a.stats in ('auto', 'reuse'):
        st = stats_from_c5_qa(c5q)
        if st is not None:
            stats_src = 'c5_qa_json:%s' % c5q
            if st['c5_src'] and os.path.normcase(os.path.normpath(st['c5_src'])) != \
                    os.path.normcase(os.path.normpath(src)):
                emit('  [警告] c5 面积 QA 的 src（%s）≠ 本次源文件——照用但已在 QA json 标注'
                     % st['c5_src'])
    if st is None:
        if a.stats == 'reuse':
            raise SystemExit('--stats reuse 但缺 c5 面积 QA：%s' % c5q)
        emit('  c5 面积 QA 不可用 → 本脚本 10m 原生网格重扫（%s）' % os.path.basename(dst))
        st = scan_stats(dst)
        stats_src = 'own_scan'
    # 4) 面积表
    area_dir = os.path.join(dest, '成品', '面积表')
    os.makedirs(area_dir, exist_ok=True)
    classes = write_area_csv(os.path.join(area_dir, '%s_面积.csv' % nm), st, names)
    # 5) 预览（.ovr 就绪后读路径自动利用）
    prev_dir = os.path.join(dest, '成品', '预览')
    os.makedirs(prev_dir, exist_ok=True)
    png = make_preview(dst, os.path.join(prev_dir, '%s_预览10m.png' % nm))
    # 6) 第一次面积表对账（可选，冒烟证据）
    ref = ref_area_check(st, a.ref_area_dir, nm, a.year) if a.ref_area_dir else None
    # 7) 评估/QA 引用
    ev_js = os.path.join(a.eval_dir, 'eval2_%s_%d.json' % (short, a.year))
    ev_md = os.path.join(a.eval_dir, 'eval2_%s_%d.md' % (short, a.year))
    cmp_md = os.path.join(a.eval_dir, 'compare_run1_run2_%s.md' % short)
    c5_md = os.path.join(a.c5_reports_dir, 'qa_%s_%s.md' % (short, a.c5_label))
    c5_csv = os.path.join(a.c5_reports_dir, 'qa_%s_%s_class.csv' % (short, a.c5_label))
    clip_qa = os.path.join(a.src_dir, '%s_%d_10m_qa.json' % (nm, a.year))
    dv = st.get('c5_datav_km2')
    if dv is None:
        try:
            dv = round(datav_area_km2(short), 2)
        except Exception as e:
            emit('  [警告] DataV 球面面积计算失败：%s' % str(e)[:80])
            dv = None
    plan_km2 = round(sum(r['prov_area_km2'] for r in read_plan2() if r['prov'] == short), 2)
    entry = dict(short=short, name=nm, tif=dst, src=src, md5=md5, src_md5=md5_src,
                 meta=meta, stats=st, stats_src=stats_src, ovr=ovr, classes=classes,
                 png=png, ref=ref, datav_km2=dv, plan_km2=plan_km2,
                 c5_refs=dict(qa_json=c5q, qa_md=c5_md, qa_csv=c5_csv, clip_qa=clip_qa),
                 ev_refs=dict(json=ev_js, md=ev_md, compare=cmp_md))
    # 8) QA json
    qa_dir = os.path.join(dest, '成品', 'QA')
    os.makedirs(qa_dir, exist_ok=True)
    qa = dict(prov=full, prov_tag=short, year=a.year, smoke=bool(a.smoke),
              product=os.path.relpath(dst, dest).replace('\\', '/'), src=src,
              md5=md5, src_md5=md5_src, size_gb=round(meta['size_gb'], 3),
              width=meta['width'], height=meta['height'], dtype=meta['dtype'],
              nodata=meta['nodata'], crs=meta['crs'], px=meta['px'], py=meta['py'],
              left=meta['left'], top=meta['top'], palette_n=meta['n_cmap'],
              tif_pages=meta['tif_pages'], internal_overview=meta['internal_ov'],
              data_px=st['data_px'], n_classes=st['n_classes'],
              px_by_class={str(k): v for k, v in st['px_by_class'].items()},
              area_km2={str(k): v for k, v in st['area_km2'].items()},
              total_km2=st['data_km2'], nodata_km2=st['zero_km2'],
              coverage_px_pct=st['coverage_px_pct'], coverage_km2_pct=st['coverage_km2_pct'],
              stats_source=stats_src, stats_minutes=st.get('minutes'),
              ovr=dict(mode=ovr.get('mode'), ok=ovr.get('ok'), detail=ovr.get('detail'),
                       exists=ovr['exists'], size_mb=ovr['size_mb']),
              preview=os.path.relpath(png, dest).replace('\\', '/'),
              datav_km2=dv, plan_km2=plan_km2, buffer_deg=BUFFER_DEG,
              ref_area=ref, c5_refs=entry['c5_refs'], eval_refs=entry['ev_refs'],
              note='统计口径：10m 原生网格直读（nearest，零重采样）；面积=块心纬度加权 cell_m2；'
                   '类码 0=nodata 不计；覆盖率双口径分开报',
              generated=time.strftime('%Y-%m-%d %H:%M:%S'))
    PC.jsave(qa, os.path.join(qa_dir, '%s_qa.json' % nm))
    # c5 面积详表副本（有则收）
    detail_copy = None
    if os.path.exists(c5_md):
        detail_copy = os.path.join(qa_dir, '%s_qa_c5详表.md' % nm)
        shutil.copy2(c5_md, detail_copy)
        if os.path.exists(c5_csv):
            shutil.copy2(c5_csv, os.path.join(qa_dir, '%s_qa_c5详表_class.csv' % nm))
    entry['qa_json'] = os.path.join(qa_dir, '%s_qa.json' % nm)
    entry['qa_detail_copy'] = detail_copy
    emit('  QA → %s（%.1f 亿 px，%d 类，%.1f km²，stats=%s）' % (
        os.path.basename(entry['qa_json']), st['data_px'] / 1e8, st['n_classes'],
        st['data_km2'], stats_src.split(':')[0]))
    if ref:
        emit('  与第一次面积表：%s → ΣΔ=%+d px，逐类差异 %d 个（max|Δpx|=%d）' % (
            os.path.basename(ref['ref_csv']), ref['d_px'], ref['diff_classes'],
            ref['max_abs_dpx_class']))
    return entry


def write_qa_summary(dest, entries, a, names):
    provs = entries
    L = ['# 第二次交付 · 成品 QA 汇总（川/粤/宁 %d）' % a.year, '']
    if a.smoke:
        L += ['> ⚠️ 冒烟自检件（非交付）；输入=第一次交付成品（只读）。', '']
    L += ['> 门禁：uint8/单波段/nodata=0/EPSG:4326/10m 原生方度格/内嵌 256 调色板/主 tif 单页；',
          '> 统计：10m 原生网格直读（nearest），面积=块心纬度加权 `p5common.cell_m2`；',
          '> 覆盖率**双口径**分报。逐件详情见 `成品/QA/<省>_qa.json`。', '']
    # 总表
    L += ['## 成品总表', '',
          '| 省 | 尺寸 px | 文件 GB | 数据像元 | 覆盖%(px) | 覆盖%(km²) | 类数 | 面积 km² | DataV 差 | MD5 |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for e in provs:
        st = e['stats']
        dv_txt = ('%+.2f%%' % (100 * (st['data_km2'] - e['datav_km2']) / e['datav_km2'])
                  if e['datav_km2'] else '—')
        L.append('| %s | %d×%d | %.2f | %d | %.2f | %.2f | %d | %.1f | %s | `%s` |' % (
            e['name'], e['meta']['width'], e['meta']['height'], e['meta']['size_gb'],
            st['data_px'], st['coverage_px_pct'], st['coverage_km2_pct'], st['n_classes'],
            st['data_km2'], dv_txt, e['md5'][:16]))
    L.append('')
    # 门禁明细
    L += ['## 门禁与金字塔逐件', '']
    for e in provs:
        m = e['meta']
        L.append('- **%s**：uint8 ✓｜nodata=0 ✓｜EPSG:4326 ✓｜10m 原生格 ✓｜调色板 %d 项 ✓｜'
                 '主文件 %d 页（%s）｜.ovr=%s（%s）｜统计来源=%s' % (
                     e['name'], m['n_cmap'], m['tif_pages'],
                     '无内建概览 ✓' if not m['internal_ov'] else '⚠️含内建概览',
                     e['ovr'].get('mode'), e['ovr'].get('detail') or '—',
                     e['stats_src'].split(':')[0]))
    L.append('')
    # 面积对账
    L += ['## 省面积对账（vs DataV 全域球面 / plan 主省合计）', '',
          '| 省 | DataV km² | plan 主省合计 km² | 本成品 km² | vs DataV |', '|---|---|---|---|---|']
    for e in provs:
        st = e['stats']
        L.append('| %s | %s | %.1f | %.1f | %s |' % (
            e['name'], '%.1f' % e['datav_km2'] if e['datav_km2'] else '—', e['plan_km2'],
            st['data_km2'],
            '%+.2f%%' % (100 * (st['data_km2'] - e['datav_km2']) / e['datav_km2'])
            if e['datav_km2'] else '—'))
    L.append('')
    # 第一次面积表逐类对账（可选）
    refs = [e for e in provs if e['ref']]
    if refs:
        L += ['## 与第一次交付面积表逐类对账（`--ref-area-dir` 提供）', '',
              '| 省 | 第一次 km² | 本成品 km² | ΣΔpx | 逐类差异数 | max\\|Δpx\\| | 说明 |',
              '|---|---|---|---|---|---|---|']
        for e in refs:
            r = e['ref']
            L.append('| %s | %.3f | %.3f | %+d | %d | %d | %s |' % (
                e['name'], r['ref_km2'], r['second_km2'], r['d_px'], r['diff_classes'],
                r['max_abs_dpx_class'],
                '逐类 0 差（同输入复算一致）' if r['diff_classes'] == 0 else '差异见 QA json'))
        L.append('')
        L += ['> 注：对账仅在输入与第一次相同（冒烟）时才有"0 差"意义；正式第二次产物与第一次'
              '样本池不同，逐类差异属预期。', '']
    # 逐瓦 QA 台账
    n_tile_qa = len(glob.glob(os.path.join(a.qa_dir, 'T*_qa.json')))
    n_tiles = len(glob.glob(os.path.join(a.tiles_dir, 'T*_10m.tif')))
    n_plan = len(read_plan2())
    L += ['## 逐瓦 QA 台账', '',
          '- 逐瓦 QA（c3）：`%s/T*_qa.json` 现有 **%d / %d**；瓦片栅格 `%s` 现有 **%d / %d**。'
          % (a.qa_dir.replace('\\', '/'), n_tile_qa, n_plan,
             a.tiles_dir.replace('\\', '/'), n_tiles, n_plan),
          '- 逐瓦台账（EECU）：见 `台账/生产台账摘要.md`（原始 CSV 不复制）。', '']
    # c5 详表引用
    L += ['## c5 面积详表副本（有则收）', '']
    for e in provs:
        if e.get('qa_detail_copy'):
            L.append('- %s：`%s`（原件 `%s`）' % (
                e['name'], os.path.relpath(e['qa_detail_copy'], dest).replace('\\', '/'),
                e['c5_refs']['qa_md']))
        else:
            L.append('- %s：c5 面积详表（label=%s）未就绪（`%s`）' % (
                e['name'], a.c5_label, e['c5_refs']['qa_md']))
    L += ['', '> 生成：c6_deliver.py ｜ %s' % time.strftime('%Y-%m-%d %H:%M:%S'), '']
    fp = os.path.join(dest, '成品', 'QA', 'qa汇总.md')
    with open(fp, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    return fp


def main():
    ap = argparse.ArgumentParser(
        prog='c6_deliver.py',
        description='第二次分类（川/粤/宁 40 瓦 + 补瓦 T2202 = 41 瓦 · 2023）交付打包：c5 成品 → 交付树（对齐第一次五省批）')
    ap.add_argument('--dest', default=DEST_DEFAULT, help='交付根（默认 %s）' % DEST_DEFAULT)
    ap.add_argument('--prov', default='all', help='省简称/全名（川,粤,宁）；all=三省')
    ap.add_argument('--year', type=int, default=YEAR)
    ap.add_argument('--smoke', action='store_true',
                    help='冒烟自检模式：全部产物落 <dest>/_smoke/，README 加非交付横幅')
    ap.add_argument('--src-dir', default=MOSAIC2, help='省口径成品目录（默认第二次 mosaic2；'
                                                       '冒烟可指第一次 D:/生产_2023五省/成品/分省）')
    ap.add_argument('--eval-dir', default=EVAL2_DIR, help='c4 评估目录')
    ap.add_argument('--c5-reports-dir', default=REPORTS2, help='c5 面积 QA/栅格对比目录')
    ap.add_argument('--c5-label', default=C5_QA_LABEL, help='c5 qa 子命令标签（默认 second）')
    ap.add_argument('--qa-dir', default=QADIR2, help='c3 逐瓦 QA 目录（计数引用）')
    ap.add_argument('--tiles-dir', default=TILES2, help='c3 瓦片栅格目录（计数引用）')
    ap.add_argument('--ledger-dir', default=LEDGER_DIR, help='EECU 台账目录（只读汇总）')
    ap.add_argument('--stats', choices=['auto', 'reuse', 'own'], default='auto',
                    help='面积统计来源：auto=优先 c5 面积 QA 复用、缺则重扫；reuse=必须复用；own=强制重扫')
    ap.add_argument('--ovr', choices=['auto', 'copy', 'build', 'none'], default='auto',
                    help='外置金字塔：auto=源有 .ovr 则复制+抽检、无则重建；copy/build/none 强制')
    ap.add_argument('--ref-area-dir', default='', help='（可选）第一次面积表目录，做逐类对账')
    a = ap.parse_args()

    if a.smoke:
        a.dest = os.path.join(a.dest, '_smoke')
    dest = a.dest
    smoke = bool(a.smoke)
    emit('=== c6 交付打包（%s%s）→ %s ===' % (
        '冒烟' if smoke else '正式', '·' + a.prov, dest))
    os.makedirs(dest, exist_ok=True)
    # 幂等：清掉上一轮重建项（保留 _smoke 等同级目录）
    for p in ('成品', '台账'):
        d = os.path.join(dest, p)
        if os.path.isdir(d):
            shutil.rmtree(d)
    for p in ('交付包_README.md', 'MANIFEST_MD5.txt'):
        fp = os.path.join(dest, p)
        if os.path.exists(fp):
            os.remove(fp)

    names = v31_names()
    colors, colors_note = load_colors()
    provs = resolve_provs(a.prov)
    emit('省范围：%s ｜ 色源：%s ｜ 统计：%s ｜ ovr：%s' % (
        ','.join(provs), colors_note, a.stats, a.ovr))

    # ---- 逐省打包 ----
    entries = []
    for short in provs:
        e = build_one(short, a, dest, names)
        if e:
            entries.append(e)
    if not entries:
        raise SystemExit('无任何省成品可打包（第二次产物未就绪？先跑 c5 mosaic；'
                         '冒烟可用 --src-dir %s --prov 宁）。' % FIRST_PROV_DIR)

    # ---- 色卡（出现类=三省统计并集） ----
    present = set()
    for e in entries:
        present |= {k for k, v in e['stats']['px_by_class'].items() if v > 0}
    write_color_card(os.path.join(dest, '成品', '色卡'), colors, names, present)

    # ---- 面积表汇总 ----
    per_prov = {e['short']: e for e in entries}
    write_area_summary(os.path.join(dest, '成品', '面积表', '汇总.csv'), per_prov, names)

    # ---- QA 汇总 ----
    qa_md = write_qa_summary(dest, entries, a, names)
    emit('QA 汇总 → %s' % qa_md)

    # ---- 评估收集 ----
    evals, ev_copied, ev_missing, ev_cmp = collect_eval(a.eval_dir, provs,
                                                        os.path.join(dest, '成品', '评估'), a.year)
    emit('评估：复制 %d 件；compare 收录 %d 省%s' % (
        len(ev_copied), ev_cmp, ('；缺 %s' % ','.join(os.path.basename(x) for x in ev_missing))
        if ev_missing else ''))

    # ---- 台账摘要 ----
    os.makedirs(os.path.join(dest, '台账'), exist_ok=True)
    led = build_ledger_summary(os.path.join(dest, '台账', '生产台账摘要.md'), smoke=smoke)
    emit('台账摘要：%d 文件 / %d 行 / ΣECU·h=%.2f（desc 去重口径）' % (
        led['n_files'], led['n_rows'], led['total_eecu']))

    # ---- 池统计（README 用） ----
    pool = PC.jload(POOL_STATS) or {}
    if 'n' not in pool:
        pool = dict(n=1764020, missing_codes=['8'])

    # ---- README（先算清单，后写入；MANIFEST 最后收口） ----
    files_pre = list_files(dest)
    ctx = dict(smoke=smoke, order=[(e['short'], e['name'], e) for e in entries],
               evals=evals, pool=pool, files=files_pre, n_plan=len(read_plan2()))
    readme_fp = build_readme(dest, ctx)
    emit('README → %s' % readme_fp)
    man_fp, files = write_manifest(dest)
    emit('MANIFEST → %s（%d 件）' % (man_fp, len(files)))

    # ---- 收尾摘要 ----
    emit('=== 完成：%d 件交付文件 ｜ 根：%s ===' % (len(files), dest))
    for rel, md5, n in files:
        emit('  %s（%s）' % (rel, md5[:12]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
