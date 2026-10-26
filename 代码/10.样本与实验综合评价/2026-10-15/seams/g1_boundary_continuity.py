# -*- coding: utf-8 -*-
"""g1_boundary_continuity — 空白行/接缝连续性**诊断**脚本（非生产硬门；阻断门定义见文件尾）

背景：2026-10-15 用户报 30.00149246° 一行空白。定位结论（seams/报告_空白行_测地楔形根因_2026-10-15.md）：
    GEE 侧 box 为 ee.Geometry.Rectangle(meta['box'])（数字坐标 → 服务端默认 geodesic=True），
    clip/export 的南边是**球面大圆**，在中经线向北拱 0.0030–0.0042°（@24–40°N）；
    瓦片南网格线与该拱之间的透镜区被排除 → 每瓦南边一个对称三角形 nodata。
    既有 QA 按 box ±0.02° 内缩抽点（c3 L392-393）→ 楔高 ≤0.0042° 永不被采到（概率 0），
    且 c3 L417 把资产掩膜 255 与本地 nodata 0 视作一致 → 漏检。

⚠️ 三档检查的**定级（务必按此解读；阈值未校准，不得当生产阻断门）**：
  T1d 测地透镜几何检测：按大圆公式预测“region 外”掩膜 ∩ 瓦片 nodata 计数。
       —— **诊断量**，因为瓦片内该区可能天然无 AEF 数据（海面/真实缺测）→ 只有换成
       「预期**有效输入**区域内输出 nodata = 0」才是阻断门（需 EVM=输入有效掩膜，见文件尾）。
  T2d 逐行内部断带：L(r)=行内最长**内部**零游程，L(r)−median(L(r±5))>16 px 报警（阈值**启发式未校准**；
       连续光学缺测/大湖/城市带也会真空白 → 必须人工复判或与对照年比对）。
  T3d 覆盖率粗对照：coverage_px_pct 与 land_frac 的关系。
       —— **不能用 land_frac 作“应有覆盖率”**：省界陆域 ≠ AEF 有效覆盖，沿海瓦还含被分类海面；
       因此 99.99 / 100*land_frac−0.05 只能作**提示性警报**，不得写成硬门（主报告亦不采用）。

* 门槛/纪律：只读；类别码 0=nodata；不写任何成品；结果落 seams/gate_report_*.json + gate_rows.csv
* 用法：
    python g1_boundary_continuity.py tiles --tiles-dir F:/lc_work/prod2nd_2023/rasters_batch
    python g1_boundary_continuity.py raster --file <tif>            # 仅逐行断带诊断（大图分块）
    python g1_boundary_continuity.py tiles --tiles-dir <dir> --no-t2 --plan <plan.csv>

—— 建议的**正式阻断门（G-SEAM，需跑前预注册）** ——
在「预期有效输入掩膜 EVM」上判定：`输出 nodata = 0`（或 ≤ 预注册容差）。
**EVM 金标准（唯一）**：导出**原始 AEF 全波段共同有效掩膜**，且 ①**不先 clip**、
②用**已验证的平面 region（geodesic=False）**、③**固定全局 grid（显式 crsTransform）**导出。
—— 若沿用同一套错误测地几何导出 sidecar，只会在同一位置复制同一缺口（自证，无效）。
—— `aef.select(0).mask()` 仅为近似（单波段）；须证明 64 波段 mask 一致，严谨用
   `aef.mask().reduce(ee.Reducer.min())`（全波段共同有效）后再判。
**降级替代（诊断 / 须独立校准，不得与金标准等价并列作硬阻断）**：
  (a) 「接缝对照带」：取距南边界 0.02° 的参考行有效像元模式，仅对该模式有效的列要求在边界带有值；
  (b) 预注册容差 `N_tol`：由已知无缺陷对照带（修后重导 / 另年份同瓦）逐 y0 带标定。
本文件实现的 T1d/T2d/T3d 只用于**报警与排序**，通过与否均不构成阻断依据。
"""
import os
import sys
import csv
import json
import math
import argparse
import statistics

import numpy as np
import rasterio
from rasterio.windows import Window

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
PXDEG = 8.983152841195215e-05
PLAN2 = r'F:/lc_work/prod2nd_2023/plan/tiles_secondrun.csv'


def arc_lat(lat0, lon, lon_a, lon_b):
    """同纬两端点球面大圆在 lon 处的纬度（度）——region 南/北边实测几何。"""
    t0 = math.tan(math.radians(lat0))
    d2 = math.radians(abs(lon_b - lon_a)) / 2.0
    lm = (lon_a + lon_b) / 2.0
    return math.degrees(math.atan(t0 * math.cos(math.radians(lon - lm)) / math.cos(d2)))


def longest_interior_zero_run(row_bool):
    """True=有值；返回首末有值之间的最长零游程（px）与其起点列。"""
    idx = np.where(row_bool)[0]
    if idx.size < 2:
        return 0, -1
    seg = row_bool[idx[0]:idx[-1] + 1]
    if seg.all():
        return 0, -1
    b = np.concatenate(([False], ~seg, [False])).astype(np.int8)
    d = np.diff(b)
    st = np.where(d == 1)[0]
    en = np.where(d == -1)[0]
    k = int(np.argmax(en - st))
    return int((en - st)[k]), int(idx[0] + st[k])


def t1_lens_test(fp, box, margin_rows=80):
    """T1：测地南边透镜区内的 nodata 像元数（判据 0）。"""
    with rasterio.open(fp) as ds:
        W, H, t = ds.width, ds.height, ds.transform
        n = min(margin_rows, H)
        lo = H - n
        blk = ds.read(1, window=Window(0, lo, W, n))
        rr, cc = np.mgrid[lo:H, 0:W]
        lon = t.c + (cc + 0.5) * t.a
        lat = t.f - (rr + 0.5) * abs(t.e)
        arc = np.array([arc_lat(box['y0'], float(x), box['x0'], box['x1']) for x in lon[0, :]])
        lens = (lat < arc[None, :]) & (lat >= box['y0'] - 1e-9) & \
               (lon >= box['x0']) & (lon <= box['x1'])
    return int((lens & (blk == 0)).sum()), int(lens.sum())


def t2_row_anomaly(fp, rows=None, win=5, min_extra=16, chunk=256, verbose_every=0):
    """T2：逐行内部零游程异常（相对 ±win 行中位数）。返回异常行列表。"""
    out = []
    with rasterio.open(fp) as ds:
        W, H = ds.width, ds.height
        r0, r1 = (0, H) if rows is None else rows
        # 先算全行 L(r)（分块）
        L = np.zeros(r1 - r0, dtype=np.int32)
        S = np.full(r1 - r0, -1, dtype=np.int64)
        for a in range(r0, r1, chunk):
            b = min(r1, a + chunk)
            blk = ds.read(1, window=Window(0, a, W, b - a))
            for i in range(b - a):
                L[a - r0 + i], S[a - r0 + i] = longest_interior_zero_run(blk[i] > 0)
        for i in range(len(L)):
            lo = max(0, i - win)
            hi = min(len(L), i + win + 1)
            med = statistics.median(L[lo:i].tolist() + L[i + 1:hi].tolist()) if hi - lo > 1 else 0
            if L[i] - med > min_extra and L[i] > 32:
                out.append(dict(row=r0 + i, L=int(L[i]), median_local=int(med), start_col=int(S[i])))
    return out


def t3_coverage(fp, land_frac, verbose=True):
    """T3d 诊断：覆盖率与 land_frac 的粗对照。

    ⚠️ land_frac（省界陆域占比）≠ AEF 有效覆盖占比，沿海瓦还含被分类海面 →
    此处给的不是“应有覆盖率”，只是**提示性警报**阈值（未校准，不作阻断依据）。
    """
    with rasterio.open(fp) as ds:
        W, H = ds.width, ds.height
        nz = 0
        for r in range(0, H, 2048):
            h = min(2048, H - r)
            blk = ds.read(1, window=Window(0, r, W, h))
            nz += int((blk > 0).sum())
    cov = 100.0 * nz / (W * H)
    # 提示线：内陆满瓦 99.99%（经验值，未校准）；沿海瓦 = 100*land_frac − 0.05（同样未校准）
    if abs(land_frac - 1.0) < 1e-9:
        thr, ok = 99.99, cov >= 99.99
    else:
        thr, ok = 100.0 * land_frac - 0.05, cov >= 100.0 * land_frac - 0.05
    if verbose:
        print('    T3d(诊断) coverage_px_pct=%.4f  land_frac=%.3f  提示线 %.3f  %s'
              % (cov, land_frac, thr, '超线' if not ok else '在线内'))
    return cov, thr, ok


def run_tiles(tiles_dir, plan=PLAN2, do_t2=True):
    print('# ⚠️ 诊断档运行（T1d/T2d/T3d 均非生产阻断门；阈值未校准，见文件头）')
    print('# 阻断门 G-SEAM 需「预期有效输入掩膜 EVM」：本轮未评估（无 AEF 有效掩膜 sidecar）')
    boxes = {}
    if plan and os.path.exists(plan):
        for r in csv.DictReader(open(plan, encoding='utf-8')):
            boxes[r['tile']] = dict(x0=float(r['x0']), y0=float(r['y0']),
                                    x1=float(r['x1']), y1=float(r['y1']),
                                    land_frac=float(r['land_frac']))
    res = []
    files = sorted(f for f in os.listdir(tiles_dir) if f.endswith('_10m.tif'))
    for f in files:
        t = f.split('_')[0]
        fp = os.path.join(tiles_dir, f)
        box = boxes.get(t)
        rec = dict(tile=t, fp=fp, grade='DIAGNOSTIC',
                   blocking_gate='NOT_EVALUATED (needs EVM = AEF valid mask)',
                   warning='T1d 透镜区可能天然无 AEF 数据（海面/真实缺测）→ 仅诊断；'
                           'T3d 提示线基于 land_frac，非“应有覆盖率”')
        if box:
            n_lens, n_lens_area = t1_lens_test(fp, box)
            rec.update(t1d_lens_nodata_px=n_lens, t1d_lens_area_px=n_lens_area)
        if do_t2:
            an = t2_row_anomaly(fp)
            rec.update(t2d_anomaly_rows=len(an), t2d_top=an[:5],
                       t2d_note='启发式阈值（L−median>16px）未校准；连续缺测/大湖可真空白')
        if box:
            cov, thr, ok = t3_coverage(fp, box['land_frac'], verbose=False)
            rec.update(coverage_px_pct=round(cov, 4), t3d_hint_line=round(thr, 3),
                       t3d_over_hint=bool(not ok))
        res.append(rec)
        print('%-7s T1d透镜nodata=%9d(面积%9d)  T2d断带行=%-4s(top L=%s)  cov=%7.3f%% 超提示线=%s' % (
            t, rec.get('t1d_lens_nodata_px', -1), rec.get('t1d_lens_area_px', -1),
            rec.get('t2d_anomaly_rows', '-'), (rec.get('t2d_top') or [{}])[0].get('L', '-'),
            rec.get('coverage_px_pct', -1), rec.get('t3d_over_hint', '-')))
    return res


def run_raster(fp, chunk_len=0):
    an = t2_row_anomaly(fp)
    print('%s  T2 异常行 %d 个' % (os.path.basename(fp), len(an)))
    for a in an[:20]:
        print('   row %d  L=%d 局部中位=%d 起点列=%d' % (a['row'], a['L'], a['median_local'], a['start_col']))
    return an


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p1 = sub.add_parser('tiles'); p1.add_argument('--tiles-dir', required=True)
    p1.add_argument('--plan', default=PLAN2); p1.add_argument('--no-t2', action='store_true')
    p1.add_argument('--out', default=os.path.join(HERE, 'gate_report.json'))
    p2 = sub.add_parser('raster'); p2.add_argument('--file', required=True)
    a = ap.parse_args()
    if a.cmd == 'raster':
        an = run_raster(a.file)
        json.dump(an, open(os.path.join(HERE, 'gate_rows_%s.json' % os.path.basename(a.file)), 'w'),
                  ensure_ascii=False, indent=1)
        return
    res = run_tiles(a.tiles_dir, a.plan, not a.no_t2)
    n_hint = sum(1 for r in res if r.get('t3d_over_hint'))
    print('\n共 %d 瓦（诊断档）；T1d 报出透镜 nodata 的瓦 = %d；超 T3d 提示线的瓦 = %d'
          % (len(res), sum(1 for r in res if r.get('t1d_lens_nodata_px', 0) > 0), n_hint))
    print('⚠️ 以上不构成阻断判定：正式门 G-SEAM 需 EVM（AEF 有效掩膜），本轮未评估（见文件头）')
    json.dump(res, open(a.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    with open(os.path.join(HERE, 'gate_rows.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['tile', 'grade', 't1d_lens_nodata_px', 't1d_lens_area_px', 't2d_anomaly_rows',
                    'coverage_px_pct', 't3d_hint_line', 't3d_over_hint', 'blocking_gate'])
        for r in res:
            w.writerow([r.get('tile'), r.get('grade'), r.get('t1d_lens_nodata_px'),
                        r.get('t1d_lens_area_px'), r.get('t2d_anomaly_rows'),
                        r.get('coverage_px_pct'), r.get('t3d_hint_line'),
                        r.get('t3d_over_hint'), r.get('blocking_gate')])
    print('WROTE', a.out)


if __name__ == '__main__':
    main()
