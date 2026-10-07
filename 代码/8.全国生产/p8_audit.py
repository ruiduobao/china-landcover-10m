# -*- coding: utf-8 -*-
"""
p8_audit.py — 片区成品"标注点反审"（全国生产每片区完成后的质量闸门）
为什么需要它（见 技术文档/29）：
  与 ESA WorldCover 对照只能看出"大类像不像"；**用我们自己的人工标注库在成品图上取点**，
  才能在小范围内暴露 130 草地 0.057、91 混交林 0.008 这类量级的崩溃。
  这一闸门比外部产品对照灵敏得多，且完全本地、几分钟出表。

做三件事：
  1) 从年度子集取该区域内的标注点，读成品图在这些点上的类别 → 逐类命中率 + 主要误分去向
  2) 邻域共现（5×5）：每类的空间纯度 + 主要"寄生"邻居 → 定位"海绵类"（如 121 落叶灌丛）
  3) 输出 CSV/JSON，并给出**是否放行**的判据（见 --min-recall）

用法：
  python p8_audit.py --raster <成品.tif> --year 2023 --bbox 118.5 38.5 135 54 --region 东北三省
  python p8_audit.py --raster ... --year 2023 --bbox ... --pts-csv <自定义标注点.csv>
"""
import os, sys, json, time, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '0.本地流水线'))
import prod_conf as C
from lc_conf import CLASSES

sys.stdout.reconfigure(encoding='utf-8')


def emit(m):
    print(time.strftime('[%H:%M:%S] ') + str(m), flush=True)


def load_points(year, bbox, pts_csv=None):
    if pts_csv:
        d = pd.read_csv(pts_csv)
    else:
        fp = os.path.join(C.KB, '数据/本地处理/全国清洗训练/年度子集_含稀有类',
                          'r7_train_%d.parquet' % year)
        if not os.path.isfile(fp):
            fp = os.path.join(C.KB, '数据/本地处理/全国清洗训练/年度子集',
                              'r7_train_%d.parquet' % year)
        d = pd.read_parquet(fp, columns=['row_id', 'lon', 'lat', 'class_new'])
    x0, y0, x1, y1 = bbox
    return d[(d.lon >= x0) & (d.lon < x1) & (d.lat >= y0) & (d.lat < y1)].reset_index(drop=True)


def sample_raster(fp, pts, a_nb_cell_m=500.0):
    import rasterio
    with rasterio.open(fp) as s:
        tr, H, W = s.transform, s.height, s.width
        r = ((tr.f - pts.lat.to_numpy()) / abs(tr.e)).astype(int)
        c = ((pts.lon.to_numpy() - tr.c) / tr.a).astype(int)
        ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
        rr, cc = r[ok], c[ok]
        win = rasterio.windows.Window(cc.min(), rr.min(),
                                      cc.max() - cc.min() + 1, rr.max() - rr.min() + 1)
        blk = np.asarray(s.read(1, window=win))
        vals = blk[rr - rr.min(), cc - cc.min()]
        # 邻域共现用的抽样块（整幅的抽稀副本）
        px_m = abs(tr.a) * 111320 * np.cos(np.radians(tr.f + H * tr.e / 2))
        decim = max(1, int(round(a_nb_cell_m / max(px_m, 1e-6))))
        emit('  邻域格元 %.0f m / 像元 %.0f m → 抽稀 %d 倍（5×5 邻域覆盖 %.1f km）'
             % (a_nb_cell_m, px_m, decim, 5 * decim * px_m / 1000))
        arr = np.asarray(s.read(1, out_shape=(max(1, H // decim), max(1, W // decim)),
                                resampling=rasterio.enums.Resampling.mode))
    return pts[ok].assign(map=vals), arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raster', required=True)
    ap.add_argument('--year', type=int, default=2023)
    ap.add_argument('--bbox', type=float, nargs=4, required=True)
    ap.add_argument('--region', default='未命名')
    ap.add_argument('--pts-csv', default='')
    # 邻域分析的目标格元边长（米）。5×5 邻域 = 该值的 5 倍范围；
    # 绝对纯度值随窗口大小显著变化，报告里必须写明用了哪一档（2026-09-15 踩过）
    ap.add_argument('--nb-cell-m', type=float, default=500.0)
    ap.add_argument('--min-recall', type=float, default=0.50)
    ap.add_argument('--min-n', type=int, default=300)
    ap.add_argument('--outdir', default='')
    a = ap.parse_args()
    out = a.outdir or os.path.join(C.LEDGER, '片区反审')
    os.makedirs(out, exist_ok=True)
    stem = '%s_%d' % (a.region, a.year)

    pts = load_points(a.year, a.bbox, a.pts_csv or None)
    emit('%s：区域内标注点 %d' % (a.region, len(pts)))
    pts, arr = sample_raster(a.raster, pts, a.nb_cell_m)
    nz = int((pts['map'] > 0).sum())
    emit('落在图幅内 %d，图上非零 %d（%.1f%%）' % (len(pts), nz, 100.0 * nz / max(len(pts), 1)))

    rows = []
    for cl in sorted(pts.class_new.unique()):
        g = pts[pts.class_new == cl]
        if len(g) < 20:
            continue
        hit = int((g['map'] == cl).sum())
        cov = int((g['map'] > 0).sum())
        top = g['map'].value_counts().head(3).to_dict()
        rows.append({'class': int(cl), 'name': CLASSES[int(cl)][1], 'n': int(len(g)),
                     'n_cov': cov, 'cov%': round(100.0 * cov / len(g), 1),
                     'hit': hit,
                     # 命中率只在"图上有值"的点上算——否则会把未覆盖区当成错分（2026-09-15 踩过）
                     'recall': round(hit / cov, 3) if cov else 0.0,
                     'recall_all': round(hit / len(g), 3),
                     '误分成': ' / '.join('%d:%d' % (k, v) for k, v in top.items()
                                          if k != cl and k != 0)})
    d = pd.DataFrame(rows).sort_values('n', ascending=False)
    d.to_csv(os.path.join(out, 'recall_%s.csv' % stem), index=False, encoding='utf-8-sig')
    big = d[d.n_cov >= a.min_n]
    overall = float(d.hit.sum() / max(d.n_cov.sum(), 1))
    cov_all = float(d.n_cov.sum() / max(d.n.sum(), 1))
    emit('\n=== 逐类命中率（成品图 vs 人工标注）===')
    emit(d.head(20).to_string(index=False))
    emit('\n加权总体命中率 %.4f（n=%d）' % (overall, int(d.n.sum())))

    # 邻域共现
    R = 2
    acc = np.zeros(256 * 256, np.int64)
    m0 = arr > 0
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            if dy == 0 and dx == 0:
                continue
            nb = np.roll(np.roll(arr, dy, 0), dx, 1)
            m = m0 & (nb > 0)
            if m.sum():
                acc += np.bincount(arr[m].astype(np.int64) * 256 + nb[m].astype(np.int64),
                                   minlength=65536)
    tot = acc.reshape(256, 256)
    nb_rows = []
    for c in range(1, 256):
        row = tot[c]
        if row.sum() < 500:
            continue
        selfp = float(row[c] / row.sum())
        others = sorted(((k, float(v) / row.sum()) for k, v in enumerate(row)
                         if k != c and v / row.sum() > 0.05), key=lambda x: -x[1])
        nb_rows.append({'class': c, 'name': CLASSES[c][1], 'purity': round(selfp, 3),
                        'nb_cell_m': a.nb_cell_m,
                        'parasites': ' / '.join('%d %s %.0f%%' % (k, CLASSES[k][1], 100 * v)
                                                for k, v in others[:3])})
    nb = pd.DataFrame(nb_rows).sort_values('purity')
    nb.to_csv(os.path.join(out, 'neighborhood_%s.csv' % stem), index=False, encoding='utf-8-sig')
    emit('\n=== 空间纯度最低的 8 类（"海绵类"候选）===')
    emit(nb.head(8).to_string(index=False))

    bad = big[big.recall < a.min_recall]
    verdict = 'PASS' if bad.empty else 'REVIEW'
    rep = {'region': a.region, 'year': a.year, 'n_pts': int(len(pts)),
           'coverage': round(cov_all, 4),
           'overall_recall': round(overall, 4), 'min_recall': a.min_recall,
           'classes_below': bad[['class', 'name', 'n', 'recall']].to_dict('records'),
           'verdict': verdict, 'time': time.strftime('%Y-%m-%d %H:%M:%S')}
    json.dump(rep, open(os.path.join(out, 'verdict_%s.json' % stem), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    emit('\n判据：≥%d 样本的类，命中率须 ≥%.2f' % (a.min_n, a.min_recall))
    emit('结论：%s%s' % (verdict, '' if verdict == 'PASS' else
                       ' —— 不达标类：' + ', '.join('%d(%s %.2f)' % (r['class'], r['name'], r['recall'])
                                                 for r in rep['classes_below'][:6])))
    emit('输出目录：%s' % out)
    return 0 if verdict == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
