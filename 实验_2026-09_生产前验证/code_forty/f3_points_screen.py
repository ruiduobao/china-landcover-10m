# -*- coding: utf-8 -*-
"""f3_points_screen.py — 现有乔灌园地 v7 点 × ForTy 2020 分类一致性筛查（工作副本，并行分文件）

* 输入：D:/lcwork_forty/ForTy_china/<tile>/*.tif（f2 工作副本）
        数据/单类样本的单独处理_2026.10.12/乔灌园地/乔灌园地v7_点表.csv（160,931 点）
        F:/lc_work/forty_exp/map/forty_file_index.csv（文件 bbox 索引）
* 规则源（**预注册，先写死再跑**）：
  ① 每点取所在像元 ForTy 类码（1 原始林 / 2 天然更新林 / 3 人工林 / 4 短轮伐种植林 / 5 树作物与农林复合 / 6 其他土地 / 255 nodata）；
  ② 上下文纯度：5×5（50 m）与 11×11（110 m）窗口内"第 5 类占比"（仅统计有效像元）→ 刻画"地块格局"（本项目园地第一性判据）；
  ③ 11×11 窗口众数类 dom11；
  ④ 一致率 = 中心类==5 的比例（另报"5 或 3/4（种植相关类）"的放宽一致率——ForTy 自身说明 Planted/Plantation 与树作物存在混淆）。
  判据（用于后续提纯，不在本脚本执行删除）：
    - 分层一致率按 srcset × grade 报告；差异 ≥5pp 视为该 stratum 存在系统性问题；
    - 单点风险档：center∈{1,2} 且 dom11∈{1,2} 且 c5_11x11=0 → "疑似天然林"（C 档建议剔除）；
    - center=255 → "未覆盖"（不计入一致率分母）。
* 门槛：覆盖率（有效类）≥0.95；每文件采样点数与分配点数一致
* 输出：F:/lc_work/forty_exp/points/v7_forty_screen.parquet（uid, forty_class, dom11, c5_5x5, c5_11x11, row, col, file）
        F:/lc_work/forty_exp/points/parts/*.parquet（分片中间件）
        F:/lc_work/forty_exp/points/screen_summary.json（分层一致率）
* 用法：python f3_points_screen.py [--workers 6]
"""
import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import rasterio
from shapely import STRtree
from shapely.geometry import box

WORK = r'F:/lc_work/forty_exp'
MAP = os.path.join(WORK, 'map')
DCOPY = r'D:/lcwork_forty/ForTy_china'
PTS = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12/乔灌园地/乔灌园地v7_点表.csv'
OUTD = os.path.join(WORK, 'points')


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def worker(task):
    rel, rows = task          # rows: list of (uid, lon, lat)
    fp = os.path.join(DCOPY, rel.replace('/', os.sep))
    out = []
    with rasterio.open(fp) as s:
        a = s.read(1)
        t = s.transform
        from rasterio.warp import transform as rtrans
        lons = [x[1] for x in rows]; lats = [x[2] for x in rows]
        xs, ys = rtrans('EPSG:4326', s.crs, lons, lats)
        for (uid, lon, lat), x, y in zip(rows, xs, ys):
            c = int((x - t.c) / t.a)
            r = int((y - t.f) / t.e)
            if r < 0 or c < 0 or r >= a.shape[0] or c >= a.shape[1]:
                out.append((uid, -1, -1, np.nan, np.nan, r, c, rel)); continue
            win11 = a[max(0, r - 5):r + 6, max(0, c - 5):c + 6]
            win5 = a[max(0, r - 2):r + 3, max(0, c - 2):c + 3]
            def frac5(w):
                v = w[w != 255]
                return float((v == 5).mean()) if v.size else np.nan
            v11 = win11[win11 != 255]
            dom = -1
            if v11.size:
                cnt = np.bincount(v11.astype(np.uint8), minlength=7)
                cnt[0] = 0
                dom = int(cnt.argmax())
            out.append((uid, int(a[r, c]), dom, frac5(win5), frac5(win11), r, c, rel))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    os.makedirs(os.path.join(OUTD, 'parts'), exist_ok=True)
    idx = pd.read_csv(os.path.join(MAP, 'forty_file_index.csv'), encoding='utf-8-sig')
    idx = idx[idx.in_china == True].reset_index(drop=True)
    pts = pd.read_csv(PTS, encoding='utf-8-sig')
    emit('点 %d，索引文件 %d' % (len(pts), len(idx)))
    geoms = [box(r.lon_min, r.lat_min, r.lon_max, r.lat_max) for r in idx.itertuples()]
    tree = STRtree(geoms)
    # 点 → 文件（用于分组）；多点命中取 bbox 面积最小者
    assign = np.full(len(pts), -1)
    for i, (lo, la) in enumerate(zip(pts.lon.to_numpy(), pts.lat.to_numpy())):
        hits = tree.query(box(lo, la, lo, la))
        if len(hits):
            areas = [(geoms[j].area, j) for j in hits]
            assign[i] = min(areas)[1]
    cover = (assign >= 0).sum()
    emit('落入中国文件 bbox：%d/%d = %.3f' % (cover, len(pts), cover / len(pts)))
    pts['fidx'] = assign
    tasks = []
    for fi, g in pts[pts.fidx >= 0].groupby('fidx'):
        rel = idx.iloc[int(fi)].relpath
        tasks.append((rel, list(zip(g.uid.astype(int), g.lon.astype(float), g.lat.astype(float)))))
    emit('分组 %d 个文件任务' % len(tasks))
    t0, done, recs = time.time(), 0, []
    parts = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for res in ex.map(worker, tasks, chunksize=1):
            recs += res
            done += 1
            if done % 100 == 0:
                emit('  完成 %d/%d（%.1f 分钟）' % (done, len(tasks), (time.time() - t0) / 60))
            if done % 200 == 0:
                df = pd.DataFrame(recs, columns=['uid', 'forty_class', 'dom11', 'c5_5x5', 'c5_11x11', 'row', 'col', 'file'])
                df.to_parquet(os.path.join(OUTD, 'parts', 'part_%05d.parquet' % (done // 200)), index=False)
                recs = []
    if recs:
        pd.DataFrame(recs, columns=['uid', 'forty_class', 'dom11', 'c5_5x5', 'c5_11x11', 'row', 'col', 'file']).to_parquet(
            os.path.join(OUTD, 'parts', 'part_last.parquet'), index=False)
    parts = sorted(glob.glob(os.path.join(OUTD, 'parts', '*.parquet')))
    d = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    d = d.drop_duplicates(subset=['uid'])
    d.to_parquet(os.path.join(OUTD, 'v7_forty_screen.parquet'), index=False)
    emit('抽样完成：%d 点 → v7_forty_screen.parquet（%.1f 分钟）' % (len(d), (time.time() - t0) / 60))
    # ---- 一致率汇总 ----
    m = pts.merge(d, on='uid', how='left')
    m['covered'] = m.forty_class.notna() & (m.forty_class > 0)
    m['agree'] = (m.forty_class == 5)
    m['agree_wide'] = m.forty_class.isin([3, 4, 5])
    m['suspect_forest'] = m.forty_class.isin([1, 2]) & m.dom11.isin([1, 2]) & (m.c5_11x11.fillna(0) == 0)
    summ = {'n_points': int(len(m)), 'n_covered': int(m.covered.sum()),
            'coverage': round(float(m.covered.mean()), 4),
            'agree_c5': round(float(m.loc[m.covered, 'agree'].mean()), 4),
            'agree_wide': round(float(m.loc[m.covered, 'agree_wide'].mean()), 4),
            'suspect_forest': int(m.suspect_forest.sum()),
            'class_dist': m.loc[m.covered, 'forty_class'].astype(int).value_counts().sort_index().to_dict(),
            'by_srcset': {}, 'by_grade': {}}
    for k, g in m[m.covered].groupby('srcset'):
        summ['by_srcset'][k] = {'n': int(len(g)), 'agree_c5': round(float(g.agree.mean()), 4),
                                'agree_wide': round(float(g.agree_wide.mean()), 4),
                                'c5_11x11_med': round(float(g.c5_11x11.median()), 3),
                                'suspect_forest': int(g.suspect_forest.sum())}
    for k, g in m[m.covered].groupby('grade'):
        summ['by_grade'][k] = {'n': int(len(g)), 'agree_c5': round(float(g.agree.mean()), 4),
                               'agree_wide': round(float(g.agree_wide.mean()), 4)}
    json.dump(summ, open(os.path.join(OUTD, 'screen_summary.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    emit(json.dumps({k: summ[k] for k in ('n_covered', 'coverage', 'agree_c5', 'agree_wide', 'suspect_forest')},
                    ensure_ascii=False))
    for k, v in summ['by_srcset'].items():
        emit('  %-16s n=%6d 类5=%s 放宽=%s 怀疑天然林=%d' % (k, v['n'], v['agree_c5'], v['agree_wide'], v['suspect_forest']))


if __name__ == '__main__':
    main()
