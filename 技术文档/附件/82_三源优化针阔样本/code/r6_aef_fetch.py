# -*- coding: utf-8 -*-
"""r6_aef_fetch.py — 为 opt_v1 的 veg30 点补 AEF64（2023），供 R6 重训对比

* 输入（只读）：
    opt/needle_broad_opt_v1.parquet            融合样本 248,828
    F:/lc_work/年度子集_含稀有类/r7_train_2023.parquet   本地已有 AEF64（v2 点全在此）
* 特征源：GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL（AEF64，A00–A63，10 m，2023）
* 关键工程教训（2026-10-25 实测）：
    把全国点塞进一个 FeatureCollection 调 sampleRegions 会 **Computation timed out**
    ——即使 tileScale=4。因为 region 覆盖全国，GEE 要展开全图 tile。
    → 必须按 **1° 空间格** 分单元请求（实测单格 ≤772 点，755 格），每格 request footprint
      仅 1°×1°，正常 10–20 s 返回。
    - 已在本地 r7_2023 的点直接复用，不重复取（探针实测 GEE 与本地逐位相同）。
    - 每格落 aef_parts/cell_{cx}_{cy}.parquet，**幂等断点续跑**。
* 校验：探针 _r6_probe.py 实测 GEE 取回值与本地 r7 逐位相同（max|diff|=0, corr=1.0）
* 输出：opt/aef_parts/*.parquet；opt/aef_all_points.parquet；opt/r6_aef_fetch.json
* 用法：python r6_aef_fetch.py [--cell 1.0] [--max-cells 0]
* 资源：单进程，< 4 GB
"""
import argparse
import json
import os
import sys
import time
import warnings

sys.stdout.reconfigure(encoding='utf-8')
PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
sys.path.insert(0, os.path.join(PROJ, '代码', '8.全国生产'))
import prod_conf as C

for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy'):
    os.environ.pop(k, None)
os.environ['HTTP_PROXY'] = C.PROXY_URL
os.environ['HTTPS_PROXY'] = C.PROXY_URL
os.environ['ALL_PROXY'] = C.PROXY_URL

import numpy as np
import pandas as pd

OPT = r'F:/lc_work/nb_national_2026/opt'
PARTS = os.path.join(OPT, 'aef_parts')
R7 = r'F:/lc_work/年度子集_含稀有类/r7_train_2023.parquet'
A = ['A%02d' % i for i in range(64)]
# 项目级健康检查（2026-10-25 s0b_health）：仅 2 个账号健康
ACCTS = [('ybzljyma', 'modern-tangent-509301-q8'),
         ('nnjxsn', 'melodic-gamma-509323-u5')]


def emit(m):
    print(time.strftime('[%H:%M:%S] ') + str(m), flush=True)


def init_ee():
    import ee
    for a, pid in ACCTS:
        try:
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter('always')
                C.apply_account_env(a)
                ee.Initialize(project=pid)
                msg = ' '.join(str(x.message) for x in ws)
                if 'restricted' in msg or 'quota' in msg:
                    emit('  acct %s RESTRICTED' % a)
                    continue
            emit('  using %s / %s' % (a, pid))
            return ee, a
        except Exception as e:
            emit('  acct %s FAIL %s' % (a, repr(e)[:140]))
    raise RuntimeError('no usable GEE account')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', type=float, default=1.0)
    ap.add_argument('--max-cells', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(PARTS, exist_ok=True)
    t0 = time.time()

    opt = pd.read_parquet(os.path.join(OPT, 'needle_broad_opt_v1.parquet'),
                          columns=['point_id', 'lon', 'lat', 'tile', 'prov', 'eco',
                                   'leaf_final', 'tier', 'src'])
    v30 = opt[opt.src == 'veg30'].copy()
    emit('veg30 点 %d' % len(v30))

    r7 = pd.read_parquet(R7, columns=['lon', 'lat'])
    r7k = set((r7.lon.round(5).astype(str) + '_' + r7.lat.round(5).astype(str)))
    del r7
    v30['k'] = v30.lon.round(5).astype(str) + '_' + v30.lat.round(5).astype(str)
    v30['in_r7'] = v30.k.isin(r7k)
    emit('veg30 坐标命中本地 r7_2023：%d / %d' % (int(v30.in_r7.sum()), len(v30)))
    need = v30[~v30.in_r7].copy()
    if need.in_r7.sum() == 0 and 'in_r7' in need:
        pass
    emit('需 GEE 取样 %d 点' % len(need))

    need['cx'] = np.floor(need.lon / args.cell).astype(int)
    need['cy'] = np.floor(need.lat / args.cell).astype(int)
    cells = sorted(need.groupby(['cx', 'cy']).size().index.tolist())
    emit('空间单元 %d（%.2f°）' % (len(cells), args.cell))

    done = set()
    for f in os.listdir(PARTS):
        if f.startswith('cell_') and f.endswith('.parquet'):
            p = f[5:-8].split('_')
            done.add((int(p[0]), int(p[1])))
    emit('已完成单元 %d / %d（断点续跑）' % (len(done), len(cells)))

    todo = [c for c in cells if c not in done]
    if todo:
        ee, acct = init_ee()
        img = (ee.ImageCollection(C.AEF).filterDate('2023-01-01', '2024-01-01')
               .mosaic().select(A))
        log, ncells_done = [], 0
        for (cx, cy) in todo:
            d = need[(need.cx == cx) & (need.cy == cy)]
            fc = ee.FeatureCollection([
                ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]),
                           {'pid': int(r.point_id)}) for r in d.itertuples()])
            t1 = time.time()
            got, err = None, ''
            for attempt in range(4):
                try:
                    got = img.sampleRegions(collection=fc, scale=10, geometries=False,
                                            tileScale=4).getInfo()
                    err = ''
                    break
                except Exception as e:
                    err = repr(e)[:120]
                    emit('   cell %s 第 %d 次失败 %s' % ((cx, cy), attempt + 1, err))
                    time.sleep(4 * (attempt + 1))
            if got is None:
                log.append(dict(cell=[cx, cy], n=len(d), ok=0, err=err))
                emit('cell %s FAILED' % ((cx, cy),))
                continue
            g = {int(f['properties']['pid']): f['properties'] for f in got['features']}
            rows = []
            for r in d.itertuples():
                p = g.get(int(r.point_id))
                rows.append([r.point_id, r.lon, r.lat] +
                            ([np.nan] * 64 if p is None else [p.get(x, np.nan) for x in A]))
            out = pd.DataFrame(rows, columns=['point_id', 'lon', 'lat'] + A)
            out[A] = out[A].astype('float32')
            out.to_parquet(os.path.join(PARTS, 'cell_%d_%d.parquet' % (cx, cy)), index=False)
            nok = int(out[A].notna().all(1).sum())
            log.append(dict(cell=[cx, cy], n=len(d), ok=nok, s=round(time.time() - t1, 1)))
            ncells_done += 1
            if ncells_done % 20 == 0 or ncells_done <= 3:
                emit('  单元 %d/%d 累计点 %d  已用 %.1f min  最近 %s n=%d %.1fs'
                     % (len(done) + ncells_done, len(cells),
                        sum(x['n'] for x in log), (time.time() - t0) / 60,
                        (cx, cy), len(d), time.time() - t1))
            if args.max_cells and ncells_done >= args.max_cells:
                emit('达到 --max-cells=%d，暂停' % args.max_cells)
                break
        json.dump(dict(acct=acct, cells=len(cells), log=log,
                       elapsed_min=round((time.time() - t0) / 60, 1)),
                  open(os.path.join(OPT, 'r6_aef_fetch_log_%d.json' % int(time.time())), 'w',
                       encoding='utf-8'), ensure_ascii=False, indent=1)

    # 拼接
    parts = [f for f in os.listdir(PARTS) if f.startswith('cell_') and f.endswith('.parquet')]
    if len(parts) >= len(cells):
        allp = pd.concat([pd.read_parquet(os.path.join(PARTS, f)) for f in parts],
                         ignore_index=True).drop_duplicates('point_id')
        allp.to_parquet(os.path.join(OPT, 'aef_all_points.parquet'), index=False)
        nok = int(allp[A].notna().all(1).sum())
        emit('拼接 %d 行，AEF 完整 %d（缺失 %d）' % (len(allp), nok, len(allp) - nok))
        json.dump(dict(n_part_cells=len(parts), rows=len(allp), aef_complete=nok,
                       aef_missing=len(allp) - nok,
                       elapsed_min=round((time.time() - t0) / 60, 1)),
                  open(os.path.join(OPT, 'r6_aef_fetch.json'), 'w', encoding='utf-8'), indent=1)
    else:
        emit('单元未齐（%d/%d），暂不拼接' % (len(parts), len(cells)))
    emit('完成 %.1f min' % ((time.time() - t0) / 60))


if __name__ == '__main__':
    main()
