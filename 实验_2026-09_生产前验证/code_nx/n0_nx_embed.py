# -*- coding: utf-8 -*-
"""n0_nx_embed.py — 为宁夏中性判读集取 2023 年 AEF 嵌入（GEE 一次性取样，成本可忽略）

* 输入：F:/lc_work/v31_exp/data/m3/nx_neutral_points.csv（205 点）
* 规则源：与生产一致——GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL，2023 年 mosaic，64 波段 A00–A63
* 门槛：一次 sampleRegions（scale=10, tileScale=4）；点数 ≤300
* 输出：F:/lc_work/v31_exp/data/m3/nx_neutral_embed.csv（point_id + A00–A63 + lon/lat）
* 用法：python n0_nx_embed.py [--acct zixen8v8]
幂等：输出已存在且行数一致则跳过（--force 重写）。
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csv
import v31_common as VC

M3D = r'F:/lc_work/v31_exp/data/m3'
SRC = os.path.join(M3D, 'nx_neutral_points.csv')
OUT = os.path.join(M3D, 'nx_neutral_embed.csv')
YEAR = 2023
ACCT = 'zixen8v8'
AEF = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'   # 与 代码/8.全国生产/prod_conf.py 一致


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def main():
    a = sys.argv[1:]
    acct = a[a.index('--acct') + 1] if '--acct' in a else ACCT
    rows = list(csv.DictReader(open(SRC, encoding='utf-8-sig')))
    if os.path.exists(OUT) and '--force' not in a:
        n = sum(1 for _ in csv.DictReader(open(OUT, encoding='utf-8-sig')))
        if n == len(rows):
            emit('已存在 %s（%d 行，--force 重写）' % (OUT, n))
            return
    import ee
    VC.ensure_ctx(acct)
    emit('账号 %s / 项目 %s' % (acct, VC.pid_of(acct)))
    pts = [ee.Feature(ee.Geometry.Point([float(r['lon']), float(r['lat'])]),
                      {'point_id': r['point_id']}) for r in rows]
    fc = ee.FeatureCollection(pts)
    aef = (ee.ImageCollection(AEF)
           .filterDate('%d-01-01' % YEAR, '%d-01-01' % (YEAR + 1))
           .mosaic().select(VC.FEATS))
    emit('请求 sampleRegions（%d 点 × 64 波段）…' % len(rows))
    got = aef.sampleRegions(collection=fc, scale=10, geometries=False, tileScale=4).getInfo()
    feats = got['features']
    emit('返回 %d 点' % len(feats))
    cols = ['point_id'] + VC.FEATS
    recs = []
    for f in feats:
        p = f['properties']
        if all(k in p for k in VC.FEATS):
            recs.append(p)
    with open(OUT, 'w', encoding='utf-8-sig', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in recs:
            w.writerow({k: r.get(k) for k in cols})
    emit('写出 %s：%d 点（缺 %d）' % (OUT, len(recs), len(rows) - len(recs)))


if __name__ == '__main__':
    main()
