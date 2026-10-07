# -*- coding: utf-8 -*-
"""m5_build_points.py — D0 点位文件构建（本地，零 GEE 成本）
* 任务 1：M3 全部点位合并 → data/m3/m5_all_points.csv（point_id/lon/lat/group）
* 任务 2：FCS10 灌丛层物理体检抽样 → data/m3/m5_audit_points.csv
         层 = 干旱省(新青藏甘宁蒙)×湿/润其余 × 原码 120/121，每层 5,000（层内不足则全取）
* 输入：data/m3/m3{a,b,cd}_points.csv；F:/lc_work/年度子集_含稀有类/r7_train_2023.parquet
* 规则源：doc 45 §一.C；复用 m3b_shrub_probe 的流式读法
* 输出：两个 csv；抽查计数打印
* 用法：python m5_build_points.py
"""
import os, sys, csv
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v31_common as VC
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

M3D = os.path.join(VC.DATA, 'm3')
SRC = r'F:/lc_work/年度子集_含稀有类/r7_train_2023.parquet'
DRY = ('新疆', '西藏', '青海', '甘肃', '宁夏', '内蒙古')
N_PER_STRATUM = 5000
SEED = 20261007

def merge_m3():
    rows = []
    for fn, grp in (('m3a_points.csv', 'A'), ('m3b_points.csv', 'B'), ('m3cd_points.csv', 'CD')):
        with open(os.path.join(M3D, fn), encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                rows.append((r['point_id'], float(r['lon']), float(r['lat']), grp))
    out = os.path.join(M3D, 'm5_all_points.csv')
    with open(out, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['point_id', 'lon', 'lat', 'group'])
        w.writerows(rows)
    print('M3 合并点位 %d → %s' % (len(rows), out))
    return out

def tag_dry(f10):
    """DataV 省界点内多边形判干/湿（GCJ-02 偏移数百米，对干湿分区无影响）。"""
    import json
    from shapely.geometry import shape, Point
    from shapely.prepared import prep
    gj = json.load(open(r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/边界/china_100000_full.json',
                        encoding='utf-8'))
    polys = []
    for ft in gj['features']:
        name = ft.get('properties', {}).get('name', '')
        if name[:2] in DRY or name[:3] in DRY:
            g = shape(ft['geometry'])
            if not g.is_valid:
                from shapely import make_valid
                g = make_valid(g)
            polys.append(prep(g))
    lon = f10['lon'].to_numpy(); lat = f10['lat'].to_numpy()
    dry = np.zeros(len(f10), dtype=bool)
    for i in range(len(f10)):
        p = Point(lon[i], lat[i])
        for g in polys:
            if g.contains(p):
                dry[i] = True
                break
    f10['dry'] = dry
    print('干旱省点 %d / 湿润点 %d' % (dry.sum(), len(f10) - dry.sum()))
    return f10


def audit_sample():
    pf = pq.ParquetFile(SRC)
    cols = ['row_id', 'lon', 'lat', 'class_new', 'src', 'province'] \
        if 'province' in pf.schema.names else ['row_id', 'lon', 'lat', 'class_new', 'src']
    f10 = []
    for b in pf.iter_batches(batch_size=200_000, columns=cols):
        d = b.to_pandas()
        m = d['src'].fillna('').astype(str) == 'glc_fcs10_2023_shrub'
        if m.any():
            f10.append(d[m])
    f10 = pd.concat(f10, ignore_index=True)
    print('灌丛层点:', len(f10), ' 码分布:', f10.class_new.value_counts().to_dict())
    f10 = tag_dry(f10)
    f10['code'] = f10['class_new'].astype(int)
    rng = np.random.default_rng(SEED)
    picks = []
    for dry in (True, False):
        for code in (120, 121):
            s = f10[(f10.dry == dry) & (f10.code == code)]
            n = min(N_PER_STRATUM, len(s))
            if n == 0:
                print('层 dry=%s code=%d 空' % (dry, code))
                continue
            picks.append(s.sample(n=n, random_state=int(rng.integers(1 << 31))))
            print('层 干旱=%-5s 码%d: %d → 抽 %d' % (dry, code, len(s), n))
    out = pd.concat(picks, ignore_index=True)
    out['point_id'] = ['M5A-%05d' % i for i in range(len(out))]
    out = out[['point_id', 'lon', 'lat', 'code', 'dry', 'row_id']]
    fp = os.path.join(M3D, 'm5_audit_points.csv')
    out.to_csv(fp, index=False, encoding='utf-8-sig')
    print('体检抽样 %d → %s' % (len(out), fp))

if __name__ == '__main__':
    merge_m3()
    audit_sample()
