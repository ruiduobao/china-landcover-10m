# -*- coding: utf-8 -*-
"""m3a_sample.py — M3-A 通用独立验证点抽样（GEE · ESA WorldCover v200 分层）

设计（doc 42 §2）：
  区域：核心 5 窗（w1 松嫩 / w2 秦岭 / w3 南方丘陵 / w4 西北甘旱 / w5 三江）
        + 可选 2 区（w6 华北平原 / w7 云贵高原）
  分层：WorldCover v200 类别（独立于 FCS 系产品）× 区域；另加"边界带"子层（3×3 邻域类别不一致）
  每类点数：核心 55 / 可选 45（按类均分，非按面积比例 —— 稀有类需要可评估的最小样本）
  空间独立性：本地过滤（与母库、开发池 ≥1 km；点间 ≥1 km）
用法：
  python m3a_sample.py sample     # 逐窗 stratifiedSample（GEE，数百万像素级，成本~0）
  python m3a_sample.py filter     # 本地排除过滤 + 出 CSV/KML/表单
"""
import os, sys, time, json, csv, math
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v31_common as VC
import numpy as np
import pandas as pd

OUTD = os.path.join(VC.DATA, 'm3')
RAW = os.path.join(OUTD, 'm3a_raw.json')

WC_NAMES = {10: '乔木', 20: '灌丛', 30: '草地', 40: '耕地', 50: '建成区',
            60: '裸地/稀疏', 70: '冰雪', 80: '水体', 90: '草本湿地', 95: '红树林', 100: '苔藓地衣'}
REGIONS = {
    'w1': dict(bbox=[122.0, 43.0, 126.5, 47.5], note='农林交错（松嫩）', core=True, n=180),
    'w2': dict(bbox=[105.5, 31.6, 110.0, 36.1], note='森林（秦岭）', core=True, n=180),
    'w3': dict(bbox=[108.0, 23.0, 114.0, 29.0], note='南方丘陵', core=True, n=180),
    'w4': dict(bbox=[101.0, 36.0, 107.0, 42.0], note='干旱过渡（西北甘旱）', core=True, n=180),
    'w5': dict(bbox=[133.0, 47.0, 135.0, 49.0], note='湿地（三江平原）', core=True, n=180),
    'w6': dict(bbox=[114.0, 34.0, 118.0, 37.0], note='华北平原（可选）', core=False, n=150),
    'w7': dict(bbox=[101.5, 23.5, 105.5, 27.0], note='云贵高原（可选）', core=False, n=150),
}
SEED = 20260927


def sample():
    """逐窗抽样；zone 由 50m 邻域 min/max 是否一致判定（edge=边界带，core=均质区）。"""
    import ee
    acct = VC.jload(os.path.join(VC.CFG, 'accounts_pool.json'))['uploader']
    ee, pid = VC.ctx(acct)
    VC.emit('GEE 账号 %s（%s）' % (acct, pid))
    wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map')
    nbr = wc.reduceNeighborhood(reducer=ee.Reducer.minMax(), kernel=ee.Kernel.square(2, 'pixels'))
    out = {}
    for r, meta in REGIONS.items():
        box = ee.Geometry.Rectangle(meta['bbox'])
        fc = wc.stratifiedSample(numPoints=meta['n'], classBand='Map', region=box,
                                 scale=30, seed=SEED, geometries=True, dropNulls=True, tileScale=4)
        try:
            res = nbr.sampleRegions(collection=fc, scale=10, geometries=True, tileScale=4).getInfo()
        except Exception as e:
            VC.emit('%s 邻域判定失败（%s），全部记 core' % (r, str(e)[:60]))
            res = fc.getInfo()
        pts = []
        for f in res.get('features', []):
            g = f.get('geometry') or {}
            c = (g.get('coordinates') or [None, None])
            if c[0] is None:
                continue
            p = f.get('properties', {})
            wc_v = p.get('Map', 0)
            mn, mx = p.get('Map_min'), p.get('Map_max')
            zone = 'core' if (mn is None or mx is None or mn == mx) else 'edge'
            pts.append(dict(lon=round(float(c[0]), 6), lat=round(float(c[1]), 6),
                            wc=int(wc_v), zone=zone))
        out[r] = pts
        from collections import Counter
        cnt = Counter(p['wc'] for p in pts)
        VC.emit('%s %s：%d 点（%s）｜ edge=%d' % (
            r, meta['note'], len(pts), ' '.join('%s:%d' % (WC_NAMES.get(k, k), v) for k, v in sorted(cnt.items())),
            sum(1 for p in pts if p['zone'] == 'edge')))
        os.makedirs(OUTD, exist_ok=True)
        VC.jsave(out, RAW)
    VC.emit('原始候选 → %s（合计 %d 点）' % (RAW, sum(len(v) for v in out.values())))


def _xyz(lon, lat):
    """经纬度 → 单位球直角坐标（用于精确的球面近邻距离，R=6371km）。"""
    a = np.radians(np.asarray(lat, dtype='float64'))
    o = np.radians(np.asarray(lon, dtype='float64'))
    return np.stack([np.cos(a) * np.cos(o), np.cos(a) * np.sin(o), np.sin(a)], axis=1)


R_KM = 6371.0088


def filter_pts():
    """本地排除：与母库/开发池 ≥1 km；点间 ≥1 km；输出 CSV + KML + 判读表单。"""
    from scipy.spatial import cKDTree
    out = VC.jload(RAW, {})
    if not out:
        raise SystemExit('先跑 sample')
    # 排除源：母库 + 开发池 + R1 窗留出点
    frames = []
    mlib = pd.read_parquet(r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/本地处理/样本重建/r7_train.parquet',
                           columns=['lon', 'lat'])
    frames.append(mlib)
    VC.emit('母库点 %d' % len(mlib))
    vp = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/本地处理/样本底座/validation_pool_v2.parquet'
    if os.path.exists(vp):
        d = pd.read_parquet(vp)
        if 'lon' in d.columns:
            frames.append(d[['lon', 'lat']]); VC.emit('开发池 %d' % len(d))
    ex = pd.concat(frames, ignore_index=True).dropna()
    tree = cKDTree(_xyz(ex.lon.to_numpy(), ex.lat.to_numpy()))
    rows = []
    for r, pts in out.items():
        d = pd.DataFrame(pts)
        if not len(d):
            continue
        d['region'] = r
        xyz = _xyz(d.lon.to_numpy(), d.lat.to_numpy())
        dist, _ = tree.query(xyz, k=1)
        d['dist_km'] = (2 * R_KM * np.arcsin(np.clip(dist / 2, 0, 1))).round(3)
        q = np.percentile(d.dist_km, [10, 25, 50, 75, 90])
        VC.emit('%s 距母库分位(km) P10/25/50/75/90 = %s ｜ ≥1km 比例 %.0f%%' % (
            r, '/'.join('%.2f' % x for x in q), 100.0 * (d.dist_km >= 1.0).mean()))
        # 点间稀释（≥0.5 km；母库排除保持 1 km）
        keep = []
        sub = d.sort_values(['zone', 'wc']).reset_index(drop=True)
        tree2 = None
        for i, row in sub.iterrows():
            if row.dist_km < 1.0:
                continue
            p = _xyz([row.lon], [row.lat])
            if tree2 is not None:
                dd, _ = tree2.query(p, k=1)
                if 2 * R_KM * math.asin(min(dd[0] / 2, 1)) < 0.5:
                    continue
            keep.append(i)
            tree2 = cKDTree(np.vstack([tree2.data, p]) if tree2 is not None else p)
        sel = sub.loc[keep]
        rows.append(sel)
    allp = pd.concat(rows, ignore_index=True)
    allp['point_id'] = ['M3A-%s-%04d' % (r, i) for i, r in enumerate(allp.region)]
    os.makedirs(OUTD, exist_ok=True)
    fp = os.path.join(OUTD, 'm3a_points.csv')
    allp[['point_id', 'region', 'lon', 'lat', 'wc', 'zone', 'dist_km']].to_csv(fp, index=False, encoding='utf-8-sig')
    VC.emit('过滤后 %d 点 → %s' % (len(allp), fp))
    for r in REGIONS:
        d = allp[allp.region == r]
        VC.emit('  %s: %d 点（核心=%d 边界=%d）' % (r, len(d), (d.zone == 'core').sum(), (d.zone == 'edge').sum()))
    # KML
    kml = ['<?xml version="1.0" encoding="UTF-8"?>', '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
           '<name>M3A 独立验证点</name>']
    for _, p in allp.iterrows():
        kml.append('<Placemark><name>%s</name><description>区域=%s 类别=%s 带=%s 距母库=%.2fkm</description>'
                   '<Point><coordinates>%.6f,%.6f,0</coordinates></Point></Placemark>' % (
                       p.point_id, p.region, WC_NAMES.get(p.wc, p.wc), p.zone, p.dist_km, p.lon, p.lat))
    kml.append('</Document></kml>')
    open(os.path.join(OUTD, 'm3a_points.kml'), 'w', encoding='utf-8').write('\n'.join(kml))
    # 判读表单（双盲两列 + 仲裁）
    form = os.path.join(OUTD, 'm3a_判读表单.csv')
    with open(form, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['point_id', 'region', 'lon', 'lat', 'strat', 'zone',
                    'label_A', 'conf_A', 'label_B', 'conf_B', 'final_label', 'nagoya_notes',
                    'image_src', 'image_date', 'interp_A', 'interp_B', 'arbiter'])
        for _, p in allp.iterrows():
            w.writerow([p.point_id, p.region, p.lon, p.lat, WC_NAMES.get(p.wc, p.wc), p.zone,
                        '', '', '', '', '', '', '', '', '', '', ''])
    VC.emit('表单 → %s' % form)


if __name__ == '__main__':
    {'sample': sample, 'filter': filter_pts}[sys.argv[1] if len(sys.argv) > 1 else 'sample']()
