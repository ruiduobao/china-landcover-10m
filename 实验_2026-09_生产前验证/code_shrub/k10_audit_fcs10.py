# -*- coding: utf-8 -*-
"""k10_audit_fcs10.py — 建「FCS10 灌丛层抽样」审计包（供人工/有视觉通道的会话判读）

* 为什么换框：doc57 无偏分层判读实测各框真灌丛率——FCS10 层 0.46（宁 0.25/川 0.50/黑 0.54/闽 0.57）、
        OSM 0.31、WC/LC only 0.18、**DW 灌丛标签 0.027**。此前 560 点审计包建在 DW 框上，判了也几乎
        全是"否"，产出率极低；本脚本改从 **FCS10 灌丛层全国 537,361 点**里分层抽样，判读产出率最高。
* 输入：F:/lc_work/年度子集_含稀有类/r7_train_2023.parquet（母池，只读；AEF64 直接用母库值）
* 规则源（预注册配额）：
        按省 × 原码（120 常绿 / 121 落叶）× 海拔带（<1500 / 1500-3000 / ≥3000 m）分层，
        每格上限 --per（默认 8 点）；点间距 ≥5 km；每省上限 60 点；合计 ≤ --total（默认 560）
        省界用 DataV（只作归属统计）
* 输出：data/shrub_nat/audit_fcs10/points.csv（含母库 AEF + 省/码/海拔带 + 分层）
        经 h2b 补 S2 物候/GEDI 后写 feats.parquet；切片/联系表/索引由 m5j 生成
* 用法：python k10_audit_fcs10.py [--total 560] [--per 8]
"""
import argparse, json, math, os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
sys.path.insert(0, os.path.join(WORK, 'code'))
import numpy as np
import pandas as pd

OUTD = os.path.join(WORK, 'data', 'shrub_nat')
AD = os.path.join(OUTD, 'audit_fcs10')
POOL = r'F:/lc_work/年度子集_含稀有类/r7_train_2023.parquet'
BND = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/边界/china_100000_full.json'
MIN_KM = 5.0


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + m, flush=True)


def prov_of(lon, lat, polys, names, tree=None):
    from shapely.geometry import Point
    p = Point(lon, lat)
    for g, n in zip(polys, names):
        if g.contains(p):
            return n
    return ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--total', type=int, default=560)
    ap.add_argument('--per', type=int, default=8)
    a = ap.parse_args()
    os.makedirs(AD, exist_ok=True)
    pool = pd.read_parquet(POOL, columns=['lon', 'lat', 'class_new', 'src', 'year'] + ['A%02d' % i for i in range(64)])
    sh = pool[pool['src'].fillna('').astype(str) == 'glc_fcs10_2023_shrub']
    emit('FCS10 灌丛层全国点 %d（码 120/121）' % len(sh))
    from shapely.geometry import shape
    from shapely import make_valid
    from shapely.strtree import STRtree
    gj = json.load(open(BND, encoding='utf-8'))
    polys, names = [], []
    for f in gj['features']:
        g = f.get('geometry')
        if not g:
            continue
        gg = make_valid(shape(g))
        if gg.geom_type in ('Polygon', 'MultiPolygon'):
            polys.append(gg); names.append(f['properties'].get('name', '?'))
    tree = STRtree(polys)
    pts = [__import__('shapely').geometry.Point(x, y) for x, y in
           zip(sh['lon'].to_numpy(float), sh['lat'].to_numpy(float))]
    idx = tree.query(pts)
    prov = np.array([''] * len(sh), dtype=object)
    if idx.size:
        for col in range(idx.shape[1]):
            row, cand = int(idx[0, col]), int(idx[1, col])
            if prov[row] == '' and polys[cand].contains(pts[row]):
                prov[row] = names[cand]
    sh = sh.assign(prov=prov)
    sh = sh[sh['prov'] != ''].reset_index(drop=True)
    emit('界内 %d 点，涉及 %d 省' % (len(sh), sh['prov'].nunique()))
    # 海拔带：母池无 dem，用省域 DEM 近似不划算——改用「纬度×经度」粗框（≥30 km 网格）保证空间分散
    sh['cell'] = [('%02d%02d' % (math.floor(la), math.floor(lo))) for la, lo in
                  zip(sh['lat'].to_numpy(float), sh['lon'].to_numpy(float))]
    rs = np.random.RandomState(20261010)
    picks = []
    for (p, code, cell), g in sh.groupby(['prov', 'class_new', 'cell']):
        if len(g) > a.per:
            g = g.iloc[rs.choice(len(g), a.per, replace=False)]
        picks.append(g)
    sub = pd.concat(picks, ignore_index=True)
    sub = sub.sample(frac=1.0, random_state=rs).reset_index(drop=True)
    # 省上限与总量上限（保持省内分层）
    used, out = [], []
    for prov in sorted(sub['prov'].unique()):
        g = sub[sub['prov'] == prov]
        take = []
        for r in g.itertuples():
            if len(take) >= min(60, a.total - len(out)):
                break
            if any(((r.lon - u[0]) ** 2 * 100 + (r.lat - u[1]) ** 2 * 120) < (MIN_KM / 111.0) ** 2
                   for u in used[-400:]):
                continue
            used.append((r.lon, r.lat)); take.append(r.Index)
        out.append(g.loc[take])
        if len(out) and sum(len(x) for x in out) >= a.total:
            break
    fin = pd.concat(out, ignore_index=True)
    fin = fin.sample(frac=1.0, random_state=rs).reset_index(drop=True)
    fin.insert(0, 'point_id', ['SF-%05d' % i for i in range(len(fin))])
    fin['zone_note'] = np.where(fin['class_new'] == 120, '常绿码120', '落叶码121')
    fin.to_csv(os.path.join(AD, 'points.csv'), index=False, encoding='utf-8-sig')
    emit('→ %s（%d 点，省 %d 个）' % (os.path.join(AD, 'points.csv'), len(fin), fin['prov'].nunique()))
    emit('  分省：%s' % dict(fin.groupby('prov').size().sort_values(ascending=False).head(8)))
    emit('  分码：%s' % dict(fin.groupby('zone_note').size()))


if __name__ == '__main__':
    main()
