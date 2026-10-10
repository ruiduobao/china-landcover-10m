# -*- coding: utf-8 -*-
"""h2_merge_nat.py — 候选合并 + 省域归属 + 生态区带 + 分层（本地，零 GEE）

* 输入：data/shrub_nat/parts/cand_*.parquet（h1 产物）
* 规则源：
        省域归属——PROJ/数据/边界/china_100000_full.json（DataV，GCJ-02；仅作归属统计用）
        区带 zone——按 (dem, bio01, bio12) 决策树（预注册，见 zone_of）
        分层 tier——A 多源一致（wc20 或 lc100 或 modis 认灌丛）/ B 单源 DW
        band——P（dwp≥0.35）/ N（0.15≤dwp<0.35，硬负样本）
* 门槛：剔除 dem≤0（海面）；区带与省的交叉表落盘
* 输出：data/shrub_nat/cand_nat.parquet + data/shrub_nat/cand_nat_stats.json
* 用法：python h2_merge_nat.py
"""
import json, os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
sys.path.insert(0, os.path.join(WORK, 'code'))
import numpy as np
import pandas as pd

OUTD = os.path.join(WORK, 'data', 'shrub_nat')
BND = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/边界/china_100000_full.json'


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + m, flush=True)


def zone_of(dem, bio01, bio12):
    """WorldClim/V1/BIO 的 bio01 单位是 °C×10（实测中位 45=4.5°C，2026-10-10 复核），
    bio12 为 mm。bio01=-999 或缺测按温带处理并在 stats 里计数。"""
    if bio01 <= -999:
        return '缺失'
    if dem >= 3000:
        return '青藏高原高山'
    t = bio01 / 10.0
    if t >= 20:
        return '热带'
    if t >= 13 and bio12 >= 900:
        return '亚热带'
    if t <= -2:
        return '寒温带'
    if bio12 < 300:
        return '干旱'
    if bio12 < 500:
        return '半干旱'
    return '温带'


def main():
    import glob
    files = sorted(glob.glob(os.path.join(OUTD, 'parts', 'cand_*.parquet')))
    if not files:
        emit('没有 parts 文件'); return
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    emit('合并 %d 个 part，共 %d 行' % (len(files), len(df)))
    df = df.rename(columns={c: c.lower() for c in df.columns})
    if 'rid' in df.columns:
        df = df.drop(columns=['rid'])
    df = df.drop_duplicates(subset=['lon', 'lat']).reset_index(drop=True)
    df = df[df['dem'] > 0].reset_index(drop=True)
    emit('去重+剔海面后 %d 行；band 分布 %s' % (len(df), df['band'].value_counts().to_dict()))

    # ---- 省域归属（STRtree + bbox 预筛，避免 290k×34 全量 contains）----
    from shapely.geometry import shape, Point
    from shapely import make_valid
    from shapely.strtree import STRtree
    gj = json.load(open(BND, encoding='utf-8'))
    names, geos = [], []
    for f in gj['features']:
        g = f.get('geometry')
        if not g:
            continue
        try:
            gg = make_valid(shape(g))
            if gg.is_empty:
                continue
            names.append(f.get('properties', {}).get('name', '?'))
            geos.append(gg)
        except Exception:
            continue
    tree = STRtree(geos)
    pts = [Point(x, y) for x, y in zip(df['lon'].to_numpy(float), df['lat'].to_numpy(float))]
    idx = tree.query(pts)          # [2, n]：与点包围框相交的多边形索引
    xs = df['lon'].to_numpy(float); ys = df['lat'].to_numpy(float)
    minx = np.array([g.bounds[0] for g in geos]); miny = np.array([g.bounds[1] for g in geos])
    maxx = np.array([g.bounds[2] for g in geos]); maxy = np.array([g.bounds[3] for g in geos])
    prov = np.array([''] * len(df), dtype=object)
    if idx.size:
        for col in range(idx.shape[1]):
            row = int(idx[0, col]); cand = int(idx[1, col])
            if prov[row] != '':
                continue
            # STRtree 只保证包围框相交，逐点再做 bbox 快筛 + contains
            if not (minx[cand] <= xs[row] <= maxx[cand] and miny[cand] <= ys[row] <= maxy[cand]):
                continue
            if geos[cand].contains(pts[row]):
                prov[row] = names[cand]
    # 兜底：DataV 部分多省界有几何碎片化间隙（实测新疆界内 7 km 仍 outside，2026-10-10），
    # 对未归属点按"最近省"归属，距离 >0.3°（≈30 km）判为界外/海域剔除
    left = np.where(prov == '')[0]
    if len(left):
        d_left = []
        for row in left:
            p = pts[row]
            best, bd = None, 1e9
            for j in range(len(geos)):
                if not (minx[j] - 0.3 <= xs[row] <= maxx[j] + 0.3 and miny[j] - 0.3 <= ys[row] <= maxy[j] + 0.3):
                    continue
                d = geos[j].distance(p)
                if d < bd:
                    bd, best = d, j
            if best is not None and bd <= 0.3:
                prov[row] = names[best]; d_left.append(bd)
        emit('未归属兜底：最近省回填 %d/%d 点（中位距离 %.4f°）' % (
            len(d_left), len(left), float(np.median(d_left)) if d_left else -1))
    df['prov'] = prov
    # 界外/无省归属点（实测多为盒网格伸入哈萨克斯坦等境外，2026-10-10 复核）→ 剔除并单独计数
    n_out = int((df['prov'] == '').sum())
    df = df[df['prov'] != ''].reset_index(drop=True)
    emit('剔除界外/无省点 %d（保留 %d）' % (n_out, len(df)))
    df['zone'] = [zone_of(r.dem, r.bio01, r.bio12) for r in df.itertuples()]
    df['tier'] = np.where((df['wc20'] > 0) | (df['lc20'] > 0) | (df['mcd'] > 0), 'A', 'B')
    df['box'] = df['box'].astype(str)
    df.to_parquet(os.path.join(OUTD, 'cand_nat.parquet'), index=False)

    miss = int((df['prov'] == '').sum())
    st = dict(n=int(len(df)), n_out_of_country=n_out,
              band=df['band'].value_counts().to_dict(),
              tier=df['tier'].value_counts().to_dict(),
              zone=df['zone'].value_counts().to_dict(),
              box_n=int(df['box'].nunique()),
              by_prov_band={p: {b: int(n) for b, n in g.groupby('band').size().items()}
                            for p, g in df.groupby('prov')},
              by_zone_tier={z: {t: int(n) for t, n in g.groupby('tier').size().items()}
                            for z, g in df.groupby('zone')},
              dwp_med_by_zone={z: round(float(g['dwp'].median()), 4) for z, g in df.groupby('zone')})
    json.dump(st, open(os.path.join(OUTD, 'cand_nat_stats.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    emit('省域未归属 %d 点' % miss)
    emit('zone: %s' % st['zone'])
    emit('band: %s' % st['band'])


if __name__ == '__main__':
    main()
