# -*- coding: utf-8 -*-
"""f5c_water_occ.py — 水体丰度证据（服务端采样，避免栅格下载算力墙）

* 输入：GEE `JRC/GSW1_4/GlobalSurfaceWater` occurrence（1984–2021 水出现频率，30 m 原生）
* 规则源（预注册，服务用户问 1/2）：
        · 点位频率 occ_pt：该 30 m 像元自身的水出现频率（0–100）
        · 邻域频率 occ_90：5×5 像元（约 150 m）窗口内的**最大值** —— 代表"该点附近是否有水"
        判据候选：湖河滩地/木本沼泽/草本沼泽/盐渍湿地 需 occ_90 ≥ {1, 10, 50}
* 门槛：每批 ≤4000 点（GEE 交互上限）；点集=池内 14/15/16/17 全量分层抽样 + 判读参考点全量
* 输出：data/eco_gate/_water/water_occ_points.csv（point_id,occ_pt,occ_90,scope）
* 用法：python f5c_water_occ.py [--acct bx15mw]
"""
import argparse, os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
sys.path.insert(0, os.path.join(WORK, 'code'))
import numpy as np, pandas as pd, v31_common as VC

OUTD = os.path.join(WORK, 'data', 'eco_gate', '_water')
M3 = os.path.join(WORK, 'data', 'm3')
POOL = r'F:/lc_work/prod5p_2023/data/train2023_clean.parquet'
WET = [14, 15, 16, 17]
SEED = 20261010


def emit(m): print(time.strftime('[%m-%d %H:%M:%S] ')+str(m), flush=True)


def collect(df, scope):
    import ee
    pts = [ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]), {'point_id': r.point_id})
           for r in df.itertuples()]
    occ = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence')
    pt = occ.rename('occ_pt')
    nb = occ.reduceNeighborhood(reducer=ee.Reducer.max(),
                                kernel=ee.Kernel.square(radius=2, units='pixels')).rename('occ_90')
    stack = pt.addBands(nb)
    rows = []
    step = 3000
    for k in range(0, len(pts), step):
        for att in range(4):
            try:
                got = stack.sampleRegions(collection=ee.FeatureCollection(pts[k:k+step]),
                                          scale=30, geometries=False, tileScale=4).getInfo()['features']
                rows += [dict(point_id=f['properties']['point_id'],
                              occ_pt=f['properties'].get('occ_pt'),
                              occ_90=f['properties'].get('occ_90'), scope=scope) for f in got]
                emit('  %s %d/%d' % (scope, min(k+step, len(pts)), len(pts)))
                break
            except Exception as e:
                emit('  重试%d %s' % (att+1, str(e)[:80])); time.sleep(10)
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--acct', default='bx15mw'); a = ap.parse_args()
    os.makedirs(OUTD, exist_ok=True)
    fp = os.path.join(OUTD, 'water_occ_points.csv')
    VC.ensure_ctx(a.acct)
    rng = np.random.RandomState(SEED)
    rows = []
    # 池内湿类（14/15/16/17）全量（若 ≤3000 用全量，否则抽样）
    pool = pd.read_parquet(POOL, columns=['lon', 'lat', 'class_new'])
    pool['cls'] = VC.to_v31(pool['class_new'].to_numpy(int))
    for c in WET:
        s = pool[pool.cls == c]
        if len(s) == 0: continue
        if len(s) > 3000:
            s = s.iloc[rng.choice(len(s), 3000, replace=False)]
        s = s.assign(point_id=['POOL-%d-%05d' % (c, i) for i in range(len(s))])
        rows += collect(s[['point_id', 'lon', 'lat']], 'pool_%d' % c)
    # 参考点（判读真值）
    R = pd.read_csv(os.path.join(M3, 'gap_points.csv'), encoding='utf-8-sig')
    R = R.assign(point_id=R.point_id.astype(str))
    rows += collect(R[['point_id', 'lon', 'lat']], 'gap')
    Rs = pd.read_csv(os.path.join(M3, 'refset_all.csv'), encoding='utf-8-sig')
    Rs = Rs.assign(point_id=Rs.point_id.astype(str))
    rows += collect(Rs[['point_id', 'lon', 'lat']], 'refset')
    out = pd.DataFrame(rows)
    out.to_csv(fp, index=False, encoding='utf-8-sig')
    emit('水丰度证据 %d 点 → %s' % (len(out), fp))
    for sc, g in out.groupby('scope'):
        v = pd.to_numeric(g.occ_90, errors='coerce')
        emit('  %-10s n=%4d occ_90 中位 %s；≥1 %.1f%%；≥50 %.1f%%' % (
            sc, len(g), None if v.isna().all() else round(float(v.median()), 1),
            100*float((v >= 1).mean()), 100*float((v >= 50).mean())))


if __name__ == '__main__': main()
