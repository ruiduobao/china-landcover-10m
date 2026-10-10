# -*- coding: utf-8 -*-
"""g8_t2_build.py — T2 扩增点选取 + 物候/地形特征 + 抽检包

* 输入：池（AEF）+ judged_all.csv（78 正/328 负）
* 规则源（预注册）：
  T2 选取：p_shrub ≥0.60；每省 ≤120；与**已判读点**间距 ≥1 km（防空间自相关虚高）；
          省内点间 ≥300 m；另记录 A(灌丛层内)/B(全池发现) 两个变体
  抽检：T2 中每省随机 30 点（种子固定）→ 交影像判读，**用于测 T2 真实精度**（不能只报模型概率）
* 输出：data/shrub/t2_points.csv（含 p_shrub / 变体 / 是否抽检）
        data/shrub/t2_s2dem.csv（物候+地形，GEE 取）
* 用法：python g8_t2_build.py [--acct bx15mw]
"""
import argparse, os, sys, time
sys.path.insert(0, r'F:/lc_work/v31_exp/code')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
import v31_common as VC
from sklearn.ensemble import RandomForestClassifier

W = r'F:/lc_work/v31_exp'; OUTD = os.path.join(W, 'data', 'shrub')
POOL = r'F:/lc_work/prod5p_2023/data/train2023_clean.parquet'
SHRUB_SRC = 'glc_fcs10_2023_shrub'
PROV = {'宁夏': (104.0, 35.0, 108.0, 40.0), '四川': (97.0, 26.0, 109.0, 34.5),
        '黑龙江': (121.0, 43.0, 135.5, 53.5), '福建': (115.5, 23.0, 120.5, 28.5)}
P_MIN = 0.60; PER_PROV = 120; MIN_KM = 0.30; MIN_KM_JUDGED = 1.0
N_AUDIT_PER = 30
SEED = 20261010

def emit(m): print(time.strftime('[%m-%d %H:%M:%S] ')+str(m), flush=True)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--acct', default='bx15mw'); a = ap.parse_args()
    j = pd.read_csv(os.path.join(OUTD, 'judged_all.csv'), encoding='utf-8-sig')
    cand = pd.read_csv(os.path.join(OUTD, 'cand_all.csv'), encoding='utf-8-sig')
    miss = [c for c in VC.FEATS if c not in j.columns]
    if miss:
        j = j.merge(cand[['point_id'] + miss], on='point_id', how='left')
    tr = j[j.Q1_shrub.isin(['是', '否'])]
    y = (tr.Q1_shrub == '是').astype(int).to_numpy()
    clf = RandomForestClassifier(n_estimators=300, class_weight='balanced_subsample',
                                n_jobs=15, random_state=7).fit(
        np.nan_to_num(tr[VC.FEATS].to_numpy('float32'), nan=-999), y)
    pool = pd.read_parquet(POOL, columns=['lon', 'lat', 'class_new', 'src'] + VC.FEATS)
    pool['prov'] = ''
    for p, (x0, y0, x1, y1) in PROV.items():
        pool.loc[(pool.lon >= x0) & (pool.lon <= x1) & (pool.lat >= y0) & (pool.lat <= y1), 'prov'] = p
    s = pool[pool.prov != ''].copy()
    s['p_shrub'] = clf.predict_proba(np.nan_to_num(s[VC.FEATS].to_numpy('float32'), nan=-999))[:, 1]
    judged = j[['lon', 'lat']].to_numpy()
    rng = np.random.RandomState(SEED)
    rows = []
    for p in PROV:
        g = s[(s.prov == p) & (s.p_shrub >= P_MIN)].sort_values('p_shrub', ascending=False)
        used, n = [], 0
        for r in g.itertuples():
            if n >= PER_PROV: break
            # 与已判读点及本批已选点保持间距
            if len(judged) and np.min(np.hypot((judged[:, 0] - r.lon) * 111.3, (judged[:, 1] - r.lat) * 111.0)) < MIN_KM_JUDGED:
                continue
            if any(((r.lon - u) ** 2 * 100 + (r.lat - v) ** 2 * 120) < (MIN_KM / 111.0) ** 2 for u, v in used):
                continue
            used.append((r.lon, r.lat)); n += 1
            rows.append(dict(prov=p, lon=float(r.lon), lat=float(r.lat), p_shrub=round(float(r.p_shrub), 4),
                             src=str(r.src), class_new=int(r.class_new), aef_idx=int(r.Index)))
        emit('T2 %s：命中 %d → 取 %d' % (p, len(g), n))
    t2 = pd.DataFrame(rows)
    t2.insert(0, 'point_id', ['T2-%s-%04d' % (r.prov, i) for i, r in enumerate(t2.itertuples())])
    t2['variant'] = np.where(t2.src.fillna('') == SHRUB_SRC, 'A_灌丛层内', 'B_全池发现')
    # 抽检集
    aud = []
    for p, g in t2.groupby('prov'):
        k = min(N_AUDIT_PER, len(g))
        sel = g.iloc[rng.choice(len(g), k, replace=False)]
        aud += list(sel.point_id)
    t2['audit'] = t2.point_id.isin(set(aud))
    t2.to_csv(os.path.join(OUTD, 't2_points.csv'), index=False, encoding='utf-8-sig')
    emit('T2 合计 %d（抽检 %d）→ t2_points.csv' % (len(t2), int(t2.audit.sum())))
    # 物候/地形
    import ee
    VC.ensure_ctx(a.acct)
    def s2():
        def msk(im):
            scl = im.select('SCL'); return im.updateMask(scl.neq(3).And(scl.neq(8)).And(scl.neq(9)))
        col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterDate('2022-12-01', '2024-01-01')
               .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60)).map(msk))
        b = []
        for tag, x, y in (('djf', '2022-12-01', '2023-03-01'), ('jja', '2023-06-01', '2023-09-01')):
            c = col.filterDate(x, y).median()
            b.append(c.normalizedDifference(['B8', 'B4']).rename('ndvi_' + tag))
        return ee.Image.cat(b)
    img = ee.Image.cat([s2(), ee.Image('USGS/SRTMGL1_003').rename('dem')])
    pts = [ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]), {'point_id': r.point_id})
           for r in t2.itertuples()]
    got = []
    for k in range(0, len(pts), 3000):
        for att in range(4):
            try:
                r = img.sampleRegions(collection=ee.FeatureCollection(pts[k:k + 3000]), scale=30,
                                      geometries=False, tileScale=4).getInfo()['features']
                got += [dict(point_id=f['properties']['point_id'],
                             **{c: f['properties'].get(c) for c in ('ndvi_djf', 'ndvi_jja', 'dem')}) for f in r]
                break
            except Exception as e:
                emit('  取样重试%d %s' % (att + 1, str(e)[:70])); time.sleep(8)
    f = pd.DataFrame(got)
    f.to_csv(os.path.join(OUTD, 't2_s2dem.csv'), index=False, encoding='utf-8-sig')
    emit('物候/地形 %d/%d' % (len(f), len(t2)))

if __name__ == '__main__': main()
