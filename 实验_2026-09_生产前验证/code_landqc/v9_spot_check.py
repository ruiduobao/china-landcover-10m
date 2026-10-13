# -*- coding: utf-8 -*-
"""v9_spot_check.py — 其余地类质检的独立 GEE 抽查（双路比对）

* 输入：--group <crop|veg|water|urban>；out/land_qc_<tag>.parquet（v8 产物）
* 方法：随机抽 --n 点，两条独立路径重取：
    路A：v7_land_fetch.build()（同一构建，验证块产物/合并保真）
    路B：**独立最小构建**（不复用 v7 代码；GSW/DW/S2/dem 各写一遍原始表达式，不做 unmask/rename 处理）
* 比对：统一走 v8.derive 归一 → A vs 产物（max|Δ|，应 0.0）；B vs A（两者均非空前，应 0.0）
* 输出：reports/land_qc_2026-10-13/spot_<tag>.json + 行内结论
* 用法：python v9_spot_check.py --group water --n 40 --acct vidalmvtpronet18
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'; CODE = os.path.join(WORK, 'code')
OUT = os.path.join(WORK, 'data', 'land_qc', 'out')
REP = os.path.join(WORK, 'reports', 'land_qc_2026-10-13')
COLS = ['max_extent', 'occurrence', 'water', 'trees', 'grass', 'crops', 'built', 'bare',
        'snow_and_ice', 'shrub', 'dem', 's2_ndvi_xia']


def build_B(ee):
    """独立最小构建（写法与 v7 不同：分开 filter/select，逐源 unmask 后才 cat —— 平台要求，
    否则 sampleRegions 遇任一源掩膜即整点丢特征，无法比对）。"""
    g = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select(['max_extent', 'occurrence']).unmask(-1)
    dwm = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1').filterDate('2023-01-01', '2024-01-01')
           .select(['water', 'trees', 'grass', 'crops', 'built', 'bare', 'snow_and_ice',
                    'shrub_and_scrub']).mean().unmask(-1)
           .rename(['water', 'trees', 'grass', 'crops', 'built', 'bare', 'snow_and_ice', 'shrub']))
    dem = ee.Image('USGS/SRTMGL1_003').select('elevation').rename('dem').unmask(-9999)
    col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    def m(im):
        s = im.select('SCL')
        return im.updateMask(s.neq(3).And(s.neq(8)).And(s.neq(9)))
    c = col.filterDate('2023-06-01', '2023-09-01').filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60)).map(m).median()
    ndvi = c.normalizedDifference(['B8', 'B4']).multiply(1000).unmask(-9999).rename('s2_ndvi_xia')
    return ee.Image.cat([g, dwm, dem, ndvi]).toFloat()


def norm(t):
    """v9 侧的轻量归一（与 v8.derive 同语义，只覆盖 v9 抓取的列）。"""
    t = t.copy()
    for c in ('occurrence', 'max_extent'):
        if c in t.columns:
            t[c] = pd.to_numeric(t[c], errors='coerce').where(lambda x: x >= 0)
    for c in ('water', 'trees', 'grass', 'crops', 'built', 'bare', 'snow_and_ice', 'shrub'):
        t[c] = pd.to_numeric(t[c], errors='coerce').fillna(-1).clip(0, 1)
    for c in ('dem', 's2_ndvi_xia'):
        t[c] = pd.to_numeric(t[c], errors='coerce').where(lambda x: x > -9000)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', required=True)
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--acct', required=True)
    a = ap.parse_args()
    sys.path.insert(0, CODE)
    import ee, v31_common as VC, v7_land_fetch as V7
    VC.ensure_ctx(a.acct)
    prod = pd.read_parquet(os.path.join(OUT, 'land_qc_%s.parquet' % a.group))
    prod['point_id'] = prod.point_id.astype(str)
    pool = prod[prod.has_ev == 1] if 'has_ev' in prod.columns else prod
    s = pool.sample(min(a.n, len(pool)), random_state=np.random.RandomState(a.seed)).reset_index(drop=True)
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(x), float(y)]), {'pid': str(p)})
                               for p, x, y in zip(s.point_id, s.lon, s.lat)])
    grabs = {}
    for tag, img in (('A', V7.build(ee)[0]), ('B', build_B(ee))):
        r = img.sampleRegions(collection=fc, scale=30, geometries=False, tileScale=4).getInfo()['features']
        got = {f['properties']['pid']: f['properties'] for f in r}
        t = pd.DataFrame([dict(point_id=p, **{c: got.get(p, {}).get(c) for c in COLS})
                          for p in s.point_id])
        grabs[tag] = norm(t)
    # A vs 产物（同口径归一后比）
    mm = norm(s[['point_id'] + COLS]).merge(grabs['A'], on='point_id', suffixes=('_o', '_A'))
    rep, bad = {}, []
    for c in COLS:
        dd = (mm[c + '_o'] - mm[c + '_A']).abs()
        mx = round(float(dd.max()) if dd.notna().any() else -1.0, 6)
        rep['prod_vs_A_' + c] = mx
        if mx > 1e-6:
            bad.append(c)
    mb = grabs['A'].merge(grabs['B'], on='point_id', suffixes=('_A', '_B'))
    for c in COLS:
        x, y = mb[c + '_A'], mb[c + '_B']
        both = x.notna() & y.notna()
        mx = round(float((x[both] - y[both]).abs().max()) if both.any() else -1.0, 6)
        rep['B_vs_A_' + c] = mx
        n_nanA = int(x.isna().sum())
        n_nanB = int(y.isna().sum())
        if mx > 1e-6 or abs(n_nanA - n_nanB) > 0:
            bad.append('B:%s(mismatch)' % c)
    os.makedirs(REP, exist_ok=True)
    json.dump(dict(group=a.group, n=len(s), acct=a.acct, diffs=rep, mismatch_bands=bad),
              open(os.path.join(REP, 'spot_%s.json' % a.group), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    print('MISMATCH BANDS:', bad if bad else 'NONE（全部 0.0）')


if __name__ == '__main__':
    main()
