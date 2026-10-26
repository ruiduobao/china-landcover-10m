# -*- coding: utf-8 -*-
"""m3_gee_struct.py — 方法C/A 的 GEE 侧结构层与第三外部产品抽样
* 输入：labels/methods/work_base.parquet（分层抽样）
* 层：CGLS-LC100 C3（2015, 100m）forest_type + tree-coverfraction；
      Hansen GFC 2024 v1.12（30m）treecover2000 + lossyear；
      GEDI L2A monthly（2019-2024）rh98/rh50/cover —— 存在性与覆盖另行报告
* 输出：labels/methods/gee_struct_sample.parquet；labels/methods/m3_gee.json
* 用法：python m3_gee_struct.py [--n 40000] [--chunk 8000]
"""
import os, sys, json, time, argparse, warnings
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

WORK = r'F:/lc_work/nb_national_2026/labels/methods'
ACCTS = ['als74akz', 'zitwwufh', 'ybzljyma', 'gm9ufoo4', 'nnjxsn', 'seqsiu', 'iu5f4z', 'xysongf']


def init_ee():
    import ee
    for a in ACCTS:
        try:
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter('always')
                C.apply_account_env(a)
                pid = C.ACCOUNTS[a]['proj']
                ee.Initialize(project=pid)
                msgs = ' '.join(str(x.message) for x in ws)
                if 'restricted' in msgs or 'quota' in msgs:
                    print('  acct', a, 'RESTRICTED'); continue
            print('  using', a, pid)
            return ee, a
        except Exception as e:
            print('  acct', a, 'FAIL', repr(e)[:100])
    raise RuntimeError('no usable GEE account')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=40000)
    ap.add_argument('--chunk', type=int, default=8000)
    args = ap.parse_args()

    df = pd.read_parquet(os.path.join(WORK, 'work_base.parquet'),
                         columns=['stable_uid', 'lon', 'lat', 'cl_24', 'region', 'cz',
                                  'pheno_axis', 'leaf_fcs10_s', 'leaf_modis_s'])
    # 分层：类×大区 至少保底，余量随机
    frac = min(1.0, args.n / len(df))
    samp = df.groupby(['cl_24', 'region'], group_keys=False).apply(
        lambda g: g.sample(max(50, int(round(len(g) * frac))), random_state=7), include_groups=False)
    samp = samp.sample(min(args.n, len(samp)), random_state=8).reset_index(drop=True)
    print('sample', len(samp))

    ee, acct = init_ee()
    cgls = ee.ImageCollection('COPERNICUS/Landcover/100m/Proba-V-C3/Global').first() \
        .select(['forest_type', 'tree-coverfraction'])
    hansen = ee.Image('UMD/hansen/global_forest_change_2024_v1_12') \
        .select(['treecover2000', 'lossyear', 'gain'])

    rows = []
    t00 = time.time()
    for i in range(0, len(samp), args.chunk):
        c = samp.iloc[i:i + args.chunk]
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]), {'uid': str(r.stable_uid)})
            for r in c.itertuples()])
        t0 = time.time()
        got = {}
        for nm, img, sc in [('cgls', cgls, 100), ('hansen', hansen, 30)]:
            for attempt in range(3):
                try:
                    r = img.sampleRegions(collection=fc, scale=sc, geometries=False,
                                          tileScale=8).getInfo()
                    got[nm] = {f['properties']['uid']: f['properties'] for f in r['features']}
                    break
                except Exception as e:
                    print('   ', nm, 'retry', attempt, repr(e)[:120]); time.sleep(3 * (attempt + 1))
        for uid in c['stable_uid'].astype(str):
            a = got.get('cgls', {}).get(uid, {})
            b = got.get('hansen', {}).get(uid, {})
            rows.append(dict(stable_uid=uid,
                             cgls_ft=a.get('forest_type'), cgls_tcd=a.get('tree-coverfraction'),
                             hansen_tc2000=b.get('treecover2000'), hansen_lossyr=b.get('lossyear'),
                             hansen_gain=b.get('gain')))
        print('chunk', i // args.chunk, 'n', len(c), 't', round(time.time() - t0, 1),
              'elapsed', round(time.time() - t00, 1), flush=True)

    out = pd.DataFrame(rows)
    # 编码
    FT_N = {1, 3}   # evergreen needleleaf, deciduous needleleaf
    FT_B = {2, 4}
    out['leaf_cgls_s'] = out['cgls_ft'].map(lambda x: 1 if x in FT_N else (0 if x in FT_B else np.nan))
    # Hansen 结构：树覆盖 > 0 视为有树（2000 基线）
    out['hansen_forest'] = np.where(out['hansen_tc2000'].isna(), np.nan,
                                    (out['hansen_tc2000'] >= 20).astype(float))
    p = os.path.join(WORK, 'gee_struct_sample.parquet')
    out.to_parquet(p, index=False)
    stats = dict(acct=acct, n=len(out), path=p,
                 cgls_ft_counts={str(k): int(v) for k, v in out['cgls_ft'].value_counts(dropna=False).items()},
                 cgls_decided=int(out['leaf_cgls_s'].notna().sum()),
                 hansen_nonnull=int(out['hansen_tc2000'].notna().sum()))
    json.dump(stats, open(os.path.join(WORK, 'm3_gee.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1, default=str)
    print(json.dumps(stats, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
