# -*- coding: utf-8 -*-
"""k6_fix_feats.py — 修正 judged_feats：① 恢复 k1 统一标签 ② 以 scale=10 重取 AEF 与母库对齐

* 输入：data/shrub_nat/judged_feats.parquet（k2 产物）+ data/shrub_nat/judged_pool.csv（k1 标签）
* 规则源：AEF 与母池同源——母池 A00..A63 来自 GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL 的 10 m 原生网格；
        k2 用 scale=20 采样会把 2×2 个 10 m 像元做均値，造成与母库的系统性偏差（实测中位差 5e-3）。
        本脚本以 scale=10 重取并校验与母池的一致性（同点 |ΔA00| 中位数 <1e-3 视为同源）。
* 门槛：一致性校验输出同源比例与中位差；不一致则在输出里标记 aef_consistent=0
* 输出：data/shrub_nat/judged_feats.parquet（覆盖）+ data/shrub_nat/aef_check.json
* 用法：python k6_fix_feats.py [--acct bx15mw]
"""
import argparse, json, os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
sys.path.insert(0, os.path.join(WORK, 'code'))
import numpy as np
import pandas as pd
import v31_common as VC

OUTD = os.path.join(WORK, 'data', 'shrub_nat')
POOL = r'F:/lc_work/年度子集_含稀有类/r7_train_2023.parquet'


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + m, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acct', default='bx15mw')
    a = ap.parse_args()
    global ee
    import ee as _ee
    ee = _ee
    VC.ensure_ctx(a.acct)
    j = pd.read_parquet(os.path.join(OUTD, 'judged_feats.parquet'))
    poolcsv = pd.read_csv(os.path.join(OUTD, 'judged_pool.csv'), encoding='utf-8-sig')
    poolcsv['point_id'] = poolcsv['point_id'].astype(str)
    j = j.drop(columns=[c for c in ('is_shrub', 'shrub_type', 'conf', 'src_exp', 'note') if c in j.columns])
    j = j.merge(poolcsv[['point_id', 'is_shrub', 'shrub_type', 'conf', 'src_exp']],
                on='point_id', how='left')
    emit('标签恢复：%s' % j['is_shrub'].value_counts().to_dict())
    # ---- scale=10 重取 AEF ----
    pts = [ee.Feature(ee.Geometry.Point([float(x), float(y)]), {'point_id': pid})
           for pid, x, y in zip(j['point_id'], j['lon'].to_numpy(float), j['lat'].to_numpy(float))]
    aef = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL') \
        .filterDate('2023-01-01', '2024-01-01').mosaic().select(VC.FEATS).toFloat()
    rows = []
    for k in range(0, len(pts), 1500):
        for att in range(5):
            try:
                r = aef.sampleRegions(collection=ee.FeatureCollection(pts[k:k + 1500]), scale=10,
                                      geometries=False, tileScale=4).getInfo()['features']
                rows += [f['properties'] for f in r]
                break
            except Exception as e:
                emit('  AEF 块 %d 重试%d %s' % (k, att + 1, str(e)[:70]))
                time.sleep(8 * (att + 1))
        if (k // 1500) % 3 == 0:
            emit('  AEF %d/%d' % (min(k + 1500, len(pts)), len(pts)))
    fa = pd.DataFrame(rows).drop_duplicates(subset=['point_id'])
    for c in VC.FEATS:
        if c in fa.columns:
            j[c] = j['point_id'].map(dict(zip(fa['point_id'], fa[c])))
    j['aef_ok'] = j['A00'].notna().astype(int)
    emit('AEF(10m) 重取 %d/%d' % (int(j['aef_ok'].sum()), len(j)))
    # ---- 与母池一致性 ----
    pool = pd.read_parquet(POOL, columns=['lon', 'lat', 'A00', 'A63'])
    pk = {(round(lo, 6), round(la, 6)): (a0, a63)
          for lo, la, a0, a63 in zip(pool['lon'].to_numpy(float), pool['lat'].to_numpy(float),
                                     pool['A00'].to_numpy(float), pool['A63'].to_numpy(float))}
    d0, d63, hit = [], [], 0
    for r in j.itertuples():
        kk = (round(float(r.lon), 6), round(float(r.lat), 6))
        if kk in pk and pd.notna(r.A00):
            hit += 1
            d0.append(abs(pk[kk][0] - float(r.A00)))
            d63.append(abs(pk[kk][1] - float(r.A63)))
    med = float(np.median(d0)) if d0 else -1
    chk = dict(n_in_pool=hit, n_compared=len(d0), median_abs_dA00=med,
               median_abs_dA63=(float(np.median(d63)) if d63 else -1),
               aef_consistent=bool(med < 1e-3))
    json.dump(chk, open(os.path.join(OUTD, 'aef_check.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    emit('母池一致性：命中 %d，|ΔA00| 中位 %.2e → %s' % (
        hit, med, '同源' if chk['aef_consistent'] else '不同源'))
    j.to_parquet(os.path.join(OUTD, 'judged_feats.parquet'), index=False)
    emit('→ judged_feats.parquet 已更新')


if __name__ == '__main__':
    main()
