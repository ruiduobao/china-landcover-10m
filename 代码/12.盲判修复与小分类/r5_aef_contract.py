# -*- coding: utf-8 -*-
"""r5_aef_contract.py — V 点 AEF64 取数 + 按 materials/_CONTRACT.json 出 R/V 交付件

* 输入：private/uid_map.csv（r2 产物）、blind/points_blind.csv
* 规则源：materials/_CONTRACT.json（experiments 组定义）；主 agent 2026-10-15：
        V 带 AEF64、无真值（cl_true/true_label/t24/label_after 禁止出现）；
        R 未经 QC **不得**写成修复成品 → 只出 `R_repair_candidates.parquet`（QC 字段显式 pending）。
* 门槛：AEF 取数用 reduceRegions + first()（sampleRegions 会丢点，见 r3 经验）；
        分块 ≤200 点、单请求 ≤90 s、重试 ≤4；AEF 全 64 维非空才算有效（记入 judge_status）。
* 输出：cache/aef_parts/*.parquet、materials/V_eval.parquet、materials/R_repair_candidates.parquet、
        materials/_READY_STATUS.json、qc/aef_fetch_summary.json
* 用法：python r5_aef_contract.py --acct p2vlkaia [--skip-aef]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rc_conf import *   # noqa

sys.path.insert(0, os.path.join(WORK, 'v31_exp', 'code'))
AEFD = os.path.join(CACHE, 'aef_parts')
WIN = 'w2'          # 五窗中的四川窗（本次两框为川东北/川东新框，另以 box/tile 列精确标注）


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def fetch_aef(acct, pts, chunk=200, retries=4):
    import ee
    import v31_common as VC
    VC.ensure_ctx(acct)
    os.makedirs(AEFD, exist_ok=True)
    feats = ['A%02d' % i for i in range(64)]
    frames = []
    for k in range(0, len(pts), chunk):
        sub = pts.iloc[k:k + chunk]
        fp = os.path.join(AEFD, 'aef_c%04d.parquet' % k)
        if os.path.exists(fp):
            frames.append(pd.read_parquet(fp))
            continue
        box = ee.Geometry.BBox(float(sub['lon'].min()), float(sub['lat'].min()),
                               float(sub['lon'].max()), float(sub['lat'].max()))
        # 关键：AEF 年度集是分块影像 → 必须 filterBounds + mosaic（`first()` 会取到不覆盖本区的块，
        # 实测 80/80 全 null）；逐 band unmask(-9999) 后 cat（与生产 g0/f11 同源做法）。
        col = ee.ImageCollection(AEF_COLLECTION).filterDate('%d-01-01' % S2_YEAR,
                                                           '%d-01-01' % (S2_YEAR + 1)).filterBounds(box)
        img = col.mosaic()
        img = ee.Image.cat([img.select(b).unmask(-9999).rename(b) for b in feats]).toFloat()
        emit('  AEF %s：候选影像 %d 景，已 mosaic' % (S2_YEAR, col.size().getInfo()))
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(x), float(y)]), {'point_id': p})
                                   for p, x, y in zip(sub['point_id'], sub['lon'], sub['lat'])])
        rows = None
        for att in range(retries):
            try:
                r = img.reduceRegions(collection=fc, reducer=ee.Reducer.first(),
                                      scale=10, tileScale=4).getInfo()['features']
                rows = [dict(f['properties']) for f in r]
                break
            except Exception as e:
                emit('  AEF 块 %d 重试 %d：%s' % (k // chunk, att + 1, str(e)[:100]))
                time.sleep(min(30, 6 * (att + 1)))
        if rows is None:
            raise SystemExit('AEF 取数失败（块 %d）' % (k // chunk))
        pd.DataFrame(rows).to_parquet(fp, index=False)
        frames.append(pd.DataFrame(rows))
        emit('  AEF 块 %d：%d 点' % (k // chunk, len(rows)))
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acct', default='p2vlkaia')
    ap.add_argument('--skip-aef', action='store_true')
    a = ap.parse_args()
    ensure_dirs()
    side = pd.read_csv(os.path.join(PRIV, 'uid_map.csv'), encoding='utf-8-sig')
    side['point_id'] = side['point_id'].astype(str)
    v = side[side['role'] == ROLE_V].copy().reset_index(drop=True)
    r = side[side['role'] == ROLE_R].copy().reset_index(drop=True)
    emit('V %d 点 / R %d 点' % (len(v), len(r)))

    if not a.skip_aef:
        aef = fetch_aef(a.acct, v[['point_id', 'lon', 'lat']])
        aef['point_id'] = aef['point_id'].astype(str)
        aef.to_parquet(os.path.join(CACHE, 'v_aef2023.parquet'), index=False)
    else:
        aef = pd.read_parquet(os.path.join(CACHE, 'v_aef2023.parquet'))
    feats = ['A%02d' % i for i in range(64)]
    for f in feats:
        if f not in aef.columns:
            aef[f] = np.nan
    m = v.merge(aef[['point_id'] + feats], on='point_id', how='left')
    m[feats] = m[feats].replace(-9999, np.nan)
    ok = m[feats].notna().all(axis=1)
    m['judge_status'] = np.where(ok, 'pending', 'no_aef')
    rid = {'F': 'SC_F', 'O': 'SC_O'}
    vout = pd.DataFrame(dict(
        vid=m['point_id'], rid=[rid.get(b, 'SC_?') for b in m['box']], lon=m['lon'], lat=m['lat'],
        block=m['block'], win=WIN, judge_status=m['judge_status'],
        group=m['group'], box=m['box'], tile='',
        pair=m['pair_id'].fillna('').astype(str),
        stratum=('forest_pair_' + m['slope_cls'].astype(str)).where(m['group'] == 'forest_pair', 'dispute_' + m['change'].astype(str))))
    vout = pd.concat([vout, m[feats]], axis=1)
    vout.to_parquet(os.path.join(MAT, 'V_eval.parquet'), index=False)
    emit('V_eval.parquet：%d 行；AEF 全 64 维有效 %d / %d' % (len(vout), int(ok.sum()), len(vout)))

    # R 点回接池原始 point_id（契约要求：用于回接 AEF），按坐标精确匹配
    pool = pd.read_parquet(POOL_V2, columns=['point_id', 'lon', 'lat'])
    key = {(round(float(lo), 6), round(float(la), 6)): str(pid)
           for pid, lo, la in zip(pool['point_id'], pool['lon'], pool['lat'])}
    rkey = [(round(float(lo), 6), round(float(la), 6)) for lo, la in zip(r['lon'], r['lat'])]
    rpool = [key.get(k, '') for k in rkey]
    emit('R 点回接池 point_id：%d/%d 命中' % (sum(1 for x in rpool if x), len(rpool)))

    rout = pd.DataFrame(dict(
        rid=[rid.get(b, 'SC_?') for b in r['box']], sample_uid=r['uid'], point_id=rpool,
        blind_id=r['point_id'], lon=r['lon'], lat=r['lat'],
        label_before=[CL24.get(int(c), '') if c == c else '' for c in r['cl_pool']],
        label_after='', action='pending', review_status='pending', temporal_status='pending',
        scope='point', rule_version='pending_QC', confidence='', passes=0,
        evidence_type='', evidence_ids='', block=r['block'], win=WIN,
        group=r['group'], box=r['box'], pair=r['pair_id'],
        slope_cls=r['slope_cls'], aspect=r['aspect'], elev=r['elev'], slope=r['slope'],
        c_v1=[CL24.get(int(c), '') if c == c else '' for c in r['c_v1']],
        c_v2=[CL24.get(int(c), '') if c == c else '' for c in r['c_v2']],
        src_label=r['src_label'], dist_km=r['dist_km']))
    rout.to_parquet(os.path.join(MAT, 'R_repair_candidates.parquet'), index=False)
    emit('R_repair_candidates.parquet：%d 行（QC 字段全部 pending，不是修复成品）' % len(rout))

    jdump(os.path.join(MAT, '_READY_STATUS.json'), dict(
        updated=time.strftime('%Y-%m-%d %H:%M:%S'),
        V_eval=dict(path=os.path.join(MAT, 'V_eval.parquet'), rows=len(vout),
                    aef_valid=int(ok.sum()), true_labels='SEALED (audit)；本文件不含任何真值列',
                    forbidden_absent=['cl_true', 'true_label', 't24', 'label_after']),
        R_repair=dict(status='CANDIDATES_ONLY（未经盲判 QC，不得当修复成品）',
                      path=os.path.join(MAT, 'R_repair_candidates.parquet'), rows=len(rout),
                      final_path=os.path.join(MAT, 'R_repair.parquet'),
                      next='blind judge 结论 → 解盲匹配 → 双路一致 → 才写 R_repair.parquet'),
        boxes={k: v['bbox'] for k, v in BOXES.items()}, win=WIN,
        s2_metrics=os.path.join(BLIND, 's2_metrics.csv'),
        blind_face=dict(points=os.path.join(BLIND, 'points_blind.csv'),
                        series=os.path.join(BLIND, 'points_series.csv'))))
    emit('契约件与状态 → materials/')


if __name__ == '__main__':
    main()
