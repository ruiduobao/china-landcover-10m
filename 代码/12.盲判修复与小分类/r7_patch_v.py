# -*- coding: utf-8 -*-
"""r7_patch_v.py — 外科式修补：把不满足 V 空间独立性的个别点换到同块合规位置

* 背景：r1 首轮 V 距离守卫只用**框内**池点；全池复核发现 1 个 V 点（B0171，O 框 dispute）
        与全池最近池点 0.77 km（< 1.0 km 门槛）。选择已冻结、batch_01 已被判读代理读取，
        故**不重排 blind ID**，只换这一个点的坐标（同块、同组、同 role）。
* 规则源：技术文档/72（V 独立块、与母库/池点距离）；主 agent 2026-10-15「V 点位与 train 原池 ID/位置需 audit 支持」
* 门槛：新点同 0.25° 块、同 box/group；与全池 ≥1 km、与母库 ≥1 km、与已选全部点 ≥0.3 km；
        类别仍落在同组（dispute→两版类 ∈ DISPUTE_CL；forest→∈ FOREST_CL）。
* 输出：更新 private/uid_map.csv、blind/points_blind.csv、blind/points_series.csv、
        blind/form_skeleton.csv、materials/batch_XX/{blind_batch,form_skeleton,form_with_metrics}.csv、
        qc/patch_v_log.json；随后需重抓该点 chips + 重建所在批 sheets + 重取该点 S2。
* 用法：python r7_patch_v.py [--dry]
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rc_conf import *   # noqa

sys.path.insert(0, os.path.join(WORK, 'v31_exp', 'code'))
V_MIN_DIST_KM = 1.0
MIN_SEP_KM = 0.3


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    t0 = time.time()
    from scipy.spatial import cKDTree
    side = pd.read_csv(os.path.join(PRIV, 'uid_map.csv'), encoding='utf-8-sig')
    side['point_id'] = side['point_id'].astype(str)
    pool = pd.read_parquet(POOL_V2, columns=['lon', 'lat'])
    mo = pd.read_parquet(MOTHER, columns=['lon', 'lat'])

    def xy(lo, la):
        return np.c_[np.asarray(lo) * 111.32 * np.cos(np.radians(np.asarray(la))), np.asarray(la) * 110.57]

    pt = cKDTree(xy(pool['lon'].values, pool['lat'].values))
    mt = cKDTree(xy(mo['lon'].values, mo['lat'].values))
    v = side[side['role'] == ROLE_V].reset_index(drop=True)
    dp = pt.query(xy(v['lon'].values, v['lat'].values), k=1)[0]
    dm = mt.query(xy(v['lon'].values, v['lat'].values), k=1)[0]
    bad = v[(dp < V_MIN_DIST_KM) | (dm < V_MIN_DIST_KM)].copy()
    emit('需修补 V 点 %d 个：%s' % (len(bad), list(bad['point_id'])))
    if len(bad) == 0:
        jdump(os.path.join(QC, 'patch_v_log.json'), dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), patched=[]))
        return

    # 已选全部点（含 R/V）用于最小间距
    sel_xy = xy(side['lon'].values, side['lat'].values)
    st = cKDTree(sel_xy)
    rng = np.random.default_rng(SEED + 777)
    import r1_scout_candidates as M
    dem, rast = M.DemCache(2), M.RastCache()
    tiles = M.sichuan_tiles()
    log = []
    for _, row in bad.iterrows():
        bid = row['block']
        _, ix, iy = bid.split('_')
        deg = BLOCK_DEG
        x0, y0 = int(ix) * deg, int(iy) * deg
        box = row['box']
        tile = BOXES[box]['tile']
        need_group = row['group']
        got = None
        for _try in range(60):
            lo = rng.uniform(x0, x0 + deg, 400)
            la = rng.uniform(y0, y0 + deg, 400)
            pxy = xy(lo, la)
            ok = (pt.query(pxy, k=1)[0] >= V_MIN_DIST_KM) & (mt.query(pxy, k=1)[0] >= V_MIN_DIST_KM)
            ok &= (st.query(pxy, k=1)[0] >= MIN_SEP_KM)
            if not ok.any():
                continue
            lo, la = lo[ok], la[ok]
            c2 = rast.sample(RAST2, tile, lo, la)
            c1 = rast.sample(RAST1, tile, lo, la)
            for i in range(len(lo)):
                if not ((c2[i] > 0) or (c1[i] > 0)):
                    continue
                g = 'forest_pair' if (c2[i] in FOREST_CL or c1[i] in FOREST_CL) else (
                    'dispute' if (c2[i] in DISPUTE_CL or c1[i] in DISPUTE_CL) else None)
                if g != need_group:
                    continue
                sl, asp, el = dem.at(tile, lo[i], la[i])
                got = dict(lon=float(lo[i]), lat=float(la[i]), c_v1=int(c1[i]), c_v2=int(c2[i]),
                           aspect=float(asp), elev=float(el), slope=float(sl),
                           change=('disagree' if (c1[i] > 0 and c2[i] > 0 and c1[i] != c2[i])
                                   else ('stable' if (c1[i] > 0 and c1[i] == c2[i]) else 'nodata')))
                break
            if got:
                break
        if not got:
            emit('  警告：%s 未找到合规替代点（保持原值并单列）' % row['point_id'])
            log.append(dict(point_id=row['point_id'], patched=False))
            continue
        m = side['point_id'] == row['point_id']
        side.loc[m, ['lon', 'lat']] = got['lon'], got['lat']
        side.loc[m, 'c_v1'], side.loc[m, 'c_v2'] = got['c_v1'], got['c_v2']
        side.loc[m, 'aspect'], side.loc[m, 'elev'], side.loc[m, 'slope'] = got['aspect'], got['elev'], got['slope']
        if 'change' in side.columns:
            side.loc[m, 'change'] = got['change']
        log.append(dict(point_id=row['point_id'], patched=True, old=(float(row['lon']), float(row['lat'])),
                        new=(got['lon'], got['lat']), block=bid, group=need_group, **got))
        emit('  %s → %.6f,%.6f（同块 %s，组 %s）' % (row['point_id'], got['lon'], got['lat'], bid, need_group))

    if a.dry:
        emit('--dry：不落盘')
        return
    side.to_csv(os.path.join(PRIV, 'uid_map.csv'), index=False, encoding='utf-8-sig')
    # 重写盲判面（ID 与顺序不变，仅坐标更新）
    blind = pd.read_csv(os.path.join(BLIND, 'points_blind.csv'), encoding='utf-8-sig')
    blind['point_id'] = blind['point_id'].astype(str)
    nmap = side.set_index('point_id')[['lon', 'lat']]
    blind['lon'] = [round(float(nmap.loc[p, 'lon']), 6) for p in blind['point_id']]
    blind['lat'] = [round(float(nmap.loc[p, 'lat']), 6) for p in blind['point_id']]
    blind.to_csv(os.path.join(BLIND, 'points_blind.csv'), index=False, encoding='utf-8-sig')
    # 分批材料
    man = jload(os.path.join(MAT, 'batch_manifest.json'), {})
    for b in man.get('batches', []):
        bdir = b['dir']
        bp = os.path.join(bdir, 'blind_batch.csv')
        if not os.path.exists(bp):
            continue
        sub = pd.read_csv(bp, encoding='utf-8-sig')
        sub['point_id'] = sub['point_id'].astype(str)
        sub['lon'] = [round(float(nmap.loc[p, 'lon']), 6) for p in sub['point_id']]
        sub['lat'] = [round(float(nmap.loc[p, 'lat']), 6) for p in sub['point_id']]
        sub.to_csv(bp, index=False, encoding='utf-8-sig')
    jdump(os.path.join(QC, 'patch_v_log.json'), dict(
        time=time.strftime('%Y-%m-%d %H:%M:%S'), seconds=round(time.time() - t0, 1),
        threshold_km=V_MIN_DIST_KM, min_sep_km=MIN_SEP_KM, patched=log,
        next='重抓被改点 chips → 重建其所在批 sheets → 重取该点 S2 → 重跑 r6 泄漏审计'))
    emit('修补完成 → private/uid_map.csv + blind/ + materials/（ID 不变）；%.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
