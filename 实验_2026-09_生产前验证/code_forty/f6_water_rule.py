# -*- coding: utf-8 -*-
"""f6_water_rule.py — 水体邻域距离场 + 规则验证（用户问 1/2：湖河滩地/木本沼泽须在水体附近）

* 输入：data/eco_gate/_water/jrc_occ_chn_1km.tif（JRC 水出现频率）
        data/m3/gap_points.csv + gap_imagery/judge_*.csv（G2/G3 判读真值）
        F:/lc_work/prod5p_2023/data/train2023_clean.parquet（池，用于规则的全量效果评估）
* 规则源（预注册）：
        · 距离场：对 occurrence≥50（常年水）与 ≥1（曾出现水）分别做 EDT；按纬度分带计算（sampling 校正）
        · 待验假设 H1：真「湖河滩地(16)」距水 ≤ D；假(判否)者显著更远 → 用 D 剪池
        · 待验假设 H2：真「木本沼泽(14)」距水 ≤ D 且水深/丰度更高 → 多数判否说明该层不可救
        · D 取值扫描 {1, 2, 3, 5, 10} km，报告判读真值上的**召回/误杀**权衡
* 门槛：规则只在判读样本 ≥10 有效点时给结论；输出保留阈值扫描表
* 输出：data/eco_gate/_water/dist_water_1km.tif（距离场 km，uint8×10 量化）
        data/m3/water_rule_scan.json + .csv
* 用法：python f6_water_rule.py
"""
import collections
import csv
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
sys.path.insert(0, os.path.join(WORK, 'code'))
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

WD = os.path.join(WORK, 'data', 'eco_gate', '_water')
M3 = os.path.join(WORK, 'data', 'm3')
SCAN_J = os.path.join(M3, 'water_rule_scan.json')
SCAN_C = os.path.join(M3, 'water_rule_scan.csv')
POOL = r'F:/lc_work/prod5p_2023/data/train2023_clean.parquet'
BANDS = [(18.0, 24.0), (24.0, 30.0), (30.0, 36.0), (36.0, 42.0), (42.0, 48.0), (48.0, 54.0)]


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def dist_field(occ, thr, path):
    """按纬度分带计算到最近水体像元的距离（km）；输出 uint8（0.5 km 量化，上限 127.5 km）。"""
    d = np.full(occ.shape, 127.5, 'float32')
    for y0, y1 in BANDS:
        r0 = max(0, int(round((54.0 - y1) / 0.01)))
        r1 = min(occ.shape[0], int(round((54.0 - y0) / 0.01)))
        if r1 <= r0:
            continue
        sub = occ[r0:r1]
        latc = 0.5 * (y0 + y1)
        dy = 0.01 * 110.574
        dx = 0.01 * 111.32 * np.cos(np.deg2rad(latc))
        water = sub >= thr
        if not water.any():
            continue
        # 距离到最近水体像元：EDT 作用于"非水"掩膜
        dist = ndimage.distance_transform_edt(~water, sampling=(dy, dx))
        d[r0:r1] = np.minimum(d[r0:r1], dist.astype('float32'))
        emit('    带 %d-%d°N 水像元 %.3f%%  距离中位 %.1f km' % (
            y0, y1, 100 * float(water.mean()), float(np.median(dist[~water])) if (~water).any() else -1))
    with rasterio.open(os.path.join(WD, 'jrc_occ_chn_1km.tif')) as src:
        prof = src.profile.copy()
    prof.update(dtype='uint8', count=1, compress='deflate', tiled=True, nodata=255)
    with rasterio.open(path, 'w', **prof) as dst:
        dst.write(np.clip(np.round(d / 0.5), 0, 255).astype('uint8'), 1)
    emit('  距离场 → %s' % path)
    return d


def sample_dist(d, tfm, lon, lat):
    col = (lon - tfm.c) / tfm.a
    row = (tfm.f - lat) / abs(tfm.e)
    ci = np.clip(np.round(col).astype('int64'), 0, d.shape[1] - 1)
    ri = np.clip(np.round(row).astype('int64'), 0, d.shape[0] - 1)
    return d[ri, ci]


def main():
    with rasterio.open(os.path.join(WD, 'jrc_occ_chn_1km.tif')) as ds:
        occ = ds.read(1)
        tfm = ds.transform
    emit('JRC 读取 %s' % (occ.shape,))
    d50 = dist_field(occ, 50, os.path.join(WD, 'dist_water_perm_1km.tif'))
    d1 = dist_field(occ, 1, os.path.join(WD, 'dist_water_ever_1km.tif'))

    # ---- 判读真值上的规则验证 ----
    gp = pd.read_csv(os.path.join(M3, 'gap_points.csv'), encoding='utf-8-sig')
    js = []
    for i in range(1, 6):
        fp = os.path.join(M3, 'gap_imagery', 'judge_%d.csv' % i)
        if os.path.exists(fp):
            js.append(pd.read_csv(fp, encoding='utf-8-sig'))
    G = gp.merge(pd.concat(js, ignore_index=True), on='point_id', how='inner')
    G['d_perm'] = sample_dist(d50, tfm, G.lon.to_numpy(), G.lat.to_numpy())
    G['d_ever'] = sample_dist(d1, tfm, G.lon.to_numpy(), G.lat.to_numpy())
    # 参考集（exp 系列）也有 湖河滩地/木本沼泽 真值
    R = pd.read_csv(os.path.join(M3, 'refset_all.csv'), encoding='utf-8-sig')
    R['d_perm'] = sample_dist(d50, tfm, R.lon.to_numpy(), R.lat.to_numpy())
    R['d_ever'] = sample_dist(d1, tfm, R.lon.to_numpy(), R.lat.to_numpy())
    R['Q1'] = '是'   # exp 集的 truth_code 即影像真值

    rows, res = [], {}
    for tag, code, name in (('G2|16', 16, '湖河滩地'), ('G3|14', 14, '木本沼泽'), ('G3|15', 15, '草本沼泽')):
        a = G[G.want.str.startswith('%d' % code)]
        b = R[R.truth_code == code].copy()
        b['want'] = '%d%s' % (code, name)
        m = pd.concat([a[['point_id', 'want', 'Q1', 'd_perm', 'd_ever']],
                       b[['point_id', 'want', 'Q1', 'd_perm', 'd_ever']]], ignore_index=True)
        v = m[m.Q1 != '无法判读']
        yes, no = v[v.Q1 == '是'], v[v.Q1 == '否']
        res['%s_%s' % (tag, name)] = dict(
            n=len(m), judged=len(v), yes=len(yes), no=len(no),
            d_perm_yes_med=round(float(yes.d_perm.median()), 2) if len(yes) else None,
            d_perm_no_med=round(float(no.d_perm.median()), 2) if len(no) else None,
            d_ever_yes_med=round(float(yes.d_ever.median()), 2) if len(yes) else None,
            d_ever_no_med=round(float(no.d_ever.median()), 2) if len(no) else None)
        r = res['%s_%s' % (tag, name)]
        emit('%s：真(是) n=%d 常水距中位 %s km；假(否) n=%d 常水距中位 %s km' % (
            name, r['yes'], r['d_perm_yes_med'], r['no'], r['d_perm_no_med']))
        for D in (0.5, 1, 2, 3, 5, 10, 20):
            keep_yes = int((yes.d_perm <= D).sum())
            keep_no = int((no.d_perm <= D).sum())
            rows.append(dict(rule='%s 常水距<=%s km' % (name, D), D_km=D,
                             n_yes=r['yes'], kept_yes=keep_yes, killed_yes=r['yes'] - keep_yes,
                             n_no=r['no'], kept_no=keep_no, killed_no=r['no'] - keep_no,
                             recall=round(keep_yes / max(1, r['yes']), 3),
                             false_kill=round((r['yes'] - keep_yes) / max(1, r['yes']), 3),
                             no_removed=round((r['no'] - keep_no) / max(1, r['no']), 3)))
    VC.jsave(res, SCAN_J)
    with open(SCAN_C, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    emit('阈值扫描 → %s' % SCAN_C)
    for r in rows:
        if r['n_yes'] >= 5:
            emit('  %-22s 召回 %.2f 误杀 %.2f | 假点清除率 %.2f' % (
                r['rule'], r['recall'], r['false_kill'], r['no_removed']))
    # ---- 池侧规模 ----
    pool = pd.read_parquet(POOL, columns=['lon', 'lat', 'class_new'])
    import v31_common as VC2
    pool['cls'] = VC2.to_v31(pool['class_new'].to_numpy(int))
    pool['d_perm'] = sample_dist(d50, tfm, pool.lon.to_numpy(), pool.lat.to_numpy())
    for code, nm in ((16, '湖河滩地'), (14, '木本沼泽'), (15, '草本沼泽'), (17, '盐渍湿地')):
        s = pool[pool.cls == code]
        if len(s) == 0:
            continue
        res.setdefault('pool', {})[nm] = dict(
            n=int(len(s)), d_perm_med=round(float(s.d_perm.median()), 2),
            pct_within_1km=round(100 * float((s.d_perm <= 1).mean()), 2),
            pct_within_5km=round(float(100 * (s.d_perm <= 5).mean()), 2),
            pct_far_over20km=round(float(100 * (s.d_perm > 20).mean()), 2))
        emit('  池 %-6s n=%5d 常水距中位 %.1f km；≤1km %.1f%%；>20km %.1f%%' % (
            nm, len(s), s.d_perm.median(), 100 * (s.d_perm <= 1).mean(), 100 * (s.d_perm > 20).mean()))
    VC.jsave(res, SCAN_J)


if __name__ == '__main__':
    main()
