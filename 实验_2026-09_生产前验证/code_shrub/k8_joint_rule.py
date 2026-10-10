# -*- coding: utf-8 -*-
"""k8_joint_rule.py — 量测「FCS10 层 × OSM 邻近距离」联合规则在判读点上的精度

* 输入：judged_feats.parquet（1,803 带目视标签点）+ osm_scrub_nat.json（全国 OSM scrub/heath 要素）
        + data/shrub/judged_all.csv（doc57 分层：标注哪些点在 FCS10 框内）
* 规则源（预注册）：
        OSM 距离 d = 点到最近 OSM scrub/heath 要素的球面距离（km）；按 d ∈ {<0.5,<1,<2,<5,<10} 分档
        联合规则 = （在 FCS10 框内）×（d < 阈值）
* 门槛：只量测 OSM 覆盖到的判读点（当前 OSM 抓到 lon≤101°E，须在报告里注明覆盖偏差）；
        每格 n/灌丛/precision/Wilson CI
* 输出：data/shrub_nat/joint_rule.json + data/shrub_nat/joint_rule.md
* 用法：python k8_joint_rule.py
"""
import json, math, os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
sys.path.insert(0, os.path.join(WORK, 'code'))
import numpy as np
import pandas as pd

OUTD = os.path.join(WORK, 'data', 'shrub_nat')


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + m, flush=True)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def main():
    j = pd.read_parquet(os.path.join(OUTD, 'judged_feats.parquet'))
    d57 = pd.read_csv(os.path.join(WORK, 'data', 'shrub', 'judged_all.csv'), encoding='utf-8-sig')
    osm = json.load(open(os.path.join(OUTD, 'osm_scrub_nat.json'), encoding='utf-8'))
    recs = osm if isinstance(osm, list) else osm.get('features', [])
    op = np.array([[r['lon'], r['lat']] for r in recs])
    emit('OSM 要素 %d，范围 lon %.1f-%.1f' % (len(op), op[:, 0].min(), op[:, 0].max()))
    lon = j['lon'].to_numpy(float); lat = j['lat'].to_numpy(float)
    y = (j['is_shrub'] == '是').to_numpy()
    ok = j['is_shrub'].isin(['是', '否']).to_numpy()
    # 剔除 OSM 未覆盖区的判读点（用 OSM bbox 判定，非格级）
    cov = (lon >= op[:, 0].min()) & (lon <= op[:, 0].max()) & (lat >= op[:, 1].min()) & (lat <= op[:, 1].max())
    k57 = {(round(a, 6), round(b, 6)): s for a, b, s in
           zip(d57.lon.to_numpy(float), d57.lat.to_numpy(float), d57['stratum'])}
    in_fcs10 = np.array([k57.get((round(a, 6), round(b, 6)), '') == 'FCS10' for a, b in zip(lon, lat)])
    emit('判读点 %d；OSM 覆盖内 %d；其中 doc57 FCS10 框内 %d' % (len(j), cov.sum(), in_fcs10.sum()))
    # 最近 OSM 距离（bins 优化：按 1° 格建索引）
    ob = np.floor(op[:, 1] * 1).astype(int) * 1000 + np.floor(op[:, 0] * 1).astype(int)
    from collections import defaultdict
    grid = defaultdict(list)
    for i, b in enumerate(ob):
        grid[b].append(i)
    dist = np.full(len(j), np.nan)
    for i in range(len(j)):
        gi = math.floor(lat[i] * 1) * 1000 + math.floor(lon[i] * 1)
        best = None
        for dla in (-1, 0, 1):
            for dlo in (-1, 0, 1):
                for k in grid.get(gi + dla * 1000 + dlo, ()):  # 纬度带也要扩
                    pass
        # 简化为 3°×3° 搜索窗（足够覆盖 <10 km 距离）
        for dla in (-1, 0, 1, 2):
            for dlo in (-1, 0, 1):
                for bb in grid.get(gi + dla * 1000 + dlo, ()):
                    dx = (op[bb, 0] - lon[i]) * 96.0 * math.cos(math.radians(lat[i]))
                    dy = (op[bb, 1] - lat[i]) * 111.0
                    d = math.hypot(dx, dy)
                    if best is None or d < best:
                        best = d
        for dlb in (-1, 0, 1):
            for bb in grid.get(gi + dlb, ()):  # 同纬度带跨经度格
                pass
        if best is not None:
            dist[i] = best
    j['osm_km'] = dist
    j['in_fcs10'] = in_fcs10
    m = ok & cov & np.isfinite(dist)
    sub = j[m]
    emit('可用测点 %d（灌丛 %d）' % (len(sub), y[m].sum()))
    res = dict(n=int(len(sub)), n_pos=int(y[m].sum()), osm_cover_note='OSM 抓取进行中，仅 lon≤101°E 有覆盖',
               by_dist={}, joint={})
    lines = ['# FCS10 × OSM 邻近联合规则 %s' % time.strftime('%Y-%m-%d %H:%M'), '',
             '- 可用测点 %d（灌丛 %d）；**OSM 抓取仍在进行，当前只覆盖 lon≤101°E，结论对东部不适用**' % (
                 len(sub), y[m].sum()),
             '- FCS10 框 = doc57 的 FCS10 分层点（n=%d）' % int(in_fcs10.sum()),
             '', '## 单独 OSM 距离分档', '', '| 距离 | n | 灌丛 | precision | 95%CI |', '|---|---|---|---|---|']
    for th in (0.5, 1, 2, 5, 10, 25):
        mm = (sub['osm_km'] < th).to_numpy()
        n, k = int(mm.sum()), int(y[m][mm].sum())
        prec = k / n if n else None
        lo, hi = wilson(k, n)
        res['by_dist'][str(th)] = dict(n=n, shrub=k, precision=(round(prec, 4) if prec is not None else None),
                                       wilson=[lo, hi])
        lines.append('| <%.1f km | %d | %d | %s | [%.2f,%.2f] |' % (
            th, n, k, ('%.3f' % prec) if prec is not None else '—', lo, hi))
    lines += ['', '## 联合规则：FCS10 框内 且 OSM < 阈值', '',
              '| 规则 | n | 灌丛 | precision | 95%CI |', '|---|---|---|---|---|']
    f = sub['in_fcs10'].to_numpy()
    for th in (0.5, 1, 2, 5, 10, 25):
        mm = f & (sub['osm_km'] < th).to_numpy()
        n, k = int(mm.sum()), int(y[m][mm].sum())
        prec = k / n if n else None
        lo, hi = wilson(k, n)
        res['joint']['fcs10&osm<%.1f' % th] = dict(n=n, shrub=k,
                                                  precision=(round(prec, 4) if prec is not None else None),
                                                  wilson=[lo, hi])
        lines.append('| FCS10 且 OSM<%.1f km | %d | %d | %s | [%.2f,%.2f] |' % (
            th, n, k, ('%.3f' % prec) if prec is not None else '—', lo, hi))
        emit('FCS10 & OSM<%.1f：n=%d 灌=%d precision=%s' % (th, n, k, ('%.3f' % prec) if prec is not None else '—'))
    j.to_parquet(os.path.join(OUTD, 'judged_feats_osm.parquet'), index=False)
    json.dump(res, open(os.path.join(OUTD, 'joint_rule.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    open(os.path.join(OUTD, 'joint_rule.md'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    emit('→ joint_rule.json / joint_rule.md')


if __name__ == '__main__':
    main()
