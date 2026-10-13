# -*- coding: utf-8 -*-
"""o11_gem_judge_pack.py — GEM 候选：国界过滤 + 省归属 + 分层判读包（240 点）

* 输入：data/orchard/gem_points.parquet（o9 全国候选）
* 规则源：中国国界（unary_union + contains_xy）；按宏观区带分层（预注册配额）：
    海南热带（橡胶/槟榔）、云南（橡胶/茶）、华南（桂粤闽）、西南（川渝黔）、华东华中、北方；
    每层 ≥2 km 贪心间距；seed 20261012
* 门槛：总 ~240 点；输出即判读点序
* 输出：data/orchard/gem/judge_points_gem.csv + judge_plan_gem.json
* 用法：python o11_gem_judge_pack.py
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import v31_common as VC
from shapely.geometry import shape
from shapely import make_valid, contains_xy

WORK = r'F:/lc_work/v31_exp'
ORD = os.path.join(WORK, 'data', 'orchard')
BND = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/边界/china_100000_full.json'
QUOTA = {'海南': 50, '云南': 50, '华南(桂粤闽台)': 50, '西南(川渝黔湘鄂)': 40, '华东华中': 30, '北方其他': 20}
MIN_KM = 2.0
SEED = 20261012


def zone_of(lon, lat):
    if 108 <= lon <= 111.5 and 18 <= lat <= 20.5:
        return '海南'
    if 97 <= lon < 106.5 and 21 <= lat <= 29.5:
        return '云南'
    if 104 <= lon <= 122 and 20.5 <= lat < 26.5:
        return '华南(桂粤闽台)'
    if 103 <= lon <= 114 and 26.5 <= lat <= 33:
        return '西南(川渝黔湘鄂)'
    if 114 < lon <= 123 and 26.5 <= lat <= 36:
        return '华东华中'
    return '北方其他'


def main():
    rng = np.random.default_rng(SEED)
    d = pd.read_parquet(os.path.join(ORD, 'gem_points.parquet'))
    gj = json.load(open(BND, encoding='utf-8'))
    geoms = []
    for ft in gj['features']:
        g = shape(ft['geometry'])
        if not g.is_valid:
            g = make_valid(g)
        geoms.append(g)
    from shapely.ops import unary_union
    china = unary_union(geoms)
    inb = d[(d.lon >= 73) & (d.lon <= 136) & (d.lat >= 18) & (d.lat <= 54)].reset_index(drop=True)
    keep = contains_xy(china, inb.lon.to_numpy(), inb.lat.to_numpy())
    cn = inb[keep].reset_index(drop=True)
    cn['zone'] = [zone_of(lo, la) for lo, la in zip(cn.lon, cn.lat)]
    print('候选 %d → 中国境内 %d；分区:' % (len(d), len(cn)))
    print(cn.zone.value_counts().to_dict())
    rows, plan = [], {}
    fixed = []
    for z, quota in QUOTA.items():
        g = cn[cn.zone == z].reset_index(drop=True)
        R = MIN_KM / 111.0
        idx = rng.permutation(len(g))
        sel = []
        for i in idx:
            lo, la = g.lon.iloc[i], g.lat.iloc[i]
            if fixed:
                arr = np.asarray(fixed)
                dd = np.hypot((arr[:, 0] - lo) * np.cos(np.radians(la)), arr[:, 1] - la)
                if dd.min() < R:
                    continue
            sel.append(i); fixed.append((lo, la))
            if len(sel) >= quota:
                break
        plan[z] = dict(avail=int(len(g)), want=quota, got=len(sel))
        for i in sel:
            rows.append({'point_id': 'GJD-%03d' % len(rows), 'lon': float(g.lon.iloc[i]),
                         'lat': float(g.lat.iloc[i]), 'zone': z})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ORD, 'gem', 'judge_points_gem.csv'), index=False, encoding='utf-8-sig')
    VC.jsave({'n_cn': int(len(cn)), 'plan': plan, 'n_total': int(len(df)), 'seed': SEED, 'min_km': MIN_KM},
             os.path.join(ORD, 'gem', 'judge_plan_gem.json'))
    print(json.dumps(plan, ensure_ascii=False, indent=1))
    print('判读点 %d → gem/judge_points_gem.csv' % len(df))


if __name__ == '__main__':
    main()
