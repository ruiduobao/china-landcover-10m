# -*- coding: utf-8 -*-
"""n16_expA2_points.py — 实验 A 第二轮点表：按 FCS10 森林层分层抽样（针/阔/混交候选）

* 背景（为什么这么做）：第一轮用 M3-A wc=10 点判读得到 45 个森林点，林型分布退化（39 常绿阔叶/4 常绿针叶/2 落叶阔叶，
  针叶与落叶类 <15/类）→ T1/T2 均无法裁决。第二轮改为**按 FCS10-2023 训练池的森林层码分层抽样**：
  针叶 71/72/81/82；落叶阔叶 61/62；常绿阔叶 51/52；混交 91/92 —— FCS10 码**仅用于选点分层**，
  真值仍由 z18 影像判读产出（纪律：产品标签不得当真值）。
* 输入：F:/lc_work/年度子集_含稀有类/r7_train_2023.parquet（lon/lat/class_new=GLC 30 类码）
* 规则源：区域框取 M3-A 同区外扩 0.5°（w1 东北 / w2 秦岭 / w3 南方丘陵）；每层配额（近似）：
  针叶 80（w1:40/w2:20/w3:20）、落叶阔叶 60（30/20/10）、常绿阔叶 40（5/15/20）、混交 20（10/10/0）；
  贪心 ≥3 km 点间距；随机种子固定 20261009。
* 门槛：各层不足时按可得数量取；输出含 stratum 与原 row_id 便于追溯。
* 输出：data/m3/expA2_points.csv + data/m3/expA2_plan.json
* 用法：python n16_expA2_points.py
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import v31_common as VC

M3D = r'F:/lc_work/v31_exp/data/m3'
POOL = r'F:/lc_work/年度子集_含稀有类/r7_train_2023.parquet'
SEED = 20261009
MIN_KM = 3.0
NEEDLE = [71, 72, 81, 82]
DECID_B = [61, 62]
EVER_B = [51, 52]
MIXED = [91, 92]
QUOTA = {  # stratum → {region: n}
    'needle': {'w1': 40, 'w2': 20, 'w3': 20},
    'decid_broad': {'w1': 30, 'w2': 20, 'w3': 10},
    'ever_broad': {'w1': 5, 'w2': 15, 'w3': 20},
    'mixed': {'w1': 10, 'w2': 10, 'w3': 0},
}


def greedy_spaced(pts, n, min_km, rng):
    """贪心抽样：随机序 + 与已选最小距离 ≥ min_km。"""
    if len(pts) == 0:
        return pts
    idx = rng.permutation(len(pts))
    sel, sel_xy = [], []
    R = min_km / 111.0
    for i in idx:
        lo, la = float(pts.lon.iloc[i]), float(pts.lat.iloc[i])
        if sel_xy:
            d = np.hypot((np.array(sel_xy)[:, 0] - lo) * np.cos(np.radians(la)), np.array(sel_xy)[:, 1] - la)
            if d.min() < R:
                continue
        sel.append(i)
        sel_xy.append((lo, la))
        if len(sel) >= n:
            break
    return pts.iloc[sel]


def main():
    rng = np.random.default_rng(SEED)
    a = pd.read_csv(os.path.join(M3D, 'm3a_points.csv'))
    boxes = a.groupby('region').agg(lon0=('lon', 'min'), lon1=('lon', 'max'),
                                    lat0=('lat', 'min'), lat1=('lat', 'max'))
    d = pd.read_parquet(POOL, columns=['row_id', 'lon', 'lat', 'class_new', 'src'])
    rows, plan = [], {}
    for stratum, codes in (('needle', NEEDLE), ('decid_broad', DECID_B),
                           ('ever_broad', EVER_B), ('mixed', MIXED)):
        s = d[d.class_new.isin(codes)]
        for region, n in QUOTA[stratum].items():
            b = boxes.loc[region]
            sub = s[(s.lon >= b.lon0 - 0.5) & (s.lon <= b.lon1 + 0.5) &
                    (s.lat >= b.lat0 - 0.5) & (s.lat <= b.lat1 + 0.5)]
            got = greedy_spaced(sub, n, MIN_KM, rng)
            plan['%s/%s' % (stratum, region)] = dict(want=n, avail=int(len(sub)), got=int(len(got)))
            for r in got.itertuples():
                rows.append({'point_id': 'EXPA2-%03d' % len(rows), 'lon': float(r.lon), 'lat': float(r.lat),
                             'region': region, 'stratum': stratum, 'fcs10_code': int(r.class_new),
                             'row_id': int(r.row_id)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(M3D, 'expA2_points.csv'), index=False, encoding='utf-8-sig')
    VC.jsave({'seed': SEED, 'min_km': MIN_KM, 'quota': QUOTA, 'plan': plan,
              'n_total': int(len(df)), 'by_stratum': df.stratum.value_counts().to_dict(),
              'by_region': df.region.value_counts().to_dict(),
              'note': 'FCS10 码仅用于选点分层；真值由 z18 影像判读出（见 judge_rubric_A2.md）'},
             os.path.join(M3D, 'expA2_plan.json'))
    print(json.dumps(plan, ensure_ascii=False, indent=1))
    print('总点数 %d；分层 %s；区域 %s' % (len(df), df.stratum.value_counts().to_dict(),
                                     df.region.value_counts().to_dict()))
    print('→', os.path.join(M3D, 'expA2_points.csv'))


if __name__ == '__main__':
    main()
