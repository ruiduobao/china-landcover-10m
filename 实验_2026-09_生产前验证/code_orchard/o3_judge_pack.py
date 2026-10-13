# -*- coding: utf-8 -*-
"""o3_judge_pack.py — 乔灌园地：判读包抽样（分层，各框精度量测 + 判别器训练集）

* 输入：data/orchard/mk_orchard_points.csv（母库框）
        data/orchard/cand_fcs10.parquet（FCS10-12 框）
        data/orchard/cand_vec.parquet（OSM/SDPT/GBIF/人工林框）
* 规则源：分层配额（预注册）：
    mk_orchard 100（现有样本质量）、fcs10_12 160（主框）、osm_orchard 80、osm_vineyard 20、
    sdpt_treecrop 40（不足则全取）、gbif_orchard_species 40、plantedforest_neg 40（负样本 sanity）
  每框内 ≥2 km 贪心去重（防空间自相关）；seed 20261012
* 门槛：总 ~480 点；输出点序即判读点序
* 输出：data/orchard/judge_points.csv（point_id,lon,lat,frame）+ data/orchard/judge_plan.json
* 用法：python o3_judge_pack.py
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import v31_common as VC

WORK = r'F:/lc_work/v31_exp'
OUTD = os.path.join(WORK, 'data', 'orchard')
QUOTA = {'mk_orchard': 100, 'fcs10_12': 160, 'osm_orchard': 80, 'osm_vineyard': 20,
         'sdpt_treecrop': 40, 'gbif_orchard_species': 40, 'sdpt_plantedforest_neg': 40}
MIN_KM = 2.0
SEED = 20261012


def greedy(df, n, min_km, rng, fixed):
    if len(df) == 0 or n <= 0:
        return df
    R = min_km / 111.0
    idx = rng.permutation(len(df))
    sel, pts = [], list(fixed)
    arr = df[['lon', 'lat']].to_numpy()
    for i in idx:
        lo, la = arr[i]
        if pts:
            a = np.asarray(pts)
            d = np.hypot((a[:, 0] - lo) * np.cos(np.radians(la)), a[:, 1] - la)
            if d.min() < R:
                continue
        sel.append(i)
        pts.append((lo, la))
        if len(sel) >= n:
            break
    return df.iloc[sel]


def main():
    rng = np.random.default_rng(SEED)
    frames = {}
    m = pd.read_csv(os.path.join(OUTD, 'mk_orchard_points.csv'), encoding='utf-8-sig')
    frames['mk_orchard'] = m.rename(columns={'row_id': 'src_id'})[['lon', 'lat', 'src_id']]
    for f in ['cand_fcs10.parquet', 'cand_vec.parquet']:
        p = os.path.join(OUTD, f)
        if os.path.exists(p):
            d = pd.read_parquet(p)
            if 'src_id' not in d.columns:
                d['src_id'] = d.get('tile', '')
            for fr, g in d.groupby('frame'):
                frames[fr] = g[['lon', 'lat', 'src_id']].reset_index(drop=True)
    rows, plan = [], {}
    fixed = []
    for fr, quota in QUOTA.items():
        g = frames.get(fr, pd.DataFrame(columns=['lon', 'lat', 'src_id']))
        got = greedy(g.reset_index(drop=True), quota, MIN_KM, rng, fixed)
        plan[fr] = dict(avail=int(len(g)), want=quota, got=int(len(got)))
        for r in got.itertuples():
            fixed.append((float(r.lon), float(r.lat)))
            rows.append({'point_id': 'OJD-%03d' % len(rows), 'lon': float(r.lon), 'lat': float(r.lat),
                         'frame': fr, 'src_id': str(getattr(r, 'src_id', ''))})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTD, 'judge_points.csv'), index=False, encoding='utf-8-sig')
    VC.jsave({'seed': SEED, 'min_km': MIN_KM, 'quota': QUOTA, 'plan': plan,
              'n_total': int(len(df)), 'by_frame': df.frame.value_counts().to_dict()},
             os.path.join(OUTD, 'judge_plan.json'))
    print(json.dumps(plan, ensure_ascii=False, indent=1))
    print('判读点 %d → %s' % (len(df), os.path.join(OUTD, 'judge_points.csv')))


if __name__ == '__main__':
    main()
