# -*- coding: utf-8 -*-
"""f2_gap_points.py — 补缺口点集（针对用户 5 问的判读专项）

* 输入：F:/lc_work/prod5p_2023/data/train2023_clean.parquet（2023 池，含 src / class_new）
* 规则源（预注册）：三个缺口各建一支点集，全部**从训练池候选里抽**（贴合成品实际输入分布）：
  G1「落叶针叶 vs 落叶阔叶」——池内 7 类（东北大兴安岭/小兴安岭为主）＋ 5 类同域对照，各 30 点
                              用途：回答"落叶组内针/阔能否用地理范围 + 特征区分"
  G2「湖河滩地 vs 裸地（水体邻域）」——三江评估区内，池内 16 类 30 点 ＋ 邻近 22 类 30 点
                              用途：检验"湖河滩地仅在水体附近"这条生态规则
  G3「木本沼泽 vs 森林/草地（水分）」——三江评估区内，池内 14 类 30 点 ＋ 邻域 11/5 各 15 点
                              用途：检验"木本沼泽须水分极丰富"
  抽样规则：每支内按 (lon,lat) 空间去重 ≥2 km；固定种子 20261010；输出自带 fcs10_code/src 便于分层
* 门槛：每支点数 ≤90；点必须落在池内（非凭空造点）
* 输出：data/m3/gap_points.csv（point_id,tag,lon,lat,cls_pool,fcs10_code,src）
        data/m3/gap_plan.json
* 用法：python f2_gap_points.py
"""
import collections
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
sys.path.insert(0, os.path.join(WORK, 'code'))
import numpy as np
import pandas as pd
import v31_common as VC

POOL = r'F:/lc_work/prod5p_2023/data/train2023_clean.parquet'
OUT = os.path.join(WORK, 'data', 'm3', 'gap_points.csv')
PLAN = os.path.join(WORK, 'data', 'm3', 'gap_plan.json')
SEED = 20261010
MIN_KM = 2.0
NAMES = VC.V31_NAMES()
SHRUB_SRC = 'glc_fcs10_2023_shrub'


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def pick(df, n, rng, used, box=None, extra=None, tag=''):
    sub = df
    if box:
        x0, y0, x1, y1 = box
        sub = sub[(sub.lon >= x0) & (sub.lon <= x1) & (sub.lat >= y0) & (sub.lat <= y1)]
    if extra is not None:
        sub = sub[extra(sub)]
    if len(sub) == 0:
        emit('  %s 候选为空' % tag)
        return []
    idx = rng.permutation(len(sub))
    out = []
    for i in idx:
        if len(out) >= n:
            break
        r = sub.iloc[i]
        lon, lat = float(r.lon), float(r.lat)
        if any(((lon - a) ** 2 * 100 + (lat - b) ** 2 * 120) < (MIN_KM / 111.0) ** 2 for a, b in used):
            continue
        used.append((lon, lat))
        out.append((lon, lat, int(r.cls), str(r.fcs10_code), str(r.src)))
    emit('  %s 抽 %d/%d' % (tag, len(out), len(sub)))
    return out


def main():
    rng = np.random.RandomState(SEED)
    cols = ['lon', 'lat', 'class_new', 'src', 'fcs10_code'] if 'fcs10_code' in pd.read_parquet(
        POOL, columns=['class_new']).columns else ['lon', 'lat', 'class_new', 'src']
    df = pd.read_parquet(POOL, columns=cols)
    df['cls'] = VC.to_v31(df['class_new'].to_numpy(int))
    if 'fcs10_code' not in df.columns:
        df['fcs10_code'] = ''
    emit('池 %d 点' % len(df))
    used = []
    rows = []

    # ---- G1 落叶针叶 vs 落叶阔叶（东北为主） ----
    NE = (118.0, 42.0, 135.0, 54.0)
    for want, nm in ((7, '7落叶针叶'), (5, '5落叶阔叶'), (8, '8针阔混交')):
        pts = pick(df[df.cls == want], 30, rng, used, box=NE, tag=nm)
        rows += [dict(tag='G1', want=nm, lon=a, lat=b, cls_pool=c, fcs10_code=d, src=s) for a, b, c, d, s in pts]

    # ---- G2 湖河滩地（16） vs 裸地（22）：三江评估区 ----
    SJ = (129.5, 42.3, 135.0, 49.5)
    for want, nm in ((16, '16湖河滩地'), (22, '22裸地'), (23, '23水体')):
        pts = pick(df[df.cls == want], 30, rng, used, box=SJ, tag=nm)
        rows += [dict(tag='G2', want=nm, lon=a, lat=b, cls_pool=c, fcs10_code=d, src=s) for a, b, c, d, s in pts]

    # ---- G3 木本沼泽（14） vs 草地（11）/森林（5,6） ----
    for want, nm in ((14, '14木本沼泽'), (15, '15草本沼泽'), (11, '11草地'), (5, '5落叶阔叶林'), (6, '6常绿针叶林')):
        n = 30 if want in (14, 15) else 15
        pts = pick(df[df.cls == want], n, rng, used, box=SJ, tag=nm)
        rows += [dict(tag='G3', want=nm, lon=a, lat=b, cls_pool=c, fcs10_code=d, src=s) for a, b, c, d, s in pts]

    out = pd.DataFrame(rows)
    out.insert(0, 'point_id', ['GAP-%s-%04d' % (r.tag, i) for i, r in enumerate(out.itertuples())])
    out.to_csv(OUT, index=False, encoding='utf-8-sig')
    plan = {'n': int(len(out)), 'seed': SEED, 'min_km': MIN_KM,
            'by_tag': {t: dict(collections.Counter(g['want'])) for t, g in out.groupby('tag')}}
    VC.jsave(plan, PLAN)
    emit('缺口点集 %d → %s' % (len(out), OUT))
    for t, v in plan['by_tag'].items():
        emit('  %s: %s' % (t, v))


if __name__ == '__main__':
    main()
