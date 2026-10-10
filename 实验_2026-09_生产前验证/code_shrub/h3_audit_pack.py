# -*- coding: utf-8 -*-
"""h3_audit_pack.py — 全国灌丛样本目视审计包（分层配额预注册 + 抽点 + 索引）

* 输入：data/shrub_nat/cand_nat.parquet（h1+h2 产物：band/tier/zone/prov 已带）
* 规则源（预注册，写死本脚本，改了要在报告里说明）：
        审计配额 = 区带 zone × 层 band/tier 定额——
          P_A（DW+多源一致，正向最可信）30/zone
          P_B（仅 DW 单源，正向存疑）        25/zone
          N_A（模糊带+多源一致，硬负样本）    12/zone
          N_B（模糊带+单源，硬负样本）        13/zone
        区带 = 干旱/半干旱/温带/亚热带/热带/寒温带/青藏高原高山（h2_merge_nat 决策树）
        富灌丛带加抽：四川、云南、内蒙古、新疆、西藏、青海、甘肃、宁夏、陕西、山西、河北、辽宁
              额外按该省 P_A 候选占比补足（slug_cap=60 点）
* 门槛：合计 ≈600–780 点；省内点间距 ≥5 km（防空间自相关）；不与 doc57 四省已判读点相距 <2 km
* 输出：data/shrub_nat/audit_points.csv（point_id/lon/lat/band/tier/zone/prov/… + hint）
        data/shrub_nat/audit_plan.json（配额与实取数）
* 用法：python h3_audit_pack.py [--total 60]
"""
import argparse, json, os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
sys.path.insert(0, os.path.join(WORK, 'code'))
import numpy as np
import pandas as pd

OUTD = os.path.join(WORK, 'data', 'shrub_nat')
PREV = os.path.join(WORK, 'data', 'shrub', 'judged_all.csv')   # doc57 四省已判读点（去重用）
QUOTA = {'P_A': 30, 'P_B': 25, 'N_A': 12, 'N_B': 13}
EXTRA = {'四川', '云南', '内蒙古', '新疆', '西藏', '青海', '甘肃', '宁夏', '陕西', '山西', '河北', '辽宁'}
MIN_KM = 5.0
EXTRA_CAP = 60


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + m, flush=True)


def key(df):
    return df['band'].astype(str) + '_' + df['tier'].astype(str)


def pick(sub, n, rs, used, min_km=MIN_KM):
    """从 sub 里按最小间距抽 n 点。"""
    got, idx = [], list(range(len(sub)))
    rs.shuffle(idx)
    sub = sub.iloc[idx]
    for r in sub.itertuples():
        if any(((r.lon - u[0]) ** 2 * 100 + (r.lat - u[1]) ** 2 * 120) < (min_km / 111.0) ** 2
               for u in used[-800:]):
            continue
        used.append((r.lon, r.lat))
        got.append(r.Index)
        if len(got) >= n:
            break
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--total', type=int, default=60, help='每区带每层的附加配额上限')
    a = ap.parse_args()
    df = pd.read_parquet(os.path.join(OUTD, 'cand_nat.parquet'))
    df['k'] = key(df)
    emit('候选 %d 点；k 分布 %s' % (len(df), df['k'].value_counts().to_dict()))
    # 已判读点（doc57）坐标，做 ≥2 km 回避
    prev = []
    if os.path.exists(PREV):
        try:
            pv = pd.read_csv(PREV, encoding='utf-8-sig')
            prev = list(zip(pv['lon'].to_numpy(float), pv['lat'].to_numpy(float)))
            emit('doc57 已判读点 %d（回避半径 2 km）' % len(prev))
        except Exception as e:
            emit('读已判读点失败 %s' % str(e)[:60])
    used = list(prev)
    rs = np.random.RandomState(20261010)
    sel = {}
    for zone in sorted(df['zone'].unique()):
        zz = df[df['zone'] == zone]
        for k, q in QUOTA.items():
            sub = zz[zz['k'] == k]
            if len(sub) == 0:
                continue
            take = pick(sub, q, rs, used)
            for ix in take:
                sel[ix] = [zone, k, '']
        emit('%s：P_A=%d P_B=%d N_A=%d N_B=%d' % (
            zone, len(zz[zz.k == 'P_A']), len(zz[zz.k == 'P_B']),
            len(zz[zz.k == 'N_A']), len(zz[zz.k == 'N_B'])))
    # 回填省名，用于"富灌丛省加抽"配额
    for ix, v in sel.items():
        v[2] = str(df.at[ix, 'prov'])
    # ---- 富灌丛省加抽 P_A（每省 ≤EXTRA_CAP，已选该省点计入配额）----
    extra = {}
    for prov in EXTRA:
        already = sum(1 for v in sel.values() if v[2] == prov)
        room = EXTRA_CAP - already
        sub = df[(df['prov'] == prov) & (df['k'] == 'P_A')]
        if len(sub) == 0 or room <= 0:
            extra[prov] = max(0, room) if len(sub) else 0
            continue
        take = pick(sub, room, rs, used)
        for ix in take:
            sel[ix] = [prov + '·加抽', 'P_A', prov]
        extra[prov] = len(take)
    emit('加抽：%s' % extra)

    df['point_id'] = ['SHN-%06d' % i for i in range(len(df))]
    out = df.loc[sorted(sel.keys())].copy()
    out['aud_cell'] = [sel[i][0] for i in out.index]
    out['k'] = [sel[i][1] for i in out.index]
    # 复位重排 id（按区带/省聚合，便于分组判读）
    out = out.sort_values(['aud_cell', 'prov', 'k']).reset_index(drop=True)
    out['point_id'] = ['SHP-%s-%04d' % (r.aud_cell.replace('·加抽', 'X')[:6], i)
                       for i, r in enumerate(out.itertuples())]
    fp = os.path.join(OUTD, 'audit_points.csv')
    out.to_csv(fp, index=False, encoding='utf-8-sig')
    plan = dict(quota=QUOTA, extra_prov=extra, extra_cap=EXTRA_CAP, min_km=MIN_KM,
                total=int(len(out)),
                by_zone={z: int(n) for z, n in out.groupby('aud_cell').size().items()},
                by_k={k: int(n) for k, n in out.groupby('k').size().items()},
                by_prov_top10={p: int(n) for p, n in out.groupby('prov').size().sort_values(ascending=False).head(10).items()})
    json.dump(plan, open(os.path.join(OUTD, 'audit_plan.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    emit('→ 审计点 %d（%s）' % (len(out), fp))
    emit('  分区带：%s' % plan['by_zone'])


if __name__ == '__main__':
    main()
