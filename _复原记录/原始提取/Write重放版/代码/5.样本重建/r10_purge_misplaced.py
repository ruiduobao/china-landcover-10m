# -*- coding: utf-8 -*-
"""
r10_purge_misplaced.py — 错位样本剔除（省级行政区白名单硬规则，s0_conf.ECO_PROVINCE）
* 背景：GLC_FCS10 2023 在长三角/杭州湾沿岸把养殖塘/潮间带错标为 140（地衣苔藓），
  在长江中下游淡水湖区错标为 183（盐渍湿地）；s0_conf 纬度带规则过滤不住。
* 动作（精度优先，不补量）：
  1. 对 r7_train（基线）+ r7_rare_samples（本轮补样）逐点判定省级白名单违规；
  2. 产出剔除清单 数据/本地处理/全国清洗训练/错位样本剔除清单.parquet（留档审计）；
  3. 就地应用：
     - 年度子集/r7_train_{y}.parquet            剔除（8 个）
     - 年度子集_含稀有类/r7_train_{y}.parquet   剔除（8 个）
     - 稀有类补样/r7_rare_samples.parquet       剔除
     - sample_year_qc.parquet                   标记 qc_status='exclude_year',
                                                qc_reason='province_misplaced'
* 注意：r7_train / r7_train_validity 本体不动（row_id 与行号对齐，删行会错位）；
  消费方一律以本清单或年度子集为准。
"""
import os, sys, time
import numpy as np
import pandas as pd

PROJ = r'Z:/Mywork/论文/BaiduSyncdisk/中国土地覆盖数据'
sys.path.insert(0, os.path.join(PROJ, '代码', '5.样本重建'))
import s0_conf as C
from s1_geom import province_violation, province_of

BASE = os.path.join(PROJ, '数据/本地处理/全国清洗训练')
WORKR = os.path.join(PROJ, '数据/本地处理/样本重建')
LIST_OUT = os.path.join(BASE, '错位样本剔除清单.parquet')
SUB_DIRS = [os.path.join(BASE, '年度子集'), os.path.join(BASE, '年度子集_含稀有类')]
YEARS = list(range(2017, 2025))
MARK_STATUS, MARK_REASON = 'exclude_year', 'province_misplaced'


def build_purge():
    cols = ['lon', 'lat', 'class_new', 'src']
    t = pd.read_parquet(os.path.join(WORKR, 'r7_train.parquet'))
    if 'row_id' not in t.columns:            # 2026-09-09 起母库带显式 row_id
        t.insert(0, 'row_id', np.arange(len(t), dtype=np.int64))
    r = pd.read_parquet(os.path.join(BASE, '稀有类补样/r7_rare_samples.parquet'))
    t['origin'] = 'base_r7_train'
    r2 = r[['row_id'] + cols].copy()                    # 补样 row_id = 9,000,000+
    r2['origin'] = 'rare_topup'
    s = pd.concat([t[['row_id'] + cols], r2], ignore_index=True)
    s['province'] = province_of(s.lon.to_numpy(), s.lat.to_numpy())
    bad = province_violation(s.lon.to_numpy(), s.lat.to_numpy(),
                             s.class_new.to_numpy())
    p = s[bad].copy()
    p['reason'] = 'province_whitelist_violation'
    p = p[['row_id', 'class_new', 'lon', 'lat', 'province', 'src', 'origin', 'reason']]
    p.to_parquet(LIST_OUT, index=False)
    print(f'剔除清单: {len(p):,} 点 → {LIST_OUT}', flush=True)
    for c, g in p.groupby('class_new'):
        b = g[g.origin == 'rare_topup']
        print(f'  {c} {len(g):>5} 点（基线 {len(g)-len(b)}, 补样带入 {len(b)}）: '
              f'{dict(g.province.value_counts().head(5))}', flush=True)
    return p


def apply_purge(p):
    ids = p.row_id.to_numpy(np.int64)
    # 1) 年度子集两套
    for d in SUB_DIRS:
        for y in YEARS:
            fp = os.path.join(d, f'r7_train_{y}.parquet')
            if not os.path.isfile(fp):
                continue
            df = pd.read_parquet(fp)
            n0 = len(df)
            df = df[~df.row_id.isin(ids)].reset_index(drop=True)
            df.to_parquet(fp, index=False)
            print(f'{os.path.basename(d)}/{y}: {n0:,} → {len(df):,} '
                  f'(-{n0-len(df):,})', flush=True)
    # 2) 稀有类补样本体
    fp = os.path.join(BASE, '稀有类补样/r7_rare_samples.parquet')
    r = pd.read_parquet(fp)
    n0 = len(r)
    r = r[~r.row_id.isin(ids)].reset_index(drop=True)
    r.to_parquet(fp, index=False)
    print(f'r7_rare_samples: {n0:,} → {len(r):,}', flush=True)
    # 3) QC 表打标记（基线点才在 QC 表里）
    fp = os.path.join(BASE, 'sample_year_qc.parquet')
    if os.path.isfile(fp):
        qc = pd.read_parquet(fp)
        m = qc.row_id.isin(ids)
        n_mark = int(m.sum())
        qc.loc[m, 'qc_status'] = MARK_STATUS
        qc.loc[m, 'qc_reason'] = MARK_REASON
        qc.to_parquet(fp, index=False)
        print(f'sample_year_qc: 标记 {n_mark:,} 个 point-years 为 '
              f'{MARK_STATUS}/{MARK_REASON}', flush=True)
        del qc


def main():
    t0 = time.time()
    p = build_purge()
    apply_purge(p)
    print(f'({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
