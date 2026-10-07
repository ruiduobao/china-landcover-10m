# -*- coding: utf-8 -*-
"""
e10_sample_qa.py — 样本一键质检（省界白名单 + 生态规则 + 国界 + NN 间距 + 类计数）
* 输入: 数据/本地处理/样本重建/r7_train.parquet（母库）+ 全国清洗训练/稀有类补样/r7_rare_samples.parquet（可选）
* 规则源: s0_conf.ECO_LAT/ECO_EXTRA_HARD/ECO_PROVINCE + s1_geom.china_contains/province_violation
* 门槛: 同类 NN 中位 ≥500m；白名单/生态/国界违规 = 0；稀有类(<2000)列缺口
* 输出: 控制台 PASS/FAIL 表 + 数据/本地处理/全国清洗训练/样本QA/qc_report.json
* 用法: python e10_sample_qa.py [--plots]   （--plots 额外重生成 e9 分布图）
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
KB = r'Z:/地理所/论文/中国土地覆盖数据_2017-2024'
sys.path.insert(0, os.path.join(PROJ, '代码', '0.本地流水线'))
sys.path.insert(0, os.path.join(PROJ, '代码', '5.样本重建'))
import s0_conf as C
from s1_geom import china_contains, province_violation
from lc_conf import class_name

TRAIN = os.path.join(PROJ, '数据/本地处理/样本重建/r7_train.parquet')
RARE = os.path.join(KB, '数据/本地处理/全国清洗训练/稀有类补样/r7_rare_samples.parquet')
OUTD = os.path.join(PROJ, '数据/本地处理/全国清洗训练/样本QA')

NN_FLOOR_M = 500


def nn_median_km(lon, lat):
    xy = np.c_[lon * np.cos(np.radians(lat.mean())), lat] * 111.32
    t = cKDTree(xy)
    d, _ = t.query(xy, k=2)
    return float(np.median(d[:, 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plots', action='store_true')
    a = ap.parse_args()
    os.makedirs(OUTD, exist_ok=True)

    t = pd.read_parquet(TRAIN, columns=['lon', 'lat', 'class_new'])
    parts = [('base', t)]
    if os.path.isfile(RARE):
        r = pd.read_parquet(RARE, columns=['lon', 'lat', 'class_new'])
        parts.append(('rare', r))
    df = pd.concat([p for _, p in parts], ignore_index=True)
    print(f'质检对象: 母库 {len(t):,} + 稀有类 {len(df)-len(t):,} = {len(df):,} 点\n')

    lon, lat, cls = df.lon.to_numpy(), df.lat.to_numpy(), df.class_new.to_numpy()
    rep = {'time': __import__('time').strftime('%Y-%m-%d %H:%M'),
           'n_total': int(len(df)), 'classes': {}, 'fail': False}

    inb = china_contains(lon, lat)
    pviol = province_violation(lon, lat, cls)
    eviol = C.eco_violation(lon, lat, cls)
    print(f"{'类':<4}{'名称':<12}{'n':>8}{'国界违规':>8}{'省界违规':>8}{'生态违规':>8}{'NN中位km':>9}")
    for c in sorted(df.class_new.unique()):
        m = cls == c
        n = int(m.sum())
        nnm = nn_median_km(lon[m], lat[m]) if n > 5 else float('nan')
        row = {'n': n,
               'boundary_viol': int((~inb[m]).sum()),
               'province_viol': int(pviol[m].sum()),
               'eco_viol': int(eviol[m].sum()),
               'nn_median_km': round(nnm, 2) if n > 5 else None,
               'nn_pass': bool(nnm >= NN_FLOOR_M / 1000) if n > 5 else None}
        rep['classes'][int(c)] = {'name': class_name(int(c)), **row}
        bad = row['boundary_viol'] + row['province_viol'] + row['eco_viol']
        flag = 'FAIL' if bad else ('NN弱' if row['nn_pass'] is False else 'ok')
        if bad or row['nn_pass'] is False:
            rep['fail'] = True
        print(f"{c:<4}{class_name(int(c)):<12}{n:>8}{row['boundary_viol']:>8}"
              f"{row['province_viol']:>8}{row['eco_viol']:>8}{nnm:>9.2f}  {flag}")

    n_bad = int((~inb).sum() + pviol.sum() + eviol.sum())
    rep['total_viol'] = n_bad
    print(f"\n总违规: {n_bad}  → {'❌ FAIL' if rep['fail'] else '✅ PASS'}")
    json.dump(rep, open(os.path.join(OUTD, 'qc_report.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('报告:', os.path.join(OUTD, 'qc_report.json'))
    if a.plots:
        os.system(f'python "{os.path.join(PROJ, "代码/4.全国清洗训练/e9_class_maps.py")}"')


if __name__ == '__main__':
    main()
