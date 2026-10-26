# -*- coding: utf-8 -*-
"""h19_accuracy_yearly.py — 逐年分层随机抽样精度量化（≥60 点/类/年；独立参考=CLCD 年度）

* 抽样：每年 稀疏 60 + 裸地 60（固定种子；按类分层，区域内自然分布）
* 参考：CLCD_v01_<year>（独立产品，未参与样本筛选）
    稀疏 ← CLCD {3 草地, 4 灌丛}；裸地 ← CLCD {7 裸地}；其余（耕地/林地/水体/冰雪/不透水/湿地/0）记 other 并从指标中剔除
    另报 WorldCover v200(2021) 参考（稀疏←{20,30}，裸地←{60}），仅 2021 年
* 指标：2×2 混淆、OA、PA/UA（Wilson 95% CI）、参考未定义比例
* 输出：data/yearly/accuracy_yearly.json + reports/accuracy_yearly.md
* 用法：python h19_accuracy_yearly.py
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
BASE = r'F:/lc_work/bare_sparse_v3'
YEARS = list(range(2017, 2025))
N_PER_CLASS = 60
SEED = 31


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0, c - h) * 100, 1), round(min(1, c + h) * 100, 1))


def main():
    cv = pd.read_parquet(os.path.join(BASE, 'data', 'yearly', 'clcd_values.parquet'))
    wc = None
    wcp = os.path.join(BASE, 'data', 'final', 'final_2023_v2.parquet')
    if os.path.isfile(wcp):
        d23 = pd.read_parquet(wcp, columns=['pid', 'wc'])
        wc = d23
    out = {}
    for y in YEARS:
        fp = os.path.join(BASE, 'data', 'yearly', 'topup_%d.parquet' % y)
        if not os.path.isfile(fp):
            continue
        d = pd.read_parquet(fp, columns=['pid', 'lon', 'lat', 'cl'])
        samp = []
        for cl in (13, 22):
            g = d[d.cl == cl]
            samp.append(g.sample(min(N_PER_CLASS, len(g)), random_state=SEED))
        s = pd.concat(samp, ignore_index=True)
        s = s.merge(cv[cv.year == y][['pid', 'clcd']], on='pid', how='left')
        s['our'] = np.where(s.cl == 13, 'sparse', 'bare')
        s['ref'] = np.where(s.clcd.isin([3, 4]), 'sparse',
                            np.where(s.clcd == 7, 'bare', 'other'))
        defined = s[s.ref != 'other']
        n_def = len(defined)
        rec = {'n_sample': int(len(s)), 'n_defined': int(n_def),
               'ref_other_pct': round(100 * (len(s) - n_def) / max(len(s), 1), 1)}
        if n_def:
            tp_sp = int(((defined.our == 'sparse') & (defined.ref == 'sparse')).sum())
            tp_ba = int(((defined.our == 'bare') & (defined.ref == 'bare')).sum())
            n_sp = int((defined.our == 'sparse').sum()); n_ba = int((defined.our == 'bare').sum())
            r_sp = int((defined.ref == 'sparse').sum()); r_ba = int((defined.ref == 'bare').sum())
            rec.update({
                'OA': round(100 * (tp_sp + tp_ba) / n_def, 1), 'OA_ci': wilson(tp_sp + tp_ba, n_def),
                'sparse_PA': round(100 * tp_sp / max(n_sp, 1), 1), 'sparse_PA_ci': wilson(tp_sp, n_sp),
                'sparse_UA': round(100 * tp_sp / max(r_sp, 1), 1), 'sparse_UA_ci': wilson(tp_sp, r_sp),
                'bare_PA': round(100 * tp_ba / max(n_ba, 1), 1), 'bare_PA_ci': wilson(tp_ba, n_ba),
                'bare_UA': round(100 * tp_ba / max(r_ba, 1), 1), 'bare_UA_ci': wilson(tp_ba, r_ba),
                'n_ref_sparse': r_sp, 'n_ref_bare': r_ba})
            rec['confusion'] = {'our_sparse_ref_sparse': tp_sp, 'our_sparse_ref_bare': int(
                ((defined.our == 'sparse') & (defined.ref == 'bare')).sum()),
                'our_bare_ref_bare': tp_ba, 'our_bare_ref_sparse': int(
                ((defined.our == 'bare') & (defined.ref == 'sparse')).sum())}
        if y == 2021 and wc is not None:
            s2 = s.merge(wc[['pid', 'wc']], on='pid', how='left')
            s2['ref_wc'] = np.where(s2.wc.isin([20, 30]), 'sparse',
                                    np.where(s2.wc == 60, 'bare', 'other'))
            dd = s2[s2.ref_wc != 'other']
            if len(dd):
                a = int(((dd.our == 'sparse') & (dd.ref_wc == 'sparse')).sum())
                b = int(((dd.our == 'bare') & (dd.ref_wc == 'bare')).sum())
                rec['wc2021'] = {'n_defined': int(len(dd)),
                                 'OA': round(100 * (a + b) / len(dd), 1),
                                 'sparse_PA': round(100 * a / max(int((dd.our == 'sparse').sum()), 1), 1),
                                 'bare_PA': round(100 * b / max(int((dd.our == 'bare').sum()), 1), 1)}
        out[y] = rec
        print('%d: n=%d defined=%d OA=%s%% (%s) 稀疏 PA=%s UA=%s ｜ 裸地 PA=%s UA=%s' % (
            y, len(s), n_def, rec.get('OA'), rec.get('OA_ci'), rec.get('sparse_PA'),
            rec.get('sparse_UA'), rec.get('bare_PA'), rec.get('bare_UA')), flush=True)
    json.dump(out, open(os.path.join(BASE, 'data', 'yearly', 'accuracy_yearly.json'), 'w',
                        encoding='utf-8'), ensure_ascii=False, indent=1)
    L = ['# 逐年分层随机抽样精度量化（独立参考 = CLCD 年度产品）\n',
         '> 每年每类 ≥60 点；参考标签：稀疏←CLCD{3,4}、裸地←CLCD{7}；其余记为 other 并剔除。\n',
         '| 年 | 样本 | 有效参考 | OA | 稀疏 PA | 稀疏 UA | 裸地 PA | 裸地 UA |',
         '|---|---:|---:|---|---|---|---|---|']
    for y, r in out.items():
        L.append('| %s | %d | %d | %s%% (%s) | %s%% | %s%% | %s%% | %s%% |' % (
            y, r['n_sample'], r['n_defined'], r['OA'], r['OA_ci'],
            r['sparse_PA'], r['sparse_UA'], r['bare_PA'], r['bare_UA']))
    open(os.path.join(BASE, 'reports', 'accuracy_yearly.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('→ data/yearly/accuracy_yearly.json, reports/accuracy_yearly.md')


if __name__ == '__main__':
    main()
