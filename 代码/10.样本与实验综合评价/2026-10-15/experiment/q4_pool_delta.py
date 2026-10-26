# -*- coding: utf-8 -*-
"""q4_pool_delta.py — 证据审计·样本端复核：旧池 vs 新池 逐类点数与来源构成

* 输入（只读）：
    - 旧池：F:/lc_work/prod5p_2023/data/train2023_clean.parquet（2,060,577 点；class_new=旧码空间）
    - 新池：F:/lc_work/prod2nd_2023/data/train2023_v2.parquet（1,764,020 点；cl=24 类码）
* 规则源：v31_common.to_v31（旧码→24 类，与 c4/生产同源）；类名 v31_map.json；
  位置键 = c4_eval_second.pos_key（(lon,lat) 6 位，双池同口径）。
* 门槛/自检：旧池总点数 == 2,060,577；新池总点数 == 1,764,020；新池缺席类 == {8}；
  对 doc 68 §10.2 引用值逐项对账（落叶灌丛 425,696→105,130；乔灌园地 4,854→141,686），偏差 >1% 告警。
* 附加（2026-10-15 审计修订）：**同点位置键重叠**——按 24 类码分组，旧池类 c 位置键集合 ∩ 新池类 c
  位置键，报"新池类 c 点落旧池同类点位的比例"（核"森林多为同点筛选而非换教师"之说；src 字段只是
  类目录名，不能单独证明标签来源）。
* 输出（全落本目录）：q4_池类目对照.csv、q4_池来源构成.json、q4_池点位重叠.csv
* 用法：python q4_pool_delta.py
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, r'F:/lc_work/v31_exp/code')
import v31_common as VC

OUT = os.path.dirname(os.path.abspath(__file__))
POOL1 = r'F:/lc_work/prod5p_2023/data/train2023_clean.parquet'
POOL2 = r'F:/lc_work/prod2nd_2023/data/train2023_v2.parquet'
NAMES = VC.V31_NAMES()


def main():
    p1 = pd.read_parquet(POOL1, columns=['class_new', 'src', 'lon', 'lat'])
    assert len(p1) == 2060577, len(p1)
    p1['c24'] = VC.to_v31(p1.class_new.to_numpy(int))
    p2 = pd.read_parquet(POOL2, columns=['cl', 'src', 'lon', 'lat'])
    assert len(p2) == 1764020, len(p2)
    p2 = p2.rename(columns={'cl': 'c24'})
    c1, c2 = p1.c24.value_counts(), p2.c24.value_counts()
    codes = sorted(set(int(k) for k in c1.index) | set(int(k) for k in c2.index))
    assert '8' not in [str(c) for c in p2.c24.unique()], '新池出现类 8'
    rows = []
    for c in codes:
        n1, n2 = int(c1.get(c, 0)), int(c2.get(c, 0))
        rows.append([c, NAMES.get(str(c), '?'), n1, n2, n2 - n1,
                     round(100 * n1 / len(p1), 3), round(100 * n2 / len(p2), 3)])
    with open(os.path.join(OUT, 'q4_池类目对照.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        import csv
        w = csv.writer(f)
        w.writerow(['24类码', '类名', '旧池n', '新池n', 'Δn', '旧池占比%', '新池占比%'])
        w.writerows(rows)
    # 来源构成（关键类，top5 src）
    out = dict(old_total=len(p1), new_total=len(p2), class_delta={
        str(c): dict(old=int(c1.get(c, 0)), new=int(c2.get(c, 0))) for c in codes})
    key_srcs = {}
    for c in [2, 4, 5, 6, 9, 10, 16]:
        o = p1[p1.c24 == c].src.value_counts().head(5)
        n = p2[p2.c24 == c].src.value_counts().head(5)
        key_srcs[str(c)] = dict(
            name=NAMES.get(str(c), '?'),
            old={str(k): int(v) for k, v in o.items()},
            new={str(k): int(v) for k, v in n.items()})
    out['key_class_src_top5'] = key_srcs
    # 对账 doc 68 §10.2 引用值（全国池口径）
    def chk(c, v_old, v_new):
        a, b = int(c1.get(c, 0)), int(c2.get(c, 0))
        return dict(class24=c, expect_old=v_old, got_old=a, expect_new=v_new, got_new=b,
                    old_ok=abs(a - v_old) <= max(1, 0.01 * v_old),
                    new_ok=abs(b - v_new) <= max(1, 0.01 * v_new))
    out['doc68_reconciliation'] = [chk(10, 425696, 105130), chk(2, 4854, 141686)]
    # ---- 同点位置键重叠（核"森林多为同点筛选"之说；位置键=c4 同式） ----
    import numpy as np

    def pos_key(lon, lat, nd=6):
        a = np.round(np.asarray(lon, dtype=float) * 10 ** nd).astype('int64') + 200_000_000
        b = np.round(np.asarray(lat, dtype=float) * 10 ** nd).astype('int64') + 100_000_000
        return a * 400_000_000 + b

    k1 = pos_key(p1.lon.to_numpy(), p1.lat.to_numpy())
    k2 = pos_key(p2.lon.to_numpy(), p2.lat.to_numpy())
    k1_all, k2_all = np.unique(k1), np.unique(k2)
    ov_rows = []
    for c in codes:
        i1 = np.where(p1.c24.to_numpy() == c)[0]
        i2 = np.where(p2.c24.to_numpy() == c)[0]
        if len(i1) == 0 or len(i2) == 0:
            ov_rows.append([c, NAMES.get(str(c), '?'), len(i1), len(i2), '', '', ''])
            continue
        s_same = float(np.isin(np.unique(k2[i2]), np.unique(k1[i1])).sum())
        # 新点侧命中率（按点计，含重复点）
        new_hit_same = float(np.isin(k2[i2], np.unique(k1[i1])).mean())
        new_hit_any = float(np.isin(k2[i2], k1_all).mean())
        old_hit_same = float(np.isin(k1[i1], np.unique(k2[i2])).mean())
        ov_rows.append([c, NAMES.get(str(c), '?'), len(i1), len(i2),
                        round(100 * new_hit_same, 2), round(100 * new_hit_any, 2),
                        round(100 * old_hit_same, 2)])
    out['position_overlap_note'] = ('新池类 c 点命中旧池同类 6 位位置键比例（%）＝"同点再用"；'
                                    '命中旧池任意类位置键＝"该点仍来自旧池点位空间"；'
                                    '反向 = 旧池类 c 被新池同类保留。位置键=6 位 (lon,lat)，c4 同式。')
    with open(os.path.join(OUT, 'q4_池点位重叠.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['24类码', '类名', '旧池n', '新池n', '新池命中旧池同类位置%',
                    '新池命中旧池任意类位置%', '旧池命中新池同类位置%'])
        w.writerows(ov_rows)
    for r in ov_rows:
        if r[0] in (2, 4, 5, 6, 7, 9, 10, 16, 15, 14, 21):
            print('重叠 类%s %s: 旧%s→新%s | 新命中旧同类 %s%% / 旧池任意类 %s%% / 旧被新保留 %s%%' % tuple(r))
    with open(os.path.join(OUT, 'q4_池来源构成.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for c, d in key_srcs.items():
        print('类%s %s: 旧 %s | 新 %s' % (c, d['name'], d['old'], d['new']))
    for r in out['doc68_reconciliation']:
        print('对账 类%d：期望 %d→%d，实测 %d→%d（old_ok=%s new_ok=%s）' % (
            r['class24'], r['expect_old'], r['expect_new'], r['got_old'], r['got_new'],
            r['old_ok'], r['new_ok']))
    print('q4 完成：%d 类目行' % len(rows))


if __name__ == '__main__':
    main()
