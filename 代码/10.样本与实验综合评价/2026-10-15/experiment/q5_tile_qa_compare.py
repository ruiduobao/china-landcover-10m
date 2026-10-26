# -*- coding: utf-8 -*-
"""q5_tile_qa_compare.py — 证据审计·瓦片级独立佐证：逐瓦类构成（QA 面积表）第一次 vs 第二次

* 输入（只读）：
    - 第一次瓦片 QA：F:/lc_work/prod5p_2023/qa/{tile}_面积.csv（code,name,area_km2,share_pct）
    - 第二次瓦片 QA：F:/lc_work/prod2nd_2023/qa/{tile}_面积.csv
* 规则源：share_pct 为瓦内数据像元占比（两轮同口径，出自各自生产驱动的 qa()）；本脚本只做 1:1 对照。
* 门槛/自检：对账 eval2 小结引用的 4 瓦（川 T1606/T1607/T1706/T1707）类 2/类 10 数值（偏差 >0.05pp 告警）。
* 输出（全落本目录）：q5_瓦片QA类构成对照.csv、q5_瓦片QA对照核对.json
* 用法：python q5_tile_qa_compare.py
"""
import csv
import json
import os

QA1 = r'F:/lc_work/prod5p_2023/qa'
QA2 = r'F:/lc_work/prod2nd_2023/qa'
OUT = os.path.dirname(os.path.abspath(__file__))
TILES = [('川', t) for t in ['T1604', 'T1605', 'T1606', 'T1607', 'T1706', 'T1707']] + \
        [('粤', t) for t in ['T1801', 'T1901', 'T2001', 'T1802', 'T1902', 'T2002',
                             'T2102', 'T1903', 'T2003', 'T2103']] + \
        [('宁', t) for t in ['T1608', 'T1509', 'T1609', 'T1709', 'T1610', 'T1710']]
CLASSES = [1, 2, 4, 5, 9, 10, 11, 21, 22]
NAMES = {1: '草本旱地', 2: '乔灌园地', 4: '常绿阔叶林', 5: '落叶阔叶林', 9: '常绿灌丛',
         10: '落叶灌丛', 11: '草地', 21: '不透水面', 22: '裸地'}
# eval2 小结引用值（川四瓦 类2 / 类10：run1→run2 瓦内占比%）
# 注：四川小结 §4.3 类 2 的四个值 "7.23/10.70/23.86/34.54" 未标明逐瓦对应，实测映射为
#     T1606=34.54 / T1607=7.23 / T1706=23.86 / T1707=10.70（本表按实测映射对账）
CHECK = {'T1606': {'2': (0.0, 34.54), '10': (4.15, 0.00)},
         'T1607': {'2': (0.0, 7.23), '10': (13.69, 0.07)},
         'T1706': {'2': (0.0, 23.86), '10': (6.28, 0.03)},
         'T1707': {'2': (0.0, 10.70), '10': (9.67, 0.37)}}


def read_area(qa_dir, tile):
    fp = os.path.join(qa_dir, '%s_面积.csv' % tile)
    if not os.path.exists(fp):
        return None
    out = {}
    with open(fp, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            try:
                out[int(r['code'])] = float(r['share_pct'])
            except (ValueError, KeyError):
                pass
    return out


def main():
    rows, checks = [], {}
    for prov, tile in TILES:
        a, b = read_area(QA1, tile), read_area(QA2, tile)
        if a is None or b is None:
            print('缺 QA 跳过 %s（run1=%s run2=%s）' % (tile, a is not None, b is not None))
            continue
        for c in CLASSES:
            va, vb = a.get(c), b.get(c)
            if va is None and vb is None:
                continue
            rows.append([prov, tile, c, NAMES[c], va if va is not None else '',
                         vb if vb is not None else '',
                         '' if (va is None or vb is None) else round(vb - va, 3)])
        if tile in CHECK:
            ok = True
            for cs, (e1, e2) in CHECK[tile].items():
                g1, g2 = a.get(int(cs), 0.0), b.get(int(cs), 0.0)
                if abs(g1 - e1) > 0.05 or abs(g2 - e2) > 0.05:
                    ok = False
                    print('⚠ %s 类%s 对账偏差：期望 %.2f→%.2f 实测 %.2f→%.2f' % (tile, cs, e1, e2, g1, g2))
            checks[tile] = ok
    with open(os.path.join(OUT, 'q5_瓦片QA类构成对照.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['省', '瓦', '类码', '类名', '一次share_pct', '二次share_pct', 'Δpp'])
        w.writerows(rows)
    with open(os.path.join(OUT, 'q5_瓦片QA对照核对.json'), 'w', encoding='utf-8') as f:
        json.dump({'checks_4tiles_chuan': checks}, f, ensure_ascii=False, indent=1)
    assert all(checks.values()), '川四瓦对账失败'
    print('q5 完成：%d 行；川四瓦类2/类10对账通过（%s）' % (len(rows), checks))


if __name__ == '__main__':
    main()
