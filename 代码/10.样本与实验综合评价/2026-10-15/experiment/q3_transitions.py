# -*- coding: utf-8 -*-
"""q3_transitions.py — 证据审计·栅格级复核：全通过一致率 / 核心区一致率 / 转移结构

* 输入（只读）：F:/lc_work/prod2nd_2023/reports/mosaic2/compare_{川,粤,宁}_run1_run2.json
  （含 24×24 全通过转移矩阵 transition_matrix、classes 表、common/changed/agree_fullpass）
* 规则源 / 口径（对报告 §5.0 的独立复算，写死）：
    - 全域一致率 = Σdiag / Σmatrix（对账 agree_fullpass）
    - 核心区 = 两侧类码均 ∉ {2 乔灌园地, 9 常绿灌丛, 10 落叶灌丛} 的矩阵像元；核心区一致率 = 核心区对角/核心区
    - 变化涉 {2,9,10} 占比 = 非对角且任一侧 ∈ {2,9,10} 的像元 / 非对角像元
    - km² 换算 = 省总 km² / 共同域 px（块心均值口径的审计近似；逐类精确值以 compare_*_class.csv 为准）
* 门槛：本脚本对账目标 = 报告 拼接对比实跑报告.md §5.0 表（0.9527/0.9454/0.9807；86.87/88.53/96.74%），
  偏差 >0.0002 即告警。
* 输出（全落本目录）：q3_栅格口径核对.json、q3_最大转移对.csv、q3_转移汇总.csv
* 用法：python q3_transitions.py
"""
import csv
import json
import os

OUT = os.path.dirname(os.path.abspath(__file__))
MOSAIC2 = r'F:/lc_work/prod2nd_2023/reports/mosaic2'
PROVS = [('川', '四川省', 0.9527, 0.8687), ('粤', '广东省', 0.9454, 0.8853),
         ('宁', '宁夏回族自治区', 0.9807, 0.9674)]
S = {2, 9, 10}
NAMES = {1: '草本旱地', 2: '乔灌园地', 3: '灌溉耕地', 4: '常绿阔叶林', 5: '落叶阔叶林',
         6: '常绿针叶林', 7: '落叶针叶林', 8: '针阔混交林', 9: '常绿灌丛', 10: '落叶灌丛',
         11: '草地', 12: '地衣苔藓', 13: '稀疏植被', 14: '木本沼泽', 15: '草本沼泽',
         16: '湖河滩地', 17: '盐渍湿地', 18: '红树林', 19: '海岸盐沼', 20: '潮滩',
         21: '不透水面', 22: '裸地', 23: '水体', 24: '永久冰雪'}


def main():
    out = {}
    top_rows, agg_rows = [], []
    for tag, full, rep_core, rep_share in PROVS:
        d = json.load(open(os.path.join(MOSAIC2, 'compare_%s_run1_run2.json' % tag), encoding='utf-8'))
        tm = d['transition_matrix']
        codes = [c['code'] for c in d['classes']]           # 1..24 顺序
        total = sum(sum(r) for r in tm)
        diag = sum(tm[i][i] for i in range(24))
        changed = total - diag
        agree_full = diag / total
        core_tot = core_ok = 0
        s_tot = s_ok = 0
        chg_s = 0
        nons_changes = []
        for i in range(24):
            for j in range(24):
                ci, cj = codes[i], codes[j]
                if ci not in S and cj not in S:
                    core_tot += tm[i][j]
                    if i == j:
                        core_ok += tm[i][j]
                else:
                    s_tot += tm[i][j]
                    if i == j:
                        s_ok += tm[i][j]
                if i != j and (ci in S or cj in S):
                    chg_s += tm[i][j]
                if i != j and ci not in S and cj not in S:
                    nons_changes.append((tm[i][j], ci, cj))
        core_agree = core_ok / core_tot
        s_agree = s_ok / s_tot
        share_s = chg_s / changed
        nons_changes.sort(reverse=True)
        factor = d['first_km2'] / total                 # km2 per px（审计近似）
        # 最大非对角对
        pairs = []
        for i in range(24):
            for j in range(24):
                if i != j and tm[i][j] > 0:
                    pairs.append((tm[i][j], codes[i], codes[j]))
        pairs.sort(reverse=True)
        for k, (px, a, b) in enumerate(pairs[:10], 1):
            top_rows.append([tag, k, a, NAMES[a], b, NAMES[b], px, round(px * factor, 1)])
        # 结构汇总
        f2i = lambda c: NAMES.get(c, '?')
        def cell(a, b):
            return tm[codes.index(a)][codes.index(b)]
        agg = dict(
            agree_full_recalc=round(agree_full, 6), agree_full_json=d['agree_fullpass'],
            core_agree=round(core_agree, 4), core_cover=round(core_tot / total, 4),
            s_region_cover=round(s_tot / total, 4), s_region_agree=round(s_agree, 4),
            changed_share_involving_S=round(share_s, 4),
            changed_share_not_involving_S=round(1 - share_s, 4),
            nons_changed_share_of_common=round(sum(x[0] for x in nons_changes) / total, 4),
            nons_top5=[('%d %s→%d %s' % (a, NAMES[a], b, NAMES[b]), px)
                       for px, a, b in nons_changes[:5]],
            rep_core=rep_core, rep_share=rep_share,
            core_dev=round(core_agree - rep_core, 4), share_dev=round(share_s - rep_share, 4),
            km2_per_px=round(factor, 8),
            mask_note='core=两侧均∉{2,9,10}（占共同域 core_cover）；S 区=任一侧∈{2,9,10}；'
                      '核心区数字只说明剩余类别稳定，不等于全图变化均为标签口径置换（选择性条件化）')
        out[tag] = agg
        # 专题对（川/粤/宁 各自的结构性转移，km²）
        if tag == '川':
            spec = [('类4 常绿阔叶林→类2 乔灌园地', cell(4, 2)), ('类4→类5 落叶阔叶林', cell(4, 5)),
                    ('类4→类6 常绿针叶林', cell(4, 6)), ('类1 草本旱地→类2', cell(1, 2)),
                    ('类10→类2', cell(10, 2)), ('类10→类1', cell(10, 1))]
        elif tag == '粤':
            spec = [('类9 常绿灌丛→类2', cell(9, 2)), ('类4→类2', cell(4, 2)), ('类1→类2', cell(1, 2)),
                    ('类10→类2', cell(10, 2)), ('类10→类1', cell(10, 1)), ('类4→类6', cell(4, 6))]
        else:
            spec = [('类10→类1', cell(10, 1)), ('类10→类22 裸地', cell(10, 22)),
                    ('类10→类11 草地', cell(10, 11)), ('类1→类2', cell(1, 2)),
                    ('类10→类21 不透水面', cell(10, 21)), ('类11→类10', cell(11, 10))]
        for name, px in spec:
            agg_rows.append([tag, name, px, round(px * factor, 1)])
        assert abs(agree_full - d['agree_fullpass']) < 1e-6, (tag, agree_full, d['agree_fullpass'])
        assert abs(core_agree - rep_core) < 2e-4, '%s 核心区对账偏差 %+.4f' % (tag, core_agree - rep_core)
        assert abs(share_s - rep_share) < 2e-4, '%s 涉S占比对账偏差 %+.4f' % (tag, share_s - rep_share)
        print('%s: 全域=%.4f(报 %.4f) 核心区=%.4f(报 %.4f) 涉{2,9,10}=%.2f%%(报 %.2f%%) 变化=%.6f' % (
            tag, agree_full, d['agree_fullpass'], core_agree, rep_core, 100 * share_s,
            100 * rep_share, changed / total))
    with open(os.path.join(OUT, 'q3_栅格口径核对.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, 'q3_最大转移对.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['省', '序', '一次类码', '一次类名', '二次类码', '二次类名', '像元', '≈km2'])
        w.writerows(top_rows)
    with open(os.path.join(OUT, 'q3_转移汇总.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['省', '转移对', '像元', '≈km2'])
        w.writerows(agg_rows)
    print('q3 完成：核心区/涉S 与报告 §5.0 对账通过（偏差 <2e-4）')


if __name__ == '__main__':
    main()
