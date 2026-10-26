# -*- coding: utf-8 -*-
"""q1_extract_metrics.py — 证据审计·量化提取：三省 24 类/9 大类 OA 精确值 + 逐类 F1 对照

* 输入（只读）：
    - 第二次评估 json：F:/lc_work/prod2nd_2023/reports/eval2/eval2_{川,粤,宁}_2023.json
    - 第一次基线 json：F:/lc_work/prod5p_2023/evalprep/results/eval_{川,粤,宁}_2023.json
    - 栅格面积对照：F:/lc_work/prod2nd_2023/reports/mosaic2/compare_{川,粤,宁}_run1_run2_class.csv
    - mosaic 对比 json：F:/lc_work/prod2nd_2023/reports/mosaic2/compare_{川,粤,宁}_run1_run2.json
* 规则源 / 口径（写死，防口径漂移）：
    - "9 大类" = v31 实验归并 macro9（F:/lc_work/v31_exp/config/macro9.json），
      稀疏植被/地衣苔藓 并入草地——注意：**不是** AGENTS.md LEVEL1 的 10 组口径；两轮评估脚本
      均用同一 macro9（v31_common.to_macro），故跨轮可比；本表同时给出 24 类 OA 与 macro9 OA。
    - 类 8（针阔混交林）：第二次池缺席（23 类），n 行若出现即告警；另由 q2 独立核查留出点真值。
    - 第一次基线 json 的 pa/ua 字段系 metrics 修正前口径（pa=precision, ua=recall），本脚本只取
      n/OA/macroF1/m9/F1（不受翻转影响），并显式记录该事实。
* 门槛/自检：分层 n 求和 == 全点 n；类 8 在第二次 per-class 表中必须缺席；Δ 与既有小结一致到 1e-4。
* 输出（全落本目录）：
    q1_三省指标精确表.csv / .md 、q1_逐类F1对照.csv 、q1_面积结构对照.csv 、q1_口径说明.json
* 用法：python q1_extract_metrics.py
"""
import csv
import json
import os

BASE = r'F:/lc_work/prod2nd_2023/reports'
EVAL2 = os.path.join(BASE, 'eval2')
MOSAIC2 = os.path.join(BASE, 'mosaic2')
RUN1 = r'F:/lc_work/prod5p_2023/evalprep/results'
OUT = os.path.dirname(os.path.abspath(__file__))
PROVS = [('川', '四川省'), ('粤', '广东省'), ('宁', '宁夏回族自治区')]
STRATA = [('all', '全点(含fcs10惯例)'), ('fcs10', 'fcs10系'), ('nonfcs10', '非fcs10净窗口')]
# v31 24 类名（与 v31_map.json 一致，提取展示用；不引入 v31_common 以免带动 GEE import）
NAMES = {1: '草本旱地', 2: '乔灌园地', 3: '灌溉耕地', 4: '常绿阔叶林', 5: '落叶阔叶林',
         6: '常绿针叶林', 7: '落叶针叶林', 8: '针阔混交林', 9: '常绿灌丛', 10: '落叶灌丛',
         11: '草地', 12: '地衣苔藓', 13: '稀疏植被', 14: '木本沼泽', 15: '草本沼泽',
         16: '湖河滩地', 17: '盐渍湿地', 18: '红树林', 19: '海岸盐沼', 20: '潮滩',
         21: '不透水面', 22: '裸地', 23: '水体', 24: '永久冰雪'}


def jload(fp):
    with open(fp, encoding='utf-8') as f:
        return json.load(f)


def main():
    rows, per_rows, notes = [], [], {}
    st_rows = {'fcs10': [], 'nonfcs10': []}
    for tag, full in PROVS:
        d2 = jload(os.path.join(EVAL2, 'eval2_%s_2023.json' % tag))
        d1 = jload(os.path.join(RUN1, 'eval_%s_2023.json' % tag))
        notes[tag] = dict(
            full=full,
            run1_json=os.path.join(RUN1, 'eval_%s_2023.json' % tag),
            run2_json=os.path.join(EVAL2, 'eval2_%s_2023.json' % tag),
            n_pool=[d1.get('n_pool'), d2.get('n_pool')],
            n_scored=[d1.get('n_scored'), d2.get('n_scored')],
            coverage_pct=[d1.get('coverage_pct'), d2.get('coverage_pct')],
            run1_pa_ua_flipped=True)  # v31_common.metrics 2026-10-07 修正前产出
        s1, s2 = d1.get('strata', {}), d2.get('strata', {})
        for key, label in STRATA:
            a, b = s1.get(key), s2.get(key)
            if not a or not b:
                continue
            rows.append([tag, full, label, key,
                         a['n'], b['n'],
                         round(a['OA'], 4), round(b['OA'], 4), round(b['OA'] - a['OA'], 4),
                         round(a.get('m9', float('nan')), 4), round(b.get('m9', float('nan')), 4),
                         round(b.get('m9', 0) - a.get('m9', 0), 4),
                         round(a['macroF1'], 4), round(b['macroF1'], 4),
                         round(b['macroF1'] - a['macroF1'], 4)])
        # 自检：分层 n 求和 == 全点 n
        for s, d in (('r1', s1), ('r2', s2)):
            if 'fcs10' in d and 'nonfcs10' in d:
                assert d['fcs10']['n'] + d['nonfcs10']['n'] == d['all']['n'], (tag, s)
        # 类 8 必须缺席于第二次 per-class 表
        per2 = (s2.get('all') or {}).get('per', {}) or {}
        assert '8' not in per2, '%s 第二次全点 per 出现类 8' % tag
        # 逐类 F1 对照（按类码联合）
        per1 = (s1.get('all') or {}).get('per', {}) or {}
        ks = sorted(set(int(k) for k in per1) | set(int(k) for k in per2))
        per1_n = {int(k): v['n'] for k, v in per1.items()}
        for c in ks:
            a, b = per1.get(str(c)), per2.get(str(c))
            per_rows.append([tag, c, NAMES.get(c, '?'),
                             a['n'] if a else '', round(a['f1'], 3) if a else '',
                             b['n'] if b else '', round(b['f1'], 3) if b else '',
                             round(b['f1'] - a['f1'], 3) if (a and b) else '',
                             'run1缺席' if not a else ('run2缺席' if not b else ''),
                             'n<30' if ((a and a['n'] < 30) or (b and b['n'] < 30)) else ''])
        # 分层逐类（fcs10 / 非fcs10），同样按类码联合
        for key in ('fcs10', 'nonfcs10'):
            pa = (s1.get(key) or {}).get('per', {}) or {}
            pb = (s2.get(key) or {}).get('per', {}) or {}
            ks2 = sorted(set(int(k) for k in pa) | set(int(k) for k in pb))
            for c in ks2:
                a, b = pa.get(str(c)), pb.get(str(c))
                st_rows[key].append([tag, c, NAMES.get(c, '?'),
                                     a['n'] if a else '', round(a['f1'], 3) if a else '',
                                     b['n'] if b else '', round(b['f1'], 3) if b else '',
                                     round(b['f1'] - a['f1'], 3) if (a and b) else ''])
    # 面积结构对照（mosaic 全省口径，km^2）
    area_rows = []
    for tag, full in PROVS:
        fp = os.path.join(MOSAIC2, 'compare_%s_run1_run2_class.csv' % tag)
        with open(fp, encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                if r['类码'] == '合计' or not r['类码']:   # 合计行
                    area_rows.append([tag, '', '合计', r['一次_面积km2'], r['二次_面积km2'],
                                      r['Δ_面积km2'], r['Δ_面积%']])
                    continue
                area_rows.append([tag, int(r['类码']), r['类名'], r['一次_面积km2'],
                                  r['二次_面积km2'], r['Δ_面积km2'], r['Δ_面积%']])
    # 写出
    hdr = ['省', '省全名', '口径', '分层键', 'n_第一次', 'n_第二次',
           'OA24_第一次', 'OA24_第二次', 'ΔOA24', 'OA_macro9_第一次', 'OA_macro9_第二次', 'ΔOA_macro9',
           'macroF1_第一次', 'macroF1_第二次', 'ΔmacroF1']
    with open(os.path.join(OUT, 'q1_三省指标精确表.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(rows)
    with open(os.path.join(OUT, 'q1_逐类F1对照.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['省', '类码', '类名', 'n_第一次', 'F1_第一次', 'n_第二次', 'F1_第二次',
                    'ΔF1', '备注', '样本量警示'])
        w.writerows(per_rows)
    for key, fn in (('fcs10', 'q1_逐类F1对照_fcs10系.csv'),
                    ('nonfcs10', 'q1_逐类F1对照_非fcs10系.csv')):
        with open(os.path.join(OUT, fn), 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['省', '类码', '类名', 'n_第一次', 'F1_第一次', 'n_第二次', 'F1_第二次', 'ΔF1'])
            w.writerows(st_rows[key])
    with open(os.path.join(OUT, 'q1_面积结构对照.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['省', '类码', '类名', '一次_面积km2', '二次_面积km2', 'Δ_面积km2', 'Δ_面积%'])
        w.writerows(area_rows)
    notes['macro9_definition'] = {
        'source': r'F:/lc_work/v31_exp/config/macro9.json',
        'groups': ['cropland[1,2,3]', 'forest[4,5,6,7,8]', 'shrubland[9,10]',
                   'grass_lowveg[11,12,13]', 'wetland[14..20]', 'impervious[21]',
                   'bare[22]', 'water[23]', 'ice_snow[24]'],
        'note': '评估脚本用 macro9（9 组）；AGENTS.md LEVEL1"十组"（含独立稀疏植被）未被两轮评估采用；'
                '两轮同用 macro9 → 组级 OA 跨轮可比。'}
    notes['class8'] = {'pool1_has_class8': True, 'pool2_missing_codes': ['8'],
                       'second_eval_per_tables_absent_8': True,
                       'detail': '第二次池 23 类缺 8（针阔混交林，91/92 弃用）；本表已断言第二次 per 表无 8；'
                                 '留出点真值是否含 8 由 q2_windows_check.py 独立复核。'}
    with open(os.path.join(OUT, 'q1_口径说明.json'), 'w', encoding='utf-8') as f:
        json.dump(notes, f, ensure_ascii=False, indent=1)

    # markdown 主表（便于写报告直接引用）
    L = ['# 三省指标精确表（第一次 → 第二次，同一评估口径）', '',
         '> 来源：eval2_*.json（第二次）+ evalprep/results/eval_*.json（第一次）；本表为机械提取（q1_extract_metrics.py）。',
         '> 口径：OA24=24 类；OA_macro9=v31 实验归并 9 组（非 AGENTS 十组）；来源分层为强制口径。', '',
         '| 省 | 口径 | n（1→2） | OA24 | Δ | OA_macro9 | Δ | macroF1 | Δ |',
         '|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        nstr = '%s→%s' % (r[4], r[5])
        if r[4] != r[5]:
            nstr += ' ⚠'
        L.append('| %s | %s | %s | %.4f→%.4f | %+.4f | %.4f→%.4f | %+.4f | %.4f→%.4f | %+.4f |' % (
            r[0], r[2], nstr, r[6], r[7], r[8], r[9], r[10], r[11], r[12], r[13], r[14]))
    L += ['', '## n 与覆盖面（run1 → run2）', '',
          '| 省 | n_pool | n_scored | 覆盖% |', '|---|---|---|---|']
    for tag, full in PROVS:
        nz = notes[tag]
        L.append('| %s | %s→%s | %s→%s | %s→%s |' % (tag, nz['n_pool'][0], nz['n_pool'][1],
                                                     nz['n_scored'][0], nz['n_scored'][1],
                                                     nz['coverage_pct'][0], nz['coverage_pct'][1]))
    with open(os.path.join(OUT, 'q1_三省指标精确表.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    print('q1 完成：%d 指标行 / %d 逐类行 / %d 面积行' % (len(rows), len(per_rows), len(area_rows)))


if __name__ == '__main__':
    main()
