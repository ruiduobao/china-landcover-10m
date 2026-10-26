# -*- coding: utf-8 -*-
"""a7_derive_merge.py — 合并级判定表（glue）

背景：本轮盲判多按结构大类给出（乔木林/灌丛/草地/耕地/水体/裸地/稀疏植被），
      a6 的细类别名表无法匹配这些词 → 全部落"未匹配"。
本脚本按**合并级结构组**判定 确认/否定/未知，供 QC 三率使用；
细类精度（F1）仍只在判读能到细类时计算，不在此表。

* 输入：--form 判读表（blind_id, 实际类别）、--sidecar 私有侧表（cl_pool/cl_name）
* 输出：qc/verdicts_merge_<tag>.csv（blind_id, verdict, level, prior_group, judged_group）
* 用法：python a7_derive_merge.py --form qc/pass1_norm.csv --sidecar private/uid_map.csv --tag pass1
"""
import argparse, csv, os, sys
sys.stdout.reconfigure(encoding='utf-8')

GROUP_OF_CODE = {}
for c in (1, 3): GROUP_OF_CODE[c] = 'crop'
for c in (4, 5, 6, 7, 8): GROUP_OF_CODE[c] = 'forest'
for c in (9, 10): GROUP_OF_CODE[c] = 'shrub'
GROUP_OF_CODE[11] = 'grass'
GROUP_OF_CODE[2] = 'orchard'
GROUP_OF_CODE[12] = 'lichen'
GROUP_OF_CODE[13] = 'sparse'
GROUP_OF_CODE[14] = 'wetland'
GROUP_OF_CODE[15] = 'wetland'
for c in (16, 17, 18, 19, 20): GROUP_OF_CODE[c] = 'coastal_wet'
GROUP_OF_CODE[21] = 'built'
GROUP_OF_CODE[22] = 'bare'
GROUP_OF_CODE[23] = 'water'
GROUP_OF_CODE[24] = 'snow'

# 判读词（含大类与细类） -> 结构组
JUD2GROUP = {
    '乔木林': 'forest', '阔叶林（常绿/落叶不可分）': 'forest',
    '常绿阔叶林': 'forest', '落叶阔叶林': 'forest', '常绿针叶林': 'forest',
    '落叶针叶林': 'forest', '针阔混交林': 'forest',
    '灌丛': 'shrub', '灌木/草地': 'shrub', '常绿灌木': 'shrub', '落叶灌木': 'shrub',
    '草地': 'grass',
    '草本旱地': 'crop', '灌溉耕地': 'crop', '耕地': 'crop',
    '乔灌园地': 'orchard', '园地': 'orchard',
    '地衣苔藓': 'lichen', '稀疏植被': 'sparse',
    '木本沼泽': 'wetland', '草本沼泽': 'wetland',
    '湖河滩地': 'coastal_wet', '盐渍湿地': 'coastal_wet', '红树林': 'coastal_wet',
    '海岸盐沼': 'coastal_wet', '潮滩': 'coastal_wet',
    '不透水面': 'built', '裸地': 'bare', '水体': 'water', '永久冰雪': 'snow',
    '无法判定': 'unknown',
}
FINE = {'常绿阔叶林', '落叶阔叶林', '常绿针叶林', '落叶针叶林', '针阔混交林',
        '草本旱地', '灌溉耕地', '乔灌园地', '常绿灌木', '落叶灌木', '水体', '裸地',
        '稀疏植被', '草地', '不透水面', '永久冰雪', '木本沼泽', '草本沼泽'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ws', default=r'F:/lc_work/label_repair_2023_20261015')
    ap.add_argument('--form', required=True)
    ap.add_argument('--sidecar', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--outdir', default=None)
    a = ap.parse_args()

    forms = list(csv.DictReader(open(a.form, encoding='utf-8-sig')))
    side = {r['point_id']: r for r in csv.DictReader(open(a.sidecar, encoding='utf-8-sig'))}
    outdir = a.outdir or os.path.join(a.ws, 'qc')
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for r in forms:
        bid = str(r.get('blind_id') or r.get('point_id')).strip()
        raw = str(r.get('实际类别', '') or '').strip()
        s = side.get(bid, {})
        prior_code = s.get('cl_pool', '')
        try:
            prior_code = int(float(prior_code)) if prior_code not in ('', None) else None
        except Exception:
            prior_code = None
        jg = JUD2GROUP.get(raw, 'unknown')
        pg = GROUP_OF_CODE.get(prior_code, 'unknown') if prior_code else None
        if jg == 'unknown' or pg is None:
            verdict = '未知'
        else:
            verdict = '确认' if jg == pg else '否定'
        rows.append(dict(blind_id=bid, verdict=verdict,
                         level='fine' if raw in FINE else 'merge',
                         prior_code=prior_code, prior_group=pg, judged_group=jg,
                         judged_text=raw, role=s.get('role', ''), source=s.get('src_label', '')))
    fp = os.path.join(outdir, 'verdicts_merge_%s.csv' % a.tag)
    with open(fp, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    from collections import Counter
    print('合并级判定 %s：%s' % (a.tag, dict(Counter(r['verdict'] for r in rows))))
    print('  其中细类级判读 %d 点' % sum(1 for r in rows if r['level'] == 'fine'))
    print('-> %s' % fp)


if __name__ == '__main__':
    main()
