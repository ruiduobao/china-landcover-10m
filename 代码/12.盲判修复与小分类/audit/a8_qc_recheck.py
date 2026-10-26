# -*- coding: utf-8 -*-
"""a8_qc_recheck.py — 复判一致率（三态，未知独立成态）+ 否定审计准确率

* 输入：--pass1 首判表（含 blind_id/实际类别/置信）、--recheck 复判表（同列）、
        --verdicts 首判合并级判定（a7 输出）、--ledger 抽样底账（a2/本地生成）
* 口径（doc72 §5）：三态 = 结构组一致 / 不一致 / 任一方无法判定(未知)，未知独立成态；
        实际类别一致率（仅双方均可判）另报；否定审计按首判=否定的点核"是否其实为确认"。
* 门槛：三态一致率 ≥0.90、否定审计准确率 ≥0.90；不达标扩大复判。
* 输出：qc/qc_report.json + 文本摘要
* 用法：python a8_qc_recheck.py --pass1 qc/pass1_norm.csv --recheck qc/recheck/recheck_1_b0102.csv [更多]
"""
import argparse, csv, glob, json, os, sys
sys.stdout.reconfigure(encoding='utf-8')

# 与 a7 相同的组映射
CODE2GROUP = {}
for c in (1, 3): CODE2GROUP[c] = 'crop'
for c in (4, 5, 6, 7, 8): CODE2GROUP[c] = 'forest'
for c in (9, 10): CODE2GROUP[c] = 'shrub'
CODE2GROUP[11] = 'grass'; CODE2GROUP[2] = 'orchard'; CODE2GROUP[12] = 'lichen'
CODE2GROUP[13] = 'sparse'; CODE2GROUP[14] = 'wetland'; CODE2GROUP[15] = 'wetland'
for c in (16, 17, 18, 19, 20): CODE2GROUP[c] = 'coastal_wet'
CODE2GROUP[21] = 'built'; CODE2GROUP[22] = 'bare'; CODE2GROUP[23] = 'water'; CODE2GROUP[24] = 'snow'

TEXT2GROUP = {
    '乔木林': 'forest', '阔叶林（常绿/落叶不可分）': 'forest', '常绿阔叶林': 'forest',
    '落叶阔叶林': 'forest', '常绿针叶林': 'forest', '落叶针叶林': 'forest', '针阔混交林': 'forest',
    '灌丛': 'shrub', '灌木/草地': 'shrub', '常绿灌木': 'shrub', '落叶灌木': 'shrub',
    '草地': 'grass', '草本旱地': 'crop', '灌溉耕地': 'crop', '耕地': 'crop',
    '乔灌园地': 'orchard', '园地': 'orchard', '地衣苔藓': 'lichen', '稀疏植被': 'sparse',
    '木本沼泽': 'wetland', '草本沼泽': 'wetland',
    '湖河滩地': 'coastal_wet', '盐渍湿地': 'coastal_wet', '红树林': 'coastal_wet',
    '海岸盐沼': 'coastal_wet', '潮滩': 'coastal_wet',
    '不透水面': 'built', '裸地': 'bare', '水体': 'water', '永久冰雪': 'snow',
}
FINE = {'常绿阔叶林', '落叶阔叶林', '常绿针叶林', '落叶针叶林', '针阔混交林',
        '草本旱地', '灌溉耕地', '乔灌园地', '常绿灌木', '落叶灌木', '水体', '裸地',
        '稀疏植被', '草地', '不透水面', '永久冰雪', '木本沼泽', '草本沼泽'}


def load(paths):
    rows = {}
    for p in paths:
        for pat in ([p] if not any(ch in p for ch in '*?') else glob.glob(p)):
            for r in csv.DictReader(open(pat, encoding='utf-8-sig')):
                bid = str(r.get('blind_id') or r.get('point_id') or '').strip()
                if bid:
                    rows[bid] = r
    return rows


def grp(r):
    txt = str(r.get('实际类别', '') or '').strip()
    if not txt or txt.startswith('无法'):
        return 'unknown', txt
    return TEXT2GROUP.get(txt, 'unknown'), txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ws', default=r'F:/lc_work/label_repair_2023_20261015')
    ap.add_argument('--pass1', required=True)
    ap.add_argument('--recheck', nargs='+', required=True)
    ap.add_argument('--verdicts', default=None)
    ap.add_argument('--outdir', default=None)
    a = ap.parse_args()

    p1 = load([a.pass1]); rc = load(a.recheck)
    print('首判 %d 点；复判 %d 点；交集 %d' % (len(p1), len(rc), len(set(p1) & set(rc))))
    ver = {}
    if a.verdicts and os.path.exists(a.verdicts):
        for r in csv.DictReader(open(a.verdicts, encoding='utf-8-sig')):
            ver[r['blind_id']] = r['verdict']

    n_same = n_diff = n_unknown = 0; fine_n = fine_same = 0
    audit_neg = audit_neg_ok = 0
    disagree = []
    for bid in sorted(set(p1) & set(rc)):
        g1, t1 = grp(p1[bid]); g2, t2 = grp(rc[bid])
        if g1 == 'unknown' or g2 == 'unknown':
            n_unknown += 1
        elif g1 == g2:
            n_same += 1
        else:
            n_diff += 1
            disagree.append(dict(blind_id=bid, pass1=t1, recheck=t2))
        if t1 in FINE and t2 in FINE:
            fine_n += 1
            if t1 == t2:
                fine_same += 1
        if ver.get(bid) == '否定':
            audit_neg += 1
            if g1 == g2:
                audit_neg_ok += 1

    denom = n_same + n_diff
    tri = round(n_same / denom, 4) if denom else None
    fine = round(fine_same / fine_n, 4) if fine_n else None
    neg_acc = round(audit_neg_ok / audit_neg, 4) if audit_neg else None
    res = dict(n_pass1=len(p1), n_recheck=len(rc), n_common=len(set(p1) & set(rc)),
               three_state=dict(same=n_same, diff=n_diff, unknown=n_unknown,
                                rate=tri, gate=0.90, pass_=bool(tri is not None and tri >= 0.90)),
               fine_agreement=dict(n=fine_n, same=fine_same, rate=fine),
               negation_audit=dict(n=audit_neg, ok=audit_neg_ok, rate=neg_acc,
                                   gate=0.90, pass_=bool(neg_acc is not None and neg_acc >= 0.90)),
               unknown_share=round(n_unknown / max(1, len(set(p1) & set(rc))), 4),
               disagreements=disagree[:50])
    outdir = a.outdir or os.path.join(a.ws, 'qc')
    os.makedirs(outdir, exist_ok=True)
    fp = os.path.join(outdir, 'qc_report.json')
    json.dump(res, open(fp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('三态一致率 %s（门0.90，%s）；一致 %d / 分歧 %d / 未知 %d'
          % (tri, 'PASS' if res['three_state']['pass_'] else 'FAIL', n_same, n_diff, n_unknown))
    print('细类一致率 %s（n=%d）' % (fine, fine_n))
    print('否定审计准确率 %s（n=%d）' % (neg_acc, audit_neg))
    if disagree:
        print('分歧样例：', disagree[:5])
    print('-> %s' % fp)


if __name__ == '__main__':
    main()
