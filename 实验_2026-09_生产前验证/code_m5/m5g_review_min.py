# -*- coding: utf-8 -*-
"""m5g_review_min.py — M3-B 最小人工复核清单生成 + 判后合并（D2 裁决用）

* 输入：data/m3/m3b_判读表单_材料版.csv（863 点材料版：切片 + 11 指标 + 机械提示）
        data/m3/m3b_ai_labels.csv（AI 第一遍预判读：arm/w/label/conf/note）
        data/m3/ai_batches/out_reliab_0.csv、out_reliab_1.csv（96 点复判第二遍）
        数据/边界/china_100000_full.json（DataV 省界，干湿分区：新青藏甘宁蒙=干旱）
* 规则源：技术文档/47 §五 最小复核方案（4 组约 250 点）：
        G1 复判不一致 6 + 无法判读 28；G2 低置信"是"48；G3 干旱 A 臂随机 60；
        G4 每窗随机"否"各 20（w1–w4，共 80）。组间去重按 G1>G2>G3>G4 优先。
* 门槛：抽样固定种子 20261008（可复现）；只读不改原表单（材料版与 AI 版均不动）。
* 输出：data/m3/m3b_最小复核清单.csv（可直接判读，列与材料版同构 + 组别/理由/AI参考）
        make 模式统计打印（各组点数、去重后总数、A 臂当前率）
        merge 模式：把判完的清单并回 863 点 → data/m3/m3b_判读表单_人工复核合并.csv
* 用法：python m5g_review_min.py make
        python m5g_review_min.py merge [--filled <判完的清单>] [--out <合并表单>]
        判完后：python m3c_analyze.py --form B --form-file data/m3/m3b_判读表单_人工复核合并.csv
幂等：输出已存在时 make 默认跳过（--force 重写）；merge 每次覆盖输出。
"""
import os
import sys
import csv
import glob
import json
import time
import random

sys.stdout.reconfigure(encoding='utf-8')

M3D = r'F:/lc_work/v31_exp/data/m3'
FORM_MAT = os.path.join(M3D, 'm3b_判读表单_材料版.csv')
AI_LAB = os.path.join(M3D, 'm3b_ai_labels.csv')
RELIAB = os.path.join(M3D, 'ai_batches', 'out_reliab_*.csv')
FORM_OUT = os.path.join(M3D, 'm3b_最小复核清单.csv')
MERGED_OUT = os.path.join(M3D, 'm3b_判读表单_人工复核合并.csv')
BOUNDARY = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/边界/china_100000_full.json'
DRY = ('新疆', '西藏', '青海', '甘肃', '宁夏', '内蒙古')
SEED = 20261008
Q1 = 'Q1_是否灌丛(是/否/无法判读)'


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def read_csv(fp):
    with open(fp, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(fp, rows, cols):
    with open(fp, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, '') for c in cols})


def tag_dry(rows):
    """DataV 省界点内多边形判干/湿（与 m5_build_points.tag_dry 同规则）。"""
    from shapely.geometry import shape, Point
    from shapely.prepared import prep
    gj = json.load(open(BOUNDARY, encoding='utf-8'))
    polys = []
    for ft in gj['features']:
        name = ft.get('properties', {}).get('name', '')
        if name[:2] in DRY or name[:3] in DRY:
            g = shape(ft['geometry'])
            if not g.is_valid:
                from shapely import make_valid
                g = make_valid(g)
            polys.append(prep(g))
    for r in rows:
        p = Point(float(r['lon']), float(r['lat']))
        r['dry'] = any(g.contains(p) for g in polys)
    return rows


def make(force=False):
    if os.path.exists(FORM_OUT) and not force:
        emit('已存在 %s（--force 重写）' % FORM_OUT)
        return
    mat = read_csv(FORM_MAT)
    ai = {r['point_id']: r for r in read_csv(AI_LAB)}
    rel = {}
    for fp in sorted(glob.glob(RELIAB)):
        for r in read_csv(fp):
            rel.setdefault(r['point_id'], []).append(r)
    mat = tag_dry(mat)
    n_dry_a = sum(1 for r in mat if r['dry'] and ai[r['point_id']]['arm'].startswith('A'))
    emit('干旱 A 臂点数 = %d' % n_dry_a)

    rnd = random.Random(SEED)
    groups = {}

    # G1：复判不一致 + 无法判读
    g1 = set()
    for pid, rr in rel.items():
        if any(x['label'] != ai[pid]['label'] for x in rr):
            g1.add((pid, '复判不一致：主判=%s/置信%s，复判=%s' % (
                ai[pid]['label'], ai[pid]['conf'], '/'.join(x['label'] for x in rr))))
    for pid, r in ai.items():
        if r['label'] == '无法判读':
            g1.add((pid, 'AI 无法判读'))
    # G2：低置信"是"
    g2 = [(pid, '低置信"是"（决定分母）') for pid, r in ai.items()
          if r['label'] == '是' and r['conf'] == '低']
    # G3：干旱 A 臂随机 60
    dry_a = [r['point_id'] for r in mat
             if r['dry'] and ai[r['point_id']]['arm'].startswith('A')]
    g3 = [(pid, '干旱 A 臂随机 60（AI 最可能低估）') for pid in rnd.sample(dry_a, min(60, len(dry_a)))]
    # G4：每窗随机"否"各 20
    g4 = []
    for w in ('w1', 'w2', 'w3', 'w4'):
        pool = [pid for pid, r in ai.items() if r['label'] == '否' and r['w'] == w]
        g4 += [(pid, '每窗随机"否"审计（%s）' % w) for pid in rnd.sample(pool, min(20, len(pool)))]

    for tag, lst in (('G1_不确定', g1), ('G2_低置信是', g2), ('G3_干旱A', g3), ('G4_随机否', g4)):
        for pid, why in lst:
            groups.setdefault(pid, (tag, why))

    cols_in = list(mat[0].keys())
    cols_out = (['point_id', '组别', '复核理由', 'AI判读', 'AI置信', 'AI备注']
                + [c for c in cols_in if c != 'point_id'] + ['dry'])
    rows = []
    for r in mat:
        pid = r['point_id']
        if pid not in groups:
            continue
        tag, why = groups[pid]
        o = dict(r)
        o['组别'] = tag
        o['复核理由'] = why
        o['AI判读'] = ai[pid]['label']
        o['AI置信'] = ai[pid]['conf']
        o['AI备注'] = ai[pid]['note']
        o['dry'] = '是' if r['dry'] else ''
        rows.append(o)
    order = {'G1_不确定': 0, 'G2_低置信是': 1, 'G3_干旱A': 2, 'G4_随机否': 3}
    rows.sort(key=lambda r: (order[r['组别']], r['w'], r['point_id']))
    write_csv(FORM_OUT, rows, cols_out)
    from collections import Counter
    c = Counter(r['组别'] for r in rows)
    a = [r for r in rows if r['arm'].startswith('A')]
    a_yes = sum(1 for r in a if r['AI判读'] == '是')
    emit('写出 %s：%d 点（%s）' % (FORM_OUT, len(rows), dict(c)))
    emit('  其中 A 臂 %d，AI 判"是" %d（该子集率 %.3f；全 A 臂率 0.107）' % (
        len(a), a_yes, a_yes / max(1, len(a))))


def merge(filled=None, out=MERGED_OUT):
    filled = filled or FORM_OUT
    base = read_csv(FORM_MAT)
    ai = {r['point_id']: r for r in read_csv(AI_LAB)}
    for r in base:  # 先把 AI 判读填入全表（人工复核点稍后覆盖）
        r[Q1] = ai[r['point_id']]['label']
        if not r.get('Q2_若否主要地物(24类名)'):
            r['Q2_若否主要地物(24类名)'] = ai[r['point_id']]['note'] if ai[r['point_id']]['label'] == '否' else ''
        r['判读员'] = 'AI预判读(2026-10-08)'
    n_h = 0
    for r in read_csv(filled):
        if not r.get(Q1, '').strip():
            continue
        j = ai[r['point_id']]
        base_by = {b['point_id']: b for b in base}[r['point_id']]
        base_by[Q1] = r[Q1].strip()
        if r.get('Q2_若否主要地物(24类名)', '').strip():
            base_by['Q2_若否主要地物(24类名)'] = r['Q2_若否主要地物(24类名)'].strip()
        if r.get('备注', '').strip():
            base_by['备注'] = r['备注'].strip()
        base_by['判读员'] = r.get('判读员', '').strip() or '人工复核'
        base_by['复核员'] = r.get('复核员', '').strip()
        n_h += 1
    cols = list(base[0].keys())
    write_csv(out, base, cols)
    emit('人工覆盖 %d 点 → 合并表单 %s（%d 行，可直接喂 m3c_analyze --form B）' % (n_h, out, len(base)))
    from collections import Counter
    a = [r for r in base if r['arm'].startswith('A')]
    yes = sum(1 for r in a if r[Q1].startswith('是'))
    no = sum(1 for r in a if r[Q1].startswith('否'))
    emit('  A 臂合并后：n=%d 是=%d 否=%d 率=%.3f' % (len(a), yes, no, yes / max(1, yes + no)))


if __name__ == '__main__':
    a = sys.argv[1:]
    if not a or a[0] == 'make':
        make(force='--force' in a)
    elif a[0] == 'merge':
        f = a[a.index('--filled') + 1] if '--filled' in a else None
        o = a[a.index('--out') + 1] if '--out' in a else MERGED_OUT
        merge(f, o)
    else:
        raise SystemExit(__doc__)
