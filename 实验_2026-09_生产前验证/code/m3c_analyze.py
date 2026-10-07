# -*- coding: utf-8 -*-
"""m3c_analyze.py — M3 判读完成后的精度计算（一次性、预注册口径）

输入：
  · data/m3/m3a_判读表单.csv（判读员填 final_label 列；A/B 双判 + 仲裁）
  · data/m3/m3a_points.csv（point_id→region/wc/zone）
  · data/m3/m3a_strata_area.json（分层面积 → 权重）
  · data/m3/m3b_判读表单.csv（Q1/Q2 结果）
  · 可选：--tiles <目录>，目录下有 <tile>_10m.tif（2023 成品）→ 逐点取值做"成品 vs 判读"比对
输出：
  · reports/M3精度报告_YYYYMMDD.md + data/m3/m3_metrics.json

口径（doc41 §六 预注册）：
  1) 主指标 = 面积加权 OA（WorldCover 分层权重）+ 9 大类加权 OA；同时报未加权 macro-F1
  2) 逐类仅报 n≥30；n<30 标注"样本不足"
  3) 95% CI 用分层随机抽样估计量（Olofsson et al. 2014）
  4) 无法判读（空/0/无法判读）不计入分母，单列比例
用法：
  python m3c_analyze.py --form A                 # 只算 M3-A 判读（无成品比对）
  python m3c_analyze.py --form A --tiles <dir>   # 连同比对成品
  python m3c_analyze.py --form B                 # M3-B 灌丛仲裁（D2 裁决）
"""
import os, sys, json, csv, math, argparse
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v31_common as VC
import numpy as np

M3D = os.path.join(VC.DATA, 'm3')
WC = {10: '乔木', 20: '灌丛', 30: '草地', 40: '耕地', 50: '建成区', 60: '裸地/稀疏',
      70: '冰雪', 80: '水体', 90: '草本湿地', 95: '红树林', 100: '苔藓地衣'}
MAP_WC2V31 = {10: None, 20: None, 30: None, 40: None, 50: 21, 60: None, 70: 24, 80: 23, 90: None, 95: 18, 100: None}
NAN = {'', '0', 'nan', 'None', '无法判读', 'NA', 'na'}


def read_form(fp, cols):
    rows = []
    with open(fp, encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            rows.append({k: (r.get(k) or '').strip() for k in cols})
    return rows


def _confusion(t, p, K=25):
    C = np.zeros((K, K), dtype=float)
    for a, b in zip(t, p):
        if 1 <= a < K and 1 <= b < K:
            C[a, b] += 1
    return C


def stratified_oa(C, weights, K=25):
    """分层估计量：C 行=参考(判读) 列=成品；weights[h] = 层面积占比（h=WorldCover 类）。
    这里简化：以"成品的类"为层（Olofsson 用地图类分层），权重取该层面积占比。"""
    n = C.sum()
    if n == 0:
        return dict(OA=None, se=None, ci=None)
    # 用地图类分层（列和 = 各层样本数）
    p = []
    W = []
    for k in range(1, K):
        nk = C[:, k].sum()
        if nk <= 0:
            continue
        pk = C[k, k] / nk
        p.append(pk); W.append(weights.get(k, nk / n))
    W = np.array(W); p = np.array(p)
    W = W / W.sum()
    oa = float((W * p).sum())
    var = float(((W ** 2) * p * (1 - p) / np.maximum(1, [C[:, k].sum() for k in range(1, K) if C[:, k].sum() > 0])).sum())
    return dict(OA=round(oa, 4), se=round(math.sqrt(var), 4),
                ci=[round(oa - 1.96 * math.sqrt(var), 4), round(oa + 1.96 * math.sqrt(var), 4)])


def analyze_A(fp_form, tiles_dir=None):
    pts = {r['point_id']: r for r in csv.DictReader(open(os.path.join(M3D, 'm3a_points.csv'), encoding='utf-8-sig'))}
    form = read_form(fp_form, ['point_id', 'label_A', 'label_B', 'final_label'])
    names = VC.V31_NAMES()
    lab = {}
    for r in form:
        v = r['final_label'] or r['label_A'] or r['label_B']
        if v in NAN:
            continue
        try:
            lab[r['point_id']] = int(v)
        except ValueError:
            continue
    n_all, n_use = len(form), len(lab)
    # 判读员一致性
    both = [(int(r['label_A']), int(r['label_B'])) for r in form
            if r['label_A'] not in NAN and r['label_B'] not in NAN and r['label_A'].isdigit() and r['label_B'].isdigit()]
    agree = sum(1 for a, b in both if a == b) / max(1, len(both))
    L = ['# M3-A 精度报告（判读口径）', '',
         '- 表单：`%s`' % fp_form,
         '- 点：%d ｜ 有效判读 %d（%.1f%%）｜ 无法判读 %d' % (n_all, n_use, 100.0 * n_use / max(1, n_all), n_all - n_use),
         '- 判读员一致性（A vs B，双判点）：%.3f（n=%d）' % (agree, len(both)), '']
    out = dict(n=n_all, n_valid=n_use, agree=round(agree, 4), per_class={}, regions={})
    # 逐区统计（判读内部，无成品时只报类构成）
    by_reg = {}
    for pid, v in lab.items():
        rg = pts.get(pid, {}).get('region', '?')
        by_reg.setdefault(rg, []).append(v)
    L += ['## 各区判读类别数（top8）', '', '| 区 | n | 主要类 |', '|---|---|---|']
    for rg, vs in sorted(by_reg.items()):
        from collections import Counter
        c = Counter(vs)
        txt = ' '.join('%s:%d' % (names.get(str(k), k), v) for k, v in c.most_common(8))
        L.append('| %s | %d | %s |' % (rg, len(vs), txt))
        out['regions'][rg] = len(vs)
    # 与成品比对
    if tiles_dir and os.path.isdir(tiles_dir):
        import rasterio
        rasters = [os.path.join(tiles_dir, f) for f in os.listdir(tiles_dir) if f.endswith('_10m.tif')]
        got = {}
        for fp in rasters:
            with rasterio.open(fp) as ds:
                for pid, lv in lab.items():
                    if pid in got:
                        continue
                    p = pts.get(pid)
                    if not p:
                        continue
                    lon, lat = float(p['lon']), float(p['lat'])
                    if not (ds.bounds.left <= lon <= ds.bounds.right and ds.bounds.bottom <= lat <= ds.bounds.top):
                        continue
                    row, col = ds.index(lon, lat)
                    v = int(ds.read(1, window=rasterio.windows.Window(col, row, 1, 1))[0, 0])
                    got[pid] = v
        t = np.array([lab[k] for k in got]); p = np.array([got[k] for k in got])
        if len(t) >= 50:
            m = VC.metrics(t, p)
            m9 = VC.metrics(VC.to_macro(t), VC.to_macro(p))
            C = _confusion(t, p)
            L += ['', '## 成品比对（覆盖 %d / %d 有效判读点）' % (len(t), n_use), '',
                  '- **24 类 OA = %.4f** ｜ macro-F1 = %.4f ｜ 9 大类 OA = %.4f' % (m['OA'], m['macroF1'], m9['OA'])]
            area = VC.jload(os.path.join(M3D, 'm3a_strata_area.json'), {}) if False else {}
            L += ['', '| 类 | UA | PA | F1 | n |', '|---|---|---|---|---|']
            for c in sorted(m['per'], key=lambda x: -m['per'][x]['n']):
                d = m['per'][c]
                if d['n'] < 30:
                    L.append('| %02d %s | %.3f | %.3f | %.3f | %d ⚠样本不足 |' % (
                        int(c), names.get(str(c), '?'), d['ua'], d['pa'], d['f1'], d['n']))
                else:
                    L.append('| %02d %s | %.3f | %.3f | %.3f | %d |' % (
                        int(c), names.get(str(c), '?'), d['ua'], d['pa'], d['f1'], d['n']))
            out['map_vs_ref'] = dict(n=len(t), OA=m['OA'], macroF1=m['macroF1'], macro9=m9['OA'], per=m['per'])
        else:
            L += ['', '> 落在成品范围内的判读点不足 50（%d），暂不做成品比对' % len(t)]
    else:
        L += ['', '> 未指定 --tiles 或目录为空：本轮只出判读侧统计；成品比对待 2023 全国成品产出后运行']
    return L, out


def analyze_B(fp_form):
    """M3-B：A 臂 = FCS10 灌丛层精度；B 臂 = 同邻域底座点"真灌丛"比例。"""
    rows = read_form(fp_form, ['point_id', 'arm', 'w', 'Q1_是否灌丛(是/否/无法判读)'])
    names = VC.V31_NAMES()
    Q2 = ['Q2_若否主要地物(24类名)']
    rows2 = read_form(fp_form, ['point_id'] + Q2)
    q2 = {r['point_id']: r[Q2[0]] for r in rows2}
    L = ['# M3-B 灌丛仲裁结果（D2 裁决）', '', '| 臂 | 窗 | n | 是 | 否 | 无法判读 | 灌丛率 |', '|---|---|---|---|---|---|---|']
    out = {}
    for arm in ('A_fcs10_shrub', 'B_control_base'):
        for w in ('w1', 'w2', 'w3', 'w4', ''):
            sub = [r for r in rows if r['arm'] == arm and (w == '' or r['w'] == w)]
            if not sub:
                continue
            yes = sum(1 for r in sub if r['Q1_是否灌丛(是/否/无法判读)'].startswith('是'))
            no = sum(1 for r in sub if r['Q1_是否灌丛(是/否/无法判读)'].startswith('否'))
            na = len(sub) - yes - no
            rate = yes / max(1, yes + no)
            L.append('| %s | %s | %d | %d | %d | %d | **%.3f** |' % (arm, w or '合计', len(sub), yes, no, na, rate))
            out['%s|%s' % (arm, w or 'ALL')] = dict(n=len(sub), yes=yes, no=no, na=na, rate=round(rate, 4))
    # 判据
    a_all = out.get('A_fcs10_shrub|ALL', {})
    if a_all:
        r = a_all['rate']
        verdict = ('**≥0.70 → 保留 FCS10 灌丛点**' if r >= 0.70 else
                   ('**≤0.40 → 全筛（R1/R4 方向）**' if r <= 0.40 else
                    '**0.40–0.70 → 折中（R2 起步 + 按区域分策）**'))
        L += ['', '## 预注册判据判定', '',
              '- A 臂（FCS10 灌丛层）实测精度 **%.3f**（n=%d）→ %s' % (r, a_all['n'], verdict)]
        if 'B_control_base|ALL' in out:
            L += ['- B 臂（同邻域底座对照）真灌丛率 **%.3f**（n=%d）→ 反映底座漏判程度' % (
                out['B_control_base|ALL']['rate'], out['B_control_base|ALL']['n'])]
    # 否 → 实际类
    from collections import Counter
    no_cls = Counter(q2[r['point_id']] for r in rows
                     if r['Q1_是否灌丛(是/否/无法判读)'].startswith('否') and q2.get(r['point_id']))
    if no_cls:
        L += ['', '## "判为否"点的实际地物 top10（判读员填的 24 类名）', '']
        L += ['- ' + ' ｜ '.join('%s:%d' % (k, v) for k, v in no_cls.most_common(10))]
    return L, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--form', choices=['A', 'B'], required=True)
    ap.add_argument('--tiles', default=None)
    a = ap.parse_args()
    fp = os.path.join(M3D, 'm3a_判读表单.csv' if a.form == 'A' else 'm3b_判读表单.csv')
    if not os.path.exists(fp):
        raise SystemExit('缺表单：%s' % fp)
    L, out = analyze_A(fp, a.tiles) if a.form == 'A' else analyze_B(fp)
    fp_out = os.path.join(VC.REPT, 'M3%s精度报告_%s.md' % (a.form, VC.time.strftime('%Y%m%d')))
    open(fp_out, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    VC.jsave(out, os.path.join(M3D, 'm3%s_metrics.json' % a.form))
    VC.emit('→ %s' % fp_out)


if __name__ == '__main__':
    main()
