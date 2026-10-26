# -*- coding: utf-8 -*-
"""t1b_sichuan_stats.py — 川 records_川.npz 统计：类码×坡向/坡度/高程、两版对照、山阴/日照、受控合并

* 输入：terrain/records_川.npz（t1 采样：每像素 二版类码/一版类码/elev/slope/asp/tile/lon/lat）
* 规则源：v31_map.json 24 类码；阳坡=asp∈[135,225]，阴坡=asp∈[315,360)∪[0,45]（与 2026-10-09 aspect_check 同口径）
    日照代理 hillshade=cos(zen)cos(s)+sin(zen)sin(s)cos(sun_az−asp)；方位扫描找最优 θ*
* 门槛：无（诊断）；分层控制 = 瓦×高程带×坡度带（Mantel–Haenszel 合并 OR）
* 输出：tables_川.json + CSV（逐类坡向/OR 分层/两版转移/方位扫描/AUC）
* 用法：python t1b_sichuan_stats.py
"""
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
OUTD = r'F:/lc_work/prod2nd_2023/reports/quality_review_2026-10-15/terrain'
NAME = {1: '草本旱地', 2: '乔灌园地', 3: '灌溉耕地', 4: '常绿阔叶', 5: '落叶阔叶', 6: '常绿针叶',
        7: '落叶针叶', 8: '针阔混交', 9: '常绿灌丛', 10: '落叶灌丛', 11: '草地', 12: '地衣苔藓',
        13: '稀疏植被', 14: '木本沼泽', 15: '草本沼泽', 16: '湖河滩地', 17: '盐渍湿地', 18: '红树林',
        19: '海岸盐沼', 20: '潮滩', 21: '不透水面', 22: '裸地', 23: '水体', 24: '永久冰雪'}
ELEV_EDGES = [0, 500, 1000, 1500, 2500, 3500, 4500, 9999]
SLOPE_EDGES = [0, 5, 15, 25, 90]
SECT = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']


def sun_shady(asp):
    sunny = (asp >= 135) & (asp <= 225)
    shady = (asp <= 45) | (asp >= 315)
    return sunny, shady


def odds_ci(a, b, c, d):
    """OR=(a/b)/(c/d)，Haldane 0.5 修正；返回 OR 与 95%CI。"""
    a, b, c, d = float(a), float(b), float(c), float(d)
    if min(a, b, c, d) == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    orr = (a * d) / (b * c)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return float(orr), (float(orr * np.exp(-1.96 * se)), float(orr * np.exp(1.96 * se)))


def mh_or(strata):
    """strata: list of (a,b,c,d)=shady/sunny × clsA/clsB。返回 MH OR + RBG 95%CI。"""
    R = S = PR = QR = 0.0
    for a, b, c, d in strata:
        n = a + b + c + d
        if n < 20 or min(a + b, c + d) == 0:
            continue
        R += a * d / n
        S += b * c / n
        P = (a + d) / n
        Q = (b + c) / n
        PR += P * a * d / n
        QR += Q * b * c / n
    if R == 0 or S == 0:
        return None, None, 0
    orr = R / S
    var = PR / (2 * R * R) + (PR + QR) / (2 * R * S) + QR / (2 * S * S)
    se = np.sqrt(var)
    return float(orr), (float(orr * np.exp(-1.96 * se)), float(orr * np.exp(1.96 * se))), 1


def auc_mw(score, pos):
    """Mann–Whitney AUC of score for separating pos=1 vs 0."""
    r = np.argsort(np.argsort(score)) + 1.0
    n1 = pos.sum()
    n0 = len(pos) - n1
    r1 = r[pos].sum()
    return float((r1 - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    z = np.load(os.path.join(OUTD, 'records_川.npz'))
    cls2 = z['cls2'].astype(np.int16)
    cls1 = z['cls1'].astype(np.int16)
    elev = z['elev']
    slope = z['slope']
    asp = z['asp']
    tile = z['tile']
    lon, lat = z['lon'], z['lat']
    n = cls2.size
    out = {'n_records': int(n), 'sampling': '森林类40%子采样/参照类6%子采样，条件分布无偏',
           'coverage_note': '窗读像素约占省域类码像元1%，非面积随机概率样本，不声称代表全省逐像元',
           'OR_def': 'OR = [ENF阴坡/ENF阳坡] / [EBF阴坡/EBF阳坡]，OR>1 表示常绿针叶相对常绿阔叶更偏阴坡',
           'ci_note': '像元级 CI 因空间伪重复过窄，仅作描述；主报告用 0.5°块 block-bootstrap CI',
           'asp_bins': SECT, 'sunny_def': '[135,225]', 'shady_def': '[315,360)U[0,45]'}
    sunny, shady = sun_shady(asp)
    aspbin = ((asp + 22.5) % 360 // 45).astype(np.int8)
    eb = np.digitize(elev, ELEV_EDGES[1:-1])
    sb = np.digitize(slope, SLOPE_EDGES[1:-1])
    print('records=%.2fM  slope>5:%.2fM  slope>15:%.2fM' % (n / 1e6, (slope > 5).sum() / 1e6, (slope > 15).sum() / 1e6))

    # ---------- S1 逐类坡向（二版），slope>5 与 slope>15 两口径 ----------
    rows = []
    for c in [1, 4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23]:
        for smin, tag in ((5, 'slope>5'), (15, 'slope>15')):
            m = (cls2 == c) & (slope > smin)
            k = int(m.sum())
            if k < 200:
                continue
            su = int((m & sunny).sum())
            sh = int((m & shady).sum())
            hist = [int(((m & (aspbin == i)).sum())) for i in range(8)]
            rows.append({'class': int(c), 'name': NAME.get(int(c)), 'subset': tag, 'n': k,
                         'sunny_pct': round(100 * su / k, 2), 'shady_pct': round(100 * sh / k, 2),
                         'sunny_shady_ratio': round(su / max(sh, 1), 3),
                         'aspect_8bin_pct': [round(100 * h / k, 2) for h in hist]})
    out['per_class_aspect_2nd'] = rows
    print('\n== S1 二版逐类 阳坡/阴坡（sunny/shady_ratio）==')
    for r in rows:
        print('  cls%2d %s %s n=%d sunny=%.1f%% shady=%.1f%% ratio=%.2f' %
              (r['class'], r['name'], r['subset'], r['n'], r['sunny_pct'], r['shady_pct'], r['sunny_shady_ratio']))

    # ---------- S2 ENF(6) vs EBF(4) 受控对照 ----------
    def contrast(mask_sel, tag, run='2nd'):
        cA, cB = (6, 4)
        m = mask_sel & np.isin(cls2 if run == '2nd' else cls1, (cA, cB))
        A = m & ((cls2 if run == '2nd' else cls1) == cA)
        B = m & ((cls2 if run == '2nd' else cls1) == cB)
        a_sh, a_su = int((A & shady).sum()), int((A & sunny).sum())
        b_sh, b_su = int((B & shady).sum()), int((B & sunny).sum())
        orr, ci = odds_ci(a_sh, a_su, b_sh, b_su)
        return {'tag': tag, 'run': run, 'n_ENF': int(A.sum()), 'n_EBF': int(B.sum()),
                'shady_pct_6': round(100 * a_sh / max(A.sum(), 1), 2),
                'sunny_pct_6': round(100 * a_su / max(A.sum(), 1), 2),
                'shady_pct_4': round(100 * b_sh / max(B.sum(), 1), 2),
                'sunny_pct_4': round(100 * b_su / max(B.sum(), 1), 2),
                'OR_6vs4': round(orr, 3), 'OR_95CI': [round(ci[0], 3), round(ci[1], 3)]}
    S2 = [contrast(slope > 5, 'slope>5'), contrast(slope > 15, 'slope>15'),
          contrast((slope > 15) & (elev > 2500), 'slope>15 & elev>2500')]
    out['S2_ENF_vs_EBF_controlled'] = S2
    print('\n== S2 ENF(6) vs EBF(4) 阳/阴坡 OR（OR=[ENF阴/ENF阳]/[EBF阴/EBF阳]，>1=针叶偏阴坡）==')
    for r in S2:
        print('  %-22s OR=%.2f CI=%s (6: shady %.1f%% n=%d | 4: shady %.1f%% n=%d)' %
              (r['tag'], r['OR_6vs4'], r['OR_95CI'], r['shady_pct_6'], r['n_ENF'],
               r['shady_pct_4'], r['n_EBF']))

    # 分层：瓦 × 高程带 × 坡度带 的 MH 合并
    sel = (slope > 5) & np.isin(cls2, (4, 6))
    strata = []
    for t in np.unique(tile):
        for e in np.unique(eb):
            for s in np.unique(sb):
                m = sel & (tile == t) & (eb == e) & (sb == s)
                if m.sum() < 50:
                    continue
                a = int((m & (cls2 == 6) & shady).sum())
                b = int((m & (cls2 == 6) & sunny).sum())
                c = int((m & (cls2 == 4) & shady).sum())
                d = int((m & (cls2 == 4) & sunny).sum())
                strata.append((a, b, c, d))
    mh, mhci, ok = mh_or(strata)
    out['S2b_MH_pooled'] = {'strata': len(strata), 'OR_MH': mh, 'OR_95CI': mhci}
    print('  MH 合并（瓦×高程×坡度,%d 层）: OR=%.2f CI=%s' % (len(strata), mh, mhci))

    # 分高程带
    per_e = []
    for e in np.unique(eb):
        m = sel & (eb == e)
        if m.sum() < 2000:
            continue
        a = int((m & (cls2 == 6) & shady).sum()); b = int((m & (cls2 == 6) & sunny).sum())
        c = int((m & (cls2 == 4) & shady).sum()); d = int((m & (cls2 == 4) & sunny).sum())
        orr, ci = odds_ci(a, b, c, d)
        per_e.append({'elev_band': [ELEV_EDGES[int(e)], ELEV_EDGES[int(e) + 1]], 'OR': round(orr, 3),
                      'CI': [round(ci[0], 2), round(ci[1], 2)],
                      'n6': a + b, 'n4': c + d,
                      'shady_pct6': round(100 * a / max(a + b, 1), 1), 'shady_pct4': round(100 * c / max(c + d, 1), 1)})
    out['S2c_per_elev'] = per_e
    print('  分高程带:')
    for r in per_e:
        print('    %s m OR=%.2f %s  shady6=%.1f%% shady4=%.1f%%' % (r['elev_band'], r['OR'], r['CI'],
                                                                      r['shady_pct6'], r['shady_pct4']))
    # 分瓦 + 分 0.5°块
    per_t = []
    for t in np.unique(tile):
        m = sel & (tile == t)
        if m.sum() < 2000:
            continue
        a = int((m & (cls2 == 6) & shady).sum()); b = int((m & (cls2 == 6) & sunny).sum())
        c = int((m & (cls2 == 4) & shady).sum()); d = int((m & (cls2 == 4) & sunny).sum())
        orr, ci = odds_ci(a, b, c, d)
        per_t.append({'tile': int(t), 'OR': round(orr, 3), 'n6': a + b, 'n4': c + d})
    out['S2d_per_tile'] = per_t
    o = np.array([r['OR'] for r in per_t])
    print('  分瓦 n=%d，OR>1 瓦数=%d，OR 中位=%.2f，范围 %.2f–%.2f' %
          (len(per_t), int((o > 1).sum()), float(np.median(o)), o.min(), o.max()))

    # ---------- S3 方位扫描 ----------
    m6 = (cls2 == 6) & (slope > 5)
    m4 = (cls2 == 4) & (slope > 5)
    arcs = np.radians(asp)
    scan = []
    for th in range(0, 360, 5):
        t = np.radians(th)
        amp = float(np.mean(np.cos(arcs[m6] - t)) - np.mean(np.cos(arcs[m4] - t)))
        scan.append(amp)
    scan = np.array(scan)
    k = int(np.argmax(np.abs(scan)))
    th_best = 5 * k
    # 圆锥统计：ENF 与 EBF 的平均方向差
    def circmean(mask):
        return np.arctan2(np.mean(np.sin(arcs[mask])), np.mean(np.cos(arcs[mask])))
    dvec = np.array([np.mean(np.sin(arcs[m6])) - np.mean(np.sin(arcs[m4])),
                     np.mean(np.cos(arcs[m6])) - np.mean(np.cos(arcs[m4]))])
    R = float(np.hypot(*dvec))
    th_vec = float((np.degrees(np.arctan2(dvec[0], dvec[1]))) % 360)
    out['S3_azimuth'] = {'best_theta_deg': th_best, 'amplitude': round(float(scan[k]), 4),
                         'vector_diff_dir_deg': round(th_vec, 1), 'vector_R': round(R, 4),
                         'sun_candidates': {'Landsat/S2 ~10:30': [135, 165], 'noon': [180],
                                            'ecology_axis': [0, 180]}}
    print('\n== S3 方位扫描 ==')
    print('  最优方位 θ*=%.0f°（即 ENF 相对 EBF 在该方位余弦上更富集）amplitude=%.4f R=%.4f 向量差方向=%.0f°' %
          (th_best, scan[k], R, th_vec))

    # ---------- S4 hillshade / 单变量 AUC ----------
    def hs(sun_az, sun_el):
        z = np.radians(90 - sun_el)
        s = np.radians(slope)
        return np.cos(z) * np.cos(s) + np.sin(z) * np.sin(s) * np.cos(np.radians(sun_az) - arcs)
    sel2 = (slope > 5) & np.isin(cls2, (4, 6))
    pos = (cls2[sel2] == 6)
    S4 = {}
    for tag, score in (('hillshade_az135_el45', hs(135, 45)), ('hillshade_az160_el40', hs(160, 40)),
                       ('hillshade_az180_el50', hs(180, 50)), ('cos_asp_minus_180', np.cos(arcs - np.pi)),
                       ('cos_asp_minus_135', np.cos(arcs - np.radians(135))),
                       ('slope_deg', slope)):
        S4[tag] = round(auc_mw(score[sel2].astype(np.float64), pos), 4)
    out['S4_AUC_ENF_vs_EBF'] = S4
    print('\n== S4 单变量区分 ENF/EBF 的 AUC（0.5=无关）==')
    for k2, v in S4.items():
        print('  %-22s AUC=%.3f' % (k2, v))

    # ---------- S5 两版同点对照 ----------
    m_forest = np.isin(cls1, (4, 5, 6, 7, 8)) & np.isin(cls2, (4, 5, 6, 7, 8)) & (slope > 5)
    t12 = {}
    for a in (4, 5, 6):
        for b in (4, 5, 6):
            m = m_forest & (cls1 == a) & (cls2 == b)
            k2 = int(m.sum())
            if k2 < 100:
                continue
            t12['%d->%d' % (a, b)] = {'n': k2, 'shady_pct': round(100 * float((m & shady).sum()) / k2, 1),
                                      'sunny_pct': round(100 * float((m & sunny).sum()) / k2, 1),
                                      'mean_elev': int(elev[m].mean())}
    out['S5_transition_aspect'] = t12
    r1 = contrast(slope > 5, 'slope>5', run='1st')
    out['S5_run1_OR'] = r1
    print('\n== S5 两版对照 ==')
    print('  一版 OR(6vs4, slope>5)=%.2f CI=%s | 二版=%.2f' % (r1['OR_6vs4'], r1['OR_95CI'], S2[0]['OR_6vs4']))
    print('  转移（一版→二版；sunny/shady 为该转移组内占比，分母=该转移像元数）：')
    for k2, v in sorted(t12.items()):
        print('    %s n=%d sunny=%.1f%% shady=%.1f%% (其余为 E/W 向) elev~%d' % (k2, v['n'], v['sunny_pct'], v['shady_pct'], v['mean_elev']))

    # 一版逐类坡向（同口径）
    rows1 = []
    for c in [4, 5, 6]:
        smin = 5
        m = (cls1 == c) & (slope > smin)
        k2 = int(m.sum())
        if k2 < 200:
            continue
        su = int((m & sunny).sum()); sh = int((m & shady).sum())
        rows1.append({'class': c, 'n': k2, 'sunny_pct': round(100 * su / k2, 2), 'shady_pct': round(100 * sh / k2, 2)})
    out['S5_run1_per_class'] = rows1
    print('  一版逐类: ' + ' | '.join('cls%d shady=%.1f%% (n=%d)' % (r['class'], r['shady_pct'], r['n']) for r in rows1))

    # ---------- S6 0.5°块一致性 ----------
    blk = (np.floor(lon * 2).astype(np.int32) * 100000 + np.floor(lat * 2).astype(np.int32))
    per_b = []
    for b in np.unique(blk):
        m = sel & (blk == b)
        if m.sum() < 2000:
            continue
        a = int((m & (cls2 == 6) & shady).sum()); bb = int((m & (cls2 == 6) & sunny).sum())
        c = int((m & (cls2 == 4) & shady).sum()); d = int((m & (cls2 == 4) & sunny).sum())
        orr, _ = odds_ci(a, bb, c, d)
        per_b.append(orr)
    per_b = np.array(per_b)
    out['S6_blocks'] = {'n_blocks': int(per_b.size), 'frac_OR_gt1': float((per_b > 1).mean()),
                        'median_OR': float(np.median(per_b)),
                        'q25_q75': [float(np.percentile(per_b, 25)), float(np.percentile(per_b, 75))]}
    print('\n== S6 0.5°块 ==')
    print('  块数=%d，OR>1 占比=%.1f%%，中位 OR=%.2f，IQR=[%.2f, %.2f]' %
          (per_b.size, 100 * (per_b > 1).mean(), np.median(per_b),
           np.percentile(per_b, 25), np.percentile(per_b, 75)))

    # ---------- S6a2 0.5°块 block-bootstrap（仅重采样块计数） ----------
    tab2 = []
    tab1 = []
    for b in np.unique(blk):
        m = sel & (blk == b)
        if m.sum() < 2000:
            continue
        tab2.append((int((m & (cls2 == 6) & shady).sum()), int((m & (cls2 == 6) & sunny).sum()),
                     int((m & (cls2 == 4) & shady).sum()), int((m & (cls2 == 4) & sunny).sum())))
        m1 = (slope > 5) & np.isin(cls1, (4, 6)) & (blk == b)
        if m1.sum() >= 2000:
            tab1.append((int((m1 & (cls1 == 6) & shady).sum()), int((m1 & (cls1 == 6) & sunny).sum()),
                         int((m1 & (cls1 == 4) & shady).sum()), int((m1 & (cls1 == 4) & sunny).sum())))
    def boot_or(tabs, B=2000, seed=7):
        if not tabs:
            return None, None, None
        T = np.array(tabs, float)
        rng2 = np.random.default_rng(seed)
        vals = []
        for _ in range(B):
            idx = rng2.integers(0, T.shape[0], T.shape[0])
            s = T[idx].sum(axis=0) + 0.5
            vals.append((s[0] * s[3]) / (s[1] * s[2]))
        vals = np.array(vals)
        return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), float(np.median(vals))
    bb2 = boot_or(tab2)
    bb1 = boot_or(tab1)
    out['S6a_block_bootstrap_0p5deg'] = {'n_blocks_run2': len(tab2), 'OR_run2_median': bb2[2],
                                         'OR_run2_95CI': [bb2[0], bb2[1]],
                                         'n_blocks_run1': len(tab1), 'OR_run1_median': bb1[2],
                                         'OR_run1_95CI': [bb1[0], bb1[1]]}
    print('  block-bootstrap(0.5°,B=2000): 二版 OR 中位=%.2f CI=[%.2f, %.2f] (块=%d)；一版 OR 中位=%.2f CI=[%.2f, %.2f] (块=%d)' %
          (bb2[2], bb2[0], bb2[1], len(tab2), bb1[2], bb1[0], bb1[1], len(tab1)))

    # ---------- S6b 0.05° 细块（~5km） ----------
    blk2 = (np.floor(lon * 20).astype(np.int64) * 1000000 + np.floor(lat * 20).astype(np.int64))
    per_b2 = []
    for b in np.unique(blk2):
        m = sel & (blk2 == b)
        if m.sum() < 800:
            continue
        a = int((m & (cls2 == 6) & shady).sum()); bb = int((m & (cls2 == 6) & sunny).sum())
        c = int((m & (cls2 == 4) & shady).sum()); d = int((m & (cls2 == 4) & sunny).sum())
        orr, _ = odds_ci(a, bb, c, d)
        per_b2.append((orr, a + bb, c + d))
    o2 = np.array([x[0] for x in per_b2])
    out['S6b_fine_blocks_0p05deg'] = {'n_blocks': int(o2.size), 'frac_OR_gt1': float((o2 > 1).mean()),
                                      'median_OR': float(np.median(o2)),
                                      'q25_q75': [float(np.percentile(o2, 25)), float(np.percentile(o2, 75))]}
    print('  细块(0.05°) n=%d，OR>1 占比=%.1f%%，中位 OR=%.2f，IQR=[%.2f, %.2f]' %
          (o2.size, 100 * (o2 > 1).mean(), np.median(o2), np.percentile(o2, 25), np.percentile(o2, 75)))

    # ---------- S7 一版同名 AUC（同点） ----------
    sel1 = (slope > 5) & np.isin(cls1, (4, 6))
    pos1 = (cls1[sel1] == 6)
    S7 = {}
    for tag, score in (('hillshade_az135_el45', hs(135, 45)), ('hillshade_az180_el50', hs(180, 50)),
                       ('cos_asp_minus_180', np.cos(arcs - np.pi))):
        S7[tag] = round(auc_mw(score[sel1].astype(np.float64), pos1), 4)
    out['S7_AUC_run1'] = S7
    print('  一版同口径 AUC: ' + ' | '.join('%s=%.3f' % kv for kv in S7.items()))

    json.dump(out, open(os.path.join(OUTD, 'tables_川.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1, default=float)
    print('\n已写 tables_川.json')


if __name__ == '__main__':
    main()
