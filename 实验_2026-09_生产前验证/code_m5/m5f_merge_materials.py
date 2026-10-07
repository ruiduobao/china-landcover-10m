# -*- coding: utf-8 -*-
"""m5f_merge_materials.py — D0 材料合并：M3 判读表单加强版 + 灌丛层体检统计（doc45 §一）
* 输入：data/m3/m5_geemat/{m3ind,audit}/*.csv（指标）＋ m3chip/chip_*.csv + chips/*.png（切片）
        data/m3/m3a|m3b|m3cd_判读表单.csv（原始表单）
* 规则源：指标提示规则（doc45 §一B，辅助不替代人工）：ch_mean≥5 偏森林 / 1.5–5 灌丛带 / <1.5 偏草本；
          NDVI 振幅大（≥0.35）且 tc2000 低 → 草本物候
* 门槛：指标缺失留空；ERR 行剔除；同一 point_id 取最后一条
* 输出：data/m3/*_判读表单_材料版.csv（原列 + 材料列 + 提示列）；data/m3/m5_audit_stats.json；
        reports/D0_体检统计_YYYYMMDD.md
* 用法：python m5f_merge_materials.py
"""
import os, sys, csv, json, time, glob
from collections import defaultdict
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v31_common as VC

M3D = os.path.join(VC.DATA, 'm3')
GEO = os.path.join(M3D, 'm5_geemat')
IND_COLS = ['ch_mean', 'tc2000', 'lossyear', 'nd_djf', 'nd_mam', 'nd_jja', 'nd_son',
            'ndvi_amp', 'ndvi_mean', 'ndvi_winter', 'ndvi_summer']
FORMS = {'A': 'm3a_判读表单.csv', 'B': 'm3b_判读表单.csv', 'CD': 'm3cd_判读表单.csv'}


def load_ind(sub, ch_sub=None):
    rows = {}
    for fp in glob.glob(os.path.join(GEO, sub, 'ind_*.csv')):
        with open(fp, encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                if not r.get('point_id') or r.get('ch_mean') == 'ERR':
                    continue
                rows[r['point_id']] = r
    for fp in glob.glob(os.path.join(GEO, ch_sub or sub, 'ch_*.csv')):
        with open(fp, encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                pid = r.get('point_id')
                if not pid:
                    continue
                rows.setdefault(pid, {'point_id': pid})
                rows[pid]['ch_mean'] = r.get('ch_mean', '')
    return rows


def load_chips():
    ok = {}
    for fp in (glob.glob(os.path.join(GEO, 'm3chip', 'chip_*.csv'))
               + glob.glob(os.path.join(GEO, 'm3chip2', 'chip_*.csv'))):
        with open(fp, encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                if r.get('status') == 'ok' and r.get('png') and os.path.exists(r['png']):
                    ok[r['point_id']] = r['png']
    return ok


def hint(rec):
    """指标提示（doc45 §一B 规则；冲突/缺指标返回空）。"""
    try:
        ch = float(rec['ch_mean']) if rec.get('ch_mean') not in ('', None) else None
    except ValueError:
        ch = None
    try:
        amp = float(rec['ndvi_amp']) if rec.get('ndvi_amp') not in ('', None) else None
    except ValueError:
        amp = None
    try:
        tc = float(rec['tc2000']) if rec.get('tc2000') not in ('', None) else None
    except ValueError:
        tc = None
    if ch is None and amp is None:
        return ''
    tags = []
    if ch is not None:
        if ch >= 5:
            tags.append('高度≥5m·偏森林/郁闭木本')
        elif ch >= 1.5:
            tags.append('高度1.5–5m·灌丛典型带')
        else:
            tags.append('高度<1.5m·偏草本/低矮')
    if amp is not None and tc is not None:
        if amp >= 0.35 and tc < 30:
            tags.append('物候振幅大+树覆盖低·草本倾向')
        elif amp < 0.2 and tc >= 30:
            tags.append('物候平稳+树覆盖高·常绿木本倾向')
    return '；'.join(tags)


def merge_forms(ind, chips):
    n_forms = {}
    for tag, fn in FORMS.items():
        fp = os.path.join(M3D, fn)
        if not os.path.exists(fp):
            continue
        rows = list(csv.DictReader(open(fp, encoding='utf-8-sig')))
        base_cols = list(rows[0].keys()) if rows else []
        out = os.path.join(M3D, fn.replace('.csv', '_材料版.csv'))
        n_have_i, n_have_c = 0, 0
        with open(out, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(base_cols + ['chip'] + IND_COLS + ['指标提示'])
            for r in rows:
                pid = r.get('point_id', '')
                iv = ind.get(pid, {})
                cp = chips.get(pid, '')
                if iv:
                    n_have_i += 1
                if cp:
                    n_have_c += 1
                w.writerow([r.get(c, '') for c in base_cols] + [cp]
                           + [iv.get(c, '') for c in IND_COLS] + [hint(iv) if iv else ''])
        n_forms[tag] = dict(file=out, n=len(rows), n_ind=n_have_i, n_chip=n_have_c)
        print('%s: %d 点，指标 %d，切片 %d → %s' % (tag, len(rows), n_have_i, n_have_c, out))
    return n_forms


def audit_stats():
    ind = load_ind('audit', 'audit_ch')
    # 层键从点表取
    strata = defaultdict(list)
    with open(os.path.join(M3D, 'm5_audit_points.csv'), encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            iv = ind.get(r['point_id'])
            if not iv:
                continue
            strata[(r['dry'], r['code'])].append(iv)
    out = {}
    for (dry, code), vs in sorted(strata.items()):
        chs, amps, tcs = [], [], []
        for v in vs:
            if v.get('ch_mean') not in ('', None):
                chs.append(float(v['ch_mean']))
            if v.get('ndvi_amp') not in ('', None):
                amps.append(float(v['ndvi_amp']))
            if v.get('tc2000') not in ('', None):
                tcs.append(float(v['tc2000']))
        hi = sum(1 for c in chs if c >= 5)
        band = sum(1 for c in chs if 1.5 <= c < 5)
        lo = sum(1 for c in chs if c < 1.5)
        n = len(vs)
        rec = dict(n=n, n_ch=len(chs),
                   ch_median=round(sorted(chs)[len(chs) // 2], 2) if chs else None,
                   tc_median=round(sorted(tcs)[len(tcs) // 2], 1) if tcs else None,
                   amp_median=round(sorted(amps)[len(amps) // 2], 3) if amps else None,
                   pct_forest_like=round(hi / len(chs), 3) if chs else None,
                   pct_shrub_band=round(band / len(chs), 3) if chs else None,
                   pct_low=round(lo / len(chs), 3) if chs else None)
        out['%s|%s' % (dry, code)] = rec
        print('层 dry=%s code=%s: n=%d ch中位=%s 树覆盖中位=%s 振幅中位=%s 森林像=%s 灌丛带=%s 低矮=%s' % (
            dry, code, n, rec['ch_median'], rec['tc_median'], rec['amp_median'],
            rec['pct_forest_like'], rec['pct_shrub_band'], rec['pct_low']))
    VC.jsave(out, os.path.join(M3D, 'm5_audit_stats.json'))
    # 报告
    L = ['# D0 灌丛层物理体检统计 %s' % time.strftime('%Y-%m-%d %H:%M'), '',
         '- 样本：`glc_fcs10_2023_shrub`（537,361 点）分层抽 20,000（干/湿 × 原码 120/121 各 5,000）',
         '- 指标：LARSE GEDI L4B 冠层高度（1 km, MU 波段）、Hansen treecover2000、S2 四季 NDVI 振幅',
         '- 判读参考带：高度 ≥5 m 偏森林 / 1.5–5 m 灌丛典型带 / <1.5 m 偏草本；振幅 ≥0.35 且树覆盖 <30% 草本倾向',
         '', '| 层 | n | 冠层高度中位(m) | 树覆盖中位(%) | NDVI振幅中位 | 森林像% | 灌丛带% | 低矮% |',
         '|---|---|---|---|---|---|---|---|']
    for k, v in out.items():
        dry, code = k.split('|')
        L.append('| %s %s | %d | %s | %s | %s | %s | %s | %s |' % (
            '干旱省' if dry == 'True' else '湿润省', code, v['n'], v['ch_median'], v['tc_median'],
            v['amp_median'], v['pct_forest_like'], v['pct_shrub_band'], v['pct_low']))
    fp = os.path.join(VC.REPT, 'D0_体检统计_%s.md' % time.strftime('%Y%m%d'))
    open(fp, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('→', fp)
    return out


def main():
    ind = load_ind('m3ind', 'm3ch')
    chips = load_chips()
    print('M3 指标 %d 点；切片 %d 张' % (len(ind), len(chips)))
    forms = merge_forms(ind, chips)
    stats = audit_stats()
    VC.jsave(dict(forms=forms, n_ind=len(ind), n_chip=len(chips)), os.path.join(M3D, 'm5_merge_summary.json'))


if __name__ == '__main__':
    main()
