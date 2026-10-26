# -*- coding: utf-8 -*-
"""q6_ning_plus6.py — 证据审计·宁夏 n 1945→1951 的配对分母分解（固定共同有效点集）

* 输入（只读）：
    - 留出窗：F:/lc_work/v31_exp/data/yearly/eval_2023_w{1..5}.parquet
    - 第一次分类图：D:/生产_2023五省/成品/五省合幅/五省合幅_2023_10m.tif（合幅口径，与基线 json 同源）
    - 第二次分类图：F:/lc_work/prod2nd_2023/rasters_batch/T{1608,1509,1609,1709,1610,1710}_10m.tif（瓦片口径）
* 规则源：**逐条复用 c4_eval_second 的函数**（build_exclusions/assign_prov/sample_local/read_plan2/
  locals_for_prov），保证与正式评估同一代码路径；本轮只抽 1964 个宁夏点，属轻量复算（非重扫影像）。
* 目的（预注册问题）：+6 点（第一次 miss→第二次有效）对 OA 的影响必须与"固定共同有效点集下的模型变化"分开：
    base = run1 在 run1 有效集（1945）上的 OA；new = run2 在 run2 有效集（1951）上的 OA；
    Δ_fixed = run2−run1 在"两轮共同有效集"上的配对差（纯模型/点位变化）；
    Δ_comp  = Δ_total − Δ_fixed（覆盖/构成效应，含 +6 加入与 run1 独有的 miss 点差异）。
  另单报：only2（run1 miss→run2 有效）6 点的真值类/预测/对错；only1（run1 有效→run2 miss）计数。
* 输出（全落本目录）：q6_宁_plus6分解.json、q6_宁_共点预测缓存.csv（逐点，traceability）
* 用法：python q6_ning_plus6.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r'F:/地理所/论文/中国土地覆盖数据_2017-2024/代码/9.第二次分类川粤宁')
import c4_eval_second as C4
import v31_common as VC

OUT = os.path.dirname(os.path.abspath(__file__))
RUN1_MOSAIC = r'D:/生产_2023五省/成品/五省合幅/五省合幅_2023_10m.tif'


def safe(p, t):
    v = C4.VC.metrics(t, p)
    return dict(n=v['n'], OA=round(v['OA'], 4),
                m9=round(C4.VC.metrics(C4.VC.to_macro(t), C4.VC.to_macro(p))['OA'], 4))


def main():
    # 1) 载荷并做与 c4 一致的排除/归省
    fr = []
    for w in ['w1', 'w2', 'w3', 'w4', 'w5']:
        fr.append(pd.read_parquet(os.path.join(C4.EVAL_DIR, 'eval_2023_%s.parquet' % w),
                                  columns=['row_id', 'lon', 'lat', 'class_new', 'src']))
    ev = pd.concat(fr, ignore_index=True)
    keep, xd = C4.build_exclusions(ev, 'both')
    ev = ev[keep].reset_index(drop=True)
    prov, _ = C4.assign_prov(ev.lon.to_numpy(), ev.lat.to_numpy(), list(C4.PC.PROV_SHORT))
    ev['prov'] = prov
    sub = ev[ev.prov == '宁夏回族自治区'].reset_index(drop=True)
    assert len(sub) == 1964, len(sub)
    # 2) 取值（同一路径函数；run1=合幅，run2=宁 6 瓦）
    lons, lats = sub.lon.to_numpy(), sub.lat.to_numpy()
    v1, _ = C4.sample_local([RUN1_MOSAIC], lons, lats)
    plan = C4.read_plan2()
    fps2, miss_tiles = C4.locals_for_prov('宁', plan)
    assert not miss_tiles, miss_tiles
    v2, _ = C4.sample_local(fps2, lons, lats)
    sub['pred1'] = v1
    sub['pred2'] = v2
    m1 = sub.pred1.isna() | (sub.pred1 <= 0) | (sub.pred1 > 250)
    m2 = sub.pred2.isna() | (sub.pred2 <= 0) | (sub.pred2 > 250)
    t = VC.to_v31(sub.class_new.to_numpy(int))
    isf = C4.PC.is_fcs10(sub.src).to_numpy()
    v1a, v2a = pd.to_numeric(sub.pred1, errors='coerce').fillna(-1).to_numpy(int), \
        pd.to_numeric(sub.pred2, errors='coerce').fillna(-1).to_numpy(int)
    both = (~m1.to_numpy()) & (~m2.to_numpy())
    only2 = m1.to_numpy() & (~m2.to_numpy())
    only1 = (~m1.to_numpy()) & m2.to_numpy()          # run1 有效→run2 miss
    bothmiss = m1.to_numpy() & m2.to_numpy()
    out = dict(n_pool=len(sub), run1_n_miss=int(m1.sum()), run2_n_miss=int(m2.sum()),
               n_common_valid=int(both.sum()), n_only2=int(only2.sum()), n_only1=int(only1.sum()),
               n_both_miss=int(bothmiss.sum()))
    # 3) 口径矩阵：base(1945) / new(1951) / fixed(common)
    base = safe(t[~m1.to_numpy()], v1a[~m1.to_numpy()])
    new = safe(t[~m2.to_numpy()], v2a[~m2.to_numpy()])
    fixed = dict(run1=safe(t[both], v1a[both]), run2=safe(t[both], v2a[both]))
    fix_oa = fixed['run2']['OA'] - fixed['run1']['OA']
    fix_m9 = fixed['run2']['m9'] - fixed['run1']['m9']
    out.update(base_run1=base, new_run2=new, fixed_common_set=fixed,
               delta_total_OA=round(new['OA'] - base['OA'], 4),
               delta_fixed_OA=round(fix_oa, 4),
               delta_comp_OA=round((new['OA'] - base['OA']) - fix_oa, 4),
               delta_total_m9=round(new['m9'] - base['m9'], 4),
               delta_fixed_m9=round(fix_m9, 4),
               delta_comp_m9=round((new['m9'] - base['m9']) - fix_m9, 4))
    # 4) 分层（非fcs10 两轮同集 1308）：直接对固定集断言
    nf = ~isf
    fixed_nf = dict(run1=safe(t[nf & both], v1a[nf & both]), run2=safe(t[nf & both], v2a[nf & both]))
    out['fixed_nonfcs10'] = fixed_nf
    out['fixed_nonfcs10_delta'] = dict(
        OA=round(fixed_nf['run2']['OA'] - fixed_nf['run1']['OA'], 4),
        m9=round(fixed_nf['run2']['m9'] - fixed_nf['run1']['m9'], 4))
    # 5) +6 点明细
    add = []
    names = VC.V31_NAMES()
    for i in np.where(only2)[0]:
        add.append(dict(lon=float(sub.lon.iloc[i]), lat=float(sub.lat.iloc[i]),
                        truth24=int(t[i]), truth_name=names.get(str(int(t[i])), '?'),
                        fcs10=bool(isf[i]), pred1=('miss' if m1.iloc[i] else int(v1a[i])),
                        pred2=int(v2a[i]), pred2_correct=bool(v2a[i] == t[i])))
    out['only2_points'] = add
    out['only1_points'] = [dict(lon=float(sub.lon.iloc[i]), lat=float(sub.lat.iloc[i]),
                                truth24=int(t[i]), pred1=int(v1a[i]))
                           for i in np.where(only1)[0]]
    # 6) 自检：复现既有读数
    assert out['base_run1']['OA'] == 0.8062 and out['new_run2']['OA'] == 0.6540, out
    json.dump(out, open(os.path.join(OUT, 'q6_宁_plus6分解.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    sub[['row_id', 'lon', 'lat', 'class_new', 'src', 'pred1', 'pred2']].to_csv(
        os.path.join(OUT, 'q6_宁_共点预测缓存.csv'), index=False, encoding='utf-8-sig')
    print('base(1945)=%.4f m9=%.4f | new(1951)=%.4f m9=%.4f' % (
        base['OA'], base['m9'], new['OA'], new['m9']))
    print('共同有效集 n=%d：run1=%.4f→run2=%.4f（Δfixed OA %+.4f / m9 %+.4f）' % (
        out['n_common_valid'], fixed['run1']['OA'], fixed['run2']['OA'], fix_oa, fix_m9))
    print('Δtotal=%+.4f = Δfixed %+.4f + Δcomp %+.4f' % (
        out['delta_total_OA'], out['delta_fixed_OA'], out['delta_comp_OA']))
    print('非fcs10 固定集 n=%d：%.4f→%.4f（Δ %+.4f）' % (
        fixed_nf['run1']['n'], fixed_nf['run1']['OA'], fixed_nf['run2']['OA'],
        out['fixed_nonfcs10_delta']['OA']))
    for p in add:
        print('  +6点', p['lon'], p['lat'], p['truth_name'], 'fcs10=%s' % p['fcs10'],
              p['pred1'], '→', p['pred2'], '对' if p['pred2_correct'] else '错')
    print('q6 完成')


if __name__ == '__main__':
    main()
