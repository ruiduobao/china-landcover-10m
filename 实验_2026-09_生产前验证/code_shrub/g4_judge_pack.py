# -*- coding: utf-8 -*-
"""g4_judge_pack.py — 组装判读包：候选分层配额 → 判读点表（含物候/证据提示）

* 输入：data/shrub/cand_all.csv（候选 + DW/WC/LC 标签 + AEF + S2 物候 + DEM/bio）
* 规则源（预注册配额，每省）：L1 共识 全取 ｜ L4 仅DW 30 ｜ L3 单源 20 ｜ OSM 25 ｜ FCS10 25
        合计每省 ≤ 120，四省 ≤ 480 → 判读子代理 6 个（每批 ~80 点）
* 门槛：同一省同层内按经纬度均匀取样（不扎堆）；缺 AEF 的点剔除
* 输出：data/shrub/judge_points.csv（point_id,prov,stratum,lon,lat,dw_prob,ndvi_amp,ndvi_djf,ndvi_jja,dem,hint）
        data/shrub/judge_plan.json
* 用法：python g4_judge_pack.py
"""
import json, os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
sys.path.insert(0, os.path.join(WORK, 'code'))
import numpy as np, pandas as pd
import v31_common as VC
OUTD = os.path.join(WORK, 'data', 'shrub')
SEED = 20261010
QUOTA = {'L1': 999, 'L2': 30, 'L4': 30, 'L3': 20, 'OSM': 25, 'FCS10': 25}

def emit(m): print(time.strftime('[%m-%d %H:%M:%S] ')+str(m), flush=True)

def main():
    df = pd.read_csv(os.path.join(OUTD, 'cand_all.csv'), encoding='utf-8-sig')
    emit('候选 %d' % len(df))
    df = df[df.get('A00').notna()] if 'A00' in df.columns else df
    rng = np.random.RandomState(SEED)
    keep = []
    for (prov, st), g in df.groupby(['prov', 'stratum']):
        q = QUOTA.get(st, 20)
        if len(g) <= q:
            sel = g
        else:
            # 空间均匀：按 lon 排序后等间隔取
            g2 = g.sort_values('lon')
            idx = np.linspace(0, len(g2) - 1, q).astype(int)
            sel = g2.iloc[idx]
        emit('  %-4s %-6s %3d → %3d' % (prov, st, len(g), len(sel)))
        keep.append(sel)
    out = pd.concat(keep, ignore_index=True)
    # 判读提示（物候：振幅大=落叶倾向；小=常绿倾向）
    def hint(r):
        h = []
        a = r.get('ndvi_amp')
        if pd.notna(a):
            if a < 0.15: h.append('物候振幅小→常绿倾向')
            elif a > 0.30: h.append('物候振幅大→落叶倾向')
            else: h.append('物候振幅中等')
        p = r.get('dw_prob')
        if pd.notna(p): h.append('DW灌丛概率%.2f' % float(p))
        e = r.get('dem')
        if pd.notna(e): h.append('海拔%.0fm' % float(e))
        return '；'.join(h)
    out['hint'] = out.apply(hint, axis=1)
    cols = ['point_id', 'prov', 'stratum', 'lon', 'lat', 'dw_prob', 'dw_label', 'wc20', 'lc100',
            'ndvi_djf', 'ndvi_jja', 'ndvi_amp', 'dem', 'slope', 'bio01', 'bio12', 'occ', 'rh98',
            'src', 'hint']
    out[[c for c in cols if c in out.columns]].to_csv(
        os.path.join(OUTD, 'judge_points.csv'), index=False, encoding='utf-8-sig')
    plan = {'n': int(len(out)), 'quota': QUOTA,
            'by_prov_stratum': {('%s|%s' % k): int(v) for k, v in out.groupby(['prov', 'stratum']).size().items()},
            'seed': SEED}
    VC.jsave(plan, os.path.join(OUTD, 'judge_plan.json'))
    emit('判读点 %d → %s' % (len(out), os.path.join(OUTD, 'judge_points.csv')))
    emit('%s' % plan['by_prov_stratum'])

if __name__ == '__main__': main()
