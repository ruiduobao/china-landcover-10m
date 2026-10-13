# -*- coding: utf-8 -*-
"""o25_assembly.py — 乔灌园地 v4 专题样本集装配（橡胶/茶园/多源果园/目视锚点 → 交付目录）

* 输入：data/orchard/{rubber_sea_cn_points, tea_2022_cn_points}.parquet + scored_{rubber,tea}.parquet
        data/orchard/scored_gem.parquet（pool+gem 打分层）＋ aef/*（橡胶/茶园 AEF 2023）＋ feats/*（pool AEF）
        data/orchard/{judge2_merged, gem_judge_merged}.csv + judge_r2/judge_r2_*.csv（历次判读）
* 规则源（**预注册**）：
  橡胶样本 = Rubber_SEA 2020 中国区全部候选（30k）＋ p_treecrop；推荐默认子集 = p≥0.5
  茶园样本 = TeaMap 2022 中国区候选（30k）＋ p_treecrop（注：模型 OOD，仅供参考）；推荐默认 = 全量
  多源果园候选 = 池 p≥0.5（1,601）＋ GEM 610（均带 frame/p/判读标签）
  目视确认锚点 = 历次判读 Q1=是 的全部点（去重）
* 输出：F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12/乔灌园地/v4_专题样本/
* 用法：python o25_assembly.py
"""
import glob
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import v31_common as VC

WORK = r'F:/lc_work/v31_exp'
ORD = os.path.join(WORK, 'data', 'orchard')
DEST = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12/乔灌园地/v4_专题样本'


def emit(m):
    print(m, flush=True)


def thin(df, min_km, seed=20261013):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    R = min_km / 111.0
    keep, pts = [], []
    arr = df[['lon', 'lat']].to_numpy()
    for i in idx:
        lo, la = arr[i]
        if pts:
            a = np.asarray(pts)
            if np.hypot((a[:, 0] - lo) * np.cos(np.radians(la)), a[:, 1] - la).min() < R:
                continue
        keep.append(i); pts.append((lo, la))
    return df.iloc[keep]


def main():
    os.makedirs(DEST, exist_ok=True)
    stats = {}
    # ---- 橡胶 ----
    rub = pd.read_parquet(os.path.join(ORD, 'rubber_sea_cn_points.parquet'))
    scra = pd.read_parquet(os.path.join(ORD, 'scored_rubber.parquet'))
    rub = rub.assign(lon6=rub.lon.round(6), lat6=rub.lat.round(6)).merge(
        scra.assign(lon6=scra.lon.round(6), lat6=scra.lat.round(6))[['lon6', 'lat6', 'p_treecrop']],
        on=['lon6', 'lat6'], how='left')
    aef = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(ORD, 'aef', 'rubber_p*.parquet')))],
                    ignore_index=True)
    rub = rub.assign(lon6=rub.lon.round(6), lat6=rub.lat.round(6)).merge(
        aef.assign(lon6=aef.lon.round(6), lat6=aef.lat.round(6))[['lon6', 'lat6'] + VC.FEATS],
        on=['lon6', 'lat6'], how='left')
    rub['subtype'] = '橡胶'
    rub['conf'] = np.where(rub.p_treecrop >= 0.7, 'A(模型强+产品)', np.where(rub.p_treecrop >= 0.5, 'B(模型中+产品)', 'C(仅产品)'))
    rub['rec'] = rub.p_treecrop >= 0.5
    rub = rub.drop(columns=['tile', 'lon6', 'lat6'], errors='ignore')
    rub.to_parquet(os.path.join(DEST, '乔灌园地v4_橡胶样本.parquet'), index=False)
    rub.drop(columns=VC.FEATS).to_csv(os.path.join(DEST, '乔灌园地v4_橡胶样本_精简.csv'),
                                      index=False, encoding='utf-8-sig')
    stats['rubber'] = {'n': int(len(rub)), 'p>=0.5': int((rub.p_treecrop >= 0.5).sum()),
                       'p>=0.7': int((rub.p_treecrop >= 0.7).sum()),
                       'aef_ok': int(rub[VC.FEATS].notna().all(axis=1).sum()) if all(c in rub for c in VC.FEATS) else 0}
    emit('橡胶 %d（p≥0.5 %d）' % (stats['rubber']['n'], stats['rubber']['p>=0.5']))
    # ---- 茶园 ----
    tea = pd.read_parquet(os.path.join(ORD, 'tea_2022_cn_points.parquet'))
    sct = pd.read_parquet(os.path.join(ORD, 'scored_tea.parquet'))
    tea = tea.assign(lon6=tea.lon.round(6), lat6=tea.lat.round(6)).merge(
        sct.assign(lon6=sct.lon.round(6), lat6=sct.lat.round(6))[['lon6', 'lat6', 'p_treecrop']],
        on=['lon6', 'lat6'], how='left')
    ta = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(ORD, 'aef', 'tea_p*.parquet')))],
                   ignore_index=True)
    tea = tea.assign(lon6=tea.lon.round(6), lat6=tea.lat.round(6)).merge(
        ta.assign(lon6=ta.lon.round(6), lat6=ta.lat.round(6))[['lon6', 'lat6'] + VC.FEATS],
        on=['lon6', 'lat6'], how='left')
    tea['subtype'] = '茶'
    tea['conf'] = 'B(专题产品)'
    tea['rec'] = True
    tea = tea.drop(columns=['prov', 'lon6', 'lat6'], errors='ignore')
    if 'prov' in tea.columns:
        pass
    tea.to_parquet(os.path.join(DEST, '乔灌园地v4_茶园样本.parquet'), index=False)
    tea.drop(columns=VC.FEATS).to_csv(os.path.join(DEST, '乔灌园地v4_茶园样本_精简.csv'),
                                      index=False, encoding='utf-8-sig')
    stats['tea'] = {'n': int(len(tea)), 'aef_ok': int(tea[VC.FEATS].notna().all(axis=1).sum()) if all(c in tea for c in VC.FEATS) else 0}
    emit('茶园 %d' % stats['tea']['n'])
    # ---- 多源果园候选（pool p>=0.5 + gem 610）----
    sc = pd.read_parquet(os.path.join(ORD, 'scored_gem.parquet'))
    pool = sc[(sc.src == 'pool') & (sc.p_treecrop >= 0.5)].copy()
    pool['subtype'] = '果园/待判'
    pool['conf'] = np.where(pool.p_treecrop >= 0.7, 'A(模型强)', 'B(模型中)')
    gem = sc[sc.src == 'gem'].copy()
    gem['subtype'] = '果园/待判(热带)'
    gem['conf'] = np.where(gem.p_treecrop >= 0.7, 'A(模型强+GEM)', 'B(模型中+GEM)')
    jm = pd.read_csv(os.path.join(ORD, 'gem_judge_merged.csv'), encoding='utf-8-sig')
    gem = gem.assign(lon6=gem.lon.round(6), lat6=gem.lat.round(6)).merge(
        jm.assign(lon6=jm.lon.round(6), lat6=jm.lat.round(6))[['lon6', 'lat6', 'Q1', 'Q2']],
        on=['lon6', 'lat6'], how='left').drop(columns=['lon6', 'lat6'])
    multi = pd.concat([pool, gem], ignore_index=True)
    multi.to_csv(os.path.join(DEST, '乔灌园地v4_多源果园候选.csv'), index=False, encoding='utf-8-sig')
    stats['multi'] = {'pool_p05': int(len(pool)), 'gem': int(len(gem))}
    emit('多源果园 %d' % len(multi))
    # ---- 目视确认锚点 ----
    anchors = []
    j2 = pd.read_csv(os.path.join(ORD, 'judge2_merged.csv'), encoding='utf-8-sig')
    a = j2[(j2.Q1 == '是')][['point_id', 'lon', 'lat', 'Q2', 'conf', 'note']].copy()
    a['round'] = 'v2_480'; anchors.append(a)
    g2 = jm[jm.Q1 == '是'][['point_id', 'lon', 'lat', 'Q2', 'conf', 'note']].copy()
    g2['round'] = 'gem_208'; anchors.append(g2)
    r2 = pd.concat([pd.read_csv(os.path.join(ORD, 'judge_r2', 'judge_r2_rubber_0%d.csv' % k), encoding='utf-8-sig')
                    for k in range(5)], ignore_index=True)
    t2 = pd.concat([pd.read_csv(os.path.join(ORD, 'judge_r2', 'judge_r2_tea_0%d.csv' % k), encoding='utf-8-sig')
                    for k in range(5)], ignore_index=True)
    r2j = pd.read_csv(os.path.join(ORD, 'judge_points_r2.csv'), encoding='utf-8-sig')
    for tag, j in [('r2_rubber', r2), ('r2_tea', t2)]:
        j = j.merge(r2j[['point_id', 'lon', 'lat']], on='point_id', how='left')
        a = j[j.Q1 == '是'][['point_id', 'lon', 'lat', 'Q2', 'conf', 'note']].copy()
        a['round'] = tag; anchors.append(a)
    anc = pd.concat(anchors, ignore_index=True)
    anc = anc.drop_duplicates(subset=['lon', 'lat'])
    anc.to_csv(os.path.join(DEST, '乔灌园地v4_目视确认锚点.csv'), index=False, encoding='utf-8-sig')
    stats['anchors'] = int(len(anc))
    emit('目视锚点 %d' % len(anc))
    VC.jsave(stats, os.path.join(DEST, 'v4_stats.json'))
    print('→', DEST)


if __name__ == '__main__':
    main()
