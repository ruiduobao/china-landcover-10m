# -*- coding: utf-8 -*-
"""r5_fuse.py — 三源融合：生成带证据等级的高精度针阔样本

* 输入（只读）：
    opt/needle_broad_veg30_v1.parquet   30m 植被类型样本 138,175（R2 产出，几何精确）
    opt/r1_points.parquet               v2 110,653 点在 30m/植被图/PFT 上的取值
    labels/needle_broad_national_v2.parquet  v2 主样本（FCS10 口径）
* 规则源：技术文档/81（V1–V9）+ 本轮实测（Q1/Q3/R1/R4）
* **核心设计（不裁决"谁对"，改为显式分级）**：
    R1 三源投票：FCS10(v2) × 30m植被类型 × PFT2013 —— 但**30m 与 100万植被图同源
       （一致率 0.9837），所以按实测只算"两个独立源"：P = FCS10，V = 植被图/30m 合体
    R2 区域赋权（依据实测判别效度，不是拍脑袋）：
       - 北方/西部（东北/华北/西北/青藏/西南）：FCS10 有判别效度（+15.8~+37.9pp）
         → P 与 V 同权
       - 南方（华东/华中/华南）：FCS10 判别效度 ≤0 甚至反向（−2.6pp，非森林区更高）
         → **V 单方权威**，P 只作记录不作裁决
       - 华南（粤/桂）：V 本身也失效（针叶占比 1981→2013 掉 20.4pp）→ 标 TIER_C
    R3 证据分级 tier：
       TIER_A  双源一致（P==V）+ 无疑义 + PFT 支持（或 PFT 无记录）
       TIER_B  双源一致，但 PFT 反对 / 或有软歧义
       TIER_C  双源冲突（P!=V）—— 保留但标冲突，供下游决策
       TIER_D  疑义大（硬歧义群系 / 华南针叶）—— 单独清单，不建议直接进训练
* 输出：
    opt/needle_broad_opt_v1.parquet      融合样本（含 tier / src_flags / conf）
    opt/opt_v1_stats.json                全量统计
    opt/opt_v1_by_prov.csv / by_tile.csv / by_form.csv
* 用法：python r5_fuse.py
* 资源：单进程，< 3 GB
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
OPT = r'F:/lc_work/nb_national_2026/opt'
LAB = r'F:/lc_work/nb_national_2026/labels'
PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'

# 区域分组（依据 Q1/Q3 实测判别效度，见 doc82）
SOUTH_FCS10_VOID = {'华东', '华中', '华南'}     # FCS10 判别效度 ≤ 0
SOUTH_VEG_VOID = {'广东省', '广西壮族自治区'}   # 植被图针叶标注也失效（Q3 核心B）

VEG_ECO = {
    '东北': ['黑龙江省', '吉林省', '辽宁省'],
    '华北': ['北京市', '天津市', '河北省', '山西省', '内蒙古自治区'],
    '西北': ['陕西省', '甘肃省', '宁夏回族自治区', '新疆维吾尔自治区'],
    '青藏': ['青海省', '西藏自治区'],
    '西南': ['四川省', '重庆市', '云南省', '贵州省'],
    '华中': ['河南省', '湖北省', '湖南省', '江西省', '安徽省'],
    '华东': ['江苏省', '浙江省', '上海市', '山东省', '福建省', '台湾省'],
    '华南': ['广东省', '广西壮族自治区', '海南省', '香港特别行政区', '澳门特别行政区'],
}
PROV2ECO = {p: e for e, ps in VEG_ECO.items() for p in ps}


def emit(m):
    print(time.strftime('[%H:%M:%S] ') + str(m), flush=True)


def key(lon, lat, nd=5):
    return (np.round(np.asarray(lon) * 10**nd).astype('int64').astype(str) + '_' +
            np.round(np.asarray(lat) * 10**nd).astype('int64').astype(str))


def main():
    t0 = time.time()

    # ---------- 1. 读三份数据 ----------
    v30 = pd.read_parquet(os.path.join(OPT, 'needle_broad_veg30_v1.parquet'))
    emit('30m 样本 %d（针 %d / 阔 %d）' % (
        len(v30), int((v30.leaf == 'needle').sum()), int((v30.leaf == 'broad').sum())))

    v2 = pd.read_parquet(os.path.join(LAB, 'needle_broad_national_v2.parquet'),
                         columns=['lon', 'lat', 'tile', 'leaf', 'fcs10_raw', 'prov'])
    r1 = pd.read_parquet(os.path.join(OPT, 'r1_points.parquet'),
                         columns=['point_id', 'v30_cls', 'pft_state', 'veg_leaf'])
    r1 = r1.merge(v2[['lon', 'lat']], left_index=True, right_index=True, how='left')
    emit('v2 %d / r1 %d' % (len(v2), len(r1)))

    # ---------- 2. 在 v2 点上补 30m 与 PFT 证据（v2 点本身带 FCS10 标签）----------
    v2 = v2.copy()
    v2['k'] = key(v2.lon, v2.lat)
    r1['k'] = key(r1.lon, r1.lat)
    r1s = r1.drop_duplicates('k')[['k', 'v30_cls', 'pft_state', 'veg_leaf']]
    v2 = v2.merge(r1s, on='k', how='left')
    v2['src'] = 'v2_fcs10'
    v2['P'] = v2.leaf                                    # FCS10 判读
    v2['V'] = v2.v30_cls.map({'needle': 'needle', 'broad': 'broad'}).fillna('na')
    v2['T'] = v2.pft_state.map({'needle': 'needle', 'broad': 'broad'}).fillna('na')
    v2['eco'] = v2.prov.map(PROV2ECO).fillna('NA')

    # ---------- 3. 30m 样本：V 已知，P 未知（需就地取 FCS10？不做——避免循环）----------
    # 30m 点的 P 证据来自「该点是否落在 v2 的 0.01° 格里」；不在则 P=na。
    # 这是保守做法：不重新去 FCS10 栅格取值（本地栅格与表格标签实测不一致，见 doc80 §10.5）
    v30 = v30.copy()
    v30['eco'] = v30.prov.map(PROV2ECO).fillna('NA')
    v30['V'] = v30.leaf
    v30['P'] = 'na'
    v30['T'] = 'na'
    v30['src'] = 'veg30'
    v2k = set(key(v2.lon, v2.lat, 2))       # 0.01° 格
    v30['_gk'] = key(v30.lon, v30.lat, 2)
    in_v2_cell = v30._gk.isin(v2k).values
    emit('30m 点落在 v2 的 0.01° 格内：%d / %d (%.2f%%)' % (
        int(in_v2_cell.sum()), len(v30), 100 * in_v2_cell.mean()))
    v30 = v30.drop(columns=['_gk'])

    # ---------- 4. 证据分级（对 v2 点做，30m 点按 V 单源+歧义分级）----------
    def tier_row(r):
        P, V, T, eco, prov = r['P'], r['V'], r['T'], r['eco'], r['prov']
        # 30m 独有样本（P 未知）
        if P == 'na':
            amb = r.get('ambig_level', 0)
            if amb == 1 or (prov in SOUTH_VEG_VOID and V == 'needle'):
                return 'TIER_D', 0.55, 'veg_only_ambiguous'
            return ('TIER_A', 0.75, 'veg_only') if amb == 0 else ('TIER_B', 0.65, 'veg_only_soft')
        # v2 点：P 已知
        if V == 'na':
            # 无植被图证据：南方 FCS10 无判别效度 → 不可用（D）；北方 FCS10 有效 → C
            if eco in SOUTH_FCS10_VOID:
                return 'TIER_D', 0.50, 'no_veg_south_unusable'
            return 'TIER_C', 0.60, 'no_veg_north_fcs10_only'
        if P == V:
            if T in ('na', V):
                return 'TIER_A', 0.95, 'two_source_agree'
            return 'TIER_B', 0.80, 'agree_but_pft_opposes'
        # 冲突
        if eco in SOUTH_FCS10_VOID:
            # 南方：FCS10 无判别效度 → 信 V
            if prov in SOUTH_VEG_VOID and V == 'needle':
                return 'TIER_D', 0.50, 'conflict_south_both_weak'
            return 'TIER_B', 0.70, 'conflict_south_trust_veg'
        if T != 'na' and T == V:
            return 'TIER_B', 0.75, 'conflict_two_sources_beat_fcs10'
        if T != 'na' and T == P:
            return 'TIER_C', 0.60, 'conflict_pft_sides_fcs10'
        return 'TIER_C', 0.55, 'conflict_no_tiebreak'

    emit('给 v2 点分级…')
    v2t = v2.apply(lambda r: pd.Series(tier_row(r), index=['tier', 'conf', 'reason']), axis=1)
    v2 = pd.concat([v2.reset_index(drop=True), v2t.reset_index(drop=True)], axis=1)

    emit('给 30m 点分级…')
    v30t = v30.apply(lambda r: pd.Series(tier_row(r), index=['tier', 'conf', 'reason']), axis=1)
    v30 = pd.concat([v30.reset_index(drop=True), v30t.reset_index(drop=True)], axis=1)

    # ---------- 5. 最终标签：冲突点按分级规则定 --------
    # TIER_A/B/C 采用"权威源"：南方用 V，北方用 P（一致时无差别）
    def final_leaf(r):
        if r['P'] == 'na':
            return r['V']
        if r['V'] == 'na':
            return r['P']          # 无植被图证据，用 FCS10（tier 已表达风险）
        if r['P'] == r['V']:
            return r['P']
        if r['eco'] in SOUTH_FCS10_VOID:
            return r['V']
        if r['reason'] == 'conflict_two_sources_beat_fcs10':
            return r['V']
        if r['reason'] == 'conflict_pft_sides_fcs10':
            return r['P']
        return r['P']          # 默认保留 FCS10（北方有判别效度）

    v2['leaf_final'] = v2.apply(final_leaf, axis=1)
    v30['leaf_final'] = v30['V']

    # ---------- 6. 合并输出 ----------
    cols_common = ['lon', 'lat', 'tile', 'prov', 'eco', 'leaf_final', 'tier', 'conf',
                   'reason', 'src']
    out = pd.concat([
        v2[cols_common + ['P', 'V', 'T']].assign(form=None, ambig_level=0),
        v30[cols_common + ['gen', 'ambig_level']].rename(columns={'gen': 'form'})
            .assign(P='na', V=v30['V'], T='na'),
    ], ignore_index=True)
    out = out.drop_duplicates(subset=['lon', 'lat'])
    out['point_id'] = np.arange(len(out), dtype='int64') + 800_000_000
    out.to_parquet(os.path.join(OPT, 'needle_broad_opt_v1.parquet'), index=False)
    emit('写 needle_broad_opt_v1.parquet：%d 行' % len(out))

    # ---------- 7. 统计 ----------
    st = {
        'n_total': int(len(out)),
        'by_leaf_final': {k: int(v) for k, v in out.leaf_final.value_counts().items()},
        'by_tier': {k: int(v) for k, v in out.tier.value_counts().items()},
        'by_src': {k: int(v) for k, v in out.src.value_counts().items()},
        'tier_x_leaf': {},
        'by_reason': {k: int(v) for k, v in out.reason.value_counts().items()},
        'tiles': int(out.tile.nunique()),
        'provs': int(out.prov.nunique()),
        'note': ('tier: A=双源一致(或30m无疑义) B=一致但有异议/软歧义 C=冲突 D=疑义大；'
                 '南方(华东/华中/华南)FCS10 无判别效度故按植被图定；'
                 '粤/桂针叶标 TIER_D（植被图本身失效，Q3 实证）'),
    }
    for t, g in out.groupby('tier'):
        st['tier_x_leaf'][t] = {k: int(v) for k, v in g.leaf_final.value_counts().items()}

    out.groupby(['prov', 'tier']).size().unstack(fill_value=0).to_csv(
        os.path.join(OPT, 'opt_v1_by_prov.csv'), encoding='utf-8-sig')
    out.groupby(['tile', 'tier']).size().unstack(fill_value=0).to_csv(
        os.path.join(OPT, 'opt_v1_by_tile.csv'), encoding='utf-8-sig')
    fm = out[out.form.notna()].groupby(['form', 'tier']).size().unstack(fill_value=0)
    fm.sort_values(fm.columns[0] if len(fm.columns) else None, ascending=False).to_csv(
        os.path.join(OPT, 'opt_v1_by_form.csv'), encoding='utf-8-sig')

    json.dump(st, open(os.path.join(OPT, 'opt_v1_stats.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    emit('完成 %.1f min' % ((time.time() - t0) / 60))
    emit(json.dumps({k: st[k] for k in ('n_total', 'by_leaf_final', 'by_tier', 'tiles', 'provs')},
                    ensure_ascii=False))


if __name__ == '__main__':
    main()
