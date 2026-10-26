# -*- coding: utf-8 -*-
"""e6_prepare_qarm.py — 高证据园地质量臂·**注册准备（不跑 RF）**

* 目的：按 doc69 §7.2 / 08 报告定义固定质量规则，在原 4 区之外选 1 个新小区，注册
        「质量优选 same count vs 随机 same count（非园地完全一致）」臂。**待主 agent 批准后才跑。**
* 输入：乔灌园地v7_2023 gpkg 元数据；train2023_v2.parquet；五窗 eval（仅取参考 n/地理）
* 门槛：本脚本零 RF、零预测；输出注册 JSON（status=prepared_awaiting_approval）
* 输出：<WS>/prereg/qarm_registration.json、<WS>/explore/e6_qarm_candidates.csv
* 用法：PYTHONDONTWRITEBYTECODE=1 python e6_prepare_qarm.py
"""
import os, sys, json, time, sqlite3, hashlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.dont_write_bytecode = True
sys.path.insert(0, r'F:/地理所/论文/中国土地覆盖数据_2017-2024/代码/9.第二次分类川粤宁')
sys.path.insert(0, r'F:/lc_work/prod5p_2023/mosaic')
sys.path.insert(0, r'F:/lc_work/v31_exp/code')
import c4_eval_second as C4
import p5common as PC
import v31_common as VC

GPKG = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12/乔灌园地/2023/乔灌园地v7_2023_WGS84.gpkg'
POOL2 = r'F:/lc_work/prod2nd_2023/data/train2023_v2.parquet'
EVAL_DIR = r'F:/lc_work/v31_exp/data/yearly'
FEATS = ['A%02d' % i for i in range(64)]
N_LIMIT = 20000
SEEDS = [7, 11, 23]
# 固定质量规则（与 qsrc_orchard.py QRULES['Q1_minimal'] 一致；待批准）
QRULE = dict(name='Q1_minimal',
             desc="排除 grade == 'C_专题产品(存疑)'（柑橘 2020；08 报告判读真园率 13.8%，源标'存疑'）",
             exclude_grades=['C_专题产品(存疑)'])
# 候选新框（原 4 区外；只按参考 n/地理；须质量规则在窗内实际生效）
CAND = dict(rid='GD2', prov='广东省', ptag='粤', win='w3',
            bbox=(113.0, 23.0, 114.0, 24.0), tile='T2002', tile_box=(113.0, 22.0, 115.0, 24.0),
            rationale='原 4 区外新瓦 T2002；参考 n=592（非fcs10 409，类4 92）；'
                      '窗内含柑橘（Q1 实际剔除 ~75%）——质量规则在此有作用（川新框窗内无柑橘，规则无效）。')


def emit(m):
    print('[%s] %s' % (time.strftime('%m-%d %H:%M:%S'), m), flush=True)


def block_id(lon, lat, step=0.25):
    bx = np.floor(np.asarray(lon, float) / step).astype('int64')
    by = np.floor(np.asarray(lat, float) / step).astype('int64')
    return np.array(['%d_%d' % (a, b) for a, b in zip(bx, by)], dtype=object)


def main():
    names = VC.V31_NAMES()
    # 参考点
    ev = pd.read_parquet(os.path.join(EVAL_DIR, 'eval_2023_%s.parquet' % CAND['win']),
                         columns=['row_id', 'lon', 'lat', 'class_new', 'src'])
    keep, xd = C4.build_exclusions(ev)
    ev = ev[keep].reset_index(drop=True)
    prov, _ = C4.assign_prov(ev.lon.to_numpy(), ev.lat.to_numpy(), list(PC.PROV_SHORT))
    ev['pf'] = prov
    ev['t24'] = VC.to_v31(ev.class_new.to_numpy().astype('int64'))
    ev['isf'] = PC.is_fcs10(ev.src).to_numpy()
    x0, y0, x1, y1 = CAND['bbox']
    mref = (ev.pf == CAND['prov']) & (ev.lon >= x0) & (ev.lon < x1) & (ev.lat >= y0) & (ev.lat < y1)
    ref = ev[mref].copy()
    ref['block'] = block_id(ref.lon.to_numpy(), ref.lat.to_numpy())
    blocks = sorted(ref.block.unique().tolist())
    # 稳定哈希定序 + 贪心 60/40
    cnt = ref.block.value_counts().to_dict()
    order = sorted(blocks, key=lambda b: hashlib.md5(b.encode()).hexdigest())
    dev, acc = [], 0
    for b in order:
        if acc < 0.6 * len(ref) or not dev:
            dev.append(b); acc += int(cnt.get(b, 0))
        else:
            break
    dev_set = set(dev)
    conf = [b for b in order if b not in dev_set]

    # 池窗 + 质量规则
    pool = pq.read_table(POOL2, columns=['point_id', 'lon', 'lat', 'cl']).to_pandas()
    tx0, ty0, tx1, ty1 = CAND['tile_box']
    tb = (tx0 - 2.0, ty0 - 2.0, tx1 + 2.0, ty1 + 2.0)
    mw = ((pool.lon >= tb[0]) & (pool.lon < tb[2]) & (pool.lat >= tb[1]) & (pool.lat < tb[3]))
    win = pool[mw].copy()
    win['block'] = block_id(win.lon.to_numpy(), win.lat.to_numpy())
    excl_blocks = set(dev) | set(conf)
    win_k = win[~win.block.isin(excl_blocks)].copy()
    n_win_before, n_win_after = len(win), len(win_k)
    # 质量 grade 回接（cl2 ↔ gpkg uid）
    g = pd.read_sql_query('select uid,grade,subtype,srcset from "乔灌园地v7_2023_WGS84"',
                          sqlite3.connect(GPKG))
    g = g.set_index(g.uid.astype(str))
    gm = g.reindex(win_k.point_id.astype(str).to_numpy())
    win_k['grade'] = gm.grade.to_numpy()
    win_k['subtype'] = gm.subtype.to_numpy()
    n_orch = int((win_k.cl == 2).sum())
    q1_mask = (~win_k.grade.isin(QRULE['exclude_grades'])).to_numpy()
    n_q1 = int(((win_k.cl == 2) & q1_mask).sum())
    # K = min(基线抽样中的园地数, Q1 可用数)；使两臂同 count、非园地完全一致
    base_orch_expected = int(round(N_LIMIT * n_orch / max(n_win_after, 1)))
    K = int(min(base_orch_expected, n_q1))
    # 位置键零重叠
    def pk(lon, lat, nd=6):
        a = np.round(np.asarray(lon, float) * 10 ** nd).astype('int64') + 200_000_000
        b = np.round(np.asarray(lat, float) * 10 ** nd).astype('int64') + 100_000_000
        return a * 400_000_000 + b
    inter = len(set(pk(win_k.lon.to_numpy(), win_k.lat.to_numpy()).tolist()) &
                set(pk(ref.lon.to_numpy(), ref.lat.to_numpy()).tolist()))

    reg = dict(
        schema='improve_pilot_qarm/v1', status='BLOCKED_no_trusted_local_source',
        generated=time.strftime('%Y-%m-%d %H:%M:%S'),
        note='高证据园地质量臂注册准备。**未运行**；待主 agent 批准规则/新框后再跑。',
        spec='doc70 v1.0 + doc69 §7.2 + 08 报告',
        region=dict(rid=CAND['rid'], prov=CAND['prov'], ptag=CAND['ptag'], win=CAND['win'],
                    bbox=list(CAND['bbox']), tile=CAND['tile'], training_bbox=list(tb),
                    rationale=CAND['rationale']),
        reference=dict(refn=int(len(ref)), refn_nonfcs10=int((~ref.isf).sum()),
                       n_dev=int(ref.block.isin(dev).sum()), n_conf=int(ref.block.isin(conf).sum()),
                       n_conf_cl4=int((ref[ref.block.isin(conf)].t24 == 4).sum()),
                       ref_by_class={int(c): int(v) for c, v in ref.t24.value_counts().sort_index().items()},
                       dev_blocks=dev, conf_blocks=conf, block_step=0.25),
        quality_rule=QRULE,
        arms=dict(
            random_orchard='取窗内前 K 个园地点（按 seed permutation 序）；非园地取前 (20000-K) 个，完全一致',
            quality_orchard='取窗内前 K 个通过质量规则（grade∉C_专题产品(存疑)）的园地点（同 permutation 序）；'
                            '非园地取前 (20000-K) 个，与 random 臂**完全相同**'),
        K=K, K_basis=dict(base_orch_expected_in_20k=base_orch_expected, n_q1_available=n_q1,
                          shortfall=(base_orch_expected - K)),
        window=dict(n_window_before=n_win_before, n_window_after=n_win_after,
                    n_orchard=n_orch, n_q1=n_q1, n_citrus_excluded=n_orch - n_q1,
                    orchard_by_grade={k: int(v) for k, v in win_k.grade.value_counts().items()},
                    poskey_overlap_train_ref=inter),
        seeds=SEEDS, n_train_limit=N_LIMIT,
        rf_params=dict(n_estimators=100, min_samples_leaf=2, max_leaf_nodes=5000, n_jobs=10),
        deadline_s=1200, target_metrics=['class4_F1', 'class1_F1', 'OA24', 'macro9', 'source_stratified'],
        main_gate='非FCS10 确认层 macro9 非劣 + 目标类 F1 改善（同首轮）',
        stop_rules='TIMEOUT / 组级回归 / 目标类无改善 → 不采纳；预测类 2 归零记回归风险',
        not_rerun_same_confirm=True,
        assessment=dict(
            trusted_local_source=False,
            verdict='BLOCKED_no_trusted_local_source',
            detail='grade 高tier（A_官方地块 37,733 / A_三源 13,024 / B_双源 11,003）全部是**海南橡胶**，'
                   '按 doc69 §7.2 不可迁移到川/粤；其余专题源（茶 6.2%、苹果 0%、柑橘 13.8%，08 报告判读）'
                   '在 10 m 尺度真园率均低且未经独立核验。故 川/粤 无"可信本地质量源"；'
                   'Q1（仅排 C_专题产品(存疑)）是弱先验干预，非真实质量提升。',
            recommendation='按协调者纪律：H2 在本轮 川/粤 判 BLOCKED；不擅自排 C 全量当新臂。'
                           'metadata 与 prepared 臂规格留档，待主 agent 决定是否另立新证据轮次。',
        ),
    )
    os.makedirs(os.path.join(WS, 'prereg'), exist_ok=True)
    json.dump(reg, open(os.path.join(WS, 'prereg', 'qarm_registration.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1, default=str)
    pd.DataFrame([dict(rid=CAND['rid'], bbox=str(CAND['bbox']), tile=CAND['tile'], refn=len(ref),
                       refn_nonfcs10=int((~ref.isf).sum()), n_conf_cl4=reg['reference']['n_conf_cl4'],
                       n_window=n_win_after, n_orchard=n_orch, n_q1=n_q1, K=K,
                       shortfall=base_orch_expected - K)]).to_csv(
        os.path.join(WS, 'explore', 'e6_qarm_candidates.csv'), index=False, encoding='utf-8-sig')
    emit('→ prereg/qarm_registration.json（status=prepared_awaiting_approval）')
    emit('  %s %s refn=%d 非=%d conf类4=%d | 窗园地=%d Q1=%d 柑橘剔除=%d | K=%d 缺口=%d | 位置键重叠=%d' % (
        CAND['rid'], CAND['tile'], len(ref), int((~ref.isf).sum()), reg['reference']['n_conf_cl4'],
        n_orch, n_q1, n_orch - n_q1, K, base_orch_expected - K, inter))


if __name__ == '__main__':
    main()
