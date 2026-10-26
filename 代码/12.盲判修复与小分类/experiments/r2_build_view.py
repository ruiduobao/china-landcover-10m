# -*- coding: utf-8 -*-
"""r2_build_view.py — 盲判修复与小分类·样本视图构建（共享上下文 C + R；零 RF）

* 输入：<WS>/materials/R_repair.parquet、V_eval.parquet（materials 子 agent 产出，只读）
        F:/lc_work/prod2nd_2023/data/train2023_v2.parquet（只读池，R 的 AEF 回接来源）
        F:/lc_work/v31_exp/data/yearly/eval_2023_w{2,3,4}.parquet（reference blocks 来源，只读）
* 规则源：prereg/label_repair_prereg.json（冻结）+ doc72 §2/§6/§7
* 门槛（写死，禁止事后改）：
    - 训练窗 = tile±2° − 全部 V blocks − 已用 reference blocks；record 泄漏 0
    - R 合格：置信∈{high,medium}、passes==2、review_status 一致、temporal 支持 2023、scope=point
    - 修复规则机械执行：针阔不清不改；常绿/落叶须当年物候；园地须管理时序证据
    - 两臂 C 相同、R 点集相同：baseline.v2=C∪R(label_before)；view-repair=C∪R(label_after/isolate+同窗同序回填)
    - selected unique R == 全部合格 R 且 >0；R 占比 ≤ probe_r_share_cap；禁止复制 R 当独立点
* 输出：<WS>/cache/view_<region>_<size>.npz（各臂 X/y、V 评价 X（**无真值**）、selectedIDs、块、flags）
        <WS>/cache/view_manifest.json（哈希与全部自检证据）
* 用法：
    python r2_build_view.py --size probe            # N=2000
    python r2_build_view.py --size real20k          # N=20000（唯一一次，仅旁证）
    python r2_build_view.py --size probe --regions SC_F,SC_O
"""
import argparse
import hashlib
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import lr_common as LC  # noqa: E402

# ---- 类别轴（机械执行修复规则；与 doc72 §4 一致） ----
FOREST = {4, 5, 6, 7, 8}
NEEDLE = {6, 7, 8}
BROAD = {4, 5}
EVERGREEN = {4, 6, 8}
DECIDUOUS = {5, 7}
ORCHARD = 2

REVIEW_OK = {'agree'}
TEMPORAL_OK = {'applicable_2023', 'ok', 'supported'}


def emit(m):
    print('[%s] %s' % (time.strftime('%m-%d %H:%M:%S'), m), flush=True)


# --------------------------------------------------------------------------- #
# 修复规则机械执行（冻结规则 → 逐点裁决）
# --------------------------------------------------------------------------- #
def validate_repair(label_before: int, label_after, action: str, evidence_type: str,
                    scope: str = 'point') -> tuple:
    """返回 (ok, reason)。action∈{relabel,isolate,keep}。规则冻结，见 prereg.repair_rules。"""
    if scope != 'point':
        return False, 'scope 必须=point（禁止扩整地块/山体）'
    if action == 'keep':
        return True, 'keep'
    if action == 'isolate':
        return True, 'isolate'
    if action != 'relabel' or label_after is None or int(label_after) == int(label_before):
        return False, 'relabel 无效或未改变类别'
    a, b = int(label_before), int(label_after)
    # 针阔轴变更：**允许**，但须显式高分结构复判一致证据 + 2023 适用；
    # 不得由 S2/物候/坡向推针阔（doc72 §4/§5）。
    if a in FOREST and b in FOREST and ((a in NEEDLE) != (b in NEEDLE)):
        if evidence_type != 'highres_structure_recheck':
            return False, '针阔变更须显式高分结构复判一致证据（不得由 S2/物候/坡向推针阔）'
    # 常绿/落叶轴（同针阔侧）：必须有当年物候证据
    if a in FOREST and b in FOREST and ((a in EVERGREEN) != (b in EVERGREEN)):
        if evidence_type != 'phenology_sameyear':
            return False, '常绿/落叶变更缺当年物候证据'
    # 园地进出：必须有管理/时序证据
    if a == ORCHARD or b == ORCHARD:
        if evidence_type != 'orchard_mgmt_timeseries':
            return False, '园地变更缺管理/时序证据'
    return True, 'ok'


def _eligibility_mask(R: pd.DataFrame) -> np.ndarray:
    """R 合格性（doc72 §5）：置信/复判/时序/scope。"""
    n = len(R)
    m = np.ones(n, bool)
    if 'confidence' in R:
        m &= R['confidence'].isin(['high', 'medium']).to_numpy()
    if 'passes' in R:
        m &= (R['passes'].to_numpy() == 2)
    if 'review_status' in R:
        m &= R['review_status'].isin(REVIEW_OK).to_numpy()
    if 'temporal_status' in R:
        m &= R['temporal_status'].isin(TEMPORAL_OK).to_numpy()
    if 'scope' in R:
        m &= (R['scope'].astype(str).to_numpy() == 'point')
    return m


# --------------------------------------------------------------------------- #
# 视图构建（纯函数：可被 r1_selfcheck 用合成数据直接测试）
# --------------------------------------------------------------------------- #
def build_arms(pool_win: pd.DataFrame, R: pd.DataFrame, V: pd.DataFrame,
               region: dict, prereg: dict, n_total: int, rng_seed: int,
               v_blocks, ref_blocks=None) -> dict:
    """构建两臂训练视图。

    pool_win: 训练窗内池点（未过滤；本函数自行剔除 v_blocks ∪ ref_blocks 作为上下文 C 的候选）
    v_blocks: V 所在 0.25° 块（R 与 C 都不得使用；保证评价空间独立）
    ref_blocks: 已用评价保留块（仅 C 候选剔除；**R 可落在参考块内**，因 R 是训练审计修复组）
    返回 dict(...) 或 {empty, reason}。
    """
    errs = []
    v_blocks = set(v_blocks)
    ref_blocks = set(ref_blocks or [])
    R = R.copy()
    lb = R['label_before'].to_numpy()
    la = R['label_after'].to_numpy() if 'label_after' in R else np.array([None] * len(R), object)
    act = R['action'].astype(str).to_numpy()
    ev = R['evidence_type'].astype(str).to_numpy() if 'evidence_type' in R else \
        np.array(['structural'] * len(R), object)
    sc = R['scope'].astype(str).to_numpy() if 'scope' in R else np.array(['point'] * len(R), object)
    R['ok_rule'], R['rule_reason'] = zip(*[
        validate_repair(o, n, a, e, s) for o, n, a, e, s in zip(lb, la, act, ev, sc)])
    elig = _eligibility_mask(R) & R['ok_rule'].to_numpy()
    R_elig = R[elig].copy()
    R_rej = R[~elig]
    if 'is_sealed' in R_elig.columns:
        R_elig = R_elig[~R_elig['is_sealed'].astype(bool)]
    R_elig['_pk'] = LC.pos_key(R_elig.lon, R_elig.lat)
    # R 可落在已用参考块内（训练审计组），但不得落在 V 块（否则污染评价空间）
    n_R_in_ref = int(R_elig['block'].isin(ref_blocks).sum())
    R_elig = R_elig[~R_elig['block'].isin(v_blocks)]
    n_eligible = len(R_elig)
    if n_eligible == 0:
        return dict(empty=True, reason='NO_REPAIR_SUPPORT: 本区无合格 R 点', errors=errs)

    # ---- R 占比上限（稳定哈希定序截断；禁止复制 R） ----
    cap = float(prereg['design']['probe_r_share_cap'])
    r_max = int(np.floor(cap * n_total))
    if n_eligible > r_max:
        order = np.argsort([hashlib.md5(str(u).encode()).hexdigest() for u in R_elig.sample_uid])
        R_used = R_elig.iloc[order[:r_max]].copy()
    else:
        R_used = R_elig.copy()
    n_R = len(R_used)

    # ---- 共享上下文 C（同窗、固定 rng；剔 V 块+已用参考块；排除 R 位置避免重复计点） ----
    r_pos = set(R_used['_pk'].tolist())
    win = pool_win.copy()
    win = win[~win['block'].isin(v_blocks | ref_blocks)].copy()   # C 候选：剔 V 块 + 已用参考块
    win['_pk'] = LC.pos_key(win.lon, win.lat)
    win = win[~win['_pk'].isin(r_pos)].reset_index(drop=True)
    n_C = int(n_total - n_R)
    if len(win) < n_C:
        return dict(empty=True, reason='NO_REPAIR_SUPPORT: 训练窗剩余点不足 C',
                    n_window=len(win), n_C=n_C, errors=errs)
    rng = np.random.RandomState(rng_seed)
    perm = rng.permutation(len(win))          # 同一预注册随机序，C 与回填共用
    C_idx = perm[:n_C]

    # ---- 两臂点集与标签 ----
    iso = R_used['action'].astype(str).values == 'isolate'
    n_iso = int(iso.sum())
    back_idx = perm[n_C:n_C + n_iso]          # 回填：紧接 C、同一 perm 顺序
    if len(back_idx) < n_iso:
        errs.append('回填点不足，candidate 计数将 < baseline')
        back_idx = back_idx

    C_ids = win.loc[C_idx, 'point_id'].to_numpy()
    back_ids = win.loc[back_idx, 'point_id'].to_numpy()
    base_ids = np.concatenate([C_ids, R_used['sample_uid'].to_numpy()])
    base_y = np.concatenate([win.loc[C_idx, 'cl'].to_numpy(),
                             R_used['label_before'].to_numpy()]).astype(int)
    keep_mask = ~iso
    cand_ids = np.concatenate([C_ids, R_used.loc[keep_mask, 'sample_uid'].to_numpy(), back_ids])
    la_keep = R_used.loc[keep_mask, 'label_after'].to_numpy()
    act_keep = R_used.loc[keep_mask, 'action'].astype(str).to_numpy()
    cand_y = np.concatenate([
        win.loc[C_idx, 'cl'].to_numpy(),
        np.where(act_keep == 'relabel', la_keep, R_used.loc[keep_mask, 'label_before'].to_numpy()),
        win.loc[back_idx, 'cl'].to_numpy()]).astype(int)

    # ---- 断言 ----
    checks = {}
    r_keep_ids = R_used.loc[keep_mask, 'sample_uid'].to_numpy()
    checks['C_identical'] = True
    checks['R_kept_ids_in_both'] = bool(set(r_keep_ids.tolist()).issubset(set(base_ids.tolist())))
    iso_ids = R_used.loc[iso, 'sample_uid'].to_numpy()
    checks['isolated_absent_in_candidate'] = bool(len(set(iso_ids.tolist()) &
                                                    set(cand_ids.tolist())) == 0)
    checks['backfill_disjoint'] = bool(len(set(back_ids.tolist()) &
                                          (set(C_ids.tolist()) | set(R_used['sample_uid'].tolist()))) == 0)
    checks['all_R_selected'] = bool(n_R == n_eligible)
    checks['R_nonempty'] = bool(n_R > 0)
    checks['n_baseline'] = int(len(base_ids))
    checks['n_candidate'] = int(len(cand_ids))
    checks['same_count'] = bool(len(base_ids) == len(cand_ids))
    checks['n_R_used'] = int(n_R)
    checks['n_R_eligible'] = int(n_eligible)
    checks['r_share'] = round(n_R / float(max(len(base_ids), 1)), 4)
    checks['r_share_cap'] = cap
    checks['r_share_ok'] = bool(checks['r_share'] <= cap + 1e-9)
    checks['n_isolate'] = n_iso
    checks['n_relabel'] = int((R_used['action'].astype(str).values == 'relabel').sum())
    checks['n_keep'] = int((R_used['action'].astype(str).values == 'keep').sum())
    bc = pd.Series(base_y).value_counts().sort_index()
    cc = pd.Series(cand_y).value_counts().sort_index()
    checks['class_prior_delta'] = {int(k): int(cc.get(k, 0) - bc.get(k, 0))
                                   for k in sorted(set(bc.index) | set(cc.index))}
    checks['unique_ids'] = bool(len(set(base_ids.tolist())) == len(base_ids) ==
                                len(set(cand_ids.tolist())) == len(cand_ids))
    v_pos = set(LC.pos_key(V.lon, V.lat).tolist())
    train_pos = set(LC.pos_key(win.lon, win.lat).tolist()) | set(r_pos)
    checks['V_not_in_train'] = bool(len(v_pos & train_pos) == 0)
    checks['C_no_forbidden_block'] = bool(
        len(set(win['block'].tolist()) & (v_blocks | ref_blocks)) == 0)
    checks['R_not_in_V_blocks'] = bool(len(set(R_used['block'].tolist()) & v_blocks) == 0)
    checks['n_R_in_ref_blocks'] = int(n_R_in_ref)      # 允许：R 是训练审计组，记录不作失败
    checks['R_blocks_disjoint_V_blocks'] = bool(
        len(set(R_used['block'].tolist()) & set(V['block'].tolist())) == 0)
    hard = ['R_kept_ids_in_both', 'isolated_absent_in_candidate', 'backfill_disjoint',
            'all_R_selected', 'R_nonempty', 'unique_ids', 'V_not_in_train',
            'C_no_forbidden_block', 'R_not_in_V_blocks', 'R_blocks_disjoint_V_blocks']
    checks['hard_pass'] = all(bool(checks[k]) for k in hard)
    if not checks['hard_pass']:
        errs.append('硬自检未通过：' + ','.join(k for k in hard if not bool(checks[k])))

    return dict(empty=False, win=win, C_idx=C_idx, back_idx=back_idx, R_used=R_used,
                R_rej=R_rej, base_ids=base_ids, base_y=base_y, cand_ids=cand_ids,
                cand_y=cand_y, r_pos=r_pos, checks=checks, errors=errs)


# --------------------------------------------------------------------------- #
# 真实数据装配
# --------------------------------------------------------------------------- #
def load_reference_blocks() -> set:
    blocks = set()
    for w in LC.REF_WINDOWS:
        fp = os.path.join(LC.EVAL_DIR, 'eval_2023_%s.parquet' % w)
        if os.path.exists(fp):
            d = pd.read_parquet(fp, columns=['lon', 'lat'])
            blocks |= set(LC.block_id(d.lon.to_numpy(), d.lat.to_numpy()).tolist())
    return blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', choices=['probe', 'real20k'], default='probe')
    ap.add_argument('--regions', default='SC_F,SC_O')
    a = ap.parse_args()
    LC.ensure_dirs()
    cfg = LC.load_prereg()
    n_total = cfg['design']['n_total_sizes'][a.size]
    emit('size=%s n_total=%d' % (a.size, n_total))

    Rfp = os.path.join(LC.MAT, 'R_repair.parquet')
    Vfp = os.path.join(LC.MAT, 'V_eval.parquet')
    if not (os.path.exists(Rfp) and os.path.exists(Vfp)):
        emit('缺 materials（%s / %s）—— 等 materials 子 agent 产出后再跑；不假造 R/V。'
             % (Rfp, Vfp))
        return 2
    R = pd.read_parquet(Rfp)
    V = pd.read_parquet(Vfp)
    bad = set(cfg['materials']['V_columns_forbidden_in_experiment']) & set(V.columns)
    if bad:
        emit('拒绝：V 表含禁止真值列 %s（V 真值由 audit 封存）' % sorted(bad))
        return 3

    ref_blocks = load_reference_blocks()
    emit('reference blocks = %d' % len(ref_blocks))
    pool = pd.read_parquet(LC.POOL2, columns=['point_id', 'lon', 'lat', 'cl', 'src'] + LC.FEATS)
    emit('池装载 %d 点 / %d 类' % (len(pool), pool.cl.nunique()))
    assert pool.cl.min() >= 1 and pool.cl.max() <= 24, '池 cl 非 24 码空间'

    manifest = dict(size=a.size, n_total=n_total,
                    generated=time.strftime('%Y-%m-%d %H:%M:%S'),
                    prereg_md5=LC.md5_file(LC.PREREG), regions={}, honesty=[
                        'R=训练审计修复点，可入训练但绝不评价；V=独立块新评价点，唯一评价集',
                        'V 真值由 audit 封存，本 npz 不含真值；预测表只写 pred',
                        'probe 非原生产 20k 结论；real20k 仅一次受控旁证',
                        '本地 sklearn RF ≠ GEE smileRF'])
    for rid in [r.strip() for r in a.regions.split(',') if r.strip()]:
        Rreg = cfg['regions'].get(rid)
        if Rreg is None:
            emit('跳过未注册区域 %s' % rid); continue
        tx0, ty0, tx1, ty1 = Rreg['tile_box']
        tb = (tx0 - 2.0, ty0 - 2.0, tx1 + 2.0, ty1 + 2.0)
        win = pool[(pool.lon >= tb[0]) & (pool.lon < tb[2]) &
                   (pool.lat >= tb[1]) & (pool.lat < tb[3])].copy()
        win['block'] = LC.block_id(win.lon.to_numpy(), win.lat.to_numpy())
        Rr = R[R.rid == rid].copy() if 'rid' in R.columns else R.iloc[0:0].copy()
        Vr = V[V.rid == rid].copy() if 'rid' in V.columns else V.iloc[0:0].copy()
        if len(Vr) == 0:
            emit('%s：无 V 点 → NO_EVAL_SUPPORT，跳过' % rid)
            manifest['regions'][rid] = dict(empty=True, reason='NO_EVAL_SUPPORT')
            continue
        v_blocks = set(Vr['block'].tolist())
        res = build_arms(win, Rr, Vr, Rreg, cfg, n_total, rng_seed=7,
                         v_blocks=v_blocks, ref_blocks=ref_blocks)
        if res.get('empty'):
            emit('%s：%s' % (rid, res['reason']))
            manifest['regions'][rid] = dict(empty=True, reason=res['reason'])
            continue
        w = res['win']

        def X_of(idx):
            return w.loc[idx, LC.FEATS].to_numpy(np.float32)

        # R 的 AEF：优先池按 point_id 回接（权威）；否则用 R 自带 AEF
        r_aef = None
        if 'point_id' in res['R_used'].columns:
            m = pool.set_index('point_id')[LC.FEATS]
            r_aef = m.reindex(res['R_used']['point_id'].to_numpy())[LC.FEATS].to_numpy(np.float32)
        if (r_aef is None or np.isnan(r_aef).any()) and all(f in res['R_used'] for f in LC.FEATS):
            r_aef = res['R_used'][LC.FEATS].to_numpy(np.float32)
        if r_aef is None or np.isnan(r_aef).any():
            emit('%s：R 缺 AEF 回接（需 materials 提供 point_id 或 AEF 列）→ 跳过' % rid)
            manifest['regions'][rid] = dict(empty=True, reason='R_AEF_MISSING')
            continue

        base_X = np.vstack([X_of(res['C_idx']), r_aef])
        keep_mask = res['R_used']['action'].astype(str).values != 'isolate'
        cand_X = np.vstack([X_of(res['C_idx']), r_aef[keep_mask], X_of(res['back_idx'])])
        ev_X = Vr[LC.FEATS].to_numpy(np.float32)
        isf = Vr['is_fcs10'].to_numpy(np.int64) if 'is_fcs10' in Vr else np.zeros(len(Vr), np.int64)
        out = os.path.join(LC.CACHE, 'view_%s_%s.npz' % (rid, a.size))
        np.savez_compressed(
            out,
            base_X=base_X, base_y=res['base_y'], cand_X=cand_X, cand_y=res['cand_y'],
            eval_X=ev_X, eval_block=Vr['block'].to_numpy(dtype=object).astype(str),
            eval_is_fcs10=isf, eval_lon=Vr['lon'].to_numpy(np.float64),
            eval_lat=Vr['lat'].to_numpy(np.float64),
            eval_vid=Vr['vid'].to_numpy(dtype=object).astype(str),
            eval_judge=Vr['judge_status'].to_numpy(dtype=object).astype(str)
            if 'judge_status' in Vr else np.array(['unknown'] * len(Vr), object).astype(str),
            base_ids=res['base_ids'].astype(str), cand_ids=res['cand_ids'].astype(str),
            C_ids=w.loc[res['C_idx'], 'point_id'].to_numpy().astype(str),
            back_ids=w.loc[res['back_idx'], 'point_id'].to_numpy().astype(str),
            R_ids=res['R_used']['sample_uid'].to_numpy().astype(str),
            R_action=res['R_used']['action'].astype(str).to_numpy().astype(str),
            R_before=res['R_used']['label_before'].to_numpy(np.int64),
            R_after=res['R_used']['label_after'].fillna(-1).to_numpy(np.int64),
        )
        manifest['regions'][rid] = dict(
            npz=out, md5=LC.md5_file(out), n_window=len(win), n_eval_V=len(Vr),
            checks=res['checks'], errors=res['errors'], rule_rejected=int(len(res['R_rej'])))
        emit('%s → %s（V %d；R %d；hard_pass=%s；R_share=%.4f）' % (
            rid, os.path.basename(out), len(Vr), res['checks']['n_R_used'],
            res['checks']['hard_pass'], res['checks']['r_share']))
    DL, _ = LC.import_control()
    DL.atomic_write_json(os.path.join(LC.CACHE, 'view_manifest.json'), manifest)
    emit('→ cache/view_manifest.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
