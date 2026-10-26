# -*- coding: utf-8 -*-
"""r0_prereg.py — 盲判修复与小分类·预注册校验 / 冻结 / 目标冻结（零 RF / 零样本）

* 输入：代码/12.盲判修复与小分类/prereg/label_repair_prereg.json
* 规则源：doc72（主 agent 冻结）+ 协调者 R/V 角色与防稀释指令
* 门槛（硬约束，改这里必须同步主 agent）：
    year=2023；seeds=[7,11,23]；RF 100/leaf2/nodes5000；n_train_limit=20000；
    design.n_total_sizes={probe:2000,real20k:20000}；probe_r_share_cap≤0.10；
    planned_fits=24；arms 恰为 {baseline.v2,view-repair}；roles R120/V80（R 不可评价、V 不可训练）；
    structure_groups/axes/tasks 齐备；SC_F/SC_O 非 geo_only；pred 表禁真值列；
    limit_s=1200；n_jobs=10；cpu_cap=15；mem_cap_gb=10；record_leak=0
* 输出：--check 校验；--freeze 写 WS/prereg/label_repair_prereg.frozen.json（不改正文）；
        --freeze-targets 写 WS/prereg/targets.frozen.json（R QC 后、看 V 真值前）
* 用法：
    python r0_prereg.py --check
    python r0_prereg.py --freeze --by "<main-agent-id>"
    python r0_prereg.py --freeze-targets --targets-json <file> --by "<id>"
    python r0_prereg.py --verify
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import lr_common as LC  # noqa: E402

FROZEN = dict(
    year=2023, seeds=[7, 11, 23], n_train_limit=20000,
    n_trees=100, min_leaf=2, max_nodes=5000,
    limit_s=1200, n_jobs=10, cpu_cap=15, mem_cap_gb=10,
    arms={'baseline.v2', 'view-repair'},
    sizes={'probe': 2000, 'real20k': 20000}, r_share_cap_max=0.10, planned_fits=24,
    forbid_pred_cols={'cl_true', 'true_label', 't24', 'label_after'},
    required_real_regions={'SC_F', 'SC_O'},
    roles={'R': (120, True, False), 'V': (80, False, True)},   # quota, can_train, can_eval
    structure_groups=['orchard', 'woody_forest', 'cropland'],
    tasks={'SC_F': 'needle_broadleaf_axis', 'SC_O': 'orchard_vs_forestcrop'},
)


def validate(cfg):
    errs, warns = [], []
    if cfg.get('year') != FROZEN['year']:
        errs.append('year 必须 = 2023')
    if cfg.get('seeds') != FROZEN['seeds']:
        errs.append('seeds 必须 = [7,11,23]（三 seed 配对）')
    m = cfg.get('model', {})
    for k, fk in (('n_estimators', 'n_trees'), ('min_samples_leaf', 'min_leaf'),
                  ('max_leaf_nodes', 'max_nodes'), ('n_jobs', 'n_jobs')):
        if m.get(k) != FROZEN[fk]:
            errs.append(f'model.{k}={m.get(k)} != 冻结 {FROZEN[fk]}')
    if cfg.get('n_train_limit') != FROZEN['n_train_limit']:
        errs.append('n_train_limit 必须 = 20000')
    des = cfg.get('design', {})
    if des.get('n_total_sizes') != FROZEN['sizes']:
        errs.append('design.n_total_sizes 必须 = {probe:2000, real20k:20000}')
    if not (0 < des.get('probe_r_share_cap', 1) <= FROZEN['r_share_cap_max']):
        errs.append('design.probe_r_share_cap 必须 ≤ 0.10（R 占比上限）')
    if des.get('planned_fits') != FROZEN['planned_fits']:
        errs.append('design.planned_fits 必须 = 24（2 size × 2 region × 2 arm × 3 seed）')
    arms = {a.get('id') for a in cfg.get('arms', [])}
    if arms != FROZEN['arms']:
        errs.append(f'arms 必须恰为 {sorted(FROZEN["arms"])}（不扫权重/阈值）')
    # 角色
    roles = cfg.get('sample_roles', {})
    for rk, (q, ct, ce) in FROZEN['roles'].items():
        r = roles.get(rk, {})
        if r.get('quota') != q:
            errs.append(f'sample_roles.{rk}.quota 必须 = {q}')
        if bool(r.get('can_train')) != ct or bool(r.get('can_eval')) != ce:
            errs.append(f'sample_roles.{rk} 训练/评价权限必须 can_train={ct}, can_eval={ce}')
    # 资源
    res = cfg.get('resources', {})
    for k, fk in (('limit_s', 'limit_s'), ('n_jobs', 'n_jobs'),
                  ('cpu_cap', 'cpu_cap'), ('mem_cap_gb', 'mem_cap_gb')):
        if res.get(k) != FROZEN[fk]:
            errs.append(f'resources.{k}={res.get(k)} != 冻结 {FROZEN[fk]}')
    # 评价结构
    ev = cfg.get('evaluation', {})
    sg = ev.get('structure_groups', {})
    for k in FROZEN['structure_groups']:
        if k not in sg:
            errs.append(f'evaluation.structure_groups 缺 {k}')
    if sg.get('orchard') != [2]:
        errs.append('structure_groups.orchard 必须 = [2]')
    if sg.get('woody_forest') != [4, 5, 6, 7, 8]:
        errs.append('structure_groups.woody_forest 必须 = [4,5,6,7,8]')
    if sg.get('cropland') != [1, 3]:
        errs.append('structure_groups.cropland 必须 = [1,3]')
    if 'axes' not in ev or 'needle_broadleaf' not in ev['axes']:
        errs.append('evaluation.axes.needle_broadleaf 必须存在（针阔轴）')
    tasks = ev.get('tasks', {})
    for rid, typ in FROZEN['tasks'].items():
        if tasks.get(rid, {}).get('type') != typ:
            errs.append(f'evaluation.tasks.{rid}.type 必须 = {typ}')
    forb = set(ev.get('pred_table_forbids', []))
    if not FROZEN['forbid_pred_cols'].issubset(forb):
        errs.append('evaluation.pred_table_forbids 必须含 cl_true/true_label/t24/label_after')
    if ev.get('class_min_n') != 30:
        errs.append('evaluation.class_min_n 必须 = 30')
    # 区域
    regs = cfg.get('regions', {})
    real = {k for k, v in regs.items() if not v.get('geo_only')}
    if not FROZEN['required_real_regions'].issubset(real):
        errs.append('必须含非 geo_only 区域 SC_F（四川林型）与 SC_O（四川园地）')
    # 修复规则 / 泄漏 / 封存
    rr = cfg.get('repair_rules', {})
    if not rr.get('needle_broadleaf'):
        errs.append('repair_rules 必须显式声明针阔不清不改')
    if not rr.get('frozen'):
        warns.append('repair_rules.frozen 应为 true')
    if not (cfg.get('training_selection') or {}).get('record_leak') == 0:
        errs.append('training_selection.record_leak 必须 = 0')
    if not (cfg.get('sealed_confirm') or {}).get('agent_must_not_read'):
        errs.append('必须声明 sealed_confirm.agent_must_not_read=true')
    tf = cfg.get('target_freeze', {})
    if tf.get('status') not in ('PENDING_R_QC', 'FROZEN'):
        warns.append('target_freeze.status 应为 PENDING_R_QC（R QC 后冻结目标）')
    return dict(ok=not errs, errors=errs, warnings=warns,
                n_errors=len(errs), n_warnings=len(warns))


def _code_hash():
    return LC.code_hash_of(os.path.join(HERE, 'lr_common.py'),
                           os.path.join(HERE, 'r0_prereg.py'),
                           os.path.join(HERE, 'r1_selfcheck.py'),
                           os.path.join(HERE, 'r2_build_view.py'),
                           os.path.join(HERE, 'r3_run_unit.py'))


def freeze(by):
    """冻结预注册：**不改正文**（改正文即改 md5），哈希写旁路 sidecar。"""
    cfg = LC.load_prereg()
    res = validate(cfg)
    if not res['ok']:
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 1
    prereg_md5 = LC.md5_file(LC.PREREG)
    os.makedirs(os.path.join(LC.WS, 'prereg'), exist_ok=True)
    out = LC.frozen_sidecar()
    DL, _ = LC.import_control()
    DL.atomic_write_json(out, dict(prereg_path=LC.PREREG, prereg_md5=prereg_md5,
                                   code_hash=_code_hash(),
                                   frozen_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                                   frozen_by=by, validate=res))
    print(json.dumps(dict(frozen=True, prereg_md5=prereg_md5, code_hash=_code_hash(),
                          sidecar=out, note='prereg 正文未改动；哈希在 sidecar'),
                     ensure_ascii=False, indent=2))
    return 0


def freeze_targets(by, targets_json=None):
    """R QC 后、看 V 真值前写死目标（sidecar）。"""
    cfg = LC.load_prereg()
    tf = cfg.get('target_freeze', {})
    body = dict(status='FROZEN', frozen_at=time.strftime('%Y-%m-%d %H:%M:%S'), frozen_by=by,
                note='目标在看 V 真值前写死；不得据 V 结果挑目标',
                fixed_now=tf.get('fixed_now'), structure_groups=cfg['evaluation']['structure_groups'],
                axes=cfg['evaluation']['axes'])
    if targets_json:
        with open(targets_json, 'r', encoding='utf-8') as f:
            body['targets'] = json.load(f)
    else:
        body['targets'] = tf.get('candidate_targets')
    out = LC.targets_sidecar()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    DL, _ = LC.import_control()
    DL.atomic_write_json(out, body)
    print(json.dumps(dict(targets_frozen=True, sidecar=out), ensure_ascii=False, indent=2))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--freeze', action='store_true')
    ap.add_argument('--freeze-targets', action='store_true')
    ap.add_argument('--targets-json', default=None)
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--by', default='unknown')
    a = ap.parse_args()
    if a.freeze:
        return freeze(a.by)
    if a.freeze_targets:
        return freeze_targets(a.by, a.targets_json)
    if a.verify:
        print(json.dumps(dict(prereg=LC.verify_frozen(), targets=LC.verify_targets_frozen()),
                         ensure_ascii=False, indent=2))
        return 0
    res = validate(LC.load_prereg())
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if res['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
