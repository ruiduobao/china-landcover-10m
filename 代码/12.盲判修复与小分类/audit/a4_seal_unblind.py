# -*- coding: utf-8 -*-
"""a4_seal_unblind.py — 主门冻结 → 预测封存 → V 一次解盲 → 聚合发布（顺序强制）

* 输入/子命令：
    gate    : 冻结主采纳门（目标类/层级/阈值）→ audit/registry/gate_freeze.json
              前置：R 修复 QC 报告（a3 输出）已存在；V 未解盲
    seal    : 登记 baseline/candidate 预测、选中 ID、解盲顺序日志哈希 → audit/registry/unblind_seal.json
              前置：gate_freeze.json 已存在（顺序：R QC → 主门 → 预测封存）
    unblind : 一次解封 V 真值 → audit/sealed/v_truth_release_<ts>.json（默认只出聚合）
              前置：status==SEALED 且 --confirm-once；二次调用默认拒绝
    release : 向主 agent 发布聚合（n<--suppress-small-n 的类级单元置空并计数）
              前置：已解盲
* 规则源：doc72 §7.2/§7.3（主门在 R QC 后、V 解封前写死；一次解盲；不用 V 找阈值/新增修复规则）
* 门槛：顺序违规即拒绝；重复解盲即拒绝（--force 才覆盖并留痕）；V 点级真值默认不发布
* 用法：python a4_seal_unblind.py gate --qc-report audit/out/qc_review_x.json --target-classes 4 2
        python a4_seal_unblind.py seal --preds results/base.csv results/cand.csv --selected-ids results/selected.csv
        python a4_seal_unblind.py unblind --confirm-once
        python a4_seal_unblind.py release --metrics results/v_metrics.json
* 说明：本脚本只管"顺序与封存"，不重算指标；指标由 a2/a3 口径与实验产物提供。
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

WS_DEFAULT = r'F:/lc_work/label_repair_2023_20261015'


def norm(p):
    return p.replace('\\', '/')


def fh(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def paths(a):
    ws = norm(a.ws)
    reg = os.path.join(ws, 'audit', 'registry')
    return ws, reg, os.path.join(reg, 'gate_freeze.json'), os.path.join(reg, 'unblind_seal.json')


def load(path):
    return json.load(open(path, encoding='utf-8')) if os.path.exists(path) else None


def save(path, rec):
    tmp = path + '.tmp'
    json.dump(rec, open(tmp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def cmd_gate(a):
    ws, reg, gpath, spath = paths(a)
    os.makedirs(reg, exist_ok=True)
    seal = load(spath)
    if seal and seal.get('status') == 'UNBLINDED':
        raise SystemExit('拒绝：V 已解盲，主门不能再冻结（doc72 §7.3 要求解盲前写死）')
    if not a.qc_report or not os.path.exists(a.qc_report):
        raise SystemExit('拒绝：缺少 R 修复 QC 报告（a3 输出）——主门须在 R QC 后冻结')
    qc = load(a.qc_report)
    rec = {
        'schema': 'gate_freeze/v1',
        'frozen_at': datetime.now().isoformat(timespec='seconds'),
        'frozen_by': 'audit_subagent（独立审计）',
        'order': 'R 修复 QC 完成 → 主门冻结 → 预测封存 → V 一次解盲',
        'qc_report': {'path': norm(a.qc_report), 'sha256': fh(a.qc_report),
                      'gate': (qc or {}).get('gate'),
                      'repair_supported': ((qc or {}).get('repair_eligibility') or {}).get('supported'),
                      'unknown_rate': ((qc or {}).get('unknown') or {}).get('rate')},
        'target': {'classes': a.target_classes, 'level': a.target_level,
                   'merge_map': a.merge_map},
        'thresholds': {'target_f1_delta_min': a.target_f1_min,
                       'noninferior_point_min': a.noninf_point,
                       'noninferior_ci_lower_min': a.noninf_lower,
                       'regression_f1_max_drop': a.regress_drop,
                       'min_class_n': a.min_class_n,
                       'seeds': a.seeds},
        'rules': ['判完不改阈值；确认只启用一次',
                  '类参考 n<%d 只作诊断，不作类级达标' % a.min_class_n,
                  '有限复核样本不能保证地表精度；V 聚合一致率不称地表真值精度',
                  '主门在 V 解盲前冻结，不得据 V 结果回改'],
    }
    if (qc or {}).get('gate') != 'PASS':
        rec['warning'] = ('QC gate=%s（未过 0.90 门）：门仍按协议冻结，但修复臂应记 NO_REPAIR_SUPPORT / '
                          'INSUFFICIENT_SUPPORT，不得因 V 结果放宽' % (qc or {}).get('gate'))
    save(gpath, rec)
    print('主门已冻结 → %s' % gpath)
    print('  目标类 %s（%s）｜targetF1Δ≥%.2f｜非劣点估≥%.2f/下界≥%.2f｜退化门≤-%.2f｜n<%d 只作诊断'
          % (a.target_classes, a.target_level, a.target_f1_min, a.noninf_point,
             a.noninf_lower, a.regress_drop, a.min_class_n))
    if rec.get('warning'):
        print('  警告：%s' % rec['warning'])


def cmd_seal(a):
    ws, reg, gpath, spath = paths(a)
    g = load(gpath)
    if not g:
        raise SystemExit('拒绝：未冻结主门（先跑 gate；顺序：R QC → 主门 → 封存）')
    old = load(spath)
    if old and old.get('status') == 'UNBLINDED' and not a.force:
        raise SystemExit('拒绝：已解盲，不能再封存（如需重记加 --force 并说明原因）')
    files = {}
    for p in (a.preds or []):
        if not os.path.exists(p):
            raise SystemExit('预测文件不存在：%s' % p)
        files[norm(p)] = {'sha256': fh(p), 'size': os.path.getsize(p)}
    for p in (a.selected_ids or []):
        if not os.path.exists(p):
            raise SystemExit('选中 ID 文件不存在：%s' % p)
        files[norm(p)] = {'sha256': fh(p), 'size': os.path.getsize(p)}
    if not files:
        raise SystemExit('拒绝：未提供任何预测/选中 ID 文件，无法封存')
    order = a.unblind_order
    if order and os.path.exists(order):
        order_rec = {'path': norm(order), 'sha256': fh(order)}
    else:
        order = os.path.join(reg, 'unblind_order_%s.json' % datetime.now().strftime('%Y%m%d_%H%M%S'))
        save(order, {'schema': 'unblind_order/v1',
                     'generated_at': datetime.now().isoformat(timespec='seconds'),
                     'rule': '解盲按此顺序一次性读取 V 真值；顺序在预测哈希之后落盘，不得事后重排'})
        order_rec = {'path': norm(order), 'sha256': fh(order), 'auto_generated': True}
    rec = {'schema': 'unblind_seal/v1', 'status': 'SEALED',
           'sealed_at': datetime.now().isoformat(timespec='seconds'),
           'gate_freeze': {'path': norm(gpath), 'sha256': fh(gpath)},
           'artifacts': files, 'unblind_order': order_rec,
           'note': 'V 真值仍封存于 audit/sealed/；实验代理只能拿到 R 合格标签与 V 坐标/特征/排除块'}
    if old:
        rec['supersedes'] = {'sealed_at': old.get('sealed_at'), 'status': old.get('status')}
    save(spath, rec)
    print('预测已封存（%d 个产物）→ %s' % (len(files), spath))
    print('  解盲顺序日志：%s' % order_rec['path'])


def cmd_unblind(a):
    ws, reg, gpath, spath = paths(a)
    seal = load(spath)
    if not seal or seal.get('status') != 'SEALED':
        raise SystemExit('拒绝：未封存预测或状态非 SEALED（当前：%s）' % (seal or {}).get('status'))
    if not a.confirm_once:
        raise SystemExit('拒绝：一次解盲需显式 --confirm-once')
    seal['status'] = 'UNBLINDED'
    seal['unblinded_at'] = datetime.now().isoformat(timespec='seconds')
    seal['unblind_count'] = int(seal.get('unblind_count', 0)) + 1
    if seal['unblind_count'] > 1 and not a.force:
        seal['unblind_count'] -= 1
        raise SystemExit('拒绝：本封存已解盲过（unblind_count=%d）。重复解盲需 --force 并写明原因' % seal['unblind_count'])
    sd = os.path.join(ws, 'audit', 'sealed')
    os.makedirs(sd, exist_ok=True)
    out = os.path.join(sd, 'v_truth_release_%s.json' % datetime.now().strftime('%Y%m%d_%H%M%S'))
    rec = {'schema': 'v_truth_release/v1',
           'released_at': seal['unblinded_at'],
           'release_scope': 'point_level' if a.point_level else 'aggregate_only',
           'point_level_reason': a.reason if a.point_level else None,
           'gate_freeze_sha256': seal['gate_freeze']['sha256'],
           'policy': ['解盲一次；此后不得用 V 结果寻找 cap/权重/阈值或新增修复规则（doc72 §7.2）',
                      'V 点级真值默认不外发；主 agent 只见聚合，不足以回指阈值',
                      'V 一致率不称地表真值精度（参考为 AI 预判读，未完成正式人工复核）'],
           'truth_source': norm(a.truth) if a.truth else None}
    if a.truth and os.path.exists(a.truth):
        rec['truth_sha256'] = fh(a.truth)
    save(out, rec)
    seal['last_release'] = {'path': norm(out), 'sha256': fh(out)}
    save(spath, seal)
    print('V 已解盲（第 %d 次）→ %s（scope=%s）' % (seal['unblind_count'], out, rec['release_scope']))


def cmd_release(a):
    ws, reg, gpath, spath = paths(a)
    seal = load(spath)
    if not seal or seal.get('status') != 'UNBLINDED':
        raise SystemExit('拒绝：尚未解盲，无 V 结果可发布')
    if not a.metrics or not os.path.exists(a.metrics):
        raise SystemExit('拒绝：需要 --metrics（V 评价聚合结果 JSON）')
    m = load(a.metrics)
    sup = []
    if isinstance(m, dict):
        for k, v in list(m.items()):
            if isinstance(v, dict) and isinstance(v.get('n'), (int, float)) and v['n'] < a.suppress_small_n:
                m[k] = {kk: (None if kk in ('ua', 'pa', 'f1', 'rate') else vv) for kk, vv in v.items()}
                sup.append(k)
    rec = {'schema': 'v_release_to_main/v1',
           'released_at': datetime.now().isoformat(timespec='seconds'),
           'suppress_small_n': a.suppress_small_n,
           'suppressed_units': sup,
           'metrics': m,
           'caveats': ['聚合一致率，非地表真值精度；参考为 AI 预判读，未完成正式人工复核',
                       '已解盲一次：不得据本结果调整阈值或新增修复规则',
                       'n<%d 的类级单元已置空（%d 个）' % (a.suppress_small_n, len(sup))],
           'adoption_gate': load(gpath)}
    out = a.out or os.path.join(ws, 'audit', 'review', 'v_release_to_main_%s.json'
                                % datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    save(out, rec)
    print('已向主 agent 发布聚合（置空 %d 个 n<%d 单元）→ %s' % (len(sup), a.suppress_small_n, out))


def main():
    ap = argparse.ArgumentParser(description='主门冻结 / 预测封存 / V 一次解盲 / 聚合发布')
    ap.add_argument('--ws', default=WS_DEFAULT)
    sub = ap.add_subparsers(dest='cmd', required=True)

    g = sub.add_parser('gate')
    g.add_argument('--qc-report', required=True)
    g.add_argument('--target-classes', nargs='*', type=int, default=[])
    g.add_argument('--target-level', default='细类')
    g.add_argument('--merge-map', default=None, help='合并级映射（名称→码集合）文件路径')
    g.add_argument('--target-f1-min', type=float, default=0.03)
    g.add_argument('--noninf-point', type=float, default=-0.01)
    g.add_argument('--noninf-lower', type=float, default=-0.02)
    g.add_argument('--regress-drop', type=float, default=0.05)
    g.add_argument('--min-class-n', type=int, default=30)
    g.add_argument('--seeds', nargs='*', type=int, default=[7, 11, 23])
    g.set_defaults(func=cmd_gate)

    s = sub.add_parser('seal')
    s.add_argument('--preds', nargs='*', default=[])
    s.add_argument('--selected-ids', nargs='*', default=[])
    s.add_argument('--unblind-order', default=None)
    s.add_argument('--force', action='store_true')
    s.set_defaults(func=cmd_seal)

    u = sub.add_parser('unblind')
    u.add_argument('--confirm-once', action='store_true')
    u.add_argument('--truth', default=None)
    u.add_argument('--point-level', action='store_true')
    u.add_argument('--reason', default=None)
    u.add_argument('--force', action='store_true')
    u.set_defaults(func=cmd_unblind)

    r = sub.add_parser('release')
    r.add_argument('--metrics', required=True)
    r.add_argument('--suppress-small-n', type=int, default=30)
    r.add_argument('--out', default=None)
    r.set_defaults(func=cmd_release)

    a = ap.parse_args()
    a.func(a)


if __name__ == '__main__':
    main()
