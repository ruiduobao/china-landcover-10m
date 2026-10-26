# -*- coding: utf-8 -*-
"""r1_selfcheck.py — 盲判修复与小分类·零训练自检（合成夹具 + 规则/泄漏/口径断言）

* 输入：无外部数据（合成夹具）+ prereg + 代码 11 control（deadline/ns_exclude）
* 规则源：prereg/label_repair_prereg.json（冻结规则）+ doc72
* 门槛：本脚本**不训练任何 RF**；只验证：
    1) 单一配置源 rc_conf 可 import；control 复用可用（deadline/ns_exclude）
    2) verify_frozen / verify_targets_frozen 语义（未冻结→ok=False）
    3) 修复规则机械执行：针阔不清不改 / 常绿落叶须当年物候 / 园地须管理时序 / scope=point
    4) 视图构建硬断言：C 两臂相同、isolate 不残留、回填不相交、selected R 全量且非空、
       R 占比≤上限、V 不进训练、无禁止块残留、同 count、特征一致
    5) 结构组/轴映射与预测表禁真值列（schema）
* 输出：<WS>/logs/r1_selfcheck.json
* 用法：python r1_selfcheck.py
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import lr_common as LC            # noqa: E402
import r2_build_view as RB        # noqa: E402


def synth_fixture(n_pool=400, n_V=40, rng_seed=0):
    rng = np.random.RandomState(rng_seed)
    lon = 106.0 + rng.rand(n_pool) * 2.0
    lat = 32.0 + rng.rand(n_pool) * 2.0
    cl = rng.choice([1, 2, 4, 5, 6, 22], n_pool)
    pool = pd.DataFrame(dict(point_id=['P%04d' % i for i in range(n_pool)],
                             lon=lon, lat=lat, cl=cl, src='synth'))
    for f in LC.FEATS:
        pool[f] = rng.rand(n_pool).astype('float32')
    pool['block'] = LC.block_id(pool.lon, pool.lat)

    rows = [
        # uid, before, after, action, conf, passes, evidence
        ('R000', 4, 5, 'relabel', 'high', 2, 'phenology_sameyear'),
        ('R001', 6, 7, 'relabel', 'high', 2, 'phenology_sameyear'),
        ('R002', 1, 2, 'relabel', 'medium', 2, 'orchard_mgmt_timeseries'),
        ('R003', 2, 1, 'relabel', 'high', 2, 'orchard_mgmt_timeseries'),
        ('R004', 22, 13, 'relabel', 'high', 2, 'structural'),
        ('R005', 5, 5, 'keep', 'high', 2, 'structural'),
        ('R006', 1, 1, 'keep', 'medium', 2, 'structural'),
        ('R007', 22, 22, 'isolate', 'high', 2, 'structural'),
        ('R008', 13, 13, 'isolate', 'medium', 2, 'structural'),
        ('R009', 2, 2, 'isolate', 'high', 2, 'orchard_mgmt_timeseries'),
        ('R010', 4, 4, 'keep', 'high', 2, 'structural'),
        ('R011', 6, 6, 'keep', 'high', 2, 'structural'),
        ('R012', 1, 1, 'keep', 'high', 2, 'structural'),
        ('R013', 22, 22, 'keep', 'high', 2, 'structural'),
        ('R014', 4, 4, 'keep', 'high', 2, 'structural'),
        ('R015', 5, 5, 'keep', 'high', 2, 'structural'),
        ('R016', 1, 1, 'keep', 'high', 2, 'structural'),
        ('R017', 22, 22, 'keep', 'high', 2, 'structural'),
        ('R018', 4, 6, 'relabel', 'high', 2, 'highres_structure_recheck'),  # ok 针阔+高分结构复判
        # 违例 3 条
        ('R100', 4, 6, 'relabel', 'high', 2, 'phenology_sameyear'),   # 针阔轴 → 拒绝
        ('R101', 4, 5, 'relabel', 'high', 2, 'structural'),           # 常绿落叶缺物候 → 拒绝
        ('R102', 1, 2, 'relabel', 'high', 2, 'structural'),           # 园地缺管理证据 → 拒绝
    ]
    nR = len(rows)
    R = pd.DataFrame(rows, columns=['sample_uid', 'label_before', 'label_after', 'action',
                                    'confidence', 'passes', 'evidence_type'])
    R['point_id'] = ['P%04d' % (i % n_pool) for i in range(nR)]
    # R 与 V 分处不同 0.25° 块（避免被 V 块排除误删）
    R['lon'] = 106.00 + rng.rand(nR) * 0.20
    R['lat'] = 32.00 + rng.rand(nR) * 0.20
    R['block'] = LC.block_id(R.lon, R.lat)
    R['win'] = 'w2'
    R['review_status'] = 'agree'
    R['temporal_status'] = 'applicable_2023'
    R['scope'] = 'point'
    R['rule_version'] = 'v1'

    vlon = 107.5 + rng.rand(n_V) * 0.4
    vlat = 33.5 + rng.rand(n_V) * 0.4
    V = pd.DataFrame(dict(vid=['V%03d' % i for i in range(n_V)], lon=vlon, lat=vlat,
                          block=LC.block_id(vlon, vlat), win='w2',
                          is_fcs10=rng.choice([0, 1], n_V), judge_status='ok'))
    return pool, R, V


def main():
    LC.ensure_dirs()
    cfg = LC.load_prereg()
    out = dict(generated=time.strftime('%Y-%m-%d %H:%M:%S'), checks={})

    # 1) 配置源 + control 复用
    try:
        import rc_conf as RC
        DL, NS = LC.import_control()
        out['checks']['config_and_control'] = dict(
            ok=(RC.WS == LC.WS) and getattr(DL, 'kill_process_tree', None) is not None
               and getattr(NS, 'NamespaceSafeExcluder', None) is not None,
            ws=LC.WS, single_source='rc_conf.py')
    except Exception as e:  # pragma: no cover
        out['checks']['config_and_control'] = dict(ok=False, err=str(e))

    # 2) 冻结语义
    out['checks']['verify_frozen'] = dict(ok=not LC.verify_frozen()['ok'],
                                          note='冻结前应 ok=False')
    out['checks']['verify_targets'] = dict(ok=not LC.verify_targets_frozen()['ok'],
                                           note='目标冻结前应 ok=False')

    # 3) 规则机械执行
    cases = [
        ((4, 6, 'relabel', 'phenology_sameyear', 'point'), False, '针阔变更缺高分结构复判证据→拒绝'),
        ((4, 6, 'relabel', 'highres_structure_recheck', 'point'), True, '针阔变更+高分结构复判一致→允许'),
        ((6, 7, 'relabel', 'phenology_sameyear', 'point'), True, '针叶常绿落叶+物候 允许'),
        ((4, 5, 'relabel', 'structural', 'point'), False, '常绿落叶缺物候 拒绝'),
        ((1, 2, 'relabel', 'structural', 'point'), False, '园地缺管理证据 拒绝'),
        ((1, 2, 'relabel', 'orchard_mgmt_timeseries', 'point'), True, '园地+管理证据 允许'),
        ((22, 13, 'relabel', 'structural', 'point'), True, '裸→稀疏 允许'),
        ((5, 5, 'keep', 'structural', 'point'), True, 'keep 允许'),
        ((13, 13, 'isolate', 'structural', 'point'), True, 'isolate 允许'),
        ((4, 5, 'relabel', 'phenology_sameyear', 'patch'), False, 'scope=patch 拒绝'),
    ]
    rule_ok = all(bool(RB.validate_repair(*c)[0]) == exp for c, exp, _ in cases)
    out['checks']['repair_rules'] = dict(ok=rule_ok, n_cases=len(cases))

    # 4) 视图构建（合成）
    pool, R, V = synth_fixture()
    v_blocks = set(V['block'].tolist())
    ref_blocks = {'424_128', '425_129'}          # 已用参考块（仅剔 C；R 可落在其中）
    res = RB.build_arms(pool, R, V, cfg['regions']['SC_F'], cfg, n_total=200,
                        rng_seed=7, v_blocks=v_blocks, ref_blocks=ref_blocks)
    c = res.get('checks', {})
    out['checks']['view_build'] = dict(
        ok=bool(c.get('hard_pass')) and not res.get('empty'),
        n_eligible=c.get('n_R_eligible'), n_used=c.get('n_R_used'),
        r_share=c.get('r_share'), n_baseline=c.get('n_baseline'),
        n_candidate=c.get('n_candidate'), n_isolate=c.get('n_isolate'),
        n_relabel=c.get('n_relabel'), rule_rejected=int(len(res.get('R_rej', []))),
        detail={k: c.get(k) for k in ('C_identical', 'R_kept_ids_in_both',
                                      'isolated_absent_in_candidate', 'backfill_disjoint',
                                      'all_R_selected', 'R_nonempty', 'same_count',
                                      'unique_ids', 'V_not_in_train', 'C_no_forbidden_block',
                                      'R_not_in_V_blocks', 'R_blocks_disjoint_V_blocks',
                                      'n_R_in_ref_blocks', 'r_share_ok')},
        class_prior_delta=c.get('class_prior_delta'))

    # 5) 结构组 / 轴映射
    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 22])
    st = LC.to_structure(arr, cfg)
    nb = LC.to_axis(arr, 'needle_broadleaf', cfg)
    ed = LC.to_axis(arr, 'evergreen_deciduous', cfg)
    out['checks']['structure_mapping'] = dict(
        ok=(list(st) == [3, 1, 3, 2, 2, 2, 2, 2, 0] and
            list(nb) == [0, 0, 0, 2, 2, 1, 1, 1, 0] and
            list(ed) == [0, 0, 0, 1, 2, 1, 2, 1, 0]),
        structure=list(map(int, st)), needle_broadleaf=list(map(int, nb)),
        evergreen_deciduous=list(map(int, ed)))

    # 6) 预测表禁真值列
    forb = set(cfg['evaluation']['pred_table_forbids'])
    schema = set(cfg['evaluation']['pred_table_schema'])
    out['checks']['pred_schema'] = dict(ok=(not (forb & schema)),
                                        schema=sorted(schema), forbidden=sorted(forb))

    out['all_ok'] = all(v.get('ok') for v in out['checks'].values())
    DL, _ = LC.import_control()
    fp = os.path.join(LC.LOGS, 'r1_selfcheck.json')
    DL.atomic_write_json(fp, out)
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    print('→ %s ｜ ALL_OK=%s' % (fp, out['all_ok']))
    return 0 if out['all_ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
