# -*- coding: utf-8 -*-
"""q2_windows_check.py — 证据审计·独立复验：留出窗构造、评估点泄漏、码空间与来源分层

* 输入（只读）：
    - 五窗留出：F:/lc_work/v31_exp/data/yearly/eval_2023_w{1..5}.parquet（row_id/lon/lat/class_new/src）
    - 旧池：F:/lc_work/prod5p_2023/data/train2023_clean.parquet（row_id/…）
    - 新池：F:/lc_work/prod2nd_2023/data/train2023_v2.parquet（point_id/cl/lon/lat/src）
    - 错位清单：数据/本地处理/全国清洗训练/错位样本剔除清单.parquet
    - 第二次评估 json（n 对账）：F:/lc_work/prod2nd_2023/reports/eval2/eval2_{川,粤,宁}_2023.json
* 规则源：**直接 import c4_eval_second 的 build_exclusions/pos_key/assign_prov**（同代码同口径），
  v31_common.to_v31（旧码→24 类）；PC.is_fcs10（src 以 glc_fcs10 开头）。
* 门槛/自检：
    ① 泄漏三键（旧池 rid / 新池 point_id 键 / 新池 6 位位置键）+ 错位清单 必须全 0；
    ② 逐省 n_pool 必须等于 eval2 json 的 n_pool（1417/2912/1964）；
    ③ 类 8 真值点在川/粤/宁 必须为 0（否则第二次缺 8 类会系统性误分，需在报告显式量化）；
    ④ fcs10 层灌丛（真值 9/10）占比复核。
* 输出（全落本目录）：q2_windows_leakage.json、q2_province_truth_classes.csv、q2_windows_bounds.csv
* 用法：python q2_windows_check.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

CODE_DIR = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/代码/9.第二次分类川粤宁'
sys.path.insert(0, CODE_DIR)
import c4_eval_second as c4          # 同源函数：build_exclusions / pos_key / assign_prov（import 不执行 main）
sys.path.insert(0, c4.EVAL_DIR + r'/..')  # 仅备查，无实际写入
import p5common as PC
import v31_common as VC

EVAL_DIR = c4.EVAL_DIR
OUT = os.path.dirname(os.path.abspath(__file__))
PROV2 = ['川', '粤', '宁']
FULL = {'川': '四川省', '粤': '广东省', '宁': '宁夏回族自治区'}


def main():
    # 1) 五窗汇总
    fr = []
    win_bounds = {}
    for w in ['w1', 'w2', 'w3', 'w4', 'w5']:
        d = pd.read_parquet(os.path.join(EVAL_DIR, 'eval_2023_%s.parquet' % w),
                            columns=['row_id', 'lon', 'lat', 'class_new', 'src'])
        d['win'] = w
        fr.append(d)
        win_bounds[w] = dict(n=int(len(d)), lon=[round(float(d.lon.min()), 3), round(float(d.lon.max()), 3)],
                             lat=[round(float(d.lat.min()), 3), round(float(d.lat.max()), 3)])
    ev = pd.concat(fr, ignore_index=True)
    n0 = len(ev)
    print('五窗留出合计 %d（分窗 %s）' % (n0, {w: v['n'] for w, v in win_bounds.items()}))

    # 2) 排除（直接用 c4 的实现，同代码同口径）
    keep, xd = c4.build_exclusions(ev, 'both')
    ev = ev[keep].reset_index(drop=True)
    print('排除交叉：v1_rid=%d | v2_pid=%d | v2_pos=%d | misaligned=%d | 保留=%d' % (
        xd['excl_v1_rid'], xd['excl_v2_pid'], xd['excl_v2_pos'], xd['excl_misaligned'], len(ev)))

    # 3) 归省（同 c4：五省几何）
    prov, bdist = c4.assign_prov(ev.lon.to_numpy(), ev.lat.to_numpy(), list(PC.PROV_SHORT))
    ev['prov'] = prov
    ev['near_boundary'] = bdist < 0.005

    # 4) 24 类码（同 c4：to_v31）
    ev['t24'] = VC.to_v31(ev.class_new.to_numpy(int))
    ev['isf'] = PC.is_fcs10(ev.src).to_numpy()

    # 5) 对账 + 类 8 检查 + fcs10 构成
    out = dict(n_windows_raw=n0, window_bounds=win_bounds, exclusions=xd, provinces={})
    for tag in PROV2:
        sub = ev[ev.prov == FULL[tag]]
        j = json.load(open(os.path.join(c4.RESULTS, 'eval2_%s_2023.json' % tag), encoding='utf-8'))
        n_pool_json = j['n_pool']
        cnt = sub.t24.value_counts().sort_index()
        c8 = int((sub.t24 == 8).sum())
        c7 = int((sub.t24 == 7).sum())
        sub_f = sub[sub.isf]
        n_f = len(sub_f)
        shr_910 = float(((sub_f.t24 == 9) | (sub_f.t24 == 10)).mean()) if n_f else None
        t910 = sub_f.t24.value_counts().sort_index().to_dict()
        sub_n = sub[~sub.isf]
        out['provinces'][tag] = dict(
            n_pool_mine=int(len(sub)), n_pool_json=int(n_pool_json),
            match=bool(len(sub) == n_pool_json),
            window_parts={w: int(v) for w, v in sub.win.value_counts().items()},
            near_boundary=int(sub.near_boundary.sum()),
            truth_class8=c8, truth_class7=c7,
            classes={int(k): int(v) for k, v in cnt.items()},
            fcs10_n=n_f, fcs10_shrub_share=shr_910,
            fcs10_class_counts={int(k): int(v) for k, v in t910.items()},
            nonfcs10_n=len(sub_n))
        print('%s: n_pool=%d(json %d) match=%s | 类8真值=%d 类7=%d | fcs10 n=%d 灌丛占比=%s' % (
            tag, len(sub), n_pool_json, len(sub) == n_pool_json, c8, c7, n_f,
            None if shr_910 is None else round(shr_910, 4)))
    # 断言组
    assert xd['excl_v1_rid'] == xd['excl_v2_pid'] == xd['excl_v2_pos'] == xd['excl_misaligned'] == 0, '泄漏非 0'
    for tag in PROV2:
        assert out['provinces'][tag]['match'], '%s n_pool 对账失败' % tag
        assert out['provinces'][tag]['truth_class8'] == 0, '%s 类8真值非 0' % tag
    out['asserts'] = dict(leakage_all_zero=True, n_pool_match=True, class8_zero=True)
    with open(os.path.join(OUT, 'q2_windows_leakage.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # 6) 明细 CSV
    rows = []
    for tag in PROV2:
        p = out['provinces'][tag]
        for c, n in p['classes'].items():
            rows.append([tag, c, VC.V31_NAMES().get(str(c), '?'), n,
                         p['fcs10_class_counts'].get(c, 0)])
    with open(os.path.join(OUT, 'q2_province_truth_classes.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        import csv
        w = csv.writer(f)
        w.writerow(['省', '24类码', '类名', '评估池真值点数(含未计分miss)', '其中fcs10系'],
                   )
        w.writerows(rows)
    with open(os.path.join(OUT, 'q2_windows_bounds.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        import csv
        w = csv.writer(f)
        w.writerow(['窗', '点数', 'lon_min', 'lon_max', 'lat_min', 'lat_max'])
        for wname, v in win_bounds.items():
            w.writerow([wname, v['n'], v['lon'][0], v['lon'][1], v['lat'][0], v['lat'][1]])
    print('q2 完成：全部断言通过（泄漏 0 / n 对账 / 类 8 真值 0）')


if __name__ == '__main__':
    main()
