# -*- coding: utf-8 -*-
"""q8_converge.py — 汇总裁剪：把 q1–q6 产物装配成"主报告可直接引用"的速查件

* 输入（只读，本目录）：q1_三省指标精确表.csv、q1_面积结构对照.csv、q1_逐类F1对照_非fcs10系.csv、
  q3_转移汇总.csv、q3_栅格口径核对.json、q2_windows_leakage.json、q4_池来源构成.json、
  q6_宁_plus6分解.json
* 规则源：无新计算——只做既有产物的重排/筛选（不重扫影像、不新增统计口径）。
* 输出：三省_面积与配对速查.md 、 审计速查.json
* 用法：python q8_converge.py
"""
import csv
import json
import os

OUT = os.path.dirname(os.path.abspath(__file__))
KEYS = ['1', '2', '3', '4', '5', '6', '9', '10', '14', '15', '16', '18', '19', '20', '21', '22', '23']


def rd(fn):
    with open(os.path.join(OUT, fn), encoding='utf-8-sig') as f:
        return list(csv.reader(f))


def main():
    # 1) 面积速查（c5 面积表重排，无新计算）
    area = rd('q1_面积结构对照.csv')
    L = ['# 三省关键类面积与配对速查（供主报告引用；机械装配，无新计算）', '',
         '> 来源：c5 `compare_*_run1_run2_class.csv`（10 m 全通过计数）与 `trans_*_24x24.csv` 汇总件；',
         '> 单位 km²；"一次/二次"=第一/第二次分类图（省口径共同域，全省裁剪形态）。', '',
         '## 1. 关键类面积（km²，第一次→第二次）', '',
         '| 省 | 类 | 一次 | 二次 | Δ | Δ% |', '|---|---|---|---|---|---|']
    for tag in ['川', '粤', '宁']:
        for r in area[1:]:
            if r[0] == tag and r[1] in KEYS:
                L.append('| %s | %s %s | %s | %s | %s | %s |' % (tag, r[1], r[2], r[3], r[4], r[5], r[6]))
    # 2) 配对转移（q3 装配）
    tr = rd('q3_转移汇总.csv')
    L += ['', '## 2. 重点转移对（矩阵像元→km²，审计近似）', '',
          '| 省 | 转移对 | ≈km² |', '|---|---|---|']
    for r in tr[1:]:
        L.append('| %s | %s | %s |' % (r[0], r[1], r[3]))
    # 3) 开发 n 表
    met = rd('q1_三省指标精确表.csv')
    nf = rd('q1_逐类F1对照_非fcs10系.csv')
    L += ['', '## 3. 评估点 n 表（五窗留出 ∩ 省界，防泄漏三键全 0）', '',
          '| 省 | n_pool | n 计分 | 全点 | fcs10 | 非fcs10 | 覆盖 |', '|---|---|---|---|---|---|---|']
    cov = {'川': '99.2%', '粤': '100.0%', '宁': '99.3%'}
    npool = {'川': 1417, '粤': 2912, '宁': 1964}
    for tag in ['川', '粤', '宁']:
        n2 = {r[2]: r[5] for r in met[1:] if r[0] == tag}
        nsc = {'川': 1406, '粤': 2911, '宁': 1951}[tag]
        L.append('| %s | %d | %d | %s | %s | %s | %s |' % (
            tag, npool[tag], nsc, n2.get('全点(含fcs10惯例)'), n2.get('fcs10系'), n2.get('非fcs10净窗口'), cov[tag]))
    L += ['', '> 宁 n 计分 1945→1951 的 +6 为覆盖/接缝效应；固定共同有效集 n=1945 上 run1 0.8062→run2 0.6530（q6）。', '']
    with open(os.path.join(OUT, '三省_面积与配对速查.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    # 4) 速查 JSON
    bundle = dict(
        generated='2026-10-15',
        metrics=[{k: v for k, v in zip(['省', '省全名', '口径', '分层键', 'n1', 'n2', 'OA24_1', 'OA24_2',
                                        'dOA24', 'm9_1', 'm9_2', 'dm9', 'mf1_1', 'mf1_2', 'dmf1'], r)}
                 for r in met[1:]],
        leakage_asserts=json.load(open(os.path.join(OUT, 'q2_windows_leakage.json'), encoding='utf-8'))['asserts'],
        core_region=json.load(open(os.path.join(OUT, 'q3_栅格口径核对.json'), encoding='utf-8')),
        pool_doc68_reconciliation=json.load(open(os.path.join(OUT, 'q4_池来源构成.json'),
                                                 encoding='utf-8'))['doc68_reconciliation'],
        ning_plus6=json.load(open(os.path.join(OUT, 'q6_宁_plus6分解.json'), encoding='utf-8')))
    with open(os.path.join(OUT, '审计速查.json'), 'w', encoding='utf-8') as f:
        json.dump(bundle, f, ensure_ascii=False, indent=1)
    print('q8 完成：三省_面积与配对速查.md + 审计速查.json')


if __name__ == '__main__':
    main()
