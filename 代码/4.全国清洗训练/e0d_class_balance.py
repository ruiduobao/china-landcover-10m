# -*- coding: utf-8 -*-
"""
e0d_class_balance.py — P1.3 三套分布设计（评审阶段E，只定义参数与缺口，不重采样）
* E1 训练分布: 大类上限 15万（r7_train 中 121/10/201/130 超限），其余维持
* E2 代表性分布: 方法=按生态区×类面积加权（CLCD/FCS10 面积先验），用于 Olofsson
  面积校正；本脚本只登记方法与占位，P3.4 评估时实现
* E3 稀有类专项池: 现状 n < 目标下限的类，列出缺口与候选专题源
* 输出: 数据/本地处理/全国清洗训练/样本诊断/class_balance_design.json
* 用法: python e0d_class_balance.py
"""
import os, sys, json
import pandas as pd

PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
sys.path.insert(0, os.path.join(PROJ, '代码', '0.本地流水线'))
from lc_conf import class_name
TRAIN = os.path.join(PROJ, '数据/本地处理/样本重建/r7_train.parquet')
OUT = os.path.join(PROJ, '数据/本地处理/全国清洗训练/样本诊断/class_balance_design.json')

# E1 训练分布目标（评审 E1 表）：大 8-15万 / 中 2-8万 / 小 0.5-2万 / 极稀有 0.2-0.5万
CAP_BIG = 150_000
BIG_CLASSES = []          # 运行时按 n>CAP_BIG 填
# E3 稀有类目标下限（极稀有类 2000-5000；本地无源类标记 no_local_source）
RARE_TARGET_MIN = 2000
RARE_SOURCES = {
    91: ['CN30米精细2020 (s3b_more_sources)', 'GLC_FCS10 91类 3×3纯净重采 (s4)'],
    92: ['GLC_FCS10 92类 3×3纯净重采 (s4)', 'CN30米精细2020'],
    140: ['青藏高原裸岩带 FCS10 定向补采', 'GLC_FCS30D 140 稳定核'],
    183: ['新疆内陆盐湖带 FCS10/GWL 定向补采', '河口三角洲盐渍湿地专题'],
    184: ['GMW v3 2020 红树林 (rare_class/gmw)', '海岸湿地年度图 1=红树'],
    185: ['中国盐沼 2022 (rare_class/saltmarsh)', '海岸湿地年度图 3=盐沼'],
    52:  ['FCS10 52类 3×3纯净重采 (s4)', 'SDPT V2 常绿疏闭'],
    62:  ['SDPT V2 落叶疏闭', 'FCS10 62类重采'],
    11:  ['AOMC苹果+柑橘+TeaMap+橡胶 (rare_class)', 'SDPT Tree crops'],
    180: ['GWL_FCS30 稳定湿地 (s3)', 'FCS10 180类重采'],
}

def main():
    train = pd.read_parquet(TRAIN, columns=['class_new'])
    n_by_c = train.class_new.value_counts().sort_index()
    e1, e3 = {}, []
    total_after = 0
    for c, n in n_by_c.items():
        cn = class_name(int(c))
        if n > CAP_BIG:
            e1[str(c)] = {'name': cn, 'n_now': int(n), 'cap': CAP_BIG,
                          'n_after': CAP_BIG, 'action': '训练分布截顶(随机保留,分层0.25°格)'}
            total_after += CAP_BIG
        else:
            e1[str(c)] = {'name': cn, 'n_now': int(n), 'cap': None,
                          'n_after': int(n), 'action': '维持'}
            total_after += int(n)
        if n < RARE_TARGET_MIN:
            e3.append({'class': int(c), 'name': cn, 'n_now': int(n),
                       'gap': RARE_TARGET_MIN - int(n),
                       'candidate_sources': RARE_SOURCES.get(int(c), ['待定'])})
    design = {
        'time': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
        'E1_训练分布': {
            'rule': '大类>15万截顶（0.25°格分层随机），其余维持；'
                    '与 e3 训练的 class_weight=balanced_subsample 配合',
            'n_now': int(n_by_c.sum()), 'n_after_cap': int(total_after),
            'capped_classes': {k: v for k, v in e1.items() if v['cap']}},
        'E2_代表性分布': {
            'method': '生态区×一级类面积加权抽选（面积先验=CLCD2020 30m 类面积），'
                      '仅用于面积估计/Olofsson 校正，不与训练集混用',
            'status': '方法已登记，P3.4 实现抽样'},
        'E3_稀有类专项池': e3,
        'note': '本脚本只登记设计参数，不修改 r7_train；重采样在 P3.3 实施'}
    json.dump(design, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f"E1 截顶类: {[k for k, v in e1.items() if v['cap']]}  "
          f"总量 {n_by_c.sum():,} → {total_after:,}")
    print('E3 稀有类缺口:')
    for r in e3:
        print(f"  {r['class']} {r['name']}: {r['n_now']:,} (缺 {r['gap']:,}) ← "
              f"{'; '.join(r['candidate_sources'])}")
    print('输出:', OUT)

if __name__ == '__main__':
    main()
