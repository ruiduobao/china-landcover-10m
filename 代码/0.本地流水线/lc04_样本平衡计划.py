# -*- coding: utf-8 -*-
"""
lc04_样本平衡计划.py — 基于类别统计生成"年度样本配额计划"（03文档 §5.1 √面积 + 稀有保底）
* 不依赖GEE/嵌入；输入为 lc01 的 class_new 分布。
* 目标：给出每年每类 建议采样点数 的配额表，指导后续GEE采样 & 训练入模。
运行: python 代码/0.本地流水线/lc04_样本平衡计划.py
"""
import os, json, csv, math
from lc_conf import CLASSES, OUT

# 中国陆域各类近似面积(km²)——用于√面积加权。GLC_FCS30D 可估，这里先用样本量作为面积代理。
N_TARGET = 900_000          # 年度目标总样本数（03文档 §6.5 ≈86万，取90万）
N_MIN = 3000                # 稀有类保底（03文档 §5.1）
AREA_WEIGHT: dict = {}      # 由样本量赋值（仅作面积代理，注释说明）

def main():
    stats = json.load(open(os.path.join(OUT, '统计', 'samples_cn_stats.json')))
    class_new = {int(k): v for k, v in stats['class_new'].items()}

    # 面积代理 = 样本量（类样本量≈∝面积）
    # √面积加权配额： n_k ∝ sqrt(n_sample_k)，稀有类保底 N_MIN
    weights = {c: math.sqrt(class_new.get(c, 1)) for c in CLASSES}
    # 稀有类（样本少）若sqrt太小，额外抬到 sqrt(N_MIN) 水平再加权，避免被淹没
    wsum = sum(weights.values())
    plan = {}
    for c in CLASSES:
        n = class_new.get(c, 0)
        raw = N_TARGET * weights[c] / wsum
        # 稀有类保底：配额至少 N_MIN，且不因面积代理太小而接近0
        n_q = max(raw, N_MIN) if n > 0 else N_MIN
        plan[c] = round(n_q)
    # 归一化回 N_TARGET（保底后可能略超，这里直接输出，统计总规模）
    total = sum(plan.values())

    g = os.path.join(OUT, '计划')
    os.makedirs(g, exist_ok=True)
    with open(os.path.join(g, 'sample_plan_annual.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['code','cname','src_samples','quota_per_year','level0'])
        for c in sorted(CLASSES):
            en, cn, l1, l0 = CLASSES[c]
            w.writerow([c, cn, class_new.get(c,0), plan[c], l0])
    print(f'年度目标总样本:{N_TARGET:,}  保底后合计:{total:,}  类别数:{len(CLASSES)}')
    print('\n配额表(每类/年) 前15 与 稀有类:')
    for c in sorted(CLASSES, key=lambda x: -plan[x])[:15]:
        print(f'  {c:>3} {CLASSES[c][1]:<12} 源样本:{class_new.get(c,0):>9,} 配额:{plan[c]:>7,}')
    print('  ...')
    for c in sorted(CLASSES, key=lambda x: plan[x])[:6]:
        print(f'  {c:>3} {CLASSES[c][1]:<12} 源样本:{class_new.get(c,0):>9,} 配额:{plan[c]:>7,}')
    print('\n[提示] 缺92与180源样本为0，配额=N_MIN需专题产品补充种子。')
    print('输出:', os.path.join(g, 'sample_plan_annual.csv'))

if __name__ == '__main__':
    main()
