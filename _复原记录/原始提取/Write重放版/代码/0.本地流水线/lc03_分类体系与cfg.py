# -*- coding: utf-8 -*-
"""
lc03_分类体系与cfg.py — 落地分类体系表 / 码表映射 / cfg.json（从 lc_conf 导出）
运行: python 代码/0.本地流水线/lc03_分类体系与cfg.py
"""
import os, csv, json
from lc_conf import CLASSES, CODE_MAP, TARGET_FEATURES, OUT, LEVEL1, LEVEL0

def main():
    # 1) 分类体系表 CSV
    g = os.path.join(OUT, '分类体系')
    os.makedirs(g, exist_ok=True)
    with open(os.path.join(g, 'classification_system.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['code','ename','cname','level1','level1_code','level0','level0_code'])
        for c in sorted(CLASSES):
            en, cn, l1, l0 = CLASSES[c]
            w.writerow([c, en, cn, l1, LEVEL1[l1], l0, LEVEL0[l0]])
    print('分类体系表:', os.path.join(g, 'classification_system.csv'))

    # 2) 码表映射 CSV
    with open(os.path.join(g, 'codemap.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['glc_fcs30d_code','new_code','note'])
        for old, new in sorted(CODE_MAP.items()):
            w.writerow([old, new, CLASSES.get(new, ('?','?','?','?'))[1]])
    print('码表映射:', os.path.join(g, 'codemap.csv'))

    # 3) cfg.json（特征顺序单一事实来源，供 GEE 端使用）
    cfg = {'feature_order': TARGET_FEATURES,      # 训练列序=GEE select列序，顺序不可变
           'classes': sorted(CLASSES),            # 全部类别码
           'variables_per_split': 8, 'min_leaf': 5, 'n_trees': 300,
           'gap_classes': {c: msg for c, msg in
                {92:'open mixed forest (use mixture/cover product)',
                 180:'woody swamp (use GWL/wetland product)'}.items()}}
    with open(os.path.join(g, 'cfg.json'), 'w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print('cfg.json:', os.path.join(g, 'cfg.json'))

if __name__ == '__main__':
    main()
