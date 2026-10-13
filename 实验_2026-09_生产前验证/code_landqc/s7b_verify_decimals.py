# -*- coding: utf-8 -*-
"""s7b_verify_decimals.py — 全交付树 DBF 小数位扫描（F 字段 decimal=0 视为异常）

* 用法：python s7b_verify_decimals.py [--classes 草地,裸地] [--scope land|forest|orchard|shrub|swamp|all]
* 输出：逐文件 PASS/FAIL 与汇总；FAIL 明细含字段名
"""
import argparse, glob, os, sys
import shapefile

sys.stdout.reconfigure(encoding='utf-8')
DEST = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12'
LANDS = ['草本旱地', '灌溉耕地', '草地', '地衣苔藓', '稀疏植被', '裸地', '湖河滩地', '盐渍湿地',
         '红树林', '海岸盐沼', '潮滩', '水体', '不透水面', '永久冰雪']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scope', default='all')
    ap.add_argument('--classes', default='')
    a = ap.parse_args()
    only = [x.strip() for x in a.classes.split(',') if x.strip()]
    pats = {
        'land': [os.path.join(DEST, c, '**', '*_WGS84.shp') for c in (only or LANDS)],
        'forest': [os.path.join(DEST, '林地_*', '**', '*_WGS84.shp')],
        'orchard': [os.path.join(DEST, '乔灌园地', '**', '*_WGS84.shp')],
        'shrub': [os.path.join(DEST, '灌木', '**', '*_WGS84.shp')],
        'swamp': [os.path.join(DEST, '*沼泽', '**', '*_WGS84.shp')],
    }
    fs = []
    for k in (pats if a.scope == 'all' else [a.scope]):
        for p in pats[k]:
            fs += glob.glob(p, recursive=True)
    fs = sorted(set(fs))
    bad, n = [], 0
    for f in fs:
        r = shapefile.Reader(f, encoding='utf-8')
        b = [(x[0], x[3]) for x in r.fields[1:] if x[1] == 'F' and x[3] == 0]
        r.close()
        n += 1
        if b:
            bad.append((os.path.relpath(f, DEST), b))
    print('扫描 %d 文件；F 字段被截为整数的：%d' % (n, len(bad)))
    for x in bad[:20]:
        print('  ✗', x[0], x[1])
    print('结论：%s' % ('全部通过（无 decimal=0 的 F 字段）' if not bad else '存在 %d 个待修文件' % len(bad)))


if __name__ == '__main__':
    main()
