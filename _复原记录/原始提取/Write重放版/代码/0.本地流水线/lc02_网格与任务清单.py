# -*- coding: utf-8 -*-
"""
lc02_网格与任务清单.py — 生成中国区地面格网清单 + tasks.jsonl 骨架（不依赖GEE）
* 输入：lc01 输出的 per_tile_points（每1°瓦片样本点数=有陆地）
* 输出：tile_grid.csv（陆地瓦片清单 + 类别覆盖 + 优先权）、tasks.jsonl（任务骨架）
运行: python 代码/0.本地流水线/lc02_网格与任务清单.py
"""
import os, json, csv, logging
from lc_conf import OUT, CLASSES

log = logging.getLogger('lc02')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', encoding='utf-8')

def main():
    stats = json.load(open(os.path.join(OUT, '统计', 'samples_cn_stats.json')))
    per_tile = stats['per_tile_points']     # key 'lon_lat' -> n
    # 陆地瓦片 = 有样本的点（>0）。无样本=海洋/无数据。
    tiles = sorted([(k, v) for k, v in per_tile.items() if v > 0],
                   key=lambda x: x[1], reverse=True)
    log.info(f'陆地瓦片数: {len(tiles)}')

    # 生成 CSV
    rows = []
    for key, n in tiles:
        lon, lat = key.split('_')
        rows.append({'lon': lon, 'lat': lat, 'points': n, 'priority': 1})
    with open(os.path.join(OUT, '网格', 'tile_grid.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['lon','lat','points','priority'])
        w.writeheader(); w.writerows(rows)

    # 任务清单骨架：每陆地瓦片 × 锚年(2020) 1个全量分类占位
    plans = []
    for key, n in tiles:
        lon, lat = key.split('_')
        plans.append({'id': f'CNLC10_2020_{key}_full', 'type': 'classify_full',
                      'year': 2020, 'lon': lon, 'lat': lat, 'status': 'PENDING'})
    with open(os.path.join(OUT, '计划', 'tasks.jsonl'), 'w', encoding='utf-8') as f:
        for p in plans:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')

    # 摘要：陆地瓦片数、样本总量、类别覆盖
    print(f'陆地瓦片数(有样本): {len(tiles)}')
    print(f'依赖瓦片点总计: {sum(n for _, n in tiles):,}')
    print(f'输出: {OUT}/网格/tile_grid.csv, {OUT}/计划/tasks.jsonl')
    # 缺口类提示
    from lc_conf import GAP_CLASSES
    print(f'\n[提示] 以下类 GLC_FCS30D 无种子，需专题产品补充:')
    for c, msg in GAP_CLASSES.items():
        print(f'  code {c}: {msg}')

if __name__ == '__main__':
    main()
