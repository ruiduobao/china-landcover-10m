# -*- coding: utf-8 -*-
"""
ng1_plan.py — 下一代补样：本地分片清单生成（零 EECU，零 GEE）

把每个 spec 的区域窗口切成 NX×NY 个小窗（试点 2×2），每窗一个分片；
分片清单落 K 盘 ng_idx/plan_<spec>.json，供账号专属工作器读取。
正式批量时把 REGIONS 的 box 换成生态区/省域多边形即可，工作器接口不变。

用法: python ng1_plan.py [--nx 2] [--ny 2]
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ng0_paths as P
import ng0_spec as S


def split_box(box, nx, ny):
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) / nx, (y1 - y0) / ny
    out = []
    for iy in range(ny):
        for ix in range(nx):
            out.append([round(x0 + ix * dx, 4), round(y0 + iy * dy, 4),
                        round(x0 + (ix + 1) * dx, 4), round(y0 + (iy + 1) * dy, 4)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nx', type=int, default=2)
    ap.add_argument('--ny', type=int, default=2)
    a = ap.parse_args()
    P.ensure_all(list(S.SPECS))
    made = []
    for key, spec in S.SPECS.items():
        box = S.REGIONS[spec['region']]['box']
        boxes = split_box(box, a.nx, a.ny)
        plan = {'spec': key, 'cls': spec['cls'], 'gate': spec['gate'],
                'region': spec['region'], 'region_note': S.REGIONS[spec['region']]['note'],
                'cand_per_shard': spec['cand'], 'target': spec['target'],
                'bbox': box, 'grid': [a.nx, a.ny],
                'shards': [{'shard': i, 'box': b} for i, b in enumerate(boxes)]}
        fp = P.shard_plan_path(key)
        json.dump(plan, open(fp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        made.append(fp)
        print(f'{key:14s} {spec["tag"]:8s} {a.nx}x{a.ny}={len(boxes)} 片  '
              f'候选/片 {spec["cand"]:>6,}  目标 {spec["target"]:>4}  → {fp}')
    print('\n合计分片:', sum(len(json.load(open(f))["shards"]) for f in made))


if __name__ == '__main__':
    main()
