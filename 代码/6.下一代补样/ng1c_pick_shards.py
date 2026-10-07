# -*- coding: utf-8 -*-
"""
ng1c_pick_shards.py — 从批量分片清单里按"每分区取前 N 片"挑子集（本地，零 EECU）

为什么不是全局取前 N：全局按门槛像元数排序会让点全挤在少数分区（红树林会只剩
粤西+海南两个海岸段，盐沼会漏掉一半海岸段）。**带状类/生态区类的价值在覆盖均衡**，
所以按 zone 分组、每区取门槛像元数最高的 N 片，保证每个海岸段/生态区都有代表。

用法: python ng1c_pick_shards.py --spec saltmarsh185 --per-zone 1 [--dry]
      python ng1c_pick_shards.py --all --per-zone 1
"""
import os
import sys
import json
import argparse
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ng0_paths as P
import ng0_spec as S


def pick(spec, per_zone, dry=False):
    src = os.path.join(P.IDX, f'plan_{spec}_batch.json')
    if not os.path.isfile(src):
        print(f'[{spec}] 无批量清单，跳过'); return None
    plan = json.load(open(src, encoding='utf-8'))
    by_zone = {}
    for sh in plan['shards']:
        by_zone.setdefault(sh['zone'], []).append(sh)
    keep = []
    for z, shs in by_zone.items():
        shs.sort(key=lambda x: -x['presence'])
        keep += shs[:per_zone]
    keep.sort(key=lambda x: (-x['presence']))
    for i, sh in enumerate(keep):
        sh['shard'] = i
    plan['shards'] = keep
    plan['picked_per_zone'] = per_zone
    zones_kept = sorted({sh['zone'] for sh in keep})
    print(f'[{spec}] {len(by_zone)} 个分区 → 每区取 {per_zone} 片 = {len(keep)} 片；'
          f'覆盖 {len(zones_kept)} 个分区')
    if dry:
        return plan
    # 备份原批量清单，再覆盖活动清单
    bak = os.path.join(P.IDX, f'plan_{spec}_batch_full.json')
    if not os.path.isfile(bak):
        shutil.copy2(src, bak)
    json.dump(plan, open(src, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    act = P.shard_plan_path(spec)
    json.dump(plan, open(act, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--per-zone', type=int, default=1)
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    specs = list(S.SPECS) if a.all else ([a.spec] if a.spec else [])
    if not specs:
        raise SystemExit('给 --spec 或 --all')
    for sp in specs:
        try:
            pick(sp, a.per_zone, a.dry)
        except Exception as e:
            print(f'[{sp}] ERR {str(e)[:150]}')


if __name__ == '__main__':
    main()
