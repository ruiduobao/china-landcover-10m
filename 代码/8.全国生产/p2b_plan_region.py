# -*- coding: utf-8 -*-
"""
p2b_plan_region.py — 区域生产分区表（按省划分区，一省一分区）
用途：东北三省 2023 年分类（黑/吉/辽 三分区，3 个账号各认领一省）。
做法：2° 格网铺满区域外包 → 每格 0.25° 子采样判定「落在目标省内的比例」→
      按多数省归入该省分区 → 每个省一个分区文件，供一个账号独立认领。
输出：代码/8.全国生产/plan/tiles_ne2023_<省>.json ＋ tiles_ne2023_汇总.csv
用法：python p2b_plan_region.py --provs 黑龙江省 吉林省 辽宁省 --deg 2.0
"""
import os, sys, io, json, time, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '5.样本重建'))
import prod_conf as C
from s1_geom import province_of

sys.stdout.reconfigure(encoding='utf-8')
BBOX_NE = (118.5, 38.5, 135.0, 54.0)      # 东北三省外包
SUB = 8                                    # 每格 8×8 子采样


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--provs', nargs='+', default=['黑龙江省', '吉林省', '辽宁省'])
    ap.add_argument('--deg', type=float, default=C.TILE_DEG)
    ap.add_argument('--min-frac', type=float, default=0.05)
    a = ap.parse_args()
    os.makedirs(C.PLAN, exist_ok=True)
    x0, y0, x1, y1 = BBOX_NE
    d = a.deg
    NX = int(round((x1 - x0) / d)); NY = int(round((y1 - y0) / d))
    print(f'区域外包 {BBOX_NE}，{d}° 格网 → {NX}×{NY}={NX*NY} 格', flush=True)
    gx = np.linspace(x0 + d / (2 * SUB), x0 + NX * d - d / (2 * SUB), NX * SUB)
    gy = np.linspace(y0 + d / (2 * SUB), y0 + NY * d - d / (2 * SUB), NY * SUB)
    GX, GY = np.meshgrid(gx, gy)
    prov = province_of(GX.ravel(), GY.ravel())
    prov = prov.reshape(GX.shape)
    keep = np.isin(prov, a.provs)
    print(f'区域内目标省像元 {int(keep.sum())} / {keep.size}', flush=True)

    rows = []
    for j in range(NY):
        for i in range(NX):
            r0, r1 = j * SUB, (j + 1) * SUB
            c0, c1 = i * SUB, (i + 1) * SUB
            blk = prov[r0:r1, c0:c1]
            n_tot = blk.size
            in_t = int(np.isin(blk, a.provs).sum())
            if in_t / n_tot < a.min_frac:
                continue
            names, cnts = np.unique(blk, return_counts=True)
            maj = names[np.argmax(cnts)]
            if maj not in a.provs:
                maj = a.provs[int(np.argmax([cnts[names == p].sum() if (names == p).any() else 0
                                             for p in a.provs]))]
            rows.append(dict(tile='N%02d%02d' % (i, j), ix=i, iy=j, part=str(maj),
                             x0=round(x0 + i * d, 3), y0=round(y0 + j * d, 3),
                             x1=round(x0 + (i + 1) * d, 3), y1=round(y0 + (j + 1) * d, 3),
                             land_frac=round(in_t / n_tot, 3)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.PLAN, 'tiles_ne2023_汇总.csv'), index=False, encoding='utf-8-sig')
    print('\n分区结果：')
    for p in a.provs:
        g = df[df.part == p]
        if not len(g):
            print(f'  {p}: 0 格'); continue
        fp = os.path.join(C.PLAN, 'tiles_ne2023_%s.json' % p)
        json.dump({'tile_deg': d, 'region': '东北三省-2023', 'part': p, 'years': [2023],
                   'n_tiles': len(g), 'tiles': g.to_dict('records')},
                  open(fp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print(f'  {p}: {len(g)} 格（陆域占比均值 {g.land_frac.mean():.2f}）→ {os.path.basename(fp)}')
    print('\n输出:', os.path.join(C.PLAN, 'tiles_ne2023_汇总.csv'))


if __name__ == '__main__':
    main()
