# -*- coding: utf-8 -*-
"""
p2_plan_tiles.py — 生成全国生产分区块表
做法：按 TILE_DEG(默认 2°) 铺满中国外包，用「0.25° 子网格 + 内陆判定」算陆域占比，
      并统计母库点数（决定优先级与可训练性）。只读，不碰 GEE。
输出：代码/8.全国生产/plan/tiles.json + tiles.csv
用法：python p2_plan_tiles.py [--deg 2.0] [--min-land 0.02]
"""
import os, sys, json, time, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prod_conf as C
sys.path.insert(0, os.path.join(C.PROJ, '代码', '5.样本重建'))
from s1_geom import china_contains

sys.stdout.reconfigure(encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--deg', type=float, default=C.TILE_DEG)
    ap.add_argument('--min-land', type=float, default=0.02)
    a = ap.parse_args()
    os.makedirs(C.PLAN, exist_ok=True)
    x0, y0, x1, y1 = C.BBOX
    d = a.deg
    NX = int(round((x1 - x0) / d)); NY = int(round((y1 - y0) / d))
    print(f'格网 {d}° → {NX}×{NY} = {NX*NY} 格', flush=True)

    # 母库点落格（决定优先级/可训练性）
    m = pd.read_parquet(C.MOTHER, columns=['lon', 'lat'])
    ix = ((m.lon.to_numpy() - x0) // d).astype(int).clip(0, NX - 1)
    iy = ((m.lat.to_numpy() - y0) // d).astype(int).clip(0, NY - 1)
    cnt = pd.Series(iy * NX + ix).value_counts()

    rows = []
    sub = 8   # 每格 8×8 子采样判定陆域
    for j in range(NY):
        for i in range(NX):
            tx0, ty0 = x0 + i * d, y0 + j * d
            gx = np.linspace(tx0 + d / (2 * sub), tx0 + d - d / (2 * sub), sub)
            gy = np.linspace(ty0 + d / (2 * sub), ty0 + d - d / (2 * sub), sub)
            GX, GY = np.meshgrid(gx, gy)
            n_in = int(china_contains(GX.ravel(), GY.ravel()).sum())
            land = n_in / (sub * sub)
            npts = int(cnt.get(j * NX + i, 0))
            if land >= a.min_land or npts > 0:
                rows.append(dict(tile='T%02d%02d' % (i, j), ix=i, iy=j,
                                 x0=round(tx0, 3), y0=round(ty0, 3),
                                 x1=round(tx0 + d, 3), y1=round(ty0 + d, 3),
                                 land_frac=round(land, 3), n_points=npts))
    df = pd.DataFrame(rows).sort_values(['land_frac', 'n_points'], ascending=False)
    df.to_csv(os.path.join(C.PLAN, 'tiles.csv'), index=False, encoding='utf-8-sig')
    plan = {'tile_deg': d, 'bbox': C.BBOX, 'n_tiles': len(df),
            'n_tiles_land': int((df.land_frac > 0.5).sum()),
            'total_points': int(df.n_points.sum()),
            'years': C.YEARS,
            'n_exports_if_all_years': int(len(df) * len(C.YEARS)),
            'tiles': df.to_dict('records')}
    json.dump(plan, open(os.path.join(C.PLAN, 'tiles.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(f'落格 {len(df)} 个（其中陆域占比>50% {plan["n_tiles_land"]} 个）', flush=True)
    print(f'8 年全跑 → {plan["n_exports_if_all_years"]} 个导出任务', flush=True)
    print(df.head(10)[['tile', 'land_frac', 'n_points']].to_string(index=False))
    print('输出:', os.path.join(C.PLAN, 'tiles.json'))


if __name__ == '__main__':
    main()
