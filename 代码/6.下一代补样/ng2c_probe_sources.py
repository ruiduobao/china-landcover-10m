# -*- coding: utf-8 -*-
"""
ng2c_probe_sources.py — 家族 5–15 的候选数据源探活（只读，零额度）

对每个候选数据集：先按 Image 读波段名，失败再按 ImageCollection 数影像数；
输出可用清单 + 波段，供 ng0_spec 决定各家族门槛怎么落地。
**不要假设专题数据集存在**（184 的红树林产品就是这么踩的，见技术文档 24 §三.6）。

用法: python ng2c_probe_sources.py [--acct <账号>] [--anchor <项目>]
"""
import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ng0_paths as P

ACC_ROOT = 'F:/地理所/论文/博士论文/中期/实验区域/02_中间数据/gee_accounts'
PROXY = {'http': 'socks5h://127.0.0.1:7890', 'https': 'socks5h://127.0.0.1:7890'}

CAND = [
    # --- 灌丛 / 草地 / 稀疏 / 裸地 ---
    'GOOGLE/DYNAMICWORLD/V1',
    'ESA/WorldCover/v200',
    'MODIS/061/MOD44B',
    'projects/global-pasture-watch/assets/gsvh-30m/v1/short-veg-height_m',
    # --- 水体 / 潮滩 / 盐沼 / 盐渍 ---
    'JRC/GSW1_4/GlobalSurfaceWater',
    'JRC/GSW1_4/MonthlyHistory',
    'COPERNICUS/S1_GRD',
    # --- 森林结构与类型 ---
    'LARSE/GEDI/GEDI04_A_002_MONTHLY',
    'LARSE/GEDI/GEDI02_A_002_MONTHLY',
    'NASA/ORNL/global_forest_classification_2020/V1',
    'JAXA/ALOS/PALSAR/YEARLY/FNF4',
    'UMD/hansen/global_forest_change_2023_v1_11',
    # --- 园地 ---
    'BIOPAMA/GlobalOilPalm/v1',
    'ESA/WorldCereal/2021/MODELS/v100',
    'ESA/WorldCereal/2021/MARKERS/v100',
    # --- 冰雪 ---
    'MODIS/061/MOD10A1',
    'MODIS/061/MOD10A2',
    # --- 地形 ---
    'COPERNICUS/DEM/GLO30',
    'COPERNICUS/DEM/GLO30_2024_1',
    'NASA/NASADEM_HGT/001',
]


def boot(acct, pid):
    d = os.path.join(ACC_ROOT, acct)
    os.environ['HOME'] = d
    os.environ['USERPROFILE'] = d
    os.environ.setdefault('HTTP_PROXY', PROXY['http'])
    os.environ.setdefault('HTTPS_PROXY', PROXY['https'])
    import ee
    last = None
    for k in range(5):
        try:
            ee.Initialize(project=pid)
            return ee
        except Exception as e:
            last = e
            time.sleep(15 + 10 * k)
    raise last


def probe(ee, ds):
    rec = {'id': ds, 'kind': '', 'bands': [], 'n_images': None, 'ok': False, 'err': ''}
    try:
        b = ee.Image(ds).bandNames().getInfo()
        rec.update(kind='Image', bands=b, ok=True)
        return rec
    except Exception as e1:
        rec['err'] = str(e1)[:70]
    try:
        ic = ee.ImageCollection(ds)
        n = ic.size().getInfo()
        b = ic.first().bandNames().getInfo()
        rec.update(kind='ImageCollection', bands=b, n_images=n, ok=True, err='')
        return rec
    except Exception as e2:
        rec['err'] = (rec['err'] + ' | ' + str(e2)[:70])[:150]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acct', default='zixen8v8')
    ap.add_argument('--anchor', default='braided-horizon-508210-a5')
    a = ap.parse_args()
    P.ensure_all()
    ee = boot(a.acct, a.anchor)
    out = []
    print(f'探活 {len(CAND)} 个候选数据集（账号 {a.acct}）\n')
    for ds in CAND:
        r = probe(ee, ds)
        out.append(r)
        if r['ok']:
            extra = f"images={r['n_images']}" if r['n_images'] is not None else ''
            print(f"✅ {ds:62s} {r['kind']:16s} {extra:14s} {str(r['bands'])[:70]}")
        else:
            print(f"❌ {ds:62s} {r['err'][:80]}")
        time.sleep(0.4)
    fp = os.path.join(P.QA, 'probe_sources.json')
    json.dump(out, open(fp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    ok = [r['id'] for r in out if r['ok']]
    print(f'\n可用 {len(ok)}/{len(out)}  → {fp}')
    print('不可用:', [r['id'] for r in out if not r['ok']])


if __name__ == '__main__':
    main()
