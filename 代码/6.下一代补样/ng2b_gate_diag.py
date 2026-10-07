# -*- coding: utf-8 -*-
"""
ng2b_gate_diag.py — 门槛诊断（只读）：逐条件通过率，找出"门槛过严/区域不适生"的真因

做法：对 spec 的区域随机撒 N 个点，**不加掩膜**地把门槛用到的所有波段取回来，
逐条件计算通过率与联合通过率；输出每条件命中数 + 建议。
这是"试点先验证再批量"里最省额度的一步——不发掩膜版采样，不落 parquet。

用法: python ng2b_gate_diag.py --acct <账号> --anchor <项目> --spec saltmarsh185 [--n 5000]
      python ng2b_gate_diag.py --acct <账号> --anchor <项目> --all --n 5000
"""
import os
import io
import sys
import json
import time
import argparse

import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ng0_paths as P
import ng0_spec as S

ACC_ROOT = 'F:/地理所/论文/博士论文/中期/实验区域/02_中间数据/gee_accounts'
PROXY = {'http': 'socks5h://127.0.0.1:7890', 'https': 'socks5h://127.0.0.1:7890'}


def boot(acct, pid):
    d = os.path.join(ACC_ROOT, acct)
    os.environ['HOME'] = d
    os.environ['USERPROFILE'] = d
    os.environ.setdefault('HTTP_PROXY', PROXY['http'])
    os.environ.setdefault('HTTPS_PROXY', PROXY['https'])
    import ee
    last = None
    for _k in range(4):
        try:
            ee.Initialize(project=pid)
            return ee
        except Exception as _e:      # 代理 SSL EOF 是常态（晚高峰更频繁），必须重试
            last = _e
            time.sleep(8 + 6 * _k)
    raise last


# 每个 spec 的条件列表： (标签, 判定函数, 用到的波段)
def conds_for(spec, band_names):
    def c(name, fn):
        return (name, fn)
    C = []
    if spec == 'moss140':
        C += [c('wc21==100', lambda d: d.wc21 == 100),
              c('dem>3400', lambda d: d.dem > 3400),
              c('slope<22', lambda d: d.slope < 22),
              c('ndvi_p90<0.36', lambda d: d.ndvi_p90 < 0.36)]
    elif spec == 'mangrove184':
        C += [c('wc21==95', lambda d: d.wc21 == 95),
              c('4<occ<92', lambda d: (d.gsw_occ > 4) & (d.gsw_occ < 92)),
              c('tra>=1', lambda d: d.gsw_tra >= 1),
              c('dem<30', lambda d: d.dem < 30)]
    elif spec == 'saltmarsh185':
        C += [c('tra>=2', lambda d: d.gsw_tra >= 2),
              c('5<occ<90', lambda d: (d.gsw_occ > 5) & (d.gsw_occ < 90)),
              c('wc!=95', lambda d: d.wc21 != 95),
              c('wc!=80', lambda d: d.wc21 != 80),
              c('0.15<ndvi<0.70', lambda d: (d.ndvi_p75 > 0.15) & (d.ndvi_p75 < 0.70)),
              c('dem<25', lambda d: d.dem < 25)]
    elif spec == 'paddy12':
        C += [c('wc21==40', lambda d: d.wc21 == 40),
              c('VH涨>3dB', lambda d: (d.vh_p50 - d.vh_p10) > 3.0),
              c('VV_p50>-18', lambda d: d.vv_p50 > -18)]
    return C


def diag(ee, spec_key, n=5000):
    cfg = S.SPECS[spec_key]
    box = S.REGIONS[cfg['region']]['box']
    geom = ee.Geometry.Rectangle(box, proj='EPSG:4326', geodesic=False)
    stack, gate, bands, desc = S.build_gate(ee, cfg['gate'], geom)
    pts = ee.FeatureCollection.randomPoints(geom, n, seed=424242, maxError=10)
    pts = pts.map(lambda f: f.set({'dx': f.geometry().coordinates().get(0),
                                   'dy': f.geometry().coordinates().get(1)}))
    t0 = time.time()
    samp = stack.sampleRegions(collection=pts, scale=cfg.get('scale', 20),
                               tileScale=4)
    url = samp.getDownloadURL(filetype='csv', selectors=bands + ['dx', 'dy'])
    d = None
    for _t in range(4):                 # 代理瞬断重试（晚高峰常态）
        try:
            r = requests.get(url, proxies=PROXY, timeout=900)
            r.raise_for_status()
            d = pd.read_csv(io.BytesIO(r.content))
            break
        except Exception as _e:
            print(f'   下载重试 {_t+1}/4: {str(_e)[:70]}', flush=True)
            time.sleep(10 + 10 * _t)
    if d is None:
        raise RuntimeError('download exhausted')
    d = d.dropna()
    dt = time.time() - t0
    print(f'\n===== {spec_key} ({cfg["tag"]}) 区域 {cfg["region"]} '
          f'候选 {n} → 有效 {len(d)}  ({dt:.0f}s) =====')
    print(f'  门槛定义: {desc}')
    if not len(d):
        print('  ❌ 该区域取不到任何有效像元（数据缺失或窗口错误）')
        return None
    res = {'spec': spec_key, 'n_cand': n, 'n_valid': int(len(d)), 'sec': round(dt, 1),
           'conds': {}, 'joint': 0}
    joint = np.ones(len(d), bool)
    for name, fn in conds_for(spec_key, bands):
        try:
            m = np.asarray(fn(d), bool)
        except Exception as e:
            print(f'  {name:18s} 计算失败 {str(e)[:60]}')
            continue
        joint &= m
        res['conds'][name] = int(m.sum())
        print(f'  {name:18s} 通过 {int(m.sum()):>7,}  ({m.mean()*100:5.1f}%)')
    res['joint'] = int(joint.sum())
    print(f'  {"联合通过":18s} {int(joint.sum()):>7,}  ({joint.mean()*100:5.1f}%)')
    if joint.sum() == 0:
        # 找出"最杀手"的条件
        loosest = min(res['conds'], key=lambda k: res['conds'][k]) if res['conds'] else None
        print(f'  ⚠️ 联合为 0 —— 最严条件是「{loosest}」，需放宽或改区域/改判据')
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acct', required=True)
    ap.add_argument('--anchor', required=True)
    ap.add_argument('--spec')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--n', type=int, default=5000)
    a = ap.parse_args()
    P.ensure_all()
    ee = boot(a.acct, a.anchor)
    specs = list(S.SPECS) if a.all else [a.spec]
    out = []
    for sp in specs:
        try:
            r = diag(ee, sp, a.n)
            if r:
                out.append(r)
        except Exception as e:
            print(f'[{sp}] ❌ {str(e)[:250]}')
    fp = os.path.join(P.QA, 'gate_diag.json')
    json.dump(out, open(fp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n输出:', fp)


if __name__ == '__main__':
    main()
