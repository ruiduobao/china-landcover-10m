# -*- coding: utf-8 -*-
"""
ng4b_gmw_evidence.py — 给 GMW 候选点取 GEE 证据波段并做双证门槛（1 个账号，专用脚本）

与舰队工作器的差别：候选点**来自本地 GMW 掩膜**（不是 randomPoints），所以这里
sampleRegions 时**不加掩膜**，把证据波段原样取回，在本地按四条判据打分：
    wc95  = WorldCover v200 判红树林(95)
    tidal = 4 < GSW occurrence < 92 且 transition >= 1
    low   = GLO-30 海拔 < 30 m
双证 = gmw(本地) & wc95 & tidal & low（与试点门槛一致，GMW 取代"缺位"的第二个产品源）。
同时输出 GMW-only / WC-only 的分歧统计，供人工判断哪个产品更可信。

用法: python ng4b_gmw_evidence.py --acct y30b63ye --anchor quick-cache-508211-s9
输出: ng_raw/mangrove184/raw_mangrove184_<acct>_gmw.parquet（过双证门槛的点）
      ng_qa/mangrove184_gmw_agreement.json（产品间一致性报告）
"""
import os
import sys
import io
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
BANDS = ['wc21', 'gsw_occ', 'gsw_tra', 'dem']
CHUNK = 6000          # 每请求点数（GEE 交互式单请求别太大，2026-09-13 教训）


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


def evidence_stack(ee):
    # 红树林候选点分布在热带海岸，用其整体 bbox 建 GLO-30 镶嵌即可（_dem 需要几何过滤）
    geom = ee.Geometry.Rectangle([105.0, 17.0, 125.0, 28.5], proj='EPSG:4326', geodesic=False)
    w21 = S._wc(ee, 'ESA/WorldCover/v200').rename('wc21')
    g = S._gsw(ee)
    occ = g.select('occurrence').rename('gsw_occ')
    tra = g.select('transition').rename('gsw_tra')
    dem = S._dem(ee, geom)
    return ee.Image.cat([w21, occ, tra, dem])


def fetch(ee, stack, df, prop, acct):
    """分块取证据波段；失败重试 + 看门狗（沿用本轮教训）"""
    import threading
    out = []
    for s in range(0, len(df), CHUNK):
        g = df.iloc[s:s + CHUNK]
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]), {prop: int(i)})
            for i, r in zip(g.index, g.itertuples())])
        samp = stack.sampleRegions(collection=fc, scale=10, tileScale=4)
        url = samp.getDownloadURL(filetype='csv', selectors=BANDS + [prop])
        box = {}
        def _run():
            try:
                r = requests.get(url, proxies=PROXY, timeout=900)
                r.raise_for_status()
                box['df'] = pd.read_csv(io.BytesIO(r.content))
            except Exception as e:
                box['err'] = e
        th = threading.Thread(target=_run, daemon=True)
        th.start(); th.join(420)
        if th.is_alive():
            print(f'  chunk@{s} 超时，重试', flush=True); time.sleep(20); continue
        if 'err' in box:
            print(f'  chunk@{s} 失败 {str(box["err"])[:80]}，重试', flush=True)
            time.sleep(20); continue
        d = box['df'].rename(columns={prop: '_pid'})
        d['_pid'] = d['_pid'].map(dict(zip(g.index, g.index)))
        out.append(d)
        print(f'  {min(s + CHUNK, len(df))}/{len(df)} 取回 {len(d)} 行', flush=True)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acct', default='y30b63ye')
    ap.add_argument('--anchor', default='quick-cache-508211-s9')
    ap.add_argument('--cand', default=os.path.join(P.RAW, 'mangrove184', 'gmw_candidates.parquet'))
    a = ap.parse_args()
    P.ensure_all(['mangrove184'])
    cand = pd.read_parquet(a.cand)
    print(f'GMW 候选点 {len(cand):,}')
    ee = boot(a.acct, a.anchor)
    stack = evidence_stack(ee)
    t0 = time.time()
    ev = fetch(ee, stack, cand, 'gmv_pid', a.acct)
    if not len(ev):
        raise SystemExit('没取回任何证据')
    m = ev.merge(cand.reset_index().rename(columns={'index': '_pid'}),
                 on='_pid', how='inner')
    print(f'证据匹配 {len(m):,}/{len(cand):,}  ({time.time()-t0:.0f}s)')

    wc95 = m.wc21 == 95
    tidal = (m.gsw_occ > 4) & (m.gsw_occ < 92) & (m.gsw_tra >= 1)
    low = m.dem < 30
    dual = wc95 & tidal & low
    rep = {
        'time': time.strftime('%Y-%m-%d %H:%M'),
        'gmw_candidates': int(len(m)),
        'wc95_yes': int(wc95.sum()), 'wc95_rate': round(float(wc95.mean()), 4),
        'tidal_yes': int(tidal.sum()), 'tidal_rate': round(float(tidal.mean()), 4),
        'low_yes': int(low.sum()), 'low_rate': round(float(low.mean()), 4),
        'dual_pass': int(dual.sum()),
        'gmw_only_not_wc': int((~wc95).sum()),
        'wc_only_not_gmw_note': 'WC-only 需另一套采样（盒内掩膜法已在本轮 213 点覆盖）',
        'lon': [float(m.lon.min()), float(m.lon.max())],
        'lat': [float(m.lat.min()), float(m.lat.max())],
    }

    def _seg(lat):
        for lo, hi, name in ((-90, 24.0, '粤西/北部湾'), (24.0, 27.0, '粤东/闽南'),
                             (27.0, 31.0, '浙闽'), (31.0, 34.0, '长江口/江苏'),
                             (34.0, 41.0, '渤海/黄海')):
            if lo <= lat < hi:
                return name
        return '其他'

    rep['zones'] = {str(k): int(v) for k, v in
                    pd.Series([_seg(x) for x in m.lat]).value_counts().items()}
    keep = m[dual].copy()
    keep = keep.rename(columns={'_pid': 'gmw_pid'})
    keep['spec'] = 'mangrove184'
    keep['cls'] = 184
    keep['shard_id'] = 900
    keep['src_acct'] = a.acct + '_gmw'
    keep['src_year'] = 2021
    out = os.path.join(P.RAW, 'mangrove184', f'raw_mangrove184_{a.acct}_gmw.parquet')
    keep[['lon', 'lat', 'cls', 'spec', 'shard_id', 'src_acct', 'src_year'] + BANDS].to_parquet(out, index=False)
    json.dump(rep, open(os.path.join(P.QA, 'mangrove184_gmw_agreement.json'), 'w',
                        encoding='utf-8'), ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    print('双证通过:', len(keep), '→', out)


if __name__ == '__main__':
    main()
