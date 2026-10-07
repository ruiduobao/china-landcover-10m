# -*- coding: utf-8 -*-
"""
xfeat.py — v-final 特征增强：构建"独立于标签源"的判别特征栈并采样
背景：本地可分性探针显示 52vs51 AUC=0.785、62vs61 AUC=0.833 —— 64 维 AEF 嵌入里
      "郁闭度/组成"维度缺失。本脚本补一层物理特征：
        x_tc / x_ntv / x_nv  MODIS MOD44B 植被连续场（树冠/非树植被/非植被覆盖度，250m）
        x_med / x_amp / x_win / x_rw  MODIS MOD13Q1 NDVI 中位/季节振幅/冬季绿度/冬中比
        x_gh                GEDI rh98 冠层高度（可选，--nogedi 关闭）
        x_elev / x_slope    Copernicus GLO-30 高程/坡度
      ——与 GLC_FCS30（本项目标签源）和 WorldCover 均相互独立。
用途：
  1) 验证模式  --zone 130,44   仅采该 2° 区母库点，供本地重跑可分性探针
  2) 全量模式  --chunk N       按 --index 分块采样（供多账号并行）
输出：parquet（row_id + 特征列）
用法：
  python xfeat.py --zone 130,44
  python xfeat.py --make-index            # 生成全库按点分块索引
"""
import os, io, sys, json, time, argparse, random
import numpy as np
import pandas as pd
import requests

PROXY = {'http': 'socks5h://127.0.0.1:7890', 'https': 'socks5h://127.0.0.1:7890'}
CRED = 'F:/地理所/论文/博士论文/中期/实验区域/02_中间数据/gee_accounts'
MOTHER = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/本地处理/样本重建/r7_train.parquet'
WORK = r'F:/lc_work'
XDIR = os.path.join(WORK, 'xfeat')
IDX_FP = os.path.join(WORK, 'xfeat_index.parquet')
CHUNK = 25000
FEAT_COLS = ['x_tc', 'x_ntv', 'x_nv', 'x_med', 'x_amp', 'x_win', 'x_rw',
             'x_gh', 'x_elev', 'x_slope']


def emit(m):
    print(time.strftime('[%H:%M:%S] ') + str(m), flush=True)


def bring_up(acct, proj):
    d = os.path.join(CRED, acct)
    os.environ['HOME'] = d
    os.environ['USERPROFILE'] = d
    os.environ.setdefault('HTTP_PROXY', PROXY['http'])
    os.environ.setdefault('HTTPS_PROXY', PROXY['https'])
    import ee
    for i in range(5):
        try:
            ee.Initialize(project=proj)
            break
        except Exception as x:
            if i == 4:
                raise
            emit('init retry %d: %s' % (i + 1, str(x)[:90]))
            time.sleep(20)
    emit('[ee] %s @ %s ready' % (acct, proj))
    return ee


def build_stack(ee, with_gedi=True):
    """静态基线特征栈（2020–2022 基准）。与推理端必须完全一致。"""
    vcf = ee.ImageCollection('MODIS/061/MOD44B').filterDate('2020-01-01', '2023-01-01')
    x_tc = vcf.select('Percent_Tree_Cover').mean().rename('x_tc')
    x_ntv = vcf.select('Percent_NonTree_Vegetation').mean().rename('x_ntv')
    x_nv = vcf.select('Percent_NonVegetated').mean().rename('x_nv')

    nd = ee.ImageCollection('MODIS/061/MOD13Q1').filterDate('2020-01-01', '2023-01-01').select('NDVI')
    pc = nd.reduce(ee.Reducer.percentile([10, 50, 90]))
    x_med = pc.select('NDVI_p50').rename('x_med')
    x_amp = pc.select('NDVI_p90').subtract(pc.select('NDVI_p10')).rename('x_amp')

    x_win = (ee.ImageCollection('MODIS/061/MOD13Q1')
             .filterDate('2021-01-01', '2021-03-01').select('NDVI').mean().rename('x_win'))
    x_rw = x_win.divide(x_med.where(x_med.gt(1), 1)).rename('x_rw')

    dem = ee.ImageCollection('COPERNICUS/DEM/GLO30').select('DEM').mosaic().rename('x_elev')
    slope = ee.Terrain.slope(dem).rename('x_slope')

    bands = [x_tc, x_ntv, x_nv, x_med, x_amp, x_win, x_rw, dem, slope]
    if with_gedi:
        g = (ee.ImageCollection('LARSE/GEDI/GEDI02_A_002_MONTHLY')
             .filterDate('2020-01-01', '2023-01-01').select('rh98').max().rename('x_gh'))
        bands.append(g)
    # 逐波段 unmask(-1)：GEDI 覆盖稀疏、MODIS 在水体/异常处有掩膜，
    # 不解掩膜会让 sampleRegions 丢掉整点（实测 2 万点只回 2768）
    return ee.Image.cat([b.unmask(-1) for b in bands]).toFloat()


def make_index():
    os.makedirs(XDIR, exist_ok=True)
    emit('读母库 %s' % MOTHER)
    d = pd.read_parquet(MOTHER, columns=['row_id', 'lon', 'lat', 'class_new'])
    d = d.sort_values('row_id').reset_index(drop=True)
    d['chunk_id'] = (np.arange(len(d)) // CHUNK).astype(np.int32)   # 按点分块
    d.to_parquet(IDX_FP, index=False)
    emit('索引 %d 点 → %d 块（每块 %d）→ %s' % (len(d), d.chunk_id.max() + 1, CHUNK, IDX_FP))


def pull(url, tries=3, timeout=900):
    last = None
    for j in range(tries):
        try:
            r = requests.get(url, proxies=PROXY, timeout=timeout)
            r.raise_for_status()
            return pd.read_csv(io.BytesIO(r.content))
        except Exception as x:
            last = x
            emit('  download retry %d/%d: %s' % (j + 1, tries, str(x)[:90]))
            time.sleep(15 + 10 * j)
    raise RuntimeError('download exhausted: %s' % str(last)[:120])


def sample_pts(ee, stack, df, tag, tile=2, scale=10):
    fc = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([float(r['lon']), float(r['lat'])]),
                   {'xf_rid': int(r['row_id'])})
        for r in df[['row_id', 'lon', 'lat']].to_dict('records')])
    samp = stack.sampleRegions(collection=fc, properties=['xf_rid'],
                               scale=scale, tileScale=tile, geometries=False)
    url = samp.getDownloadURL(filetype='csv', selectors=['xf_rid'] + FEAT_COLS)
    out = pull(url)
    out = out.rename(columns={'xf_rid': 'row_id'})
    out['x_tag'] = tag
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acct', default='seqsiu')
    ap.add_argument('--proj', default='')
    ap.add_argument('--zone', default='')       # "130,44"
    ap.add_argument('--chunk', type=int, default=-1)
    ap.add_argument('--make-index', action='store_true')
    ap.add_argument('--nogedi', action='store_true')
    ap.add_argument('--scale', type=int, default=10)
    a = ap.parse_args()

    if a.make_index:
        make_index()
        return 0
    os.makedirs(XDIR, exist_ok=True)
    if not os.path.isfile(IDX_FP):
        make_index()
    idx = pd.read_parquet(IDX_FP)

    if a.zone:
        zx, zy = (int(v) for v in a.zone.split(','))
        pts = idx[(idx.lon // 2 * 2 == zx) & (idx.lat // 2 * 2 == zy)].copy()
        emit('区 (%d,%d) 母库点 %d' % (zx, zy, len(pts)))
        if len(pts) > 45000:
            pts = pts.sample(45000, random_state=5).sort_values('row_id')
            emit('  → 抽样到 %d' % len(pts))
        tag = 'z%d_%d' % (zx, zy)
    else:
        pts = idx[idx.chunk_id == a.chunk].copy()
        tag = 'c%04d' % a.chunk
        emit('块 %d：%d 点' % (a.chunk, len(pts)))
    if not len(pts):
        emit('无点，退出'); return 0

    proj = a.proj
    if not proj:
        sys.path.insert(0, os.path.join(os.path.dirname(CRED), 'gee_accounts'))
        import io as _io, re
        txt = _io.open(os.path.join(CRED, a.acct, '_任务登记.md'), encoding='utf-8', errors='replace').read()
        m = re.search(r'^\|\s*([a-z][a-z0-9-]{10,40})\s*\|', txt, re.M)
        proj = m.group(1)
    ee = bring_up(a.acct, proj)
    stack = build_stack(ee, with_gedi=not a.nogedi)

    t0 = time.time()
    if a.zone:
        out = sample_pts(ee, stack, pts, tag, scale=a.scale)
    else:
        outs = []
        for s in range(0, len(pts), 8000):          # 分小批，避免单次下载过大
            sub = pts.iloc[s:s + 8000]
            outs.append(sample_pts(ee, stack, sub, tag, scale=a.scale))
        out = pd.concat(outs, ignore_index=True)
    fp = os.path.join(XDIR, 'xfeat_%s.parquet' % tag)
    out.to_parquet(fp, index=False)
    emit('%s: %d/%d 点有值，耗时 %.0fs → %s' % (tag, out.x_tc.notna().sum() if len(out) else 0,
                                              len(pts), time.time() - t0, fp))
    emit(out[FEAT_COLS].describe().T.to_string())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
