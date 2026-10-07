# -*- coding: utf-8 -*-
"""m5a_materials.py — D0 判读材料 GEE worker：M3 点位 chip+物理指标 / 灌丛层体检指标（doc45 §一）
* 输入：--points csv(point_id,lon,lat[,code,dry]) --task ind|chip --acct <账号> --shard i --nshards N --out <dir>
* 规则源：S2 SR Harmonized 2023 年 SCL 掩膜（去 3/8/9）；ETH 冠层高度（资产探针顺序回退）；Hansen treecover2000/lossyear
* 门槛：ind 分块 300（超时自动减半重试×2）；chip 256×256 @10m EPSG:4326（computePixels 优先、getDownloadURL 回退）
* 输出：<out>/ind_<shard>.csv（point_id,eth_h,tc2000,lossyear,nd_01..nd_12,ndvi_amp,ndvi_mean,ndvi_winter,ndvi_summer）
        <out>/chips/<point_id>.png（左真彩右假彩）+ <out>/chip_<shard>.csv（point_id,png,status）
* 幂等：输出文件已有 point_id / 已存在 PNG 跳过；用法：python m5a_materials.py --task ind --acct xx --shard 0 --nshards 12
"""
import os, sys, io, csv, time, argparse, json
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v31_common as VC
import numpy as np

RES = 8.983152841195215e-05
HAN_CANDS = ['UMD/hansen/global_forest_change_2025_v1_13', 'UMD/hansen/global_forest_change_2024_v1_12']
CH_CANDS = ['LARSE/GEDI/GEDI04_B_002',        # GEDI L4B 1km 平均冠层高度（官方目录）
             'NASA/JPL/global_forest_canopy_height_2019']


def resolve_pid(acct, override=None):
    if override:
        return override
    try:
        pid = VC.pid_of(acct)
        if pid:
            return pid
    except Exception:
        pass
    m = VC.jload(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              '..', 'config', 'm5_pids.json'), {})
    return m.get(acct, '')


def init(acct, pid_override=None):
    os.environ['HTTPS_PROXY'] = os.environ['HTTP_PROXY'] = os.environ['GEE_PROXY'] = VC.C.PROXY_URL
    VC.C.apply_account_env(acct)
    import ee
    pid = resolve_pid(acct, pid_override)
    if not pid:
        raise SystemExit('账号 %s 无可用项目 ID' % acct)
    print('pid=%s init...' % pid, flush=True)
    for i in range(4):
        try:
            ee.Initialize(project=pid)
            print('init ok', flush=True)
            return ee, pid
        except Exception as e:
            if i == 3:
                raise
            print('init retry %d: %s' % (i, str(e)[:80]), flush=True)
            time.sleep(10 + 5 * i)


def resolve_assets(ee, pid):
    """ETH IC + Hansen Image；探针确定可用资产（零下载，仅元数据）。"""
    ch = None
    for cand in CH_CANDS:
        try:
            obj = ee.Image(cand)
            obj.bandNames().getInfo()
            ch = obj
            globals()['_CH_BAND'] = obj.bandNames().getInfo()[0]
            print('冠层高度资产: %s 波段=%s' % (cand, _CH_BAND), flush=True)
            break
        except Exception:
            continue
    if ch is None:
        print('⚠ 冠层高度资产不可用，ch_mean 列将为空', flush=True)
    han = None
    for cand in HAN_CANDS:
        try:
            han = ee.Image(cand)
            han.bandNames().getInfo()
            print('Hansen 资产: %s' % cand, flush=True)
            break
        except Exception:
            han = None
    if han is None:
        raise SystemExit('Hansen 全部候选不可用')
    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterDate('2022-12-01', '2024-01-01')
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))
          .map(lambda im: im.updateMask(im.select('SCL').neq(3).And(im.select('SCL').neq(8))
                                        .And(im.select('SCL').neq(9)))))
    return ch, han, s2


def chip_grid(lon, lat, px=192):
    """computePixels 网格（proto 字段名：affine_transform.scale_x/shear_x/translate_x/...）"""
    half = px // 2
    return {'dimensions': {'width': px, 'height': px},
            'affine_transform': {'scale_x': RES, 'shear_x': 0.0, 'translate_x': lon - half * RES,
                                 'shear_y': 0.0, 'scale_y': -RES, 'translate_y': lat + half * RES},
            'crs_code': 'EPSG:4326'}


def read_points(fp, shard, nshards):
    rows = []
    with open(fp, encoding='utf-8-sig') as f:
        for i, r in enumerate(csv.DictReader(f)):
            if i % nshards == shard:
                rows.append((r['point_id'], float(r['lon']), float(r['lat']),
                             r.get('code', ''), r.get('dry', '')))
    return rows


def done_ids(fp):
    if not os.path.exists(fp):
        return set()
    with open(fp, encoding='utf-8-sig') as f:
        return {r['point_id'] for r in csv.DictReader(f) if r.get('point_id')}


def append(fp, header, rows):
    new = not os.path.exists(fp)
    with open(fp, 'a', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerows(rows)


SEASONS = {'nd_djf': ('2022-12-01', '2023-03-01'), 'nd_mam': ('2023-03-01', '2023-06-01'),
           'nd_jja': ('2023-06-01', '2023-09-01'), 'nd_son': ('2023-09-01', '2023-12-01')}


def ndvi_seasons(ee, s2):
    """四季 NDVI 中位数合成（DJF/MAM/JJA/SON，各一带；doc45 预注册口径）。"""
    bands = []
    for name, (d0, d1) in SEASONS.items():
        img = (s2.filterDate(d0, d1).median().normalizedDifference(['B8', 'B4']).rename(name))
        bands.append(img)
    return ee.Image.cat(bands)


def run_ind(ee, ch, han, s2, pts, out, shard):
    """指标采集：三次 reduceRegions 单遍（NDVI 12 月@10m / Hansen@30m / 冠层高度@1000m），客户端按 pid 合并。"""
    os.makedirs(out, exist_ok=True)
    fp = os.path.join(out, 'ind_%d.csv' % shard)
    todo = [(p, lo, la, c, d) for p, lo, la, c, d in pts if p not in done_ids(fp)]
    hdr = (['point_id', 'ch_mean', 'tc2000', 'lossyear', 'nd_djf', 'nd_mam', 'nd_jja', 'nd_son']
           + ['ndvi_amp', 'ndvi_mean', 'ndvi_winter', 'ndvi_summer'])
    print('ind 待跑 %d/%d' % (len(todo), len(pts)), flush=True)
    seasons = ndvi_seasons(ee, s2)
    han2 = han.select(['treecover2000', 'lossyear'])
    chb = ch.select(_CH_BAND).rename('ch_mean') if ch else None
    size = 400
    i = 0
    while i < len(todo):
        chunk = todo[i:i + size]
        t0 = time.time()
        try:
            fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([lo, la]), {'pid': p})
                                       for p, lo, la, c, d in chunk])
            buf15 = fc.map(lambda f: f.setGeometry(f.geometry().buffer(30)))
            t1 = time.time()
            ha = han2.reduceRegions(collection=fc, reducer=ee.Reducer.first(),
                                    scale=30, tileScale=4).getInfo()
            print('  step hansen %.1fs' % (time.time() - t1), flush=True)
            t1 = time.time()
            nd = seasons.reduceRegions(collection=buf15, reducer=ee.Reducer.median(),
                                       scale=30, tileScale=4).getInfo()
            print('  step ndvi4 %.1fs' % (time.time() - t1), flush=True)
            t1 = time.time()
            chd = ({f['properties']['pid']: f['properties'].get('ch_mean') for f in
                    chb.reduceRegions(collection=fc.map(lambda f: f.setGeometry(f.geometry().buffer(1000))),
                                      reducer=ee.Reducer.median(), scale=1000, tileScale=4)
                    .getInfo().get('features', [])} if chb is not None else {})
            print('  step ch %.1fs' % (time.time() - t1), flush=True)
            rows = []
            ndm = {f['properties']['pid']: f['properties'] for f in nd.get('features', [])}
            ham = {f['properties']['pid']: f['properties'] for f in ha.get('features', [])}
            for p, lo, la, c, d in chunk:
                pr = ndm.get(p) or {}
                ph = ham.get(p) or {}
                nds = [pr.get('nd_djf'), pr.get('nd_mam'), pr.get('nd_jja'), pr.get('nd_son')]
                vals = [v if v is not None else '' for v in nds]
                fl = [float(v) for v in nds if v is not None]
                amp = round(max(fl) - min(fl), 4) if len(fl) >= 3 else ''
                mean = round(sum(fl) / len(fl), 4) if len(fl) >= 3 else ''
                win = round(float(nds[0]), 4) if nds[0] is not None else ''
                summ = round(float(nds[2]), 4) if nds[2] is not None else ''
                cv = chd.get(p)
                rows.append([p, '' if cv is None else round(cv, 2),
                             '' if ph.get('treecover2000') is None else int(ph['treecover2000']),
                             '' if ph.get('lossyear') is None else int(ph['lossyear'])]
                            + vals + [amp, mean, win, summ])
            append(fp, hdr, rows)
            print('chunk %d-%d ok n=%d %.1fs' % (i, i + size, len(rows), time.time() - t0), flush=True)
            i += size
        except Exception as e:
            print('chunk 失败(%.1fs): %s' % (time.time() - t0, str(e)[:140]), flush=True)
            if size > 60:
                size //= 3
                print('分块缩小 → %d' % size, flush=True)
            else:
                rows = [[c[0], 'ERR', '', ''] + [''] * 12 for c in chunk]
                append(fp, hdr, rows)
                i += size
    print('ind shard %d 完成' % shard, flush=True)


def stretch(b):
    lo, hi = np.percentile(b[b > 0], (2, 98)) if (b > 0).any() else (1, 2)
    out = np.clip((b.astype('float32') - lo) / max(1e-6, hi - lo) * 255, 0, 255).astype('uint8')
    return out


def run_chip(ee, s2, pts, out, shard, acct):
    os.makedirs(out, exist_ok=True)
    cdir = os.path.join(out, 'chips')
    os.makedirs(cdir, exist_ok=True)
    fp = os.path.join(out, 'chip_%d.csv' % shard)
    from PIL import Image
    gs0 = s2.filter(ee.Filter.calendarRange(6, 8, 'month'))
    done = set()
    if os.path.exists(fp):
        for r in csv.DictReader(open(fp, encoding='utf-8-sig')):
            if r.get('status') == 'ok' and r.get('png') and os.path.exists(r['png']):
                done.add(r['point_id'])
    todo = [(p, lo, la, c, d) for p, lo, la, c, d in pts if p not in done]
    print('chip 待跑 %d/%d' % (len(todo), len(pts)), flush=True)
    for k, (p, lo, la, c, d) in enumerate(todo):
        png = os.path.join(cdir, '%s.png' % p)
        t0 = time.time()
        try:
            if not os.path.exists(png) or os.path.getsize(png) < 1000:
                half = 96
                gs = (gs0.filterBounds(ee.Geometry.Rectangle(
                    [lo - half * RES, la - half * RES, lo + half * RES, la + half * RES]))
                    .median().select(['B4', 'B3', 'B2', 'B8']))
                grid = chip_grid(lo, la)
                try:
                    arr = ee.data.computePixels({'expression': gs, 'fileFormat': 'NPY', 'grid': grid})
                    if isinstance(arr, (bytes, bytearray)):
                        arr = np.load(io.BytesIO(bytes(arr)))
                    arr = np.stack([arr['B4'], arr['B3'], arr['B2'], arr['B8']], -1)
                except Exception as e1:
                    url = gs.getDownloadURL({'scale': 10, 'crs': 'EPSG:4326', 'format': 'NPY',
                                             'region': ee.Geometry.Rectangle(
                                                 [lo - half * RES, la - half * RES, lo + half * RES, la + half * RES])})
                    s, _pid = VC.sess(acct)
                    r = s.get(url, timeout=120)
                    r.raise_for_status()
                    z = np.load(io.BytesIO(r.content))
                    arr = np.stack([z['B4'], z['B3'], z['B2'], z['B8']], -1)
                tc = np.dstack([stretch(arr[..., 0]), stretch(arr[..., 1]), stretch(arr[..., 2])])
                fc = np.dstack([stretch(arr[..., 3]), stretch(arr[..., 0]), stretch(arr[..., 1])])
                Image.fromarray(np.hstack([tc, fc])).save(png)
            append(fp, ['point_id', 'png', 'status'], [[p, png, 'ok']])
        except Exception as e:
            append(fp, ['point_id', 'png', 'status'], [[p, png, 'ERR:' + str(e)[:80]]])
        t1 = time.time() - t0
        if (k + 1) % 10 == 0 or k < 3:
            print('chip %d/%d %.1fs' % (k + 1, len(todo), t1), flush=True)
    print('chip shard %d 完成' % shard, flush=True)


if __name__ == '__main__':
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=['ind', 'chip'], required=True)
    ap.add_argument('--points', required=True)
    ap.add_argument('--acct', required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--out', required=True)
    ap.add_argument('--pid', default=None)
    a = ap.parse_args()
    ee, pid = init(a.acct, a.pid)
    ch, han, s2 = resolve_assets(ee, pid)
    pts = read_points(a.points, a.shard, a.nshards)
    print('账号 %s 项目 %s  点位 %d（shard %d/%d）' % (a.acct, pid, len(pts), a.shard, a.nshards), flush=True)
    if a.task == 'ind':
        run_ind(ee, ch, han, s2, pts, a.out, a.shard)
    else:
        run_chip(ee, s2, pts, a.out, a.shard, a.acct)
    print('ALL DONE %.0fs' % (time.time() - t0), flush=True)
