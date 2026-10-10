# -*- coding: utf-8 -*-
"""h1_nat_cand.py — 全国灌丛候选点生成（2° 盒随机点 + 便宜证据层采样 + 本地分层）

* 输入：GEE 便宜源（见 build_cheap）——DW v1 2023（label 众数 + shrub_and_scrub 年均）、
        ESA WC v200(20)、Copernicus LC100(20)、MODIS MCD12Q1 IGBP6/7、Hansen treecover2000、
        SRTM DEM/坡度、WorldClim bio01/bio12、JRC GSW occurrence
        （S2 物候 / GEDI 冠层高度 / AEF64 属贵层，**不在本阶段取**，由 h2b 对入选点补取）
* 规则源（预注册分层，写死本脚本）：
        P 带 = dwp ≥0.35（正向候选）；N 带 = 0.15 ≤ dwp <0.35（模糊带→硬负样本）
        ——阈值沿用 doc57 四省实验定义，放量期间不再调整
* 门槛：每盒 25,000 随机点；入选去重 ≥1 km；每盒 P ≤ --cap 点、N ≤ cap/4 点
* 输出：data/shrub_nat/parts/cand_<part>.parquet
* 用法：python h1_nat_cand.py --part 0 --nparts 10 --acct bx15mw   （h1_drive.py 多账号并行）
* 踩坑记录：Image.sample() 在掩膜图上返回点数远小于 numPixels（100 m 聚合下 66k 合格像元只回 7 点），
        且 getInfo() 超 5,000 要素会中断 —— 故一律用"随机点 + sampleRegions + 本地过滤"。
"""
import argparse, json, os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
sys.path.insert(0, os.path.join(WORK, 'code'))
import numpy as np
import pandas as pd
import v31_common as VC

OUTD = os.path.join(WORK, 'data', 'shrub_nat')
PARTD = os.path.join(OUTD, 'parts')
BOXES = []
_lon = 73.0
while _lon < 135.0:
    _la = 17.5
    while _la < 54.0:
        BOXES.append((round(_lon, 1), round(_la, 1), round(_lon + 2, 1), round(_la + 2, 1)))
        _la += 2.0
    _lon += 2.0
SEED = 20261010
P_MIN, N_MIN = 0.35, 0.15
N_RAND = 25000   # 默认值；可用 --rand 覆盖
MIN_KM = 1.0


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + 'P%d %s' % (os.getpid(), m), flush=True)


def build_cheap(r):
    dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1').filterDate('2023-01-01', '2023-12-31').filterBounds(r)
    dwi = ee.Image.cat([dw.select('label').mode().rename('dwl'),
                        dw.select('shrub_and_scrub').mean().rename('dwp')])
    wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map')
    lc = ee.Image('COPERNICUS/Landcover/100m/Proba-V-C3/Global/2019').select('discrete_classification')
    mcd = ee.ImageCollection('MODIS/061/MCD12Q1').filterDate('2023-01-01', '2023-12-31').first().select('LC_Type1')
    dem = ee.Image('USGS/SRTMGL1_003')
    tc = ee.Image('UMD/hansen/global_forest_change_2023_v1_11').select('treecover2000')
    occ = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence')
    bio = ee.Image('WORLDCLIM/V1/BIO').select(['bio01', 'bio12'])
    # 逐波段 unmask：sampleRegions 会丢掉"任一波段为 null"的整点（2026-10-10 实测：11 波段堆叠后
    # 25,000 随机点仅回 16 条，逐波段 unmask 才恢复正常命中率）
    img = ee.Image.cat([
        dwi.unmask(-999), wc.eq(20).unmask(0).rename('wc20'), lc.eq(20).unmask(0).rename('lc20'),
        mcd.eq(6).Or(mcd.eq(7)).unmask(0).rename('mcd'), tc.unmask(-1).rename('tc'),
        dem.unmask(-999).rename('dem'), ee.Terrain.slope(dem).unmask(-999).rename('slope'),
        bio.unmask(-999), occ.unmask(-1).rename('occ')]).toFloat()
    return img


def land_boxes():
    """用 DataV 全国界做盒级陆地过滤（缓存 data/shrub_nat/boxes_land.json）。
    返回 BOXES 的子集（仅留陆地交叠 ≥5% 的盒），省掉远洋盒的空跑。"""
    fp = os.path.join(OUTD, 'boxes_land.json')
    if os.path.exists(fp):
        return json.load(open(fp, encoding='utf-8'))
    from shapely.geometry import shape, box as sbox, MultiPolygon
    from shapely import make_valid
    gj = json.load(open(r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/边界/china_100000_full.json',
                        encoding='utf-8'))
    polys = []
    for f in gj['features']:
        g = f.get('geometry')
        if not g:
            continue
        try:
            gg = make_valid(shape(g))
            if gg.is_empty:
                continue
            polys.append(gg)
        except Exception:
            continue
    geo = MultiPolygon(polys) if all(p.geom_type == 'Polygon' for p in polys) else \
        (polys[0] if len(polys) == 1 else None)
    keep = []
    for b in BOXES:
        sb = sbox(*b)
        ok = False
        try:
            for g in polys:           # DataV 有 invalid 几何，逐要素算最稳
                if sb.intersects(g) and sb.intersection(g).area / sb.area >= 0.05:
                    ok = True
                    break
        except Exception:
            ok = True
        if ok:
            keep.append(b)
    VC.jsave(keep, fp)
    return keep


def main():
    global ee
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', type=int, required=True)
    ap.add_argument('--nparts', type=int, required=True)
    ap.add_argument('--cap', type=int, default=1500, help='每盒入选上限')
    ap.add_argument('--rand', type=int, default=25000, help='每盒随机点数')
    ap.add_argument('--tag', default='', help='输出 part 的文件名后缀（加抽轮次用）')
    ap.add_argument('--acct', default='bx15mw')
    ap.add_argument('--boxes', default='', help="只跑指定盒，格式 'lon_lat,lon_lat'（重跑失败盒用）")
    a = ap.parse_args()
    os.makedirs(PARTD, exist_ok=True)
    import ee as _ee
    ee = _ee
    VC.ensure_ctx(a.acct)

    LAND = land_boxes()
    if a.boxes:
        want = set(x for x in a.boxes.split(',') if x)
        mine = [b for b in LAND if '%.1f_%.1f' % b[:2] in want]
    else:
        mine = [b for i, b in enumerate(LAND) if i % a.nparts == a.part]
    out_rows = []
    t0 = time.time()
    for k, box in enumerate(mine):
        bid = '%06d' % LAND.index(box)
        r = ee.Geometry.Rectangle(list(box))
        rs = np.random.RandomState(SEED + LAND.index(box))
        try:
            img = build_cheap(r)
            nr = a.rand
            lon = rs.uniform(box[0], box[2], nr)
            lat = rs.uniform(box[1], box[3], nr)
            pts = [ee.Feature(ee.Geometry.Point([float(x), float(y)]), {'rid': i})
                   for i, (x, y) in enumerate(zip(lon, lat))]
            got = []
            for s in range(0, len(pts), 4000):
                fc = ee.FeatureCollection(pts[s:s + 4000])
                for att in range(6):
                    try:
                        got += [f['properties'] for f in
                                img.sampleRegions(collection=fc, scale=200, geometries=False,
                                                 tileScale=4).getInfo()['features']]
                        break
                    except Exception as e:
                        emit('  %s 块 %d 重试%d %s' % (box, s, att + 1, str(e)[:60]))
                        time.sleep(8 * (att + 1))
            if not got:
                continue
            df = pd.DataFrame(got)
            df['lon'] = lon[df['rid'].to_numpy()]
            df['lat'] = lat[df['rid'].to_numpy()]
            df['box'] = bid
            df['band'] = np.where(df['dwp'] >= P_MIN, 'P', np.where(df['dwp'] >= N_MIN, 'N', ''))
            df = df[df['band'] != '']
            # 本地去重（≥1 km）+ 每带上限
            sel = []
            for bn in ('P', 'N'):
                sub = df[df.band == bn]
                lim = a.cap if bn == 'P' else a.cap // 4
                if len(sub) > lim:
                    take = min(lim * 3, len(sub))
                    sub = sub.iloc[rs.choice(len(sub), take, replace=False)]
                used = []
                for rec in sub.itertuples():
                    if any(((rec.lon - u[0]) ** 2 * 100 + (rec.lat - u[1]) ** 2 * 120) <
                           (MIN_KM / 111.0) ** 2 for u in used):
                        continue
                    used.append((rec.lon, rec.lat))
                    sel.append(rec._asdict())
                    if len(used) >= lim:
                        break
            out_rows += sel
        except Exception as e:
            emit('  盒 %s 失败 %s' % (box, str(e)[:80]))
        if (k + 1) % 5 == 0:
            emit('%s %d/%d 盒，入选 %d（%.0fs）' % (a.acct, k + 1, len(mine), len(out_rows), time.time() - t0))
    if out_rows:
        df = pd.DataFrame(out_rows)
        tag = ('cand_%s' % a.tag) if a.tag else ('cand_retry' if a.boxes else 'cand_%03d' % a.part)
        fp = os.path.join(PARTD, '%s.parquet' % tag)
        df.to_parquet(fp, index=False)
        emit('→ %s（%d 行，命中率 %.1f%%）' % (fp, len(df), 100.0 * len(df) / (N_RAND * max(1, len(mine)))))
    else:
        emit('本部分无数据')


if __name__ == '__main__':
    main()
