# -*- coding: utf-8 -*-
"""p1_core.py — CGLS-LC100 C3 forest_type 三年一致 100 m 核心区（needle / broad）
* 输入：GEE `COPERNICUS/Landcover/100m/Proba-V-C3/Global/{2017,2018,2019}`
        瓦片清单 `代码/8.全国生产/plan/tiles.csv`
* 规则源：技术文档/79 §3.1（预注册，不得修改判据）
        核心 = ft2017==ft2018==ft2019 & ft2019∈{1..4} & tcf19>=60 & ddi19>=2
        needle = ft19∈{1,3}   broad = ft19∈{2,4}   ft19=5 混交不取
        连续区核心 = 对每类 3×3 腐蚀（focal_min），去掉 100 m 边缘混像元
* 门槛：GEE 单请求 ≤48 MB → 8 个布尔量打包进 1 个 uint8 波段（~7.5 MB/瓦）
        腐蚀改为本地 numpy 3×3 腐蚀（请求外围留 1 像元 margin，跨瓦边正确）
* 输出（`F:/lc_work/nb_national_2026/cgls/`）：
        core/{TILE}_needle_core.tif、core/{TILE}_broad_core.tif（uint8，1=核心）
        p1_stats.json、p1_stats_by_prov.csv、p1_tiles.csv
        p1_checkpoint.jsonl（逐瓦台账，幂等续跑）、p1_failed.json
* 用法：
        python p1_core.py                 # 下载 + 聚合（自动续跑）
        python p1_core.py --limit 20      # 只跑前 20 个待做瓦（测速/分段）
        python p1_core.py --agg-only      # 只聚合已有台账
        python p1_core.py --tiles T0123,T0124
"""
import os, sys, json, time, math, argparse, warnings, traceback

sys.stdout.reconfigure(encoding='utf-8')
PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
sys.path.insert(0, os.path.join(PROJ, '代码', '8.全国生产'))
import prod_conf as C

for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy',
          'NO_PROXY', 'no_proxy'):
    os.environ.pop(k, None)
os.environ['HTTP_PROXY'] = C.PROXY_URL
os.environ['HTTPS_PROXY'] = C.PROXY_URL
os.environ['ALL_PROXY'] = C.PROXY_URL
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine

# ---------------- 路径 ----------------
ROOT = r'F:/lc_work/nb_national_2026/cgls'
CORE = os.path.join(ROOT, 'core')
CKPT = os.path.join(ROOT, 'p1_checkpoint.jsonl')
FAILED = os.path.join(ROOT, 'p1_failed.json')
STATS = os.path.join(ROOT, 'p1_stats.json')
PROVCSV = os.path.join(ROOT, 'p1_stats_by_prov.csv')
TILECSV = os.path.join(ROOT, 'p1_tiles.csv')
TILES_CSV = os.path.join(PROJ, '代码', '8.全国生产', 'plan', 'tiles.csv')
BOUNDARY = C.BOUNDARY

# ---------------- CGLS 原生网格（已实测 EPSG:3857 / 100 m） ----------------
AID = 'COPERNICUS/Landcover/100m/Proba-V-C3/Global/%d'
R_E = 6378137.0
X0G, Y0G, PX = -20037550.0, 15538800.0, 100.0
MARGIN = 1                      # 腐蚀用外围 1 像元（保证跨瓦边界正确）
PACK_BANDS = ['pack']

# 账号顺序：先用已验证可用的，失败轮换
ACCTS = ['zitwwufh', 'gm9ufoo4', 'nnjxsn', 'seqsiu', 'iu5f4z', 'xysongf',
         'ybzljyma', 'zixen8v8', 'y30b63ye', 'nhqz5uj', '1yxth5g', 'ief3nj', '11e0n00']

# ---------------- 省 → 生态区（任务口径 8 组；青藏单列） ----------------
REGION = {
    '辽宁省': '东北', '吉林省': '东北', '黑龙江省': '东北',
    '北京市': '华北', '天津市': '华北', '河北省': '华北', '山西省': '华北', '内蒙古自治区': '华北',
    '陕西省': '西北', '甘肃省': '西北', '宁夏回族自治区': '西北', '新疆维吾尔自治区': '西北',
    '青海省': '青藏', '西藏自治区': '青藏',
    '重庆市': '西南', '四川省': '西南', '贵州省': '西南', '云南省': '西南',
    '河南省': '华中', '湖北省': '华中', '湖南省': '华中',
    '上海市': '华东', '江苏省': '华东', '浙江省': '华东', '安徽省': '华东',
    '福建省': '华东', '江西省': '华东', '山东省': '华东', '台湾省': '华东',
    '广东省': '华南', '广西壮族自治区': '华南', '海南省': '华南',
    '香港特别行政区': '华南', '澳门特别行政区': '华南',
}
REGION_ORDER = ['东北', '华北', '西北', '青藏', '西南', '华中', '华东', '华南', '其他']


# ---------------- GEE 初始化 / 账号轮换 ----------------
class GEE:
    def __init__(self):
        self.ee = None
        self.acct = None
        self.pid = None
        self.fails = 0
        self.idx = 0

    def use(self, i=None):
        import ee
        if i is not None:
            self.idx = i
        n = len(ACCTS)
        for k in range(n):
            a = ACCTS[(self.idx + k) % n]
            try:
                with warnings.catch_warnings(record=True) as ws:
                    warnings.simplefilter('always')
                    C.apply_account_env(a)
                    pid = C.ACCOUNTS.get(a, {}).get('proj')
                    if not pid:
                        continue
                    ee.Initialize(project=pid)
                    msg = ' '.join(str(w.message) for w in ws)
                if 'restricted' in msg or 'quota' in msg:
                    print('  [acct] %s RESTRICTED' % a, flush=True)
                    continue
                self.ee, self.acct, self.pid = ee, a, pid
                self.idx = (self.idx + k) % n
                print('  [acct] using %s / %s' % (a, pid), flush=True)
                return True
            except Exception as e:
                print('  [acct] %s FAIL %s' % (a, str(e)[:90]), flush=True)
        return False

    def rotate(self):
        return self.use((self.idx + 1) % len(ACCTS))


def merc(lon, lat):
    return (R_E * math.radians(lon),
            R_E * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)))


def grid_of(x0, y0, x1, y1, margin=MARGIN):
    """瓦片 → 原生 100 m Mercator 网格（含 margin），返回 (grid dict, width, height, affine, col0, row0)"""
    mx0, my0 = merc(x0, y0)
    mx1, my1 = merc(x1, y1)
    c0 = int(round((mx0 - X0G) / PX)) - margin
    c1 = int(round((mx1 - X0G) / PX)) + margin
    r0 = int(round((Y0G - my1) / PX)) - margin
    r1 = int(round((Y0G - my0) / PX)) + margin
    w, h = c1 - c0, r1 - r0
    tx = X0G + c0 * PX
    ty = Y0G - r0 * PX
    g = {'dimensions': {'width': w, 'height': h},
         'affineTransform': {'scaleX': PX, 'shearX': 0, 'translateX': tx,
                             'shearY': 0, 'scaleY': -PX, 'translateY': ty},
         'crsCode': 'EPSG:3857'}
    return g, w, h, Affine(PX, 0, tx, 0, -PX, ty), c0, r0


# ---------------- 位序（uint16 打包，10 位） ----------------
# bit0 any3(ft∈1..4 任一年)  bit1 all3(三年同且∈1..4)  bit2 gate19  bit3 sel(=all3&gate19)
# bit4 npre  bit5 bpre  bit6 mixed5(ft19==5)  bit7 valid
# bit8 any3_gated(任一年森林且当年过 tcf/ddi 门)  bit9 any5(ft∈1..5 任一年)
B_ANY3, B_ALL3, B_GATE, B_SEL, B_NPRE, B_BPRE, B_MIX, B_VALID, B_ANYG, B_ANY5 = range(10)


def build_pack(ee):
    """构造 10 位打包影像（单 uint16 波段，请求体积 ~15 MB/瓦，仍低于 48 MB）"""
    imgs = [ee.Image(AID % y).select(['forest_type', 'tree-coverfraction',
                                      'data-density-indicator'])
            for y in (2017, 2018, 2019)]
    ft = [i.select('forest_type') for i in imgs]
    tcf = [i.select('tree-coverfraction') for i in imgs]
    ddi = [i.select('data-density-indicator') for i in imgs]
    isf = [(x.gte(1)).And(x.lte(4)) for x in ft]          # 纯林型 1–4
    isf5 = [(x.gte(1)).And(x.lte(5)) for x in ft]         # 含混交 5
    gate = [tcf[i].gte(60).And(ddi[i].gte(2)) for i in range(3)]

    any3 = isf[0].Or(isf[1]).Or(isf[2])
    any5 = isf5[0].Or(isf5[1]).Or(isf5[2])
    anyg = (isf[0].And(gate[0])).Or(isf[1].And(gate[1])).Or(isf[2].And(gate[2]))
    all3 = ft[0].eq(ft[1]).And(ft[1].eq(ft[2])).And(isf[2])
    gate19 = gate[2]
    sel = all3.And(gate19)
    npre = sel.And(ft[2].eq(1).Or(ft[2].eq(3)))
    bpre = sel.And(ft[2].eq(2).Or(ft[2].eq(4)))
    mixed = ft[2].eq(5)
    valid = ft[2].mask()

    bits = [any3, all3, gate19, sel, npre, bpre, mixed, valid, anyg, any5]
    pack = ee.Image(0).toUint16()
    for i, x in enumerate(bits):
        pack = pack.bitwiseOr(x.unmask(0).toUint16().leftShift(i))
    return pack.rename('pack')


def erode3x3(m):
    """布尔 3×3 腐蚀（全邻域为 1 才保留）"""
    m = m.astype(bool)
    h, w = m.shape
    out = np.ones((h, w), dtype=bool)
    p = np.zeros((h + 2, w + 2), dtype=bool)
    p[1:-1, 1:-1] = m
    for dy in range(3):
        for dx in range(3):
            out &= p[dy:dy + h, dx:dx + w]
    return out


# ---------------- 省界栅格化（一次载入，per-tile rasterize） ----------------
class ProvLayer:
    """省界 → 一次重投影到 EPSG:3857（与 CGLS 原生网格同 CRS），逐瓦 rasterize。"""

    def __init__(self, path):
        import shapely.geometry as sg
        from shapely.ops import transform as shp_transform
        from rasterio.warp import transform as warp_xy
        d = json.load(open(path, encoding='utf-8'))
        self.feats = [f for f in d['features'] if f['properties'].get('name')]
        self.names = [f['properties']['name'] for f in self.feats]
        self.shapes = []
        for f in self.feats:
            g = sg.shape(f['geometry'])
            # GCJ-02/经纬度 → EPSG:3857（仅做栅格化定位，不做精度修正）
            g = shp_transform(lambda x, y, z=None: warp_xy('EPSG:4326', 'EPSG:3857', x, y), g)
            self.shapes.append(g)
        self.bounds = [s.bounds for s in self.shapes]
        self.index = {n: i + 1 for i, n in enumerate(self.names)}

    def labels(self, transform, width, height):
        from rasterio.features import rasterize
        tb = (transform.c, transform.f + transform.e * height,
              transform.c + transform.a * width, transform.f)
        shapes, vals = [], []
        for shp, b, i in zip(self.shapes, self.bounds, range(1, len(self.shapes) + 1)):
            if b[2] <= tb[0] or b[0] >= tb[2] or b[3] <= tb[1] or b[1] >= tb[3]:
                continue
            shapes.append((shp, i)); vals.append(i)
        if not shapes:
            return None
        return rasterize(shapes, out_shape=(height, width), transform=transform,
                         fill=0, dtype='int16', all_touched=False)


# ---------------- 单瓦处理 ----------------
def fetch_tile(gee, pack, x0, y0, x1, y1):
    g, w, h, tr, c0, r0 = grid_of(x0, y0, x1, y1)
    last = None
    for attempt in range(5):
        try:
            b = gee.ee.data.computePixels({'expression': pack, 'fileFormat': 'GEO_TIFF',
                                           'bandIds': PACK_BANDS, 'grid': g})
            with rasterio.io.MemoryFile(b) as mem:
                with mem.open() as ds:
                    arr = ds.read(1)
                    if ds.width != w or ds.height != h:
                        raise RuntimeError('grid mismatch %dx%d vs %dx%d' % (ds.width, ds.height, w, h))
            return arr, g, w, h, tr, c0, r0
        except Exception as e:
            last = e
            msg = str(e)[:160]
            print('    retry %d: %s' % (attempt, msg), flush=True)
            if any(s in msg for s in ('restricted', 'quota', 'permission', '403',
                                      'Unauthorized', 'invalid_grant')):
                if not gee.rotate():
                    raise RuntimeError('no usable account: %s' % msg)
            time.sleep(min(20, 2 ** attempt) + (0.3 if attempt else 0))
    raise RuntimeError('fetch failed: %s' % str(last)[:200])


def write_tif(path, arr, transform, crs='EPSG:3857'):
    prof = dict(driver='GTiff', width=arr.shape[1], height=arr.shape[0], count=1,
                dtype='uint8', crs=crs, transform=transform, compress='deflate',
                tiled=True, blockxsize=256, blockysize=256, BIGTIFF='YES')
    tmp = path + '.tmp'
    with rasterio.open(tmp, 'w', **prof) as dst:
        dst.write(arr.astype(np.uint8), 1)
    os.replace(tmp, path)


def process_tile(gee, pack, prov, row):
    t = row.tile
    x0, y0, x1, y1 = float(row.x0), float(row.y0), float(row.x1), float(row.y1)
    arr, g, w, h, tr, c0, r0 = fetch_tile(gee, pack, x0, y0, x1, y1)

    bit = lambda k: ((arr >> k) & 1).astype(bool)
    npre_f = bit(B_NPRE)
    bpre_f = bit(B_BPRE)
    ncore_f = erode3x3(npre_f)
    bcore_f = erode3x3(bpre_f)

    # 裁掉 margin，回到瓦片本体
    sl = (slice(MARGIN, h - MARGIN), slice(MARGIN, w - MARGIN))
    ncore = ncore_f[sl]
    bcore = bcore_f[sl]
    hh, hw = ncore.shape          # hh = 行数（纬度方向），hw = 列数（经度方向）
    tr_in = Affine(PX, 0, X0G + (c0 + MARGIN) * PX, 0, -PX, Y0G - (r0 + MARGIN) * PX)

    pn = os.path.join(CORE, '%s_needle_core.tif' % t)
    pb = os.path.join(CORE, '%s_broad_core.tif' % t)
    write_tif(pn, ncore.astype(np.uint8), tr_in)
    write_tif(pb, bcore.astype(np.uint8), tr_in)

    # ---- 逐行真实地面面积（EPSG:3857 的 100 m 像元 = 100·cosφ 地面米）----
    yrow = Y0G - (r0 + MARGIN + np.arange(hh) + 0.5) * PX
    lat = np.arctan(np.sinh(yrow / R_E))
    row_km2 = 1e4 * np.cos(lat) ** 2 / 1e6          # km² / 像元
    n_rows = ncore.sum(1).astype(np.float64)
    b_rows = bcore.sum(1).astype(np.float64)
    n_geo = float((n_rows * row_km2).sum())
    b_geo = float((b_rows * row_km2).sum())

    # ---- 省归属（只在瓦片本体上栅格化）----
    b_any3 = bit(B_ANY3)[sl]; b_all3 = bit(B_ALL3)[sl]
    b_anyg = bit(B_ANYG)[sl]; b_any5 = bit(B_ANY5)[sl]
    prov_counts = {}
    lab = prov.labels(tr_in, hw, hh)
    if lab is not None:
        sel = bit(B_SEL)[sl]
        for pidx in np.unique(lab):
            if pidx == 0:
                continue
            m = (lab == pidx)
            nn = int(ncore[m].sum()); bb = int(bcore[m].sum())
            prov_counts[str(int(pidx))] = [int(m.sum()), nn, bb,
                                           int(b_any3[m].sum()), int(b_all3[m].sum()),
                                           int(b_anyg[m].sum()), int(b_any5[m].sum()),
                                           int(sel[m].sum())]

    rec = dict(
        tile=t, x0=x0, y0=y0, x1=x1, y1=y1,
        land_frac=float(row.land_frac),
        crs='EPSG:3857', w=int(hw), h=int(hh), n_px=int(hw * hh),
        n_valid=int(bit(B_VALID)[sl].sum()),
        n_any3=int(b_any3.sum()),
        n_all3=int(b_all3.sum()),
        n_any5=int(b_any5.sum()),
        n_anyg=int(b_anyg.sum()),
        n_gate19=int(bit(B_GATE)[sl].sum()),
        n_sel=int(bit(B_SEL)[sl].sum()),
        n_needle_pre=int(npre_f[sl].sum()),
        n_broad_pre=int(bpre_f[sl].sum()),
        n_needle_core=int(ncore.sum()),
        n_broad_core=int(bcore.sum()),
        n_mixed5=int(bit(B_MIX)[sl].sum()),
        needle_km2_nominal=round(float(ncore.sum()) * 0.01, 4),
        broad_km2_nominal=round(float(bcore.sum()) * 0.01, 4),
        needle_km2_geo=round(n_geo, 4),
        broad_km2_geo=round(b_geo, 4),
        lat_top=round(float(np.degrees(lat[0])), 5), lat_bot=round(float(np.degrees(lat[-1])), 5),
        prov=prov_counts,
    )
    return rec


# ---------------- checkpoint ----------------
def load_ckpt():
    recs = {}
    if os.path.exists(CKPT):
        for line in open(CKPT, encoding='utf-8'):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                recs[r['tile']] = r
            except Exception:
                pass
    return recs


def append_ckpt(rec):
    with open(CKPT, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        f.flush(); os.fsync(f.fileno())


def tif_ok(t):
    a = os.path.join(CORE, '%s_needle_core.tif' % t)
    b = os.path.join(CORE, '%s_broad_core.tif' % t)
    return all(os.path.exists(p) and os.path.getsize(p) > 0 for p in (a, b))


# ---------------- 聚合 ----------------
def aggregate(recs):
    recs = [r for r in recs if r is not None]
    if not recs:
        return None, None
    tot = {}
    for k in ('n_valid', 'n_any3', 'n_all3', 'n_any5', 'n_anyg', 'n_gate19', 'n_sel',
              'n_needle_pre', 'n_broad_pre', 'n_needle_core', 'n_broad_core', 'n_mixed5',
              'n_px'):
        tot[k] = int(sum(r[k] for r in recs))
    tot['needle_km2_nominal'] = float(sum(r['needle_km2_nominal'] for r in recs))
    tot['broad_km2_nominal'] = float(sum(r['broad_km2_nominal'] for r in recs))
    tot['needle_km2_geo'] = float(sum(r['needle_km2_geo'] for r in recs))
    tot['broad_km2_geo'] = float(sum(r['broad_km2_geo'] for r in recs))

    # 分省聚合：8 个计数列 [n_px, needle_core, broad_core, any3, all3, anyg, any5, sel]
    prov_rows = {}
    for r in recs:
        for pidx, v in r.get('prov', {}).items():
            d = prov_rows.setdefault(int(pidx), np.zeros(8, dtype=np.int64))
            d += np.array(v[:8], dtype=np.int64)
    return tot, prov_rows


def write_prov_csv(prov, prov_rows):
    cols = ['prov', 'region', 'n_px', 'needle_core_px', 'broad_core_px',
            'n_any3', 'n_all3', 'C1', 'C1_gated', 'needle_km2', 'broad_km2']
    rows = []
    for pidx, v in prov_rows.items():
        name = prov.names[pidx - 1]
        rows.append(dict(prov=name, region=REGION.get(name, '其他'),
                         n_px=int(v[0]), needle_core_px=int(v[1]), broad_core_px=int(v[2]),
                         n_any3=int(v[3]), n_all3=int(v[4]),
                         C1=(round(float(v[4]) / float(v[3]), 4) if v[3] else None),
                         C1_gated=(round(float(v[7]) / float(v[5]), 4) if v[5] else None),
                         needle_km2=round(v[1] * 0.01, 2), broad_km2=round(v[2] * 0.01, 2)))
    df = pd.DataFrame(rows, columns=cols)
    for c in ('n_px', 'needle_core_px', 'broad_core_px', 'n_any3', 'n_all3'):
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype('int64')
    for c in ('needle_km2', 'broad_km2'):
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0).astype('float64')
    if df.empty:
        pd.DataFrame(columns=cols).to_csv(PROVCSV, index=False, encoding='utf-8-sig')
        return pd.DataFrame(columns=cols)

    # 生态区合计（按省口径求和；腐蚀不跨省累加，属近似，仅用于分省表展示）
    agg = df.groupby('region', as_index=False)[['n_px', 'needle_core_px', 'broad_core_px',
                                               'n_any3', 'n_all3']].sum()
    agg['prov'] = '【生态区合计】'
    agg['C1'] = [round(float(a) / b, 4) if b else None for a, b in zip(agg['n_all3'], agg['n_any3'])]
    agg['C1_gated'] = None
    agg['needle_km2'] = (agg['needle_core_px'] * 0.01).round(2)
    agg['broad_km2'] = (agg['broad_core_px'] * 0.01).round(2)
    agg = agg[cols]
    rg_o = {r: i for i, r in enumerate(REGION_ORDER)}
    df['_o'] = df['region'].map(rg_o).fillna(99)
    df = df.sort_values(['_o', 'needle_core_px'], ascending=[True, False]).drop(columns=['_o'])
    tot = {'prov': '【全国合计】', 'region': '全国'}
    for c in ('n_px', 'needle_core_px', 'broad_core_px', 'n_any3', 'n_all3'):
        tot[c] = int(df[c].sum())
    tot['C1'] = round(float(tot['n_all3']) / tot['n_any3'], 4) if tot['n_any3'] else None
    tot['needle_km2'] = round(tot['needle_core_px'] * 0.01, 2)
    tot['broad_km2'] = round(tot['broad_core_px'] * 0.01, 2)
    agg['_o'] = agg['region'].map(rg_o).fillna(99)
    agg = agg.sort_values('_o').drop(columns=['_o'])
    out = pd.concat([df, agg[cols], pd.DataFrame([tot], columns=cols)], ignore_index=True)
    out.to_csv(PROVCSV, index=False, encoding='utf-8-sig')
    return out


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='最多处理多少个待做瓦')
    ap.add_argument('--tiles', default='', help='只处理这些瓦（逗号分隔）')
    ap.add_argument('--agg-only', action='store_true')
    ap.add_argument('--deadline-min', type=float, default=58.0, help='墙钟预算（分钟）')
    ap.add_argument('--skip-no-land', action='store_true',
                    help='land_frac==0 的瓦不查询（仍写全零掩膜并标注）')
    args = ap.parse_args()

    os.makedirs(CORE, exist_ok=True)
    tiles = pd.read_csv(TILES_CSV, encoding='utf-8-sig')
    prov = ProvLayer(BOUNDARY)

    if not args.agg_only:
        gee = GEE()
        if not gee.use():
            raise SystemExit('NEED_MANUAL: 无可用 GEE 账号')
        pack = build_pack(gee.ee)

        ck = load_ckpt()
        order = tiles.sort_values('land_frac', ascending=False)
        todo = []
        for row in order.itertuples():
            if args.tiles and row.tile not in args.tiles.split(','):
                continue
            if row.tile in ck and tif_ok(row.tile):
                continue
            if args.skip_no_land and float(row.land_frac) <= 0.0:
                continue
            todo.append(row)
        if args.limit:
            todo = todo[:args.limit]
        print('待处理 %d 瓦 / 总 %d 瓦；已完成 %d' % (len(todo), len(tiles), len(ck)), flush=True)

        failed = {}
        if os.path.exists(FAILED):
            try:
                failed = json.load(open(FAILED, encoding='utf-8'))
            except Exception:
                failed = {}
        t_start = time.time()
        for i, row in enumerate(todo, 1):
            el = (time.time() - t_start) / 60.0
            if el >= args.deadline_min:
                print('达到墙钟预算 %.1f min，停在 %d/%d' % (el, i - 1, len(todo)), flush=True)
                break
            t0 = time.time()
            try:
                rec = process_tile(gee, pack, prov, row)
                append_ckpt(rec)
                ck[rec['tile']] = rec
                failed.pop(rec['tile'], None)
                print('[%3d/%3d] %s %.1fs n=%d b=%d (pre n=%d b=%d) C1=%.3f' % (
                    i, len(todo), row.tile, time.time() - t0,
                    rec['n_needle_core'], rec['n_broad_core'],
                    rec['n_needle_pre'], rec['n_broad_pre'],
                    rec['n_all3'] / max(1, rec['n_any3'])), flush=True)
            except Exception as e:
                failed[row.tile] = dict(err=str(e)[:400], t=time.strftime('%Y-%m-%d %H:%M:%S'))
                print('[%3d/%3d] %s FAILED %s' % (i, len(todo), row.tile, str(e)[:160]), flush=True)
                traceback.print_exc(limit=2)
                # 连续失败则换号
                if len(failed) % 3 == 0:
                    gee.rotate()
                time.sleep(2)
            json.dump(failed, open(FAILED, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    # ---- 聚合 ----
    ck = load_ckpt()
    recs = [ck[t] for t in ck]
    tot, prov_rows = aggregate(recs)
    if tot is None:
        raise SystemExit('无任何已完成瓦')

    reg_tab = {}
    for pidx, v in prov_rows.items():
        nm = prov.names[pidx - 1]
        rg = REGION.get(nm, '其他')
        d = reg_tab.setdefault(rg, np.zeros(8, dtype=np.int64))
        d += np.array(v, dtype=np.int64)

    def ratio(a, b):
        return round(float(a) / float(b), 4) if b else None

    stats = dict(
        built=time.strftime('%Y-%m-%d %H:%M:%S'),
        spec='技术文档/79 §3.1（预注册 C1 判据）',
        grid=dict(crs='EPSG:3857', pixel_m=100.0,
                  note='GEE 原生网格；100 m 为 WebMercator 名义米，地面面积另按 cos²φ 折算'),
        tiles_total=int(len(tiles)), tiles_done=int(len(recs)),
        tiles_missing=[t for t in tiles.tile if t not in ck],
        national=dict(
            n_px=int(tot['n_px']),
            n_valid=int(tot['n_valid']),
            n_any3=int(tot['n_any3']), n_all3=int(tot['n_all3']),
            n_any5=int(tot['n_any5']), n_anyg=int(tot['n_anyg']),
            n_gate19=int(tot['n_gate19']), n_sel=int(tot['n_sel']),
            n_mixed5=int(tot['n_mixed5']),
            n_needle_pre=int(tot['n_needle_pre']), n_broad_pre=int(tot['n_broad_pre']),
            n_needle_core=int(tot['n_needle_core']), n_broad_core=int(tot['n_broad_core']),
            needle_km2_nominal=round(tot['needle_km2_nominal'], 2),
            broad_km2_nominal=round(tot['broad_km2_nominal'], 2),
            needle_km2_geo=round(tot['needle_km2_geo'], 2),
            broad_km2_geo=round(tot['broad_km2_geo'], 2),
            core_km2_nominal=round(tot['needle_km2_nominal'] + tot['broad_km2_nominal'], 2),
            core_km2_geo=round(tot['needle_km2_geo'] + tot['broad_km2_geo'], 2),
        ),
        C1=dict(
            definition='三年林型一致的森林像元 / 三年任一年是森林的像元（ft∈{1,2,3,4}，无 tcf/ddi 门）',
            national=ratio(tot['n_all3'], tot['n_any3']),
            definition_gated='三年同&过 2019 tcf/ddi 门的像元 / 任一年森林且过该年 tcf/ddi 门的像元',
            national_gated=ratio(tot['n_sel'], tot['n_anyg']),
            definition_with_mixed='(三年同 & ∈1..4) / (任一年 ∈1..5，含混交)',
            national_with_mixed=ratio(tot['n_all3'], tot['n_any5']),
            threshold=0.70,
            by_region={rg: dict(n_any3=int(v[3]), n_all3=int(v[4]),
                                C1=ratio(v[4], v[3]),
                                n_anyg=int(v[5]), n_sel=int(v[7]),
                                C1_gated=ratio(v[7], v[5]),
                                n_any5=int(v[6]),
                                C1_with_mixed=ratio(v[4], v[6]))
                       for rg, v in reg_tab.items()},
        ),
        erosion=dict(
            needle=dict(pre=int(tot['n_needle_pre']), post=int(tot['n_needle_core']),
                        retention=ratio(tot['n_needle_core'], tot['n_needle_pre'])),
            broad=dict(pre=int(tot['n_broad_pre']), post=int(tot['n_broad_core']),
                       retention=ratio(tot['n_broad_core'], tot['n_broad_pre'])),
            all=dict(pre=int(tot['n_needle_pre'] + tot['n_broad_pre']),
                     post=int(tot['n_needle_core'] + tot['n_broad_core']),
                     retention=ratio(tot['n_needle_core'] + tot['n_broad_core'],
                                     tot['n_needle_pre'] + tot['n_broad_pre'])),
        ),
        by_region={rg: dict(n_px=int(v[0]), needle_core_px=int(v[1]), broad_core_px=int(v[2]),
                            needle_km2=round(v[1] * 0.01, 2), broad_km2=round(v[2] * 0.01, 2),
                            n_any3=int(v[3]), n_all3=int(v[4]), C1=ratio(v[4], v[3]),
                            n_any5=int(v[6]), C1_with_mixed=ratio(v[4], v[6]))
                   for rg, v in reg_tab.items()},
    )
    # 零值瓦
    stats['tiles_zero_needle'] = [r['tile'] for r in recs if r['n_needle_core'] == 0]
    stats['tiles_zero_broad'] = [r['tile'] for r in recs if r['n_broad_core'] == 0]
    stats['tiles_zero_both'] = [r['tile'] for r in recs
                                if r['n_needle_core'] == 0 and r['n_broad_core'] == 0]
    if os.path.exists(FAILED):
        stats['failed'] = json.load(open(FAILED, encoding='utf-8'))

    json.dump(stats, open(STATS, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    df = pd.DataFrame(recs).drop(columns=['prov'])
    df.to_csv(TILECSV, index=False, encoding='utf-8-sig')
    write_prov_csv(prov, prov_rows)

    print(json.dumps({k: stats[k] for k in ('tiles_done', 'national', 'C1', 'erosion')},
                     ensure_ascii=False, indent=1))
    if stats['tiles_missing']:
        print('未完成瓦 %d 个：%s' % (len(stats['tiles_missing']),
                                   ','.join(stats['tiles_missing'][:40])), flush=True)


if __name__ == '__main__':
    main()
