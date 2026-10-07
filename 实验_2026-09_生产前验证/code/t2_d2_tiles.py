# -*- coding: utf-8 -*-
"""t2_d2_tiles.py — D2 规则（R4：剔除 glc_fcs10_2023_shrub 灌丛层）下的代表瓦片复测试产
与 t_all.py 的区别：
  · 训练池剔除 `src == 'glc_fcs10_2023_shrub'`（其余 FCS10 混类抽样保留 → 稀有类不丢）
  · 独立注册表 data/tiles2_registry.json；资产名 v31t2_* / v31r2_*
  · 产物落 `试点成果_2023瓦片_R4复测/`，保留原 `试点成果_2023瓦片/` 作对照
瓦片：秦岭（w2 森林）、松嫩（w1 农林）—— 二者灌丛层占池 25%（检验代价与收益）；三江未跑（该窗无灌丛层，R4≡R0）
子命令：pools | upload | submit | poll | download | colorize | qa | preview
"""
import os, sys, time, math, json, csv, subprocess, collections
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v31_common as VC
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

TILES = {
    'qinling': dict(box=[107.0, 33.0, 109.0, 35.0], note='森林（秦岭）', host='e5h08k'),
    'songnen': dict(box=[123.0, 45.0, 125.0, 47.0], note='农林（松嫩）', host='hqzub6'),
}
YEAR = 2023
SRC = r'F:/lc_work/年度子集_含稀有类/r7_train_2023.parquet'
DROP_SRC = 'glc_fcs10_2023_shrub'
SHARD = 3000
P = VC.cfg('frozen_params.json')
T2D = os.path.join(VC.DATA, 'tiles2')
RESD = os.path.join(VC.RES, 'tiles2')
REGD = os.path.join(VC.DATA, 'tiles2_registry.json')
DST = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/试点成果_2023瓦片_R4复测'
GEESCRIPT = r'C:/Users/Administrator/.zcode/skills/geefast-download/scripts/gee_rest_compute_pixels.py'


def reg(k=None, v=None):
    d = VC.jload(REGD, {})
    if k is None:
        return d
    if v is None:
        return d.get(k)
    d[k] = v; VC.jsave(d, REGD)
    return v


# ---------- 1. 建池（本地，流式） ----------
def build_pools():
    os.makedirs(T2D, exist_ok=True)
    boxes = {t: [v['box'][0] - 2.0, v['box'][1] - 2.0, v['box'][2] + 2.0, v['box'][3] + 2.0]
             for t, v in TILES.items()}
    cols = ['row_id', 'lon', 'lat', 'class_new', 'src'] + VC.FEATS
    acc = {t: [] for t in TILES}
    n_drop = collections.Counter()
    pf = pq.ParquetFile(SRC)
    for b in pf.iter_batches(batch_size=200_000, columns=cols):
        d = b.to_pandas()
        s = d['src'].fillna('').astype(str)
        drop = (s == DROP_SRC).to_numpy()
        for t, bx in boxes.items():
            m = ((d.lon >= bx[0]) & (d.lon <= bx[2]) & (d.lat >= bx[1]) & (d.lat <= bx[3])).to_numpy()
            n_drop[t] += int((m & drop).sum())
            m = m & (~drop)
            if m.any():
                acc[t].append(d.loc[m, [c for c in cols if c != 'src']])
    for t in TILES:
        d = pd.concat(acc[t], ignore_index=True)
        d['row_id'] = d['row_id'].astype('int64')
        h = (d.row_id.to_numpy() * 2654435761) % (2 ** 31)
        d = d.iloc[np.argsort(h, kind='stable')[:20000]]
        d.to_parquet(os.path.join(T2D, 'pool_%s.parquet' % t), index=False)
        VC.emit('%s 池 %d 点（±2° 盒；剔除灌丛层 %d 点）' % (t, len(d), n_drop[t]))


# ---------- 2. 上传（幂等：片级时间闸门，全部瓦片一次处理） ----------
def upload():
    for t, meta in TILES.items():
        acct = meta['host']
        base = 'projects/%s/assets' % VC.pid_of(acct)
        name = 'v31t2_%s' % t
        if reg('pool2_%s' % t):
            VC.emit('%s 已注册，跳过' % t); continue
        fp = os.path.join(T2D, 'pool_%s.parquet' % t)
        d = pd.read_parquet(fp)
        have = VC.list_assets(acct, base) or set()
        nch = math.ceil(len(d) / SHARD)
        missing = [i for i in range(nch) if ('%s_%03d' % (name, i)) not in have]
        guard = reg('shards2_%s' % t) or {}
        now = time.time()
        todo = [i for i in missing if now - float(guard.get(str(i), 0)) > 2700]
        if todo:
            ee, _ = VC.ctx(acct)
            for i in todo:
                ch = d.iloc[i * SHARD:(i + 1) * SHARD]
                recs = [ee.Feature(ee.Geometry.Point([float(r['lon']), float(r['lat'])]),
                                   {'cl': int(r['class_new']), 'lon': float(r['lon']), 'lat': float(r['lat']),
                                    **{f: float(r[f]) for f in VC.FEATS}}) for r in ch.to_dict('records')]
                t2 = ee.batch.Export.table.toAsset(
                    collection=ee.FeatureCollection(recs), assetId='%s/%s_%03d' % (base, name, i),
                    description='v31t2_%s_%03d_%s' % (name, i, time.strftime('%m%d%H%M%S')))
                t2.start()
                guard[str(i)] = now
                VC.emit('  %s 片%03d 提交 %s' % (name, i, t2.id))
                time.sleep(2)
            reg('shards2_%s' % t, guard)
            continue
        if missing:
            VC.emit('%s：%d 片仍在跑（%s），等待' % (t, len(missing), missing)); continue
        ee, _ = VC.ctx(acct)
        fc = ee.FeatureCollection('%s/%s_000' % (base, name))
        for i in range(1, nch):
            fc = fc.merge(ee.FeatureCollection('%s/%s_%03d' % (base, name, i)))
        t2 = ee.batch.Export.table.toAsset(collection=fc, assetId='%s/%s_merged' % (base, name),
                                           description='v31t2m_%s_%s' % (name, time.strftime('%m%d%H%M%S')))
        t2.start()
        reg('pool2_%s' % t, '%s/%s_merged' % (base, name))
        VC.emit('  %s 合并提交 %s' % (name, t2.id))


# ---------- 3. 训练 + 导出 ----------
def submit():
    for t, meta in TILES.items():
        if reg('raster2_%s' % t):
            VC.emit('%s 栅格已提交过' % t); continue
        src = reg('pool2_%s' % t)
        if not src:
            VC.emit('%s 池未注册（可能仍在合并）' % t); continue
        acct = meta['host']
        base, nm = src.rsplit('/', 1)
        have = VC.list_assets(acct, base)
        if have is None or nm not in have:
            VC.emit('%s 池资产未就绪，稍后再跑' % t); continue
        n_q, err = VC.qdepth(acct)
        if n_q is None or n_q > 3:
            VC.emit('%s 队列=%s，稍后' % (acct, n_q)); continue
        ee, pid = VC.ctx(acct)
        box = ee.Geometry.Rectangle(meta['box'])
        tr = VC.remap_fc_v31(ee.FeatureCollection(src))
        clf = ee.Classifier.smileRandomForest(numberOfTrees=P['n_trees'], minLeafPopulation=P['min_leaf'],
                                              maxNodes=P['max_nodes'], seed=7).train(tr, 'cl', VC.FEATS)
        aef = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
               .filterDate('%d-01-01' % YEAR, '%d-01-01' % (YEAR + 1))
               .filterBounds(box).mosaic().select(VC.FEATS))
        cls = aef.classify(clf).rename('class').uint8().clip(box)
        aid = 'projects/%s/assets/v31r2_%s_%d' % (pid, t, YEAR)
        desc = 'v31r2_%s_%d_%s' % (t, YEAR, time.strftime('%m%d%H%M%S'))
        t2 = ee.batch.Export.image.toAsset(image=cls, description=desc, assetId=aid, scale=10,
                                           crs='EPSG:4326', region=meta['box'], maxPixels=10 ** 12, shardSize=16)
        t2.start()
        reg('raster2_%s' % t, dict(asset=aid, desc=desc, host=acct, task=t2.id,
                                   submitted=time.strftime('%Y-%m-%d %H:%M:%S')))
        VC.emit('%s %d 导出提交 @%s → %s' % (t, YEAR, acct, t2.id)); time.sleep(3)


# ---------- 4. 轮询 ----------
def poll():
    for t, meta in TILES.items():
        info = reg('raster2_%s' % t)
        if not isinstance(info, dict):
            VC.emit('%s 未提交' % t); continue
        try:
            e, _ = VC.ctx(info['host'])
            ts = e.data.getTaskStatus(info['task'])
            st = ts[0]['state'] if ts else '?'
            eecu = round(float(ts[0].get('batch_eecu_usage_seconds') or 0) / 3600, 2) if ts else 0.0
            VC.emit('%s[%s] %s  EECU·h=%.2f' % (t, info['host'], st, eecu))
            if st in ('COMPLETED', 'FAILED', 'CANCELLED'):
                i2 = dict(info); i2['state'] = st; i2['eecu_h'] = eecu; reg('raster2_%s' % t, i2)
        except Exception as ex:
            VC.emit('%s 状态查询失败 %s' % (t, str(ex)[:80]))


# ---------- 5. 下载 ----------
def download():
    os.makedirs(RESD, exist_ok=True)
    only = sys.argv[sys.argv.index('--tile') + 1] if '--tile' in sys.argv else None
    workers = sys.argv[sys.argv.index('--workers') + 1] if '--workers' in sys.argv else '12'
    for t, meta in TILES.items():
        if only and t != only:
            continue
        info = reg('raster2_%s' % t)
        if not isinstance(info, dict) or info.get('state') != 'COMPLETED':
            VC.emit('%s 未完成（state=%s）' % (t, (info or {}).get('state'))); continue
        out = os.path.join(RESD, '%s_10m.tif' % t)
        if os.path.exists(out) and os.path.getsize(out) > 1e7:
            VC.emit('%s 已存在' % t); continue
        env = dict(os.environ)
        env['HOME'] = VC.C.cred_home(info['host']); env['USERPROFILE'] = env['HOME']
        env['EE_PROJECT'] = VC.pid_of(info['host'])
        cmd = [sys.executable, GEESCRIPT, '--image', info['asset'], '--bbox', ','.join(str(x) for x in meta['box']),
               '--bands', 'class', '--scale', '10', '--auto-tile', '--workers', workers,
               '--dtype', 'uint16', '--output', out, '--overwrite']
        t0 = time.time()
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        dt = (time.time() - t0) / 60
        sz = os.path.getsize(out) / 1e6 if os.path.exists(out) else 0
        VC.emit('%s 下载 %s %.1f 分钟 %.0f MB' % (t, '✅' if r.returncode == 0 else '❌', dt, sz))
        if r.returncode != 0:
            VC.emit((r.stdout or '')[-300:] + (r.stderr or '')[-200:])


# ---------- 6. 上色 ----------
def colorize():
    import rasterio
    from rasterio.enums import Resampling
    sys.path.insert(0, os.path.join(VC.ROOT, 'code'))
    from d_tiles import COLORS
    os.makedirs(DST, exist_ok=True)
    pal = {i: (230, 230, 230, 0) for i in range(256)}
    for k, c in COLORS.items():
        pal[k] = (int(round(c[0] * 255)), int(round(c[1] * 255)), int(round(c[2] * 255)), 255)
    for k in range(25, 256):
        pal[k] = (200, 200, 200, 255)
    for t in TILES:
        src = os.path.join(RESD, '%s_10m.tif' % t)
        if not os.path.exists(src):
            VC.emit('%s 缺下载文件' % t); continue
        dst = os.path.join(DST, '%s_10m.tif' % t)
        tmp = dst + '.tmp.tif'
        with rasterio.open(src) as ds:
            prof = ds.profile.copy()
            prof.update(dtype='uint8', compress='deflate', predictor=2, tiled=True,
                        blockxsize=512, blockysize=512, count=1, nodata=None)
            with rasterio.open(tmp, 'w', **prof) as out:
                for r0 in range(0, ds.height, 1024):
                    hh = min(1024, ds.height - r0)
                    blk = ds.read(1, window=rasterio.windows.Window(0, r0, ds.width, hh))
                    out.write(np.where(blk > 250, 0, blk).astype('uint8'), 1,
                              window=rasterio.windows.Window(0, r0, ds.width, hh))
                out.write_colormap(1, pal)
                out.build_overviews([2, 4, 8, 16, 32], Resampling.nearest)
                out.update_tags(ns='rio_overview', resampling='nearest')
        os.replace(tmp, dst)
        with open(os.path.join(DST, '%s.clr' % t), 'w', encoding='utf-8') as f:
            f.write('# value R G B\n')
            for k, c in sorted(COLORS.items()):
                f.write('%d %d %d %d\n' % (k, round(c[0] * 255), round(c[1] * 255), round(c[2] * 255)))
        VC.emit('%s ✅ %.0f MB（调色板+金字塔 nearest）' % (t, os.path.getsize(dst) / 1e6))


# ---------- 7. QA + 对照 ----------
def _stats(fp):
    import rasterio
    cnt = collections.Counter(); npx = 0
    with rasterio.open(fp) as ds:
        res = ds.res
        for r0 in range(0, ds.height, 2048):
            hh = min(2048, ds.height - r0)
            blk = ds.read(1, window=rasterio.windows.Window(0, r0, ds.width, hh))
            lat0 = ds.xy(r0, 0, offset='ul')[1]; lat1 = ds.xy(r0 + hh - 1, 0, offset='ul')[1]
            cell = (res[0] * 111.32 * np.cos(np.deg2rad((lat0 + lat1) / 2))) * (abs(res[1]) * 110.57)
            v, c = np.unique(blk, return_counts=True)
            for a, b in zip(v.tolist(), c.tolist()):
                cnt[a] += b * cell
            npx += hh * ds.width
    return cnt, npx


def qa():
    names = VC.V31_NAMES()
    L = ['# D2 规则（R4：剔除灌丛层）瓦片复测 QA %s' % time.strftime('%Y-%m-%d %H:%M'), '',
         '> 训练池 = 2023 年度子集 ±2° 盒，**剔除 `glc_fcs10_2023_shrub`**，cap 20k；其余配置与试点一致',
         '> 对照 = `试点成果_2023瓦片/`（全池 R0 版）', '']
    for t, meta in TILES.items():
        fp = os.path.join(DST, '%s_10m.tif' % t)
        if not os.path.exists(fp):
            L.append('## %s 未产出' % t); continue
        cnt, npx = _stats(fp)
        tot = sum(cnt.values()); zero = cnt.get(0, 0); cov = (tot - zero) / tot
        valid = sorted(((k, v) for k, v in cnt.items() if k > 0), key=lambda x: -x[1])
        rows = [dict(code=k, name=names.get(str(k), '?'), area_km2=round(v, 2),
                     share_pct=round(100.0 * v / (tot - zero), 3)) for k, v in valid]
        with open(os.path.join(DST, '%s_面积.csv' % t), 'w', newline='', encoding='utf-8-sig') as f:
            wr = csv.DictWriter(f, fieldnames=['code', 'name', 'area_km2', 'share_pct'])
            wr.writeheader(); wr.writerows(rows)
        L += ['## %s（%s）' % (t, meta['note']), '',
              '- 覆盖率 %.2f%% ｜ 类数 %d ｜ 总面积 %.0f km²' % (cov * 100, len(valid), sum(v for _, v in valid))]
        old = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/试点成果_2023瓦片/%s_面积.csv' % t
        if os.path.exists(old):
            od = {int(r['code']): r for r in csv.DictReader(open(old, encoding='utf-8-sig'))}
            L += ['', '| 编码 | 地类 | R0 全池 km² | R4 复测 km² | Δ km² | Δ% |', '|---|---|---|---|---|---|']
            keys = sorted(set([int(k) for k in od.keys()] + [r['code'] for r in rows]))
            new = {r['code']: r for r in rows}
            for k in keys:
                a = float(od[k]['area_km2']) if k in od else 0.0
                b = new[k]['area_km2'] if k in new else 0.0
                if a < 1 and b < 1:
                    continue
                L.append('| %d | %s | %.0f | %.0f | %+.0f | %s |' % (
                    k, names.get(str(k), '?'), a, b, b - a,
                    ('%+.1f%%' % (100 * (b - a) / a)) if a > 0 else '—'))
        info = reg('raster2_%s' % t) or {}
        L.append('')
        L.append('- EECU·h = %s ｜ 资产 = `%s`' % (info.get('eecu_h', '?'), info.get('asset', '?')))
        L.append('')
        VC.emit('%s QA：覆盖率 %.2f%% 类数 %d' % (t, cov * 100, len(valid)))
    fp = os.path.join(VC.REPT, 'D2R4瓦片复测QA_%s.md' % time.strftime('%Y%m%d'))
    open(fp, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    VC.emit('→ %s' % fp)


# ---------- 8. 预览 ----------
def preview():
    import rasterio
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    sys.path.insert(0, os.path.join(VC.ROOT, 'code'))
    from d_tiles import COLORS
    names = VC.V31_NAMES()
    for t, meta in TILES.items():
        fp = os.path.join(DST, '%s_10m.tif' % t)
        if not os.path.exists(fp):
            continue
        cnt, _ = _stats(fp)
        rows = [dict(code=int(k), name=names.get(str(k), '?'), area_km2=v) for k, v in cnt.items() if k > 0]
        with rasterio.open(fp) as ds:
            step = max(1, int(ds.width / 3000))
            arr = ds.read(1, out_shape=(ds.height // step, ds.width // step))
        h, w = arr.shape
        rgb = np.full((h, w, 3), 0.92, dtype='float32')
        for k, c in COLORS.items():
            rgb[arr == k] = c
        fig, ax = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={'width_ratios': [3, 1]})
        ax[0].imshow(rgb, interpolation='nearest')
        ax[0].set_title('%s · %s · 24 类地表覆盖（2023,10m, D2-R4 规则）' % (t, meta['note']), fontsize=14)
        ax[0].set_xticks([]); ax[0].set_yticks([]); ax[1].axis('off')
        txt = [meta['note'], '', '逐类面积 top12（km²）：']
        for r in sorted(rows, key=lambda x: -x['area_km2'])[:12]:
            txt.append('%2d %-6s %8.0f' % (r['code'], r['name'], r['area_km2']))
        txt += ['', '总覆盖 %.0f km²' % sum(r['area_km2'] for r in rows), '', '类数 %d / 24' % len(rows)]
        ax[1].text(0, 1, '\n'.join(txt), va='top', fontsize=11)
        patches = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k2]) for k2 in sorted(COLORS)]
        labels = ['%d %s' % (k2, names.get(str(k2), '?')) for k2 in sorted(COLORS)]
        fig.legend(patches, labels, loc='lower center', ncol=8, fontsize=8, frameon=False)
        plt.subplots_adjust(bottom=0.11); plt.tight_layout()
        png = os.path.join(DST, '%s_预览10m.png' % t)
        plt.savefig(png, dpi=120); plt.close()
        VC.emit('%s 预览 → %s' % (t, png))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'pools'
    {'pools': build_pools, 'upload': upload, 'submit': submit, 'poll': poll,
     'download': download, 'colorize': colorize, 'qa': qa, 'preview': preview}[cmd]()
