# -*- coding: utf-8 -*-
"""f11_sichuan_v2.py — 四川「样本集 v2」重分类驱动（训练→导出→下载→保真）

* 输入：SRC 资产 projects/fast-drake-508210-f7/assets/prod5p/smp2023_merged（2,060,577 点，含 src）
        瓦片元数据 F:/lc_work/prod5p_2023/plan/tiles_frozen.csv（prov=宁 的 6 瓦）
        成品对照 F:/lc_work/prod5p_2023/rasters_batch/<tile>_10m.tif（R0 交付版）
* 规则源：生产配方（p1_pilot/b1_batch）+ 唯一改动 **R4 全筛**：训练集 .filter(neq('src','glc_fcs10_2023_shrub'))
        其余完全一致：±2° box → randomColumn(seed 7) → limit(20000) → RF(100,leaf2,nodes5000,seed7)
        → classify → uint8 clip → Export.toAsset(scale=10, EPSG:4326, shardSize=16)
* 门槛：保真 3000 点须 100%（unmask(255) 处理掩膜）；导出资产命名 prod5p_r4/tile_<t>_2023（不覆盖交付资产）
* 输出：F:/lc_work/prod5p_2023/rasters_r4/<tile>_10m.tif + qa_r4/<tile>_面积.csv + nx_r4_state.json
* 用法：python n3_nx_r4.py submit [--tiles T1610] ｜ poll ｜ fetch [--tiles T1610] ｜ compare
"""
import csv
import json
import math
import os
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'F:/lc_work/v31_exp/code')
import numpy as np
import v31_common as VC

ROOT = r'F:/lc_work/prod5p_2023'
RDIR = os.path.join(ROOT, 'rasters_v2_sc')          # 新品输出（不动交付目录）
QAD = os.path.join(ROOT, 'qa_v2_sc')
STATE = os.path.join(ROOT, 'sc_v2_state.json')
PROV = '川'
SRC = 'projects/fast-drake-508210-f7/assets/prod5p/smp2023_merged'
YEAR = 2023
POLICY_SRC_EXCLUDE = 'glc_fcs10_2023_shrub'   # v2-① 删 FCS10 灌丛层（判读精度 13.9%/16.5%）
POLICY_CL_EXCLUDE = 180                       # v2-② 删 30 类码 180=木本沼泽（判读精度 6.7%）
POLICY_ECO = True                             # v2-③ 生态硬约束（区域化：落叶针叶<1500m 仅四川盆地）
GEEFAST = r'C:/Users/Administrator/.zcode/skills/geefast-download/scripts/gee_rest_compute_pixels.py'
CAND = ["bx15mw", "bv9yc0", "ccstqqb", "ch2fg2t9", "chengruiduobao", "clvqm4", "e0p36771", "eib2z5bs", "es82gxq", "fcz0iuwf", "gb4286x", "gm9ufoo4", "hj74bml", "hqzub6", "hrbm3pdw", "hte4021n", "ief3nj", "iu5f4z", "j8tv79n", "k3r6ncvd", "kitmyfaceplease2", "n08zt4", "n51j08y", "nama8lg", "neg69g", "nhqz5uj", "nkhn6s9", "nnjxsn", "oh6oiel", "p2vlkaia", "pcrw36d", "pjsxq83j", "ppzynq", "q6hbzjk", "qy7gff", "r45smj3u", "rbj1et5", "rz4afbw", "s4ezbd", "save456jr", "s8xpcl1w", "seqsiu", "sgqv7bf", "sfvmtj0u", "udn4q4", "ughwvm7968.med", "v054r8u", "vh481j", "vidalmvtpronet18", "vlgvl6u", "x2zdn8", "w2qe4hiu", "x8diho2n", "xysongf", "y30b63ye", "ybzljyma", "zhnagningdan1", "zitwwufh", "zixen8v8", "zsi8emo", "zvhvax2j", "zzuoavsm"]
N_FID = 3000
PIDS = VC.jload(r'F:/lc_work/v31_exp/config/m5_pids.json', {})


def pid_of(acct):
    """项目 ID：先 VC.pid_of（老池），失败再查 m5_pids.json（63 账号舰队）。"""
    try:
        return VC.pid_of(acct)
    except Exception:
        if acct in PIDS:
            return PIDS[acct]
        raise


def log(m):
    print('[%s] %s' % (time.strftime('%m-%d %H:%M:%S'), m), flush=True)


def S():
    return VC.jload(STATE, {'tiles': {}})


def save(st):
    st['updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
    VC.jsave(st, STATE)


def tiles_meta(prov='宁'):
    """瓦片元数据：先用五省冻结表，缺的从全国 313 网格（代码/8.全国生产/plan/tiles.csv）补。"""
    m = {}
    fp0 = os.path.join(ROOT, 'plan', 'tiles_frozen.csv')
    if os.path.exists(fp0):
        for r in csv.DictReader(open(fp0, encoding='utf-8-sig')):
            if r['prov'] == prov:
                m[r['tile']] = dict(box=[float(r['x0']), float(r['y0']), float(r['x1']), float(r['y1'])],
                                    land_frac=float(r['land_frac']))
    if prov != '宁':
        nat = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/代码/8.全国生产/plan/tiles.csv'
        for r in csv.DictReader(open(nat, encoding='utf-8-sig')):
            t = r['﻿tile'] if '﻿tile' in r else r['tile']
            m.setdefault(t, dict(box=[float(r['x0']), float(r['y0']), float(r['x1']), float(r['y1'])],
                                 land_frac=float(r['land_frac'])))
    return m


def submit(only=None):
    """提交 R4 训练导出；账号策略：跳过 bad_accts，逐账号尝试，遇"未注册/受限/配额"即拉黑换号。"""
    T = tiles_meta(PROV)
    st = S()
    for i, (t, meta) in enumerate(sorted(T.items())):
        if only and t not in only:
            continue
        rec = st['tiles'].setdefault(t, {})
        if rec.get('task') and rec.get('state') not in ('FAILED', 'CANCELLED'):
            log('%s 已有任务 %s（%s），跳过' % (t, rec['task'], rec['state']))
            continue
        bad = set(st.setdefault('bad_accts', []))
        used = {v.get('acct') for v in st['tiles'].values()
                if v.get('state') in ('SUBMITTED', 'READY', 'RUNNING')}
        done = False
        for acct in [a for a in CAND if a not in bad and a not in used]:
            try:
                ee, pid = VC.ctx(acct)
                try:
                    ee.data.createFolder('projects/%s/assets/prod5p_v2' % pid)
                except Exception:
                    pass
                boxp = ee.Geometry.Rectangle([meta['box'][0] - 2, meta['box'][1] - 2,
                                              meta['box'][2] + 2, meta['box'][3] + 2])
                base = (ee.FeatureCollection(SRC).filterBounds(boxp)
                        .filter(ee.Filter.neq('src', POLICY_SRC_EXCLUDE))      # v2-① 删灌丛层
                        .filter(ee.Filter.neq('cl', POLICY_CL_EXCLUDE))       # v2-② 删木本沼泽
                        .randomColumn('rc', 7).sort('rc').limit(40000))      # 先随机压缩到 4 万（20 万贴高程会 memory limit exceeded）
                # v2-③ 生态硬约束：贴 1 km 高程后按「类×高程」剔除生理不可能组合
                #   geometries=True 关键（False 会让点几何丢失、第二步取样返回 0 —— 2026-10-10 实测）
                #   落叶针叶(<1500 m)、地衣苔藓(<1500 m) 只在**四川盆地**判为不可能（doc55 §11：
                #   该阈值全国套用会误杀东北兴安落叶松——85% 的 7 类点在东北、天然分布 300–1000 m）
                withElev = (ee.Image('USGS/SRTMGL1_003').rename('elev')
                            .sampleRegions(collection=base, scale=1000, geometries=True, tileScale=4))
                basin = ee.Geometry.Rectangle([102.5, 28.5, 108.5, 32.5], None, False)
                CRE = [51, 52]; CDB = [61, 62]; CEN = [71, 72]; CMX = [91, 92]; CDN = [81, 82]
                CGRD = [11, 12, 20]; CSHR = [120]; CICE = [220]; CMOSS = [140]
                viol = [
                    ee.Filter.And(ee.Filter.gt('elev', 3800), ee.Filter.inList('cl', CRE)),
                    ee.Filter.And(ee.Filter.gt('elev', 4500), ee.Filter.inList('cl', CDB)),
                    ee.Filter.And(ee.Filter.gt('elev', 4800),
                                  ee.Filter.inList('cl', CEN + CMX + CDN)),
                    ee.Filter.And(ee.Filter.gt('elev', 4500), ee.Filter.inList('cl', CGRD)),
                    ee.Filter.And(ee.Filter.gt('elev', 5000), ee.Filter.inList('cl', CSHR)),
                    ee.Filter.And(ee.Filter.lt('elev', 2500), ee.Filter.inList('cl', CICE)),
                    ee.Filter.And(ee.Filter.bounds(basin), ee.Filter.lt('elev', 1500),
                                  ee.Filter.inList('cl', CDN)),
                    ee.Filter.And(ee.Filter.bounds(basin), ee.Filter.lt('elev', 1500),
                                  ee.Filter.inList('cl', CMOSS)),
                ]
                ok = withElev.filter(ee.Filter.Not(ee.Filter.Or(*viol)))
                fc = ok.sort('rc').limit(20000)
                n_used = fc.size().getInfo()
                tr = VC.remap_fc_v31(fc)
                clf = (ee.Classifier.smileRandomForest(numberOfTrees=100, minLeafPopulation=2,
                       maxNodes=5000, seed=7).train(tr, 'cl', VC.FEATS))
                box = ee.Geometry.Rectangle(meta['box'])
                aef = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
                       .filterDate('%d-01-01' % YEAR, '%d-01-01' % (YEAR + 1))
                       .filterBounds(box).mosaic().select(VC.FEATS))
                cls = aef.classify(clf).rename('class').uint8().clip(box)
                aid = 'projects/%s/assets/prod5p_v2/tile_%s_%d' % (pid, t, YEAR)
                desc = 'scv2_%s_%s' % (t, time.strftime('%m%d%H%M%S'))
                task = ee.batch.Export.image.toAsset(image=cls, description=desc, assetId=aid,
                                                     scale=10, crs='EPSG:4326', region=meta['box'],
                                                     maxPixels=10 ** 12, shardSize=16)
                task.start()
                rec.update(acct=acct, task=task.id, asset=aid, desc=desc, n_train=int(n_used),
                           submitted=time.strftime('%Y-%m-%d %H:%M:%S'), state='SUBMITTED')
                log('%s 提交 @%s → %s（R4 池取样 %d 点）' % (t, acct, task.id, n_used))
                done = True
                break
            except Exception as e:
                msg = str(e)
                log('  %s 于 %s 提交失败：%s' % (t, acct, msg[:100]))
                if ('not registered' in msg) or ('restricted mode' in msg) or ('quota' in msg.lower()):
                    bad.add(acct)
                    st['bad_accts'] = sorted(bad)
                    log('  → 账号 %s 拉黑（bad_accts=%d）' % (acct, len(bad)))
                continue
        if not done:
            log('%s 无可用账号（bad=%d）' % (t, len(bad)))
    save(st)


def poll(only=None):
    st = S()
    for t, r in sorted(st['tiles'].items()):
        if r.get('state') in ('COMPLETED',):
            continue
        ee, _ = VC.ctx(r['acct'])
        try:
            s = ee.data.getTaskStatus(r['task'])[0]
            r['state'] = s['state']
            r['eecu'] = float(s.get('batch_eecu_usage_seconds', 0) or 0) / 3600.0
            log('%s %s  eecu=%.2f' % (t, s['state'], r['eecu']))
        except Exception as e:
            log('%s 查询失败 %s' % (t, str(e)[:80]))
    save(st)


def asset_grid(acct, asset, box):
    ee, _ = VC.ctx(acct)
    tr = ee.Image(asset).projection().getInfo()['transform']
    sx, sy, tx, ty = tr[0], tr[4], tr[2], tr[5]
    w = int(math.ceil((box[2] - tx) / sx))
    h = int(math.ceil((ty - box[1]) / abs(sy)))
    return '%r,0,%r,0,%r,%r' % (sx, tx, sy, ty), w, h


def fetch(only=None):
    os.makedirs(RDIR, exist_ok=True)
    st = S()
    T = tiles_meta(PROV)
    for t, r in sorted(st['tiles'].items()):
        if only and t not in only:
            continue
        if r.get('state') != 'COMPLETED' or r.get('fidelity', {}).get('rate') == 1.0:
            continue
        meta = T[t]
        raw = os.path.join(RDIR, '_raw_%s.tif' % t)
        env = dict(os.environ)
        env['HOME'] = VC.C.cred_home(r['acct'])
        env['USERPROFILE'] = env['HOME']
        env['EE_PROJECT'] = pid_of(r['acct'])
        aff, gw, gh = asset_grid(r['acct'], r['asset'], meta['box'])
        cmd = [sys.executable, GEEFAST, '--image', r['asset'],
               '--bbox', ','.join(str(x) for x in meta['box']), '--bands', 'class',
               '--scale', '10', '--affine', aff, '--grid-width', str(gw), '--grid-height', str(gh),
               '--auto-tile', '--workers', '20', '--dtype', 'uint16', '--output', raw, '--overwrite']
        log('%s 下载（%dx%d）…' % (t, gw, gh))
        t0 = time.time()
        res = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        ok = res.returncode == 0 and os.path.exists(raw) and os.path.getsize(raw) > 1e6
        log('%s 下载%s（%.1f min，%.0f MB）' % (t, '完成' if ok else '失败', (time.time() - t0) / 60,
                                            os.path.getsize(raw) / 1e6 if os.path.exists(raw) else 0))
        if not ok:
            log('  tail: %s' % ((res.stdout or res.stderr or '')[-300:].replace('\n', ' | ')))
            continue
        # 保真：资产随机 3000 点 vs 本地成品
        rng = np.random.RandomState(7)
        xs = rng.uniform(meta['box'][0] + 0.02, meta['box'][2] - 0.02, N_FID)
        ys = rng.uniform(meta['box'][1] + 0.02, meta['box'][3] - 0.02, N_FID)
        ee, _ = VC.ctx(r['acct'])
        import requests, io
        import pandas as pd
        import rasterio
        feats = [ee.Feature(ee.Geometry.Point([float(x), float(y)]), {'i': i})
                 for i, (x, y) in enumerate(zip(xs.tolist(), ys.tolist()))]
        out = (ee.Image(r['asset']).select('class').unmask(255)
               .sampleRegions(collection=ee.FeatureCollection(feats), properties=['i'],
                              scale=10, tileScale=4))
        url = out.getDownloadURL(filetype='csv', selectors=['i', 'class'])
        txt = requests.get(url, proxies=VC.C.PROXY, timeout=300).text
        df = pd.read_csv(io.StringIO(txt)).set_index('i')['class']
        with rasterio.open(raw) as ds:
            vals = list(ds.sample([(float(x), float(y)) for x, y in zip(xs, ys)]))
        bad = miss = 0
        for i, v in enumerate(vals):
            if i not in df.index:
                miss += 1
                continue
            g, loc = int(df.loc[i]), int(v[0])
            if (g > 250 and loc != 0) or (g <= 250 and loc != g):
                bad += 1
        rate = 1.0 - (bad + miss) / N_FID
        r['fidelity'] = dict(n=N_FID, bad=bad, miss=miss, rate=round(rate, 6))
        log('%s 保真 %.4f%% (bad=%d miss=%d)' % (t, 100 * rate, bad, miss))
        if rate == 1.0:
            # 转交付态：uint8 + nodata=0 + tiled
            dst = os.path.join(RDIR, '%s_10m.tif' % t)
            with rasterio.open(raw) as ds:
                prof = ds.profile.copy()
                prof.update(dtype='uint8', compress='deflate', predictor=2, tiled=True,
                            blockxsize=512, blockysize=512, count=1, nodata=0)
                with rasterio.open(dst + '.tmp.tif', 'w', **prof) as d2:
                    for r0 in range(0, ds.height, 2048):
                        hh = min(2048, ds.height - r0)
                        d2.write(ds.read(1, window=rasterio.windows.Window(0, r0, ds.width, hh)), 1, window=rasterio.windows.Window(0, r0, ds.width, hh))
            os.replace(dst + '.tmp.tif', dst)
            log('%s 成品写出 %s' % (t, dst))
    save(st)


def compare():
    """与交付 R0 成品对比：类构成（面积 km²）+ 中性判读点一致率。"""
    import rasterio
    T = tiles_meta()
    names = VC.V31_NAMES()
    rows = []
    for t, meta in sorted(T.items()):
        fp0 = os.path.join(ROOT, 'rasters_batch', '%s_10m.tif' % t)
        fp4 = os.path.join(RDIR, '%s_10m.tif' % t)
        if not (os.path.exists(fp0) and os.path.exists(fp4)):
            log('%s 缺文件，跳过对比' % t)
            continue
        c0, c4 = {}, {}
        for fp, c in ((fp0, c0), (fp4, c4)):
            with rasterio.open(fp) as ds:
                res = ds.res
                for r0 in range(0, ds.height, 2048):
                    hh = min(2048, ds.height - r0)
                    blk = ds.read(1, window=rasterio.windows.Window(0, r0, ds.width, hh))
                    lat0 = ds.xy(r0, 0, offset='ul')[1]
                    lat1 = ds.xy(r0 + hh - 1, 0, offset='ul')[1]
                    cell = (res[0] * 111.32 * math.cos(math.radians((lat0 + lat1) / 2))) * (abs(res[1]) * 110.57)
                    v, n = np.unique(blk, return_counts=True)
                    for a, b in zip(v.tolist(), n.tolist()):
                        p = c.get(a, 0.0)
                        c[a] = p + b * cell
        tot = sum(v for k, v in c4.items() if k > 0)
        rows.append((t, tot, c0, c4))
    with open(os.path.join(QAD, '构成对比.csv') if os.path.isdir(QAD) else os.path.join(RDIR, '构成对比.csv'),
              'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['tile', 'code', 'name', 'R0_km2', 'R4_km2', 'R0_%', 'R4_%'])
        for t, tot, c0, c4 in rows:
            t0 = sum(v for k, v in c0.items() if k > 0) or 1
            for k in sorted(set(list(c0.keys()) + list(c4.keys()))):
                if k <= 0:
                    continue
                w.writerow([t, k, names.get(str(k), ''), round(c0.get(k, 0), 1), round(c4.get(k, 0), 1),
                            round(100 * c0.get(k, 0) / t0, 3), round(100 * c4.get(k, 0) / tot, 3)])
    log('构成对比 → %s' % (os.path.join(RDIR, '构成对比.csv')))
    # 关键类占比打印
    for t, tot, c0, c4 in rows:
        t0 = sum(v for k, v in c0.items() if k > 0) or 1
        for nm, code in (('落叶灌丛', 10), ('草地', 11), ('草本旱地', 1), ('裸地', 22), ('稀疏植被', 13)):
            log('  %s %-6s R0 %.1f%% → R4 %.1f%%' % (t, nm, 100 * c0.get(code, 0) / t0, 100 * c4.get(code, 0) / tot))


if __name__ == '__main__':
    a = sys.argv[1:]
    cmd = a[0] if a else 'submit'
    if '--prov' in a:
        PROV = a[a.index('--prov') + 1]
    only = set(a[a.index('--tiles') + 1].split(',')) if '--tiles' in a else None
    os.makedirs(RDIR, exist_ok=True)
    os.makedirs(QAD, exist_ok=True)
    {'submit': submit, 'poll': poll, 'fetch': fetch, 'compare': compare}[cmd](only) \
        if cmd != 'compare' else compare()
