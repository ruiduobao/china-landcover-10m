# -*- coding: utf-8 -*-
"""
v7_e2e_local.py — 端到端验证（本地全量 v-final 模型）：嵌入下载 → 训练 → 分类图 → 逐类面积
为什么本地做：GEE 的 decisionTreeEnsemble 需把整片森林内联进请求（远超 48MB 上限），
而 GEE 内 smileRandomForest 没有类别权重、只能靠过采样替代 → 实测把 52 类过预测到 72% 面积。
所以端到端验证改为：**用真正的 v-final 混合模型在本地分类**，嵌入按 0.05° 分块交互下载
（GEE 单次 getDownloadURL 硬上限 48MB，实测 0.10°@30m 即 80MB 超限）。
窗口选址依据母库类密度：
  W1 (119.55,28.40) 浙南/闽北 常绿阔叶林（52 主场）
  W2 (128.85,47.40) 小兴安岭 针阔混交林（91 密度全国最高）
输出: Z:/.../数据/本地处理/全国清洗训练/端到端验证/{win}_class.tif, {win}_map.png,
      {win}_area.csv, W{1,2}_emb2022.tif
用法: python v7_e2e_local.py [--acct y30b63ye] [--proj quick-cache-508211-s9]
"""
import os, sys, io, json, time, argparse
import numpy as np
import pandas as pd
import requests

PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
KB = r'Z:/地理所/论文/中国土地覆盖数据_2017-2024'
sys.path.insert(0, os.path.join(PROJ, '代码', '0.本地流水线'))
sys.path.insert(0, os.path.join(PROJ, '代码', '4.全国清洗训练'))
from lc_conf import CLASSES
import e6_spatial_eval as E6
from sklearn.ensemble import RandomForestClassifier
import joblib

WORK = r'F:/lc_work'
OUTD = os.path.join(KB, '数据/本地处理/全国清洗训练/端到端验证')
PROXY = {'http': 'socks5h://127.0.0.1:7890', 'https': 'socks5h://127.0.0.1:7890'}
CRED = r'F:/地理所/论文/博士论文/中期/实验区域/02_中间数据/gee_accounts'
FEATS = [f'A{i:02d}' for i in range(64)]
AEF = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
YEAR = 2022
SCALE = 30
TILE = 0.075   # 实测 0.10°@30m = 80MB 超 48MB 上限；0.075° ≈ 8MB 安全
WINDOWS = {'W1_浙闽常绿阔叶': (119.50, 28.30, 119.80, 28.60),
           'W2_小兴安岭针阔混交': (128.80, 47.35, 129.10, 47.65)}
WEAK = [52, 62, 91, 140, 11]
WEAK_DEPTH = 20
TREES, WEAK_TREES = 150, 120
HOST = {52: [51, 61, 71, 72, 120, 121, 62, 92], 62: [61, 51, 71, 82, 121, 52, 120],
        91: [71, 61, 92, 82, 51, 72, 81], 140: [150, 201, 121, 130, 120, 220],
        11: [10, 12, 120, 121, 130, 71, 51]}
PALETTE = ['#1b9e77', '#66a61e', '#e6ab02', '#a6761d', '#d95f02', '#7570b3', '#e7298a',
           '#1f78b4', '#33a02c', '#b15928', '#006400', '#7fc97f', '#41ab5d', '#00441b',
           '#78c679', '#c7e9c0', '#ffffcc', '#fdd49e', '#d9d9d9', '#969696', '#525252',
           '#bdbdbd', '#fdbf6f', '#a63603', '#ffeda0', '#c6dbef', '#3182bd', '#08519c',
           '#d73027', '#4575b4', '#8c6bb1']


def emit(m):
    print(time.strftime('[%H:%M:%S] ') + str(m), flush=True)


def train_model():
    fp = os.path.join(WORK, 'vfinal_model_2022.joblib')
    if os.path.isfile(fp):
        d = joblib.load(fp)
        emit('复用已训练模型')
        return d['main'], d['weak'], d
    src = os.path.join(E6.SUB_DIR, f'r7_train_{YEAR}.parquet')
    emit('训练 v-final 模型（%s）…' % os.path.basename(src))
    df = pd.read_parquet(src, columns=['row_id', 'class_new', 'train_weight'] + FEATS)
    X = df[FEATS].to_numpy(np.float32)
    y = df.class_new.to_numpy(int)
    w = np.nan_to_num(df.train_weight.to_numpy(np.float32), nan=1.0, posinf=1.0, neginf=1.0)
    fin = np.isfinite(X).all(1)
    X, y, w = X[fin], y[fin], w[fin]
    t0 = time.time()
    main = RandomForestClassifier(n_estimators=TREES, n_jobs=13, random_state=42,
                                  min_samples_leaf=2, max_features='sqrt',
                                  class_weight='balanced_subsample').fit(X, y, sample_weight=w)
    weak = RandomForestClassifier(n_estimators=WEAK_TREES, n_jobs=13, random_state=77,
                                  max_depth=WEAK_DEPTH, min_samples_leaf=2,
                                  max_features='sqrt',
                                  class_weight='balanced_subsample').fit(X, y, sample_weight=w)
    emit('训练完成 %.0fs（主 %d 树全深度 + 弱类 %d 树 depth=%d）'
         % (time.time() - t0, TREES, WEAK_TREES, WEAK_DEPTH))
    obj = {'main': main, 'weak': weak, 'weak_classes': WEAK, 'host': HOST,
           'spec': 'v-final 2026-09-14'}
    joblib.dump(obj, fp)
    return main, weak, obj


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


def fetch_window(ee, name, box):
    """按 0.05° 分块下载嵌入（GEE 单次 48MB 上限），本地拼成整窗 GeoTIFF"""
    import rasterio
    from rasterio.merge import merge as rmerge
    x0, y0, x1, y1 = box
    d = os.path.join(WORK, 'e2e', name)
    os.makedirs(d, exist_ok=True)
    nx = int(round((x1 - x0) / TILE))
    ny = int(round((y1 - y0) / TILE))
    parts = []
    for a in range(nx):
        for b in range(ny):
            bx0 = x0 + a * TILE; by0 = y0 + b * TILE
            fp = os.path.join(d, f't{a}{b}.tif')
            parts.append(fp)
            if os.path.isfile(fp) and os.path.getsize(fp) > 10000:
                continue
            reg = ee.Geometry.Rectangle([bx0, by0, bx0 + TILE, by0 + TILE])
            emb = (ee.ImageCollection(AEF).filterDate(f'{YEAR}-01-01', f'{YEAR+1}-01-01')
                   .filterBounds(reg).mosaic().select(FEATS))
            url = emb.getDownloadURL(dict(region=reg, scale=SCALE, crs='EPSG:4326',
                                          format='GEO_TIFF'))
            ok = False
            for att in range(4):
                try:
                    r = requests.get(url, proxies=PROXY, timeout=1800)
                    r.raise_for_status()
                    open(fp, 'wb').write(r.content)
                    emit('  块 %d%d %.1f MB' % (a, b, len(r.content) / 1e6))
                    ok = True
                    break
                except Exception as x_:
                    emit('  块 %d%d 重试 %d: %s' % (a, b, att + 1, str(x_)[:70]))
                    time.sleep(10 + 15 * att)
            if not ok:
                raise RuntimeError('块 %d%d 四次失败' % (a, b))
            time.sleep(1)
    srcs = [rasterio.open(p) for p in parts]
    arr, tr = rmerge(srcs)
    prof = srcs[0].profile.copy()
    prof.update(height=arr.shape[1], width=arr.shape[2], transform=tr, count=arr.shape[0],
                dtype=arr.dtype, compress='lzw')
    out = os.path.join(OUTD, f'{name}_emb{YEAR}.tif')
    with rasterio.open(out, 'w', **prof) as dst:
        dst.write(arr)
    for s in srcs:
        s.close()
    emit('%s 拼图 %s 形状 %s' % (name, os.path.basename(out), arr.shape))
    return out, arr, tr, prof


def classify_and_report(name, arr, tr, prof, main, weak):
    import rasterio
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    nb, hh, ww = arr.shape
    X = np.moveaxis(arr, 0, -1).reshape(-1, nb).astype(np.float32)
    ok = np.isfinite(X).all(1)
    pred = np.zeros(len(X), np.int16)
    if ok.any():
        pk = main.predict(X[ok])
        pw = weak.predict(X[ok])
        for c in WEAK:
            sel = (pw == c) & np.isin(pk, HOST[c])
            pk[sel] = c
        pred[ok] = pk
    lab = pred.reshape(hh, ww)
    op = os.path.join(OUTD, f'{name}_class.tif')
    pr = prof.copy(); pr.update(count=1, dtype='int16')
    with rasterio.open(op, 'w', **pr) as dst:
        dst.write(lab.astype(np.int16), 1)
        dst.set_band_description(1, 'v-final class code')
    # 面积（像元面积按投影纬度修正；EPSG:4326 下 1 像元 ≈ (0.0002695°)^2）
    lat_c = (tr.f + hh / 2 * tr.e)
    px_km2 = (SCALE / 1000.0) ** 2
    vc = pd.Series(lab[ok.reshape(hh, ww)]).value_counts().sort_index()
    rows = [{'class': int(c), 'name': CLASSES[int(c)][1], 'n_px': int(v),
             'area_km2': round(int(v) * px_km2, 3)} for c, v in vc.items() if c > 0]
    area = pd.DataFrame(rows).sort_values('area_km2', ascending=False)
    tot = area.area_km2.sum()
    area['pct'] = (area.area_km2 / tot * 100).round(3)
    area.to_csv(os.path.join(OUTD, f'{name}_area.csv'), index=False, encoding='utf-8-sig')
    nz = len(area)
    weak_line = ', '.join(f'{c}={(area[area["class"]==c].area_km2.sum()):.2f}km²'
                          for c in WEAK)
    emit(f'{name}: 窗口 {tot:.1f} km²  非零类 {nz}/30  弱类 {weak_line}')
    # 出图
    cls_list = sorted(set(int(c) for c in area['class']))
    fwd = [0] * (max(cls_list) + 1)
    for i, c in enumerate(cls_list):
        fwd[c] = i
    idx = np.zeros_like(lab, dtype=np.int16)
    m = lab > 0
    idx[m] = np.array(fwd, dtype=np.int16)[lab[m]]
    cmap = ListedColormap(PALETTE[:len(cls_list)])
    norm = BoundaryNorm(np.arange(-0.5, len(cls_list) + 0.5), cmap.N)
    fig, ax = plt.subplots(figsize=(9, 8.4), dpi=170)
    ax.imshow(np.ma.masked_where(lab == 0, idx), cmap=cmap, norm=norm,
              extent=[tr.c, tr.c + ww * tr.a, tr.f + hh * tr.e, tr.f], interpolation='nearest')
    ax.set_title(f'v-final 分类图 · {name} · {YEAR} · {SCALE} m', fontsize=12)
    ax.set_xlabel('lon'); ax.set_ylabel('lat')
    handles = [plt.Rectangle((0, 0), 1, 1, color=PALETTE[i]) for i in range(len(cls_list))]
    ax.legend(handles, [f'{c} {CLASSES[int(c)][1]}' for c in cls_list],
              loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    plt.tight_layout()
    png = os.path.join(OUTD, f'{name}_map.png')
    plt.savefig(png, bbox_inches='tight')
    plt.close()
    emit(f'  分类图 → {png}')
    return {'window_km2': round(float(tot), 2), 'n_nonzero': nz,
            'weak_area_km2': {str(c): float(area[area['class'] == c].area_km2.sum())
                              for c in WEAK},
            'area_table': rows, 'tif': op, 'png': png}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acct', default='y30b63ye')
    ap.add_argument('--proj', default='quick-cache-508211-s9')
    a = ap.parse_args()
    os.makedirs(OUTD, exist_ok=True)
    main_rf, weak_rf, _ = train_model()
    ee = bring_up(a.acct, a.proj)
    summary = {'time': time.strftime('%Y-%m-%d %H:%M'), 'year': YEAR, 'scale_m': SCALE,
               'model': 'v-final hybrid (main depth=None + weak depth=20, host-restricted)',
               'windows': {}}
    for name, box in WINDOWS.items():
        emit('== %s %s ==' % (name, box))
        _, arr, tr, prof = fetch_window(ee, name, box)
        summary['windows'][name] = classify_and_report(name, arr, tr, prof, main_rf, weak_rf)
    json.dump(summary, open(os.path.join(OUTD, 'e2e_local_summary.json'), 'w',
                            encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
    emit('输出目录: %s' % OUTD)


if __name__ == '__main__':
    main()
