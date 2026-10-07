# -*- coding: utf-8 -*-
"""
s8_export.py — r1 步骤7：交付生成器（替代 数据/样本交付）
* 产出（全部由数据实时统计生成，杜绝文档漂移）:
    - cn_samples_r1_pool.gpkg    全量分层池（含 reject/uncovered，供追溯）
    - cn_samples_r1_train.gpkg   训练集
    - by_class/   30×(分布PNG + 类gpkg)，取自训练集
    - by_province/ 34×省级gpkg，取自训练集
    - README_字段与取值说明.md（计数/分布程序化生成）
    - r1_delivery_summary.json
* 大 gpkg 分块 append 写（pyogrio），内存友好。
"""
import os, sys, glob, json, time, shutil
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_conf as C
import s1_geom as G

CHUNK = 1_000_000
# 版本参数：python s8_export.py [r1|r2]（默认 r1）
VERSION = (sys.argv[1] if len(sys.argv) > 1 else 'r1').lower()
# Z: 为 SMB 网络盘，GPKG 随机 I/O + 索引构建极慢（实测 30 分钟仅 25MB）。
# 改为 F 盘本地写完后整体拷回。
STAGE = rf'F:\{VERSION}_stage'

def stage_path(rel):
    os.makedirs(os.path.dirname(os.path.join(STAGE, rel)), exist_ok=True)
    return os.path.join(STAGE, rel)

def copy_tree_to_deliv():
    t0 = time.time()
    os.makedirs(C.DELIV, exist_ok=True)
    for root, _, files in os.walk(STAGE):
        rel = os.path.relpath(root, STAGE)
        dst_dir = C.DELIV if rel == '.' else os.path.join(C.DELIV, rel)
        os.makedirs(dst_dir, exist_ok=True)
        for fn in files:
            src = os.path.join(root, fn)
            dst = os.path.join(dst_dir, fn)
            if fn.endswith('-journal'):
                continue
            shutil.copyfile(src, dst)
            print(f'    拷回 {os.path.relpath(dst, C.DELIV)} ({os.path.getsize(src)//1048576}MB)', flush=True)
    # 清理可能残留的 journal
    for j in glob.glob(os.path.join(C.DELIV, '**', '*-journal'), recursive=True):
        try: os.remove(j)
        except OSError: pass
    print(f'  拷回完成 ({time.time()-t0:.0f}s)', flush=True)

def to_gpkg(df, gpkg_path, layer='samples'):
    import geopandas as gpd
    import shapely
    if os.path.exists(gpkg_path):
        os.remove(gpkg_path)
    first = True
    for i in range(0, len(df), CHUNK):
        d = df.iloc[i:i + CHUNK]
        g = gpd.GeoDataFrame(d.copy(), geometry=shapely.points(d.lon, d.lat), crs=4326)
        g.to_file(gpkg_path, layer=layer, driver='GPKG', mode='w' if first else 'a')
        first = False
        print(f'    gpkg {min(i+CHUNK, len(df)):,}/{len(df):,}', flush=True)

CLASS_COLORS = {
    10: '#ffff64', 11: '#ffaF37', 12: '#aaf0f0', 51: '#006400', 52: '#4c7300',
    61: '#00a000', 62: '#a8c800', 71: '#005000', 72: '#003c00', 81: '#286400',
    82: '#285000', 91: '#a0b432', 92: '#788200', 120: '#966400', 121: '#964b00',
    130: '#ffb432', 140: '#ffdcd2', 150: '#ffebaf', 180: '#00a884', 181: '#73ffdf',
    182: '#9ebb3b', 183: '#828282', 184: '#f57ab6', 185: '#66cdab', 186: '#444f89',
    190: '#c31400', 200: '#ff5e4d', 201: '#fff5d7', 202: '#0046c8', 220: '#ffffff',
}

def plot_boundaries(ax):
    import geopandas as gpd
    names, geoms, _ = G.provinces()
    g = gpd.GeoDataFrame({'name': names}, geometry=list(geoms), crs=4326)
    g.boundary.plot(ax=ax, color='#999999', linewidth=0.3)

def by_class_export(train, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    meta = []
    for c in sorted(train.class_new.unique()):
        d = train[train.class_new == c]
        safe = C.class_name(c)
        gp = os.path.join(outdir, f'class_{c}_{safe}.gpkg')
        to_gpkg(d[['lon', 'lat', 'class_new', 'tier', 'year', 'src', 'src_conf', 'province']],
                gp, layer=f'class_{c}')
        fig, ax = plt.subplots(figsize=(10, 8), dpi=130)
        plot_boundaries(ax)
        ax.scatter(d.lon, d.lat, s=0.15, c=CLASS_COLORS.get(c, '#555555'),
                   rasterized=True, alpha=0.6)
        ax.set_xlim(72.5, 136.5); ax.set_ylim(15.5, 55)
        ax.set_title(f'class {c} {safe}  n={len(d):,}', fontsize=11)
        ax.set_xlabel('lon'); ax.set_ylabel('lat')
        fig.tight_layout()
        png = os.path.join(outdir, f'class_{c}_{safe}.png')
        fig.savefig(png); plt.close(fig)
        meta.append({'class': int(c), 'name': safe, 'n_train': int(len(d)),
                     'png': os.path.basename(png), 'gpkg': os.path.basename(gp)})
        print(f'  class {c} {safe}: {len(d):,}', flush=True)
    return meta

def by_province_export(train, outdir):
    os.makedirs(outdir, exist_ok=True)
    meta = []
    for prov, d in train.groupby('province'):
        gp = os.path.join(outdir, f'{prov}.gpkg')
        to_gpkg(d[['lon', 'lat', 'class_new', 'tier', 'year', 'src', 'src_conf']],
                gp, layer='samples')
        meta.append({'province': prov, 'n': int(len(d))})
        print(f'  {prov}: {len(d):,}', flush=True)
    return meta

def write_readme(pool, train, valid, meta_class, meta_prov):
    byc_pool = pool.class_new.value_counts().sort_index()
    byc_train = train.class_new.value_counts().sort_index()
    byt_train = train.tier.value_counts()
    lines = []
    A = lines.append
    A(f'# 中国区 30 类精细土地覆盖样本库 {VERSION}（字段与取值说明）')
    A('')
    A(f'> 生成时间: {time.strftime("%Y-%m-%d %H:%M")}  |  流水线: 代码/5.样本重建 (s0-s8, 固定种子可复现)')
    A('> 本 README 全部计数由数据实时统计生成，与文件内容严格一致。')
    A('')
    A('## 一、文件清单')
    A('')
    A('| 文件 | 内容 | 点数 |')
    A('|---|---|---|')
    A(f'| cn_samples_{VERSION}_pool.gpkg | 全量分层池（含 reject/uncovered，供追溯与嵌入空间二次清洗） | {len(pool):,} |')
    A(f'| cn_samples_{VERSION}_train.gpkg | 训练集（清洗+抽稀后） | {len(train):,} |')
    A(f'| by_class/ | 30 类分布图 PNG + 类 gpkg（训练集） | - |')
    A(f'| by_province/ | 省级 gpkg（训练集） | - |')
    A(f'| {VERSION}_validation.gpkg | 独立验证池（不入训练） | {len(valid):,} |')
    A('')
    A('## 二、字段说明')
    A('')
    A('| 字段 | 含义 |')
    A('|---|---|')
    A('| lon/lat | 经纬度（EPSG:4326 十进制） |')
    A('| class_new | 30 类产品码（见下表） |')
    A('| class_raw | 源产品原始码（部分源有） |')
    A('| tier | 样本置信层（见第三节） |')
    A('| year | 标签年份（源产品年份） |')
    A('| src | 来源溯源码（见第四节） |')
    A('| src_conf | 来源先验置信 0-1 |')
    A('| agree_n | 五产品投票一致票数 0-5（WC2021/ESRI2020/清华2017/CLCD2020/CN30米2020） |')
    A('| border | 任一 10m 票源 3×3 邻域多样性≥3（边界带） |')
    A('| stab_years | CLCD 2000-2024 同类稳定年数（底座点） |')
    A('| cell_q | 0.25° 配额格键（训练集） |')
    A('| province | 省级行政区（DataV） |')
    A('| urban | GUB 城乡分割标记（1=GUB 域内，仅底座不透水点） |')
    A('')
    A('## 三、tier 置信分层（唯一规则：五产品 level0 一致票 agree_n）')
    A('')
    A('| tier | 规则 | 训练权重建议 | 池内点数 | 训练集点数 |')
    A('|---|---|---|---|---|')
    rule = {'gold': 'agree=5 且非边界带', 'silver': 'agree≥4', 'bronze': 'agree=3',
            'external': '外部源，agree≥3 才入训练', 'uncovered': '有效票≤3（覆盖不足）',
            'reject': 'agree<3（产品矛盾）', 'validation': '独立验证池'}
    w = {'gold': 1.0, 'silver': 0.8, 'bronze': 0.6, 'external': 0.7, 'uncovered': 0, 'reject': 0, 'validation': '-'}
    for t in ['gold', 'silver', 'bronze', 'external', 'uncovered', 'reject', 'validation']:
        A(f"| {t} | {rule[t]} | {w[t]} | {int((pool.tier==t).sum()):,} | {int((train.tier==t).sum()):,} |")
    A('')
    A('## 四、src 来源溯源')
    A('')
    A('| src 前缀 | 来源 | 先验置信 |')
    A('|---|---|---|')
    srct = {
        'v2_glc_fcs30d2020_7yr_stable': ('GLC_FCS30D 2020 七年稳定核（主教师，底座）', '0.95'),
        'glc_fcs10_2023': ('GLC_FCS10 2023 10m 全覆盖重采（第二教师，3×3纯净像元）', '0.85'),
        'esri_nat_2020': ('ESRI 2020（built 经 GUB 拆城乡；青藏耕地降权）', '0.70'),
        'ne_crops': ('东北稻/玉米/大豆专项 2017-2025', '0.90'),
        'worldcereal': ('ESA WorldCereal 参考点', '0.90'),
        'eglc': ('EGLC 全球谐和参考', '0.5-0.8'),
        'gpw_grass_vhr': ('Global Pasture Watch VHR 草地', '0.95'),
        'gwl_fcs30_stable': ('GWL_FCS30 湿地 2000-2022 稳定', '0.90'),
        'gisa_gisd': ('GISA/GISD 不透水（城乡=首城市化年 1971+码 与 2000 比较）', '0.75'),
        'glc12_2020_single': ('GLC_FCS30D 园地(12)单年专项', '0.60'),
        'mountains_lc_validation': ('全球高山 LC 验证（CCI 码）', '0.5-0.75'),
        'yrd_imperv_1985_2020': ('长三角不透水验证', '0.85'),
        'hdlv_xj_2020': ('新疆 HDLV 验证', '0.80'),
        'sdpt_v2_species': ('SDPT V2 种植园树种属性（ever_dec×conifer_br→51/61/71/81，Tree crops→11）', '0.5-0.7'),
        'glc_fcs10_2023_trans': ('GLC_FCS10 2023 过渡带加密（森林-灌草交接 3×3 纯净像元）', '0.85'),
        'global_oilpalm_yop': ('全球油棕 OP-YoP（仅滇南）', '0.75'),
    }
    for k, (d, cf) in srct.items():
        A(f'| {k} | {d} | {cf} |')
    A('')
    A('## 五、30 类体系与样本量（池 / 训练集）')
    A('')
    A('| 码 | 类名 | 池内 | 训练集 |')
    A('|---|---|---|---|')
    for c in sorted(C.CLASSES):
        A(f'| {c} | {C.class_name(c)} | {int(byc_pool.get(c, 0)):,} | {int(byc_train.get(c, 0)):,} |')
    A('')
    A(f'## 六、清洗与质控（{VERSION} 统一定义）')
    A('')
    A('1. 底座=GLC_FCS30D 七年稳定+腐蚀样本网格，国界裁剪+GUB 城乡分割；')
    A('2. 外部源统一映射/国界/互斥（距底座<100m 剔除、源内 250m、跨源 100m 去重）；')
    A('3. 训练集=gold/silver/bronze 全留 + external(agree≥3)；生态硬规则（每类纬度区间+区域规则）；')
    A(f'4. 同类最小间距 {C.MIN_DIST["default"]}m（按类 200-2000m）桶贪心抽稀；0.25°格×类 ≤{C.CELL_QUOTA}；')
    A('5. 验证池独立不入训练；class_new=-1 表示仅组级/二元验证（见 s7 文档）。')
    A('')
    A('## 七、引用与致谢')
    A('')
    A('- GLC_FCS30D: Zhang et al., ESSD 2021（教师产品）')
    A('- GLC_FCS10: Liu et al., ESSD 2025（10m 第二教师）')
    A('- AlphaEarth Foundations: Google DeepMind 2025（嵌入特征，后续分类用）')
    return '\n'.join(lines)

def fill_province(df):
    """补齐省归属（新增源点可能缺 province）"""
    if 'province' not in df.columns:
        df['province'] = '未匹配'
    miss = df['province'].isna() | (df['province'].astype(str) == '未匹配')
    if miss.any():
        df.loc[miss, 'province'] = G.province_of(df.loc[miss, 'lon'].to_numpy(),
                                                 df.loc[miss, 'lat'].to_numpy())
        print(f'  省归属补齐 {int(miss.sum()):,} 点', flush=True)
    return df

def main():
    t0 = time.time()
    if VERSION in ('r2', 'r3', 'r4'):
        pool = pd.read_parquet(os.path.join(C.WORK, f'{VERSION}_pool.parquet'))
        train = pd.read_parquet(os.path.join(C.WORK, f'{VERSION}_train.parquet'))
    else:
        pool = pd.read_parquet(C.R1_POOL)
        train = pd.read_parquet(C.R1_TRAIN)
    valid = pd.read_parquet(C.R1_VALID)
    pool = fill_province(pool)
    train = fill_province(train)
    # 类码整型化（r2 因 concat 变 float，避免文件名/属性出现 10.0）
    for _df in (pool, train, valid):
        if 'class_new' in _df.columns:
            _df['class_new'] = _df['class_new'].fillna(-1).astype(int)
    print(f'[{VERSION}] pool {len(pool):,} / train {len(train):,} / valid {len(valid):,}', flush=True)

    os.makedirs(STAGE, exist_ok=True)
    def pick(df, cols):
        return df[[c for c in cols if c in df.columns]]
    print('1) pool gpkg …', flush=True)
    to_gpkg(pick(pool, ['lon', 'lat', 'class_new', 'tier', 'year', 'src', 'src_conf',
                        'agree_n', 'province']),
            stage_path(f'cn_samples_{VERSION}_pool.gpkg'))
    print('2) train gpkg …', flush=True)
    to_gpkg(pick(train, ['lon', 'lat', 'class_new', 'tier', 'year', 'src', 'src_conf',
                         'agree_n', 'cell_q', 'province']),
            stage_path(f'cn_samples_{VERSION}_train.gpkg'))
    print('3) validation gpkg …', flush=True)
    to_gpkg(pick(valid, ['lon', 'lat', 'class_new', 'class_group', 'src', 'src_conf',
                         'year', 'province']),
            stage_path(f'{VERSION}_validation.gpkg'))
    print('4) by_class …', flush=True)
    meta_c = by_class_export(train, stage_path('by_class'))
    print('5) by_province …', flush=True)
    meta_p = by_province_export(train, stage_path('by_province'))
    print('6) README …', flush=True)
    md = write_readme(pool, train, valid, meta_c, meta_p)
    open(stage_path('README_字段与取值说明.md'), 'w', encoding='utf-8').write(md)

    summary = {'pool': int(len(pool)), 'train': int(len(train)), 'valid': int(len(valid)),
               'by_class_train': {str(k): int(v) for k, v in train.class_new.value_counts().sort_index().items()},
               'by_province_train': {m['province']: m['n'] for m in meta_p},
               'elapsed_min': round((time.time() - t0) / 60, 1)}
    json.dump(summary, open(stage_path(f'{VERSION}_delivery_summary.json'), 'w',
                            encoding='utf-8'), ensure_ascii=False, indent=2, default=int)
    copy_tree_to_deliv()
    print('\n交付完成 →', C.DELIV, f'({time.time()-t0:.0f}s)')

if __name__ == '__main__':
    main()

