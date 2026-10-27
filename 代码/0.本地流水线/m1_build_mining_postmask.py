# -*- coding: utf-8 -*-
"""
m1_build_mining_postmask.py — 构建中国露天采矿/采石场“后分类处理专用”空间走廊与独立验证点

* 核心原则（用户2026-10-27指令）：
  露天采矿/采石场【不参加主分类器训练】，仅在 10m 逐年土地覆盖初版分类后，
  通过“矿业足迹走廊 + 光谱/时序动态检验（当年为裸地/稀疏/不透水 且 NDVI<0.26）”直接后处理修正。
* 输入数据（D:/地物样本数据_2026-10-27/03_露天采矿/）：
  1. Tang & Werner 2023 (31,189 面 / 8,721 km²)
  2. Maus et al. 2022 v2 (8,795 面 / 10,371 km²)
  3. OSM quarry (10,586 面 / 3,204 km²)
  4. CTAmap 2026 省级区划 (Y:/CTAmap/.../T2026_province.shp)
* 输出成果（F:/地理所/.../数据/单类样本的单独处理_2026.10.12/露天采矿_后处理_不参与训练/）：
  1. China_Mining_PostMask_WGS84.gpkg（全量三源标准化图斑，带来源/分级/面积/省归属）
  2. China_Mining_Corridor_500m_WGS84.gpkg（合并外扩 500m 潜在采矿检索走廊，供后处理加速）
  3. 露天采矿_独立验证点_WGS84.gpkg / .csv（~1,200 个高置信独立验证点，仅供论文精度评估）
  4. README_露天采矿_后处理说明.md、统计 JSON 与分省 CSV
"""
import os, sys, time, json
import numpy as np
import geopandas as gpd
import shapely
from shapely import STRtree, Point
from pyproj import Geod

sys.stdout.reconfigure(encoding='utf-8')

PROV_PATH = r'Y:/CTAmap/SCI论文/数据/2026CTAmap 省市县行政区划数据/province/T2026_province.shp'
TANG_PATH = r'D:\地物样本数据_2026-10-27\03_露天采矿\TangWerner2023_Global_Mining_74548_Zenodo7894216\China_subset\TangWerner2023_China_boundary.gpkg'
MAUS_PATH = r'D:\地物样本数据_2026-10-27\03_露天采矿\Maus2022_Global_Mining_Polygons_v2_PANGAEA942325\China_subset\Maus2022_China.gpkg'
OSM_PATH  = r'D:\地物样本数据_2026-10-27\03_露天采矿\OSM_quarry_local\OSM_quarry_China.gpkg'

DEST_DIR = r'F:\地理所\论文\中国土地覆盖数据_2017-2024\数据\单类样本的单独处理_2026.10.12\露天采矿_后处理_不参与训练'
ALIAS_DIR = r'F:\地理所\论文\中国土地覆盖数据_2017-2024\数据\单类样本的单独处理_2026.10.12\露天采矿'

GEOD = Geod(ellps='WGS84')

def emit(msg):
    print(time.strftime('[%Y-%m-%d %H:%M:%S] ') + str(msg), flush=True)

def main():
    emit("=== 开始构建露天采矿后处理母版与独立验证集 ===")
    os.makedirs(DEST_DIR, exist_ok=True)
    os.makedirs(ALIAS_DIR, exist_ok=True)

    # 1. 读取 CTAmap 省界
    emit("1. 读取 CTAmap 2026 省级行政区划...")
    prov_gdf = gpd.read_file(PROV_PATH)
    prov_names = prov_gdf['省'].astype(str).tolist()
    prov_codes = prov_gdf['省级码'].astype(str).tolist()
    prov_geoms = list(prov_gdf.geometry)
    prov_tree = STRtree(prov_geoms)
    emit(f"CTAmap 加载完成，包含 {len(prov_gdf)} 个省级单元。")

    # 2. 读取三源数据
    emit("2. 读取三源原始矢量数据...")
    t0 = time.time()
    g_tang = gpd.read_file(TANG_PATH)
    g_maus = gpd.read_file(MAUS_PATH)
    g_osm  = gpd.read_file(OSM_PATH)
    emit(f"Tang 2023: {len(g_tang)} 面; Maus v2: {len(g_maus)} 面; OSM quarry: {len(g_osm)} 面 (耗时 {time.time()-t0:.1f}s)")

    # 3. 标准化属性与坐标系
    emit("3. 标准化多边形字段与投影面积计算...")
    # 转等积投影计算面积
    g_tang_proj = g_tang.to_crs('EPSG:6933')
    g_maus_proj = g_maus.to_crs('EPSG:6933')
    g_osm_proj  = g_osm.to_crs('EPSG:6933')

    g_tang['area_km2'] = (g_tang_proj.geometry.area / 1e6).round(4)
    g_maus['area_km2'] = (g_maus_proj.geometry.area / 1e6).round(4)
    g_osm['area_km2']  = (g_osm_proj.geometry.area / 1e6).round(4)

    g_tang['src'] = 'tang2023'
    g_tang['orig_id'] = ['TANG_' + str(x) for x in g_tang['OBJECTID']]
    g_tang['name'] = g_tang['Name'].fillna('').astype(str)

    g_maus['src'] = 'maus2022'
    g_maus['orig_id'] = ['MAUS_' + str(i+1) for i in range(len(g_maus))]
    g_maus['name'] = ''

    g_osm['src'] = 'osm_quarry'
    g_osm['orig_id'] = ['OSM_' + str(x) for x in g_osm['osm_id']]
    g_osm['name'] = g_osm['name'].fillna('').astype(str)

    # 4. 空间相交分级 (Tier A: 多源重叠; Tier B: Tang 精细面; Tier C: Maus大面/OSM采石场)
    emit("4. 构建空间索引，判定多源重叠分级 (Tier A / B / C)...")
    tree_tang = STRtree(g_tang.geometry.tolist())
    tree_maus = STRtree(g_maus.geometry.tolist())
    tree_osm  = STRtree(g_osm.geometry.tolist())

    # Tang
    tang_overlap_maus = tree_maus.query(g_tang.geometry, predicate='intersects')
    tang_idx_with_maus = set(tang_overlap_maus[0])
    tang_overlap_osm = tree_osm.query(g_tang.geometry, predicate='intersects')
    tang_idx_with_osm = set(tang_overlap_osm[0])
    tang_tier = []
    for i in range(len(g_tang)):
        if (i in tang_idx_with_maus) or (i in tang_idx_with_osm):
            tang_tier.append('Tier_A_consensus')
        else:
            tang_tier.append('Tier_B_tang_fine')
    g_tang['tier'] = tang_tier

    # Maus
    maus_overlap_tang = tree_tang.query(g_maus.geometry, predicate='intersects')
    maus_idx_with_tang = set(maus_overlap_tang[0])
    maus_overlap_osm = tree_osm.query(g_maus.geometry, predicate='intersects')
    maus_idx_with_osm = set(maus_overlap_osm[0])
    maus_tier = []
    for i in range(len(g_maus)):
        if (i in maus_idx_with_tang) or (i in maus_idx_with_osm):
            maus_tier.append('Tier_A_consensus')
        else:
            maus_tier.append('Tier_C_maus_standalone')
    g_maus['tier'] = maus_tier

    # OSM
    osm_overlap_tang = tree_tang.query(g_osm.geometry, predicate='intersects')
    osm_idx_with_tang = set(osm_overlap_tang[0])
    osm_overlap_maus = tree_maus.query(g_osm.geometry, predicate='intersects')
    osm_idx_with_maus = set(osm_overlap_maus[0])
    osm_tier = []
    for i in range(len(g_osm)):
        if (i in osm_idx_with_tang) or (i in osm_idx_with_maus):
            osm_tier.append('Tier_A_consensus')
        else:
            osm_tier.append('Tier_C_osm_standalone')
    g_osm['tier'] = osm_tier

    emit(f"Tang 分级: Tier A {g_tang.tier.value_counts().to_dict()}")
    emit(f"Maus 分级: Tier A {g_maus.tier.value_counts().to_dict()}")
    emit(f"OSM 分级: Tier A {g_osm.tier.value_counts().to_dict()}")

    # 5. 合并全量图斑
    cols = ['src', 'tier', 'orig_id', 'name', 'area_km2', 'geometry']
    import pandas as pd
    combined_df = pd.concat([g_tang[cols], g_maus[cols], g_osm[cols]], ignore_index=True)
    all_polys = gpd.GeoDataFrame(combined_df, geometry='geometry', crs='EPSG:4326')
    all_polys['poly_id'] = [f"MNP_{i+1:06d}" for i in range(len(all_polys))]

    # 6. 为多边形赋值 CTAmap 省级归属
    emit("5. 为 50,570 个多边形匹配 CTAmap 2026 省级归属...")
    rep_pts = all_polys.geometry.representative_point()
    prov_matches = prov_tree.query(rep_pts, predicate='intersects')
    
    ct_prov = np.full(len(all_polys), '', dtype=object)
    ct_code = np.full(len(all_polys), '', dtype=object)
    ct_km = np.zeros(len(all_polys), dtype=float)

    for pt_idx, pr_idx in zip(prov_matches[0], prov_matches[1]):
        if ct_prov[pt_idx] == '':
            ct_prov[pt_idx] = prov_names[pr_idx]
            ct_code[pt_idx] = prov_codes[pr_idx]

    # 对界外点（海岛/沿海采石场）查找最近省份
    unassigned = np.where(ct_prov == '')[0]
    if len(unassigned) > 0:
        emit(f"发现 {len(unassigned)} 个图斑位于省界外侧，正在按最近距离就近匹配省份...")
        for idx in unassigned:
            pt = rep_pts.iloc[idx]
            nearest_pr_idx = prov_tree.nearest(pt)
            nearest_geom = prov_geoms[nearest_pr_idx]
            ct_prov[idx] = prov_names[nearest_pr_idx]
            ct_code[idx] = prov_codes[nearest_pr_idx]
            # 计算距离
            p1, p2 = shapely.shortest_line(pt, nearest_geom).coords
            _, _, dist_m = GEOD.inv(p1[0], p1[1], p2[0], p2[1])
            ct_km[idx] = round(dist_m / 1000.0, 3)

    all_polys['ct_prov'] = ct_prov
    all_polys['ct_code'] = ct_code
    all_polys['ct_km']   = ct_km

    # 调整字段顺序
    final_cols = ['poly_id', 'src', 'tier', 'orig_id', 'name', 'area_km2', 'ct_prov', 'ct_code', 'ct_km', 'geometry']
    all_polys = all_polys[final_cols]

    # 保存 China_Mining_PostMask_WGS84.gpkg
    out_mask_path = os.path.join(DEST_DIR, 'China_Mining_PostMask_WGS84.gpkg')
    emit(f"6. 导出全国露天采矿后处理母版矢量: {out_mask_path}...")
    all_polys.to_file(out_mask_path, layer='mining_polygons', driver='GPKG')
    emit(f"母版保存成功，共计 {len(all_polys)} 个多边形图斑。")

    # 7. 生成 500m 走廊掩膜 (China_Mining_Corridor_500m_WGS84.gpkg)
    out_corridor_path = os.path.join(DEST_DIR, 'China_Mining_Corridor_500m_WGS84.gpkg')
    emit(f"7. 构建 500m 缓冲空间检索走廊: {out_corridor_path}...")
    # 在等积投影下缓冲 500m
    emit("   在 EPSG:6933 等积投影下计算 500m 缓冲...")
    corridor_geom = all_polys.to_crs('EPSG:6933').geometry.buffer(500)
    emit("   执行多边形融合 (unary_union)...")
    dissolved = shapely.unary_union(corridor_geom.values)
    corridor_gdf = gpd.GeoDataFrame({'corridor_id': ['CORRIDOR_500M'], 'geometry': [dissolved]}, crs='EPSG:6933').to_crs('EPSG:4326')
    # 分解为 MultiPolygon 的单个 Polygon 以便快速空间索引
    corridor_exploded = corridor_gdf.explode(index_parts=False).reset_index(drop=True)
    corridor_exploded['part_id'] = [f"PART_{i+1:05d}" for i in range(len(corridor_exploded))]
    corridor_exploded.to_file(out_corridor_path, layer='mining_corridor_500m', driver='GPKG')
    emit(f"500m 走廊保存成功，共 {len(corridor_exploded)} 个连续区块。")

    # 8. 抽取独立人工/评估验证点 (~1,200点，标记不参与训练)
    emit("8. 抽取高精度独立验证点 (仅用于成果精度评估与混淆矩阵，绝不参与训练)...")
    # 候选：Tier_A_consensus 与 Tier_B_tang_fine 中面积 >= 2 ha (0.02 km2) 的多边形
    high_conf = all_polys[
        (all_polys['tier'].isin(['Tier_A_consensus', 'Tier_B_tang_fine'])) &
        (all_polys['area_km2'] >= 0.02)
    ].copy()
    emit(f"   高置信候选矿斑: {len(high_conf)} 个")

    # 内缩提取内部极点
    high_conf_proj = high_conf.to_crs('EPSG:6933')
    neg_buff = high_conf_proj.geometry.buffer(-20)
    valid_neg = ~neg_buff.is_empty
    cand_geoms = neg_buff[valid_neg].representative_point().to_crs('EPSG:4326')
    high_conf_valid = high_conf[valid_neg].copy()
    high_conf_valid['pt_geom'] = cand_geoms

    # 空间贪心抽稀 >= 1.5 km (保证点与点之间空间独立)
    emit("   执行 1.5 km 空间抽稀去重...")
    cand_pts = list(high_conf_valid['pt_geom'])
    cand_indices = list(high_conf_valid.index)

    selected_idx = []
    sel_points = []
    tree_pts = None

    for i, pt in enumerate(cand_pts):
        if tree_pts is None:
            selected_idx.append(cand_indices[i])
            sel_points.append(pt)
            tree_pts = STRtree(sel_points)
        else:
            # 查 0.015 度（约 1.5 km）内是否有已有选点
            nearby = tree_pts.query(pt.buffer(0.015))
            if len(nearby) == 0:
                selected_idx.append(cand_indices[i])
                sel_points.append(pt)
                tree_pts = STRtree(sel_points)
        if len(selected_idx) >= 1200:
            break

    val_sub = all_polys.loc[selected_idx].copy()
    val_pts = gpd.GeoDataFrame(
        {
            'pid': [f"MN_VAL_{i+1:04d}" for i in range(len(val_sub))],
            'lon': [round(p.x, 6) for p in val_sub.geometry.representative_point()],
            'lat': [round(p.y, 6) for p in val_sub.geometry.representative_point()],
            'cls24': 27,
            'cls_zh': '露天采矿',
            'tier': val_sub['tier'].values,
            'src': val_sub['src'].values,
            'poly_id': val_sub['poly_id'].values,
            'orig_id': val_sub['orig_id'].values,
            'area_km2': val_sub['area_km2'].values,
            'ct_prov': val_sub['ct_prov'].values,
            'ct_code': val_sub['ct_code'].values,
            'ct_km': val_sub['ct_km'].values,
            'usage': 'validation_only_not_training'  # 明确标记
        },
        crs='EPSG:4326',
        geometry=[Point(x, y) for x, y in zip(
            [p.x for p in val_sub.geometry.representative_point()],
            [p.y for p in val_sub.geometry.representative_point()]
        )]
    )

    out_val_gpkg = os.path.join(DEST_DIR, '露天采矿_独立验证点_WGS84.gpkg')
    out_val_csv  = os.path.join(DEST_DIR, '露天采矿_独立验证点_WGS84.csv')
    val_pts.to_file(out_val_gpkg, layer='mining_validation_points', driver='GPKG')
    val_pts.drop(columns='geometry').to_csv(out_val_csv, index=False, encoding='utf-8-sig')
    emit(f"独立验证点已生成: {len(val_pts)} 点 (保存至 {out_val_gpkg} 与 .csv)")

    # 9. 生成分省面积统计表
    prov_summary = all_polys.groupby('ct_prov').agg(
        图斑数=('poly_id', 'count'),
        总面积_km2=('area_km2', 'sum'),
        平均单斑面积_km2=('area_km2', 'mean'),
        最大单斑面积_km2=('area_km2', 'max')
    ).sort_values(by='总面积_km2', ascending=False).reset_index()
    prov_summary.to_csv(os.path.join(DEST_DIR, '露天采矿_分省面积统计.csv'), index=False, encoding='utf-8-sig')

    # 10. 生成总结 JSON 与 README
    stats = {
        'total_polygons': len(all_polys),
        'tang_count': len(g_tang),
        'maus_count': len(g_maus),
        'osm_count': len(g_osm),
        'total_area_km2': round(float(all_polys['area_km2'].sum()), 2),
        'tier_breakdown': all_polys['tier'].value_counts().to_dict(),
        'validation_points_count': len(val_pts),
        'province_count': int(all_polys['ct_prov'].nunique())
    }
    with open(os.path.join(DEST_DIR, '露天采矿_各源数据统计.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    readme_content = f"""# 露天采矿/采石场（后分类处理专用 · 不参与主分类器训练）

> **生成时间**：{time.strftime('%Y-%m-%d')}
> **核心定位**：**本类别不参与 AEF/OlmoEarth 主分类器随机森林/神经网络模型训练！**
> **原因说明**：
> 1. 露天采掘面/排土场与自然裸地、戈壁碎石、城市建筑工地光谱高度同谱，若直接加入主分类器训练，会导致西北戈壁与华北山地大面积将自然裸岩误判为矿区，严重破坏基础地类精度；
> 2. 采用**国际顶刊标准流程（Prior-Constrained Spectral Refinement）**：主分类器保持原有 23/24 类自然地表分类，矿坑在当年自然被分类为 `裸地 (201)`、`稀疏植被 (150)` 或 `不透水面 (190/200)`；
> 3. 在分类后处理阶段，仅对落在本目录矿业空间走廊内、且当年主分类为裸地/稀疏/不透水且满足植被抑制条件的像元，重编码为“露天采矿 (cls24=27)”。

---

## 一、文件清单

1. `China_Mining_PostMask_WGS84.gpkg`：全国多源露天采矿多边形母版（**{len(all_polys):,} 个图斑**，合并 Tang 2023、Maus v2 与 OSM quarry，统一 EPSG:4326，带来源、分级与 CTAmap 2026 省归属）；
2. `China_Mining_Corridor_500m_WGS84.gpkg`：全国矿业足迹外扩 500m 空间检索走廊（后处理时只需在此走廊内进行像素扫描）；
3. `露天采矿_独立验证点_WGS84.gpkg` 与 `.csv`：**{len(val_pts)} 个高置信独立验证点**（`usage = 'validation_only_not_training'`，专供最终产品评估与论文混淆矩阵计算，绝对不用于训练）；
4. `露天采矿_分省面积统计.csv` 与 `露天采矿_各源数据统计.json`：分省与分源统计台账。

---

## 二、后处理分类 Python 执行逻辑（供制图脚本调用）

```python
# 伪代码：在每个 2° 瓦片或单省 10m 分类栅格完成后执行
import rasterio
from rasterio.features import geometry_mask

# 1. 读取当年 10m 初版分类栅格与 NDVI 栅格
# landcover: 10m 像元矩阵; ndvi: 当年夏季中位 NDVI
# 2. 读取 China_Mining_PostMask_WGS84.gpkg 中与当前瓦片相交的多边形
mining_mask = ~geometry_mask(mining_polygons, out_shape=landcover.shape, transform=profile['transform'])

# 3. 严格重编码规则
# 仅当像元落在矿斑内，且当年被判为裸地/稀疏/不透水，且不是茂密植被或深水区时，改标为 27（露天采矿）
is_minable_background = (landcover == 201) | (landcover == 150) | (landcover == 190) | (landcover == 200)
is_active_bare = (ndvi < 260)  # 排除已复垦为茂密草木的区域

update_idx = mining_mask & is_minable_background & is_active_bare
landcover[update_idx] = 27  # 重编码为 露天采矿
```

---

## 三、多边形来源与 Tier 分级

- **总图斑数**：{len(all_polys):,} 个
  - Tang & Werner 2023：{len(g_tang):,} 个（精细部件轮廓）
  - Maus et al. 2022 v2：{len(g_maus):,} 个（矿业权与总包络）
  - OSM quarry：{len(g_osm):,} 个（中小采石场补充）
- **Tier 分级**：
  - `Tier_A_consensus`：多源重叠共识区（高置信）
  - `Tier_B_tang_fine`：Tang 单源精细轮廓
  - `Tier_C_maus_standalone`：Maus 独立大范围矿业用地
  - `Tier_C_osm_standalone`：OSM 独立采石场
"""
    with open(os.path.join(DEST_DIR, 'README_露天采矿_后处理说明.md'), 'w', encoding='utf-8') as f:
        f.write(readme_content)

    # 在 ALIAS_DIR 下放置镜像链接说明
    alias_readme = f"""# 露天采矿 目录说明

本目录的地物类型【不参与主分类器模型训练】，已按照后分类修正标准流程完成母版与独立验证点构建。
所有正式数据产物保存在：
`../露天采矿_后处理_不参与训练/`

- 全国矿区后处理母版: `China_Mining_PostMask_WGS84.gpkg`
- 500m 检索走廊: `China_Mining_Corridor_500m_WGS84.gpkg`
- 独立验证点: `露天采矿_独立验证点_WGS84.gpkg`
详细执行说明见上级同名文件夹下的 `README_露天采矿_后处理说明.md`。
"""
    with open(os.path.join(ALIAS_DIR, 'README_样本文件说明.md'), 'w', encoding='utf-8') as f:
        f.write(alias_readme)

    emit("=== 全部处理完毕，成果已安全落盘！===")

if __name__ == '__main__':
    main()
