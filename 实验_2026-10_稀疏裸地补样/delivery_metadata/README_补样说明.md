# 裸地 / 稀疏植被 东北·东部补样（池 v3 候选层，2023）

> 2026-10-25 ｜ 依据 `F:/lc_work/bare_sparse_v3/spec/预注册判据.md`（阈值先写死后跑）
> 触发：2026-10-14 诊断 `F:/lc_work/v31_exp/reports/bare_sparse_east_gap_2026-10-14.md`（东北稀疏植被缺口）

## 内容
- `稀疏植被_2023_WGS84.gpkg`：**2,457** 点（cls24=13，raw∈150/152/153）
- `裸地_2023_WGS84.gpkg`：**2,103** 点（cls24=22，raw∈200/201/202）
- `bare_sparse_v3_2023.parquet`：同上长表（含 AEF64 A00–A63、证据列、省界字段）
- `shp/`：同名 Shapefile（ASCII 字段名，浮点保留）
- `分省统计.csv`

## 来源与规则
- 源：**GLC_FCS10 2023 10 m**（E 盘本地瓦片）3×3 全纯 + 9×9 同类占比 ≥0.70；
  0.025° 格配额（稀疏 ≤2 / 裸地 ≤1）、最小间距（300/200 m）、与既有 2023 同类样本 250 m 去重。
- 分窗配额：北区（科尔沁/松嫩/呼伦贝尔/浑善达克/黑吉辽其余）+ 东区（华北/黄淮山东/华东华南/黄土高原东）。
- 独立证据（GEE 2023）：S2 夏 NDVI/NDWI、冬 NDVI；Dynamic World 概率；WorldCover；GSW；SRTM。
  keep 判据沿用 doc67 v1.1 两类口径 + 窗口加固（详见 spec §3）。
- 独立核验：栅格回读 300 点（raw+3×3 纯）、证据双路 200 点（独立构建，max|Δ| 须 0）、AEF 完备性。

## 使用
- 训练池 v3：`src='fcs10_2023_ne_east_topup'`，可直接按 src 并入/回退；**不要**改动 v2 池。
- 注：E 盘 FCS10 的 130–135°E 带已于 2026-10-25 补齐（Zenodo MD5 校验），本层已含该带裸地补样（稀疏植被该带仅约 9 km²，未采）。

## 字段
`pid, lon, lat, cls24(13/22), cls_zh, tier, src, fam, raw(FCS10 原始码), share9(9×9 同类占比), ndvi, bare/water/built/snow_and_ice/crops/shrub(DW 概率), wc(WorldCover 码), occurrence(GSW), ct_prov/ct_code/ct_km(CTAmap 省界), year, A00–A63(AEF64)`
