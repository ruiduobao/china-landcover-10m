# 附件：80 Copernicus 森林类型扩样 · 证据清单

> 对应正文：[`../../80_Copernicus森林类型扩样结果.md`](../../80_Copernicus森林类型扩样结果.md)
> 归档时间：2026-10-18。原始工作区：`F:/lc_work/nb_national_2026/cgls/`（本目录是核心证据副本，**不含 626 个 tif 与 3.2 MB 候选 parquet**）。

## 一、文件

| 本目录文件 | 原始路径 | 内容 |
|---|---|---|
| `p1_stats.json` | `cgls/p1_stats.json` | **核心区统计**：nitonal n_needle_core/n_broad_core、needle_km2_geo 261,631.8 / broad_km2_geo 839,898.44、**C1 0.7973**（gated 0.8203 / with_mixed 0.7968）、腐蚀保留率（needle 0.5972 / broad 0.5868 / all 0.5898）、分生态区 |
| `p1_tiles.csv` | `cgls/p1_tiles.csv` | 逐瓦 313 行：100 m 像元构成、needle/broad 核心像元、地理面积、C1 |
| `p2_stats.json` | `cgls/p2_stats.json` | **候选点**：107,892（针 50,818 / 阔 57,074）、去聚团前 142,583、**C4 重叠 0.0635**、参数（per_tile 400 / cap 60000 / jitter 20 m / 0.01°） |
| `p4_cgls_vs_fcs10.json` | `cgls/p4_cgls_vs_fcs10.json` | **C2/C5/C6 全部原始数**：混淆矩阵、chance_corrected（kappa 0.3321 / 平衡准确率 0.6624 / 基线 0.6412）、by_leaf、by_region、by_prov、**south_needle（1.51%，251/265 判 51）**、C5 瓦片清单、C6 分省、6 条 caveats |
| `p4_by_prov.csv` / `p4_by_region.csv` | 同左 | 分省（33 行，16 行 n<30 标注不报率）/ 分生态区（8 组 n≥86）一致率与 CI |
| `p4_tiles.csv` | `cgls/p4_tiles.csv` | 逐瓦核心像元数（P4 视角） |
| `p1_core.py` / `p2_points.py` / `p4_compare.py` | `cgls/code/` | 三个脚本（P1 含渐进式下载 + 本地 3×3 腐蚀 + 幂等 checkpoint） |

**未归档**：`core/*_core.tif`（626 个，约 22 MB）、`needle_broad_cgls_candidates.parquet`（107,892 行，3.2 MB）——均可用 `p1_core.py` + `p2_points.py` 重建（约 70 分钟）。

## 二、复核四条最快路径

1. **核 C1**：`p1_stats.json` → `C1.national=0.7973`（主口径）、`national_gated=0.8203`、`national_with_mixed=0.7968`。
2. **核 C2**：`p4_cgls_vs_fcs10.json` → `overall.rate=0.699458`、`chance_corrected.kappa=0.332109`、`chance_corrected.majority_class_baseline=0.641245`（差值 5.82pp 即由此算得）。
3. **核南方针叶互斥**：同文件 → `south_needle.cgls_needle_vs_fcs10` = `{N:4, B:261, NA:56}`、`cgls_needle_agreement_with_fcs10=0.015094`、`cgls_needle_vs_fcs10_code["51"]=251`。
4. **核 C4/C5**：`p2_stats.json` → `C4_overlap_vs_v2.rate=0.0635`；C5 的 +10 瓦由 `p1_tiles.csv` 的 `n_needle_core>0` 集合减去 v2 的 248 瓦得到。

## 三、已知缺陷与口径警告（必读）

1. **CGLS 年份口径不一致**：`labels/methods/gee_struct_sample.parquet` **未落 `cgls_year` 字段**，年份只能引 `m3_gee_struct.py` docstring 的"2015"；而 P1 用的是 **2017–2019**。这会压低 C2 一致率，但**不改变 FAIL 的方向**（0.6995 距 0.75 差 5.5pp，而口径差异量级不足以补齐）。
2. **尺度结构性上限**：100 m 像元内 60% 阔叶 + 40% 针叶时 CGLS 只能给一个标签，FCS10 的 10 m 点按面积落多数类 → 边缘冲突无法与真实标签噪声分离。
3. **本地 FCS10 栅格与表格标签无法交叉核对**（P4 独立发现）：与 `Z:/nb_national_2026/fcs10_dl/tiles/`、`ext_tiles/fcs10/` 逐点对比，中心像素一致率仅 **0.15–0.66**，偏移扫描**无稳定最优偏移**；v2 抽样点对栅格仅 **20.8%** 精确匹配。此问题**独立于本轮任务**，但会影响任何"栅格 vs 表"核对 → **建议单独立项排查**。
4. **`work_base.parquet` 的 `cgls_ft` 列全为 NaN**（463,135 个）：真实 CGLS 抽样在 `gee_struct_sample.parquet`（20,000 点，非空 15,966），P4 按 `stable_uid` 内连获取。后续脚本不要用 `work_base.cgls_ft`。
