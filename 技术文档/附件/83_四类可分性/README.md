# 附件：83 四类森林可分性 · 证据清单

> 对应正文：[`../../83_四类森林可分性总盘点与裁决.md`](../../83_四类森林可分性总盘点与裁决.md)
> 归档时间：2026-10-25。原始工作区：`F:/lc_work/nb_national_2026/opt/`。

## 一、文件

| 本目录文件 | 原始路径 | 内容 |
|---|---|---|
| `o1_result.json` | `opt/o1_result.json` | **OlmoEarth 四类探针全指标**：任务 A/B/C/D/D2/E 的 5 折 F1/AUC、经纬度基线、AEF64 同点对照、`coregion_taskD`、`confound_analysis`（纬度中位 30.4/48.2、最近距离 356 km）、`read_stats`（3.26 s/chunk、700 请求中 467 成功/233 超时跳过） |
| `o1_label_map.csv` | `opt/o1_label_map.csv` | **202 个群系 → 四类**的映射与歧义裁决（2,875 点因无白名单关键词丢弃） |
| `o2_evidence_matrix.csv` | `opt/o2_evidence_matrix.csv` | **69 条证据总表**（E01–E55 转写自已落盘文档/JSON，E56–E69 为本轮实测） |
| `o2_result.json` | `opt/o2_result.json` | **四类严格再分析**：`headline`（各特征组 5 折 F1 + Δ）、`fourclass`（all/coregion/e2e_only）、`pairs`（六对二分类）、`axes_two_class`、`confusion_structure`、`label_ceiling`、`audit_habit_grade`（**doc78 bug 的完整机制与影响**）、`verdict`（A/B/C 分档） |
| `o2_confusion_fourclass.json` | 同左 | 四类 OOF 混淆矩阵 + 混淆对空间结构 + 标签错误检验 |
| `code/o1_olmo_probe.py` | `opt/code/` | OlmoEarth 读取/映射/CV 主脚本 |
| `code/o1b_coregion.py` | `opt/code/` | **剔除地理混杂的 D 任务**（限东北+阿尔泰） |
| `code/o2_evidence.py` | `opt/code/` | 证据盘点 + 四类再分析（六阶段） |

**未归档**：`o1_olmo.parquet`（527 点 × 128 维）、`aef_v2.parquet`、`aef_all_points.parquet`、`needle_broad_veg30_v1.parquet`。

## 二、复核五条最快路径

1. **核"D 是地理假象"**：`o1_result.json` → `confound_analysis`（`median_nearest_cross_dist_km=356`、`DNF_pts_within_100km_of_ENF=11/116`），再对 `coregion_taskD.D_coregion.feat.macro_f1_mean=0.750` 与 `lonlat` 基线 0.782。
2. **核"针/阔迁移判死"**：`o2_result.json` → `label_ceiling`（跨源 0.6426/0.6387、两源标签一致率 0.6356）。
3. **核"混淆主因是标签"**：`o2_result.json` → `confusion_structure` 里 D_N 内 E2C 子集 P(预测 E_N)=0.300（海拔 2,692 m）vs E2E 0.028（968 m）。
4. **核 doc78 bug**：`o2_result.json` → `audit_habit_grade`（含逐码修正前后对照）；本地复算用 `labels/habit_stats.json` → `by_code`。
5. **核 OlmoEarth 读取**：`o1_result.json` → `read_stats`（`read_s_mean` 3.26、`nband_degraded` 0、`fail_by_reason` 仅 `deadline_skip`）。

## 三、口径警告（必读）

1. **不要引用 D 任务的 0.963 AUC**：那是 ENF/DNF 空间不重叠导致的地理假象，共区域检验后 OlmoEarth 输给坐标基线。
2. **不要只用经纬度当基线**：AEF 强编码地形（DEM R²=0.962），只扣经纬度会系统性高估净信号。主基线应为「经纬度 + 地形」。
3. **四类 coregion 子集定义**：1° 格内四类各 ≥15 点 → 76 格 / 16,758 点。全量 95,984 点的数字**不能**直接用于"四类可分"的判断。
4. **OlmoEarth 只适合稀疏点取样**：chunk 级传输（读 1 像素也拉 4 MB），shard 20.48 km，一次读超一个 shard 必超时。
5. **`work_base.gedi_rh98/rh50/cover` 三列全为空**（`m0_prep.py` 里只是占位 NaN），任何引用 GEDI 的现有分析都无效，需重新取数。
