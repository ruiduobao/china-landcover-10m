# -*- coding: utf-8 -*-
"""一次性：为数据目录树写 README 索引（2026-09-09 目录整理）"""
import io, os

R = r'Z:/Mywork/论文/BaiduSyncdisk/中国土地覆盖数据'

FILES = {}

FILES[R + '/数据/本地处理/README.md'] = """# 数据/本地处理 —— 流水线中间产物与工作区

> 定位：**脚本的工作区**，所有文件都可由 `代码/` 再生成；对外成品请看 `数据/样本交付/`。
> 总索引见 `数据/README.md`。

## 子目录导览

| 目录 | 是什么 | 状态 |
|---|---|---|
| `全国清洗训练/` | **当前核心**：年度嵌入、年度 QC、八套年度子集、独立评估、稀有类补样 | ✅ 当前权威（见其 README） |
| `样本重建/` | r1→r7 样本重建链的逐阶段产物（s0–s8、r2–r7、r8/r8b/r10） | 当前 = r7（见其 README） |
| `样本底座/` | 旧清洗链版本库 cn_samples_v1→v5（含 summary） | 历史版本，仅追溯 |
| `北京试点/` `北京试点R1/` `北京试点S/` | 早期试点产物（`北京试点S` 为 b1 脚本配置的 out_dir） | 历史，保留 |
| `统计/` `网格/` `计划/` `分类体系/` | 最早 lc01–lc07 本地链的产物（cfg/codemap/tile_grid 等） | 历史参考 |
| `models/` | 合成数据自测用的 RF 模型（synth_rf） | 历史自测 |
| `日志/` | 全部运行日志 | 追溯用 |

## 散文件

- `README与记录.md`：2026-09-02 本地链（lc01–lc07）的工作日志，非目录导览。
- `数据核查结论_缺类修正.md`、`样本提纯v2_运行记录.md`：历史核查记录。

## 规则

1. 中间产物**不要**直接写进论文——晋级成品复制到 `数据/样本交付/`。
2. 新一轮批量产物请带版本/日期后缀，成对写 `*_summary.json`。
3. 删除任何 `*_summary.json` 之前先确认没有脚本在读它。
"""

FILES[R + '/数据/本地处理/全国清洗训练/README.md'] = """# 全国清洗训练 —— r7 年度版核心工作区（当前权威）

> 更新：2026-09-09。上游：`../样本重建/`（样本母库）+ GEE 年度嵌入提取。

## ✅ 当前权威文件（用这些）

| 文件/目录 | 说明 |
|---|---|
| `年度子集_含稀有类/r7_train_{2017..2024}.parquet` | **八套年度训练子集（最终交付母本）**：A00–A63 + class_new + train_weight + qc_status |
| `年度子集/r7_train_{2017..2024}.parquet` | 未合并稀有类的 QC 后子集 |
| `sample_year_qc.parquet` | 逐点×逐年 QC 表（14,272,711 行：四态 keep/downweight/exclude_year/persistent_change） |
| `年度QC_summary.json` | QC 汇总 |
| `评估_P3.4/评估报告_P3.4.json` | 空间独立评估（25km 块留出，OA 0.7943 / macro-F1 0.6077） |
| `错位样本剔除清单.parquet` | **1,959 个省级白名单违规点**（140 沿海错标等），留档审计 |
| `稀有类补样/r7_rare_samples.parquet` | 稀有类补样 2,877 点（已剔错） |
| `chunks_index_r7.parquet` + `emb_parts_r7/` | 年度嵌入提取索引与 4,758 块成品（14.27M point-years） |
| `emb_parts_rare/` | 稀有类补样嵌入（9 块） |

## ⚠️ 使用注意

1. `../样本重建/r7_train.parquet`（母库）仍含 159 个错位点（row_id 对齐不能删行），
   **直接用母库必须先按 `错位样本剔除清单.parquet` 过滤**；年度子集已过滤。
2. `sample_year_qc` 中错位点已标记 `exclude_year/province_misplaced`。

## 历史中间件（勿当成品）

`chunks_index.parquet`、`emb_parts/`（旧单年提取）、`chunks_index_redo*.parquet`（修复重提索引）、
`models_national/`（旧 e3 单年模型）、`e1_run.log`、`patrol_run.log`、`hard_fail_chunks.json`。

## 复现入口

`代码/4.全国清洗训练/`：e1b/e1c（提取）→ e2b（年度QC/子集）→ e7（稀有类合并）→
e6（独立评估）→ e8（GPKG 导出）→ e9（分布图）；剔除：`代码/5.样本重建/r10_purge_misplaced.py`。
"""

FILES[R + '/数据/本地处理/样本重建/README.md'] = """# 样本重建 —— r1→r7 样本库重建链产物

> 更新：2026-09-09。脚本：`代码/5.样本重建/`（s0_conf 为唯一配置源）。

## 链路与现状

```
s1–s3  底座/外部/专题  → r1_*
s4/s3b FCS10/补充源    → r4_*（灌丛替换 r4_fcs10_shrub）
r2–r7  提质清洗        → r7_train.parquet（2,241,785 点，09-08 冻结）
r8/r8b/r9              → 稀有类补样候选（本地栅格，输出在 ../全国清洗训练/稀有类补样/）
r10                    → 错位样本剔除（省级白名单，清单在 ../全国清洗训练/）
s8_export              → 交付生成（旧单年冻结版，已归档至 数据/样本交付/_归档_r7冻结单年版_20260908/）
```

## ✅ 当前权威

- `r7_train.parquet`（母库 2,241,785 点）＋ `r7_train_validity.parquet`（valid_from/to 行号对齐）。
- **警告**：母库仍含 159 个错位点未删行（row_id 对齐），使用前按
  `../全国清洗训练/错位样本剔除清单.parquet` 过滤；年度子集已过滤。
- `r7_pool.parquet`（9,646,715 候选池）、`r1_validation.parquet`（旧验证池，仅追溯）。

## 历史中间层

r1–r6 各 `*_*.parquet + *_summary.json`（逐阶段审计留档）、`base_shards/`（底座分片）、
`r2_backup/`、`r4_shrub_shards/`、`n18_parts/` 相关分片。均只读追溯，不再消费。
"""

FILES[R + '/数据/外部样本源/README.md'] = """# 外部样本源 —— 只读的外部原始数据

> 规则：**只读**。不修改、不重命名、不删除；处理后的成品在 `样本重建/` 或 `全国清洗训练/`。
> 稀有类专题源逐个说明见 `rare_class/_SOURCES.md`。

## 主要来源

| 内容 | 说明 |
|---|---|
| `GLC_FCS10_UserGuides.pdf`、`glc_fcs10_2023/`、`eglc_*` | GLC_FCS10 2023 30m 全球（91/140/183/184 补样教师源） |
| `GALF_*.zip`、`grass_gpw_china.parquet`、`gpw_grassland_harm_point.gpkg` | 全球草地（GPW/GALF） |
| `WorldCereal_ReferenceData_20*_*.zip`、`worldcereal_china.parquet` | ESA WorldCereal 参考样本 |
| `esri_national_yearly.parquet` | ESRI 年度土地覆盖（全国投票产品之一） |
| `ne_china_crops.7z`、`ne_crops_samples*` | 东北作物（水稻/玉米/大豆，2017–2025） |
| `yrd_impervious/`、`Impervious surfaces ... .rar` | 长三角不透水面验证 |
| `HDLV_XJ_ValidationDataset.xls`、`hdlv_xj_validation.parquet` | 新疆 HDLV 验证集 |
| `amur_*` | Amur 区样本 |
| `mountains_lc/` | 山地土地覆盖 |
| `rare_class/` | **稀有类专题源**（苹果/柑橘/茶园/橡胶/湿地/红树林 GMW/CW/盐沼/潮滩等，12 个） |
| `gee_thematic/`、`*_run.log` | 早期 GEE 专题采样运行记录 |

## 注意

- `*.zip/*.rar/*.7z` 为原始压缩包，与其解压目录并存；空间紧张时可压缩归档，但**不要删除**（复现需要）。
- `drive-download-*.zip` 为网盘手工下载残留，内容已入库，仅存档。
"""

FILES[R + '/数据/样本数据/README.md'] = """# 样本数据 —— 最早的网格样本底料（只读）

| 内容 | 说明 |
|---|---|
| `GLC_Samples_Grid_2x2/` | GLC 2°×2° 网格样本瓦片（r1 底座的原始输入，`s0_conf.GRID_DIR` 指向这里） |
| `GLC_Samples_Grid_2x2_gpkg/` | 同上内容的 GPKG 版 |
| `其他/GLC_Samples_Grid_2x2.rar` | 原始压缩包存档 |
| `其他/基于7年的FCS生成/` | 基于 FCS30D 七年稳定样本的生成资料 |

规则：只读；后续清洗链产物见 `本地处理/样本底座/` 与 `本地处理/样本重建/`。
"""

FILES[R + '/数据/边界/README.md'] = """# 边界 —— 全项目唯一权威边界

- `china_100000_full.json`：DataV 省级边界（GeoJSON）。国界裁剪、省级统计、生态规则判定
  全部经 `代码/5.样本重建/s1_geom.py` 引用此文件，**不要另建边界副本**。
- 投影约定：面积/块留出计算用 Albers（`+proj=aea +lat_1=25 +lat_2=47 +lat_0=36 +lon_0=104`）。
"""

FILES[R + '/数据/备份/README.md'] = """# 备份 —— 冻结基线（不可修改）

| 目录 | 内容 |
|---|---|
| `r7_评审基线_20260909/` | r7 单年冻结版基线：`r7_pool/r7_train/r1_validation/r7_fcs10_shrub.parquet` + `r7_audit.json` + `r7_delivery_summary.json`。评审对照与复现基准，**禁止改动**；后续所有升级以此为对照。 |
"""

FILES[R + '/_归档/README.md'] = """# _归档 —— 非数据杂物

- `网页抓取与临时_20260909/`：项目根目录散落的网页快照（*.html）、检索结果（gh*.json/grep1.json）、
  临时脚本（servir.py、sepal_cls.js）、表格（t02.xlsx）。与数据生产无关，仅存档；确认无用可整目录删除。
- 数据版本归档不在本目录，见 `数据/样本交付/_归档_*/` 与 `数据/备份/`。
"""

FILES[R + '/数据/样本交付/_归档_r7冻结单年版_20260908/README.md'] = """# r7 冻结单年版交付（2026-09-08 生成，已归档）

由 `代码/5.样本重建/s8_export.py` 从 r7 冻结母库（单锚年 2020 口径）生成，**已被 r7 年度版取代**：

- `cn_samples_r7_train.gpkg`（399 MB）／`cn_samples_r7_pool.gpkg`（1.7 GB）／`r7_validation.gpkg`
- `r7_delivery_summary.json`：当时审计汇总（train 2,241,785 / pool 9,646,715）
- `by_class/`（30×PNG+GPKG）、`by_province/`（34 省级 GPKG）

当前交付见 `../gpkg/`（22 个年度 GPKG）与 `../分布图/`。对照基线在 `数据/备份/r7_评审基线_20260909/`。
"""


def main():
    for path, text in FILES.items():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.isfile(path):
            print('SKIP(已存在)', os.path.relpath(path, R))
            continue
        io.open(path, 'w', encoding='utf-8', newline='').write(text)
        print('WRITE ', os.path.relpath(path, R))


if __name__ == '__main__':
    main()
