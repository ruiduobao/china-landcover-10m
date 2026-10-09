# 中国 2017–2024 年 10 m 精细土地覆盖数据

**China 10 m Annual Land Cover (2017–2024)** — 基于 Google AlphaEarth Foundations 年度卫星嵌入（64 维 / 10 m）与 Google Earth Engine 服务端随机森林，生产中国逐年 10 m 土地覆盖（主产品 **24 类**，另出 9 大类归并图），对标刘良云 GLC_FCS10（ESSD 2025，自报 OA 83.16%）。

> **English abstract**: This repository hosts the code, documentation and experiment records of a project producing annual 10 m land-cover maps for China (2017–2024), built on Google AlphaEarth Foundations embeddings (64-D, 10 m) with server-side Random Forest on Google Earth Engine. The main product has 24 classes (plus a 9-macro-class version), benchmarked against GLC_FCS10. All production configuration is frozen (v3.1-R1); a 2023 five-province pilot (Sichuan/Heilongjiang/Guangdong/Ningxia/Fujian, 76 tiles) has been delivered. Large data (sample library, rasters) are **not** included in this repository.

---

## 当前状态（2026-10-08）

| 里程碑 | 状态 |
|---|---|
| 生产配置 | **冻结 v3.1-R1**：24 类 + 局部单 RF（±2° 邻域借样、100 树/minLeaf2/maxNodes5000/2 万点）+ 逐年匹配样本 |
| 2023 五省先行批 | ✅ **76/76 瓦交付**（川/黑/粤/宁/闽，10 m，含调色板/金字塔/面积表/QA；1,328.4/1,600 EECU·h） |
| 跨产品对比（E14） | ✅ 与 GLC_FCS10 / GLC_FCS30D / ESRI LULC / ESA WorldCover 逐像元对比（组级一致率 46–85%） |
| 独立验证 | M3 布点完成（3,210 通用点 + 863 灌丛仲裁点 + 403 三江来源点）；**M3-B 已完成 AI 第一遍预判读**（A 臂精度 0.107 → 按预注册判据指向"筛除 FCS10 灌丛层"方向，待人工复核裁决） |
| 全国放量 | 313 瓦片 × 8 年待放量（预算 ≈4.3 万 EECU·h 量级） |

## 方法概要

```
AlphaEarth Foundations 年度嵌入（GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL，64 维，10 m）
        │  逐年 tile 级 ±2° 邻域借样（≈2 万点/瓦片，年度匹配子集）
        ▼
GEE 服务端 smileRandomForest（100 树 / minLeaf=2 / maxNodes=5000 / seed=7）
        ▼
逐瓦 10 m 分类 → uint8 + 调色板 + 金字塔（nearest）+ nodata=0
        ▼
交付：GeoTIFF + 面积表（10 m 原生网格本地统计）+ 概览页（PIL 全平面验收）
```

评估纪律：一律使用 25 km 块留出的 `validation_pool_v2`，并按标签来源分层报告（FCS10 系 / 非 FCS10）。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `代码/0.本地流水线/` | **配置源** `lc_conf.py`（类码表 / 路径），全部脚本以此为唯一配置 |
| `代码/4.全国清洗训练/` | 清洗训练主链 `e0–e10`（含样本质检门 `e10_sample_qa.py`、空间留出评估 `e6_spatial_eval.py`） |
| `代码/8.全国生产/` | **全国生产链路** `p0–p10`（部署 / 分区 / 提交 / 监控 / 下载 / 拼接）+ 账号 worker 模板 |
| `代码/6.下一代补样/`、`代码/7.vfinal/` | 稀有类补样链路；v-final 规格、可分性诊断与冻结 |
| `技术文档/` | **48 篇技术文档**（索引见 `技术文档/00_文档索引.md`；`39` 交接总览、`38` 生产执行依据、`44` 跨产品对比、`46` D0 材料与体检、`47` M3-B AI 预判读） |
| `实验_2026-09_生产前验证/` | 生产前验证实验的报告 / 代码 / 指标存档副本 |
| `AGENTS.md` / `handoff.md` / `VERSION.md` | 面向接手者的硬约束与门禁命令 / 会话交接快照 / 版本沿革（本项目以文档+VERSION 管理变更） |
| `_tools/`、`_复原记录/` | 一次性诊断脚本；2026-09-11 事故后的复原记录 |

> ⚠️ **数据不入库**：样本母库（2,252,207 点）、嵌入式年度子集、10 m 栅格成品、试点 GeoTIFF 等大数据均不在本仓库（体积原因）。代码与文档足以复现方法；成品数据可另行联系。

## 快速开始（需要 Google Earth Engine 账号）

```bash
# 账号探活 + 项目级健康检查（零 EECU，用任何账号前先跑；脚本 s0_sweep/s0b_health 在实验工作区，未随库发布）
python <工作区>/s0_sweep.py --force && python <工作区>/s0b_health.py

# 环境自检（账号/代理/磁盘/依赖）
python 代码/8.全国生产/p0_doctor.py

# 样本质检门（任何样本变更后必跑）
python 代码/4.全国清洗训练/e10_sample_qa.py

# 全国生产链路（示例：部署模型 → 分区 → 生成 worker → 提交 → 监控 → 下载 → 拼接）
python 代码/8.全国生产/p1_deploy_model.py --year 2023
python 代码/8.全国生产/p2_plan_tiles.py
python 代码/8.全国生产/p3_gen_workers.py
python 代码/8.全国生产/fleet/pw_<acct>.py --submit --max 1
python 代码/8.全国生产/p4_watch.py
python 代码/8.全国生产/p5_fetch.py --acct <acct> --tile T0123 --year 2023
python 代码/8.全国生产/p6_mosaic.py --year 2023
```

## 关键事实（复现必读）

1. **类别栅格的一切派生/概览/统计必须显式最近邻**，或直接在原生 10 m 网格本地统计（非最近邻重采样会凭空造类）。
2. **评估一律用 25 km 块留出的 `validation_pool_v2`**，并按来源分层（FCS10 系 vs 非 FCS10）——全点 OA 会被评估点来源机械主导。
3. **大图下载走 `getPixels` 并发器**并逐瓦对齐 GEE 导出网格（`projection()` → affine），不要用分块 `getDownloadURL`。
4. **方法学纪律**：实验先写死判据再跑；小样本下单次种子噪声可达 ±1.5 pp，<2 pp 的单次增益不采信。
5. **训练池来源决策逐年预注册**；任何池变更走服务端 `src` 筛除重训，样本母库本体只读。

## 主要技术文档入口

- [`技术文档/39_实验全览与交接说明.md`](技术文档/39_实验全览与交接说明.md) — 实验全表、资产、已知问题、复现指南（先读）
- [`技术文档/38_R1评审修正与2017-2024生产方案.md`](技术文档/38_R1评审修正与2017-2024生产方案.md) — 当前执行依据
- [`技术文档/44_五省2023与主流土地覆盖产品对比.md`](技术文档/44_五省2023与主流土地覆盖产品对比.md) — 跨产品一致率全景
- [`技术文档/46_D0材料生成与体检报告.md`](技术文档/46_D0材料生成与体检报告.md) · [`技术文档/47_M3B灌丛仲裁AI预判读报告.md`](技术文档/47_M3B灌丛仲裁AI预判读报告.md) — 最新实验
- [`技术文档/00_文档索引.md`](技术文档/00_文档索引.md) — 全部文档索引与有效性标注

## 致谢与说明

- 数据基础：Google AlphaEarth Foundations（Earth Engine 年度嵌入）；对照产品 GLC_FCS10/FCS30D、ESRI LULC、ESA WorldCover。
- 本项目为**进行中研究**：所有精度数字均注明口径（开发集/教师泛化/独立人工验证），最新结论以 `技术文档/` 与 `VERSION.md` 为准。
- 引用信息与许可待定；如需数据或合作，请通过 issue 联系。
