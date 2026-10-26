# 附件：85 全国六类林地样本 · 证据清单

> 对应正文：[`../../85_全国六类林地样本生产结果.md`](../../85_全国六类林地样本生产结果.md)
> 归档时间：2026-10-25。原始工作区：`F:/lc_work/nb_national_2026/forest6/`（本目录为轻量证据副本，**不含 parquet**）。

## 一、交付样本（未归档，见原工作区）

| 文件 | 点数 | 说明 |
|---|---:|---|
| `forest6_core.parquet` | **96,845** | **推荐训练用**。全 A 档：41 20,000 / 42 15,000 / 43 15,000 / 44 20,000 / **45 13,776 / 46 13,069** |
| `forest6_full.parquet` | 212,406 | 含 B 档，带 `quality_tier`（A/B/C） |
| `forest6_south_needle_excluded.parquet` | 14,484 | 华南 41 类单列（FCS10 该区无判别效度） |
| `forest6_v1.parquet` | 472,307 | 全池中间件 |
| `mixed_expanded.parquet` / `mixed_expanded_aef.parquet` | 70,000 | 扩样混交候选（45/46 各 35,000） |

## 二、证据文件

| 本目录文件 | 原始路径 | 内容 |
|---|---|---|
| `f1_stats.json` | `forest6/f1_stats.json` | **资产合并**：897,204 → 去重 **472,307**（删 424,897）、四类矛盾 **0**；粗六类点数；**逐列 110 条缺失率**（AEF64 **0%**、地形 0%、物候 1.5–8.2%）；AEF 值域 [−0.53, 0.59] |
| `H1_result.json` | `forest6/H1_result.json` | **假设检验 H1**：RR_MH **3.10 [2.97,3.24]**、perm p=0.0050；**阈值敏感性**（[0.30,0.70]→2.93 不过）；最近邻配对 2.51/2.98；标签来源混杂说明 |
| `f2a_stats.json` | `forest6/f2a_stats.json` | **混交点物候补采**：6,864/6,864 成功；**口径校验中位差 0.00、相关 0.9916、MAE 8.74**；45/46 的 amp/djf 分布 |
| `f3_stats.json` | `forest6/f3_stats.json` | **六类融合 + L2**：三档（S1/S2/S3）比较、L1 自洽（violations=0）、**L2 全部口径**（三方 0.5256 / 剔除同源 0.6133 / **常绿落叶针叶轴 0.9603**）、生态过滤 83 点逐点清单、分布核对 |
| `f3_class_compare.csv` | `forest6/f3_class_compare.csv` | 4 档 × 层级/类别 × 60 列可比性指标 |
| `L3_result.json` | `forest6/L3_result.json` | **L3 影像验证（判据失效）**：总体 0.2955、逐类、**针/阔轴 AUC 0.489–0.497**、8 个敏感性变体、AEF64 上限诊断（分层/空间块/按省留出三套 CV） |
| `f5_aef_holdout.json` | `forest6/f5_aef_holdout.json` | **全池 v1 的 AEF64 三种留出设计**（块CV 0.7114 / 按省 0.5980 / 按生态区 0.5283）+ 9 对 pairwise |
| **`f5_aef_holdout_core.json`** | `forest6/f5_aef_holdout_core.json` | **core 高纯度集的三种留出设计**（块CV OA **0.7940**/macroF1 **0.7806**、按省 **0.7051/0.6901**、按生态区 0.6354/0.6217）+ **9 对 pairwise（AUC 0.9184–0.9824、acc 0.8314–0.9311）** |
| `f5_verdict.json` | `forest6/f5_verdict.json` | 三轴乘积上限估算 **0.6986** |
| `f5_thresholds.csv` | `forest6/f5_thresholds.csv` | 各生态区各类 `djf2023/amp2023` 的 P10/P50/P90（78 行，分带阈值依据） |
| `f6_expand_stats.json` | `forest6/f6_expand_stats.json` | **全国扩样四级漏斗**（1551 亿 → 369 亿 → 61.4 亿 → 54.2 亿 → 5413 万 → 13.4 万 → **70,000**）、分生态区/省、`frac_needle_hist`、`pure_forest_diagnostic` |
| `f7_stats.json` | `forest6/f7_stats.json` | **最终交付统计**：AEF 取数 70,000/70,000、**校验 max\|diff\|=1.45e-08**、两版本逐类/分档/分区、卫生检查、重叠率 |

## 三、代码

`code/` 下 8 个脚本：`f1_pool.py`（资产合并）、`f2a_pheno_mixed.py`（GEE 物候补采）、`f2b_mixed.py`（H1 + 邻域推导）、`f3_fuse.py`（六类融合）、`f4_l3.py`（L3 影像验证）、`f5_validate.py`（**AEF 留出验证，支持 `F5_SRC`/`F5_TAG` 环境变量切换数据集**）、`f6_expand_mixed.py`（全国扩样）、`f7_final.py`（AEF 取数 + 最终交付）。

## 四、复核五条最快路径

1. **核 core 精度**：`f5_aef_holdout_core.json` → `designs.prov.macro_f1=0.6901`、`designs.block10km.oa=0.7940`；逐类看 `designs.prov.per_class`（45/46 召回 0.518/0.529）。
2. **核分轴达标**：同文件 → `pairs.needle_vs_broad.{41vs43,42vs44,45vs46}.prov.auc` = 0.9824/0.9791/0.9184。
3. **核假设 H1**：`H1_result.json` → `rr_mh=3.10`、`ci=[2.97,3.24]`；再看 `threshold_sensitivity` 里 [0.30,0.70]→2.93。
4. **核扩样漏斗**：`f6_expand_stats.json` → `funnel`。
5. **核 L3 失效原因**：`L3_result.json` → 针/阔轴 AUC 0.489–0.497；再对 `f5_aef_holdout_core.json` 的 pairwise 0.94–0.98，说明是判据而非标签。

## 五、口径警告（必读）

1. **0.85 未达标，且是结构性的**：六类精确 = 三轴同时正确，三轴 acc 中位乘积上限 ≈0.699（实测 0.6901 已接近）。要 0.85 等于要求每轴 ≥0.95。
2. **不要用 L3 的 0.2955 判标签错**：S2 物候在针/阔轴 AUC≈0.49（完全无信息），而同批点 AEF64 pairwise 0.94–0.98。**需人工高分目视才能真验收**。
3. **45/46 不是 50 m 双层冠层**：L3 证窗口近乎均质。它们是"窗口内出现哪种针叶码"的构造，72.2%/74.0% 的点窗口内只有一种针叶码。
4. **38.4% 的扩样点 500 m 尺度仍是纯林**（M_C 档），core 已排除、full 保留但标 C。
5. **FCS10 的 91/92 奇偶不代表常绿/落叶**（本轮实测推翻原规则）：cls45 的 amp 中位 697.3 是强落叶。
6. **`amp2023` 是 ×1000 量纲**，不是 ×10000。
7. **AEF64 与 FCS10 同源**，留出验证只能证明"可学"，不能证明"生态学正确"。
