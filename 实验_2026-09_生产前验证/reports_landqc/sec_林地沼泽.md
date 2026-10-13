# 单类样本制作全过程（分章草稿）——林地 4 类 / 木本·草本沼泽

> 本章供主文档 `数据/单类样本的单独处理_2026.10.12/00_样本制作全过程说明.md` 的 §3（林地）与 §4（沼泽）使用。
> **纪律：本章所有数字均来自下列文档与台账文件，逐条在括号内注明来源；未标注来源的算式仅为把已记录的分子分母相除（如命中率），不引入新数据。**
>
> 主要来源：
> - 林地：`技术文档/65_林地样本全面质检4类林.md`（下称 **65**）；`F:/lc_work/v31_exp/reports/forest_qc_2026-10-13/{预注册判据.md, report.md, stats.json, calibration.json, per_year_counts.json, verify_report.json}`；代码 `F:/lc_work/v31_exp/code/w1_forest_fetch.py`、`w2_forest_qc.py`、`w3_verify_fetch.py`、`w4_forest_chips.py`、`w5_forest_assemble.py`（下称 **w1…w5**）；更早脉络 `技术文档/36`（疏/郁闭）、`54`（森林轴可分性）、`55`/`56`（V 档生态规则与样本 v2）、`60`（样本总纲）、`61`（郁闭疏闭问题）。
> - 沼泽：`技术文档/66_湿地样本木本草本沼泽质检与扩展.md`（下称 **66**）；`F:/lc_work/v31_exp/reports/swamp_qc_2026-10-13/{预注册判据.md, calibration_swamp.json, separation_report.json, stats_final.json}`；工作区台账 `F:/lc_work/v31_exp/data/swamp_qc/{out/, qc/, lists/, osm_wetland_stats.json}`；代码 `v1_swamp_fetch.py`、`v2_swamp_candidates.py`、`v3_swamp_qc.py`、`v4_swamp_expand.py`、`v5_swamp_assemble.py`、`v6_swamp_aef.py`、`data/swamp_qc/h2b_osm_wetland.py`（下称 **v1…v6 / h2b**）；更早脉络 `技术文档/54`（湿地细分可分性）、`56`（木本沼泽层 6.7% 废弃）、`60`（样本总纲）。
> - 交付态：`数据/单类样本的单独处理_2026.10.12/{shp_export_report.json, 林地_*/README_样本文件说明.md + 逐年池成员数.csv + 剔除清单.csv, 木本沼泽/…, 草本沼泽/…}`（2026-10-13 `s1_samples_to_shp.py` 导出，全部 PASS）。

---

## 1. 林地 4 类（常绿阔叶 / 落叶阔叶 / 常绿针叶 / 落叶针叶）

**术语速查（每个术语一句话）**

| 术语 | 一句话解释 |
|---|---|
| **E/D 轴** | 森林的"常绿（Evergreen）/落叶（Deciduous）"叶习性轴——本类集唯一能用物候独立验证的轴（**54**：AEF 单独 AUC **0.990**）。 |
| **针/阔轴** | "针叶 / 阔叶"轴——**54** 实测最佳 AUC 仅 **0.724 < 0.85**，本版**不做任何自动改判**，只做区域硬规则与提示。 |
| **物候振幅 `amp`** | S2 双窗 NDVI×1000 的 `|jja − djf|`：冬季落叶越彻底振幅越大（常绿低、落叶高）；`amp_rel = amp / max(max(|djf|,|jja|),50)` 为归一化版辅助层。 |
| **bio06 / bio11** | WorldClim V1 生物气候量 @1 km：**bio06=最冷月最低温（℃×10）**（本版气候分带键）；**bio11=最冷季均温（℃×10）**（R1 用）。 |
| **AEF64** | Google AlphaEarth Foundations 年度嵌入 64 维向量（`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`，10 m 采样），本版用作 E/D 轴 LDA 判别的特征。 |
| **WorldCover / DW / MODIS** | ESA WorldCover v200（2021，11 类码，10=树覆盖、30=草、40=耕）；Dynamic World 2023（10 m 概率，trees 等）；MODIS MCD12Q1 `LC_Type1` 2022（500 m，IGBP：1 常绿针叶/2 常绿阔叶/3 落叶针叶/4 落叶阔叶/5 混交）——**MODIS 是唯一独立给出"针/阔 × 常绿/落叶"四分的产品**（**65**）。 |
| **FCS10 / FCS30D** | 刘良云组 GLC_FCS10（2023）与 GLC_FCS30D 产品；本类集 93.4% 标签来自其 7 年稳定层（见 1.3.1），故评估的"亏损在标签不在模型"。 |
| **strict / clean / 隔离** | 三层交付口径：**strict=未隔离且无任何 flag**（≥2 独立证据层支持原标签）；**clean=未隔离**（含弱证据/缺测/框级 flag）；**隔离=被 ≥2 独立层决定性反驳且零支持，或触犯硬规则**（**65** §3.1）。 |
| **tier=external** | 母库质量分层字段的一档：由外源/专题产品**构造**的点（非 FCS 多源投票点），母库要求 `agree_n ≥ 阈值` 或专题来源才保留（`代码/5.样本重建/r7_clean.py` L63-65）；本类集 external **30,744 点（6.6%）**（`base_forest_points.parquet`）。 |

### 1.1 一句话结论 + 当前文件清单

**一句话结论**：对 8 年子集并集的 4 类林 **466,135 点**做物候（2019+2023）、生态气候、独立产品、AEF 嵌入四层证据质检，**隔离 35,246 点（7.56%）**、**strict 推荐层 159,347**、**clean 可用层 430,889**；混交林 91/92（**6,864 点**）按用户指令本版弃用；母库与 8 年子集**本体未动**（MD5 前后一致），结论以 `row_id` 为键，天然覆盖 8 年（**65** §一、§3.1；`stats.json`）。

| 文件（绝对路径） | 内容 | 点数 / 行数 |
|---|---|---|
| `F:/lc_work/v31_exp/data/forest_qc/base_forest_points.parquet` | 质检基座（8 年并集 4 类林；列：point_id/lon/lat/class_new/src/tier/train_weight/aef_year/class4/prov） | 466,135 |
| `F:/lc_work/v31_exp/data/forest_qc/feats/{s2_y2019, s2_y2023, env, xs}/` | 证据块（S2 物候两年 / 生态环变量 / 交叉产品），每模式 119 块（`verify_report.json`） | 466,135 ×4 模式 |
| `F:/lc_work/v31_exp/data/forest_qc/out/forest_samples_v1.parquet/.csv` | 全量结果（原标签、`suggest_class4`、逐层证据、`n_support/n_refute`、生态与交叉产品列、`quarantined/flag/reason`、AEF64） | 466,135 |
| `…/out/forest_samples_v1_clean.{parquet,csv}` | 可用层（未隔离） | **430,889** |
| `…/out/forest_samples_v1_strict.{parquet,csv}` | 推荐层（未隔离且无 flag） | **159,347** |
| `…/qc/quarantine.csv` | 隔离点 + 理由码 + 建议改判 | **35,246** |
| `…/out/mixed_forest_9162_excluded.csv` | 弃用混交点（91/92，从 2023 子集单独提出） | **6,864** |
| `…/out/{report.md, stats.json, calibration.json, per_year_counts.json}` | 报告 / 统计 / 阈值校准表 / 逐年影响 | — |
| `…/out/chips/{clean_sheet.png, quarantine_sheet.png}` | z18 切片复核材料（干净组 vs 隔离组；单期影像**不能**判 E/D，仅供框级复核） | 各 12 点 |
| `…/{verify_report.json, subset_md5_before.json}` | 抓取核验（块 119×4 全绿、GEE 抽查 0.0 差）与 8 年子集只读基线（8 个 MD5） | — |
| `F:/地理所/…/数据/单类样本的单独处理_2026.10.12/林地_<类名>/2023/林地_<类名>_2023_{WGS84,GCJ02}.*` | **交付 SHP**：8 年并集 clean 层（字段 `pid/lon/lat/prov/cls4/clsname/layer=strict 或 clean/tier/sug/src`），逐文件回读 0.0 差、PASS | 常绿阔叶 **112,083** / 落叶阔叶 **124,304** / 常绿针叶 **134,717** / 落叶针叶 **59,785**（合计 430,889；`shp_export_report.json`） |
| `…/林地_<类名>/林地_<类名>_剔除清单.csv` | 本类隔离点（E/D 轴反驳或 R1），行数（不含表头） | 13,416 / 14,345 / 7,220 / 265（与 `stats.json` 一致） |
| `…/林地_<类名>/逐年池成员数.csv` | 该点集在 2017–2024 各年训练池中的成员数（= 各年 clean 数） | 见 1.4 表 |

> 四类总点数（并集）：常绿阔叶 **125,499** / 落叶阔叶 **138,649** / 常绿针叶 **141,937** / 落叶针叶 **60,050**（**65** §一；`base_forest_points.parquet`）。
> 30 类码 → 24 类码：4=[51,52]、5=[61,62]、6=[71,72]、7=[81,82]、8=[91,92]（`config/v31_map.json` merges）。

### 1.2 版本演进时间线

| 版本 | 日期 | 做法（一句话） | 点数 | 结论 / 为何被替代 |
|---|---|---|---|---|
| **V0 原始标签**（母库/FCS 系） | ≤2026-09 | 30 类体系：森林 5 类 ×（郁闭/疏闭 2 码）= 51/52/61/62/71/72/81/82/91/92 共 10 码 | 母库 2,252,207 中森林占比见（**60** §1.2） | 沿用 GLC_FCS 惯例；疏/郁闭的类边界是"树冠覆盖度 fc 阈值"（Open 0.15–0.4 / Closed >0.4），**不是光谱类**（**36** §二） |
| **V0.5 E6 疏/郁闭合并**（**36**） | 2026-09-24 | 二分类探针（64 维 AEF、RF、5 折 CV）+ 质心余弦 | 52 共 5,754 / 62 共 10,880 点 | 52 vs 51 **AUC 0.785**、62 vs 61 **0.833**（对照类对全 0.95+）；质心余弦 0.985 / 0.973 → **真不可分，合并为单一森林码**（24 类映射 `merges`） |
| **V1 实验 A 森林轴**（**54**） | 2026-10-09 | 两轮 z18 判读（378→113 森林点）+ 物候标签（amp<0.15 常绿 / >0.30 落叶）+ AEF/S2/S1/DEM 探针 | 22 常绿 / 67 落叶 | **常绿 vs 落叶 AEF AUC 0.990** ✅；**针 vs 阔 0.724** ❌（落叶组内针叶 0 点样本不足）→ 建议"森林 5→2"；同时记录目视"常绿"标签 19/29、43/74 与物候冲突、FCS10 森林层与判读严重不符 |
| **V2 样本集 v2 + 四川重分类**（**55/56**） | 2026-10-09/10 | 区域化生态硬约束筛选（19 条 V 档规则，仅硬否决"不可能"组合）+ 本地对照实验（1,325 判读真值点） | v2 保留 1,503,331 / 2,060,577 = 73.0%；生态筛选按类含常绿阔叶 **12,426**、常绿针叶 6,164、落叶阔叶 1,872 | 24 类 OA **+1.36 pp**、9 大类 **+2.34 pp** → 通过；**用户要求：森林细分不合并**（维持 4/5/6/7/8 五类，声明为"C 档：无可分性证据"）；另记录 97–99.3°E 有 **323 个 >3800 m 常绿阔叶训练点**（标签惯例，非掩膜可修） |
| **V3 全面质检 + 修订 1**（**65**；w1–w5） | 2026-10-12/13 | 8 年并集 4 类林，E1 物候（2019+2023）/E2 生态/E3 独立产品/E5 AEF-LDA 四层；首轮 9.9% → **修订 1**（气候带校准 + 雪窗守门 + 极端冷区停用）→ 重跑 | **466,135**（总 472,999 含混交 6,864） | 隔离 **35,246（7.56%）**；strict **159,347**（34.2%）；clean **430,889**（92.4%）；混交弃用 6,864 |
| **交付态 SHP** | 2026-10-13 | `s1_samples_to_shp.py` 导出双坐标 SHP（WGS84 + GCJ-02），clean 层为几何主体、`layer` 字段区分 strict/clean | 4 类 clean 合 430,889（其中 strict 159,347） | 逐文件回读 `roundtrip_maxdiff=0.0`、PASS（`shp_export_report.json`） |
| **待办（V4）** | — | M3 人工判读（≥3,000 点）；图例瘦身决策（**54** 建议森林 5→2） | — | 属用户侧决策，未执行（AGENTS 待办 1/6） |

### 1.3 各版本技术细节

#### 1.3.1 输入与来源（V3 基座，`base_forest_points.parquet`）

- 点集 = 8 年子集 `F:/lc_work/年度子集_含稀有类/r7_train_{2017..2024}.parquet` 中 51/52/61/62/71/72/81/82 的**并集**，按 `row_id` 去重：**466,135**（本人复核：8 年森林行合计 3,543,764，唯一 row_id 466,135；**同一 row_id 跨年 class_new 不一致的点 = 0**，与**预注册判据.md** §对象 的声明一致）。
- 标签来源 `src`：`v2_glc_fcs30d2020_7yr_stable` **435,391（93.4%）**、`glc_fcs10_2023_band130` 18,366、`glc_fcs10_2023` 10,507、`ng_evergreen52` 1,668、`eglc_GLanCE-2` 146、`eglc_GLanCE-1` 57（基座文件）。
- `tier`：gold 226,282 / silver 106,797 / bronze 102,312 / **external 30,744（6.6%）**（基座文件；external 含义见术语表）。
- `aef_year`：2023 占 459,451（98.6%），其余 2022 4,969 / 2024 636 / 2020 303 / 2018 283 / 2021 256 / 2019 166 / 2017 71（基座文件）——AEF64 按 `aef_year` 从年度子集直接取用，命中 **1.0000**（**65** §二 E5 行；w2 运行日志）。
- **混交 91/92 不入本集**：2023 子集 6,864 点另存 `mixed_forest_9162_excluded.csv`（`stats.json: mixed_9162=6864`）。

#### 1.3.2 证据层口径（w1 代码 = 唯一实现源）

| 层 | 数据与口径（代码级） | 采样 |
|---|---|---|
| **E1 物候** | `COPERNICUS/S2_SR_HARMONIZED`；SCL 掩膜剔 **3/8/9**（云影/中/高概率云）；`CLOUDY_PIXEL_PERCENTAGE < 60`；DJF=**上年 12-01 ~ 当年 03-01**、JJA=**当年 06-01 ~ 09-01**；`median()`（**未剔 11=雪**，由修订 1 的雪窗守门补偿）；NDVI×1000，`unmask(-9999)`→本地转 NaN；**2019 与 2023 两年各一套**（w1 `--what s2`） | scale=20 |
| **E2 生态** | `USGS/SRTMGL1_003` dem + `WORLDCLIM/V1/BIO` 的 bio01/bio06/bio11/bio12 | scale=1000 |
| **E3 独立产品** | `ESA/WorldCover/v200`（`Map` 类码）+ `GOOGLE/DYNAMICWORLD/V1` 2023 全年均值（trees/shrub_and_scrub/crops/grass/water）+ `MODIS/061/MCD12Q1` 2022 `LC_Type1` | scale=500 |
| **E5 嵌入** | AEF64（年度子集自带；w2 `load_aef` 按 `aef_year` 取） | scale=10（母库原生） |
| E4 库内来源 | `src`/`tier`/`train_weight` | 只作溯源与分层，**不作证据**（预注册判据.md §一） |

抓取工程：`--part i --nparts 7` 分片、块 4000 点、**逐块幂等**（存在即跳过）、失败重试 6 次（8 s×序号退避）、失败记 `_fail_*.json`（w1 代码）。共 **119 块/模式**（466,135 = 7×17 块；`verify_report.json` inventory）。

#### 1.3.3 判据（预注册 + 修订 1）

**预注册（2026-10-12，先写死再跑；`森林_qc/预注册判据.md`）**：

- **轴 1（E/D）**：
  - **矛盾（quarantine 候选）**= 原方向被 **≥2 个独立证据层决定性反驳且无任何一层支持**（两层物候各算一层；AEF 层后验 <0.10；MODIS 直接给反向 E/D）；
  - **一致（clean）**= ≥2 层支持原方向；其余 = 灰带（打 `flag_gray`，保留不隔离）；
  - **不做自动改判**（标签来自 FCS30D 7 年稳定层，改判风险大）——隔离点只给"建议改判类"列 `suggest_class4`（同类叶型内翻转 E/D）。
- **轴 2（针/阔）**：只做硬规则与提示，**不自动改判**——R1 常绿阔叶不可能（`class=4 且 bio11≤0 且 dem<2000`）；R2 热带落叶存疑（`class∈{5,7} 且 bio01≥240 且 bio12≥1200`）；R3 极端干旱区针叶存疑（`class∈{6,7} 且 bio12<200 且 dem<1500`）；R4 MODIS 针/阔冲突提示（MODIS∈{1,3} 而标签∈{4,5} 或反向，且 `dw_trees<0.2`）；前三条规则跑前先做标签分布自检，命中 > 该类 2% 则降级为 flag。
- **共同层**：Q1 非林框架信号（WorldCover∈{20,30,40} 且 `dw_trees<0.25`）；Q2 数据完整性（任一年缺 NDVI 或 env）。

**首轮实测暴露两点（隔离 45,918 = 9.9%）→ 修订 1（2026-10-13，重跑前写死；首轮数字作废）**：

1. **阈值适用域**：首轮把"小省"回退全国阈值，**未实现预注册的"并入气候带"**——冷区常绿针叶的**真实冬枯**（被隔离类 6 特征：bio06 中位 **−11.4 ℃**、dem 中位 **1,661 m**；amp 中位 490 > 全国阈 382）被误判"落叶"（类 6 隔离率 **13.35%**）。
2. **雪窗不可辨**：SCL 未剔 11（雪），某年冬窗若被雪/裸土支配（DJF<150），该年物候对 E/D 不可辨。

修订三条（均在 w2 代码中实现）：

- **R-a 气候带校准**：按 bio06 分 5 带（z1 >30 / z2 0~30 / z3 −50~0 / z4 −100~−50 / z5 ≤−100，℃×10）；训练集 = **"标签 ∩ MODIS E/D 一致"** 的点；阈 = 常绿/落叶 **amp2019 中位的中点**，灰带 = **0.35 × 两类 IQR 的均值**（下限 20）；带内每类 <200 时并入更暖带/全国缺省；**2019 定阈、2023 独立应用**（预注册）。
- **R-b 雪窗守门**：某年 DJF<150 → 该年 verdict 置空（非决断）。
- **R-c 极端冷区停用**：bio06 ≤ −100（最冷月最低温 ≤ −10 ℃）→ **物候层整体非决断**（冬季信号由雪/冻害主导），改由 AEF/MODIS 判，均不决断则落灰带 flag 保留。

**校准结果（`calibration.json` / `stats.json:calib`，单位 ℃×10）**：

| 带 | bio06 范围 | thr | gray | 常绿中位 | 落叶中位 | nE/nD |
|---|---|---|---|---|---|---|
| z2 | 0~30 | **185.5** | 60.3 | 82.8 | 288.3 | 12,229 / 488 |
| z3 | −50~0 | **270.5** | 50.9 | 70.0 | 471.0 | 6,589 / 6,649 |
| z4 | −100~−50 | **316.3** | 48.5 | 70.1 | 562.5 | 2,803 / 7,049 |
| z5 | ≤−100（物候停用） | **327.4** | 49.8 | 49.2 | 605.6 | 3,681 / 54,754 |
| 全国缺省 | —（z1 暖区样本不足并入） | **326.1** | 56.5 | — | — | — |

带阈值随冬季变冷单调上升（186→327），与"冬季越冷、常绿冬枯越强"的物理预期一致（`report.md` §三）。辅助层 `amp_rel` 校准中点 **0.389**，与参考值 0.424 偏离 **8.3%（<30% → 报告不改）**（`stats.json:global_calib`）。

**E5 AEF-LDA（w2 代码）**：训练集 = "标签 ∩ MODIS 一致 ∩ 2019 且 2023 物候均支持"的**四重一致**点 **47,071 个**；LDA 用收缩协方差（**shrinkage 0.05**、ridge 1e-6）；判别阈值 margin = **2.197 = ln(0.9/0.1)**（对数似然比 >±2.197 才表态）；训练集内误分率 **0.89%**（`stats.json:aef_lda`）。

#### 1.3.4 隔离 / flag 规则与理由码

- **支持/反驳计数**：4 层（v2019、v2023、v_aef、modis_ed）逐层与 `label_ed` 比对，`n_support` / `n_refute`；`quarantine_ed = (n_refute ≥ 2) & (n_support == 0)`；`gray_ed = ~quarantine_ed & (n_support < 2)`（w2 代码）。
- **最终隔离** = `quarantine_ed` ∪ `R1`；**flag** = 未隔离 且（gray ∨ R2 ∨ R3 ∨ R4 ∨ Q1 ∨ Q2）。
- 理由码计数（`stats.json:reasons`）：`ED_CONFLICT` **34,205**、`R1_COLD_EB` **1,679**（重叠 638；合计隔离 35,246）、`R2_TROP_DECI` 99、`R3_ARID_CONIF` 112、`R4_LEAF` 288、`Q1_NONFOREST` 16,721、`Q2_MISS` 42,344。
- R1 命中 1,679 / 125,499 = **1.34% < 2% 预注册门限** → 保留为隔离硬规则（命中率由两个已记录数相除）。
- 建议改判（`suggest_class4`，仅对 `quarantine_ed`）：常绿阔叶→落叶阔叶 **12,375**、落叶阔叶→常绿阔叶 **14,345**、常绿针叶→落叶针叶 **7,220**、落叶针叶→常绿针叶 **265**（合计 34,205；`stats.json:ed_switch`）。
  - ⚠ 注：doc 65 §3.2 正文写的 11,452/12,939/6,906/2,908 与 `stats.json`/`report.md` 不一致（四数之和也恰为 34,205，属旧稿数字），**以 `stats.json` 为准**；见文末"矛盾清单"。
- **分类隔离率**（`stats.json:by_class` + `report.md` §二；首轮对照来自预注册判据.md 修订 1）：常绿阔叶 13,416（**10.69%**，首轮 10.19%）、落叶阔叶 14,345（**10.35%**，首轮 7.08%）、常绿针叶 7,220（**5.09%**，首轮 13.35%——雪区修复后大降）、落叶针叶 265（**0.44%**，首轮 7.27%）。
- 类内 flag 数：常绿阔叶 39,590、落叶阔叶 97,043、常绿针叶 75,137、落叶针叶 59,772（`stats.json:by_class`；flag 占比高主要因 z5 冷区物候停用 + MODIS 对约 2/3 点不表态，**65** §五.4）。
- 隔离集中省（`stats.json:by_prov`，前六）：云南 8,070 / 四川 5,564（**65** §3.2 写 5,527，见文末矛盾）/ 陕西 4,767（**65** 写 4,223）/ 湖北 4,167（**65** 写 4,006）/ 广西 1,492 / 贵州 1,498。

### 1.4 当前版完整流程（w1 抓取 → w2 质检 → w5 装配）

**第 1 步 w1_forest_fetch.py（证据抓取，逐块幂等）**
1. 读基座 `point_id/lon/lat`；按 `--part/--nparts=7` 取本片；块 4000 点，输出 `feats/{s2_y2019,s2_y2023,env,xs}/<pre>_p{pp}_c{kk}.parquet`。
2. 三种模式：`s2`（S2 双窗 NDVI×1000，scale=20）/`env`（SRTM+WorldClim 5 列，scale=1000）/`xs`（WorldCover+DW5+MODIS 共 7 列，scale=500）；`sampleRegions(..., geometries=False, tileScale=4)`；S2 的 `≤−9000` 哨兵本地转 NaN。
3. 幂等与重试：块文件存在即跳过；单块失败重试 6 次（8 s×i 退避），仍失败写 `_fail_*.json` 留痕。

**第 2 步 w2_forest_qc.py（质检主链）**
1. 合并 4 模式特征 + 按 `aef_year` 合并 AEF64（命中 1.0000 打印于日志）。
2. 派生 `amp2019/amp2023`、`rel2019/rel2023`、`mx2019/mx2023`；`modis_ed` 由 `LC_Type1` 映射（1/2→常绿、3/4→落叶、5→不表态）。
3. **校准**（R-a）：训练集 = `modis_agree`，按 `czone` 分组拟合 thr/gray（见 1.3.3 表）。
4. **逐年 verdict**：`a > thr+gray → 落叶`；`a < thr−gray → 常绿`；带内空；再依次执行 R-b（DJF<150 置空）、R-c（bio06≤−100 置空）。
5. **AEF LDA** 训练（四重一致 47,071）→ 全量打分 `aef_score/aef_p_ever` → `v_aef`（|margin|>2.197 才表态）。
6. **计数与判定**：`n_support/n_refute/n_decisive` → `quarantine_ed/gray_ed`；再算 R1–R4、Q1、Q2；`suggest_class4`（冲突点翻转 E/D）；`quarantined` / `flag` / `reason`。
7. **输出**：`forest_samples_v1.parquet/.csv`（含 AEF64 全列）、`qc/quarantine.csv`、`stats.json`（含 `calibration`/`per-class`/`by_prov`/`reasons`/`ed_switch`/`gates`）、`calibration.json`、`mixed_forest_9162_excluded.csv`。
8. **门（PASS 判据）**：G1 记账 466,135；G2 `point_id` 唯一；G3 证据覆盖（两年物候 ≥90%、env ≥98%、xs ≥95%）；G4 隔离点全部 `n_refute≥2 且 n_support==0` 且带理由码；G5 校准表完整。实测：**G1–G5 全 true，PASS=true**，覆盖率 s2_2019 **0.9315** / s2_2023 **0.9751** / env **0.9996** / xs **0.9962**（`stats.json:gates`）。

**第 3 步 w3_verify_fetch.py（独立核验，见 1.5）→ 第 4 步 w4_forest_chips.py（复核切片，z18 512² 拼图，含十字丝与 10 m 青框）→ 第 5 步 w5_forest_assemble.py**
1. 从 v1 结果切两层：clean（`~quarantined`）**430,889**、strict（`~quarantined & ~flag`）**159,347**，各出 parquet+csv。
2. **逐年影响**（按 `row_id` 键映射回 8 年子集，森林码 51–82）：`per_year_counts.json`；示例 2017：森林 432,474 / 隔离 33,076 / 仅 flag 246,569 / strict 152,829。
3. 生成 `report.md`（结论、分类隔离率、校准表、两层守门、验证、逐年、其它 flag、局限）。
4. 门：`rows_ok`（466,135）、`unique`（无重复）、`clean+quar == 总`、`strict ≤ clean`、`per_year_all`（各年森林 >40 万）→ 全 PASS（w5 运行日志）。
5. 交付态：`s1_samples_to_shp.py` 以 **clean 层**为几何主体导出双坐标 SHP，`layer` 字段标 strict/clean，`sug` 列给建议改判码（-9=无建议）；`逐年池成员数.csv` 即各年 clean 成员数，8 年合计与 `per_year_counts.json` 的 clean（= 森林点 − 隔离）**逐年逐值一致**（本人复核 8/8 年）。

**逐年影响表（`per_year_counts.json`；strict 为 4 类合计）**

| 年 | 森林点 | 隔离 | 仅 flag | strict |
|---|---|---|---|---|
| 2017 | 432,474 | 33,076 | 246,569 | 152,829 |
| 2018 | 431,750 | 33,064 | 245,693 | 152,993 |
| 2019 | 432,594 | 33,179 | 246,453 | 152,962 |
| 2020 | 432,734 | 33,147 | 245,408 | 154,179 |
| 2021 | 433,766 | 33,056 | 246,315 | 154,395 |
| 2022 | 462,215 | 34,402 | 269,637 | 158,176 |
| 2023 | 459,451 | 34,554 | 268,863 | 156,034 |
| 2024 | 458,780 | 34,478 | 268,823 | 155,479 |

### 1.5 验证链

1. **抓取保真（w3，三层）**：
   - V1 块清单：按 466,135 点 / 7 分片 / 块 4000 重算应有块与每块行数，逐块比对——s2_y2019、s2_y2023、env、xs **各 119/119 块，缺 0、行数异常 0**；
   - V2/V3 行数与覆盖：各模式合并行数 = **466,135** 且 `point_id` 唯一；覆盖 s2_2019 amp 0.9315 等（见 1.4 第 2 步门值）；值域自检（env 物理量级、wc/modis 类码、dw∈0–1、NDVI×1000 尺度）；
   - V5 GEE 抽查（随机 24 点，种子 20261012）：重取 s2_2019、s2_2023、env 三路逐点比对，`max|Δ|` 全为 **0.0**（`verify_report.json:checks.spot`），`V1V2_PASS` / `V5_PASS` / `PASS` 全 true。
2. **E/D 轴的独立核验（逐月 NDVI，2023 抽查 10 点；`report.md` §五）**：

| 组 | 冬均（12–2 月） | 夏均（6–8 月） | 夏−冬 | 判读 |
|---|---|---|---|---|
| 隔离·滇 类 5（标落叶阔叶） | 537 / 620 / 724 | 848 / 820 / 821 | +97 ~ +312 | 冬季仍绿 → 支持"应常绿" |
| 隔离·陕 类 4（标常绿阔叶） | 580 / 310 / 293 | 879 / 860 / 867 | **+299 ~ +574** | 冬季大幅落叶 → 支持"应落叶" |
| 对照·干净滇 类 4 | 848 / 801 | 792 / 850 | −56 / +50 | 真常绿 ✓ |
| 对照·干净黑 类 5 | 76 / 112 | 835 / 896 | **+759 / +784** | 真落叶 ✓ |

3. **三层交叉（全量口径）**：MODIS 明确给出 E/D 的点占 **32.9%**，其中与原标签同向 **85.0%**（本人复核：`modis_ed` 非空 153,391/466,135 = 32.9%）；AEF LDA 训练集内误分 **0.89%**；**针/阔轴 MODIS 一致率仅 65.4%**——与 **54** 的 AUC 0.724 互相印证，故针/阔**不做自动改判**（`report.md` §五）。
4. **只读性**：`subset_md5_before.json` 记录 8 个年度子集 MD5（2017 `0dad4a7e…` … 2024 `71d4eebc…`），质检全程未写子集/母库（**65** §一；w1–w5 只读）。
5. **交付态几何回读**：SHP 逐文件 `roundtrip_maxdiff = 0.0`、记录数=点数、PASS（`shp_export_report.json`）。

### 1.6 已知限制（诚实清单）

1. **针/阔轴无法独立验证**（**54**：AUC 0.724；本版 MODIS 一致率 65.4%）：只做 R1 硬规则 + R2/R3/R4 提示；叶型标签沿用来源（FCS30D 7 年稳定层为主，占 93.4%）——**4 类林中的"针/阔"维度本质上是继承的外部惯例**（**56** 已声明为"C 档：无可分性证据"）。
2. **物候阈值以 2019 单年校准、2023 独立应用**（预注册）；**暖区 z1 因落叶样本不足并入全国缺省**，故南方（含台湾）常绿点的判据实际用全国阈值。
3. **隔离 ≠ 判定错误**：隔离是"≥2 独立的代理证据与标签冲突"，`suggest_class4` 仅供下游参考；**未自动改标签**（改判风险交用户/下游）。
4. **flag 占比高（58.2% = 271,542/466,135）**：主因 z5 冷区物候停用 + MODIS 对约 67% 点不表态（500 m 粗）；Q2 数据缺失 42,344 多为 2019 南方云区/近岸海面（**65** §五）。
5. **近岸海面点**（env 空值 **202 个**，本人复核）与"近岸海面被分类"政策 A/B 同源，待用户定夺。
6. **复核材料限制**：chips 为**单期**影像，不能判 E/D（**54** 已证目视"常绿"标签 67% 与物候冲突），仅供框级复核。
7. **外源构造点（tier=external）30,744 点**（6.6%）无 FCS 多源投票支撑（其保留依据是母库 `agree_n ≥ 阈值` 或专题来源，`代码/5.样本重建/r7_clean.py`）；本版质检对其**不加区别**地套同一判据（w2 代码无 tier 分支），使用时可按 `tier` 分层。
8. **混交（8 类）本版弃用** 6,864 点——若恢复 5 类口径需单独质检（本版未做）。
9. **历史遗留的标签惯例问题未在样本端解决**：97–99.3°E 有 323 个 >3800 m 常绿阔叶训练点，V 档规则 R04 在四川 24 瓦内实测否决 739 像元（**55**/`report.md` §2.5），说明该类在横断山区的"常绿阔叶"是外部标签惯例，样本端只隔离了 E/D 冲突点，未处理"类定义"问题。

### 1.7 文件 → 版本对应表

| 版本/阶段 | 文件 | 说明 |
|---|---|---|
| V3 基座 | `data/forest_qc/base_forest_points.parquet` | 8 年并集 466,135 点（含 class4/class_new/src/tier/aef_year/prov） |
| V3 证据 | `data/forest_qc/feats/{s2_y2019,s2_y2023,env,xs}/<pre>_p*_c*.parquet`（119×4） | w1 产物；`_fail_*.json` 若有即缺块留痕（本次为 0） |
| V3 结果 | `data/forest_qc/out/forest_samples_v1.{parquet,csv}` | 466,135 点全字段（含 AEF64、逐层证据、reason） |
| V3 clean | `data/forest_qc/out/forest_samples_v1_clean.{parquet,csv}` | 430,889（未隔离） |
| V3 strict | `data/forest_qc/out/forest_samples_v1_strict.{parquet,csv}` | 159,347（未隔离且无 flag） |
| V3 隔离 | `data/forest_qc/qc/quarantine.csv` | 35,246 + 理由码 + `suggest_class4` |
| V3 弃用 | `data/forest_qc/out/mixed_forest_9162_excluded.csv` | 6,864（91/92，2023 子集） |
| V3 台账 | `data/forest_qc/out/{stats.json, calibration.json, per_year_counts.json, report.md}` + `data/forest_qc/{verify_report.json, subset_md5_before.json}` + `预注册判据.md` | 校准表、逐年、核验、MD5 基线、判据（含修订 1） |
| V3 复核 | `data/forest_qc/out/chips/{clean,quarantine}_sheet.png` | z18 切片 12+12 点 |
| V3 报告副本 | `F:/lc_work/v31_exp/reports/forest_qc_2026-10-13/`（5 文件） | 与 data 侧一致（`report.md/stats.json/calibration.json/per_year_counts.json/verify_report.json/预注册判据.md`） |
| 交付态 | `数据/单类样本的单独处理_2026.10.12/林地_<类名>/{2023/*.shp, 剔除清单.csv, 逐年池成员数.csv, README_样本文件说明.md}` | clean 层双坐标 SHP（layer 字段标 strict/clean）+ 逐类隔离清单 + 逐年成员数 |
| 代码 | `code/w1…w5_*.py`（w0 基座并入 w2 前置） | 抓取 → 质检 → 核验 → 切片 → 装配 |

---

## 2. 木本沼泽 / 草本沼泽（24 类码 14 / 15；30 类码 180 / 181）

**术语速查（每个术语一句话）**

| 术语 | 一句话解释 |
|---|---|
| **JRC GSW** | JRC Global Surface Water（`JRC/GSW1_4`）：`occurrence`（1984–2021 有水频率 %）、`seasonality`（年内有水月数）、`recurrence`（年际回归率 %）、`max_extent`（历史最大水范围 0–1）；另有 `MonthlyHistory` 逐月 `water==2` 掩膜，本版取 **2015-01 ~ 2021-12 共 12 个月位**的水频。 |
| **点级 vs 邻域水** | 木本沼泽树冠遮挡使**点位本身常看不到水**：现有 180 点点级淹水月数中位 **0**、方位水中位 **1**；181 草本为 **3 / 3**——"点在水中"与"点在环水陆地"是两种地物签名（**66** §三.3）。 |
| **relief500 / relief2000** | `dem − focal_min(dem, R)`（R=17 px/67 px，SRTM 30 m 原生 → **≈510 m / 2010 m**）：值小=地形低洼/汇水，值大=堤岸/坡地；是"排除河道防护林"的位势判据（**66** §二 S2 行；v1 代码）。 |
| **sampleRegions 联合掩膜陷阱** | 多波段图像 `Image.cat` 后，**任一波段在某点被掩膜 → 整点被 sampleRegions 丢弃**；GSW 对"从未有水"像元整体掩膜，首轮因此丢掉 71.6% 整行（丢的恰是需要的干点负样本）。修法：**逐源 `unmask(-1)` 再 cat**（v1 代码注释；**66** §三.1）。 |
| **LEVEE_LIKE（堤岸型）** | 冻结核判据下"点级零淹水 且 300 m 邻域历史最大水覆盖 ≥0.3"的点——签名即**河道防护林/自然岸林**（用户点名要排除的对象）；v3 代码 `levee = (~core) & (max_300 >= 0.3) & (pt_flood_months == 0)`。 |
| **DW `flooded_vegetation`** | Dynamic World 2023 的"淹没植被"概率（10 m 原生，本版取全年均值，点值 + 150 m 邻域均值 `dw_fv_150`）。 |
| **WorldCover 90** | ESA WorldCover v200 的 "Herbaceous wetland" 类码（本版候选源 B）。 |
| **MODIS 11** | MCD12Q1 `LC_Type1` 的 "Permanent wetlands"（本版仅作独立佐证）。 |
| **OSM tag** | OpenStreetMap `natural=wetland` + `wetland=*` 的众包标注；本版用 `wetland=swamp` 给**木本沼泽**提供独立分型标注，`marsh|reedbed|wet_meadow|fen|bog|无值` 归草本（**66** §六.3）。 |
| **复合体加密** | 在现有保留点 3 km 内随机撒新点、继承母点类别的扩展法（v4 系列；`lists/densify_woody.parquet`）。 |
| **FCS10 木本沼泽层** | 30 类码 180 的历史标签来源层；判读精度仅 **6.7%（1/15）**（**56** §2），是本次质检的起点问题。 |

### 2.1 一句话结论 + 当前文件清单

**一句话结论**：对现有 **180 木本沼泽（8 年并集 4,968 点）/ 181 草本沼泽（29,644 点）**做三层证据（水文 GSW/DW/产品、地形、光学/SAR 植被）质检，并沿"候选 + 复合体加密 + OSM"三路扩展，**最终交付 48,314 点（木本 10,612 / 草本 37,702）→ 并入 OSM 后 v2 = 49,016（木本 10,656 / 草本 38,360）**；全部通过冻结判据「**点级季节性淹水 ≥1 月** 或 **（8 方位中 ≥3 方位见水 且 平坦低洼）**」；该判据对背景随机点特异性极强（**森林 0.06% / 草地 0.35% / 耕地 2.1%** 通过），**河道防护林/岸林被结构性排除**——现有木本沼泽 **54.1%（2,687/4,968）被判为无水/堤岸型并隔离**（**66** §一、§五；`calibration_swamp.json`、`stats_final.json`）。

| 文件（绝对路径） | 内容 | 点数 / 行数 |
|---|---|---|
| `data/swamp_qc/lists/existing_180181.parquet` | 现有样本（180 + 181，8 年并集，列 point_id/lon/lat/class_new/src） | 180 **4,968** / 181 **29,644**（2023 子集：2,638 / 20,349） |
| `data/swamp_qc/lists/background_40k.parquet` | 背景随机点（含 class_new 分层） | 40,000（森林 8,733 = 近水 8 + 远水 8,725；耕地 6,320；草地 3,696） |
| `data/swamp_qc/lists/judged_anchors.parquet` | M3 缺口判读锚点（truth_type：木本 31 / 草本 253 / 其他 10） | 294（**已判定不可作真值**，见 2.5） |
| `data/swamp_qc/feats/{hydro,vege,vegeL}/{hydro,vege,vegeL}_p*_c*.parquet` | 现有样本证据（hydro 54 波段 / vege 14 波段 / vegeL 3 波段） | 34,612 行 ×2 层 |
| `data/swamp_qc/out/swamp_existing_v1_keep.parquet` + `qc/quarantine_swamp.csv` | 现有样本保留 / 隔离 | 保留 **24,390**（2,281 + 22,109）/ 隔离 **10,222** |
| `data/swamp_qc/lists/{candidates_rets,candidates_china}.parquet` + `out/candidate_gen_stats.json` | 遥感候选三源（A DW 淹没植被 / B WorldCover 90 / C GSW 季节水+S2 植被） | 去重后 **41,740** → 中国境内 **29,474**（A 7,708 / B 8,150 / C 13,616） |
| `data/swamp_qc/lists/densify_woody.parquet` | 复合体加密点（列 lon/lat/parent/parent_class/tag/point_id） | **9,584**（dens180 5,682 / dens181 3,902；3 km 内，实测最大 3.000 km） |
| `data/swamp_qc/lists/{osm_wetland.parquet, osm_candidates_china.parquet, osm_accepted_702.parquet}` + `osm_wetland_stats.json` | OSM 湿地要素台账（56,149）、境内目标子集（2,684）、采纳后新增 | 2,684（swamp 208 / 草本 2,476）→ 采纳 1,037 → **新增 702（木本 44 / 草本 658）** |
| `data/swamp_qc/out/swamp_samples_v1.{parquet,csv}` | **成品 v1**（existing 保留 + 候选 + 加密；列 point_id/lon/lat/class_swamp/type_src/source/prov/证据列） | **48,314**（14 木本 10,612 / 15 草本 37,702） |
| `data/swamp_qc/out/swamp_samples_v2.{parquet,csv}` | **成品 v2 = v1 + OSM** | **49,016**（10,656 / 38,360） |
| `data/swamp_qc/qc/quarantine_swamp_full.csv` | 隔离 + 被拒候选 + 被拒加密（含全部证据列） | **25,356**（现有 10,222 + 候选 8,315 + 加密 6,819） |
| `data/swamp_qc/out/{calibration_swamp.json, separation_report.json, stats_apply.json, stats_final.json, expand_gen_stats.json}` + `feats/aef/` | 冻结判据、分离度报告、两阶段统计、AEF64（2023 年度，v1 全部 48,314 + OSM 702 = 命中 **1.0000**） | — |
| `F:/lc_work/v31_exp/reports/swamp_qc_2026-10-13/`（4 文件） | 报告副本（预注册判据.md / calibration_swamp.json / separation_report.json / **stats_final.json = v1 版 48,314**，注意非最新） | — |
| 交付态 | `数据/单类样本的单独处理_2026.10.12/{木本沼泽,草本沼泽}/{2023/*.shp, 剔除清单.csv, 逐年池成员数.csv, README_样本文件说明.md}` | 木本 **10,656** / 草本 **38,360**（v2 口径；双坐标 SHP 回读 0.0、PASS） |

### 2.2 版本演进时间线

| 版本 | 日期 | 做法（一句话） | 点数 | 结论 / 为何被替代 |
|---|---|---|---|---|
| **V0 原始标签** | ≤2026-10 | 30 类码 180（木本沼泽，**GLC 无种子、属缺口类**，靠专题补样，见 **60** §1.2 `GAP_CLASSES`）/ 181（草本沼泽） | 8 年并集 4,968 / 29,644 | 沿用 FCS/专题惯例；**56** 实测木本沼泽层判读精度 **6.7%（1/15）**、草本 **0.571** |
| **V1 实验 C 湿地细分可分性**（**54**，四批 586 有效判读点） | 2026-10-09 | 三江 403 + FCS10 分层补 108 + WorldCover 树覆盖 40 + 本成品森林像元 40 点；z18 判读 | 草沼 244 / 湖河滩 53 / 裸地 21 / 木本 31 / 森林 10 … | 草沼 vs 草地 **AUC 0.708**、湖河滩 vs 裸地 **0.649**（<0.85 不可分）、水体 vs 草沼 **0.967**（可分）；P2 木本 vs 森林因"**四源皆失**"无法裁决 → 建议**湿地合并 1 类 + 水体独立** |
| **V2 样本集 v2 删层**（**56** §3） | 2026-10-10 | 按判读精度**删除 180 标签**（2023 池 2,543 点）；草本层保留 | v2 保留 1,503,331 / 2,060,577 = 73.0% | 木本沼泽层"近乎随机"（6.7%）被废弃；草本 0.571 中等 |
| **V3 全面质检 + 扩展（现行）**（**66**；v1–v6） | 2026-10-13 | 三层证据抓取（hydro/vege）→ 分离度报告 → 冻结判据 → 现有样本三态 QC + 堤岸审计 → 三路扩展（候选/加密/OSM）→ 组装 | 现有 34,612 → 保留 24,390；扩展采纳 23,924；**v1 48,314 → v2 49,016** | 冻结判据「点级淹水≥1 月 或（≥3 方位见水且平坦）」；木本现有样本隔离 54.1%（防护林问题被证实）；两处数据级修复（联合掩膜/哨兵污染） |
| **交付态 SHP** | 2026-10-13 12:25 | `s1_samples_to_shp.py` 导出双坐标 SHP（14/15 两类，各 2023 口径） | 10,656 / 38,360 | 回读 0.0、PASS（`shp_export_report.json`） |
| **待办** | — | 分型（14 vs 15）证据弱；**54** 建议合并为 1 类 + 水体独立（用户裁决）；M3 人工判读 | — | 见 2.6 |

### 2.3 各版本技术细节

#### 2.3.1 用户要求与问题定义（**66** 文首；预注册判据.md §一）

- 用户明确要求：**木本沼泽样本不得来自河道附近的防护林/自然岸林**；对现有样本过滤检验；不足则扩展；产出能支撑全国分类的两类点。
- 目标类必须满足：① 多年/季节性地表水或土壤饱和证据；② 低洼汇水位置（非堤岸）；③ 木本（180）或非木本低矮植被（181）。
- 主要混淆源：木本——**河岸防护林（堤岸高、行状、无年淹）、天然岸林（河漫滩高位）、普通森林（无水）**；草本——草地（无水）、**农田（水田：季节性淹水但为耕作）**、湖河滩地（无植被/裸滩）。

#### 2.3.2 证据层（v1 代码 = 唯一实现源）

| 层 | 内容（代码级） | 关键口径 |
|---|---|---|
| **S0 水频/月历史/方位** | `JRC/GSW1_4/GlobalSurfaceWater`：occurrence/seasonality/recurrence/max_extent **点值 + 150 m 邻域**（focal radius=5）＋ 300 m 邻域（radius=10）4 列；`MonthlyHistory` 2015-01~2021-12 的 12 个月位 `water==2` 频率（点值 + 150 m）；**8 方位**：max_extent 经 `translate` 取 N/S/E/W ±10 px（≈300 m）与 NE/NW/SE/SW ±7 px（轴向≈210 m、径向≈297 m） | 沼泽植被不是明水面 → **邻域与方位才是信号**；点级保留 `-1` 哨兵（"从未有水/无数据"），邻域一律 `unmask(0)` 后 focal |
| **S1 产品/光学** | DW 2023 全年均值 flooded_vegetation/water/trees/grass（点 + 150 m，`dw_fv_150…`）；WorldCover `Map` 类；MODIS MCD12Q1 2022 `LC_Type1`（11=永久湿地） | 独立佐证，不单独定案 |
| **S2 地形** | SRTM `dem`、`slope`、`relief500`（radius=17 ≈510 m）、`relief2000`（radius=67 ≈2010 m） | **低洼位势 → 排除堤岸**；relief2000 在交互式 API 下持续超时 → 候选/加密点降级用轻量层（见 2.3.5） |
| **S3 植被（SAR/光学）** | S1 GRD（IW、VV+VH、**不筛轨道**）：夏 2023-06~09、冬 2022-12~2023-03 均值；S2 NDVI/NDWI 三窗：春枯 03-01~05-01、夏 06-01~09-01、冬 2022-12~2023-02（SCL 剔 3/8/9、云量<60） | 淹水植被 C 波段双反弹（VV 高、VV−VH 大）；沼泽早绿/常湿（NDWI 高） |
| **S4 木本/草本候选分型** | DW trees（`dw_trees_150≥0.15`；WorldCover 90 源放宽至 **0.10**）；OSM `wetland=swamp` 标签 | 分型证据弱（见 2.6.1） |

抓取：hydro 54 波段、块 3000、scale=30；vege 14 波段、块 4000；vegeL 3 波段、块 5000；`--part/--nparts 7` 分片、逐块幂等、重试 6 次（v1 代码）。

**hydro 54 波段清单（v1 代码，逐源 `cat` 顺序）**：`occurrence/seasonality/recurrence/max_extent`（点级，`unmask(-1)`）＋ `m01…m12`（点级月水频，`unmask(-1)`）＋ 上述 4 列 + 12 月位的 **150 m** 邻域（`_150`，先 `unmask(0)` 再 focal）＋ `occ_300/seas_300/rec_300/max_300`（300 m 邻域 4 列）＋ `q_N…q_SW`（8 方位 max_extent）＋ `flooded_vegetation/water/trees/grass`（DW 点级）＋ `dw_fv_150/dw_water_150/dw_trees_150/dw_grass_150`（DW 150 m）＋ `wc`（WorldCover 类码）＋ `modis_lc`（MODIS 2022）。**vege 14 波段**：`dem/slope/relief500/relief2000` ＋ S1 夏冬 `VV/VH` 4 列 ＋ S2 三窗 `NDVI/NDWI` 6 列（S2/S1 的 `-9999` 哨兵本地转 NaN）。

**两个数据级修复的对账表（**66** §三.1–3.2）**

| 问题 | 病征 | 根因 | 修复 | 验收 |
|---|---|---|---|---|
| 联合掩膜陷阱 | 首轮 **71.6% 整行全空** | 44 波段联合图上任一波段被掩膜 → `sampleRegions` 丢整点；GSW 对"从未有水"像元整体掩膜，丢的恰是需要的干点负样本 | 逐源 `unmask(-1)` 再 `cat`（保留"从未有水"语义） | 非空率 **28.41% → 100%**；旧口径"点级波段交集"= 旧非空数 **21,268**（机理闭环） |
| 邻域哨兵污染 | `occ_150` 等出现 (−1,0) 混合值（**5.5%** 行） | `focal_mean` 作用在已 `unmask(-1)` 的图上 | 邻域一律 `unmask(0)` 后 focal；(−1,0) 按"部分无水"截断为 0 | v3 `derive()` 固化，冻结判据在修复后数据上定标 |

#### 2.3.3 四个关键发现（全部实测；**66** §三）

1. **`sampleRegions` 联合掩膜陷阱（数据级）**：首轮 44 波段联合图下 **71.6% 整行全空**；修复（逐源 `unmask(-1)` 再 cat，保留"从未有水"语义）后非空率 **28.41% → 100%**；机理闭环——旧口径"点级波段交集"精确等于旧非空数 **21,268**。
2. **邻域 focal 的哨兵污染（数据级）**：`focal_mean` 直接作用在 `unmask(-1)` 图上 → `occ_150` 出现 (−1,0) 混合值（**5.5%** 行）；修复：邻域一律 `unmask(0)` 后 focal，(−1,0) 按"部分无水"截断为 0（v3 `derive()`）。
3. **木本沼泽的水"在邻域不在点级"**：现有 180 点点级淹水月数中位 **0**、方位水中位 **1**（单侧）；181 草本点级 **3** 月、方位 **3**（环水）。→ 这既是"堤岸型"的判别签名，也说明**"水优先"掩膜抽不出木本沼泽**：实测 `WC 树覆盖 × GSW 季节性水 × 平坦` 全国采样**仅 1 个候选点**（`expand_gen_stats.json: n_sampled=1`，邻域膨胀版也近于空）。
4. **294 个 M3 缺口判读锚点不可用作真值**：水证据全为 0（与背景随机点同分布）、坡度中位 **6.7°**（坡地；`separation_report.json:anchor_木本 slope p50=6.74`）——与 **56** 的"木本沼泽层精度 6.7%"一致；本次改以**证据结构**（existing 180/181 vs 背景的可分性）定标。

#### 2.3.4 冻结判据与分离度（`calibration_swamp.json`；分离度报告后冻结）

阈值明细（`calibration_swamp.json:thresholds`）：

| 参数 | 值 | 含义 |
|---|---|---|
| `pt_flood_months_min` | **1** | 点级历史淹水月数下限（月水频阈值 `pt_month_min`） |
| `pt_month_min` | **0.143 = 1/7** | 月水频的最小非零值（MonthlyHistory 2015–2021 共 7 年） |
| `n_dir_min` | **3** | 8 方位 max_extent 计数下限（≥3 方位见水） |
| `relief2000_max` | **15 m** | 低洼位势上限（dem − focal_min @≈2010 m） |
| `slope_max` | **3°** | 平坦上限 |
| `levee_max300_min` | 0.3 | 堤岸签名：`max_300≥0.3`（配合点级零淹水） |
| `no_wet_max300_max` | 0.3 | 冻结后 **未被 apply 使用**（见 2.6.2） |

判据：`swamp_core = (pt_flood_months ≥ 1) OR (n_dir ≥ 3 AND relief2000 ≤ 15 m AND slope ≤ 3°)`

| 群体 | n | 通过率（`calibration_swamp.json: pass_rates` / `stats_apply.json`） |
|---|---|---|
| existing 180（木本沼泽） | 4,968 | **45.9%** |
| existing 181（草本沼泽） | 29,644 | **74.6%** |
| 背景·森林（含岸林/防护林） | 8,733 | **0.06%**（`bg_forest_pass=0.0006`） |
| 背景·草地 | 3,696 | **0.35%**（0.0035） |
| 背景·耕地 | 6,320 | **2.1%**（0.0212） |
| 背景·全部随机 | 40,000 | **6.1%** |
| M3 判读锚点（不可靠，仅记录） | 294 | **8.5%** |

分离度签名（`separation_report.json`，中位）：existing_180 点级淹水 0 月 / `n_dir` 1 / `relief2000` 7 m / `dw_trees_150` 0.13；existing_181 点级 3 月 / `n_dir` 3 / `relief2000` 8 m / `dw_trees_150` 0.08；背景·森林（远水 8,725 点）`occ150` 0 / `relief2000` 197 m / slope 21.2°。

**现有样本质检结果**（`stats_apply.json`、`stats_final.json`）：

| 类 | 总点数 | 保留 | 隔离 | 隔离构成 |
|---|---|---|---|---|
| 180 木本沼泽 | 4,968 | **2,281（45.9%）** | 2,687（54.1%） | `NO_WET_EVIDENCE` 2,237 + `LEVEE_LIKE` **450** |
| 181 草本沼泽 | 29,644 | **22,109（74.6%）** | 7,535（25.4%） | `NO_WET_EVIDENCE` 6,014 + `LEVEE_LIKE` **1,521** |

> 木本沼泽半数以上被隔离 = 直接回应用户关切：其签名（点从不淹水、水只在单侧、非低洼）正是**河道防护林/岸林**（**66** §五；隔离档 `qc/quarantine_swamp.csv`）。
> 保留点的木本证据率：`dw_trees_150≥0.15` 在保留 180 中仅 **28.1%**、保留 181 中 **20.2%**（`stats_apply.json: woody_frac_keep180/181`）——分型证据弱的结构性证据。

#### 2.3.5 扩展（三条路线 + 一条负结果）

1. **遥感候选（v2_swamp_candidates.py，子代理执行）**：三源 A/B/C（定义见 2.1 文件表下方表或 v2 代码头）→ 分 5°/2.5° 格用 `stratifiedSample` 采样（配额/格 ≤4,500，种子 13，scale 30；`Image.sample` 在全国框实测产出 ≈ N×掩膜占比且 N≥3e4 必超时，故弃用）→ 精去重（round(4)≈11 m 保序 + ≥1 km 贪心）→ **41,740 点**；中国境内 **29,474**；过冻结判据 **21,159**；热点前 8 个 5° 格（`candidate_gen_stats.json: grid5_top8`）：

| 5° 格 | 点数 | 对应湿地 |
|---|---|---|
| 130–135°E / 45–50°N | 4,992 | 三江平原 |
| 120–125°E / 30–35°N | 3,722 | 太湖口 |
| 110–115°E / 25–30°N | 2,867 | 洞庭 |
| 105–110°E / 20–25°N | 2,746 | 北部湾-海南 |
| 120–125°E / 40–45°N | 2,596 | 辽河 |
| 120–125°E / 35–40°N | 2,512 | 苏北-黄河口 |
| 115–120°E / 25–30°N | 2,183 | 鄱阳 |
| 100–105°E / 30–35°N | 2,060 | 若尔盖 |

分源通过率（分母=各自境内子集 A 7,708 / B 8,150 / C 13,616；两处已记录数相除）：**A 58.9%（4,539）/ B 45.0%（3,666）/ C 95.1%（12,954）**；采样台账 `candidate_gen_stats.json` 同时记录目标量 A/B/C = 72,000/49,000/49,000（末轮点名补扫若尔盖 X00、洞庭 X01）、`china_inside_frac_approx = 0.706`、`anomalies = []`、单账号（`udn4q4`）全程零失败。
2. **"水优先"掩膜（v4_swamp_expand.py）——负结果**：`WC 林地 × GSW max_extent≥1 × seasonality≥2 × slope≤3`（邻域膨胀版 max/10 px）全国分块采样，**仅 1 个候选**（`expand_gen_stats.json`）→ 该路线被证伪，记录在案。
3. **复合体加密（v4 系列）**：在现有保留点 3 km 内撒点 **9,584**（dens180 5,682 / dens181 3,902；parent 覆盖 6,113 个母点；实测距母点最大 3.000 km、中位 2.058 km）→ 过判据 **2,765**（木本 1,606 / 草本 1,159）。轻量层替代：D 条件以 **`relief500 ≤ 8`** 替代 `relief2000 ≤ 15`（半径 67 px 的 focal 在交互式 API 下持续超时；**66** §八.2；v5 代码注释）。
4. **OSM 湿地（h2b_osm_wetland.py；独立矢量源）**：Overpass `natural=wetland` + `wetland=*` 分格抓取 **56,149 要素**（`osm_wetland_stats.json`：way 48,017 / relation 7,555 / node 577；分格 2°、out geom，way/relation 取成员坐标均值）→ 中国境内 **9,622**（82.9% 落在 bbox 内境外=中亚/西伯利亚湖区）→ 目标子集 **2,684**（`swamp` 208 归 14；`marsh|reedbed|wet_meadow|fen|bog|无值` 2,476 归 15；`tidalflat|mangrove|saltmarsh` 等属他类排除）→ 过冻结判据 **1,289** → **植被守门**（`trees+grass≥0.15` 或 `flooded_veg≥0.10`，剔除大湿地质心落水 **252** 个）→ 采纳 **1,037** → 与成品 ≥1 km 去重 → **新增 702（木本 44 / 草本 658）**。

   抓取侧 wetland 取值构成（`osm_wetland_stats.json`，全球 56,149 要素）：`marsh` 10,094、`wet_meadow` 5,000、`swamp` 4,960、`bog` 3,952、`reedbed` 3,207、`tidalflat` 1,927、`mangrove` 1,438、`saltmarsh` 576、`fen` 389、空值 24,286，其余为大量单例自由文本（含"湿地公园""稻田"等噪声标签）——**这也是 OSM 只取 `swamp` 作木本独立分型、其余进草本并把潮滩/红树林/盐沼排除的原因**。

#### 2.3.6 分型与组装的规则（v5 代码）

- existing 点：用原 180/181 标签 → 14/15；加密点：继承母点类；候选点：`woody_ev = (dw_trees_150 ≥ 0.15) | (src=='B' & dw_trees_150 ≥ 0.10)` → 14，否则 15（**66** §七）。
- 每条记录带 `type_src`（existing / dens_parent / cand_woody / cand_herb / candB_woody / candB_herb）+ `source`（细化到源，如 `cand_herb_C`）+ `prov`（候选/加密点为 NaN，**未回填**）。

### 2.4 当前版完整流程（v1 → v3 → v2/v4 → v5 → OSM → v6）

1. **清单与背景**：`lists/existing_180181.parquet`（34,612）、`background_40k.parquet`（40,000，含森林/耕地/草地的 class_new 分层）、`judged_anchors.parquet`（294）。
2. **证据抓取（v1_swamp_fetch.py）**：hydro（54 波段，块 3000）/ vege（14 波段，块 4000）；候选与加密点复用同一脚本（`--list/--outdir`）；`vegeL` 轻量版（dem/slope/relief500 三列）用于候选/加密（避 relief2000 超时）。
3. **分离度报告（v3 --report）**：把 32 个特征在【锚点 31/253/10、existing 180/181、背景森林近水/远水、背景耕地、背景全部】上打 p25/p50/p75 → `out/separation_report.json`（供人工定阈；**先报告后冻结**）。
4. **冻结判据 + 三态判定 + 堤岸审计（v3 --apply）**：`calibration_swamp.json` 冻结后执行——保留 = `swamp_core`；隔离 = 非 core，其中 `LEVEE_LIKE`（`max_300≥0.3 且 点级淹水=0`）与 `NO_WET_EVIDENCE` 分档；输出 `swamp_existing_v1_keep.parquet`（24,390）、`qc/quarantine_swamp.csv`（10,222）、`background_scored.parquet`、`anchors_scored.csv`、`stats_apply.json`。
   - 哨兵语义：GSW/DW 的 `-1` → 证据计数归零、`gsw_valid` 记录有无 GSW 数据；邻域列按"部分无水"截断（v3 `derive()`）。
   - `pt_flood_months` = 点级月水频 ≥0.143 的月数；`n_flood_months` = 150 m 邻域月水频 ≥0.3 的月数；`w_spring/w_summer/w_win` = 各季 3 个月位 150 m 水频的最大值；`n_dir` = 8 方位 max_extent 之和（v3 代码）。
5. **候选生成（v2_swamp_candidates.py）** 与 **加密点生成（v4 系列）**、**水优先掩膜（v4_swamp_expand.py，负结果）**：见 2.3.5。
6. **组装（v5_swamp_assemble.py）**：同一冻结判据（候选/加密用 `relief500≤8` 变体）→ existing 保留 24,390 + 加密 2,765 + 候选 21,159 = **48,314**（by_class：14=10,612 / 15=37,702）；被拒 15,134 汇入 `qc/quarantine_swamp_full.csv`（连同现有隔离共 **25,356**；`stats_final.json`）。构成明细（`stats_final.json:by_type_src / by_source`）：

| `type_src` | 点数 | `source` 分解 |
|---|---|---|
| existing | 24,390 | existing_181 22,109 + existing_180 2,281 |
| cand_herb | 12,217 | cand_herb_C 8,127 + cand_herb_A 4,090 |
| cand_woody | 5,276 | cand_woody_C 4,827 + cand_woody_A 449 |
| dens_parent | 2,765 | dens180 1,606 + dens181 1,159 |
| candB_herb | 2,217 | cand_herb_B 2,217 |
| candB_woody | 1,449 | cand_woody_B 1,449 |
| （OSM，仅 v2） | 702 | osm_marsh_tag 658 + osm_swamp_tag 44 |

被拒明细（`stats_final.json:rejected_reasons`）：`CAND_NOT_SWAMP` 8,315 + `NO_WET_EVIDENCE` 8,251 + `DENS_NOT_SWAMP` 6,819 + `LEVEE_LIKE` 1,971 = **25,356**（其中前两项中的候选/加密部分 15,134 进交付态 `剔除清单.csv`）。
7. **OSM 并入（子代理线下执行，无独立脚本留档）**：h2b 抓取 → 境内目标子集 → 冻结判据 → 植被守门 → ≥1 km 去重 → **+702** → **v2 = 49,016**（10,656 / 38,360；`stats_final.json:v2`，2026-10-13 07:07 落盘）。
8. **AEF64 补齐（v6_swamp_aef.py）**：`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` 2023 年度、scale=10、块 2000、逐块幂等、逐 band `unmask(-9999)`；`feats/aef/aef_osm702.parquet` 为 OSM 新增点单独抓取；**命中 1.0000**（**66** §七；`logs/aef_osm.log` 首跑因 OSM 采纳表缺 `lon` 列报错、补 lon 后完成——工程留痕）。
9. **交付**：`s1_samples_to_shp.py` 出 14/15 两类双坐标 SHP（10,656 / 38,360；`shp_export_report.json` PASS），`剔除清单.csv`（木本 7,233 = DENS 4,076 + CAND 3,157；草本 7,901 = CAND 5,158 + DENS 2,743——**仅含被拒的候选/加密点，不含现有样本的 10,222 隔离点**，后者在 `qc/quarantine_swamp.csv/full`），`逐年池成员数.csv`（该点集在各年训练池中的成员数）：

| 年 | 木本沼泽在池（占交付 10,656） | 草本沼泽在池（占交付 38,360） |
|---|---|---|
| 2017 | 515（4.83%） | 15,695（40.92%） |
| 2018 | 521（4.89%） | 15,744（41.04%） |
| 2019 | 1,594（14.96%） | 20,497（53.43%） |
| 2020 | 1,626（15.26%） | 20,428（53.25%） |
| 2021 | 1,629（15.29%） | 20,396（53.17%） |
| 2022 | 1,198（11.24%） | 17,180（44.79%） |
| 2023 | 1,093（10.26%） | 16,163（42.14%） |
| 2024 | 1,090（10.23%） | 16,126（42.04%） |

> 读法：交付点集是**单份 2023 口径**成果；上表是"哪些既有年份的训练池包含该点"（新增的候选/加密/OSM 点不在任何历史年池中，故木本在池率低——**这些点在未补逐年 AEF 前只支撑 2023 年训练**，见 2.6.11）。

### 2.5 验证链

1. **特异性（判据的负样本检验）**：冻结判据在背景群体通过率——森林 **0.06%**（8,733 点）、草地 0.35%、耕地 2.1%、全部随机 6.1%（`calibration_swamp.json`；`stats_apply.json`）→ 判据几乎不误纳非湿地背景；这是"防护林结构性排除"的量化依据。
   - ⚠ 局限：背景"森林·近水"子群（`occ150≥10`）只有 **8 点**（分离度报告），故"岸林负样本"的直接检验体量很小；防护林证据主要来自**现有 180 中被隔离的 450 + 181 的 1,521 个 `LEVEE_LIKE` 点**及其签名统计，而非背景近水样本。
2. **锚点对照（仅记录，不作判据）**：M3 缺口判读 294 点通过率（`out/anchors_scored.csv`，本人复核）——木本 31 → **9.7%**、草本 253 → **7.5%**、非沼泽 10 → **30.0%**（三群均与"通过率应显著高于背景"的预注册期望相反；`stats_apply.json` 只落盘总通过率 8.5%）。`separation_report.json:anchor_*` 表明三群水证据全为 0（`occ150`/`pt_flood_months`/`n_dir` 中位 0–0.5）、木本锚坡度中位 **6.7°** → **确认该批锚点不可用作真值**（**66** §三.4）。
3. **数据级修复的机理闭环**：联合掩膜陷阱的旧/新口径精确对账（旧非空 21,268 = 旧口径点级波段交集；非空率 28.41%→100%）；哨兵污染行占比 5.5% 的量化与截断规则（**66** §三.1–3.2）。
4. **AEF64 覆盖**：v1 全部 48,314 + OSM 新增 702，命中 **1.0000**（**66** §七）。
5. **预注册门（G1–G6）**：G1 现有点全部三态、G2 键唯一、G3 证据非空率 ≥90%、G4 隔离点带理由码、G5 判据在锚点上方向一致、G6 母库/子集 MD5 前后一致（**66** 预注册判据.md §六；`stats_apply.json` 落盘了 G1/G2/G3/G4 的可核数据、G5 记录为"不可靠锚点仅记录"）。本次未见单独的门 JSON 汇总文件（工程缺口，见 2.6）。
6. **交付几何回读**：两类 SHP 逐文件 `roundtrip_maxdiff=0.0`、PASS（`shp_export_report.json`）。

### 2.6 已知限制（诚实清单）

1. **木本/草本分型弱**：DW trees 在两类间的分离极小——`separation_report.json` 中位为 **0.13（180）vs 0.08（181）**；**66** §八.1 写作 "0.08 vs 0.10"（数字与报告不一致，见文末矛盾）；**54** 的湿地分型轴 AUC 0.708 亦不可分 → **分型仅供参考**；若生产侧可接受，建议按 **54** 的"湿地合并 1 类 + 水体独立"口径使用，或对 14/15 做合并报告。
2. **判据实现偏离预注册**：预注册写的是三态（keep/quarantine/**flag** 灰带，W=1 或 D 缺 → flag 保留），v3_apply 落地为**二态**（`swamp_core` vs 隔离，理由码两档）——无 flag 中间态；`calibration_swamp.json` 的两个阈值 `levee_max300_min / no_wet_max300_max` 在 `apply` 中只用了前者。
3. **D 条件的层替代**：候选/加密点用 `relief500≤8` 替代 `relief2000≤15`（GEE 交互式超时）；v5 注释给出"保留点中等价性 ~0.8"的量级说明，未做正式等价性检验 → 候选/加密点的"低洼"判定与现有样本**口径不完全一致**。
4. **`prov` 未回填**：候选/加密/OSM 新增点的省归属为 NaN（**66** §八.3；可用 `k19_provattrib` 一次补齐）——本版交付 SHP 的 `prov` 对新增点为缺失值。
5. **OSM 分型标签本身是众包质量**（中国境内 `swamp` 仅 208 个；**66** §八.4）：已过同一冻结判据 + 植被守门，视为**补充点**（`type_src=osm_*_tag` 可筛）。
6. **v2 并入 OSM 的步骤无脚本留档**（子代理线下执行）；同批的 densify 生成器也不在 `code/`（仅 `v4_swamp_expand.py` 留下负结果台账）→ 全链路复现缺两个环节的脚本原件。
7. **现有样本隔离量很大**（木本 54.1%、草本 25.4%）：本版**不区分**"标签错"与"真实无水沼泽（如泥炭藓沼泽/高位沼泽）"——判据只认"水/低洼"证据，可能误杀无水型沼泽；且 `LEVEE_LIKE` 采用的 `max_300≥0.3` 阈值未做敏感性扫描（**66** §八仅列"判据保守"立场）。
8. **水田混淆未在判据内处理**：预注册把"农田（水田）"列为草本沼泽的必须排除源，但冻结判据不含耕地/作物层（背景耕地通过率 2.1% 是唯一旁证）→ 草本沼泽中水田混入风险未量化。
9. **交付态 `剔除清单.csv` 不完整**：只含被拒候选/加密（15,134），**不含现有样本的 10,222 隔离点**；README 的表述（"被冻结判据隔离的点"）与文件内容不符（见文末矛盾）。
10. **报告副本滞后**：`reports/swamp_qc_2026-10-13/stats_final.json`（05:44）为 **v1 的 48,314 版**，不含 OSM；最新 v2 数字（49,016）在 `data/swamp_qc/out/stats_final.json`（07:07）——引用时须指明路径。
11. **AEF 只到 2023 单年**：v6 只抓 2023 年度嵌入；本批点若要进入 2017–2024 逐年训练，需按年补 AEF（同 **63** 的逐年做法），本版未做。

### 2.7 文件 → 版本对应表

| 版本/阶段 | 文件 | 说明 |
|---|---|---|
| 判据 | `data/swamp_qc/预注册判据.md`；`data/swamp_qc/out/calibration_swamp.json` | 先写死再跑；分离度报告后冻结（含阈值与通过率） |
| V3 分离度 | `data/swamp_qc/out/separation_report.json` | 9 群 × 32 特征 p25/p50/p75（锚点/existing/背景 4 类） |
| V3 现有 QC | `data/swamp_qc/out/swamp_existing_v1_keep.parquet`、`qc/quarantine_swamp.csv`、`out/background_scored.parquet`、`out/anchors_scored.csv`、`out/stats_apply.json` | 保留 24,390 / 隔离 10,222（含 LEVEE_LIKE 分档） |
| 扩展-候选 | `data/swamp_qc/lists/{candidates_rets,candidates_rets_raw,candidates_china}.parquet`、`out/candidate_gen_stats.json`、`feats/{hydro_cand,vege_cand,vege_candL}/` | 41,740 → 境内 29,474 → 采纳 21,159 |
| 扩展-加密 | `data/swamp_qc/lists/densify_woody.parquet`、`feats/{hydro_dens,vege_dens,vege_densL}/`、`out/expand_gen_stats.json`（水优先负结果） | 9,584 → 采纳 2,765 |
| 扩展-OSM | `data/swamp_qc/{osm_wetland_stats.json, osm_wetland_merged.json}`、`h2b_osm_wetland.py`、`lists/{osm_wetland.parquet, osm_candidates_china.parquet, osm_accepted_702.parquet}`、`feats/{hydro_osm,vege_osm,aef/aef_osm702.parquet}` | 56,149 → 9,622 → 2,684 → 1,037 → 新增 702 |
| 成品 v1 | `data/swamp_qc/out/swamp_samples_v1.{parquet,csv}` + `qc/quarantine_swamp_full.csv` + `out/stats_final.json`（reports 副本同版） | 48,314（10,612 / 37,702）；被拒 25,356 汇总 |
| 成品 v2 | `data/swamp_qc/out/swamp_samples_v2.{parquet,csv}` + `out/stats_final.json`（data 侧最新，含 `v2` 段） | 49,016（10,656 / 38,360） |
| AEF | `data/swamp_qc/feats/aef/aef_p*_c*.parquet`（v1 全量）+ `aef_osm702.parquet` | 2023 年度，命中 1.0000 |
| 代码 | `code/v1_swamp_fetch.py` → `v2_swamp_candidates.py` → `v3_swamp_qc.py` → `v4_swamp_expand.py` → `v5_swamp_assemble.py` → `v6_swamp_aef.py`；`data/swamp_qc/h2b_osm_wetland.py` | 抓取→候选→判据/QC→加密/负结果→组装→特征；（densify 生成器与 OSM 并入脚本未留档） |
| 交付态 | `数据/单类样本的单独处理_2026.10.12/{木本沼泽,草本沼泽}/{2023/*.shp, 剔除清单.csv, 逐年池成员数.csv, README_样本文件说明.md}` | v2 口径 10,656 / 38,360，双坐标 SHP PASS |

---

## 附：文档间矛盾 / 需在合稿时裁决之处

| # | 位置 | 矛盾内容 | 本稿采用 |
|---|---|---|---|
| 1 | **65** §3.2 正文 vs `stats.json:ed_switch` / `report.md` | 建议改判四数：doc 写 11,452/12,939/6,906/2,908；stats 写 **12,375/14,345/7,220/265**（两者和均 34,205） | `stats.json`（机器产物）为准 |
| 2 | **65** §3.3 表 vs `calibration.json` / `report.md` | 常绿/落叶中位：doc 写 z2 96/265、z3 155/386、z4 281/351、z5 299/356；calibration 写 **82.8/288.3、70.0/471.0、70.1/562.5、49.2/605.6**；全国缺省 thr/gray doc 写 326.0/56.0，stats 写 **326.1/56.5** | `calibration.json` 为准 |
| 3 | **65** §3.2 vs `stats.json:by_prov` | 隔离集中省：doc 四川 5,527 / 陕西 4,223 / 湖北 4,006；stats 为 **5,564 / 4,767 / 4,167** | `stats.json` 为准 |
| 4 | 交接常识 vs **65** §一 | AGENTS.md 待办 1 写"真缺口仍是 M3 独立人工验证点的判读"（3,210+863+403 点待判），本次木本质检用的是 **294 个 M3 缺口锚点**，且**66** 判定其不可靠——两处"缺口/锚点"不是同一批、易混；合稿时需统一命名（M3 通用/仲裁点 vs M3 缺口判读锚点） | 分开表述，注明批次 |
| 5 | **66** §二 vs v1 代码 | 证据层描述为"DW…点+150 m"，代码 `focal_mean(radius=15)` 在 DW 10 m 原生网格上确为 150 m（一致）；但"8 方位（±300m/±420m 对角线）"与代码不符——代码为 4 正方向 ±10 px（≈300 m）+ 4 对角 ±7 px（轴向 ≈210 m、径向 ≈297 m）；v1 脚本 docstring 又写"±300m/±600m" | 以 v1 代码为准，措辞已按代码改写 |
| 6 | **66** §八.1 vs `separation_report.json` | 分型证据写作 "DW trees 在两类间仅 0.08 vs 0.10"；报告现有 180/181 的 `dw_trees_150` 中位为 **0.13 vs 0.08**（0.10 未见于任何落盘数据） | 报告值 0.13/0.08 为准，并保留 doc 原数作对照 |
| 7 | **66** §六.2 vs 数据 | "加密 …（2,211 个复合体）"；`densify_woody.parquet` 的 `parent` 去重后为 **6,113 个母点**（9,584 点，tags dens180 5,682/dens181 3,902）——"复合体"若指聚类后簇数则与"母点数"口径不同，未见定义文件 | 两者并列，标注口径不明 |
| 8 | 交付 README vs 剔除清单内容 | `木本沼泽/README_样本文件说明.md` 称剔除清单="被冻结判据隔离的点"，实际文件只含被拒候选/加密（木本 7,233=DS 4,076+RC 3,157；草本 7,901），**不含现有样本 10,222 隔离点**（后者只在工作区 `qc/quarantine_swamp*.csv`） | 以文件实际内容为准，README 表述需修正 |
| 9 | 报告副本 vs data 台账 | `reports/swamp_qc_2026-10-13/stats_final.json`（05:44）是 **v1/48,314** 版；`data/swamp_qc/out/stats_final.json`（07:07）才是 v2/49,016 版；**66** §七同时给出两个版本数字但未标注两文件的时差 | 引用时按路径+时刻区分 |
| 10 | 判据实现 vs 预注册 | 预注册三态（含 flag 灰带）与 v3 `--apply` 的二态实现不一致；`no_wet_max300_max` 阈值冻结后未被使用 | 事实并记，属实现偏离而非数字冲突 |
| 11 | **56** §3 ② vs 现样本 | doc 56 的 v2 处理"**删除 180 标签 2,543 点**"（2023 池）；本次质检的 8 年子集仍含 180（并集 4,968、2023 子集 2,638）——两者是**不同训练池文件**（v2 生产池 vs `年度子集_含稀有类`），若不注明会读成"删了又出现" | 明确指向不同文件；合稿时需一句话交代池的世代 |
| 12 | 空间/时间规模表述 | **67** §7 说本轮 14 类清洗后"训练池重建：林地/灌木/沼泽/园地取各专项 v 版"，即本两节的样本**尚未合入新训练池**；而 **65**/**66** 以"当前样本"口吻交付 → 合稿时需说明两节产物在训练池中的接入状态（未接入/计划接入） | 加注"质检视图，训练池重建时接入" |
