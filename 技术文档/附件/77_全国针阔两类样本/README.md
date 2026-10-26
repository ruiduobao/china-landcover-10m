# 附件：77 全国针阔两类样本生产结果 · 证据清单

> 对应正文：[`../../77_全国针阔两类样本生产结果.md`](../../77_全国针阔两类样本生产结果.md)
> 归档时间：2026-10-17。原始工作区：`F:/lc_work/nb_national_2026/`（本目录是其核心证据副本，**不含大 parquet 与栅格**）。
> 所有数字均可在本目录 json/csv 中复核，无需重跑。

## 一、样本与统计

| 本目录文件 | 原始路径 | 内容 |
|---|---|---|
| `stats_v1.json` | `labels/stats_v1.json` | 最终样本统计：55,712 点、76 瓦、23 省、aspect_band 构成、terrain 中位、OR（FCS10 2.186 / 旧 4.749）、vs_old |
| `oldnew_stats.json` | `labels/oldnew_stats.json` | 与旧标签不一致率 0.20111（n=51,488）+ 方向（N→B 6,138 / B→N 4,217）+ 分省 |
| `panel_lineage.json` | `labels/panel_lineage.json` | **面板标签来源实测**：vs FCS30D 0.9869 / vs FCS10 0.7502（证明 `annual/` 仍是旧标签） |
| `or_by_prov.json` | `labels/or_by_prov.json` | 分省 OR（内蒙 5.79 … 广东 0.20）+ MH 分层（FCS10 2.3204 / 旧 4.7388） |
| `final_summary.json` | `labels/final_summary.json` | **全部关键数字的机器可读汇总**（样本/OR/覆盖/不一致/两臂/8年/R2/CGLS） |
| `annual_availability_fcs10.csv` | `labels/annual_availability_fcs10.csv` | 8 年逐年可用（旧标签口径，含 fcs10 叶型列对照） |
| `判据_针阔两类样本v1.json` | `labels/判据_针阔两类样本v1.json` | 跑前冻结的判据（G1–G10、采样规则、tier 规则） |
| `needle_broad_map.json` | `exp/two_class_retrain/needle_broad_map.json` | **冻结码表**（22 类 = 19 非森林 + 31/32/33；旧 4/5/6/7/8 降属性列） |

## 二、审计 G1–G8

| 本目录文件 | 原始路径 | 结论 |
|---|---|---|
| `g1_labels.json` | `audit/out/g1_labels.json` | **G1 PASS**（1,000/1,000 = 1.000）；卫生 0 重复/0 NaN/0 越界 |
| `g2_coverage.json` | `audit/out/g2_coverage.json` | 76 瓦 / 23 省；72 瓦两类共存；22/23 省两类共存；G3/G4 按全国门槛 FAIL（v1 先行批） |
| `g3_aspect_VOID.json` | `audit/out/g3_aspect.json` | ⚠ **作废件**（文件名已标 VOID）：OR=0.988 因 1 km SRTM 地形错位不可用，保留仅供追溯 |
| `g4_panel.json` | `audit/out/g4_panel.json` | **G5 PASS**（8 年可用 0.954–0.988）；**G6 PASS**（面板一致 1.0，按构造） |
| `g5_oldnew.json` | `audit/out/g5_oldnew.json` | 不一致率 0.2011 + 分省/分区明细（本项即 77 §4.5 的直接来源） |
| `g6_leakage.json` | `audit/out/g6_leakage.json` | **G7 PASS**：精确键 0 命中；最近点中位 2.03 km |
| `g7b_hygiene.json` | `audit/out/g7b_hygiene.json` | 549,789 重训点与 1,274,776 面板点的卫生检查（0 NaN AEF / 0 越界） |
| `or_crosscheck_76tiles.json` | `audit/out/or_crosscheck_76tiles.json` | **全量 76 瓦坡向核对**：FCS10 **2.046** [1.928, 2.172] / 旧 **4.918** [4.614, 5.243]；151,033 坡点、FCS10 有效 38,509；含 `conclusion` 字段 |
| `or_crosscheck_549k.json` | `audit/out/or_crosscheck_549k.json` | **本 agent 的 2×2 复核**（549,789 点 × 两套地形 × 两标签）：NASADEM FCS10 2.060 / 旧 5.120；1 km FCS10 1.062 / 旧 0.999；band 一致率 0.5397 |
| `coverage_114.json` | `audit/out/coverage_114.json` | 114 已切瓦；全国 131,099 点（82.3%）在其中；覆盖 85 个瓦 |

> `or_crosscheck_samepoints_raw.json` 是最早一版单人核对（FCS10 格全 0、OR=null），**仅作失败留痕**，结论以 `or_crosscheck_76tiles.json` 与 `or_crosscheck_549k.json` 为准。

## 三、两臂重训

| 本目录文件 | 原始路径 | 内容 |
|---|---|---|
| `two_arm_metrics.json` | `exp/two_class_retrain/two_arm_metrics.json` | **两臂全指标**（vote3 + 3 种子 + 分区域/分省/分瓦）。ArmA macro-F1 0.7547 / ArmB 0.8539；OR 预测 5.231 / 2.371，参考 2.060 |
| `t1_prep_stats.json` | `exp/two_class_retrain/t1_prep_stats.json` | 549,789 点构造：76 瓦、FCS10 缺失 0%、未映射 0%、两臂标签分布 |
| `mixed_policy_eval.json` | `exp/two_class_retrain/mixed_policy_eval.json` | 混交 4,133 点处置（20 瓦；median needle share 0.3612；31 得 1,558 / 32 得 2,575） |

> 大文件未归档：`points_76.parquet`（50 MB）、`oof_preds_all76.npz`（3.9 MB，**可复原全部两臂指标**）、`mixed_points_dominant.parquet`。

## 四、多方法 A–G

| 本目录文件 | 原始路径 | 裁定要点 |
|---|---|---|
| `m1_vote.json` | `labels/methods/m1_vote.json` | A 投票：strict 62,946（针 6,098）；label↔FCS10 0.8025；**不采纳**（覆盖仅 13.5%） |
| `m2_pheno.json` | `labels/methods/m2_pheno.json` | B 物候：B1 3,920 / B2 2,776 / B_any 10,118（2.171%）；F 时序稳定 90.44% 标签稳定；**作降权信号** |
| `m3_gee.json` | `labels/methods/m3_gee.json` | C/E CGLS 抽样 20,000 点（decided 13,227）、Hansen 仅 485；**GEDI 未测** |
| `m3_cgls_vs_fcs10.json` | `labels/methods/m3_cgls_vs_fcs10.json` | CGLS vs FCS10：n=6,648，一致率 **0.6995**，针叶 P 0.590 / R 0.531 / F1 **0.559** |
| `m4_eco.json` | `labels/methods/m4_eco.json` | D 生态规则：D_implausible 36,713（7.876%）；D1 广东 2,540 / D6 黑龙江 20,265；**作合理性校验** |
| `m5_anchor.json` | `labels/methods/m5_anchor.json` | G 锚点 36,529 / 自举 358,902；CV AUC 0.9937/0.9987 **（循环论证）**；anchor OR 0.343 方向与独立核对相反 → **不采纳** |
| `m7_structure.json` | `labels/methods/m7_structure.json` | **C1 郁闭/疏闭 AUC = 0.500（证伪）**；C2 邻域 AUC 0.9634（空间自相关）；C3 GEDI 未测/CGLS 部分 |

## 五、8 年面板

| 本目录文件 | 原始路径 | 内容 |
|---|---|---|
| `annual_presence_summary.json` | `annual/presence_summary.json` | 8 年可用（0.9591–0.9927）、S2 证据等级、AEF 零缺失 |
| `annual_R2_gate.json` | `annual/R2门控模拟.json` | R0 跳变 0.0365 → **R2 0.0054**（准确率 0.9322 → 0.9374） |
| `annual_判据_针阔八年继承.json` | `annual/判据_针阔八年继承.json` | 继承规则 R1–R5（跑前冻结） |
| `annual_crossyear.json` | `annual/跨年一致性分析.json` | C1 逐年一致率 0.9284–0.9353；C2 OOF AUC 0.9776–0.9844；C3 跳变 |

## 六、FCS10 瓦片化

| 本目录文件 | 原始路径 | 内容 |
|---|---|---|
| `fcs10_verify_76.json` | `fcs10_dl/verify_76.json` | 切瓦核验：`transform_equal=true`、`diff_pct=0`、`real_diff=0` |
| `fcs10_state_block_check.json` | `fcs10_dl/state_block_check.json` | 5° 块完整性体检（抽样全 ok） |

---

## 复核指引（三条最快路径）

1. **核 OR**：打开 `or_crosscheck_76tiles.json` → `fcs10.OR=2.046`、`old.OR=4.918`、`conclusion`；再对照 `two_arm_metrics.json` → `arms.B.vote3.or_ref=2.05998`（独立路径互证）。
2. **核两臂**：`two_arm_metrics.json` → `arms.A.vote3` / `arms.B.vote3`（`macro_f1_2`、`needle.precision`、`pred_needle_share_of_forest`、`or_needle_broad`）。
3. **核面板缺口**：`panel_lineage.json` → `panel_vs_fcs30d=0.9869` vs `panel_vs_fcs10=0.7502`。

## 七、全国 v2 扩容（2026-10-17 补归档，对应正文 §12–§15）

| 本目录文件 | 原始路径 | 内容 |
|---|---|---|
| `stats_v2.json` | `labels/stats_v2.json` | **全国 v2 抽样全统计**：110,653 点、248 瓦、33 省、208 瓦两类齐、候选 2,024,507→429,023、泄漏剔除 58,144、24.9 min |
| `stats_v2_by_prov.csv` | `labels/stats_v2_by_prov.csv` | 分省针/阔/混点数与针叶占比（新疆 0.893 … 台湾 0.005） |
| `stats_v2_tiles.csv` | `labels/stats_v2_tiles.csv` | 逐瓦点数与两类计数（248 瓦） |
| `or_by_prov_v2.json` | `labels/or_by_prov_v2.json` | **三口径 OR**：样本 2.183 / 人口 1.908 / MH 1.903（616 层）+ 分省（东北 3.146、云南 0.835） |
| `oldnew_stats_v2.json` | `labels/oldnew_stats_v2.json` | **全国旧/新标签**：不一致 17.90%（针→阔 11.3% > 阔→针 6.7%）+ 按旧来源分层（FCS10 系 0.999–1.000 vs fcs30d 0.8095）+ by_eco / by_tier |
| `oldnew_by_prov_v2.csv` | `labels/oldnew_by_prov_v2.csv` | 分省一致率（福建 0.6205 … 台湾 0.9924） |
| `south_needle_fix.json` | `exp/two_class_retrain/south_needle_fix.json` | **南方针叶三方案全败**：S1 权重 0.400/0.003 → 0.261/0.009 → 0.184/0.014；**S2 阈值四档 0.667/0.003 → 0.191/0.006 → 0.163/0.020 → 0.137/0.131**；**S3 南方单独建模输出 0 个针叶点**。诊断：真针叶 666 点 / 1.66% |
| `t6_scores_curve.json` | `exp/two_class_retrain/t6_scores_curve.json` | **可分性定量证据**：南方 AUC **0.6992** vs 全国 **0.9495**；P(needle) 均值 针 0.0504 / 阔 0.0206；召回目标反推阈值曲线（召回 0.30 → precision 0.1008） |
| `presence_summary_fcs10.json` | `annual/presence_summary_fcs10.json` | **FCS10 八年面板（最终配方口径，109,260 纯林点 × 8 年 = 874,080 行）**：可用率 **0.9294–1.0015**（G5 全过）；含非林年 4,312–5,381、无 AEF 0–7,058、QC 排除 0–897、物候证据率 0.0784–0.3034；G6 恒等式 1.0 已标注 |
| `每年可用点数_fcs10.csv` | `annual/每年可用点数_fcs10.csv` | 每年 × 针/阔 的可用点数与可用率（正文 §14 第二张表的来源） |
| `分省平衡_fcs10.csv` | `annual/分省平衡_fcs10.csv` | 2023 可用点的分省针/阔平衡画像 |
| `presence_summary_a0_fcs10.json` | `annual/presence_summary_a0_fcs10.json` | 同上（a0 直出件，字段更全：`no_aef_record`/`aef_incomplete`/`ok_downweight`） |
| `annual_availability_fcs10.csv` | `labels/annual_availability_fcs10.csv` | 8 年逐年可用宽表（含旧标签叶型对照列） |

## 八、v2 复核指引（三条）

1. **核全国样本量**：`stats_v2.json` → `n_points=110653`、`tiles_with_both_pure=208`；分省明细对 `stats_v2_by_prov.csv`。
2. **核 OR 未归零**：`or_by_prov_v2.json` → `sample.OR=2.1833`、`population_all_candidates.OR=1.9076`、`mh_tile_elevband.OR=1.9034`；**三者同向 → 不是抽样偏差**。
3. **核"南方清零是标签真相"**：`oldnew_stats_v2.json` → `by_eco.华东.new_needle_share=0.0803` vs `old_needle_share=0.2525`，再对 `south_needle_fix.json` → `reference.true_needle_n=666`。

## 九、宁夏诊断（对应正文 §16）

本目录无独立文件；数据源为 `two_arm_metrics.json` → `strata.province.宁夏回族自治区` 的 `armA` / `armB`（纯林参考 286 点：针 127 / 阔 159）。Bootstrap 脚本未单独落盘（一次性计算，参数与结果见正文 §16：20,000 次、seed 20261017、CI95 Δmacro-F1 [−0.0930, +0.0288]）。

## 十、G3 补样通道实测（对应正文 §18）

| 本目录文件 | 原始路径 | 内容 |
|---|---|---|
| `stats_v2fill.json` | `labels/stats_v2fill.json` | **直撒点补样结果**：18 瓦 / 6,386 点（针 3,086 / 阔 3,167 / 混 133）/ 3.2 min / 卫生全 0；逐瓦 `n_added` 与 `forest_km2` |
| `s3_fill_fcs10_direct.py` | `labels/code/s3_fill_fcs10_direct.py` | 补样脚本（绕过旧训练池；逐行带即时比例抽稀，峰值 <0.3 GB） |

**结论（正文 §18.3–§18.4）**：补出的 6,386 点中 **6,265 点（98.11%）在中国境外**，境内仅 121 点；
18 瓦合计 FCS10 森林 243,834 km²，**境内仅约 698 km²（0.29%）** → G3 缺口对中国分类实质无影响，
按"中国境内森林覆盖"口径 G3 实为 ≈99.8% PASS（改判据须走预注册，本轮仍标 FAIL）。

**未归档**：`needle_broad_direct_v2fill.parquet`（0.2 MB，98% 为境外点，未并入 v2 主样本）。
