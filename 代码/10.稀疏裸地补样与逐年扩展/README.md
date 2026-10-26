# 代码说明 — 稀疏植被 / 裸地 东北·东部补样与 2017–2024 逐年扩展

> 归档自 `F:/lc_work/bare_sparse_v3/code/`（2026-10-25）。全部脚本均为独立 `python 脚本.py` 运行。
> 工作区（大产物）：`F:/lc_work/bare_sparse_v3/`；判据：`.../spec/预注册判据.md`；报告：`技术文档/85` + `实验_2026-10_稀疏裸地补样/`。

## 一、链路顺序

| 步 | 脚本 | 作用 | 关键输出 |
|---|---|---|---|
| 0 | `h0_health_all.py` / `h0c_probe_accounts.py` | 账号健康复检 / 实测可用编队（含从 `_任务登记.md` 挖项目号） | `logs/health_all.json`、`config/accounts_good.json` |
| 1 | `h1_harvest_fcs10.py` | 从本地 GLC_FCS10 2023（10 m）采集候选：3×3 全纯 + 9×9 同类≥0.70、0.025° 格配额、最小间距 | `data/candidates/cand_<region>.parquet` |
| 1b | `h1b_rebalance.py` | 东区分窗配额再平衡 | `*_rebalanced.json` |
| 1c | `h1c_holdout_filter.py` | 剔除评估留出（25 km Albers 块 + 五窗评估点 100 m） | `*_holdout.json` |
| 1d | `h1d_purity_recheck.py` | 按交付坐标逐点重读 3×3/9×9 复核 | `*_purity.json` |
| 2 | `h2_evidence.py` | 2023 证据：S2 夏 NDVI/NDWI、冬 NDVI；DW 9 波段；WC；GSW；SRTM | `data/evidence/*.parquet` |
| 3 | `h3_rules.py` | doc67 口径裁定 keep/quarantine/flag | `data/final/cand_qc.parquet` |
| 3c | `h3c_spacing_fix.py` | 球面距离重做 250/300/200 m 去重与类内间距 | `cand_qc_fixed.parquet` |
| 4 | `h4_aef.py` | AEF64（可 `--year` 逐年） | `data/aef*/*.parquet` |
| 5 | `h5_verify.py` | 栅格回读 + 证据双路 + AEF 完备性 | `verify.json` |
| 6 | `h6_deliver.py` | 2023 交付（GPKG 含 AEF64 + SHP + parquet + 分省表 + README） | `delivery/` |
| 6b | `h6b_china_filter.py` | CTAmap 国界过滤（保留 ≤10 km 边界带） | `cand_qc_china*.parquet` |
| 7 | `h7_pool_v3.py` | 池 v3 = v2 + 补样（50 m 去重） | `train2023_v3.parquet` |
| 9 | `h9_fcs10_e130_extract.py` | E130–E135 带 zip 解压与验收 | E 盘标准目录 |
| 11 | `h11_yearly_evidence.py` | 逐年动态证据（S2 夏/宽窗/冬 + DW，云量<60） | `data/evidence_yearly/<year>/` |
| 12 | `h12_yearly_rules.py` | 逐年判定（锚 2023 + 缺证据继承 + 三项护栏） | `yearly_panel.parquet`、`yearly_valid_window.parquet` |
| 13 | `h13_build_yearly.py` | 逐年层 + 与 `年度子集_含稀有类/r7_train_<year>` 合并 | `data/yearly/topup_<year>.parquet`、`data/pools/train_<year>.parquet` |
| 14 | `h14_verify_yearly.py` | 逐年核验（完备/跨年/双路） | `verify_yearly.json` |
| 14b/c | `h14b_clcd_check.py` / `h14c_clcd_values.py` | CLCD 逐年旁证 / 逐点 CLCD 值（供护栏） | `clcd_check.json`、`clcd_values.parquet` |
| 15 | `h15_run_yearly_fetch.py` / `h15b_dispatch.py` | 多账号并行抓逐年 AEF/证据；缺块轮换重试 | `logs/yz/` |
| 16 | `h16_visual_audit.py` | S2 真彩色接触表（1 km 视场 + 十字，12 点/类/年） | `reports/visual_audit/` |
| 17 | `h17_dist.py` | 逐年省级分布表 | `province_yearly.csv`、`distribution.md` |
| 18 | `h18_report_yearly.py` | 验收报告（逐年量/跨年/双路/CLCD/目视/精度） | `yearly_support_2017_2024.md` |
| 19 | `h19_accuracy_yearly.py` | 逐年分层随机抽样精度（≥60/类/年，独立参考 CLCD） | `accuracy_yearly.json` |
| 20 | `h20_publish_class_yearly.py` | 写入项目类目录：旧样本备份 → 旧+新合并 → 逐年 GPKG | 类目录 `<类>/<year>/<类>_<year>_WGS84.gpkg` |
| — | `h8_report.py` / `h8b_areas.py` | 2023 交付汇总报告 / 分窗 FCS10 面积 | `reports/*.md`、`fcs10_areas.json` |

## 二、关键参数（判据见 `spec/预注册判据.md`）

- 采集：FCS10 2023 10 m；3×3 全纯；9×9 同类 ≥0.70；0.025° 格稀疏≤2 / 裸地≤1；最小间距 300/200 m；与既有同类样本 250 m 去重（球面复核）。
- 分窗：北区 113–126°E/41.5–50.5°N（科尔沁/松嫩/呼伦贝尔/浑善达克）；东区 107–125°E/18–38.5°N；ne130 130–135°E/41.5–54°N。
- 逐年：锚 2023；NDVI 夏→宽窗→冬兜底；缺证据继承最近年；三项护栏 — CLCD 当年不透水剔、稀疏 DW crops≥0.35 剔、裸地 DW bare≥0.25。
- 账号：`config/pids_mined.json`（1,317 仓）；`--pid` 直连；`h15/h15b` 16 进程并行 + 轮换重试。

## 三、注意

- 所有路径常量写在脚本头部（`BASE` / `DATA` / `PROV_SHP` / `CLCD`），换机时只改这些。
- `h11/h4` 需要 GEE（代理 `socks5h://127.0.0.1:7890`）；`h1*`、`h12`、`h13`、`h20` 为纯本地。
- 大产物（年度池、证据块）留在 `F:/lc_work/bare_sparse_v3/`，未入项目库。
