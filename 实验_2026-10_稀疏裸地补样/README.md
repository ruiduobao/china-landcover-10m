# 实验档案 — 稀疏植被 / 裸地 东北·东部补样与 2017–2024 逐年扩展（2026-10-25）

> 主文档：`技术文档/85_稀疏植被裸地补样与2017-2024逐年支撑.md`
> 代码：`代码/10.稀疏裸地补样与逐年扩展/`（30 个脚本 + README）
> 大产物（不入库）：`F:/lc_work/bare_sparse_v3/`（候选、证据、AEF、逐年池、池 v3）

## 本目录内容

| 目录/文件 | 内容 |
|---|---|
| `spec/预注册判据.md` | 先写死后跑的窗口/配额/纯度/证据阈值 + 全部修正记录（含三项护栏、账号口径纠正） |
| `reports/bare_sparse_v3_report.md` | 2023 补样交付报告（采集/质检/核验/分省/面积对账） |
| `reports/yearly_support_2017_2024.md` | **逐年支撑验收报告**（逐年量/跨年一致性/区域分布/双路/CLCD/目视/逐年精度） |
| `reports/accuracy_yearly.md` | 逐年分层随机抽样精度（≥60 点/类/年，独立参考 CLCD；OA/PA/UA + 95%CI） |
| `reports/distribution.md` | 逐年省级分布 |
| `reports/clcd_check.md` | 逐年 CLCD 一致性旁证 |
| `reports/yearly_verify.md` | 逐年核验原始 JSON |
| `reports/visual_audit/` | S2 真彩色接触表（2018/2021/2023/2024 × 稀疏/裸地）+ `result.json`（判读记录）+ `points.csv` |
| `data_stats/` | 逐年统计、跨年/双路/CLCD/精度、国界过滤、球面复核、纯度/留出复核、fcs10 面积、逐年有效期 |
| `delivery_metadata/` | 2023 交付的 README、分省表、stats、含证据列的 parquet（GPKG/SHP 已入类目录） |

## 复现入口

```bash
# 工作区（大产物）
cd F:/lc_work/bare_sparse_v3/code
python h1_harvest_fcs10.py --region north --outdir ../data/candidates
python h1_harvest_fcs10.py --region east  --outdir ../data/candidates
python h1_harvest_fcs10.py --region ne130 --outdir ../data/candidates
python h1c_holdout_filter.py && python h1d_purity_recheck.py
python h2_evidence.py --list ../data/candidates/cand_north.parquet,../data/candidates/cand_east.parquet --acct <acct> --part 0 --nparts 2
python h3_rules.py --cand ... --ev ../data/evidence --outdir ../data/final
python h3c_spacing_fix.py && python h6b_china_filter.py
python h4_aef.py --list ../data/final/cand_qc_china_guarded.parquet --acct <acct>
python h6_deliver.py --final ../data/final/cand_qc_china_guarded.parquet --aef ../data/aef_v2
python h7_pool_v3.py
# 逐年
python h15_run_yearly_fetch.py            # 8 年 AEF + 证据（16 账号并行）
python h12_yearly_rules.py && python h13_build_yearly.py
python h14_verify_yearly.py --acct <acct> && python h14b_clcd_check.py
python h19_accuracy_yearly.py && python h17_dist.py && python h18_report_yearly.py
python h20_publish_class_yearly.py        # 旧样本备份 + 类目录逐年 GPKG
```

## 样本落点

- 类目录：`数据/单类样本的单独处理_2026.10.12/{稀疏植被,裸地}/{2017..2024}/<类>_<year>_WGS84.gpkg`
- 旧样本备份：`数据/单类样本的单独处理_2026.10.12/{稀疏植被,裸地}/旧样本_2026-10-25/<year>/`
- 池 v3：`F:/lc_work/bare_sparse_v3/data/train2023_v3.parquet`（1,768,580）
- 年度池：`F:/lc_work/bare_sparse_v3/data/pools/train_<year>.parquet`（8 个）
