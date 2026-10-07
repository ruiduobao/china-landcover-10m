# 15. Agent 复现手册——从零复现全流程的逐步操作指南

> 读者：接手的 agent 或新成员。按本文顺序执行可完整复现：r1 样本库重建 → 嵌入提取 → 清洗训练 → 逐年分类 → 精度验证。
> 环境：Windows + Git Bash + Anaconda（F:\anaconda，python 即 F:\anaconda\python.exe）+ 代理 socks5h://127.0.0.1:7890（常开）。
> 工作根目录：`Z:\Mywork\论文\中国土地覆盖数据`（**所有命令均在此目录执行**——旧脚本用相对路径）。

---

## 0. 前置检查（5 分钟）

```bash
cd "Z:/Mywork/论文/中国土地覆盖数据"
python -c "import ee, sklearn, rasterio, geopandas, scipy, pyarrow; print('依赖OK')"
# 代理探测（必须返回任意 HTTP 码）
curl -s --socks5-hostname 127.0.0.1:7890 -o /dev/null -w "%{http_code}\n" --max-time 15 https://earthengine.googleapis.com
# 关键数据存在性
ls 数据/样本数据/GLC_Samples_Grid_2x2 | wc -l          # 应 64800
ls 数据/本地处理/样本底座/cn_samples_v1.parquet          # 旧底座（投票列桥接用）
ls "E:/gisrsdata.org/还未整理/地理资源/遥感地物分类"      # E 盘投票栅格 5 套
ls "E:/地理所/论文/全球土地覆盖数据/数据/GLC_FCS10"       # FCS10 96 瓦
```

GEE 账号：`F:\地理所\论文\博士论文\中期\实验区域\02_中间数据\gee_accounts\`（本流程用 zsi8emo/s4ezbd/w2qe4hiu；先读该目录 `01_账号使用说明.md`）。新账号接入：建目录 → 设 HOME+USERPROFILE 双环境变量 → `earthengine authenticate` → ee.Initialize 实测 → 在对照表登记。

## 1. 重建 r1 样本库（纯本地，~2h，产物已交付可跳过）

```bash
cd "Z:/Mywork/论文/中国土地覆盖数据/代码/5.样本重建"
python -u run_all.py            # 断点续跑；--force 全部重跑
```
- 顺序执行 s2 底座(849万) → s3 外部点(57.8万) → s3b GALF(0.7万) → s4 FCS10重采(6.9万,~50min) → s4b 专题产品(2.9万,柑橘5GB扫~25min) → s5 投票(917万,外部点票有缓存 votes_*.parquet) → s6 训练清洗(299万) → s7 验证池(2.6万) → s8 交付(F盘暂存拷回)。
- 每步日志：`数据/本地处理/日志/s*_run.log`；产物与 summary：`数据/本地处理/样本重建/r1_*.parquet|json`。
- **断点规则**：parquet/summary 存在即跳过；thematic_shards/、votes_*.parquet、topup_shards/ 均可断点。
- 门禁自动执行（30 类全覆盖/境外0/生态0/NN≥500m/计数一致），结果 `r1_gate.json`。
- 交付落 `数据/样本交付/`（pool/train/validation gpkg + by_class 60 文件 + by_province 34 + 程序化 README）。
- ⚠️ 单独重跑某步：`python -u s6_clean_train.py`（从 WORK 目录读上游 parquet）。
- ⚠️ **勿动** `样本交付/_归档_v4v5_20260906/`（旧版留档）。

## 2. 生成试验区样本子集（b1，1 分钟）

```bash
cd "Z:/Mywork/论文/中国土地覆盖数据"
python -u 代码/5.样本重建/b1_beijing_subset.py
```
- 读 `数据/本地处理/生产配置.json` 的 bbox → 裁 r1_train → 0.25° 整格 75/25 划分 → 写配置的 train_file/holdout_file。
- 当前配置：北京 0.6°×0.5° 窗口（116.2-116.8E, 39.9-40.4N），train ~7.8k / holdout ~2.6k。
- 全国时此步跳过（train_file 直接指全国 r1_train）。

## 3. 锚年嵌入提取（p2，GEE 交互，~2 分钟）

```bash
python -u 代码/2.北京试点/p2_sample_embeddings.py
```
- 读配置 train/holdout 文件 → 2020 年 64 维嵌入（sampleRegions, scale=10）→ 写 `{out_dir}/samples_bj_{train,holdout}.parquet`（含 lon/lat/cls/tier/A00-A63）。
- 单账号 zsi8emo；每批 ≤1 万点（大批次需拆分，见 e1_worker 的 1 万点/块惯例）。

## 4. 训练 + GEE 树迁移（p3，本地+1 次 GEE 交互，~5 分钟）

```bash
python -u 代码/2.北京试点/p3_train.py
```
- 64 维 RF → top-24 重训（树数取配置 n_trees，当前 100）→ 转 decisionTreeEnsemble 字符串 → G2 校验（GEE 端 600 点预测 vs sklearn；**不一致率 ≤2.5% 为已知浮点噪声，PASS 判据看打印**）→ 产物 `{out_dir}/models_pilot/{rf_bj.joblib, trees_bj.json, p3_report.json}`。
- trees_bj.json 是 GEE 分类的唯一模型载体（features/classes/trees）。

## 5. 逐年分类——二选一

### 5a. 小区域（≤0.6°×0.5°）：交互式直下（p5d，推荐，~15-20 分钟/年）
```bash
python -u 代码/2.北京试点/p5d_interactive.py
```
- 0.05° 分块（**勿改大**：0.1°/0.25° 触发交互式 User memory limit）→ 逐块 classify+getDownloadURL(GEO_TIFF) → `{out_dir}/raster/{task_prefix}_{year}_p*.tif`。
- 内置退避重试（429/503 → 15/30/45s…6 次）；断点续传（已下块跳过）。
- ⚠️ ee 对象必须在 `PC.load_account()` 之后构造（进程内禁止切换账号）。

### 5b. 大区域：batch（p5 + p5b 监控）
```bash
python -u 代码/2.北京试点/p5_submit.py     # 提交 8 年任务（账号轮换+换仓）
python -u 代码/2.北京试点/p5b_monitor.py   # 轮询状态/取消卡死/重提
python -u 代码/2.北京试点/p5c_poll.py 120  # 纯轮询直至全 SUCCEEDED（分钟粒度）
```
- 任务清单 `{out_dir}/tasks.jsonl`（id/year/account/pid/asset/taskId/status）。
- 单任务 ≤0.25° 瓦（50MB 上限）；队列深度每账号 ≤2-3 仓；PENDING 超 2 天重提。

## 6. 本地后处理（p7，秒-分钟）

```bash
python -u 代码/2.北京试点/p7_postprocess.py
```
- mosaic 各年 `_p*.tif` → 行政区裁剪 → 3 年投票平滑（bit0）+ 城市化不可逆（bit1）→ `{out_dir}/raster/{task_prefix}_{year}_final.tif` + QC 层。
- ⚠️ p5d 交互路线跳过 p6（p5d 已产出分块 tif，命名与 p7 通配匹配）；batch 路线先跑 `p6_download.py`。

## 7. 精度验证（p8，~1 分钟）

```bash
python -u 代码/2.北京试点/p8_validate.py
```
- holdout 点查 2020 成品像元 → 混淆矩阵 OA/κ/UA/PA → `{out_dir}/p8_validation.json`。
- 参考基线：旧试点（v2 样本）OA 90.11%；r1 样本本地复训 OA 91.65%。

## 8. 全国生产（改一个 JSON 后重复 §2-§7）

```jsonc
// 数据/本地处理/生产配置.json
{ "area_name": "china_r1", "bbox": [73,18,135,54], "region": "中国",
  "out_dir": "数据/本地处理/全国R1", "task_prefix": "CNLC10_CN",
  "train_file": "数据/本地处理/样本重建/r1_train_全国嵌入版.parquet",
  "holdout_file": "...同上 holdout...", "n_trees": 100 }
```
1. 全国样本嵌入提取：改用 `代码/4.全国清洗训练/e1_worker.py`（多账号分块断点，`python e1_worker.py <账号>` 从项目根目录启动，3 账号并行 2-4h）；
2. `e2_clean.py`（马氏清洗）→ `e3_train_zones.py`（2° 分区训练+评估）；
3. p5 batch：全国拆 250-350 个 0.25° 瓦任务 × 8 年（降耗后 ~7,000-9,500 EECU·h，10 账号 10-14 天）；
4. p6 下载 → p7 镶嵌一致性 → p8 + r1_validation 独立验证。
- **region 必须用陆地多边形**（DataV "中国"全境去海，省 15-25%）。

## 9. 故障速查

| 症状 | 原因/处置 |
|---|---|
| `User memory limit exceeded`（交互） | 块太大 → STEP=0.05 |
| `Pixel grid dimensions ... 32768` | region 过大 → 分块或改 batch |
| `Total request size > 50331648` | 单块 >50MB → 缩 STEP |
| 429/503 | 退避重试（p5d 内置）；持续则错峰 |
| `restricted mode` 警告 | 非商业配额预警；batch 不受影响则继续，异常则换仓 |
| `EE client library not initialized` | load_account 必须在构造 ee 对象前；进程内禁切账号 |
| Worker 报 FileNotFoundError | 必须从项目根目录启动（相对路径） |
| GPKG 写 Z 盘极慢 | F 盘暂存写完拷回（s8 已内建） |
| 代理断连（10061） | 等代理恢复；p5d 断点续传安全重跑 |
| 2017-2025 年份缺失 | `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` 各年景数用 p2 的 size().getInfo() 核实 |

## 10. 关键文件索引

| 内容 | 路径 |
|---|---|
| 唯一区域配置 | `数据/本地处理/生产配置.json` |
| r1 流水线（s0-s8/run_all/monitor） | `代码/5.样本重建/` |
| 分类链（pilot_common/p2-p8 + p5c/p5d） | `代码/2.北京试点/` |
| 嵌入清洗与分区训练（e1-e5） | `代码/4.全国清洗训练/` |
| 权威码表 | `代码/0.本地流水线/lc_conf.py` |
| r1 中间产物 | `数据/本地处理/样本重建/` |
| 试验区产物 | `数据/本地处理/北京试点S/`（配置 out_dir） |
| GEE 账号池与登记 | `F:\...\gee_accounts\`（01 说明 + 00 对照表） |
| 技术文档 01-15 | `技术文档/`（14 总集 / 13 降耗 / 12 样本重建） |

---

## 更新记录（2026-09-09｜r7 收尾 + 交付格式升级）
新增脚本：`e2b_yearly_qc.py`（年度QC/年度子集）、`e6_spatial_eval.py`（空间独立评估）、`e7_merge_rare.py`（稀有类合并）、`e8_export_gpkg.py`（GPKG 导出）、`e9_class_maps.py`（分布图）、`e1c_worker_generic.py`（通用提取 worker）、`r8/r8b/r9`（稀有类补样）。

- **r7 年度嵌入提取完成**：4,758 块 / 14,272,711 point-years（2017–2024），12 账号编队、3,000 点/块 + 下载重试。
- **年度 QC 与八套年度子集**：`sample_year_qc.parquet`（keep 13,475,352 / downweight 570,893 / exclude_year 216,274 / persistent_change 10,192）+ `年度子集/r7_train_{2017..2024}.parquet`。
- **稀有类补样（当日第二次更正）**：初版按 FCS10/GMW/CW 将 91/140/183/184 补至 2,000 点/类；经省级白名单复检，140 在东部沿海为 FCS10 错标（养殖塘/潮间带）、183 有 148 点落在内陆淡水湖区，已剔除错位点 1,959 个。现存量：91 2,000 / **140 194** / 183 1,852 / 184 2,000，全库失衡比实际 ≈2,295×（121 vs 140）。精度优先，未按数量回补。
- **空间独立评估（剔错后重跑）**：OA 0.7943 / macro-F1 0.6077 / balanced accuracy 0.6048（n=29,831，剔除 367 个 holdout 块后）。
- **交付格式升级**：`数据/样本交付/gpkg/`（22 个 GeoPackage，保留全部属性，Point EPSG:4326）+ `数据/样本交付/分布图/`（30 类单类分布图 + 6×5 面板 + 总览图）。
- 详见 `18_V1版本的样本审稿意见落实与提升报告.md` 与 `19_样本交付与文档升级记录.md`。
