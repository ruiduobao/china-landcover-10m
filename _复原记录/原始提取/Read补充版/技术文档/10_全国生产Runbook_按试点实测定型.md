# 全国生产 Runbook（按北京试点实测定型）

> **用途**：全国跑 2017–2024 八期 10m 土地覆盖时的**操作手册**——精确路径、命令、改造清单、坑清单、质检门禁。
> **依据**：北京试点全流程实测（2026-09-05，详见 `09_北京试点执行报告与完整流程.md`），所有数字均为实测值而非估算。
> **结论先行**：路线 = **逐年全量分类 + 本地时序一致性**（两点变化检测已因 PIF 漂移废弃）；总预算 **~1.5 万 EECU·h**（36 社区仓弹药自足，主账号非必需）；36 账号并行理论墙钟 **<1 天算力 + 2 天下载 + 1 天后处理**。

---

## 0. 前置条件核对（全国开跑前逐项打勾）

| # | 项 | 位置/命令 | 状态 |
|---|---|---|---|
| 1 | 样本底座 v2（849 万点+置信分层） | `数据/本地处理/样本底座/cn_samples_v2.parquet` | ✅ |
| 2 | 中国边界（含省级） | `数据/边界/china_100000_full.json` | ✅ |
| 3 | 分类体系/码表/cfg | `数据/本地处理/分类体系/{classification_system.csv, codemap.csv, cfg.json}` | ✅ |
| 4 | 本地训练→GEE 迁移工具链 | `代码/0.本地流水线/lc05–lc07`（参考实现），试点版 `代码/2.北京试点/p3_train.py`（**用这个**，已实证树格式） | ✅ |
| 5 | 账号池就位+体检 | `gee_accounts/<账号>/_scripts/full_checkin.py`；试点版 `代码/2.北京试点/pilot_common.py` | ✅ 模板 |
| 6 | 嵌入数据核查 | `代码/2.北京试点/p1_check.py`（改 bbox 为全国） | ⬜ 开跑第一步 |
| 7 | 代理 | socks5h://127.0.0.1:7890（http:// 前缀必死） | ✅ |
| 8 | Python 依赖 | earthengine-api, google-auth, requests, rasterio, shapely, geopandas, sklearn, joblib, pyarrow | ✅ |

---

## 1. 实测参数表（全国预算直接引用）

| 参数 | 实测值 | 来源 |
|---|---|---|
| band-px 单价（top-24 特征） | **0.8×10⁻⁹ EECU·h** | 北京 8 任务实测回归 |
| 单任务 EECU（北京 1.68 万km²） | 1.57–6.86 EECU·h | tasks_pilot.jsonl |
| 单任务墙钟（提交→SUCCEEDED） | 10–30 分钟 | 试点监控 |
| 全国单年（9.6×10¹⁰ px × 24 波段） | **≈1,840 EECU·h/年**；8 年 ≈ **1.5 万** | 模型外推 |
| 10° 块单任务 | ≈66h（<100h 红线 → **可下社区仓**） | 外推 |
| 账号级 batch 并发 | 1（PENDING 不并行） | 01 文档实测 |
| 下载上限 | 50MB/请求；0.25° 分瓦 ≈23MB | p6 实测 |
| PIF 跨年漂移 | cos 0.87–0.98（类别相关）→ 两点检测废弃 | p1 实测 |
| 时序一致性必要量 | 试点平滑 27.7% 像元 | p7 实测 |

---

## 2. 全国生产七阶段（每阶段：做什么/用哪个脚本/改哪里/门禁）

### 阶段 T0：账号池扩容体检（半天）
- 每账号跑 `full_checkin.py`（36 仓逐一 enable+建根+ee 冒烟），结果回写 `_任务登记.md`；
- 新账号接入 = 建目录→authenticate→enable→**console 手动 EE 注册**（程序化不可用！）→复检；
- 门禁：可用仓清单落盘 `gee_accounts/cn_pool.json`。

### 阶段 T1：全国样本嵌入提取（1–2 天，table 任务为主）
- 输入：`cn_samples_v2.parquet` 的 **gold+silver**（249 万点，训练）+ bronze 抽样（验证）；
- 脚本：改造 `p2_sample_embeddings.py` → `n2_sample_emb.py`。改造点：
  - 分块：按 **2° 网格分块**（约 600 块），每块一个 `sampleRegions` 交互请求（table 不受 batch 并发限制，可 6–10 并发）；
  - **必须 mosaic**（试点教训：`first()` 静默丢 98%——AEF 年度集合是 UTM 多瓦片）；
  - 失败重试 + 每 2° 块落盘一个 parquet 分片；
- 抽 2020 年训练用；如做逐年样本对齐再抽各年（可选）；
- 门禁：各 2° 块行数与该块点数一致率 >99%。

### 阶段 T2：分区训练 + 树串生成（本地，免费，1 天）
- 脚本：改造 `p3_train.py` → `n3_train_zones.py`。改造点：
  - 分区：**2°×2° 网格（含相邻 8 格借样，每格每类 ≤5000）**——北京试点单区模型 OA 92.2% 验证了 24 维 RF 的能力，全国沿用；
  - 权重：gold=1.0、silver=0.5；样本权重 ×(0.7+0.3·stab_years/24)；
  - 每区输出 `trees_zone{X}.json`（features+classes+trees）→ 存 `数据/本地处理/北京试点/models_pilot` 的同级目录 `models_national/`；
- 门禁 G2：每区留出 600 点一致性 ≤3% 且分歧全在阈值边界（`p3b_g2_report.md` 判据）。

### 阶段 T3：8 年×分块分类任务（1 天算力）
- 脚本：改造 `p5_submit.py` → `n5_submit.py`。改造点：
  - 网格：**10°×10° 块 × 8 年 ≈ 28×8=224 任务**（每任务 ~66h，社区仓可扛）；或 5° 块 ×8 年 ≈ 900 任务（每 ~17h，更碎但容错好）——**推荐 10° 块起步，实测超 100h 的块再拆**；
  - 任务名：`CNLC10_{year}_B{lon0}N{lat0}`；
  - 分类器：按块中心所在分区加载对应 `trees_zone{X}.json`；
  - 仓库分配：账号内轮换（每任务一个新仓），台账 `tasks_national.jsonl`；
  - 提交纪律：每账号 1 RUNNING + ≤1 PENDING；PENDING 不过夜；
- 门禁 G3：SUCCEEDED 率 100%（失败换仓重提）。

### 阶段 T4：下载（1–2 天）
- 脚本：改造 `p6_download.py` → `n6_download.py`。已内置：0.25° 分瓦、断点续传、跨账号读各自资产；
- 全国 8 年 ≈ **504 瓦/北京 × (960/1.68 万) ≈ 2.9 万个分瓦**，504 瓦实测 ~10 分钟/年 → 全国单年 ~3 小时，8 年 × 并发 6 ≈ **半天–1 天**；
- 门禁：分瓦数齐全 + 文件大小 >1KB + 年份完整。

### 阶段 T5：镶嵌裁剪 + 时序一致性（本地，1 天）
- 脚本：改造 `p7_postprocess.py` → `n7_postprocess.py`。全国改造点：
  - 镶嵌改为 **GDAL 流式**（`gdalbuildvrt` + `gdal_translate`，绝不能全图 read 进内存——试点 1° 窗口法在全国不适用）；
  - 裁剪：市界→国界；
  - 时序一致性三规则不变（孤立年平滑/城市化不可逆/QC 层），但实现改 **rasterio 分块窗口流式**（8.6×10¹⁰ px 全图 numpy 放不进 20GB）；
- 门禁：8 期全国成品 + QC 层落盘，体积 ≈ 25–35GB/年。

### 阶段 T6：验证与发布（2 天）
- 留出集（bronze 全国抽样）混淆矩阵；与 CLCD/ESA WC/GLC_FCS10 对比切片；Olofsson 面积置信区间；
- 脚本：改造 `p8_validate.py` → `n8_validate.py`（rowcol 已修对，直接抄）。

---

## 3. 坑清单（全部踩过，全国跑前重读一遍）

| # | 坑 | 正确姿势 |
|---|---|---|
| 1 | `rasterio.rowcol` 返回 **(row, col)** | 所有解包写 `rows, cols = rowcol(...)`；p8 曾因此 OA=25% |
| 2 | numpy 2.5 多值 `rowcol` 返回 float64 | 采样入口强制 `astype(np.int64)` |
| 3 | AEF 年度集合 = 多景 UTM 瓦片 | 必须 `.mosaic()`；`first()` 静默丢 98% 点 |
| 4 | `FeatureCollection.getDownloadURL` 参数名是 `filetype` 不是 `fileFormat` | |
| 5 | `Export.image.toAsset` 不接受 `formatOptions`（COG 选项无效） | |
| 6 | 树字符串格式 | rpart 风格：`1) root 9999 9999 9999` 开头；子行 `N) band{<\|>=}thr 9999 9999 label[*]`；左=2n 右=2n+1；内部节点尾 9999、叶尾 `*` |
| 7 | sklearn↔GEE 树预测边界不一致 ~2% | 判据=分歧点全距阈值 <1e-3；阈值加 ±1e-6 边距可归零 |
| 8 | G2 校验别忘 `.classify(clf)` | `emb.sampleRegions` 不带分类返回 NaN |
| 9 | Windows 凭证 | HOME+USERPROFILE **双设**；代理 socks5h://127.0.0.1:7890 |
| 10 | 清华/分省数据 bbox 选点重复计数 | 用省界多边形 `contains_xy` |
| 11 | WC macrotile 内部瓦片名含完整产品前缀 | `ESA_WorldCover_10m_2021_v200_{N36}{E117}_Map.tif` |
| 12 | 下载 50MB 上限 | 0.25° 分瓦 + 断点续传 |
| 13 | PENDING >1–2 天服务端作废 | 队列深度 ≤2、按天滚动 |
| 14 | 主进程 `CELL_IDX` 空 | Pool worker 用 `_winit` 重建；主进程测试先手动 build |
| 15 | 存储超限仓导出必 FAILED | 派发前查 `/assets` quota（pelagic 587GB 教训） |

---

## 4. 账号与类型分配（全国版，待用户确认）

| 类型 | 账号 | 内容 |
|---|---|---|
| CN-样本嵌入提取 | 5vqm9g（+zsi8emo/s4ezbd/w2qe4hiu） | 249 万训练点 × 24 维 × 各年 |
| CN-分类推演（池化，分块异参） | 6l3cp84 / neg69g / oh6oiel / zsi8emo / s4ezbd / w2qe4hiu | 224 任务（10°块×8年） |
| CN-验证 | save456jr | 留出集采样与比对 |
| CN-下载归档 | e0p36771 | 交互式 getDownloadURL |
| 冷备 | 7lus57it | 园地/稀有类增补 |

> 禁忌：CN-分类推演池与全球项目 change_map 六账号（kitmy/zhang/ughwvm/vidal/cr/berk）零交集。

---

## 5. 质检门禁汇总（全国）

| 门 | 指标 |
|---|---|
| T1 | 样本嵌入块完整率 >99% |
| T2 | 每区 G2 不一致 ≤3% 且全在阈值边界 |
| T3 | 任务 SUCCEEDED 100%；EECU 台账零缺登记 |
| T4 | 分瓦齐全（0.25° 网格对账） |
| T5 | 8 期成品 + QC 落盘；体积对账（25–35GB/年） |
| T6 | 留出 OA ≥85%（一级类）；面积置信区间产出 |

---

## 6. 待办（全国开跑前的脚本改造，全部本地免费）

1. `n2_sample_emb.py`（2° 分块+并发+分片落盘）；
2. `n3_train_zones.py`（分区借样训练+树串批量生成）；
3. `n5_submit.py`（10° 块网格任务生成+账号池轮换+台账）；
4. `n6_download.py`（全国 0.25° 网格+断点续传——试点脚本已 80% 可用）；
5. `n7_postprocess.py`（GDAL 流式镶嵌+分块时序一致性）；
6. `n8_validate.py`（全国留出集）。

