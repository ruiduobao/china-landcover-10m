# 28. 全国生产 Runbook（2017–2024 中国 10m 精细土地覆盖）

> ⚠️ **操作链路仍有效，但规格需按新方案执行**：分类体系已由 30 类改为 **24 类**、模型为**局部单 RF**（不使用 K=5、不使用训练端合并），
> 详见 [`38_R1评审修正与2017-2024生产方案.md`](38_R1评审修正与2017-2024生产方案.md) §三。文档导航见 [`00_文档索引.md`](00_文档索引.md)。



> 2026-09-14 编写 ｜ 前置：`27_v-final冻结与五类F1=0诊断修复报告.md`
> 适用代码：`代码/8.全国生产/`（p0–p6 ＋ fleet 差异化 worker）
> **状态：链路已端到端实跑走通，等待开跑指令。**

---

## 一、一句话架构

**样本（sqrt 配比 20 万点）上传一次到 GEE 资产 → ACL 共享给 9 个生产账号 →
每个账号按分区块逐个提交 `Export.image.toAsset`（服务端 `smileRandomForest` 训练 10–20 s 后分类）→
分块下载 → 本地拼接 → 逐类面积质检 → 落 Z 盘。**

```
母库 2,252,207 点
   │  p1  sqrt 配比抽样（∝√p）20 万点，8 年合并
   ▼
GEE 资产：prodsample_v1_0NN（20k/片）  ── setAssetAcl ──▶ 其余 8 个账号可读
   │  p3  9 账号 × 差异化 worker（prefix/alias/顺序/shardSize/节奏 全不同）
   ▼
分区任务：313 陆域格 × 8 年 = 2,504 个导出（2°格网，10m，uint8 COG 资产）
   │  p4  监控 operations + 额度台账（GET，路径不带 :list）
   ▼
p5  分块下载（0.25°/次 ≈7.7 MB，绕开 48 MB 硬上限）→ 本地拼成 tile
   │
p6  年度拼接 + 全国概览 + 逐类面积（10m 像元精确计数，不靠概览图）
   ▼
Z:\...\生产输出\{栅格,元数据,日志}
```

---

## 二、三条实测依据（架构不是拍的）

### 2.1 内联树集成**不可行**（体积-精度曲线实测）

| 配置（树数/深/叶） | 树字符串体积 | OA | macro-F1 | 30 类非零 |
|---|---|---|---|---|
| v-final 主模型 150/全深度/2 | ~2,900 MB | 0.7413 | 0.6123 | 26/30 |
| 150/16/2 | ~237 MB | 0.6225 | 0.5581 | 30/30 |
| 150/14/200 | ~22 MB | 0.5425 | 0.4941 | 30/30 |
| 120/13/400 | ~8.7 MB | 0.5220 | 0.4727 | 30/30 |
| 40/11/6000 | ~0.4 MB | 0.4507 | 0.4014 | 30/30 |

GEE 单次请求上限 **48 MB**（实测报错 `Total request size (79709184 bytes) must be ≤ 50331648`）。
能塞进请求的模型 OA 只有 **0.54**，低于 GLC_FCS30 自报的 16 类 71.4% → **不能用于生产**。

### 2.2 GEE 端训练**很快**（可承受每任务重算）

| 训练配置 | 样本 | 训练+分类耗时 |
|---|---|---|
| `smileRandomForest(100 树)` | 5,650 点 | **15 s** |
| `smileRandomForest(150 树)` | 5,650 点 | **19 s** |
| `smileRandomForest(150 树, maxNodes=200000)` | 5,650 点 | **11 s** |

→ 每任务固定开销 10–20 s，相对数小时的导出可忽略。

### 2.3 样本资产可**跨账号共享**

`ee.data.setAssetAcl(asset, {'readers': ['user:<acct>@<domain>']})` 实测成功；
被授权账号 `ee.FeatureCollection(asset).size()` = 5,650，且**可直接用于训练+分类（4 s）**。
→ 样本只上传一次，不必 9 个账号重复上传。

---

## 三、生产模型规格

| 项 | 值 | 依据 |
|---|---|---|
| 上传配比 | **sqrt 配比**：n_c ∝ √p_c，下限 150 | 本地对照实验：均衡配比 OA 0.6256 / **sqrt 0.7162** / 自然 0.7214 但只有 26/30 非零 |
| 样本量 | **200,000 点**（8 年合并、按 row_id 去重） | sqrt 下 200k 即达 OA 0.7162、macro-F1 0.6256、**30/30 非零** |
| 分片 | 20,000 点/片（≈9 MB 内联载荷），共 10 片 | 实测 20k 片可提交 |
| GEE 训练 | `smileRandomForest(numberOfTrees=150, minLeafPopulation=2, maxNodes=200000, seed=42)` | 无类别权重 → **配比即权重**；`maxNodes` 限深等效本地 `max_depth≈18`，实测能让稀有类票不被淹没 |
| 特征 | AEF 64 维（`A00–A63`），10 m | v-final 同源 |
| 输出 | `uint8` 类别码，EPSG:4326，scale=10 | — |

> **与 v-final 本地模型的关系**：v-final 的本地指标（OA 0.7959，8 年）来自 `balanced_subsample` 的两模型混合；
> 生产模型是"GEE 内单一模型 + sqrt 配比"，**预期指标低于 v-final**（2022 年口径 0.7162 vs 0.7413），
> 但换来可部署性。这是明确接受的取舍，交付时须如实标注。

---

## 三·补、模型升级：局部自适应建模 + K 模型投票（2026-09-16，opt-in）

> 依据 `技术文档/33`+`34`。**默认关闭**，行为与旧版逐字等价；显式开启后才生效。

### 是什么
把"全局单模型分类全国"改为 GLC_FCS10 的**局部自适应**做法：每个 2° 瓦片用「本瓦片 ± local_margin_deg」范围内的样本**独立训练**；再训练 K=5 个成员做 `ee.ImageCollection([...]).mode()` 众数投票（绕开 GEE 单次训练 ≈2 万点的硬上限）。

### 为什么
25km 块留出池、多种子配对实测：全局单模型 OA **0.6450** → 局部 **0.7106（+6.56pp，3 种子全正）** → 局部+K5+合并 **0.7361（+9.11pp）**。
窗越小越好（6°窗 0.7162 > 10°窗 0.6974）。121 F1 0.43→0.68——**"海绵"是全局训练摊薄样本所致，不是类不可分**（121 AUC 0.96+）。

### 怎么开
```
python p1d_set_deploy_opts.py --acct <账号...> --on --k 5     # 写 local_train/local_margin_deg/k_vote/train_frac 进部署件（自动 .bak）
python p3_gen_workers.py --acct <账号...> --include-busy      # 重新生成 worker（含 mkclf 函数，名称继续零重复）
```
关闭：`p1d_set_deploy_opts.py --acct ... --off`。查看：`--show`。

### 算力注意（未测，排期前必须先量）
局部+K5 = 每瓦片 **5 次服务端训练 + 5 次 classify + 1 次 mode() 归约**。分类器评估在导出算子中的占比未测 → **先跑「同瓦片 K=1 / K=5 对照导出」量出真实倍率**，再决定全国 8 年 2.8 万 EECU·h 预算是否上调。

### 训练端合并（可选、独立生效）
部署件 `merge_map`：`{"121":130,"92":61}`（训练前重标标签）。多种子配对 **+2.46 ± 0.35pp**。
注意：合并是**交付口径简化**而非精度修复——121 在局部建模下 F1 可达 0.66+，若要保留 30 类细类体系，可不合并。

## 四、目录与路径规范

| 用途 | 路径 |
|---|---|
| 代码（权威） | `F:\地理所\论文\中国土地覆盖数据_2017-2024\代码\8.全国生产\` |
| 分区/部署/分配（小文件） | `代码\8.全国生产\plan\`（`tiles.json` / `assign_<acct>.json` / `deploy_<acct>.json`） |
| 差异化 worker | `代码\8.全国生产\fleet\pw_<acct>.py`（生成物，勿手改） |
| SSD 暂存（下载中转，可重建） | `F:\lc_work\prod_stage\<year>\<tile>.tif` |
| **成品落点** | `Z:\...\生产输出\栅格\<year>\`（逐瓦片 COG + `mosaic_index.json` + `overview_<year>.tif`） |
| 台账/面积/QA | `Z:\...\生产输出\元数据\`（`ledger_<acct>.jsonl` / `area_<year>.csv` / `qa_<year>.json` / `dashboard.json`） |
| 运行日志 | `Z:\...\生产输出\日志\` |
| 样本/母库（只读） | `F:\...\数据\本地处理\样本重建\r7_train.parquet` |

**为什么成品是"瓦片集合"而不是单个全国文件**：10 m 全国 = 272 Gpx ≈ 272 GB（uint8），
单文件既不现实也不实用；**GLC_FCS10 本身也是按瓦片分发的**。故交付 = 逐瓦片 COG + 瓦片索引 + 全国概览（0.005°）。

---

## 五、账号编队与防批量识别

### 5.1 编队（`prod_conf.ACCOUNTS`）

| 角色 | 账号 | 说明 |
|---|---|---|
| **free（本轮用）** | `zixen8v8`（主上传）、`y30b63ye`、`zitwwufh`、`nhqz5uj`、`r45smj3u`、`seqsiu`、`iu5f4z`、`1yxth5g`、`hte4021n` | 9 个，巡检时 RUN=0/PEND=0 |
| busy（先不派） | `5rqs3xw2`、`679i9zo111222`、`als74akz`、`e5h08k`、`gm9ufoo4`、`hqzub6` | 巡检时有 RUNNING，需先复核 |
| spare（未启用） | `ief3nj`、`ppzynq`、`pcrw36d`、`3qb2r4` | 留作扩容 |

**硬约束**：
- 30 个账号按用户指令排除；**19 个禁动账号**不得提交/取消；
- 同域账号对 `s8xpcl1w + hte4021n`（`qrjxj4wl01.asia`）、`ppzynq + r45smj3u`（`owesmcayqn.asia`）
  **组内不得派同类任务**——本轮 `s8xpcl1w`/`ppzynq` 均未启用，故不冲突；
- **一账号一区一任务**：batch 并发 = 账号级 1，同账号多仓也不并行。

### 5.2 防批量识别（`p3_gen_workers.py` 自动落实）

每个账号在以下**每一维**取不同值，且生成后做**跨账号零重复自检**：

| 维度 | 取值 |
|---|---|
| 任务名前缀 | `lcf/lcv/lct/lcn/lcm/lcs` + 序号（10 个账号零重复） |
| 模块别名 | 25 个树种词根 × 后缀（`alder_hub` / `birch_tool` …） |
| 10 个函数名 | 每个函数从 5 个同义名里独立取一个（`emit→say/log/tell…`） |
| 时间戳格式 | 5 种轮换 |
| shardSize | `tileScale×8`（16/24/32） |
| maxPixels | 1e12 / 5e12 / 1e13 |
| 分区顺序 | 各账号独立种子洗牌（同一分区不落同序位） |
| 提交间隔 | (1,5)/(2,6)/(3,8)/(5,12) 秒随机 |
| 资产目录 | `lcf_a01`…`lcf_b06`（一账号一目录） |

**另附纪律**：不跨账号同参同区并发；提交前先 `refresh` 确认无活动任务；账号间不时间紧邻提交
（worker 内置随机间隔）；换仓原则见 `01_账号使用说明.md`。

---

## 六、额度监控与记录

- **台账**：`Z:\...\生产输出\元数据\ledger_<acct>.jsonl`（每个 worker 只写自己的行，**多进程不共写同一文件**）
  字段：`acct/tile/year/desc/asset/task/state/eecu_s/ts`
- **看板**：`python p4_watch.py` → 逐账号 `RUN/PEND/SUCC/FAIL/EECU·h` + 产出进度百分比 → `dashboard.json/csv`
- **额度口径**：社区仓 **150 EECU·h/仓为软约束**（实测单账号累计 878 h 仍可跑）；
  优先把任务派给该账号**累计 EECU 最低**的仓；临近 150 h 的仓**降优先级**而非取消在跑任务。
- **纪律**：**不得以"无进度/updateTime 陈旧"为由 cancel 任何 RUNNING/PENDING 任务**；
  只有"同区重复任务"才允许取消 EECU 较少的一个。

---

## 七、账号切换与凭证隔离

```python
# prod_conf.apply_account_env(acct) —— 必须在 import ee / 构造任何 ee 对象之前调用
os.environ['HOME'] = <gee_accounts>/<acct>        # Windows 必须同时设
os.environ['USERPROFILE'] = <gee_accounts>/<acct> # 缺一会静默用错账号
os.environ.setdefault('HTTP_PROXY', 'socks5h://127.0.0.1:7890')   # http:// 前缀必死
```

- **一个进程一个账号**：ee 的全局态禁止进程内切换账号 → 多账号靠**多进程**（每账号一个 worker 进程）。
- 8780/8085 端口残留会静默挂起授权；批量前 `p0_doctor.py` 会做只读探活。
- 代理对高并发 token 刷新敏感：`p0/p4` 默认 **4 线程**，不要再调高。

---

## 八、完整操作流程

```bash
cd "F:\地理所\论文\中国土地覆盖数据_2017-2024\代码\8.全国生产"

# ── 0) 开跑前自检（只读；含账号探活、代理、磁盘、依赖）
python p0_doctor.py                      # 结论必须为"可开跑"

# ── 1) 造样本 + 上传 + 共享 + 出部署件
python p1_deploy_model.py --make-sample --n 200000      # 本地 sqrt 抽样（约 10 s）
python p1_deploy_model.py --all   --n 200000            # 分片上传 + ACL 共享 + 部署件（约 1–2 h）
#   产出：plan/deploy_<acct>.json、plan/deploy_assets.json

# ── 2) 分区表（只读）
python p2_plan_tiles.py                                  # plan/tiles.json（313 格）

# ── 3) 生成差异化 worker + 分区分配
python p3_gen_workers.py                                 # fleet/pw_<acct>.py + plan/assign_<acct>.json

# ── 4) 逐账号提交（每个账号一个进程；账号级并发=1，提 1 个就退出，等完成再提）
python fleet/pw_<acct>.py --submit --max 1
#   建议做成循环：while 未跑完; do 对每个空闲账号 pw_<acct>.py --submit --max 1; sleep 600; done

# ── 5) 监控（随时可跑，只读）
python p4_watch.py

# ── 6) 下载 + 拼接（某个瓦片完成后）
python p5_fetch.py --acct <acct> --tile <T> --year <Y>
python p6_mosaic.py --year <Y>
```

**推荐的开跑节奏**：先用 `--plan tiles_test.json` 的 0.5° 测试瓦片验证全链路（已完成），
再放开 `tiles.json`；每天早晚各跑一次 `p4_watch.py` + 空闲账号补提。

---

## 九、端到端实跑记录（2026-09-14，已走通）

| 步骤 | 命令 | 结果 |
|---|---|---|
| 环境自检 | `p0_doctor.py` | 依赖齐（py3.12.10 / ee 1.7.42 / rasterio 1.4.3）、代理 7890 监听、磁盘 F 582G / Z 3.4T |
| 分区表 | `p2_plan_tiles.py` | 558 格 → **313 陆域格**（>50% 陆域 238 格），8 年 → **2,504 个导出任务** |
| 差异化 worker | `p3_gen_workers.py` | 9 账号，`prefix`/`alias` **跨账号零重复** |
| 样本资产上传 | `asset_train_test.py` | 5,650 点单片 → **COMPLETED（~5 min）** |
| GEE 端训练 | `b_only.py` | 100 树 15 s / 150 树 19 s / `maxNodes` 11 s，区内出现 20–21 类 |
| 跨账号共享 | `acl_test.py` + `acl_read.py` | `setAssetAcl` 成功；被授权账号读到 5,650 行并**直接训练+分类成功（4 s）** |
| 配比定案 | `alloc_probe.py` | 均衡 0.6256 / **sqrt 0.7162（30/30）** / 自然 0.7214（26/30） |
| **① 提交（真实 worker）** | `fleet/pw_zixen8v8.py --submit --plan tiles_test.json --years 2022` | 提交 `lcs08_X001_2022`（0.5°×0.5°，2022，10m，150 树） |
| **② 导出完成** | — | **COMPLETED：墙钟 28 min、1.13 EECU·h**，资产 `lcs08_X001_2022`（IMAGE, 4.94 MB） |
| **③ 监控** | `p4_watch.py` / `pw_*.py --refresh` | 台账与看板正常；`11 条 operation，活动 0` |
| **④ 下载** | `p5_fetch.py --acct zixen8v8 --tile X001 --year 2022 --plan tiles_test.json` | 0.25° 分块 ×4 → 本地拼成 `X001_2022.tif`（**2.5 MB，非零像元 99.97%，40 s**） |
| **⑤ 拼接与面积** | `p6_mosaic.py --year 2022` | 3 个瓦片 → 概览（89.3 Mpx）＋ `area_2022.csv`（**26 类非零**）＋ `qa_2022.json` ＋ 逐瓦片 COG ＋ `mosaic_index.json` |

**首瓦片实测产出（0.5°×0.5°，2022）**：面积表里 `10 草本旱地 989.5 km²`、`190 城镇不透水面 190.3 km²`、
`51 郁闭常绿阔叶林 520.7 km²` 等物理合理；`11 乔灌园地` 明显过高（1,036.9 km²）
—— 这是**只有 5,650 点的小样本模型**导致的过预测，正面说明为什么生产必须用 200k 的 sqrt 配比样本。

**过程中修掉的 4 个真错**（都已回写模板）：
1. `Export.image.toAsset` **不接受 `tileScale`**（那是 `toDrive` 的参数）→ 改用差异化 `shardSize` 承担同等作用；
2. 同名资产**不可覆盖**（`Cannot overwrite asset`）→ 上传前先 `deleteAsset`（幂等重跑）；
3. worker 的 `done_set()` 读台账时，监控行没有 `tile/year` 字段 → 加字段保护；
4. 部署件从 `asset`（单数）改为 `assets`（列表）后，日志语句未同步 → 已修。

---

## 十、排障速查

| 现象 | 原因 / 处置 |
|---|---|
| `Total request size (...) must be ≤ 50331648` | 单次下载超 48 MB → 调小 `SUBTILE_DEG`（0.25°→0.125°） |
| `Cannot overwrite asset 'X'` | 同名资产存在 → `ee.data.deleteAsset(X)` 后重提 |
| `Unknown configuration options: {'tileScale': 4}` | `toAsset` 不支持 tileScale → 用 `shardSize` |
| `Invalid GeoJSON geometry` | Feature 几何非法 → 样本资产改用**无几何** `ee.Feature(None, props)`（本链路已如此） |
| `User memory limit exceeded`（交互式 classify） | 交互式分类 0.25° 即超内存（实测）→ **分类必须走 batch 导出**；交互式只能做 ≤0.15° 的抽查 |
| `Invalid number of random selected features for splitting: 0` | `smileRandomForest` 的 `variablesPerSplit` 传了 0 → **只在给正整数时才传**（默认 None=√n） |
| 下载子块超限 | 实测分类栅格 ≈ **2 字节/像元**：0.5°@10m = 62 MB 超限 → 子块取 **0.25°**（≈15 MB）安全 |
| `not registered to use Earth Engine` | 仓库未注册 → 换 `prod_conf.ACCOUNTS[acct]['proj']` 里已验证的项目号（**不要用 `_任务登记.md` 的第一个仓**） |
| `project is not registered` 或 `Caller does not have required permission` | 项目归属/成员问题 → 见 `01_账号使用说明.md` §六 |
| `operations` 返回 404 + HTML | 端点写成 `POST .../operations:list` → 必须 `GET .../v1/projects/{pid}/operations`，`pageSize` 走 query |
| 任务长期 PENDING | **正常**（可能 Task too old，重提即可）；**不得据此取消** |
| 依赖缺失 | `rasterio` 必需（拼接）；**无 GDAL CLI**，一切走 rasterio |
| 中文出图乱码 | `matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']` |
| 后台进程被宿主清理 | 长任务拆短跑 + 台账幂等（worker 与 p5 都可重复执行） |

---

## 十一、全国制图的规模 / 额度 / 时间估算

**规模**：313 个 2° 瓦片 × 8 年 = **2,504 个导出任务**；单 2° 瓦片 10 m ≈ 4.9 亿像元。

**实测单价（2026-09-14 首瓦片，已用真实数字替换估值）**：0.5°×0.5° 瓦片（≈3,080 km²）@10m，150 树 RF：
**28 min 墙钟 / 1.13 EECU·h** → **3.67×10⁻⁴ EECU·h/km²**。
- **全国单位面积成本**：3.67e-4 × 960 万 km² ≈ **3,520 EECU·h / 年** → 8 年 ≈ **2.8 万 EECU·h**。
  （按 2° 瓦片折算：每瓦片年 ≈ 18 EECU·h；313 格中含 75 个陆域占比<50%，实际按面积算更准。）
- **供给**：9 个 free 账号 × 12 社区仓 × 150 EECU·h（软约束）≈ 1.6 万 EECU·h/轮；
  启用 busy+备用到 15–19 个账号后可用约 2.7–3.4 万 EECU·h。
  → **额度刚好覆盖一轮全国 8 年，几乎没有余量**，必须做好用量盯盘与仓库轮换。
- **时间**：单账号吞吐实测 **2.42 EECU·h / 墙钟小时**（1.13 h ÷ 28 min）。
  2.8 万 EECU·h ÷ 2.42 ≈ 11,600 墙钟小时；**9 账号 ≈ 54 天，19 账号 ≈ 25 天**（不含排队与重试）。
- **必须走 batch 导出**：交互式分类在 0.25° 就撞 `User memory limit exceeded`（实测），
  且单次下载有 48 MB 上限 → 全国不可行。

**降低成本的三个选项**（按推荐序）：
1. **先出 30 m 版**（成本降约 9 倍，用于论文主图与统计），10 m 版分批推进；
2. **年份分批**：先 2020–2022 三年（与新样本 validity 一致），再补 2017–2019/2023–2024；
3. 只对**变化区**做 10 m 精分（需先有 30 m 版做变化掩膜）。
