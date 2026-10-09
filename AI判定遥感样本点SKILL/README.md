# 遥感样本点验证与地物类型标记 Skill

> **rs-sample-labeling** ｜ 给一份点位表（`point_id, lon, lat`），产出**逐点判读结论**：正确 / 错误 / 无法判读 + 实际类别 + 置信 + 证据，并附可复核的质控统计与 GIS 交付件。
> 面向遥感解译人员与承接样本判读任务的 AI Agent。方法来自「中国 2017–2024 逐年 10 m 土地覆盖」项目的真实判读实践（M3-B 灌丛仲裁 210 点高分影像盲判 + D0 材料链），全部命令与判据都经过实跑验证。

![全流程](assets/fig1_流程.png)

**核心方法一句话**：高分影像判结构与边界，哨兵时序判日期与物候，辅助指标只作旁证；盲判防锚定，复判与审计证可靠。

---

## 它解决什么问题

遥感产品生产与论文里，**样本点的标签是不是真的**永远是最大的不确定性来源：

| 任务 | 示例 | 判读要回答 |
|---|---|---|
| 标签核验 | 某灌丛专题层抽 600 点 | 这个点到底是不是灌丛？ |
| 细类判别 | 东北 2023 训练池标了玉米/水稻/大豆 | 到底是不是玉米？还是大豆/水稻？ |
| 成品精度 | 独立验证点 3,210 个 | 这个 10 m 像元实际是什么地类？ |

共同需求是**批量、可复现、有据可查，而且能细分到作物种级**——不是只到"耕地/林地/草地"。

---

## 判读界面长什么样

![切片解剖](assets/fig2_切片解剖.png)

每点自动抓 **z18 高分影像（≈0.5 m）**、以点为中心裁 512 px，叠加**十字丝（中心留空，不遮挡被判像元）+ 目标框（10 m，按纬度换算像素）+ 编号**；判读时按 2×2 拼成联系表（512 原生不缩水），**批次只给编号，不给先验标签**（防锚定）。

---

## 细类怎么判：高分看结构，时序看物候

单期高分图分不开玉米和大豆（同款均匀绿冠层）——这是本项目实测踩过的坑。补上**有日期的哨兵时序**后，三作物物候节律明显可分：

![三作物物候](assets/fig3_三作物物候.png)

| 作物 | NDVI 峰值 | 关键旁证 | 高分结构 |
|---|---|---|---|
| 水稻 | 最早（7 月中） | NDWI 前期高＝泡田—封行；格田+田埂 | 矩形格田、镜面水、成行秧苗 |
| 玉米 | 居中（7 月末） | 收获后**高茬** | 宽行（0.6–0.7 m）、行间露土、抽雄发黄 |
| 大豆 | 最晚（8 月上旬） | **9 月中旬叶片先黄**（油料作物早衰） | 行距相近但冠层封行早、叶色深 |

判据全集（含小麦/棉花/油菜/甘蔗/果园/茶园/大棚/蔬菜/地膜/养殖塘）与**六大区域农时表**见 [`references/地类判读图谱.md`](references/地类判读图谱.md)。

---

## 怎么证明可信：质控协议

![质控协议](assets/fig4_质控协议.png)

内置三件套，避免"自己判自己"：

- **复判抽样**：随机 ≥10% + 全部低置信 + 全部"无法判读"，重判一遍算自一致率（门槛 ≥0.90）；
- **否定审计**：随机抽"否定"点重判，算"其实是对的"比例（门槛 ≥0.90）；
- **最小复核清单**：把分歧 + 无法判读 + 低置信"确认"归集成一张表，人工只核这些。

报告强制写口径：**确认率/一致率是判读结果，不是精度**；AI 判读须标注是否经人工复核；从训练池抽的点只评标签质量，不评成品精度。

---

## 目录结构

```
AI判定遥感样本点SKILL/
├── SKILL.md                      # Skill 入口（触发条件 + 硬规则 + 命令级流程）
├── scripts/                      # 7 个脚本，全部可独立运行
│   ├── prep_points.py            # 输入质检：CSV/SHP/GPKG/GeoJSON/XLSX → 规范表 + 表单骨架 + 盲判清单
│   ├── fetch_chips.py            # 高分切片：Esri 主源 / Google 备源，十字丝 + 目标框 + 编号
│   ├── s2_timeseries.py          # 哨兵时序：免密钥 STAC，逐窗云掩膜 → NDVI/NDWI + 物候指标
│   ├── build_sheets.py           # 联系表：2×2 盲判批次（只给编号）
│   ├── qc_protocol.py            # 质控：复判/审计抽样 + 最小复核清单
│   ├── merge_stats.py            # 合并统计：确认率/分组/混淆/敏感性/复判一致率 + 报告
│   └── points_to_shp.py          # 交付：SHP（WGS84 + GCJ-02 双版）+ 字段对照
├── references/                   # 判读知识（按需读）
│   ├── 地类判读图谱.md            # 作物决策树 + 区域农时 + 时序指标 + 混淆速查
│   ├── 影像源与瓦片手册.md         # 瓦片数学、源选择、并发/代理、9 条已踩的坑
│   └── 判读协议.md                # 三态操作定义、盲判纪律、质控三率、报告口径
├── assets/                       # README 配图 + 生成脚本
├── 00_搭建大纲.md / 01_场景与功能模块分析.md / 02_三轮用户模拟迭代记录.md
└── _试跑/轮1_东北作物/            # 真数据试跑现场（24 点：切片/联系表/时序/表单/报告/SHP）
```

---

## 快速开始

**依赖**：`pillow numpy pandas`（切片与统计）；`rasterio`（哨兵时序）；`geopandas pyogrio fiona`（SHP 交付）。网络需可达 Esri / Google 瓦片与 `earth-search.aws.element84.com`（哨兵 STAC，免密钥）。

```bash
S=path/to/AI判定遥感样本点SKILL/scripts

# ① 点位质检（支持 csv/shp/gpkg/geojson/xlsx）
python $S/prep_points.py     --points points.csv --outdir run1 --prior-col 先验类 --strata-col 区

# ② 高分切片（z18≈0.5 m；--box-m 30 用于 Landsat 场景）
python $S/fetch_chips.py     --points run1/points_clean.csv --outdir run1 --workers 8

# ③ 哨兵物候（作物细类必做；>30 点建议分批或走 GEE）
python $S/s2_timeseries.py   --points run1/points_clean.csv --outdir run1 --year 2023 --start 05-01 --end 09-30

# ④ 盲判联系表 + 判读（结果填 run1/form_filled.csv：结论/实际类别/置信/证据）
python $S/build_sheets.py    --chips run1/chips --outdir run1
python $S/prep_points.py     --points run1/points_clean.csv --outdir run1 --metrics run1/s2_metrics.csv  # 指标提示列

# ⑤ 质控抽样 → 复判/审计（对 qc/*_blind.csv 再走一遍 ②④）
python $S/qc_protocol.py     sample --labels run1/form_filled.csv --outdir run1/qc --strata-col 区
python $S/qc_protocol.py     review --labels run1/form_filled.csv --recheck run1/复判填表.csv --outdir run1/qc

# ⑥ 统计报告 + ⑦ GIS 交付
python $S/merge_stats.py     --base run1/points_clean.csv --labels run1/form_filled.csv run1/复判填表.csv \
                             --outdir run1 --prior-col 先验类 --strata-col 区 --threshold 0.40
python $S/points_to_shp.py   --csv run1/merged_labels.csv --outdir run1/shp --gcj02
```

**安装为可调用技能**：把本目录（`SKILL.md` + `scripts/` + `references/`）拷到 `~/.zcode/skills/rs-sample-labeling/`，之后在会话里说"用 rs-sample-labeling 技能核验这批点位"即可自动加载。

**实测速度**：高分切片 ≈1.6 s/点（并发 8）；哨兵时序 1.5–3 min/点（免密钥跨境 S3 单文件 open 4–5 s，属固有成本）；24 点全套判读材料约 1 分钟（不含时序）。

---

## 四条硬规则（写进 Skill，判读时不得违反）

1. **盲判**——判读批次只给 `point_id`，不得出现先验标签/分层/臂别；先盲判再对照。被告知答案会让判读率虚高（本项目实测从 12% 虚高到 100%）。
2. **三件套**——每条结论必须带 **置信 + 证据（影像源与日期/季节 + 辅助指标数值）**；缺一不算完成。
3. **"无法判读"单列**——不并入"错误"、不进确认率分母；云/阴影/影像不可用要写明原因。
4. **没有日期就没有作物**——作物种级结论必须由有日期的时序影像支撑。

---

## 边界与声明

- 本 Skill 做**点位级判读与核验**，不做地类分类建模、不生产地图。
- **AI 判读 ≠ 正式人工盲判**：论文/交付结论必须声明复核状态，或补人工签字。
- 影像版权：Esri/Google 为商业底图，仅用于解译判读；成果对外发布按各源条款标注。Sentinel 数据免费开放。
- 判读结论可追溯到具体影像源与日期——这是本 Skill 与"看图说话"的区别。

---

## 出处

方法沉淀自「中国 2017–2024 逐年 10 m 土地覆盖（AEF 嵌入 + GEE 局部随机森林）」项目的判读实践：

- M3-B 灌丛仲裁：210 点 z18 影像盲判，把先验层精度从 AI 单遍 0.107 修到 0.058，判据"≤0.40 → 全筛"稳健成立（推翻需 36.3% 的"否"错判，实测 7.1%）；
- D0 材料链：4,476 点高分切片 + 哨兵四季合成 + GEDI 冠层高度 + Hansen 树覆盖；
- 生产链路：五省 76 个 2° 瓦片 10 m 交付（QA_OK 76/76）。
