# GBIF 账号开通后：操作说明（选一条路走）

> 2026-09-16 ｜ 前置：`31_外部中国地物样本点采集报告.md`
> 你已注册 GBIF 账号 ⇒ 解锁**异步下载 API**，可一次拉全量数据（含 `habitat`/`locality`/`elevation` 文本字段）。

---

## 为什么要走下载 API（而不是继续用检索 API 分区抓）

检索 API 每条只有基础字段，且 `offset ≤ 100,000` 必须分区。**下载 API 给的 CSV 里有三个字段是我们真正缺的**：

| 字段 | 为什么关键 |
|---|---|
| `habitat`（生境描述） | **这是直接的地类标签**。标本标签常写"沼泽"、"林下"、"水边"、"岩石上"、"沙地"、"果园"、"茶园" —— 比物种名强得多 |
| `locality`（产地描述） | 同样可文本挖掘，且能校验坐标是否被降精度到县中心 |
| `elevation`（海拔） | 高寒类（地衣苔藓/高寒草甸）的硬约束 |

**所以本轮的核心方法不是"用例的物种名推地类"，而是"从生境文本里挖地类标签"**——这是物种数据唯一站得住脚的用法。

---

## 三个下载（按优先级）

### 【下载 A ★最优先】8 个高精度标本馆 · 中国 · 全类群 = **867,351 条**

这 8 个数据集是**唯一坐标保留 6 位小数 + 带 locality/habitat 文本**的中国植物标本集
（其余两个大集共 310 万条被统一降精度到 2 位小数 ≈1.1 km，无用）。

```
https://www.gbif.org/occurrence/search?country=CN&has_coordinate=true&has_geospatial_issue=false&dataset_key=b54a3b16-b648-41ab-bd6f-d3881895daac&dataset_key=249ec6d9-ce13-4096-baf5-42130fc3d0b1&dataset_key=246ad1f8-e75c-419d-b97e-16dde697ae30&dataset_key=1b412457-43cf-4a6f-b5c2-7acf01be5977&dataset_key=f9336171-e9a9-4147-b9f6-2d6c133afffd&dataset_key=614d7304-af7c-4613-81d8-488e81e28229&dataset_key=196ac794-b2c2-40d4-8e6a-649d0abc324d&dataset_key=af7d2594-6d4e-47e9-9f30-b7ebb3110f7b
```

### 【下载 B】地衣 + 苔 + 藓（全国所有数据集）= **79,744 条**

（Bryophyta 61,181 / Marchantiophyta 8,594 / Lecanoromycetes 9,969 / Arthoniomycetes）

```
https://www.gbif.org/occurrence/search?country=CN&has_coordinate=true&has_geospatial_issue=false&taxon_key=35&taxon_key=9&taxon_key=180&taxon_key=313
```

### 【下载 C·可选】中国维管植物全部 = **4,500,192 条**（约 8–10 GB）

量大且大部分来自降精度数据集，只有确认前两个不够用时再下。

```
https://www.gbif.org/occurrence/search?country=CN&has_coordinate=true&has_geospatial_issue=false&taxon_key=7707728
```

---

## 两条路，选一条

### 路 1：你点链接（**推荐，凭据不出你的浏览器**）

1. 打开上面 A 的链接（会显示"867,351 条匹配"）
2. 页面上点 **Download** → 选 **Simple**（CSV）格式 → 确认
3. GBIF 后台打包，几分钟到半小时，**邮箱会收到完成通知**
4. 把 zip 存到 **`F:\lc_work\外部样本\GBIF_下载\`**（目录已建）
5. 告诉我"下好了"，我来做解压、字段核验、生境文本挖掘、AEF 嵌入验证、入库

### 路 2：我来跑（全自动）

1. 新建 **`F:\lc_work\外部样本\gbif_cred.txt`**，两行（**不要贴到聊天里**）：
   ```
   你的GBIF用户名
   你的GBIF密码
   ```
2. 告诉我一声，我执行：
   ```
   python F:\lc_work\gbif_download.py A 你的邮箱
   python F:\lc_work\gbif_download.py B 你的邮箱
   ```
   脚本会提交请求 → 轮询到 `SUCCEEDED` → 自动把 zip 拉到 `GBIF_下载\`

> 注：GBIF 用**用户名**登录，不是邮箱。凭据文件只在本机、只被这个脚本读。

---

## 拿到数据后我会做什么（五步）

1. **解压 + 字段核验**：确认 `decimalLatitude/Longitude` 小数位、`coordinateUncertaintyInMeters`、`habitat`/`locality`/`elevation` 的填充率
2. **生境文本挖掘**：中英双语关键词表打标签
   - 湿地：`沼泽|swamp|marsh|bog|水边|河边|湖畔|滩|tidal|mangrove|红树`
   - 岩石/结皮：`岩石|石上|岩面|rock|cliff|soil crust|结皮|砾石`
   - 沙地/荒漠：`沙地|沙漠|沙丘|sand|dune|desert|戈壁|盐碱`
   - 草地：`草地|草甸|草原|meadow|grassland|steppe|放牧`
   - 灌丛：`灌丛|林缘|灌丛中|scrub|shrubland|thicket`
   - 森林：`林下|林中|林内|forest|understory|canopy`
   - 园地：`果园|茶园|种植园|栽培|orchard|plantation|tea garden|cultivated`
   - 农田：`农田|耕地|田边|稻田|field|farmland|paddy|roadside`
3. **交叉过滤**：`生境标签` ∩ `AEF 嵌入预测` ∩ `坐标精度 ≤11m` 三者一致才保留
4. **产出可用样本集**：按类统计，与母库现有量对比，给出"能补哪几类、各补多少"
5. **写回**：报告落 `技术文档/32_*`，样本落 `Z:\...\数据\外部样本\`，通过 `e10_sample_qa.py` 质检后才并入

**预期（基于已做的 20 组 + 属一级验证）**：
- 有实据能补的：**51 郁闭常绿阔叶**（Machilus/Phoebe/Castanopsis 命中 52–68%）、**71 郁闭常绿针叶**（Abies/Picea）、**121 落叶灌丛**（Caragana/Rosa/Cotoneaster 约 30%）、**184/185 海岸湿地**（红树属 42%）
- 生境文本挖掘是新增量，可能解锁：**180/181 沼泽**（原来只靠物种名是 6–14%，但标本生境栏写"沼泽"的会直接命中）
- **仍然不要指望的**：**11 园地**、**140 地衣苔藓**（物种路线已实测失败，理由见 `31` 号文档 §3）

---

## 附：文件位置

| 文件 | 说明 |
|---|---|
| `F:\lc_work\gbif_download.py` | 路 2 用的一键脚本（已就绪） |
| `F:\lc_work\外部样本\query_A.json` / `query_B.json` | 路 2 的请求体（也可用于网页端"create new download"） |
| `F:\lc_work\外部样本\GBIF_下载\` | 路 1 的 zip 存放目录（已建） |
| `F:\lc_work\外部样本\gbif_cred.txt` | 路 2 的凭据文件（**需你创建**，两行） |
