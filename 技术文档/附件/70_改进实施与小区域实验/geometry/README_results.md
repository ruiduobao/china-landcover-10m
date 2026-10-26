# geometry/ — G-30N 30°N 南缘楔形：平面几何修复工程验证（结果与交接）

> 2026-10-15，几何子 agent。依据 doc70 §4.4/§5(G-30N)/§7 几何覆盖门；doc69 §6.4；seams 报告 §8.1。
> 结论一句话：**legacy 测地 clip 在 T1206 南缘造成 23,394 px（本小窗内）输入有效却输出为空的工程缺口；planar clip 将其归零，且楔外/共同有效域类码完全一致（0 分歧）。这是工程覆盖提升，不是分类标签精度提升。**

## 1. 交付件

| 文件 | 内容 |
|---|---|
| `config/planar_g30n.json` | **可执行工程配置（已验收）**：平面 clip+region、`materialize_mask=true`、模型固定 v2 训练原几何不改、seeds[7,11,23]、deadline 1200；显式列出 `NOT_ENABLED`（O-CAP/F-VIEW/T-STRATA 均 false） |
| `geom_planar.py` | **可复用 helper**：平面/测地矩形、`crsTransform` 构造、`scale`×`crsTransform` **互斥校验**、EVM 共同有效掩膜、测地拱高解析预测、网格吸附。纯函数，无 IO/无请求。 |
| `g1_30n.py` | 独立实验脚本：`--config` 实际读取上述配置 → 登记 → 单请求 5 波段（legacy/planar/legacy_mask/evm/expmask）→ 真实 GeoTIFF → 评估；`--selfcheck` 做配置自检 |
| `run_g1.py` | **harness**：父进程 deadline 先启动、子进程树超时终止、原子结果、并回共享 `registry/experiments.json` |
| `h0_health.py` | 串行账号健康（零 EECU）+ 训练池 ACL 可读性核验 |
| `g0_probe_mask.py` / `g0b_probe_geom.py` / `g0c_probe_aef.py` | 三次一次性探针（掩膜解码 / 几何测地性 / AEF 路径） |
| `g2_deliver_palette.py` | 单波段调色板交付修正（`photometric=PALETTE`，源件不改） |
| `c3_optin_patch_proposal.md` | 生产 c3 的**安全 opt-in 补丁方案**（默认不改 c3） |
| `delivery/G30N_*_{legacy,planar}_10m_palette.tif` | **单波段交付件（带调色板，可被读取器识别）** |
| `delivery/MANIFEST_MD5.txt` | 源件与交付件双 MD5 清单 |
| `out/G30N_v2_s{7,11,23}_5band.tif` | 真实分析 GeoTIFF（uint8×5：legacy/planar/legacy_mask/evm/expmask，**不含调色板**） |
| `out/G30N_v2_s{7,11,23}_{legacy,planar}_10m.tif` | 单波段小区分类图（含 ColorMap 但 `PhotometricInterpretation=1`，见 §3.2） |
| `out/G30N_v2_s{7,11,23}_eval.json` | 逐项计数与一致率 |
| `out/G30N_reconcile.json` | **分母对账**（解析弧 vs 实测边界，逐种子） |
| `reg/g30n_*.json` / `run_results/*.json` | 注册与 harness 计时记录 |

## 1.1 可运行入口（现有框架足够，无需新造）

```bash
cd F:/lc_work/improve_2023_pilot_20261015/geometry
# ① 配置自检（零云端计算；含 ee 客户端序列化检查）
python g1_30n.py --selfcheck
# ② 真实实验（harness 硬截止 1200s；读 config/planar_g30n.json）
python run_g1.py --exp G30N_seedset_v2 --limit 1200 -- --seeds 7,11,23 --tag v2
# ③ 单波段调色板交付修正（纯本地，无 GEE）
python g2_deliver_palette.py
```
`--selfcheck` 实测输出要点：`planar_geodesic_false=true`、`legacy_has_geodesic_key=false`（即 legacy 走服务端 Polygon 默认 geodesic=true）、`xor_ok=true`、`region_bbox_matches_config=true`、`PASS=true`。

## 1.2 账号与项目（口径统一）

全部请求只用一个账号：**`k3r6ncvd` → 项目 `curious-helix-508400-e4`**（`m5_pids.json` 权威；`coherent-mesh-509212-u2` 属 `eib2z5bs`，是 health 中被淘汰的 restricted-mode 账号，**本实验从未使用**）。5 次运行的 `logs/*/run.log` 逐条打印 `账号 k3r6ncvd / 项目 curious-helix-508400-e4`，无第二个项目、无串号。健康记录 `reg/health_geometry.json`（3 候选串行、零 EECU）。

## 1.3 运行序列与计时（全部来自 `run_results/*.json`，不挑较短值）

| 顺序 | exp_id | 状态 | wall_s | limit_s | rc | 说明 |
|---|---|---|---:|---:|---:|---|
| 1 | `G30N_tiny` | done | 9.7 | 300 | 0 | 8×8 A00 连通/时延测试 |
| 2 | `G30N_seed7` | done | 33.3 | 1200 | 0 | **VALIDITY_FAIL**（impl v1 未物化掩膜） |
| 3 | `G30N_seed7b` | done | 14.2 | 1200 | 0 | 首个修正版单种子成功 |
| 4 | `G30N_seedset7_11_23` | done | **76.0** | 1200 | 0 | **首个 3 种子成功**（tag 共用 → 逐种子 eval 被覆盖） |
| 5 | `G30N_seedset_v2` | done | 56.2 | 1200 | 0 | **复核重跑**（per-seed tag，产出逐种子独立产物） |

即：**首次 3 种子成功 = 76.0s（第 4 次）**；56.2s 是随后为逐种子产物可追溯而做的复核重跑（第 5 次）。两者数值逐项相同；主文引用时应说明是同一配置的两次运行，不取较短值单独呈现。逐种子单请求耗时（第 5 次）：3.5 / 37.6 / 2.0 s。

## 2. 登记值（运行前冻结）

| 项 | 值 |
|---|---|
| 区域 | 川 T1206 瓦南缘楔形中心：`cols[10854,11411) × rows[22192,22285)` = 557×93 px；bbox `[97.975026, 29.998161, 98.025062, 30.006515]`（含 30.0 接缝南侧 20 行作对照） |
| 网格 | 生产资产真实 affine：`px=py=8.983152841195215e-05°`，`x0=96.99999454769751`，`y_top=32.00005654549083`（取 `T1206_10m.tif` 表头）；显式 `crsTransform`，**不传 scale** |
| 模型 | 生产原表达式：池 `smp2023v2_merged` / `filterBounds(box±2°=95–101,28–34)` / `randomColumn(seed).sort.limit(20000)` / `smileRandomForest(100,minLeaf2,maxNodes5000,seed)` / AEF2023 64 维 |
| 两臂 | 仅 **clip 几何**不同：`legacy = ee.Geometry.Rectangle([4数字])`（客户端→`GeometryConstructors.Polygon`+数字坐标→服务端 **geodesic=true**）vs `planar = ee.Geometry.BBox(...)`（显式 `geodesic=false`）。**训练选择固定为同一 legacy 表达式** |
| 种子 | `{7,11,23}`（一次 harness 内串行，总 wall 76.0s） |
| deadline | 1200s/单元；父进程在首个请求前起算；单请求 `ee.data.setDeadline(min(60, 剩余-5)s)`、`setMaxRetries(0)` + 自管重试；不发 batch |
| 账号 | `k3r6ncvd`（proj `curious-helix-508400-e4`；ID 记录，无密钥） |

## 2.1 调色板事实澄清（audit 提问）

| 文件 | ColorMap 标签 | `PhotometricInterpretation` | 读取器是否显示调色板 |
|---|---|---|---|
| `out/G30N_*_{legacy,planar}_10m.tif` | **有**（256 项） | `1`（MinIsBlack） | ❌ 部分读取器/QGIS 报告“无调色板” |
| `delivery/G30N_*_{legacy,planar}_10m_palette.tif` | **有**（256 项） | **`3`（Palette）** | ✅ 正常 |
| `out/G30N_*_5band.tif` | **无**（多波段不做 Palette） | `1` | — （分析件，非出图件） |

即：调色板数据一直存在，问题只是 `photometric` 标签写成 MinIsBlack。修正件由 `g2_deliver_palette.py` 生成（只读源件、写新文件名、逐像元校验数据一致、`photometric=3`），**未覆盖任何原件、原件 MD5 不变**；双 hash 见 `delivery/MANIFEST_MD5.txt`。`write_tif` 已同步修正，后续运行直接产出 `photometric=PALETTE`。

## 3. 实测结果（种子 7/11/23 逐项相同）

| 指标 | 值 |
|---|---|
| EVM∩平面 tile box（分母） | 40,661 px |
| legacy 输出 nodata（工程缺口） | **23,394 px** |
| planar 输出 nodata | **0** |
| recovered（legacy 空 → planar 有值） | **23,394 px（恢复率 1.0）** |
| 楔外 legacy==planar 类码一致率 | **1.000**（分母 16,710 px，分歧 0） |
| tile box 外（南侧 20 行） | legacy/planar 均 0 非零（一致 nodata）；EVM 有效 11,140 px **另单列，不计缺口** |
| 实测南边界 | legacy 30.00378°（解析拱高 0.0037791° = 42.07 行，吻合）；planar 30.0000° |
| 单请求耗时 | 3.6 / 31.5 / 23.4 s；总 wall 76.0s / 1200s |

### 3.1 分母对账（解释 557 = 1 行差）

- 解析楔形（大圆拱）在 tile 内 = 43 行 = **23,951 px**（行 22222–22264）
- legacy 实测 nodata = 42 行 = **23,394 px**（行 22223–22264）
- 差 **557 px = 恰好 1 行（行 22222）** → 解析弧与**实际栅格化边界**的像元中心约定差（~0.5 px，与 seams 报告「mean −0.53 px」一致），非缺陷。
- 共同有效域 `common = in_tile ∩ legacy>0 ∩ planar>0` = **17,267 px** = 40,661 − 23,394 ✓
- `outside_wedge`（报告分母）= `common ∩ ¬lens` = **16,710**；边界行 = `common ∩ lens` = **557**
- **两域分歧均为 0**：outside_wedge 0/16,710；边界行 0/557；合计 0/17,267 → 该 557 px 不隐藏任何 legacy/planar 分歧。

## 4. 关键实现修正与失效记录（不得覆盖）

| 实验 | 状态 | 说明 |
|---|---|---|
| `G30N_tiny` | done | 8×8 A00 computePixels 连通/时延：0.8s，真值 min/max −0.125/−0.103 |
| `G30N_seed7` | **VALIDITY_FAIL** | impl v1：**未物化掩膜** → `ee.data.computePixels`(GEO_TIFF) **不传递图像掩膜**，被 clip 掩掉的像元返回底层数值 → legacy/planar 均零 nodata，**不能作几何结论**。原件保留（`out/G30N_seed7_*`、`run_results/G30N_seed7.json`） |
| `G30N_seed7b` / `G30N_seedset7_11_23` | done | impl v2：类别波段 `.unmask(0, False)` 显式填 0 + `legacy_mask` 波段交叉核对 → 复现并验证 |
| `G30N_seedset_v2`（v2_s7/s11/s23） | done | 最终版：per-seed 独立产物 |

**参数与门槛自始至终未改**；修正仅是实现层（掩膜物化）。

## 5. 探针结论（供复核）

1. `g0b_probe_geom`（常量图）：`Rectangle([4数字])` 与 `Rectangle(box,None,True)` 逐行相同、南缘 lat<30.00378° 整行置零 → 客户端 `Rectangle([数字])` 确为**测地**（seams 报告归因成立，**非硬拟合**）。
2. `g0c_probe_aef`（AEF 路径）：4 种 clip 与 noclip **逐值全同**；`unmask(0)` 后 34,537 px 显 0 → **掩膜存在但 computePixels 不传递**。
3. `g0_probe_mask`：`updateMask(0)` 与 `clip` 掩膜在常量图上均可见 → 排除「computePixels 完全不保掩膜」的过强结论；差异来自波段类型/来源路径，故**验收一律显式物化**。

## 6. 边界与未执行项

- 本窗仅 0.05° 宽，楔弧在窗内近乎水平（弧在两角退回 y0 的距离约 1°）→ 窗内呈「整行带状」而非完整透镜；透镜形状由解析式 `geom_planar.wedge_lens_mask` 给出（全瓦 43 行中心对称，与 seams 报告一致），**未做整 2° 宽导出**（超交互窗）。
- 未改冻结 c3；未启 batch；未做第二区域；未做其它年份/瓦扫描。
- 独立复核：请 audit 子 agent 用 `out/G30N_v2_s*_5band.tif` + `out/G30N_reconcile.json` 复算（纯 numpy，无需云端）。
