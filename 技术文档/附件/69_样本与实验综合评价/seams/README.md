# seams/ 证据清单（栅格空白行 / 网格接缝工程诊断，2026-10-15）

**主报告：`报告_空白行_测地楔形根因_2026-10-15.md`**（结论、根因链、行号、影响量、修复与验收门）

## 一句话结论
用户报的 30.00149246°「一行空白」= 每个 2° 瓦片**南边缘**的对称三角形 nodata（楔），
成因是 GEE 侧 `ee.Geometry.Rectangle(meta['box'])` 取服务端默认 `geodesic=True`：clip/export 的南边是
**球面大圆**，在中经线向北拱 0.0030–0.0042°（31–47 px @24–40°N），拱与瓦片南网格线之间的透镜区被排除。
不是拼接 bug、不是 AEF 缺测；第一批 74 瓦同样带病（掩膜逐值相同）。

## 文件与产出

| 文件 | 内容 / 单位 | 生成命令 |
|---|---|---|
| `报告_空白行_测地楔形根因_2026-10-15.md` | 主报告 | — |
| `s1_rowscan.py` | 指定行区间逐行统计（nz / 预期覆盖掩膜内缺失 / 最长内部零游程） | `python s1_rowscan.py --file <tif> --r0 111290 --r1 111335 --plan <plan.json> --tag mos2nd` |
| `rowscan_mos2nd_111290_111335.csv` | 总幅 46 行剖面（行号 / 上下纬 / nz / 零游程） | 同上 |
| `s2_geodesic_test.py` / `geodesic_test.json` | 测地大圆拱预测 vs 实测掩膜（5 瓦：Jaccard、逐列深度误差、拱高行数） | `python s2_geodesic_test.py` |
| `s3_impact.py` / `impact_prov.json` / `impact_prov.csv` | 楔∩nodata：省内（DataV GCJ-02 raw + buffer −0.005，像元中心）与 bbox 上界 | `python s3_impact.py` |
| `g1_boundary_continuity.py` | **诊断档**实现（T1d 测地透镜计数 / T2d 逐行内部断带 / T3d 覆盖率粗对照）；文件头写明正式阻断门 G-SEAM 定义（需 EVM=输入有效掩膜，先预注册） | `python g1_boundary_continuity.py tiles --tiles-dir ...` |
| `gate_report_2nd41.json` / `gate_rows.csv` / `gate_2nd41.log` | 第二批 41 瓦诊断：41/41 报出南边界透镜 nodata>0（`grade=DIAGNOSTIC`，`blocking_gate=NOT_EVALUATED`） | 见上 |
| `gate_report_1st74.json` / `gate_1st74.log` | 第一批 74 瓦诊断：74/74 报出透镜 nodata>0，且与第二批**逐值相同** | `--plan F:/lc_work/prod5p_2023/plan/tiles_frozen.csv --no-t2` |
| `sim_fill_order.py` / `sim_fill_order.json` | 拼接覆盖顺序模拟 4 案（S1 无误、S2 复现 1 行零覆盖、S3 源端 nodata 不可自愈、S4 覆盖与顺序无关） | `python sim_fill_order.py` |
| `lens_area_2nd.json` | 41 瓦解析楔面积（含海面 2,109 km²） | 见报告 §6 |
| `firstbatch_south_scan.json` | 第一批 76 瓦南边缘扫描（76/76 有楔） | 见报告 §5 |
| `lens_sample_hits.json` | 落在楔形区内的样本池点（521 点，均带 AEF 64 维真值） | 见报告 §4 |

## 关键数字（速查）
- 用户行：总幅 row 111302 / 四川省件 row 47937 / 瓦片 T1206 row 22247（纬度带 30.00148470–30.00157453）
- 楔：T1206 瓦片行 22223–22264（42 行），中心列 11131.5，末行零游程 66–22197（仅 132 px 有值）
- 实测楔高 vs 大圆拱预测：42/42.07、40/40.27、39/38.28、47/47.13、34/33.74 行（Jaccard 0.977–0.982）
- 省内**候选缺口**（楔∩nodata，省界内）：川 597.27 km²、粤 180.80 km²、宁 86.96 km²，合计 **865.0 km²**
  （候选缺口 = 楔形 nodata 计数；「楔区本应有 AEF 有效输入」仅由 521 个楔内池点支持，未逐像元证实）
- 既有 QA 漏检：`c3_driver_prod2.py` L392-393 抽样纬度内缩 0.02°（楔 ≤0.0042° 永不被采到）+ L417 255↔0 视作一致 + `c5_mosaic_compare.py` L291-313 自证 + 无覆盖率门
- 交付同源：`delivery/成品/分省/四川_2023_10m.tif` 与 `mosaic2/四川_2023_10m.tif` MD5 同为 `c2b0ffbd748f681244b372f2db9ef639`

## 未决 / 限制
- `clip(box)` 与 `Export.region=` 谁落地成掩膜：本地无法二分（两者同为测地默认）；未做云端验证。
- 「AEF 该处有数据」由 521 个楔内池点支持（点证据），非全区逐像元证明。
- 本报告仅 2023 年第一/第二批；其它年份若沿用同一 box 构造，风险同源。
