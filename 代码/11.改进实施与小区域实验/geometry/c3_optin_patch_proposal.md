# c3 生产驱动「平面几何」安全 opt-in 补丁方案（默认不改冻结 c3）

> 生成 2026-10-15，几何子 agent。依据 doc70 §4.4「默认生产 c3 不直接改写，新实验驱动可明确选择 legacy-geodesic 或 planar」。
> 本文件是**方案 + 可直接套用的最小 diff**，**未应用到 `代码/9.第二次分类川粤宁/c3_driver_prod2.py`**；是否启用由主 agent 决定。

## 1. 现状（只读核对）

`代码/9.第二次分类川粤宁/c3_driver_prod2.py`：

| 行 | 代码 | 语义 |
|---|---|---|
| L211-213 | `boxp = ee.Geometry.Rectangle([x0-2, y0-2, x1+2, y1+2])` | 训练 `filterBounds`；数字坐标 → 客户端展开为 `GeometryConstructors.Polygon`（数字坐标）→ 服务端 Polygon 默认 **geodesic=true** |
| L220 | `box = ee.Geometry.Rectangle(meta['box'])` | **推理 clip 几何**；同上 → 测地 |
| L224 | `cls = aef.classify(clf).rename('class').uint8().clip(box)` | 掩膜在此落地 |
| L228-230 | `Export.image.toAsset(..., scale=10, crs='EPSG:4326', region=meta['box'], ...)` | `region=[4 数字]` → `_canonicalize_region` 转 `Geometry.LineString([x0,y0,x1,y1])`（对角线段，仅取 bounds 作窗口）→ 同为测地默认 |

**实测依据（本工作区，2026-10-15）**：
- `g0b_probe_geom.py`：常量图下 `Rectangle([4数字])` 与 `Rectangle(box,None,True)` 逐行相同、南缘 lat<30.00378° 整行置零 → 客户端 `Rectangle([数字])` 确为**测地**（seams 报告归因成立）。
- `g0c_probe_aef.py`：`ee.data.computePixels`(GEO_TIFF) **不传递图像掩膜**（clip 前后逐值相同；`unmask(0)` 后 34,537 px 显 0）→ 任何以「导出后数 nodata」为手段的几何验收，**必须显式物化掩膜**（`.unmask(0)` 或单独 `mask()` 波段），否则自证失效。

## 2. 补丁设计原则

1. **默认零行为变化**：新增开关 `--plane-geom`，缺省 `False` → 完全走现有代码路径，字节级不变。
2. **平面语义一次性贯通三处**：`clip` 几何、`region` 几何、训练 `filterBounds` 几何分别可独立选择，避免「只改一处仍留测地」。
3. **crsTransform 与 scale 互斥**：显式 `crsTransform` 时不再传 `scale`（GEE Export 规则），由 helper 强制校验。
4. **训练窗修改单列**：改 `filterBounds` 会改变抽样选择 → 必须另立模型版本并落盘实际选中 ID（doc70 §8 风险表「原几何同时影响训练窗」）。
5. **不引入新依赖**：helper 为纯函数模块，可内联复制。

## 3. 最小 diff（建议形式）

在 `c3_driver_prod2.py` 顶部新增：

```python
# ---- 几何语义开关（doc70 §4.4；默认 False = 现有冻结行为，字节不变）----
PLANE_GEOM = bool(VC.jload(os.path.join(ROOT, 'config.json'), {}).get('plane_geom', False))
PLANE_TRAIN = bool(VC.jload(os.path.join(ROOT, 'config.json'), {}).get('plane_train', False))


def _rect(box, planar):
    """planar=True → ee.Geometry.BBox（客户端显式 geodesic=False）；False → 现有 Rectangle([4数字])。"""
    if planar:
        return ee.Geometry.BBox(float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    return ee.Geometry.Rectangle(list(box))
```

`submit_tile()` 内三处替换：

```python
-    boxp = ee.Geometry.Rectangle([meta['box'][0] - 2, meta['box'][1] - 2,
-                                  meta['box'][2] + 2, meta['box'][3] + 2])
+    _bp = [meta['box'][0] - 2, meta['box'][1] - 2,
+           meta['box'][2] + 2, meta['box'][3] + 2]
+    boxp = _rect(_bp, PLANE_TRAIN)          # 默认 False → 原行为
     fc = (ee.FeatureCollection(SRC).filterBounds(boxp)
           .randomColumn('rc', 7).sort('rc').limit(20000))
     ...
-    box = ee.Geometry.Rectangle(meta['box'])
+    box = _rect(meta['box'], PLANE_GEOM)     # 默认 False → 原行为
     aef = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
            .filterDate('%d-01-01' % YEAR, '%d-01-01' % (YEAR + 1))
            .filterBounds(box).mosaic().select(VC.FEATS))
     cls = aef.classify(clf).rename('class').uint8().clip(box)
     ...
-    task = ee.batch.Export.image.toAsset(image=cls, description=desc, assetId=aid,
-                                         scale=10, crs='EPSG:4326', region=meta['box'],
-                                         maxPixels=10 ** 12, shardSize=16)
+    # region 传**平面 Geometry 对象**（不能传 4 数字列表：_canonicalize_region 会转成测地默认几何）
+    task = ee.batch.Export.image.toAsset(image=cls, description=desc, assetId=aid,
+                                         scale=10, crs='EPSG:4326',
+                                         region=_rect(meta['box'], PLANE_GEOM),
+                                         maxPixels=10 ** 12, shardSize=16)
```

可选：显式 `crsTransform` 对齐生产格（避免 scale 吸附相位漂移）——此时**必须**去掉 `scale`：

```python
# 仅在 PLANE_GEOM 且显式对齐时启用
_ct, _gw, _gh = VC.geom_grid(meta['box'], px=8.983152841195215e-05,
                             x0=96.99999454183633, ytop=40.00000330580025)   # 全局方度格
assert (_ct is None) or (10 is None), 'crsTransform 与 scale 互斥'
task = ee.batch.Export.image.toAsset(image=cls, description=desc, assetId=aid,
                                     crs='EPSG:4326', crsTransform=_ct,   # 不再传 scale
                                     region=_rect(meta['box'], PLANE_GEOM),
                                     maxPixels=10 ** 12, shardSize=16)
```

## 4. 启用后的验收门（doc69 §6.4 / doc70 §7 几何覆盖）

1. **EVM 门（金标准）**：`aef.mask().reduce(ee.Reducer.min())`，**不先 clip**、平面 region、固定 grid 导出；在「平面 tile box 内 ∩ EVM=1」上要求输出 nodata = 0（或预注册容差）。本工作区已实测该门在本瓦小窗上 **legacy 23,394 px 缺口 → planar 0**。
2. **楔外一致**：非楔区 legacy/planar 类码逐像元一致率 = 1.0（本工作区实测 1.000，分母 16,710 px）。
3. **物化掩膜纪律**：任何「导出后数 nodata」的检查必须 `unmask(0)` 或单列 `mask()` 波段；`computePixels` 不传掩膜（实测）。
4. **边缘确定性扫描**：四边/角点 ≥128 行窄带（doc69 §6.4 第 5 条）——旧 QA 的 `y0+0.02°` 内缩抽样对楔高（≤0.0042°）**概率为 0**，必须换门。
5. **训练窗若同时改**（`PLANE_TRAIN=True`）：单列模型版本、落盘实际选中 ID、独立影响报告；**不与纯推理几何修复混算**。

## 5. 不建议的做法

- 直接 `build_overviews`（doc43 已证确定性写坏概览页）→ 概览一律走 `ov_build.py` + PIL 全盘验收。
- 用邻行/邻瓦插值填补楔区（seams 报告 §7 S3 已证拼接无法自愈，且污染类别统计）。
- 把 `region` 继续传 4 数字列表（`_canonicalize_region` → LineString → 测地默认），只改 `clip` 一处。
- 对 2017-2024 全部 313 瓦直接放量：应先按 T1d/T3d 诊断扫一遍其它年份同源风险（seams 报告 §10.5）。
