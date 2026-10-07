# 模型训练与 GEE 迁移部署（专篇）

> 回答"模型怎么训练、怎么变成 GEE 能分类的模型"。核心方案（与《05 计算资源优化》一致）：**本地 sklearn 训练随机森林 → 转换成 GEE `ee.Classifier.decisionTreeEnsemble` 决策树 → GEE 内只做推理**。另附两条备选路线（GEE 原生 RF、原型余弦分类）及其选型依据。

---

## 目录

1. [三条路线总览与选型](#1-三条路线总览与选型)
2. [本地训练管线（sklearn）](#2-本地训练管线sklearn)
3. [特征顺序一致性约定（最易踩的坑）](#3-特征顺序一致性约定最易踩的坑)
4. [转换原理：sklearn 树 → ee.Tree](#4-转换原理sklearn-树--eetree)
5. [转换代码实现](#5-转换代码实现)
6. [GEE 端使用](#6-gee-端使用)
7. [转换正确性校验（必须做）](#7-转换正确性校验必须做)
8. [备选路线 A：GEE 原生 RF](#8-备选路线-agee-原生-rf)
9. [备选路线 B：原型余弦分类（免训练）](#9-备选路线-b原型余弦分类免训练)
10. [常见坑与对策](#10-常见坑与对策)

---

## 1. 三条路线总览与选型

| 路线 | 训练在哪 | 模型怎么进 GEE | 优点 | 缺点 | 适用 |
|---|---|---|---|---|---|
| **A. 本地 sklearn → 迁移**（主推） | 本地，免费、可调参、可类加权 | 转换决策树 → `ee.Classifier.decisionTreeEnsemble` | 训练零 GEE 配额；类权重/网格搜索/概率输出全支持 | 需写转换代码 + 严格校验 | 主分类器 |
| **B. GEE 原生 RF** | GEE 内 | 训练后存 Asset 直接复用 | 零转换代码；原生概率输出 | 训练烧 GEE 配额；调参/类权重不便 | 疑难分区兜底 |
| **C. 原型余弦** | 本地算类原型向量 | 上传原型 FeatureCollection，GEE 内 argmax(v·μₖ) | 免训练、推理极廉价；官方推荐范式 | 精度取决于类内聚度 | 粗筛、变化像元快分 |

本专篇详解 A；B、C 见第 8、9 章。

---

## 2. 本地训练管线（sklearn）

### 2.1 数据准备（来自 03/05 文档）

样本表已本地化（Parquet）：`A00..A63`（或特征选择后的 top-M）+ `class` 类别码 + `tile_id` 等。**训练输入 = 提纯后的黄金样本 + 对应年份嵌入特征**。

### 2.2 训练代码

```python
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

# 1) 样本表（本地，见 05 文档"样本嵌入一次性导出"）
df = pd.read_parquet('data/samples_cn_2020.parquet')

# 特征列：顺序一经确定，GEE 侧必须完全一致（见第 3 章）
FEATS = [f'A{i:02d}' for i in range(64)]          # 或特征选择后的 top-M
X = df[FEATS].to_numpy()
y = df['class'].to_numpy()                          # 类别码（数值，如 10/12/51…）

# 2) 切分：按 tile 分组切分，防止相邻像元空间泄漏导致验证虚高
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, va_idx = next(gss.split(X, y, groups=df['tile_id'].to_numpy()))
Xtr, Xva, ytr, yva = X[tr_idx], X[va_idx], y[tr_idx], y[va_idx]

# 3) 类权重：按"1/√面积"或 'balanced'（1/样本数），稀有类提权
cw = compute_class_weight('balanced', classes=np.unique(ytr), y=ytr)
cw_map = dict(zip(np.unique(ytr), cw))

# 4) 训练（参数与 SEA-PNF 对齐：300 棵树、sqrt 特征数、叶 5 样本）
rf = RandomForestClassifier(
    n_estimators=300,
    max_features='sqrt',        # 对应 GEE variablesPerSplit≈√64
    min_samples_leaf=5,         # 对应 minLeafPopulation
    class_weight=cw_map,
    oob_score=True,
    n_jobs=-1, random_state=42,
)
rf.fit(Xtr, ytr)

# 5) 本地评估（独立验证集）
print(classification_report(yva, rf.predict(Xva)))

# 6) 保存：模型 + 特征顺序 + 类别清单（一份 json 配置，转换/校验/GEE 端共用）
import json
joblib.dump(rf, 'models/rf_cn2020_zone01.joblib')
json.dump({'features': FEATS, 'classes': rf.classes_.tolist()},
          open('models/rf_cn2020_zone01_cfg.json', 'w'))
```

### 2.3 分区训练的组织

160 个 2° 分区 × 9 年 = 至多 1440 个模型。实际做法：

- **基准年模型**：2020 年每分区 1 个（160 个），全图重锚用；
- **非重锚年**：复用邻近基准模型，只对变化像元重分类（04 文档），无需逐年新模型；
- 每个模型 = `rf_{zone}.joblib + cfg.json`，本地一次跑完（sklearn 多核，几十分钟到几小时）。

---

## 3. 特征顺序一致性约定（最易踩的坑）

sklearn 树里存的 `feature` 是**训练矩阵的列索引**（0 基）。转换后 GEE 端 `features` 列表的顺序、以及分类时 `image.select()` 的波段顺序，**三者必须完全一致**，否则树会拿错特征。

**统一约定**（用一份共享 json 保证）：

```json
{
  "features": ["A00","A02","A05", ...],   // 训练列序 = 此序
  "classes": [10, 11, 12, ...]            // 类别码，叶子 label 用
}
```

- Python 端：`X = df[FEATS]`，FEATS 与 json 一致；
- 转换端：把该 `features` 数组原样传给 `decisionTreeEnsemble`；
- GEE 端：`image.select(FEATS)`，FEATS 顺序与 json 一致。

---

## 4. 转换原理：sklearn 树 → ee.Tree

### 4.1 决策树结构的对应关系

sklearn 的 `RandomForestClassifier.estimators_` 里每棵树，其 `tree_` 存于 `sklearn.tree._tree.Tree`：

| sklearn 字段 | 含义 | GEE `ee.Tree` 字段 |
|---|---|---|
| `children_left[i]` | 节点 i 左子树（-1=叶子） | `left`（内部节点） |
| `children_right[i]` | 节点 i 右子树 | `right` |
| `feature[i]` | 分裂用特征索引（训练矩阵列号） | `featureIndex` |
| `threshold[i]` | 分裂阈值（`X <= 阈值 → 左`） | `value`（GEE 同为 `特征<=value → 左`，约定一致） |
| `value[i][0]` | 叶子处的各类样本数（n_classes 向量） | 叶子：`label`（argmax 后的类别）或 `probabilities`（归一化向量） |
| `classes_` | 内部类索引 → 原始类别码的映射 | 叶子 label 直接用原始类别码 |

### 4.2 两种叶子

- **label 叶子**（推荐）：叶子存 argmax 后的类别码，输出硬分类。官方示例格式，稳定可靠：
  ```json
  {"label": 51}
  ```
- **probabilities 叶子**：叶子存归一化的类概率向量：
  ```json
  {"probabilities": [0.02, 0.85, 0.13]}
  ```
  > ⚠️ GEE 对概率向量与类别码的**对齐规则需实测确认**（大概率按类别值排序对齐）。**先在小样本子集上验证顺序正确再用于生产**；若需要稳妥的概率输出，直接退回路线 B（GEE 原生 RF）。

### 4.3 树的 JSON 规模

300 棵树 × 深 ~15–20 层 ≈ 数千节点，序列化后约 1–3 MB。`decisionTreeEnsemble` 在代码内构造，GEE 代码长度一般可容纳；若超限，减到 100–150 棵树或按分区拆脚本。

---

## 5. 转换代码实现

```python
# convert_sklearn_to_gee.py
import json, ee
import joblib

def sklearn_tree_to_ee_tree(t, classes, use_probabilities=False):
    """t: sklearn tree_.Tree；classes: rf.classes_（内部索引→类别码）"""
    def build(idx):
        if t.children_left[idx] == -1:                # 叶子
            probs = t.value[idx][0]                    # 各类样本数
            probs = probs / probs.sum()                # 归一化
            if use_probabilities:
                return ee.Tree(ee.Dictionary({'probabilities': probs.tolist()}))
            k = int(np.argmax(probs))                  # 预测类（内部索引）
            return ee.Tree(ee.Dictionary({'label': int(classes[k])}))
        return ee.Tree(ee.Dictionary({
            'featureIndex': int(t.feature[idx]),       # 训练列索引！
            'value': float(t.threshold[idx]),          # 阈值
            'left': build(t.children_left[idx]),
            'right': build(t.children_right[idx]),
        }))
    return build(0)

def rf_to_gee_classifier(rf, features, use_probabilities=False):
    trees = [sklearn_tree_to_ee_tree(est.tree_, rf.classes_, use_probabilities)
             for est in rf.estimators_]
    return ee.Classifier.decisionTreeEnsemble({
        'features': features,          # 必须与训练列序完全一致（第 3 章）
        'trees': trees,
    }), trees

if __name__ == '__main__':
    import numpy as np
    rf = joblib.load('models/rf_cn2020_zone01.joblib')
    cfg = json.load(open('models/rf_cn2020_zone01_cfg.json'))
    clf, trees = rf_to_gee_classifier(rf, cfg['features'])
    # 输出：一棵树的调试预览 + 供 GEE 脚本嵌入的完整 json
    print(ee.Tree.toDictionary(trees[0]).getInfo())    # 目检根节点
    json.dump([t.serialize() for t in trees],
              open('models/rf_cn2020_zone01_gee_trees.json', 'w'))
```

> 注意：`ee.Tree.serialize()` 输出的是 ee.Tree 的 JSON 表示；生产时把各分区 trees json 用**脚本模板生成器**嵌入对应分区的 JS 脚本（迁移模型无法存 Asset，只能在代码内构造——每分区脚本内联自己的树 json）。

---

## 6. GEE 端使用

```js
// classify_with_migrated_rf.js（由脚本模板生成器按分区产出）

// 1) 特征顺序 = 训练列序（从 cfg.json 复制，顺序不可变）
var FEATS = ['A00','A02','A05', /* ... top-M 或 64 个，顺序与训练一致 */ ];

// 2) 迁移模型：树 json 由本地转换脚本生成后内联（或读取共享资产配置）
var rf = ee.Classifier.decisionTreeEnsemble({
  features: FEATS,
  trees: [ /* 本分区 rf_cn2020_zone01_gee_trees.json 内容 */ ]
});

// 3) 取目标年嵌入（按瓦片裁剪），只选训练用到的特征
var emb = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
            .filterBounds(tile)
            .filterDate('2020-01-01','2021-01-01')
            .first()
            .select(FEATS);

// 4) 分类（重锚年全图 / 变化像元段均值影像均可）
var cls = emb.classify(rf, 'class');

// 5) 后处理与导出（UInt8、按 1° 瓦片、crsTransform 对齐；见 02/04/05）
var out = cls.clip(tile).uint8();
Export.image.toDrive({image: out, region: tile, maxPixels: 1e13, ...});
```

---

## 7. 转换正确性校验（必须做）

**原则：GEE 端推理必须与本地 sklearn 逐点一致**，不一致即转换有 bug（多为特征顺序或阈值精度）。

```python
# verify_gee_vs_sklearn.py
# 1) 取 5000 个样本点（含类别），本地 rf.predict → y_sklearn
# 2) GEE 端：同样的点，用 decisionTreeEnsemble classify → y_gee
# 3) 对比混淆矩阵：不一致率应为 0（或 <0.1%，阈值 float32 舍入边界）
```

必查项：

- **特征顺序**：FEATS 错序是 100% 出错的典型原因；
- **阈值精度**：sklearn 用 float64 阈值，GEE 端序列化后按 float64 传输，一般无损；若边界样本不一致，检查 `tree_.threshold` 是否被 json 截断（保留 ≥10 位小数）；
- **类别映射**：`rf.classes_` 到 label 的映射错误会导致错类；
- **概率叶子顺序**：若用 probabilities 叶子，先验证类别对齐（见 4.2）。

---

## 8. 备选路线 A：GEE 原生 RF

不做转换，直接在 GEE 内训练（训练本身烧配额，但逻辑最简单，且**原生支持概率输出**）：

```js
// 1) 从黄金样本 FC 中抽取训练点，采样嵌入值
var training = emb.sampleRegions({
  collection: samplesFC,          // 提纯样本点
  scale: 10,
  properties: ['class']
});

// 2) 训练
var rf = ee.Classifier.smileRandomForest({
  numberOfTrees: 300,
  variablesPerSplit: 8,           // ≈√64
  minLeafPopulation: 5
});
var trained = rf.train(training, 'class', FEATS);

// 3) 分类 + 概率输出（HMM/平滑可直接用）
var cls = emb.classify(trained, 'class');
var prob = emb.classify(trained.setOutputMode('PROBABILITY'), 'prob');
```

**用法**：仅对转换/余弦都不达标的少数疑难分区兜底；训练一次后模型存 Asset 复用。

---

## 9. 备选路线 B：原型余弦分类（免训练）

**本地算类原型**（Karcher 均值 = 归一化均值向量），上传 FeatureCollection，GEE 内做"与每个原型点积、取 argmax"：

```python
# 本地
protos = df.groupby('class')[FEATS].apply(
    lambda g: g.to_numpy().mean(axis=0))
protos = protos.apply(lambda v: v / np.linalg.norm(v))   # 单位向量
protos.to_csv('data/prototypes_2020.csv', index_label='class')
# 上传 → 每个原型一条记录（属性 class + A00..A63），存 Asset
```

```js
// GEE：把原型变成"每类一个波段"的影像，做矩阵点积
var protoFC = ee.FeatureCollection('projects/xxx/assets/prototypes_2020');
// 用 protoFC 构建原型影像（band k = 原型 k 的 64 维），或用 reduceRegion 展开
var protosArr = protoFC.map(function(f){
  return f.toDictionary(protoFC.first().propertyNames());
});
// 简化示意：逐原型点积
var sims = ee.ImageCollection(protoFC.toList(30).map(function(f){
  var p = ee.Feature(f);
  var pImg = ee.Image.constant(p.toDictionary().values()).rename(FEATS); // 原型影像
  return emb.multiply(pImg).reduce(ee.Reducer.sum()).rename(p.get('class'));
}));
var cls = sims.toBands().reduce(ee.Reducer.max());  // 或 argmax 得类别索引
```

**适用**：粗筛（05 文档级联第一级）、变化像元快速重分类、低配环境主分类；若某类余弦精度不达标，回落路线 A 或原生 RF。

---

## 10. 常见坑与对策

| 坑 | 现象 | 对策 |
|---|---|---|
| 特征顺序不一致 | GEE 分类结果大面积错乱 | 统一 cfg.json 单一事实来源；校验脚本必跑 |
| 类别码映射错 | 少数类全错 | 校验混淆矩阵逐类核对 |
| 阈值 float32 舍入 | 边界样本不一致（<0.1%） | 阈值保留 ≥10 位小数；接受 <0.1% 边界差异 |
| probabilities 叶子类别顺序未验证 | 概率错位 → 平滑/面积统计失真 | 先小样本验证；要稳概率就退回原生 RF |
| 树 json 太大 | GEE 代码/请求超限 | 减树至 100–150，或按分区拆脚本 |
| 迁移模型无法存 Asset | 每脚本要内联树 json | 脚本模板生成器（Python 产出分区 JS）统一管理 |

---

## 附：关联文档

- 《05 计算资源优化与本地GEE分工》：为何本地训练、GEE 推理（配额金线）
- 《03 高精度样本构建方法论》：训练样本的来源与提纯
- 《02 主技术方案》第 8.4：导出规范（分类结果落盘）
- Google Earth Engine API：`ee.Classifier.decisionTreeEnsemble`、`ee.Tree`、`smileRandomForest`（官方文档为准）
