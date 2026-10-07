# -*- coding: utf-8 -*-
"""
lc06_convert_gee.py — sklearn 随机森林 → GEE 决策树JSON（06文档 §5）
* 纯 Python 生成"与 ee.Tree 序列化同构"的嵌套 dict：
    内部节点 {'featureIndex': int, 'value': float, 'left':..., 'right':...}
    叶节点   {'label': int}
* 不依赖 GEE 即可生成；生产时把此 dict 交给 ee.Classifier.decisionTreeEnsemble({features, trees})。
运行: python lc06_convert_gee.py --model models/rf.joblib --cfg models/rf_cfg.json --out models/rf_gee_trees.json
"""
import os, json, argparse
import numpy as np, joblib

def sk_tree_to_dict(tree, classes, use_probabilities=False):
    feats = tree.feature.tolist()   # 节点用其特征索引
    cl = tree.children_left.tolist(); cr = tree.children_right.tolist()
    val = tree.threshold.tolist()
    values = tree.value             # [n_nodes, 1, n_classes] 样本计数
    classes = np.asarray(classes)

    def build(idx):
        if cl[idx] == -1:           # 叶子
            p = values[idx][0].astype(float); p = p / p.sum()
            if use_probabilities:
                return {'probabilities': p.tolist()}
            return {'label': int(classes[int(np.argmax(p))])}
        return {'featureIndex': int(feats[idx]), 'value': float(val[idx]),
                'left': build(cl[idx]), 'right': build(cr[idx])}
    return build(0)

def rf_to_gee(rf, use_probabilities=False):
    return [sk_tree_to_dict(est.tree_, rf.classes_, use_probabilities)
            for est in rf.estimators_]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--cfg', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rf = joblib.load(a.model)
    cfg = json.load(open(a.cfg))
    trees = rf_to_gee(rf)
    payload = {'features': cfg['features'], 'classes': cfg['classes'],
               'n_trees': len(trees), 'gee_classifier': 'ee.Classifier.decisionTreeEnsemble'}
    payload['trees_mirror_json'] = trees   # 供校验；生产用 ee.Tree 包装
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(payload, open(a.out, 'w'), ensure_ascii=False)
    print('转换完成: 树数 =', len(trees))
    print('样例(首棵树根):', json.dumps(trees[0], ensure_ascii=False)[:180])
    print('输出:', a.out)

if __name__ == '__main__':
    main()
