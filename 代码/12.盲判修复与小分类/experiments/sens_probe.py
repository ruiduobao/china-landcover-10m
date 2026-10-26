# -*- coding: utf-8 -*-
"""sens_probe.py — 条件敏感性探针（描述性研究，非采纳试验）

依据文档72：以首遍盲判（pass1，解盲前已冻结哈希）为依据，检验
"隔离与园地先验矛盾的 R 训练点"是否影响局部分类预测。
* 输入：private/uid_map.csv（先验/role/group）、blind/judged/pass1_all200.csv（首遍判读）、
        materials/V_eval.parquet（V 点 AEF64）、../prod2nd_2023/data/train2023_v2.parquet（池）
* 设计（参照文档71研究探针）：固定训练上下文 C，两臂 =
        base = C ∪ R（原始标签）
        iso  = C ∪ R_kept（矛盾可隔离点移除后按同种子顺序补足，非复制）
        两个框：O=争议组（园地/森林/耕地）、F=配对组（林型坡向）
* 评价：V 点，以首遍盲判为参照标签（描述性，非独立真值）；只报可判点。
* 输出：experiments/sens/probe_result.json + 文本摘要
* 用法：python sens_probe.py [--N 2000] [--seed 7]
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

WS = r'F:/lc_work/label_repair_2023_20261015'
POOL = r'F:/lc_work/prod2nd_2023/data/train2023_v2.parquet'
FEAT = ['A%02d' % i for i in range(64)]
NAME2CODE = {'草本旱地':1,'乔灌园地':2,'灌溉耕地':3,'常绿阔叶林':4,'落叶阔叶林':5,'常绿针叶林':6,
 '落叶针叶林':7,'常绿灌木':9,'落叶灌木':10,'草地':11,'地衣苔藓':12,'稀疏植被':13,'木本沼泽':14,
 '草本沼泽':15,'湖河滩地':16,'盐渍湿地':17,'红树林':18,'海岸盐沼':19,'潮滩':20,'不透水面':21,
 '裸地':22,'水体':23,'永久冰雪':24,
 # 合并级/泛指 -> 不可用于细类评价（置 -1），避免把大类当细类
 '乔木林':-1,'阔叶林（常绿/落叶不可分）':-1,'灌丛':-1,'灌木/草地':-1,'耕地':-1,'园地':2,
 '无法判定':-1}


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--box', default='O')
    ap.add_argument('--rule', default='orchard',
                    help='orchard=隔离"池=园地但判读非园地"的R点；forest=隔离"池=森林但判读非森林"的R点')
    a = ap.parse_args()

    side = pd.read_csv(os.path.join(WS, 'private/uid_map.csv'))
    p1 = pd.read_csv(os.path.join(WS, 'blind/judged/pass1_all200.csv'))
    d = p1.merge(side, on='point_id')
    AC = '实际类别（24类中文名，或"无法判定"）'
    CF = '置信（高/中/低）'

    group = 'dispute' if a.box == 'O' else 'forest_pair'
    box = (107.0, 30.0, 109.0, 32.0) if a.box == 'O' else (107.0, 32.0, 109.0, 34.0)
    g = d[d.group == group]
    R = g[g.role == 'R']; V = g[g.role == 'V']

    pool = pd.read_parquet(POOL)
    wx0, wy0, wx1, wy1 = box[0]-2, box[1]-2, box[2]+2, box[3]+2
    w = pool[(pool.lon >= wx0) & (pool.lon <= wx1) & (pool.lat >= wy0) & (pool.lat <= wy1)].copy()
    emit('%s 窗内候选 %d；R=%d V=%d' % (a.box, len(w), len(R), len(V)))

    # 排除 V 所在 0.25° 块（全部训练排除）
    vblk = set(zip((V.lon//0.25).astype(int), (V.lat//0.25).astype(int)))
    wblk = list(zip((w.lon//0.25).astype(int), (w.lat//0.25).astype(int)))
    w = w[[b not in vblk for b in wblk]].copy()

    # R 点位置键
    rkeys = set(zip(R.lon.round(5), R.lat.round(5)))
    jud = {}
    for _, r in g.iterrows():
        jud[(round(r.lon, 5), round(r.lat, 5))] = dict(
            code=NAME2CODE.get(str(r[AC]), -1), conf=r[CF], prior=r.cl_pool)
    # 可隔离集：prior=2(园地) 且判读细类可判且 != 2；或 prior in {1,3}(耕地) 且判读=2
    iso = set()
    for k, v in jud.items():
        if a.rule == 'orchard':
            if v['prior'] == 2 and v['code'] >= 0 and v['code'] != 2:
                iso.add(k)
            if v['prior'] in (1, 3) and v['code'] == 2:
                iso.add(k)
        elif a.rule == 'forest':
            # 池标森林(4/5/6/7/8) 但盲判为细类且不是森林 -> 隔离
            if v['prior'] in (4, 5, 6, 7, 8) and v['code'] >= 0 and v['code'] not in (4, 5, 6, 7, 8):
                iso.add(k)
    wkeys = list(zip(w.lon.round(5), w.lat.round(5)))
    keep = np.array([k not in iso for k in wkeys])
    emit('矛盾 R 可隔离 %d 点；候选 %d，扣隔离后 %d' % (len(iso), len(w), int(keep.sum())))

    # V 特征
    ve = pd.read_parquet(os.path.join(WS, 'materials/V_eval.parquet'))
    ve = ve[ve.vid.str.replace('B', '').astype(int).isin(V.point_id.str.replace('B', '').astype(int))]
    # V 的参照标签来自 pass1
    Vj = V.merge(ve[['vid'] + FEAT], left_on='point_id', right_on='vid', how='inner')
    yv = np.array([NAME2CODE.get(str(v), -1) for v in Vj[AC]])
    ok = yv >= 0
    emit('V 可判(判读为细类) n=%d / %d' % (ok.sum(), len(Vj)))

    res = dict(box=a.box, N=a.N, seed=a.seed, n_window=len(w), n_iso=len(iso),
               n_V=int(len(Vj)), n_V_evalable=int(ok.sum()))
    if ok.sum() < 20:
        res['note'] = 'V 可判点不足，仅记录'
    else:
        Xv = Vj[FEAT].to_numpy(np.float32)[ok]
        for arm in ['base', 'iso']:
            rng = np.random.RandomState(a.seed)
            idx = np.where(np.ones(len(w), bool) if arm == 'base' else keep)[0]
            sel = rng.choice(idx, size=min(a.N, len(idx)), replace=False)
            X = w.iloc[sel][FEAT].to_numpy(np.float32); y = w.iloc[sel]['cl'].to_numpy()
            rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=2,
                                        max_leaf_nodes=5000, n_jobs=10, random_state=a.seed)
            t = time.time(); rf.fit(X, y); fit_s = time.time() - t
            pred = rf.predict(Xv)
            res[arm] = dict(acc=round(float(accuracy_score(yv[ok], pred)), 4),
                            macroF1=round(float(f1_score(yv[ok], pred, average='macro')), 4),
                            pred_orchard=int((pred == 2).sum()),
                            pred_forest=int(np.isin(pred, [4, 5, 6, 7, 8]).sum()),
                            n_train=len(sel), fit_s=round(fit_s, 1))
            emit('%s %s done %.1fs' % (a.box, arm, fit_s))
    os.makedirs(os.path.join(WS, 'experiments/sens'), exist_ok=True)
    out = os.path.join(WS, 'experiments/sens/probe_%s.json' % a.box)
    json.dump(res, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    emit('-> %s' % out)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
