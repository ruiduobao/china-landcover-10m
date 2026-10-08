# -*- coding: utf-8 -*-
"""f03_fingerprint.py — 图 F03：年度指纹（64 维 AEF）原理图【本地渲染，云端模型看不到这些数据】
* 输入：F:/lc_work/年度子集_含稀有类/r7_train_{2017..2024}.parquet（列 row_id + A00..A63 + class_new）
        F:/lc_work/v31_exp/config/v31_map.json（30→24 类映射）
* 规则：PCA(2) 在"全部年份 × 抽样点"上一次性拟合（保证同年可比、跨年漂移可见）
* 输出：论文图/output/F03_年度指纹原理图.png（三联：按年份 / 按类组 / 单点 8 年轨迹）
* 用法：python f03_fingerprint.py [--n 2500]
"""
import os, sys, json, argparse
import numpy as np
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

SRC = r'F:/lc_work/年度子集_含稀有类/r7_train_%d.parquet'
V31 = r'F:/lc_work/v31_exp/config/v31_map.json'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output')
YEARS = list(range(2017, 2025))
FEATS = ['A%02d' % i for i in range(64)]
GRP = {1: '耕地', 2: '森林', 3: '灌丛', 4: '草地', 5: '湿地', 6: '不透水', 7: '裸地', 8: '稀疏植被', 9: '水体', 10: '冰雪'}
V31_2_GROUP = {}
for c in (1, 2, 3): V31_2_GROUP[c] = 1
for c in range(4, 9): V31_2_GROUP[c] = 2
for c in (9, 10): V31_2_GROUP[c] = 3
V31_2_GROUP[11] = 4
for c in range(14, 21): V31_2_GROUP[c] = 5
V31_2_GROUP[21] = 6
V31_2_GROUP[22] = 7
V31_2_GROUP[12] = 8
V31_2_GROUP[13] = 8
V31_2_GROUP[23] = 9
V31_2_GROUP[24] = 10


def main(n):
    os.makedirs(OUT, exist_ok=True)
    d = json.load(open(V31, encoding='utf-8'))
    vmap = {}                                    # 30 类原码 -> 24 类
    for new, olds in d['merges'].items():        # merges: 新码 -> [旧码...]
        for o in olds:
            vmap[int(o)] = int(new)
    vmap.update({int(k): int(v) for k, v in d['identity'].items()})
    # 1) 抽 2023 的 row_id
    t = pq.read_table(SRC % 2023, columns=['row_id', 'class_new'])
    rng = np.random.default_rng(7)
    idx = rng.choice(len(t), size=min(n, len(t)), replace=False)
    ids = t.column('row_id').to_numpy()[np.sort(idx)]
    cls30 = t.column('class_new').to_numpy()[np.sort(idx)]
    id2cls = dict(zip(ids.tolist(), cls30.tolist()))
    # 2) 逐年取同一批 row_id
    X, Y, C, R = [], [], [], []
    traj = {}
    for y in YEARS:
        tb = pq.read_table(SRC % y, columns=['row_id'] + FEATS, filters=[('row_id', 'in', ids.tolist())])
        rid = tb.column('row_id').to_numpy()
        arr = np.stack([tb.column(f).to_numpy() for f in FEATS], 1).astype('float32')
        X.append(arr); Y += [y] * len(rid); R += rid.tolist()
        C += [id2cls.get(int(i), 0) for i in rid]
        traj[y] = {int(i): k for k, i in enumerate(rid)}
        print('年 %d: %d 点' % (y, len(rid)), flush=True)
    X = np.vstack(X); Y = np.array(Y); R = np.array(R); C = np.array(C)
    # 3) PCA 一次拟合
    p = PCA(n_components=2, random_state=7).fit(X)
    Z = p.transform(X)
    v12 = p.explained_variance_ratio_[:2] * 100
    # 漂移量化（64 维原始空间）：跨年同点距离 中位 vs 同年不同点距离 中位
    z17 = {R[k]: k for k in range(len(R)) if Y[k] == 2017}
    z23 = {R[k]: k for k in range(len(R)) if Y[k] == 2023}
    common = sorted(set(z17) & set(z23))
    if len(common) > 50:
        a = X[[z17[i] for i in common]]
        b = X[[z23[i] for i in common]]
        d_cross = float(np.median(np.linalg.norm(a - b, axis=1)))
        sub = X[Y == 2023][:600]
        dd = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=2)
        d_within = float(np.median(dd[np.triu_indices(len(sub), 1)]))
        ratio = d_cross / max(1e-9, d_within)
    else:
        d_cross = d_within = ratio = float('nan')
    print('漂移量化: 跨年同点=%.3f 同年不同点=%.3f 比值=%.2f' % (d_cross, d_within, ratio))
    # 4) 出图
    fig = plt.figure(figsize=(16.5, 5.6), dpi=150)
    # (a) 按年份
    ax1 = fig.add_subplot(131)
    cmap = plt.get_cmap('turbo', len(YEARS))
    for k, y in enumerate(YEARS):
        m = Y == y
        ax1.scatter(Z[m, 0], Z[m, 1], s=3, color=cmap(k), alpha=.5, label=str(y), linewidths=0)
    ax1.set_title('(a) 同一批点的 8 年指纹：同年聚集、跨年整体漂移', fontsize=11)
    ax1.set_xlabel('PC1 (%.1f%%)' % v12[0]); ax1.set_ylabel('PC2 (%.1f%%)' % v12[1])
    cen = [(Z[Y == y, 0].mean(), Z[Y == y, 1].mean()) for y in YEARS]
    ax1.plot([c[0] for c in cen], [c[1] for c in cen], '-', color='k', lw=1.6, zorder=5)
    for k, (cx, cy) in enumerate(cen):
        ax1.scatter([cx], [cy], s=90, color=cmap(k), edgecolor='k', linewidth=.6, zorder=6)
        ax1.annotate(str(YEARS[k]), (cx, cy), fontsize=7.5, zorder=7,
                     xytext=(3, 3), textcoords='offset points')
    ax1.legend(fontsize=7, markerscale=3, ncol=2); ax1.set_xticks([]); ax1.set_yticks([])
    # (b) 按大类（用 2023 标签着色）
    ax2 = fig.add_subplot(132)
    g = np.array([V31_2_GROUP.get(vmap.get(int(c), 0), 0) for c in C])
    m23 = Y == 2023
    cmap2 = plt.get_cmap('tab10', 10)
    for k in range(1, 11):
        m = (g == k) & m23
        if m.sum() == 0: continue
        ax2.scatter(Z[m, 0], Z[m, 1], s=4, color=cmap2(k - 1), alpha=.6, label=GRP[k], linewidths=0)
    ax2.set_title('(b) 2023 年：同一指纹空间里地类可分（按大类着色）', fontsize=11)
    ax2.set_xlabel('PC1'); ax2.set_ylabel('PC2')
    ax2.legend(fontsize=7, markerscale=3, ncol=2); ax2.set_xticks([]); ax2.set_yticks([])
    # (c) 单点 8 年轨迹
    ax3 = fig.add_subplot(133)
    # 取 200 个 8 年齐全、且 2023 为森林/耕地的点
    cand = [i for i in ids.tolist() if all(i in traj[y] for y in YEARS)]
    rng.shuffle(cand)
    cand = cand[:200]
    for i in cand[:120]:
        xs = [Z[traj[y][i], 0] for y in YEARS]
        ys = [Z[traj[y][i], 1] for y in YEARS]
        ax3.plot(xs, ys, '-', color='0.6', lw=.5, alpha=.5)
        ax3.scatter(xs[0], ys[0], s=6, color='tab:blue', zorder=3)
        ax3.scatter(xs[-1], ys[-1], s=6, color='tab:red', zorder=3)
    ax3.set_title('(c) 单像元 8 年轨迹（蓝=2017 → 红=2024）' + chr(10) + '同点跨年距离/同年不同点距离=%.2f×' % ratio, fontsize=10.5)
    ax3.set_xlabel('PC1'); ax3.set_ylabel('PC2'); ax3.set_xticks([]); ax3.set_yticks([])
    fig.suptitle('F03 年度指纹（AlphaEarth Foundations 64 维 / 10 m）：同年内可比、跨年有系统漂移，因此逐年独立训练',
                 fontsize=13.5, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fp = os.path.join(OUT, 'F03_年度指纹原理图.png')
    fig.savefig(fp); print('→', fp)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=2500)
    main(ap.parse_args().n)
