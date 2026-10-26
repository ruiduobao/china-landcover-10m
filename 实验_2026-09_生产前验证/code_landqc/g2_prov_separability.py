# -*- coding: utf-8 -*-
"""g2_prov_separability.py — 各类样本"省间可分离度"评估（统一以 AEF64 为判据）

* 口径：每类取 **2023 年交付件**（含 2023 年 AEF64，年匹配）——分类别、分省统计；
  AEF 自 GPKG 读取（float32）；省 = CTAmap `ct_prov`（空值=省界外/近岸，剔除并计数）。
* 指标（三类，统一 AEF64 空间）：
    ① 白化质心距离 d_ij = ‖Σ^{-1/2}(μi−μj)‖（Σ=类内合并协方差；无单位、越大越可分）
    ② 省对 LDA 五折**平衡准确率**（每省抽样 ≤3000；0.5=不可分、1=完全可分）
    ③ 类级：多元 LDA 五折平衡准确率 + 轮廓系数（省标签，抽样 ≤20000，原生 AEF 欧氏）
* 省集：每类取点数 ≥100 的前 12 个省（不足则全取）。
* 输出：AEF整理与可分离度/{省间可分离度_汇总.csv, 省间可分离度_矩阵_<类>.csv, 可分离度明细.json}
* 用法：python g2_prov_separability.py [--classes 草地,裸地,...]
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score

sys.stdout.reconfigure(encoding='utf-8')
ROOT = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12'
OUT = os.path.join(ROOT, 'AEF整理与可分离度')
AEFC = ['A%02d' % k for k in range(64)]
MIN_N, TOP_K, CAP_PAIR, CAP_SIL = 100, 12, 3000, 20000
SPECS = {
    '草本旱地': '草本旱地/2023/草本旱地_2023_WGS84.gpkg',
    '灌溉耕地': '灌溉耕地/2023/灌溉耕地_2023_WGS84.gpkg',
    '草地': '草地/2023/草地_2023_WGS84.gpkg',
    '地衣苔藓': '地衣苔藓/2023/地衣苔藓_2023_WGS84.gpkg',
    '稀疏植被': '稀疏植被/2023/稀疏植被_2023_WGS84.gpkg',
    '裸地': '裸地/2023/裸地_2023_WGS84.gpkg',
    '湖河滩地': '湖河滩地/2023/湖河滩地_2023_WGS84.gpkg',
    '盐渍湿地': '盐渍湿地/2023/盐渍湿地_2023_WGS84.gpkg',
    '红树林': '红树林/扩充_v1/红树林_全量_10718_2023_WGS84.gpkg',
    '海岸盐沼': '海岸盐沼/2023/海岸盐沼_2023_WGS84.gpkg',
    '潮滩': '潮滩/2023/潮滩_2023_WGS84.gpkg',
    '水体': '水体/2023/水体_2023_WGS84.gpkg',
    '不透水面': '不透水面/2023/不透水面_2023_WGS84.gpkg',
    '永久冰雪': '永久冰雪/2023/永久冰雪_2023_WGS84.gpkg',
    '林地_常绿阔叶林': '林地_常绿阔叶林/林地_常绿阔叶林_2017-2024并集_WGS84.gpkg',
    '林地_落叶阔叶林': '林地_落叶阔叶林/林地_落叶阔叶林_2017-2024并集_WGS84.gpkg',
    '林地_常绿针叶林': '林地_常绿针叶林/林地_常绿针叶林_2017-2024并集_WGS84.gpkg',
    '林地_落叶针叶林': '林地_落叶针叶林/林地_落叶针叶林_2017-2024并集_WGS84.gpkg',
    '木本沼泽': '木本沼泽/2023/木本沼泽_2023_WGS84.gpkg',
    '草本沼泽': '草本沼泽/2023/草本沼泽_2023_WGS84.gpkg',
    '灌木_常绿': '常绿灌木/2023/常绿灌木_2023_WGS84.gpkg',
    '灌木_落叶': '落叶灌木/2023/落叶灌木_2023_WGS84.gpkg',
    '乔灌园地': '乔灌园地/2023/乔灌园地v7_2023_WGS84.gpkg',
}


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def analyze(name, path):
    g = gpd.read_file(os.path.join(ROOT, path), columns=['ct_prov'] + AEFC, engine='pyogrio')
    X = g[AEFC].to_numpy(np.float32).astype(np.float64)
    prov = g.ct_prov.fillna('').astype(str).to_numpy()
    ok = (prov != '') & np.isfinite(X).all(axis=1)
    X, prov = X[ok], prov[ok]
    n_all, n_noprov = int(ok.sum()), int((~ok).sum())
    cnt = pd.Series(prov).value_counts()
    keep = cnt[cnt >= MIN_N].head(TOP_K).index.tolist()
    m = np.isin(prov, keep)
    Xs, ps = X[m], prov[m]
    if len(keep) < 2:
        return dict(cls=name, n=n_all, n_noprov=n_noprov, n_prov=len(keep), note='省不足 2 个，跳过')
    # ① 白化质心距离
    mu = Xs.mean(0)
    C = np.cov((Xs - mu).T) + np.eye(64) * 1e-6
    W = np.linalg.inv(np.linalg.cholesky(C))
    Z = (Xs - mu) @ W.T
    cent = {p: Z[ps == p].mean(0) for p in keep}
    D = pd.DataFrame(index=keep, columns=keep, dtype=float)
    for i, a in enumerate(keep):
        for b in keep[i + 1:]:
            d = float(np.linalg.norm(cent[a] - cent[b]))
            D.loc[a, b] = D.loc[b, a] = round(d, 3)
        D.loc[a, a] = 0.0
    # ② 省对 LDA 五折平衡准确率
    rng = np.random.RandomState(7)
    Zs = {}
    for p in keep:
        idx = np.where(ps == p)[0]
        idx = idx if len(idx) <= CAP_PAIR else rng.permutation(idx)[:CAP_PAIR]
        Zs[p] = Xs[idx]
    A = pd.DataFrame(index=keep, columns=keep, dtype=float)
    accs = []
    for i, a in enumerate(keep):
        A.loc[a, a] = 1.0
        for b in keep[i + 1:]:
            Xp = np.vstack([Zs[a], Zs[b]])
            yp = np.r_[np.zeros(len(Zs[a])), np.ones(len(Zs[b]))]
            s = cross_val_score(LinearDiscriminantAnalysis(), Xp, yp, cv=5, scoring='balanced_accuracy')
            A.loc[a, b] = A.loc[b, a] = round(float(s.mean()), 3)
            accs.append((float(s.mean()), a, b))
    # ③ 类级：多元 LDA 平衡准确率 + 轮廓系数
    Xm = np.vstack([Zs[p] for p in keep]); ym = np.concatenate([[p] * len(Zs[p]) for p in keep])
    ml = float(cross_val_score(LinearDiscriminantAnalysis(), Xm, ym, cv=5, scoring='balanced_accuracy').mean())
    sil = np.nan
    if len(Xs) >= 1000:
        sub = rng.permutation(len(Xs))[:min(CAP_SIL, len(Xs))]
        sil = float(silhouette_score(Xs[sub], ps[sub], metric='euclidean'))
    accs.sort(reverse=True)
    row = dict(cls=name, n=n_all, n_noprov=n_noprov, n_prov=len(keep),
               prov_top='|'.join('%s:%d' % (p, int(cnt[p])) for p in keep),
               pair_mean=round(float(np.mean([a for a, _, _ in accs])), 3),
               pair_min=round(accs[-1][0], 3), pair_min_pair='%s↔%s' % (accs[-1][1], accs[-1][2]),
               pair_max=round(accs[0][0], 3), pair_max_pair='%s↔%s' % (accs[0][1], accs[0][2]),
               d_mean=round(float(np.nanmean(D.to_numpy())), 3),
               multiclass_acc=round(ml, 3), silhouette=round(sil, 3) if sil == sil else None)
    return row, D, A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--classes', default='')
    a = ap.parse_args()
    only = [x.strip() for x in a.classes.split(',') if x.strip()]
    os.makedirs(OUT, exist_ok=True)
    rows, detail = [], {}
    for name, path in SPECS.items():
        if only and name not in only:
            continue
        f = os.path.join(ROOT, path)
        if not os.path.exists(f):
            emit('✗ 缺 %s（%s）' % (name, path)); continue
        t0 = time.time()
        res = analyze(name, f)
        if isinstance(res, tuple):
            row, D, A = res
            D.to_csv(os.path.join(OUT, '白化距离矩阵_%s.csv' % name), encoding='utf-8-sig')
            A.to_csv(os.path.join(OUT, 'LDA准确率矩阵_%s.csv' % name), encoding='utf-8-sig')
            detail[name] = dict(pair_acc=[[float(v) for v in r] for r in A.to_numpy()], prov=list(A.index))
        else:
            row = res
        rows.append(row)
        emit('%-12s n=%-7d 省=%-2d 省对均值 %.3f（min %.3f %s / max %.3f %s）｜多类 %.3f ｜轮廓 %s ｜%.0fs' % (
            name, row['n'], row['n_prov'], row.get('pair_mean', -1), row.get('pair_min', -1), row.get('pair_min_pair', ''),
            row.get('pair_max', -1), row.get('pair_max_pair', ''), row.get('multiclass_acc', -1),
            row.get('silhouette'), time.time() - t0))
    df = pd.DataFrame(rows).sort_values('multiclass_acc', ascending=False)
    df.to_csv(os.path.join(OUT, '省间可分离度_汇总.csv'), index=False, encoding='utf-8-sig')
    json.dump(detail, open(os.path.join(OUT, '可分离度明细.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    emit('→ %s' % OUT)


if __name__ == '__main__':
    main()
