# -*- coding: utf-8 -*-
"""g4_region_separability.py — 各大区内"每种地类 vs 其余地类"的可分离度（统一 AEF64）

* 输入：各类 2023 交付 GPKG（内嵌 2023 年 AEF64）；大区 = g3 的 6°×6°（B<gx><gy>），
        落区按 lon/lat 整数分格（与瓦片 2° 网格同原点，3×3 归并）。
* 口径：每类取 2023 件（年匹配 AEF）；区内类点 n≥30 且其余点合计 n≥100 才评估。
* 指标（均为 LDA 五折**平衡准确率**，0.5=不可分、1=完全可分）：
    ① acc_vs_rest：该类 vs（区内其余全部类别的混合样本；每类抽样 ≤500、合计 ≤4000，类内 ≤2000）
    ② top1–top3：区内与该类**质心（白化）最近**的 3 个其他类别及其两两可分度
* 输出：AEF整理与可分离度/{大区内类间可分度_明细.csv, 大区内类间可分度_矩阵.csv, 大区内类间可分度_n矩阵.csv}
* 分片：--part/--nparts（按大区分片，互不重叠；明细各自落 _part 文件后合并）
* 用法：python g4_region_separability.py --part 0 --nparts 3
"""
import argparse, glob, json, os, sys, time
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

sys.stdout.reconfigure(encoding='utf-8')
ROOT = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12'
OUT = os.path.join(ROOT, 'AEF整理与可分离度')
AEFC = ['A%02d' % k for k in range(64)]
X0, Y0, DEG = 73.0, 18.0, 6.0
MIN_CLS, MIN_REST, CAP_CLS, CAP_OTHER, CAP_REST = 30, 100, 2000, 500, 4000
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
    print(time.strftime('[%m-%d %H:%M:%S] ') + 'P%d %s' % (os.getpid(), str(m)), flush=True)


def load_all():
    metas, Xs = [], []
    for cls, path in SPECS.items():
        f = os.path.join(ROOT, path)
        if not os.path.exists(f):
            emit('✗ 缺 %s' % cls); continue
        g = gpd.read_file(f, columns=AEFC, engine='pyogrio')
        x = g.geometry.x.to_numpy(); y = g.geometry.y.to_numpy()
        X = g[AEFC].to_numpy(np.float32).astype(np.float64)
        ok = np.isfinite(X).all(axis=1)
        reg = np.array(['B%02d%02d' % (int((a - X0) // DEG), int((b - Y0) // DEG))
                        for a, b in zip(x, y)])
        metas.append(pd.DataFrame(dict(cls=cls, region=reg[ok], x=x[ok], y=y[ok])))
        Xs.append(X[ok])
        emit('load %-12s %d 点' % (cls, int(ok.sum())))
    return pd.concat(metas, ignore_index=True), np.vstack(Xs)


def lda_acc(Xa, Xb, rng, cap_a, cap_b):
    ia = rng.permutation(len(Xa))[:min(cap_a, len(Xa))]
    ib = rng.permutation(len(Xb))[:min(cap_b, len(Xb))]
    if len(ia) < 10 or len(ib) < 10:
        return np.nan
    X = np.vstack([Xa[ia], Xb[ib]])
    y = np.r_[np.zeros(len(ia)), np.ones(len(ib))]
    try:
        s = cross_val_score(LinearDiscriminantAnalysis(), X, y, cv=5, scoring='balanced_accuracy')
        return float(s.mean())
    except Exception:
        return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', type=int, default=0)
    ap.add_argument('--nparts', type=int, default=1)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    meta, Xall = load_all()
    regions = sorted(meta.region.unique())
    mine = regions[a.part::a.nparts]
    emit('本片 %d/%d 大区：%s …' % (len(mine), len(regions), mine[:3]))
    rng = np.random.RandomState(11)
    rows = []
    for reg in mine:
        m = (meta.region == reg).to_numpy()
        sx = Xall[m]
        scls = meta.cls.to_numpy()[m]
        # 区内各类质心（白化）
        mu = sx.mean(0)
        C = np.cov((sx - mu).T) + np.eye(64) * 1e-6
        W = np.linalg.inv(np.linalg.cholesky(C))
        Z = (sx - mu) @ W.T
        cent = {c: Z[scls == c].mean(0) for c in np.unique(scls) if (scls == c).sum() >= MIN_CLS}
        for cls in np.unique(scls):
            nc = int((scls == cls).sum())
            if nc < MIN_CLS:
                continue
            own = sx[scls == cls]
            rest_mask = scls != cls
            n_rest = int(rest_mask.sum())
            if n_rest < MIN_REST:
                continue
            # ① vs 其余（每类抽 ≤500、合计 ≤4000）
            othX, q = [], CAP_REST
            for c2 in np.unique(scls[rest_mask]):
                Xi = sx[(scls == c2)]
                take = min(CAP_OTHER, len(Xi), max(0, q))
                if take:
                    othX.append(Xi[rng.permutation(len(Xi))[:take]]); q -= take
                if q <= 0:
                    break
            Xb = np.vstack(othX)
            acc = lda_acc(own, Xb, rng, CAP_CLS, CAP_REST)
            # ② top3 最近类
            top = []
            if cls in cent:
                dist = [(c2, float(np.linalg.norm(cent[cls] - cent[c2]))) for c2 in cent if c2 != cls]
                dist.sort(key=lambda t: t[1])
                for c2, _ in dist[:3]:
                    Xc = sx[scls == c2]
                    top.append((c2, lda_acc(own, Xc, rng, CAP_CLS, CAP_OTHER)))
            r = dict(region=reg, cls=cls, n_class=nc, n_rest=n_rest, acc_vs_rest=round(acc, 3) if acc == acc else None)
            for k in range(3):
                r['top%d_cls' % (k + 1)] = top[k][0] if k < len(top) else ''
                v = top[k][1] if k < len(top) else np.nan
                r['top%d_acc' % (k + 1)] = round(v, 3) if v == v else None
            rows.append(r)
        emit('%s：%d 行（区内类 %d）' % (reg, len([x for x in rows if x['region'] == reg]), len(np.unique(scls))))
    out = os.path.join(OUT, '大区内类间可分度_明细_part%d.csv' % a.part)
    pd.DataFrame(rows).to_csv(out, index=False, encoding='utf-8-sig')
    emit('→ %s（%d 行）' % (out, len(rows)))


if __name__ == '__main__':
    main()
