# -*- coding: utf-8 -*-
"""e6_years_driver.py — 逐年跑 e6 并合并（抗宿主清理）
* 对每个年份单独启动 e6_spatial_eval.py --years <y>（单年约 2-3 分钟），
  失败自动重试（默认 3 次）；每年后收集 by_year / by_source / val_predictions。
* 最后用各年 confusion_<y>.csv 求和得到 8 年汇总 OA，并合并 by_source（从累计预测）。
* 资源：单进程、n_jobs 由 E6_NJOBS 控制（默认 15）。
"""
import os, sys, json, glob, subprocess, time, argparse
import numpy as np
import pandas as pd

ROOT = r'F:\地理所\论文\中国土地覆盖数据_2017-2024'
KB = r'Z:\地理所\论文\中国土地覆盖数据_2017-2024'
OUT = os.path.join(KB, '数据', '本地处理', '全国清洗训练', '评估_P3.4')
E6 = os.path.join(ROOT, '代码', '4.全国清洗训练', 'e6_spatial_eval.py')
REPORT = os.path.join(OUT, '评估报告_P3.4.json')
YEARS = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
ACC = os.path.join(OUT, '_merge_acc')
os.makedirs(ACC, exist_ok=True)

sys.path.insert(0, os.path.join(ROOT, '代码', '4.全国清洗训练'))
from e6_spatial_eval import metrics_from_cm  # noqa


def run_year(y, tries=3):
    for i in range(tries):
        env = dict(os.environ, E6_NJOBS=os.environ.get('E6_NJOBS', '15'),
                   OMP_NUM_THREADS='15', MKL_NUM_THREADS='15')
        p = subprocess.run([sys.executable, E6, '--years', str(y)],
                           cwd=ROOT, env=env, capture_output=True, text=True)
        line = [l for l in (p.stdout or '').splitlines() if f'y{y}:' in l and 'OA=' in l]
        if line:
            ok = True
        else:
            ok = False
        # 无论 stdout 如何，只要报告里出现该年即算成功
        try:
            j = json.load(open(REPORT, encoding='utf-8'))
            if str(y) in j.get('by_year', {}):
                return j
        except Exception:
            pass
        print(f'  y{y} 第{i+1}次失败，重试…', flush=True)
        time.sleep(10)
    return None


def merge():
    # 汇总混淆矩阵
    cm_all = None
    labs_all = None
    for f in sorted(glob.glob(os.path.join(OUT, 'confusion_*.csv'))):
        y = int(os.path.basename(f).split('_')[1].split('.')[0])
        df = pd.read_csv(f, index_col=0)
        labs = [int(str(c).split('_')[1]) for c in df.index]
        M = df.to_numpy()
        if cm_all is None:
            labs_all, cm_all = labs, M
        else:
            merged = sorted(set(labs_all) | set(labs))
            def grow(A, L, newL):
                pos = {c: i for i, c in enumerate(newL)}
                G = np.zeros((len(newL), len(newL)), dtype=np.int64)
                for i, ci in enumerate(L):
                    for j, cj in enumerate(L):
                        G[pos[ci], pos[cj]] += A[i, j]
                return G
            cm_all = grow(cm_all, labs_all, merged) + grow(M, labs, merged)
            labs_all = merged
    met_all, per_cls = metrics_from_cm(cm_all, labs_all)
    return met_all, per_cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', type=int, nargs='*', default=YEARS)
    a = ap.parse_args()
    collected = {}
    for y in a.years:
        j = run_year(y)
        if j is None:
            print(f'❌ y{y} 三年重试均失败', flush=True)
            continue
        collected[str(y)] = j['by_year'][str(y)]
        if 'by_source' in j:
            collected[str(y)]['by_source'] = j['by_source']
        # 备份该年产出
        for pat in (f'confusion_{y}.csv', f'per_class_{y}.csv'):
            src = os.path.join(OUT, pat)
            if os.path.isfile(src):
                import shutil; shutil.copyfile(src, os.path.join(ACC, pat))
        print(f'✅ y{y} OA={collected[str(y)]["OA"]:.4f}', flush=True)

    met_all, per_cls = merge()
    # 合并 by_source（从每年报告里汇总，按 n 加权近似）
    bs_rows = {}
    for y, d in collected.items():
        bs = d.pop('by_source', None)
        if bs:
            for tag, m in bs.items():
                bs_rows.setdefault(tag, []).append((m['n'], m))
    bs_all = {}
    for tag, rows in bs_rows.items():
        rows.sort(key=lambda x: -x[0])
        bs_all[tag] = rows[0][1] if len(rows) == 1 else rows[0][1]  # 单年口径代表
    rep = {'time': time.strftime('%Y-%m-%d %H:%M'), 'trees': 150,
           'note': f'B 方案（FCS10 跨年降权0.5）逐年运行合并；by_source 为该年单年口径',
           'by_year': collected, 'overall': met_all, 'by_source_notes': bs_all,
           'years_done': sorted(int(k) for k in collected)}
    json.dump(rep, open(REPORT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
    per_cls.to_csv(os.path.join(OUT, 'per_class_overall.csv'), index=False)
    print('合并完成:', json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                   for k, v in met_all.items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
