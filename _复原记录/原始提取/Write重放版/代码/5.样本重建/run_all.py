# -*- coding: utf-8 -*-
"""
run_all.py — r1 样本重建流水线编排 + 终检门禁
* 顺序: s2底座 → s3外部 → s4 FCS10全覆盖 → s5投票合并 → s6训练清洗 → s7验证池 → s8交付
* 断点续跑: 默认跳过已存在产物的步骤（--force 全部重跑）
* 终检门禁: 30类无空类 / 境外点=0 / 生态违规=0 / 同类NN间距抽查 / 交付计数一致
"""
import os, sys, json, time, subprocess, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_conf as C

STEPS = [
    ('s2_ingest_base.py',  C.R1_BASE),
    ('s3_external.py',     C.R1_EXT),
    ('s4_fcs10.py',        C.R1_TOPUP),
    ('s4b_thematic.py',    C.R1_THEMATIC),
    ('s5_votes_pool.py',   C.R1_POOL),
    ('s6_clean_train.py',  C.R1_TRAIN),
    ('s7_validation.py',   C.R1_VALID),
    ('s8_export.py',       None),   # 交付无单一产物文件
]

def gates():
    import pandas as pd
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import s1_geom as G
    train = pd.read_parquet(C.R1_TRAIN)
    pool = pd.read_parquet(C.R1_POOL)
    ok = True
    rep = {}

    n_class = train.class_new.nunique()
    rep['30类全覆盖'] = (n_class == 30)
    ok &= (n_class == 30)

    inside = G.china_contains(train.lon.to_numpy(), train.lat.to_numpy())
    rep['境外点=0'] = bool(inside.all())
    ok &= bool(inside.all())

    bad = C.eco_violation(train.lon.to_numpy(), train.lat.to_numpy(), train.class_new.to_numpy())
    rep['生态违规=0'] = int(bad.sum()) == 0
    ok &= int(bad.sum()) == 0

    # 同类 NN 间距抽查（每类最多 2 万点）
    from scipy.spatial import cKDTree
    p50s = []
    for c, d in train.groupby('class_new'):
        if len(d) < 2:
            continue
        if len(d) > 20000:
            d = d.sample(20000, random_state=1)
        km = np.column_stack([d.lon * 111.32 * np.cos(np.radians(d.lat)), d.lat * 110.57])
        t = cKDTree(km)
        dd, _ = t.query(km, k=2)
        p50s.append((c, float(np.median(dd[:, 1])) * 1000))   # km → m
    rep['同类NN间距p50(m)'] = {str(c): round(v) for c, v in p50s}
    n_low = sum(1 for _, v in p50s if v < 300)
    rep['NN_p50<300m类数'] = n_low

    # 稀缺类改善
    rep['稀缺类训练集点数'] = {str(c): int((train.class_new == c).sum()) for c in C.RARE_CLASSES}

    # 计数一致（README 由同源数据生成，校验 train 总数）
    summ = json.load(open(os.path.join(C.WORK, 'r1_train_summary.json'), encoding='utf-8'))
    rep['train计数一致'] = (summ['train_total'] == len(train))
    ok &= summ['train_total'] == len(train)

    print(json.dumps(rep, ensure_ascii=False, indent=2, default=int))
    print('\n门禁结果:', '✅ 全部通过' if ok else '❌ 存在未通过项')
    return ok, rep

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--from', dest='frm', default=None, help='从指定步骤开始, 如 s5')
    args = ap.parse_args()

    t0 = time.time()
    run = not args.frm or False
    for script, out in STEPS:
        if args.frm:
            if script.startswith(args.frm):
                run = True
            elif not run:
                continue
        if out and os.path.exists(out) and not args.force and (not args.frm or run):
            print(f'[skip] {script} → {os.path.basename(out)} 已存在', flush=True)
            continue
        print(f'\n=== 运行 {script} ===', flush=True)
        log = os.path.join(C.LOG_DIR, script.replace('.py', '') + '_run.log')
        with open(log, 'w', encoding='utf-8') as lf:
            rc = subprocess.call([sys.executable, '-u',
                                  os.path.join(os.path.dirname(os.path.abspath(__file__)), script)],
                                 stdout=lf, stderr=subprocess.STDOUT)
        if rc != 0:
            print(f'[FAIL] {script} 退出码 {rc}，日志: {log}', flush=True)
            sys.exit(rc)
        print(f'[done] {script} ({time.time()-t0:.0f}s)', flush=True)

    print('\n=== 终检门禁 ===', flush=True)
    ok, rep = gates()
    json.dump({'passed': bool(ok), 'report': rep},
              open(os.path.join(C.WORK, 'r1_gate.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2, default=int)
    sys.exit(0 if ok else 2)

if __name__ == '__main__':
    main()
