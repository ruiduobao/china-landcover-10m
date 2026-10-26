# -*- coding: utf-8 -*-
"""r6_leak_qa.py — 独立泄漏审计与只读输入哈希（QC 包，不复用盲判代理上下文）

* 输入：blind/*、materials/*（受检面）、private/*（仅本脚本读，用于比对）、
        v2 池 / 母库（距离独立性核验）、原始两版栅格 / DEM（哈希）
* 规则源：技术文档/72 协议；skill references/判读协议.md §2（盲判批次只给 point_id）
* 门槛（任一不过即 FAIL，写进报告）：
    A 受检面无先验列/关键词；B 计数 200=R120+V80（对 100 点 / 争议 100 点）；
    C R 块 ∩ V 块 = ∅；D 配对两点同 role 且同块；E V 点与 v2 池、母库距离 ≥1 km；
    F V_eval.parquet 无 cl_true/true_label/t24/label_after；G 只读输入哈希落盘。
* 输出：qc/leak_qa_report.md、qc/leak_qa.json、audit/readonly_inputs_hashes.json
* 用法：python r6_leak_qa.py
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rc_conf import *   # noqa

FORBIDDEN_KEYS = ['prior', 'cl', 'cl_pool', 'role', 'group', 'pair', 'pair_id', 'src', 'src_label', 'aspect',
                  'elev', 'slope', 'block', 'box', 'change', 'cl_name', 'uid']
FORBIDDEN_TEXT = ['先验', '坡向', '配对', 'repair_dev', 'sealed_confirm'] + list(CL24.values())
V_MIN_DIST_KM = 1.0


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


META_SKIP = ('batch_manifest.json', '_READY_STATUS.json', '_CONTRACT.json')


def scan_face(root):
    """扫描盲判面（判读批次实际读取的文件）：blind/*.csv、materials/batch_*/**。
    meta 清单（batch_manifest/_READY_STATUS/_CONTRACT）只描述路径与纪律、不构成判读输入，
    单列在 `meta_files` 里，不参与文本命中判定（其中 _CONTRACT.json 由 experiments 组写）。"""
    hits, meta = [], []
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if not d.startswith('_')]     # 跳过 _tmp* 等外部临时目录
        if os.path.basename(dp) == 'private':
            continue
        in_chips = os.path.basename(dp) == 'chips'
        for fn in fns:
            fp = os.path.join(dp, fn)
            low = fn.lower()
            if fn in META_SKIP:
                meta.append(fp)
                continue
            if fn.endswith(('.csv', '.json', '.md', '.txt')):
                try:
                    txt = open(fp, encoding='utf-8-sig', errors='ignore').read()
                except Exception:
                    continue
                head = txt.splitlines()[0] if txt else ''
                toks = [t.strip().strip('"').lower() for t in head.split(',')]
                for k in FORBIDDEN_KEYS:
                    if k.lower() in toks:
                        hits.append(dict(file=fp, kind='header', key=k, line=head[:160]))
                for t in FORBIDDEN_TEXT:
                    if t and t in txt:
                        hits.append(dict(file=fp, kind='text', key=t, where='body'))
            if fn.endswith('.png') and in_chips and not low.startswith('b'):
                hits.append(dict(file=fp, kind='chipname', key=low, line=''))
    return hits, meta


def main():
    t0 = time.time()
    ensure_dirs()
    side = pd.read_csv(os.path.join(PRIV, 'uid_map.csv'), encoding='utf-8-sig')
    blind = pd.read_csv(os.path.join(BLIND, 'points_blind.csv'), encoding='utf-8-sig')
    res = {}

    # A 面扫描
    h1, m1 = scan_face(BLIND)
    h2, m2 = scan_face(MAT)
    hits = h1 + h2
    res['A_face_scan'] = dict(pass_=len(hits) == 0, hits=hits[:40], n_hits=len(hits),
                              meta_files_not_scanned=sorted(set(m1 + m2)))

    # B 计数
    nR = int((side['role'] == ROLE_R).sum())
    nV = int((side['role'] == ROLE_V).sum())
    npair = int((side['group'] == 'forest_pair').sum())
    ndisp = int((side['group'] == 'dispute').sum())
    res['B_counts'] = dict(pass_=(len(side) == QUOTA_TOTAL and nR == QUOTA_R and nV == QUOTA_V
                                  and npair == 100 and ndisp == 100),
                           n_total=len(side), n_R=nR, n_V=nV, n_pair_pts=npair, n_dispute_pts=ndisp,
                           n_blind=len(blind), blind_ids_unique=bool(blind['point_id'].is_unique))

    # C 块不相交
    bR = set(side.loc[side['role'] == ROLE_R, 'block'])
    bV = set(side.loc[side['role'] == ROLE_V, 'block'])
    res['C_blocks_disjoint'] = dict(pass_=len(bR & bV) == 0, n_R_blocks=len(bR), n_V_blocks=len(bV),
                                    overlap=sorted(bR & bV)[:10])

    # D 配对同 role 同块
    bad = []
    for pid, g in side[side['group'] == 'forest_pair'].groupby('pair_id'):
        if g['role'].nunique() != 1 or g['block'].nunique() != 1 or len(g) != 2:
            bad.append(pid)
    res['D_pair_integrity'] = dict(pass_=len(bad) == 0, n_pairs=int(side[side['group'] == 'forest_pair']['pair_id'].nunique()),
                                   bad=bad[:10])

    # E V 点空间独立性（与 v2 池、母库）
    from scipy.spatial import cKDTree
    v = side[side['role'] == ROLE_V]
    pool = pd.read_parquet(POOL_V2, columns=['lon', 'lat'])
    pt = cKDTree(np.c_[pool['lon'].values * 111.32 * np.cos(np.radians(pool['lat'].values)),
                       pool['lat'].values * 110.57])
    vxy = np.c_[v['lon'].values * 111.32 * np.cos(np.radians(v['lat'].values)), v['lat'].values * 110.57]
    d_pool = pt.query(vxy, k=1)[0]
    d_mother = None
    if os.path.exists(MOTHER):
        mo = pd.read_parquet(MOTHER, columns=['lon', 'lat'])
        mt = cKDTree(np.c_[mo['lon'].values * 111.32 * np.cos(np.radians(mo['lat'].values)),
                           mo['lat'].values * 110.57])
        d_mother = mt.query(vxy, k=1)[0]
    okE = bool((d_pool >= V_MIN_DIST_KM).all() and (d_mother is None or (d_mother >= V_MIN_DIST_KM).all()))
    res['E_V_independence'] = dict(pass_=okE, min_dist_pool_km=round(float(d_pool.min()), 3),
                                   min_dist_mother_km=(round(float(d_mother.min()), 3) if d_mother is not None else None),
                                   threshold_km=V_MIN_DIST_KM)
    # R 点必须来自池（回接核验：同坐标池内命中）
    r = side[side['role'] == ROLE_R]
    rxy = np.c_[r['lon'].values * 111.32 * np.cos(np.radians(r['lat'].values)), r['lat'].values * 110.57]
    d_r = pt.query(rxy, k=1)[0]
    res['E_R_from_pool'] = dict(pass_=bool((d_r < 0.01).all()), max_dist_pool_km=round(float(d_r.max()), 4))

    # F V_eval 禁列
    vp = os.path.join(MAT, 'V_eval.parquet')
    cols = list(pd.read_parquet(vp).columns) if os.path.exists(vp) else []
    forb = [c for c in cols if c in ('cl_true', 'true_label', 't24', 'label_after')]
    res['F_V_eval_forbidden'] = dict(pass_=len(forb) == 0, forbidden=forb, n_cols=len(cols),
                                     has_aef64=all('A%02d' % i in cols for i in range(64)))

    # G 只读输入哈希
    inputs = dict(POOL_V2=POOL_V2, POOL_V1=POOL_V1, MOTHER=MOTHER, TILE_PLAN2=TILE_PLAN2)
    for k, v in BOXES.items():
        inputs['RAST2_%s' % v['tile']] = os.path.join(RAST2, '%s_10m.tif' % v['tile'])
        inputs['RAST1_%s' % v['tile']] = os.path.join(RAST1, '%s_10m.tif' % v['tile'])
        inputs['DEM_%s' % v['tile']] = os.path.join(DEM, 'dem_%s.tif' % v['tile'])
    hashes = {}
    for k, fp in inputs.items():
        if os.path.exists(fp):
            st = os.stat(fp)
            hashes[k] = dict(path=fp, bytes=st.st_size, mtime=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_mtime)),
                             sha256=(sha256_file(fp) if st.st_size < (1 << 31) else 'SKIP_BIG'))
        else:
            hashes[k] = dict(path=fp, missing=True)
    jdump(os.path.join(AUD, 'readonly_inputs_hashes.json'), hashes)
    res['G_readonly_hashes'] = dict(pass_=all(not h.get('missing') for h in hashes.values()), n=len(hashes))

    allpass = all(v.get('pass_') for k, v in res.items() if isinstance(v, dict))
    res['all_pass'] = bool(allpass)
    res['seconds'] = round(time.time() - t0, 1)
    jdump(os.path.join(QC, 'leak_qa.json'), res)

    L = ['# 盲判材料独立泄漏审计（r6_leak_qa.py）', '',
         '- 时间：%s ｜ 结论：**%s**' % (time.strftime('%Y-%m-%d %H:%M:%S'), 'PASS' if allpass else 'FAIL'),
         '- 受检面：`%s`、`%s`（`private/` 为唯一映射，不在受检面，判读代理不得读）' % (BLIND, MAT), '',
         '| 检查 | 结果 | 关键值 |', '|---|---|---|']
    for k, v in res.items():
        if isinstance(v, dict) and 'pass_' in v:
            L.append('| %s | %s | %s |' % (k, 'PASS' if v['pass_'] else 'FAIL',
                                           json.dumps({kk: vv for kk, vv in v.items() if kk != 'pass_'}, ensure_ascii=False)[:220]))
    L += ['', '## 口径声明', '',
          '- 本审计只证明**盲判面不含先验**与空间/角色纪律成立，不证明判读结论正确；',
          '- 高分底图无日期元数据 → `blind/points_series.csv` 一律 `date_status=unknown`；',
          '- "确认率/一致率"是判读结果不是精度；解盲匹配与 QC 三率在 r7 之后由主 agent 汇总。']
    open(os.path.join(QC, 'leak_qa_report.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    emit('泄漏审计 %s（%d 项）→ qc/leak_qa_report.md' % ('PASS' if allpass else 'FAIL',
                                                  sum(1 for v in res.values() if isinstance(v, dict) and 'pass_' in v)))
    for k, v in res.items():
        if isinstance(v, dict) and 'pass_' in v and not v['pass_']:
            emit('  FAIL %s：%s' % (k, json.dumps({kk: vv for kk, vv in v.items() if kk != 'pass_'}, ensure_ascii=False)[:200]))


if __name__ == '__main__':
    main()
