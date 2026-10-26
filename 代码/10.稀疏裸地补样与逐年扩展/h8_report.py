# -*- coding: utf-8 -*-
"""h8_report.py — 汇总补样交付报告（读各环节 stats/json → markdown）

* 输入（缺则跳过）：data/candidates/cand_{north,east}_stats.json、data/final/qc_stats.json、
  data/final/verify.json、delivery/delivery_stats.json、delivery/分省统计.csv、
  data/pool_v3_stats.json、logs/health_all.json
* 输出：reports/bare_sparse_v3_report.md
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h1_harvest_fcs10 as H  # noqa
BASE = r'F:/lc_work/bare_sparse_v3'


def jload(fp):
    try:
        return json.load(open(fp, encoding='utf-8'))
    except Exception:
        return None


def main():
    os.makedirs(os.path.join(BASE, 'reports'), exist_ok=True)
    lines = []
    A = lines.append
    A('# 裸地 / 稀疏植被 东北·东部补样（池 v3 候选层）—— 交付报告\n')
    A('> 2026-10-25 ｜ 判据：`spec/预注册判据.md` ｜ 触发与诊断：`F:/lc_work/v31_exp/reports/bare_sparse_east_gap_2026-10-14.md`\n')

    A('## 1. 采集（本地 FCS10 2023，10 m）\n')
    for reg in ('north', 'east', 'ne130'):
        s = jload(os.path.join(BASE, 'data', 'candidates', 'cand_%s_stats.json' % reg))
        if not s:
            continue
        A('- **%s**：blocks=%s，扫描 %.0f s' % (reg, s.get('blocks'), s.get('elapsed_scan_sec', 0)))
        for fam, v in (s.get('fams') or {}).items():
            A('  - %s：块内配额候选 %s → 去重后落格 %s → 封顶后 **%s**' % (
                fam, v.get('raw_candidates'), v.get('after_thin'), v.get('final')))
            for nm, sv in (v.get('subbox') or {}).items():
                A('    - %s：%s/%s' % (nm, sv.get('kept'), sv.get('target')))
        sc = s.get('self_check')
        if sc:
            A('  - 自检：%s' % json.dumps(sc, ensure_ascii=False, default=int))

    A('\n## 2. 质量裁定（预注册判据）\n')
    q = jload(os.path.join(BASE, 'data', 'final', 'qc_stats.json'))
    if q:
        A('| 类 | 候选 | keep | quarantine | flag |')
        A('|---|---|---|---|---|')
        for fam, v in q.get('total', {}).items():
            A('| %s | %s | %s | %s | %s |' % (fam, v['n'], v['keep'], v['quarantine'], v['flag']))
        A('')
        A('<details><summary>分窗明细</summary>\n')
        A('```json')
        A(json.dumps(q.get('by_subwin', {}), ensure_ascii=False, indent=1))
        A('```\n</details>')

    A('\n## 3. 独立核验\n')
    v = jload(os.path.join(BASE, 'data', 'final', 'verify.json'))
    if v:
        A('- 栅格回读：%s' % json.dumps(v.get('raster', {}), ensure_ascii=False))
        A('- 证据双路：%s' % json.dumps(v.get('evidence_double_arm', {}), ensure_ascii=False))
        A('- AEF：%s' % json.dumps(v.get('aef', {}), ensure_ascii=False))

    A('\n## 4. 交付\n')
    d = jload(os.path.join(BASE, 'delivery', 'delivery_stats.json'))
    if d:
        A('- 稀疏植被 **%s** 点 / 裸地 **%s** 点 / AEF 全缺失 %s' % (
            d.get('sparse'), d.get('bare'), d.get('aef_missing')))
    st = os.path.join(BASE, 'delivery', '分省统计.csv')
    if os.path.isfile(st):
        A('\n分省（前 15 行）：\n')
        A('```')
        A(open(st, encoding='utf-8-sig').read().split('\n\n')[0][:1500])
        A('```')

    dp = os.path.join(BASE, 'delivery', 'bare_sparse_v3_2023.parquet')
    ar = jload(os.path.join(BASE, 'data', 'final', 'fcs10_areas.json'))
    if os.path.isfile(dp):
        dd = pd.read_parquet(dp)
        A('')
        A('### 4.1 分窗交付量 / FCS10 面积对账')
        A('| 区 | 分窗 | 稀疏点 | 裸地点 | FCS10 稀疏 km2 | FCS10 裸地 km2 | 稀疏点/km2 | 裸地点/km2 |')
        A('|---|---|---|---|---|---|---|---|')
        for reg in ('north', 'east', 'ne130'):
            x0, y0, x1, y1 = H.REGIONS[reg]
            sub = dd[(dd.lon >= x0) & (dd.lon < x1) & (dd.lat >= y0) & (dd.lat < y1)].copy()
            subs = H.SUBBOX[reg]
            idx = np.full(len(sub), -1, dtype=int)
            lonv = sub.lon.to_numpy(); latv = sub.lat.to_numpy()
            for k, (nm, bb, _, _) in enumerate(subs):
                if bb is None:
                    continue
                m = (lonv >= bb[0]) & (lonv < bb[2]) & (latv >= bb[1]) & (latv < bb[3]) & (idx < 0)
                idx[m] = k
            idx[idx < 0] = len(subs) - 1
            sub['_k'] = idx
            for k, (nm, bb, _, _) in enumerate(subs):
                g2 = sub[sub._k == k]
                nsp = int((g2.fam == 'sparse').sum()); nba = int((g2.fam == 'bare').sum())
                a2 = (ar or {}).get(reg, {}).get(nm, {})
                sk = a2.get('sparse_km2'); bk = a2.get('bare_km2')
                A('| %s | %s | %d | %d | %s | %s | %s | %s |' % (
                    reg, nm, nsp, nba, sk, bk,
                    ('%.3f' % (nsp / sk)) if sk else '-', ('%.3f' % (nba / bk)) if bk else '-'))

        inside = pd.Series(False, index=dd.index)
        for reg in ('north', 'east', 'ne130'):
            x0, y0, x1, y1 = H.REGIONS[reg]
            inside |= ((dd.lon >= x0) & (dd.lon < x1) & (dd.lat >= y0) & (dd.lat < y1))
        out_ = dd[~inside]
        A('| - | bbox 外（1°块边缘溢出） | %d | %d | - | - | - | - |' % (
            int((out_.fam == 'sparse').sum()), int((out_.fam == 'bare').sum())))

    A('\n## 5. 池 v3 合并\n')
    p = jload(os.path.join(BASE, 'data', 'pool_v3_stats.json'))
    if p:
        A('- v2 %s + 补样 %s（同类 50 m 去重 %s）→ **v3 %s**' % (
            p.get('n_v2'), p.get('n_topup_added'), p.get('n_dup_removed'), p.get('n_v3')))
        A('- 补样按类：%s' % json.dumps(p.get('topup_by_class'), ensure_ascii=False))
        A('- 合并后稀疏植被 %s / 裸地 %s' % (
            p.get('by_class', {}).get('13'), p.get('by_class', {}).get('22')))
    for name, fn in [('球面距离复检（h3c）', 'spacing_fix.json'), ('国界过滤（h6b）', 'china_filter.json')]:
        v2 = jload(os.path.join(BASE, 'data', 'final', fn))
        if v2:
            A('- %s：%s' % (name, json.dumps({k: v for k, v in v2.items() if k != 'file'},
                                             ensure_ascii=False)))


    A('\n## 6. 账号与执行环境\n')
    h = jload(os.path.join(BASE, 'logs', 'health_all.json'))
    if h:
        A('- 全量健康检查 %s：113 账号中 healthy=%d（%s）；其余多为 SOCKS 连接失败/restricted-mode/项目未解析' % (
            h.get('time'), len(h.get('healthy', [])), ', '.join(h.get('healthy', []))))
    A('- 本机：24 核 / 32 线程 / 66.8 GB 内存；采集 2 进程并行（各约 2.5 GB）')
    A('- E 盘 FCS10 的 130–135°E 带已于 2026-10-25 补齐（Zenodo 4.37 GB，MD5 一致），并扩采该带裸地（中国境内裸地约 1,190 km²；稀疏仅约 9 km²）')

    A('\n## 7. 局限与使用提示\n')
    A('1. 本层是 **FCS10-2023 派生**（与既有池的 FCS30D-2020 教师不同源），按 src 可整体回退。')
    A('2. 与既有样本的 250 m 去重已做；如需更严格（如 1 km）请重跑 h1 的 dedupe 半径。')
    A('3. 池 v3 需要在服务端重建资产后才会进入生产（本次未上传 GEE）。')
    A('4. 2017–2024 逐年面板未做（本层仅 2023）；逐年扩展需按 doc63/65 的年匹配流程另跑。')
    fp = os.path.join(BASE, 'reports', 'bare_sparse_v3_report.md')
    open(fp, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print('→', fp)


if __name__ == '__main__':
    main()
