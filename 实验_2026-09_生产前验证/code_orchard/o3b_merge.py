# -*- coding: utf-8 -*-
"""o3b_merge.py — 乔灌园地：合并子代理判读 CSV → judge_merged.csv

* 输入：data/orchard/judge/judge_*.csv（point_id,Q1,Q2,Q3,conf,note；UTF-8-SIG）
        + data/orchard/judge_points.csv（点表）
* 规则源：按 point_id 去重（首见保留）；缺判点列出
* 门槛：覆盖率与 Q1 分布必须打印
* 输出：data/orchard/judge_merged.csv + results/orchard/judge_stats.json
* 用法：python o3b_merge.py
"""
import csv
import glob
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import v31_common as VC

ORD = r'F:/lc_work/v31_exp/data/orchard'
RES = r'F:/lc_work/v31_exp/results/orchard'


def main():
    os.makedirs(RES, exist_ok=True)
    files = sorted(glob.glob(os.path.join(ORD, 'judge', 'judge_*.csv')))
    rows, dup, seen = [], 0, set()
    for fp in files:
        for r in csv.DictReader(open(fp, encoding='utf-8-sig')):
            pid = (r.get('point_id') or '').strip()
            if not pid:
                continue
            if pid in seen:
                dup += 1
                continue
            seen.add(pid)
            rows.append({'point_id': pid, 'Q1': (r.get('Q1') or '').strip(),
                         'Q2': (r.get('Q2') or '').strip(), 'Q3': (r.get('Q3') or '').strip(),
                         'conf': (r.get('conf') or '').strip(), 'note': (r.get('note') or '').strip(),
                         'file': os.path.basename(fp)})
    j = pd.DataFrame(rows)
    pts = pd.read_csv(os.path.join(ORD, 'judge_points.csv'), encoding='utf-8-sig')
    m = pts.merge(j, on='point_id', how='left')
    m.to_csv(os.path.join(ORD, 'judge_merged.csv'), index=False, encoding='utf-8-sig')
    stat = {'n_points': int(len(m)), 'n_judged': int(m.Q1.notna().sum()),
            'n_missing': int(m.Q1.isna().sum()), 'duplicates': int(dup), 'files': len(files),
            'Q1_dist': m['Q1'].fillna('(缺)').value_counts().to_dict(),
            'Q2_top': m[m.Q1 == '是']['Q2'].value_counts().to_dict(),
            'Q3_top': m[m.Q1 == '否']['Q3'].value_counts().head(15).to_dict(),
            'by_frame_Q1yes': m[m.Q1 == '是'].frame.value_counts().to_dict(),
            'by_frame_total': m.frame.value_counts().to_dict()}
    VC.jsave(stat, os.path.join(RES, 'judge_stats.json'))
    print(json.dumps(stat, ensure_ascii=False, indent=1))
    print('→', os.path.join(ORD, 'judge_merged.csv'))


if __name__ == '__main__':
    main()
