#!/bin/bash
# landqc_status.sh <crop|veg|water|urban> — 本组抓取进度（点覆盖数 vs 期望；失败块；运行中进程数）
cd F:/lc_work/v31_exp/code
BASE=../data/land_qc
python - "$1" <<'EOF'
import glob, json, os, sys
import pandas as pd
tag = sys.argv[1]
gidx = {'crop': 1, 'veg': 2, 'water': 3, 'urban': 4}[tag]
BASE = '../data/land_qc'
d = pd.read_parquet(os.path.join(BASE, 'lists', 'grp%d_%s.parquet' % (gidx, tag)), columns=['point_id'])
fs = sorted(glob.glob(os.path.join(BASE, 'feats', tag, 'land_p*_c*.parquet')))
tot, ids = 0, set()
for f in fs:
    t = pd.read_parquet(f, columns=['point_id'])
    tot += len(t); ids.update(t.point_id.astype(str))
fail = 0
for f in glob.glob(os.path.join(BASE, 'feats', tag, '_fail_*.json')):
    fail += sum(1 for _ in open(f, encoding='utf-8'))
n = len(d)
print('%s: 期望 %d 点 / 块内 %d 点（去重 %d）/ 缺口 %d / 块数 %d / 失败块记录 %d'
      % (tag, n, tot, len(ids), n - len(ids), len(fs), fail))
EOF
ps -W 2>/dev/null | grep -c "python" || true
