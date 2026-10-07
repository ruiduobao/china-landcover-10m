# -*- coding: utf-8 -*-
"""
ng3_status.py — 下一代补样：进度速查（本地只读，零 EECU）

逐 spec 汇总：分片清单数 / 已落盘分片数（含空壳与坏块识别）/ 候选点总行数 /
最终过门槛样本数 / 最近落盘时间。用于无人值守时判断"在跑 / 卡住 / 跑完"。

用法: python ng3_status.py
"""
import os
import glob
import json
import time
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ng0_paths as P
import ng0_spec as S


def parquet_rows(fp):
    """轻量读取行数；坏块（非 PAR1 魔数）单独标记。"""
    try:
        with open(fp, 'rb') as f:
            head = f.read(4)
            f.seek(-4, os.SEEK_END)
            tail = f.read(4)
        if head != b'PAR1' or tail != b'PAR1':
            return -1, 'BAD_MAGIC'
        if os.path.getsize(fp) < 200:
            return -1, 'TOO_SMALL'
        import pyarrow.parquet as pq
        md = pq.ParquetFile(fp).metadata
        return int(md.num_rows), 'ok'
    except Exception as e:
        return -1, 'ERR:' + str(e)[:40]


def main():
    specs = list(S.SPECS)
    print(f'{"spec":14s}{"家族":10s}{"分片":>6}{"落盘":>6}{"候选点数":>12}'
          f'{"过门槛":>9}  最近落盘')
    tot_raw = tot_gate = 0
    for sp in specs:
        pf = P.shard_plan_path(sp)
        n_shard = len(json.load(open(pf, encoding='utf-8'))['shards']) if os.path.isfile(pf) else 0
        files = sorted(glob.glob(os.path.join(P.RAW, sp, 'raw_*.parquet')))
        rows = 0
        bad = 0
        latest = 0.0
        for f in files:
            n, st = parquet_rows(f)
            if n < 0:
                bad += 1
            else:
                rows += n
                latest = max(latest, os.path.getmtime(f))
        gate_fp = P.gate_path(sp)
        gate_n = 0
        if os.path.isfile(gate_fp):
            gate_n, _ = parquet_rows(gate_fp)
            gate_n = max(0, gate_n)
        tot_raw += max(0, rows)
        tot_gate += gate_n
        lt = time.strftime('%m-%d %H:%M', time.localtime(latest)) if latest else '-'
        flag = f'({bad} 坏块)' if bad else ''
        print(f'{sp:14s}{S.SPECS[sp]["tag"]:10s}{n_shard:>6}{len(files):>6}{rows:>12,}'
              f'{gate_n:>9,}  {lt} {flag}')
    print(f'{"合计":14s}{"":10s}{"":>6}{"":>6}{tot_raw:>12,}{tot_gate:>9,}')


if __name__ == '__main__':
    main()
