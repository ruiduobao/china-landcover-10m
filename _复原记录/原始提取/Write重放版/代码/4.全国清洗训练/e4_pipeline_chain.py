# -*- coding: utf-8 -*-
"""
e4_pipeline_chain.py — 无人值守接续链：等嵌入提取全部完成 → 自动跑 e2 清洗 → e3 训练
* 用法: python e4_pipeline_chain.py   （后台长跑）
"""
import os, sys, time, glob, subprocess

IDX = '数据/本地处理/全国清洗训练/chunks_index.parquet'
PARTS = '数据/本地处理/全国清洗训练/emb_parts'
CODE = '代码/4.全国清洗训练'
LOG = '数据/本地处理/全国清洗训练/pipeline_chain.log'

def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(time.strftime('%H:%M:%S ') + str(msg) + '\n')
    print(msg, flush=True)

def total_chunks():
    import pyarrow.parquet as pq
    t = pq.read_table(IDX, columns=['chunk_id']).to_pandas()
    return t.chunk_id.nunique()

def done_chunks():
    return len(glob.glob(os.path.join(PARTS, 'chunk_*.parquet')))

def main():
    total = total_chunks()
    log(f'接续链启动: 总块 {total}')
    # 1) 等提取完成（容差：>= 总块-5 视为完成，个别块失败可接受）
    stall = 0
    last_done = 0
    while True:
        d = done_chunks()
        log(f'已完成 {d}/{total}')
        if d >= total - 5:
            log('提取完成（或仅剩个别失败块）')
            break
        if d == last_done:
            stall += 1
            if stall >= 36:      # 6 小时无进展 → 判定卡死
                log('警告: 6 小时无进展，强制进入下一阶段')
                break
        else:
            stall = 0
        last_done = d
        time.sleep(600)
    # 2) e2 清洗
    log('== 启动 e2 清洗 ==')
    r = subprocess.run([sys.executable, '-u', os.path.join(CODE, 'e2_clean.py')],
                       capture_output=True, text=True)
    log(r.stdout[-2000:] if r.stdout else '')
    if r.returncode != 0:
        log(f'e2 失败: {r.stderr[-800:]}')
        return
    # 3) e3 分区训练
    log('== 启动 e3 分区训练 ==')
    r = subprocess.run([sys.executable, '-u', os.path.join(CODE, 'e3_train_zones.py')],
                       capture_output=True, text=True)
    log(r.stdout[-3000:] if r.stdout else '')
    if r.returncode != 0:
        log(f'e3 失败: {r.stderr[-800:]}')
        return
    log('== 全链完成: v5 清洗样本 + 分区模型 + 评估报告 ==')

if __name__ == '__main__':
    main()
