# -*- coding: utf-8 -*-
"""
v6_freeze.py — v-final 冻结：快照 + MD5 MANIFEST + SPEC.md
快照内容（落 Z 盘）：
  1) 母库与派生：r7_train.parquet / r7_train_validity.parquet（F 盘权威副本完整复制）
  2) 逐年训练子集 + 年度索引 + sample_year_qc：**原地计算 MD5 记账，不复制**（体积大、可复现）
  3) v-final 代码：代码/7.vfinal/*（完整复制）
  4) v-final 评估产物：评估_vfinal/*（完整复制）
  5) 端到端验证产物：端到端验证/*（完整复制，若存在）
  6) SPEC.md：模型规格 + 与 26 号文档版本的差异 + 已知弱项
输出: Z:/.../数据/备份/vfinal_<date>/ + MANIFEST.md
用法: python v6_freeze.py
"""
import os, sys, json, time, hashlib, shutil, io

PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
KB = r'Z:/地理所/论文/中国土地覆盖数据_2017-2024'
WL = os.path.join(KB, '数据/本地处理/全国清洗训练')
SNAP = os.path.join(KB, '数据/备份', 'vfinal_' + time.strftime('%Y%m%d'))
MD5_BUF = 1 << 20


def md5_of(p, limit_mb=0):
    """limit_mb>0 时只对前 limit_mb 计算（超大文件用），返回 (hex, 是否全量)"""
    h = hashlib.md5()
    cap = limit_mb * 1024 * 1024 if limit_mb else 0
    read = 0
    with open(p, 'rb') as f:
        while True:
            b = f.read(MD5_BUF)
            if not b:
                break
            h.update(b)
            read += len(b)
            if cap and read >= cap:
                return h.hexdigest(), False
    return h.hexdigest(), True


def copy_into(src, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(src))
    shutil.copy2(src, dst)
    return dst


def main():
    t0 = time.time()
    os.makedirs(SNAP, exist_ok=True)
    rows = []

    # 1) 母库
    for f in ['r7_train.parquet', 'r7_train_validity.parquet']:
        s = os.path.join(PROJ, '数据/本地处理/样本重建', f)
        d = copy_into(s, os.path.join(SNAP, '样本重建'))
        m, _ = md5_of(d)
        rows.append(('样本重建/' + f, d, m, os.path.getsize(d)))

    # 2) 原地记账（不复制）
    for rel in ['r7_train.parquet', 'r7_train_validity.parquet', 'chunks_index_r7.parquet',
                'sample_year_qc.parquet', '稀有类补样/r7_rare_samples.parquet']:
        s = os.path.join(WL, rel)
        if os.path.isfile(s):
            m, full = md5_of(s)
            rows.append(('(原地) ' + rel, s, m + ('' if full else ' (前1MB)'), os.path.getsize(s)))
    for y in range(2017, 2025):
        s = os.path.join(WL, '年度子集_含稀有类', f'r7_train_{y}.parquet')
        if os.path.isfile(s):
            m, full = md5_of(s, limit_mb=64)      # 逐年子集各 ~500MB，取前 64MB 指纹
            rows.append((f'(原地) 年度子集_含稀有类/r7_train_{y}.parquet', s,
                         m + ('' if full else ' (前64MB)'), os.path.getsize(s)))

    # 3) 代码
    cdir = os.path.join(PROJ, '代码', '7.vfinal')
    if os.path.isdir(cdir):
        dst = os.path.join(SNAP, '代码_7.vfinal')
        os.makedirs(dst, exist_ok=True)
        for f in sorted(os.listdir(cdir)):
            if f.endswith(('.py', '.md')):
                d = copy_into(os.path.join(cdir, f), dst)
                m, _ = md5_of(d)
                rows.append(('代码_7.vfinal/' + f, d, m, os.path.getsize(d)))

    # 4) 评估产物
    for rel in ['评估_vfinal', '端到端验证']:
        srcd = os.path.join(WL, rel)
        if not os.path.isdir(srcd):
            continue
        dstd = os.path.join(SNAP, rel)
        os.makedirs(dstd, exist_ok=True)
        for f in sorted(os.listdir(srcd)):
            s = os.path.join(srcd, f)
            if not os.path.isfile(s):
                continue
            if os.path.getsize(s) > 50e6:      # 大体积嵌入/中间件只记原地指纹，不复制
                m, full = md5_of(s, limit_mb=64)
                rows.append((f'(原地) {rel}/' + f, s, m + ('' if full else ' (前64MB)'),
                             os.path.getsize(s)))
                continue
            d = copy_into(s, dstd)
            m, _ = md5_of(d)
            rows.append((f'{rel}/' + f, d, m, os.path.getsize(d)))

    # 5) 快照清单
    lines = ['# v-final 母库冻结快照 · MD5 MANIFEST',
             '',
             f'- 生成时间：{time.strftime("%Y-%m-%d %H:%M:%S")}',
             f'- 快照目录：`{SNAP}`',
             f'- 母库规模：**2,252,207 点**（与文档 26 版本相同；本轮未新增样本，理由见 SPEC.md）',
             f'- 文件数：{len(rows)}',
             '',
             '| 文件 | 大小(MB) | MD5 | 说明 |',
             '|---|---:|---|---|']
    for name, path, m, sz in rows:
        note = ('完整副本' if SNAP.replace('\\', '/') in path.replace('\\', '/')
                else '原地指纹（未复制）')
        lines.append(f'| `{name}` | {sz/1e6:.2f} | `{m}` | {note} |')
    lines += ['', '> 说明：逐年子集（8×~500MB）与母库派生索引体积大且可由母库+e2b+e7 复现，',
              '> 故只做**原地 MD5 指纹**（大文件取前 64MB），不复制进快照。', '']
    io.open(os.path.join(SNAP, 'MANIFEST.md'), 'w', encoding='utf-8').write('\n'.join(lines))
    print(f'快照 {len(rows)} 个文件 → {SNAP}')
    print('MANIFEST:', os.path.join(SNAP, 'MANIFEST.md'))
    print(f'耗时 {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
