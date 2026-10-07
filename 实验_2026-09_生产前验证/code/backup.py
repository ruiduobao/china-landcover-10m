# -*- coding: utf-8 -*-
"""backup.py — 每次 advance 运行前的快照（用户纪律：跑前必备份）
策略：code/config/reports + STATE/jobs + 小体量数据(data 下 parquet/json) 全量拷贝；
大体量源文件（r7_train_2023.parquet ~330MB、prod_sample_sqrt.csv 84MB）只记
size+mtime+首尾各1MB的MD5指纹（免每轮复制，冻结性靠指纹不变验证）。
保留最近 30 份。
"""
import os, sys, shutil, hashlib, json, time
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v31_common as VC

BIG_FILES = [VC.SRC2023, VC.SQRT_CSV,
             os.path.join(VC.EVAL_DIR_GEE, 'eval_w1.parquet'),
             os.path.join(VC.EVAL_DIR_GEE, 'eval_w2.parquet'),
             os.path.join(VC.EVAL_DIR_GEE, 'eval_w3.parquet'),
             os.path.join(VC.EVAL_DIR_GEE, 'eval_w4.parquet')]
KEEP = 30


def fp_md5_headtail(fp, mb=1):
    h = hashlib.md5()
    n = mb * 1024 * 1024
    sz = os.path.getsize(fp)
    with open(fp, 'rb') as f:
        h.update(f.read(n))
        if sz > 2 * n:
            f.seek(-n, 2)
        h.update(f.read(n))
    return h.hexdigest()


def fingerprint(fp):
    st = os.stat(fp)
    return dict(size=st.st_size, mtime=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_mtime)),
                headtail_md5=fp_md5_headtail(fp))


def main():
    ts = time.strftime('%Y%m%d_%H%M%S')
    dst = os.path.join(VC.BAKS, ts)
    os.makedirs(dst, exist_ok=True)
    manifest = dict(time=ts, files={}, big=[])
    for sub in ['code', 'config']:
        d = os.path.join(dst, sub)
        shutil.copytree(os.path.join(VC.ROOT, sub), d, dirs_exist_ok=True)
        for f in sorted(os.listdir(d)):
            manifest['files']['%s/%s' % (sub, f)] = os.path.getsize(os.path.join(d, f))
    for fp in [VC.STATE_FP, VC.JOBS_FP]:
        if os.path.exists(fp):
            shutil.copy2(fp, dst)
            manifest['files'][os.path.basename(fp)] = os.path.getsize(fp)
    for f in os.listdir(VC.REPT):
        if f.endswith('.md'):
            shutil.copy2(os.path.join(VC.REPT, f), os.path.join(dst, f))
    ddir = os.path.join(dst, 'data')
    os.makedirs(ddir, exist_ok=True)
    for f in os.listdir(VC.DATA):
        if f.endswith(('.parquet', '.json')):
            src = os.path.join(VC.DATA, f)
            if os.path.getsize(src) < 20 * 1024 * 1024:
                shutil.copy2(src, os.path.join(ddir, f))
                manifest['files']['data/%s' % f] = os.path.getsize(src)
    for fp in BIG_FILES:
        if os.path.exists(fp):
            manifest['big'].append(dict(path=fp, **fingerprint(fp)))
    with open(os.path.join(dst, 'BACKUP_MANIFEST.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    # 冻结性校验：与上一份备份的 big 指纹比对
    olds = sorted(d for d in os.listdir(VC.BAKS) if d != ts)
    msg = []
    if olds:
        prev = VC.jload(os.path.join(VC.BAKS, olds[-1], 'BACKUP_MANIFEST.json'), {})
        pb = {b['path']: b for b in prev.get('big', [])}
        for b in manifest['big']:
            p = pb.get(b['path'])
            if p and (p['size'] != b['size'] or p['headtail_md5'] != b['headtail_md5']):
                msg.append('⚠️ 源文件变动: %s' % b['path'])
    # 清老
    allb = sorted(os.listdir(VC.BAKS))
    for d in allb[:-KEEP]:
        shutil.rmtree(os.path.join(VC.BAKS, d), ignore_errors=True)
    VC.emit('备份 → %s（%d 文件）%s' % (ts, len(manifest['files']),
                                      ('；' + ';'.join(msg)) if msg else '；大文件指纹无变化'))


if __name__ == '__main__':
    main()
