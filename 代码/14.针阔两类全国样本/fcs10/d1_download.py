# -*- coding: utf-8 -*-
"""d1_download.py — FCS10-2023 全国 zip 分块并发下载（断点续传 + MD5 校验）
* 输入 / cn_zips.json（key/size/url）
* 规则源 / 本任务纪律：并发 <=4、幂等可续、状态落 json
* 门槛 / 每块字节数 == 预期；整档 md5 == Zenodo checksum
* 输出 / Z:/fcs10_dl/zip/<key>、state/download_state.json
* 用法 / python d1_download.py [--workers 4] [--chunk-mb 32] [--only KEY...]
"""
import os, sys, json, time, hashlib, argparse, threading, queue
import urllib.request
import urllib.error

DL = r'Z:/nb_national_2026/fcs10_dl'
ZIPDIR = os.path.join(DL, 'zip')
PARTDIR = os.path.join(DL, 'parts')
STATE = os.path.join(DL, 'state', 'download_state.json')
MANIFEST = r'F:/lc_work/nb_national_2026/fcs10_dl/cn_zips.json'

CHUNK = 32 * 1024 * 1024
RETRY = 6


def md5_of(path, bufsize=1 << 22):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_state():
    if os.path.exists(STATE):
        with open(STATE, encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_state(st):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    tmp = STATE + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(st, f, ensure_ascii=False, indent=1)
    os.replace(tmp, STATE)


def fetch_range(url, start, end, dst, expect):
    """下载 [start,end] 闭区间到 dst，返回字节数。带重试。"""
    for attempt in range(RETRY):
        got = os.path.getsize(dst) if os.path.exists(dst) else 0
        if got == expect:
            return expect
        if got > expect:
            os.remove(dst)
            got = 0
        s = start + got
        if s > end:
            return got
        req = urllib.request.Request(url, headers={
            'Range': 'bytes=%d-%d' % (s, end),
            'User-Agent': 'Mozilla/5.0 (fcs10-dl)',
            'Accept-Encoding': 'identity',
        })
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                if r.status not in (200, 206):
                    raise IOError('http %s' % r.status)
                mode = 'ab' if got else 'wb'
                with open(dst, mode) as f:
                    while True:
                        b = r.read(1 << 20)
                        if not b:
                            break
                        f.write(b)
        except Exception as e:
            time.sleep(min(30, 2 ** attempt))
            continue
    got = os.path.getsize(dst) if os.path.exists(dst) else 0
    if got != expect:
        raise IOError('chunk incomplete %d/%d' % (got, expect))
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--chunk-mb', type=int, default=32)
    ap.add_argument('--only', nargs='*')
    a = ap.parse_args()
    chunk = a.chunk_mb * 1024 * 1024

    os.makedirs(ZIPDIR, exist_ok=True)
    os.makedirs(PARTDIR, exist_ok=True)
    with open(MANIFEST, encoding='utf-8') as f:
        files = json.load(f)
    if a.only:
        files = [x for x in files if any(o in x['key'] for o in a.only)]

    # checksum from Zenodo (cached)
    ck_path = os.path.join(DL, 'state', 'zenodo_checksums.json')
    os.makedirs(os.path.dirname(ck_path), exist_ok=True)
    if os.path.exists(ck_path):
        with open(ck_path, encoding='utf-8') as f:
            cks = json.load(f)
    else:
        cks = {}
        with urllib.request.urlopen('https://zenodo.org/api/records/14729665', timeout=60) as r:
            rec = json.load(r)
        for x in rec['files']:
            cks[x['key']] = dict(size=x['size'], md5=x['checksum'].split(':')[-1])
        with open(ck_path, 'w', encoding='utf-8') as f:
            json.dump(cks, f, ensure_ascii=False, indent=1)

    st = load_state()
    # 构建任务队列
    q = queue.Queue()
    tasks = {}
    for fi in files:
        key, size, url = fi['key'], fi['size'], fi['url']
        exp = cks.get(key, {}).get('size', size)
        st.setdefault(key, dict(size=exp, md5=cks.get(key, {}).get('md5'), url=url,
                                status='pending', chunks_done=0, n_chunks=0))
        zip_path = os.path.join(ZIPDIR, key)
        if os.path.exists(zip_path) and os.path.getsize(zip_path) == exp:
            h = md5_of(zip_path)
            if h == st[key]['md5']:
                st[key].update(status='done', chunks_done=-1, md5_actual=h)
                print('[skip] %s 已完成 md5 一致' % key, flush=True)
                continue
            else:
                print('[redo] %s md5 不符 %s != %s' % (key, h, st[key]['md5']), flush=True)
                os.remove(zip_path)
        n = (exp + chunk - 1) // chunk
        st[key].update(n_chunks=n, status='downloading', chunk=chunk)
        pdir = os.path.join(PARTDIR, key)
        os.makedirs(pdir, exist_ok=True)
        for i in range(n):
            s = i * chunk
            e = min(exp, s + chunk) - 1
            fp = os.path.join(pdir, 'c%05d' % i)
            tasks[(key, i)] = dict(url=url, start=s, end=e, dst=fp, expect=e - s + 1)
            if not (os.path.exists(fp) and os.path.getsize(fp) == e - s + 1):
                q.put((key, i))
        have = 0
        for i in range(n):
            fp = os.path.join(pdir, 'c%05d' % i)
            csz = min(exp, i * chunk + chunk) - i * chunk
            if os.path.exists(fp) and os.path.getsize(fp) == csz:
                have += 1
        st[key]['chunks_done'] = have
    save_state(st)
    total_chunks = q.qsize()
    print('待下载块数 %d（并发 %d，块 %d MB）' % (total_chunks, a.workers, a.chunk_mb), flush=True)
    if total_chunks == 0:
        print('无待下载块', flush=True)
        return

    done = [0]
    lock = threading.Lock()
    t0 = time.time()
    errs = []

    def worker():
        while True:
            try:
                key, i = q.get_nowait()
            except queue.Empty:
                return
            t = tasks[(key, i)]
            try:
                fetch_range(t['url'], t['start'], t['end'], t['dst'], t['expect'])
                with lock:
                    done[0] += 1
                    if done[0] % 5 == 0 or done[0] == total_chunks:
                        el = time.time() - t0
                        got = sum(os.path.getsize(tasks[k]['dst']) for k in tasks if os.path.exists(tasks[k]['dst']))
                        print('  %d/%d  %.1f MB  %.2f MB/s  ETA %.1f min' % (
                            done[0], total_chunks, got / 1e6, got / 1e6 / el,
                            (total_chunks - done[0]) * (el / done[0]) / 60), flush=True)
            except Exception as e:
                errs.append((key, i, str(e)))
                print('  [ERR] %s chunk %d: %s' % (key, i, e), flush=True)
            finally:
                q.task_done()

    ths = [threading.Thread(target=worker, daemon=True) for _ in range(a.workers)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()

    # 合并 + 校验
    for fi in files:
        key = fi['key']
        if st[key]['status'] == 'done':
            continue
        pdir = os.path.join(PARTDIR, key)
        exp = st[key]['size']
        n = st[key]['n_chunks']
        parts, miss = [], []
        for i in range(n):
            p = os.path.join(pdir, 'c%05d' % i)
            parts.append(p)
            csz = min(exp, i * chunk + chunk) - i * chunk
            if not (os.path.exists(p) and os.path.getsize(p) == csz):
                miss.append(p)
        if miss:
            st[key]['status'] = 'incomplete'
            print('[incomplete] %s 缺 %d 块' % (key, len(miss)), flush=True)
            continue
        zip_path = os.path.join(ZIPDIR, key)
        print('[merge] %s ...' % key, flush=True)
        with open(zip_path, 'wb') as out:
            for p in parts:
                with open(p, 'rb') as f:
                    while True:
                        b = f.read(1 << 22)
                        if not b:
                            break
                        out.write(b)
        h = md5_of(zip_path)
        ok = h == st[key]['md5']
        st[key].update(status='done' if ok else 'md5_mismatch', md5_actual=h)
        print('[%s] %s md5=%s' % ('OK' if ok else 'MD5FAIL', key, h), flush=True)
        save_state(st)

    save_state(st)
    print('完成。错误 %d' % len(errs), flush=True)
    for k in st:
        print('  ', k, st[k]['status'], st[k].get('md5_actual'))


if __name__ == '__main__':
    main()
