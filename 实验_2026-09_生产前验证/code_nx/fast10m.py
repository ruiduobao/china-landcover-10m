# -*- coding: utf-8 -*-
"""fast10m.py — 用 geefast-download skill 的原生 getPixels 并发下载器拉取 10m 瓦片成品
切换 HOME 到持有账号凭据目录（沿用项目既有认证方式），再调用 skill 脚本。
"""
import os, sys, subprocess, time
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v31_common as VC
import t_all as T

SCRIPT = r'C:/Users/Administrator/.zcode/skills/geefast-download/scripts/gee_rest_compute_pixels.py'
OUTD = os.path.join(VC.RES, 'tiles')


def main():
    only = sys.argv[sys.argv.index('--tile') + 1] if '--tile' in sys.argv else None
    workers = sys.argv[sys.argv.index('--workers') + 1] if '--workers' in sys.argv else '20'
    os.makedirs(OUTD, exist_ok=True)
    for t, meta in T.TILES.items():
        if only and t != only:
            continue
        w = T.winner_of(t)
        if w['state'] != 'COMPLETED':
            VC.emit('%s 未完成' % t); continue
        host, asset = w['info']['host'], w['info']['asset']
        out = os.path.join(OUTD, '%s_10m.tif' % t)
        if os.path.exists(out) and os.path.getsize(out) > 1e7:
            VC.emit('%s 已存在 %s' % (t, out)); continue
        env = dict(os.environ)
        env['HOME'] = VC.C.cred_home(host)
        env['USERPROFILE'] = VC.C.cred_home(host)
        env['EE_PROJECT'] = VC.pid_of(host)
        cmd = [sys.executable, SCRIPT, '--image', asset, '--bbox', ','.join(str(x) for x in meta['box']),
               '--bands', 'class', '--scale', '10', '--auto-tile', '--workers', workers,
               '--dtype', 'uint16', '--output', out, '--overwrite']
        VC.emit('%s @%s → %s' % (t, host, out))
        t0 = time.time()
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        dt = time.time() - t0
        tail = (r.stdout or '')[-500:].replace('\n', ' | ')
        if r.returncode != 0:
            VC.emit('  ❌ 失败（%.1f 分钟）: %s' % (dt / 60, tail[-300:]))
        else:
            sz = os.path.getsize(out) / 1e6 if os.path.exists(out) else 0
            VC.emit('  ✅ 完成 %.1f 分钟 · %.0f MB · %.1f MB/分钟' % (dt / 60, sz, sz / max(dt / 60, 1e-6)))


if __name__ == '__main__':
    main()
