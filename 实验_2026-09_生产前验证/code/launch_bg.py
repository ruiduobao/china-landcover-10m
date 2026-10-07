# -*- coding: utf-8 -*-
"""launch_bg.py — 跨平台后台启动器（Windows 用 DETACHED_PROCESS，避免父 shell 退出被杀）
用法：python launch_bg.py <任务名> -- python <脚本> [参数...]
"""
import os, sys, subprocess

def main():
    if '--' not in sys.argv:
        raise SystemExit('用法: python launch_bg.py <tag> -- <cmd...>')
    i = sys.argv.index('--')
    tag = sys.argv[1] if i > 1 else 'bg'
    cmd = sys.argv[i + 1:]
    logd = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
    logd = os.path.abspath(logd)
    os.makedirs(logd, exist_ok=True)
    logf = os.path.join(logd, '%s.log' % tag)
    f = open(logf, 'ab')
    kw = {}
    if os.name == 'nt':
        kw['creationflags'] = 0x00000008 | 0x00000200   # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, **kw)
    print('%s → PID %d 日志 %s' % (tag, p.pid, logf))

if __name__ == '__main__':
    main()
