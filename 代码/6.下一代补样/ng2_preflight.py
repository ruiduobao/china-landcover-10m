# -*- coding: utf-8 -*-
"""
ng2_preflight.py — 试点前置体检（每个 spec 一次，EE 交互式只读，不消耗 batch 额度）

检查三件事，全绿才允许进入正式批量：
  1. 账号凭证可用（ee.Initialize 成功）；
  2. spec 的每个源数据集在当前账号下**可读**（.bandNames().getInfo()）；
  3. 门槛表达式能成功构图（build_gate 返回 5 元组且 band 名匹配）。

用法: python ng2_preflight.py --acct <账号> --spec moss140
      python ng2_preflight.py --acct <账号> --all-specs
"""
import os
import sys
import json
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ng0_paths as P
import ng0_spec as S

ACC_ROOT = r'F:\地理所\论文\博士论文\中期\实验区域\02_中间数据\gee_accounts'
PROXY = {'http': 'socks5h://127.0.0.1:7890', 'https': 'socks5h://127.0.0.1:7890'}


def boot(acct, pid):
    d = os.path.join(ACC_ROOT, acct)
    os.environ['HOME'] = d
    os.environ['USERPROFILE'] = d
    os.environ.setdefault('HTTP_PROXY', PROXY['http'])
    os.environ.setdefault('HTTPS_PROXY', PROXY['https'])
    import ee
    last = None
    for _k in range(4):
        try:
            ee.Initialize(project=pid)
            return ee
        except Exception as _e:      # 代理 SSL EOF 是常态（晚高峰更频繁），必须重试
            last = _e
            time.sleep(8 + 6 * _k)
    raise last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acct', required=True)
    ap.add_argument('--anchor', required=True)
    ap.add_argument('--spec')
    ap.add_argument('--all-specs', action='store_true')
    a = ap.parse_args()
    specs = list(S.SPECS) if a.all_specs else ([a.spec] if a.spec else [])
    if not specs:
        raise SystemExit('给 --spec 或 --all-specs')

    t0 = time.time()
    print(f'[preflight] 账号 {a.acct} @ {a.anchor}')
    try:
        ee = boot(a.acct, a.anchor)
        print('  ✅ ee.Initialize')
    except Exception as e:
        print('  ❌ ee.Initialize 失败:', str(e)[:200]); raise SystemExit(2)

    all_ok = True
    for key in specs:
        cfg = S.SPECS[key]
        box = S.REGIONS[cfg['region']]['box']
        print(f'\n-- {key} ({cfg["tag"]}, 类 {cfg["cls"]}) --')
        for ds in cfg['sources']:
            ok, det = True, ''
            # 通用探活：先按 Image 取波段，失败再按 ImageCollection 取。
            # **不要 size().getInfo()** —— Dynamic World 这类超大集合会卡死数分钟（2026-09-13 教训）。
            try:
                b = ee.Image(ds).bandNames().getInfo()
                det = f'Image {b}'
            except Exception:
                try:
                    b = ee.ImageCollection(ds).limit(1).first().bandNames().getInfo()
                    det = f'Collection {b}'
                except Exception as e:
                    ok, det = False, str(e)[:150]
            all_ok &= ok
            print(f'   {"OK " if ok else "NG "} {ds:44s} {det}')

        try:
            geom = ee.Geometry.Rectangle(box, proj='EPSG:4326', geodesic=False)
            stack, gate, bands, desc = S.build_gate(ee, cfg['gate'], geom)
            have = stack.bandNames().getInfo()
            missing = [b for b in bands if b not in have]
            ok = not missing
            all_ok &= ok
            print(f'   {"✅" if ok else "❌"} gate={cfg["gate"]} bands={have}'
                  + (f' 缺={missing}' if missing else ''))
        except Exception as e:
            all_ok = False
            print('   ❌ 构图失败:', str(e)[:200])
    print(f'\n[preflight] {"✅ 全绿" if all_ok else "❌ 有 FAIL"}  ({time.time()-t0:.0f}s)')
    raise SystemExit(0 if all_ok else 3)


if __name__ == '__main__':
    main()
