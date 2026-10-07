# -*- coding: utf-8 -*-
"""
ng1b_batch_plan.py — 批量分片规划（把试点小窗换成省域/生态区格网 + 存在性预扫）

流程
  1. 读 `ng0_spec.BATCH[spec]` 的 master 分区盒（生态区/海岸段/农区），按 parcel_deg 切成 parcel；
  2. GEE 端**存在性预扫**：`gate.selfMask().reduceRegions(parcels, Reducer.count(), scale≈500)`
     → 每个 parcel 的门槛像元数。这一步只读、极廉，避免在空 parcel 上白撒 3 万随机点；
  3. 保留 count ≥ min_count 的 parcel 作为 shard，写 `ng_idx/plan_<spec>_batch.json`；
  4. 输出汇总表（各区 parcel 数 / 非空数 / 门槛像元总数），供人工核对分区是否合理。

用法:
  python ng1b_batch_plan.py --acct <账号> --anchor <项目> --spec moss140
  python ng1b_batch_plan.py --acct <账号> --anchor <项目> --all
  python ng1b_batch_plan.py --spec moss140 --no-scan      # 不预扫，全部 parcel 入列（离线干跑）
"""
import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ng0_paths as P
import ng0_spec as S

ACC_ROOT = 'F:/地理所/论文/博士论文/中期/实验区域/02_中间数据/gee_accounts'
PROXY = {'http': 'socks5h://127.0.0.1:7890', 'https': 'socks5h://127.0.0.1:7890'}
SCAN_SCALE = 500          # 预扫分辨率（m）；只判"有没有"，不需要精度
CHUNK = 60                # 每次 reduceRegions 的 parcel 数（防单请求过大）


def boot(acct, pid):
    d = os.path.join(ACC_ROOT, acct)
    os.environ['HOME'] = d
    os.environ['USERPROFILE'] = d
    os.environ.setdefault('HTTP_PROXY', PROXY['http'])
    os.environ.setdefault('HTTPS_PROXY', PROXY['https'])
    import ee
    last = None
    for k in range(5):
        try:
            ee.Initialize(project=pid)
            return ee
        except Exception as e:
            last = e
            time.sleep(15 + 10 * k)
    raise last


def parcelize(zones, deg):
    """master 盒 → (zone_name, parcel_id, box) 列表"""
    out = []
    pid = 0
    for name, (x0, y0, x1, y1) in zones:
        nx = max(1, int(round((x1 - x0) / deg)))
        ny = max(1, int(round((y1 - y0) / deg)))
        for iy in range(ny):
            for ix in range(nx):
                bx0 = round(x0 + ix * deg, 4)
                by0 = round(y0 + iy * deg, 4)
                out.append((name, pid, [bx0, by0,
                                        round(min(bx0 + deg, x1), 4),
                                        round(min(by0 + deg, y1), 4)]))
                pid += 1
    return out


def presence_scan(ee, spec_key, parcels):
    """返回 parcel_id → 门槛像元数（scale=SCAN_SCALE）"""
    cfg = S.SPECS[spec_key]
    counts = {}
    for s in range(0, len(parcels), CHUNK):
        group = parcels[s:s + CHUNK]
        feats = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Rectangle(box, proj='EPSG:4326', geodesic=False),
                       {'pid': int(p)}) for _, p, box in group])
        geom = feats.geometry()
        _, gate, _, _ = S.build_gate(ee, cfg['gate'], geom)
        masked = gate.selfMask()
        for attempt in range(3):
            try:
                res = masked.reduceRegions(collection=feats,
                                           reducer=ee.Reducer.count(),
                                           scale=SCAN_SCALE, tileScale=4)
                got = res.getInfo()
                for f in got.get('features', []):
                    pr = f.get('properties', {})
                    counts[int(pr['pid'])] = int(pr.get('count', 0) or 0)
                print(f'    预扫 {s + len(group)}/{len(parcels)} parcel 完成', flush=True)
                break
            except Exception as e:
                print(f'    预扫 chunk@{s} 第{attempt+1}次失败: {str(e)[:110]}', flush=True)
                time.sleep(15 + 15 * attempt)
        else:
            for _, p, _ in group:
                counts.setdefault(int(p), -1)
    return counts


def run_spec(ee, spec_key, no_scan=False, max_shards=0, promote=False):
    cfg = S.SPECS[spec_key]
    b = S.BATCH[spec_key]
    parcels = parcelize(b['zones'], b['parcel'])
    print(f'\n== {spec_key} ({cfg["tag"]} 类{cfg["cls"]}) 分区 {len(b["zones"])} 个 → parcel {len(parcels)} 个 '
          f'(边长 {b["parcel"]}°, 阈值 {b["min_count"]})', flush=True)
    counts = {} if no_scan else presence_scan(ee, spec_key, parcels)
    shards = []
    zone_stat = {}
    for name, pid, box in parcels:
        c = b['min_count'] if no_scan else counts.get(pid, -1)
        st = zone_stat.setdefault(name, [0, 0, 0])
        st[0] += 1
        if c >= b['min_count']:
            st[1] += 1
            st[2] += max(0, c)
            shards.append({'shard': len(shards), 'parcel': pid, 'zone': name, 'box': box,
                           'presence': int(c)})
    # 有上限时按"门槛像元数"降序取前 N 个 parcel —— 让每个 shard 的期望产出最大
    if max_shards and len(shards) > max_shards:
        shards.sort(key=lambda x: -x['presence'])
        shards = shards[:max_shards]
    for i, sh in enumerate(shards):
        sh['shard'] = i
    plan = {'spec': spec_key, 'cls': cfg['cls'], 'gate': cfg['gate'], 'mode': 'batch',
            'parcel_deg': b['parcel'], 'min_count': b['min_count'],
            'scan_scale': SCAN_SCALE if not no_scan else None,
            'cand_per_shard': cfg['cand'], 'target': cfg['target'],
            'zones': [list(z) for z in b['zones']],
            'zone_stat': {k: {'parcels': v[0], 'nonempty': v[1], 'gate_px': v[2]}
                          for k, v in zone_stat.items()},
            'shards': shards}
    fp = os.path.join(P.IDX, f'plan_{spec_key}_batch.json')
    json.dump(plan, open(fp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    tot_px = sum(v[2] for v in zone_stat.values())
    print(f'   非空 parcel {len(shards)}/{len(parcels)}，门槛像元合计 ≈{tot_px:,}（{SCAN_SCALE}m）')
    for k, v in sorted(zone_stat.items(), key=lambda kv: -kv[1][2]):
        print(f'     {k:14s} parcel {v[0]:>4} 非空 {v[1]:>4}  门槛像元≈{v[2]:>10,}')
    print(f'   → {fp}', flush=True)
    if promote:
        import shutil
        act = P.shard_plan_path(spec_key)
        bak = os.path.join(P.IDX, f'plan_{spec_key}_pilot.json')
        if os.path.isfile(act) and not os.path.isfile(bak):
            shutil.copy2(act, bak)
        shutil.copy2(fp, act)
        print(f'   ✅ 已提升为活动分片清单 {act}（试点清单备份 → {os.path.basename(bak)}）', flush=True)
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acct')
    ap.add_argument('--anchor')
    ap.add_argument('--spec')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--no-scan', action='store_true')
    ap.add_argument('--max-shards', type=int, default=0, help='每家族最多取 N 个 parcel（按门槛像元数降序）')
    ap.add_argument('--promote', action='store_true', help='把批量清单提升为活动清单 plan_<spec>.json')
    a = ap.parse_args()
    P.ensure_all(list(S.SPECS))
    specs = list(S.SPECS) if a.all else ([a.spec] if a.spec else [])
    if not specs:
        raise SystemExit('给 --spec 或 --all（可选 --no-scan 离线干跑）')
    ee = None
    if not a.no_scan:
        if not (a.acct and a.anchor):
            raise SystemExit('预扫需要 --acct 与 --anchor')
        ee = boot(a.acct, a.anchor)
    for sp in specs:
        try:
            run_spec(ee, sp, a.no_scan, a.max_shards, a.promote)
        except Exception as e:
            print(f'[{sp}] ❌ {str(e)[:200]}')


if __name__ == '__main__':
    main()
