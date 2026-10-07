# -*- coding: utf-8 -*-
"""s7_carm.py — C 臂：B + 针对性特征（doc37 §6.2，只做代表区 w2 秦岭 + w5 三江湿地）
特征（17 个，2023 生长季 4-10 月）：
  S2 SR harmonized：B5/B6/B7/B8/B8A 生长季中位数 + B5/B6/B7/B8A p90、NDRE 中位
  纹理：B8A 中位量化后 GLCM(3x3) contrast/entropy/var
  水淹：JRC GSW1_4 recurrence/seasonality；地形：SRTM elevation/slope
阶段：sample（批导采样资产→存在性幂等→完整性抽检→共享）→ submit（C_s{7,11,23} 追加进 jobs.json）
判据：瓶颈类（w2=森林组成 04-08；w5=湿地 14-20）F1 ≥ B 臂 +3pp 且不损害他类（s8 汇总裁定）。
"""
import os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v31_common as VC

WIN = VC.cfg('windows.json')['windows']
FEATX = ['s2_B5_med', 's2_B6_med', 's2_B7_med', 's2_B8_med', 's2_B8A_med',
         's2_B5_p90', 's2_B6_p90', 's2_B7_p90', 's2_B8A_p90', 'ndre_med',
         'glcm_contrast', 'glcm_entropy', 'glcm_var',
         'recurrence', 'seasonality', 'elevation', 'slope']
REGIONS = {'w2': dict(bottleneck=list(range(4, 9)), note='森林组成（秦岭）'),
           'w5': dict(bottleneck=list(range(14, 21)), note='湿地细分（三江）')}


def feature_image(w):
    import ee
    x0, y0, x1, y1 = WIN[w]['bbox']
    geom = ee.Geometry.Rectangle([x0, y0, x1, y1])
    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterDate('2023-01-01', '2024-01-01')
          .filter(ee.Filter.calendarRange(4, 10, 'month'))
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
          .filterBounds(geom))
    med = s2.median()
    p90 = s2.reduce(ee.Reducer.percentile([90]))
    parts = [med.select(b).rename('s2_%s_med' % b) for b in ['B5', 'B6', 'B7', 'B8', 'B8A']]
    parts += [p90.select('%s_p90' % b).rename('s2_%s_p90' % b) for b in ['B5', 'B6', 'B7', 'B8A']]
    b8, b5 = med.select('B8'), med.select('B5')
    parts.append(b8.subtract(b5).divide(b8.add(b5)).rename('ndre_med'))
    glcm = (med.select('B8A').divide(10000).multiply(255).toByte().rename('b8q')
            .glcmTexture(3))
    parts += [glcm.select('b8q_contrast').rename('glcm_contrast'),
              glcm.select('b8q_ent').rename('glcm_entropy'),   # GEE 实际波段名是 _ent
              glcm.select('b8q_var').rename('glcm_var')]
    # JRC GSW 在陆地/无观测区是掩膜 → 必须 unmask(0)，否则 sampleRegions 会丢 99% 点位（已实测 399/400）
    parts.append(ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
                 .select(['recurrence', 'seasonality']).unmask(0))
    dem = ee.Image('USGS/SRTMGL1_003').select('elevation')
    parts += [dem, ee.Terrain.slope(dem)]
    return ee.Image.cat(parts).toFloat()


COMBOS = [('w2', 'train'), ('w2', 'eval'), ('w5', 'train'), ('w5', 'eval')]  # 固定顺序 → 固定宿主


def sample_hosts(pool):
    """采样宿主账号：uploader 打头（w2 两个资产已在它名下），其余用 worker 分摊并行。"""
    hosts = [pool['uploader']]
    for a in pool['workers']:
        if a not in hosts:
            hosts.append(a)
        if len(hosts) >= 4:
            break
    return hosts


def host_of(pool, w, kind):
    hosts = sample_hosts(pool)
    try:
        idx = COMBOS.index((w, kind))
    except ValueError:
        idx = 0
    return hosts[idx % len(hosts)]


def assets_of(w, pool):
    """返回 {train_src, eval_src, train_out, eval_out}；输出资产按 host_of 定位（可跨账号）。"""
    up_root = 'projects/%s/assets' % VC.pid_of(pool['uploader'])
    if w == 'w5':
        train_src = VC.reg_get('train_w5') or '%s/v31s_w5_merged' % up_root
    else:
        train_src = 'projects/%s/assets/t4ca_exp/smp_%s_merged' % (VC.pid_of(pool['owner']), w)
    outs = {}
    for kind in ('train', 'eval'):
        key = 'carm_%s_%s' % (w, kind)
        host = host_of(pool, w, kind)
        outs['%s_out' % kind] = VC.reg_get(key) or (
            'projects/%s/assets/v31f_%s_%s' % (VC.pid_of(host), w, kind))
    return dict(train_src=train_src,
                eval_src=VC.reg_get('eval_%s' % w) or '%s/v31e_%s_merged' % (up_root, w),
                train_out=outs['train_out'], eval_out=outs['eval_out'])


def inflight_of(acct, tag):
    """uploader 名下是否有该 tag 前缀的在飞 v31c_ 任务（防重复提交）。"""
    try:
        s, pid = VC.sess(acct)
        r = s.get('https://earthengine.googleapis.com/v1/projects/%s/operations' % pid,
                  params={'pageSize': 100}, headers={'X-Goog-User-Project': pid}, timeout=60)
        r.raise_for_status()
        n = 0
        for o in r.json().get('operations', []):
            md = o.get('metadata', {})
            d = md.get('description', '')
            if d.startswith('v31c_') and tag in d and md.get('state') in ('PENDING', 'RUNNING'):
                n += 1
        return n
    except Exception as e:
        VC.emit('在飞查询失败（按无在飞处理）: %s' % str(e)[:80])
        return 0


def stage_sample(pool):
    """批导 train/eval 采样资产（多账号并行，每轮可投多个）；就绪后抽检完整性并共享。"""
    up_root = 'projects/%s/assets' % VC.pid_of(pool['uploader'])
    have_up = VC.list_assets(pool['uploader'], up_root) or set()
    shared = VC.jload(os.path.join(VC.DATA, 'shared_done.json'), {})
    all_cache = {}
    for w, kind in COMBOS:
        if w not in REGIONS:
            continue
        host = host_of(pool, w, kind)
        A = assets_of(w, pool)
        out = A['train_out'] if kind == 'train' else A['eval_out']
        name = out.split('/')[-1]
        # ① 已注册 / 已在宿主名下 / ② 在 uploader 名下（早期提交）
        if VC.reg_get('carm_%s_%s' % (w, kind)):
            continue
        host_root = 'projects/%s/assets' % VC.pid_of(host)
        hhave = all_cache.setdefault(host, VC.list_assets(host, host_root) or set())
        if name in hhave:
            VC.reg_set('carm_%s_%s' % (w, kind), out)
            VC.emit('%s %s 资产已在 %s 名下，登记' % (w, kind, host))
            continue
        if name in have_up:
            VC.reg_set('carm_%s_%s' % (w, kind), '%s/%s' % (up_root, name))
            VC.emit('%s %s 资产在 uploader 名下，登记' % (w, kind))
            continue
        if inflight_of(host, '%s_%s' % (w, kind)):
            VC.emit('%s %s 在 %s 上有采样任务在飞，等待' % (w, kind, host))
            continue
        n_q, err = VC.qdepth(host)
        if n_q is None or n_q > 5:
            VC.emit('  %s 队列=%s（%s），跳过' % (host, n_q, err))
            continue
        ee, _ = VC.ctx(host)
        img = feature_image(w)
        fc = ee.FeatureCollection(A['train_src'] if kind == 'train' else A['eval_src'])
        # geometries=True 必须：sampleRegions 默认丢弃几何 → 导出报 null geometry（已踩）
        sampled = img.sampleRegions(collection=fc, scale=10, tileScale=4, geometries=True)
        desc = 'v31c_%s_%s_%s' % (w, kind, time.strftime('%m%d%H%M%S'))
        t = ee.batch.Export.table.toAsset(collection=sampled, description=desc, assetId=out)
        t.start()
        VC.emit('采样提交 %s %s @%s → %s' % (w, kind, host, t.id))
        time.sleep(3)
    # 全部就绪判定
    ready = all(VC.reg_get('carm_%s_%s' % (w, k)) for w, k in COMBOS if w in REGIONS)
    if not ready:
        return
    emails = ['user:' + pool['emails'][a] for a in pool['workers'] if a in pool['emails']]
    for w, kind in COMBOS:
        if w not in REGIONS:
            continue
        full = VC.reg_get('carm_%s_%s' % (w, kind))
        holder = next((a for a in [pool['owner'], pool['uploader']] + pool['workers']
                       if VC.pid_of(a) in full), pool['uploader'])
        if not check_props(holder, full):
            raise SystemExit('NEED_MANUAL: %s 采样特征不完整（宿主 %s）' % (full, holder))
        if not shared.get(full):
            ee2, _ = VC.ctx(holder)
            ee2.data.setAssetAcl(full, {'readers': emails})
            shared[full] = time.strftime('%Y-%m-%d %H:%M')
            VC.jsave(shared, os.path.join(VC.DATA, 'shared_done.json'))
            VC.emit('共享 %s ✓（宿主 %s）' % (full.split('/')[-1], holder))
    st = VC.state()
    st.setdefault('carm_sampled', {})
    for w, _k in COMBOS:
        st['carm_sampled'][w] = time.strftime('%Y-%m-%d %H:%M')
    VC.save_state(st)
    VC.emit('C臂采样全部就绪 ✅（w2+w5）')


def check_props(acct, asset):
    """抽检前 3 个特征的完整性（存在且非空）。"""
    import io
    import pandas as pd
    try:
        e, _ = VC.ctx(acct)
        fc = e.FeatureCollection(asset).limit(3)
        url = fc.getDownloadURL(filetype='csv')
        import requests
        import prod_conf as PC
        r = requests.get(url, proxies=PC.PROXY, timeout=120)
        df = pd.read_csv(io.StringIO(r.text))
        miss = [f for f in FEATX if f not in df.columns or df[f].isna().all()]
        VC.emit('抽检 %s: 缺失特征 %s' % (asset.split('/')[-1], miss or '无'))
        return not miss
    except Exception as e:
        VC.emit('抽检失败 %s: %s' % (asset, str(e)[:100]))
        return False


def stage_submit(pool):
    st = VC.state()
    done = st.get('carm_sampled', {})
    jobs = VC.jload(VC.JOBS_FP, [])
    have = {j['job_id'] for j in jobs}
    n_new = 0
    for w in REGIONS:
        if w not in done:
            continue
        A = assets_of(w, pool)
        for s in VC.cfg('frozen_params.json')['seeds']['B']:
            jid = '%s_C_s%02d' % (w, s)
            if jid in have:
                continue
            jobs.append(dict(job_id=jid, window=w, arm='C', seed=s, label_mode='v31',
                             train_asset=A['train_out'], eval_asset=A['eval_out'],
                             feats=VC.FEATS + FEATX, group='C'))
            n_new += 1
    if n_new:
        VC.jsave(jobs, VC.JOBS_FP)
    VC.emit('s7 submit: 追加 C 任务 %d（s2 轮会提交）' % n_new)


def main():
    pool = VC.jload(os.path.join(VC.CFG, 'accounts_pool.json'))
    if not pool:
        raise SystemExit('NEED_MANUAL: accounts_pool.json 缺失')
    act = 'sample' if 'sample' in sys.argv else ('submit' if 'submit' in sys.argv else 'sample')
    if act == 'sample':
        # w5 依赖其评估/训练资产先就绪（s1/s6 完成）
        if not (VC.reg_get('eval_w5') or VC.asset_exists(
                pool['uploader'], 'projects/%s/assets/v31e_w5_merged'
                % VC.pid_of(pool['uploader']))):
            VC.emit('w5 评估资产未就绪，本轮只处理 w2')
            REGIONS.pop('w5', None)
        stage_sample(pool)
    else:
        stage_submit(pool)


if __name__ == '__main__':
    main()
