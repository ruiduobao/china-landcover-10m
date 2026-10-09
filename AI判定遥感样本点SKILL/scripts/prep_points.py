# -*- coding: utf-8 -*-
"""prep_points.py — 点位输入质检 + 判读表单骨架 + 盲判清单

* 输入：--points points.csv（必含 id/lon/lat；可含先验类、分层、来源等）
* 规则源：references/判读协议.md —— 盲判批次不得携带先验标签/分层（防锚定）
* 门槛：id 唯一、坐标可解析、经纬度落在给定 bbox 内（默认中国 73–136E/18–54N，--bbox 可改）
* 输出：<outdir>/points_clean.csv（规范化点位）
        <outdir>/form_skeleton.csv（point_id,结论,实际类别,置信,证据 —— 空白待填）
        <outdir>/blind_list.csv（盲判顺序：仅 point_id + 序号）
        <outdir>/prep_report.md（质检摘要：数量/重复/越界/先验类分布）
* 用法：python prep_points.py --points raw.csv --outdir ./run1
        python prep_points.py --points raw.csv --outdir ./run1 --prior-col 先验类 --strata-col w
"""
import argparse
import collections
import csv
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
VERDICTS = '结论（正确/错误/无法判读）'


def read_any(fp):
    """支持 CSV / XLSX / SHP（GIS 用户常给点 shapefile）→ 统一成 dict 列表（含 lon/lat）。"""
    low = fp.lower()
    if low.endswith(('.shp', '.gpkg', '.geojson')):
        import geopandas as gpd
        g = gpd.read_file(fp)
        if g.crs is None:
            print('警告：%s 无坐标系声明，按 WGS84 处理' % fp)
        elif str(g.crs).upper() not in ('EPSG:4326',):
            print('提示：%s 为 %s，转换为 WGS84' % (fp, g.crs))
            g = g.to_crs(4326)
        pts = g.geometry.representative_point() if hasattr(g.geometry, 'representative_point') else g.geometry
        d = g.drop(columns=g.geometry.name).copy()
        d['lon'] = pts.x.values
        d['lat'] = pts.y.values
        return d.to_dict('records')
    if low.endswith(('.xlsx', '.xls')):
        import pandas as pd
        d = pd.read_excel(fp)
        return d.to_dict('records')
    return list(csv.DictReader(open(fp, encoding='utf-8-sig')))


def main():
    ap = argparse.ArgumentParser(description='点位输入质检与表单骨架')
    ap.add_argument('--points', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--id-col', default='point_id')
    ap.add_argument('--lon-col', default='lon')
    ap.add_argument('--lat-col', default='lat')
    ap.add_argument('--prior-col', default=None, help='先验标签列（如 教师类）；只进报告与合并，不进盲判批次')
    ap.add_argument('--strata-col', default=None, help='分层列（区域/来源等），用于分层统计')
    ap.add_argument('--bbox', default='73,18,136,54', help='lon0,lat0,lon1,lat1；越界仅告警不删除')
    ap.add_argument('--verdict-col', default=None, help='结论列名（默认“%s”）' % VERDICTS)
    ap.add_argument('--metrics', default=None, help='s2_metrics.csv：把物候指标写成“指标提示”列（判读时交叉验证）')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    rows = read_any(a.points)
    if not rows:
        raise SystemExit('空表')
    need = [a.id_col, a.lon_col, a.lat_col]
    missing = [c for c in need if c not in rows[0]]
    if missing:
        raise SystemExit('缺少列：%s（现有列：%s）' % (missing, list(rows[0].keys())))
    lon0, lat0, lon1, lat1 = [float(x) for x in a.bbox.split(',')]
    seen, dup, out, bad = set(), [], [], []
    for r in rows:
        pid = str(r[a.id_col]).strip()
        if pid in seen:
            dup.append(pid)
        seen.add(pid)
        try:
            lon, lat = float(r[a.lon_col]), float(r[a.lat_col])
        except Exception:
            bad.append(pid)
            continue
        if not (lon0 <= lon <= lon1 and lat0 <= lat <= lat1):
            out.append((pid, lon, lat))
    with open(os.path.join(a.outdir, 'points_clean.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    vcol = a.verdict_col or VERDICTS
    hint = {}
    if a.metrics and os.path.exists(a.metrics):
        for m in csv.DictReader(open(a.metrics, encoding='utf-8-sig')):
            parts = []
            if m.get('n'):
                parts.append('S2 %s 期' % m['n'])
            if m.get('ndvi_amp'):
                parts.append('NDVI振幅%s' % m['ndvi_amp'])
            if m.get('ndvi_peak_date'):
                parts.append('峰值%s' % m['ndvi_peak_date'][5:])
            fv = m.get('flood_evidence', '')
            if fv:
                parts.append({'yes': '有淹水证据', 'weak': '弱淹水信号', 'no': '无淹水'}.get(fv, fv))
            if m.get('ndwi_min'):
                parts.append('NDWI_min%s' % m['ndwi_min'])
            hint[m['id']] = '；'.join(parts)
    head = [a.id_col, vcol, '实际类别', '置信（高/中/低）', '证据（影像特征+日期+辅助指标）']
    if hint:
        head.append('指标提示（S2 物候，判读时交叉验证；与影像冲突以影像为准）')
    with open(os.path.join(a.outdir, 'form_skeleton.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(head)
        for r in rows:
            row = [r[a.id_col], '', '', '', '']
            if hint:
                row.append(hint.get(str(r[a.id_col]), ''))
            w.writerow(row)
    with open(os.path.join(a.outdir, 'blind_list.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['seq', a.id_col])
        for i, r in enumerate(rows):
            w.writerow([i + 1, r[a.id_col]])
    L = ['# 点位输入质检摘要', '', '- 输入：`%s`（%d 行）' % (a.points, len(rows))]
    L += ['- 重复 id：%d %s' % (len(dup), dup[:5]), '- 坐标不可解析：%d %s' % (len(bad), bad[:5]),
          '- 越界告警：%d（bbox=%s）' % (len(out), a.bbox)]
    if a.prior_col:
        c = collections.Counter(str(r.get(a.prior_col, '')) for r in rows)
        L += ['', '## 先验类分布（仅供合并与统计，不进盲判批次）', '']
        L += ['| 先验类 | n |', '|---|---|'] + ['| %s | %d |' % (k or '(空)', v) for k, v in c.most_common()]
    if a.strata_col:
        c = collections.Counter(str(r.get(a.strata_col, '')) for r in rows)
        L += ['', '## 分层分布', ''] + ['- %s: %d' % (k or '(空)', v) for k, v in c.most_common()]
    L += ['', '## 产物', '', '- `points_clean.csv`、`form_skeleton.csv`（判读表单骨架）、`blind_list.csv`（盲判顺序）',
          '- **盲判纪律**：判读批次只给 `point_id`，不得展示 `%s` 等先验列' % (a.prior_col or '先验标签')]
    open(os.path.join(a.outdir, 'prep_report.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('点位 %d；重复 %d；不可解析 %d；越界 %d → %s' % (len(rows), len(dup), len(bad), len(out), a.outdir))


if __name__ == '__main__':
    main()
