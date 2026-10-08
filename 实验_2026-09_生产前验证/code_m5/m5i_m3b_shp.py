# -*- coding: utf-8 -*-
"""m5i_m3b_shp.py — M3-B 最小复核清单 → 点 shapefile（WGS84 + GCJ-02 双版本）

* 输入：F:/lc_work/v31_exp/data/m3/m3b_最小复核清单.csv（210 点，m5g_review_min.py 生成）
* 规则源：字段名转 ASCII 且 ≤10 字符（DBF 限制）；文本值实测最大 83 字节，无需截断；
  坐标双版本——WGS84（对齐 Esri/Google 影像底图）与 GCJ-02（对齐 DataV 省界/高德底图，
  我国境内两坐标系偏移 300–500 m，这是本项目已记录的口径陷阱）；GCJ-02 版仍按惯例写
  EPSG:4326 元数据（坐标值本身已偏移），用文件名区分。
* 门槛：点数 = 输入行数；两版仅坐标不同、属性逐字一致；字段名全部 ≤10 且唯一。
* 输出：F:/lc_work/v31_exp/data/m3/m3b_shp/
        m3b_复核点_WGS84.{shp,shx,dbf,prj,cpg}
        m3b_复核点_GCJ02.{shp,shx,dbf,prj,cpg}
        字段对照.txt（shp 字段 → 原列名 + 说明）
* 用法：python m5i_m3b_shp.py [--in <csv>] [--outdir <dir>]
幂等：直接覆盖输出目录内同名文件。
"""
import math
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import geopandas as gpd

M3D = r'F:/lc_work/v31_exp/data/m3'
SRC = os.path.join(M3D, 'm3b_最小复核清单.csv')
OUTD = os.path.join(M3D, 'm3b_shp')

# 原列名 → shp 字段名（ASCII，≤10 字符；数值列原样保留）
RENAME = {
    'point_id': 'point_id',
    '组别': 'grp',
    '复核理由': 'reason',
    'AI判读': 'ai_label',
    'AI置信': 'ai_conf',
    'AI备注': 'ai_note',
    'arm': 'arm',
    'w': 'win',
    '教师类(v31)': 'teacher',
    'Q1_是否灌丛(是/否/无法判读)': 'q1_shrub',
    'Q2_若否主要地物(24类名)': 'q2_other',
    'Q3_证据(影像/日期)': 'q3_evi',
    '判读员': 'judge',
    '复核员': 'checker',
    '备注': 'memo',
    'chip': 'chip',
    'ch_mean': 'ch_mean',
    'tc2000': 'tc2000',
    'lossyear': 'lossyear',
    'nd_djf': 'nd_djf',
    'nd_mam': 'nd_mam',
    'nd_jja': 'nd_jja',
    'nd_son': 'nd_son',
    'ndvi_amp': 'ndvi_amp',
    'ndvi_mean': 'ndvi_mean',
    'ndvi_winter': 'ndvi_win',
    'ndvi_summer': 'ndvi_sum',
    '指标提示': 'hint',
    'dry': 'dry',
}
NUM = ['lon', 'lat', 'ch_mean', 'tc2000', 'lossyear', 'nd_djf', 'nd_mam',
       'nd_jja', 'nd_son', 'ndvi_amp', 'ndvi_mean', 'ndvi_winter', 'ndvi_summer']
NOTE = {
    'grp': '组别：G1_不确定/G2_低置信是/G3_干旱A/G4_随机否',
    'reason': '复核理由', 'ai_label': 'AI 预判读：是/否/无法判读',
    'ai_conf': 'AI 置信：高/中/低', 'ai_note': 'AI 备注（主要地物判断依据）',
    'arm': 'A_fcs10_shrub=争议层 / B_control_base=邻域底座对照',
    'win': '窗：w1松嫩/w2秦岭/w3南方丘陵/w4西北干旱',
    'teacher': '训练池教师类（v31 码）', 'q1_shrub': '【待填】是否灌丛', 'q2_other': '【待填】若非灌丛的主要地物',
    'chip': '切片路径（448x224，左真彩右假彩）', 'ch_mean': 'GEDI 冠层高度中位（m）',
    'tc2000': 'Hansen 2000 年树覆盖（%）', 'lossyear': 'Hansen 损失年（20=2020）',
    'nd_djf': 'NDVI 冬（DJF）', 'nd_mam': 'NDVI 春（MAM）', 'nd_jja': 'NDVI 夏（JJA）', 'nd_son': 'NDVI 秋（SON）',
    'ndvi_amp': 'NDVI 振幅', 'ndvi_mean': 'NDVI 均值', 'ndvi_win': 'NDVI 冬季中位', 'ndvi_sum': 'NDVI 夏季中位',
    'hint': '机械提示', 'dry': '干旱省（新青藏甘宁蒙）标记',
}


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


# ---------- WGS84 → GCJ-02 ----------
_A, _EE = 6378245.0, 0.00669342162296594323


def _out_of_china(lon, lat):
    return not (72.004 <= lon <= 137.8347 and 0.8293 <= lat <= 55.8271)


def _t_lat(x, y):
    r = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
    r += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    r += (20.0 * math.sin(y * math.pi) + 40.0 * math.sin(y / 3.0 * math.pi)) * 2.0 / 3.0
    r += (160.0 * math.sin(y / 12.0 * math.pi) + 320.0 * math.sin(y * math.pi / 30.0)) * 2.0 / 3.0
    return r


def _t_lon(x, y):
    r = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
    r += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    r += (20.0 * math.sin(x * math.pi) + 40.0 * math.sin(x / 3.0 * math.pi)) * 2.0 / 3.0
    r += (150.0 * math.sin(x / 12.0 * math.pi) + 300.0 * math.sin(x / 30.0 * math.pi)) * 2.0 / 3.0
    return r


def wgs84_to_gcj02(lon, lat):
    if _out_of_china(lon, lat):
        return lon, lat
    dlat, dlon = _t_lat(lon - 105.0, lat - 35.0), _t_lon(lon - 105.0, lat - 35.0)
    rad = lat / 180.0 * math.pi
    m = 1 - _EE * math.sin(rad) ** 2
    sm = math.sqrt(m)
    dlat = (dlat * 180.0) / ((_A * (1 - _EE)) / (m * sm) * math.pi)
    dlon = (dlon * 180.0) / (_A / sm * math.cos(rad) * math.pi)
    return lon + dlon, lat + dlat


def build(fp_in):
    # 表头去重（防列名重复导致 pandas 改名成 xxx.1 后映射失败）：同名列只留第一次出现
    import csv as _csv
    with open(fp_in, encoding='utf-8-sig', newline='') as f:
        head = next(_csv.reader(f))
    seen = {}
    keep = []
    for i, c in enumerate(head):
        if c in seen:
            emit('  警告：列名重复 %r（第 %d 列）→ 丢弃' % (c, i + 1))
            continue
        seen[c] = i
        keep.append(i)
    df = pd.read_csv(fp_in, dtype=str, usecols=keep, encoding='utf-8-sig').fillna('')
    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    attrs = df.drop(columns=['lon', 'lat']).rename(columns=RENAME)
    order = [RENAME[c] for c in df.columns if c not in ('lon', 'lat')]
    attrs = attrs[order]
    return df, attrs


def write_shp(attrs, lon, lat, fp):
    g = gpd.GeoDataFrame(attrs.copy(), geometry=gpd.points_from_xy(lon, lat), crs='EPSG:4326')
    g.to_file(fp, driver='ESRI Shapefile', engine='pyogrio', encoding='UTF-8')


def main():
    a = sys.argv[1:]
    fp_in = a[a.index('--in') + 1] if '--in' in a else SRC
    outd = a[a.index('--outdir') + 1] if '--outdir' in a else OUTD
    os.makedirs(outd, exist_ok=True)
    df, attrs = build(fp_in)
    emit('读入 %s：%d 点，%d 字段' % (fp_in, len(df), len(attrs.columns)))
    lon, lat = df['lon'].to_numpy(), df['lat'].to_numpy()
    fp1 = os.path.join(outd, 'm3b_复核点_WGS84.shp')
    write_shp(attrs, lon, lat, fp1)
    emit('写出 %s' % fp1)
    gl = [wgs84_to_gcj02(x, y) for x, y in zip(lon, lat)]
    fp2 = os.path.join(outd, 'm3b_复核点_GCJ02.shp')
    write_shp(attrs, [p[0] for p in gl], [p[1] for p in gl], fp2)
    emit('写出 %s' % fp2)
    # 字段对照
    fpm = os.path.join(outd, '字段对照.txt')
    with open(fpm, 'w', encoding='utf-8') as f:
        f.write('M3-B 最小复核清单 shp 字段对照（生成 %s）\n' % time.strftime('%Y-%m-%d %H:%M'))
        f.write('源：%s\n\n' % fp_in)
        f.write('%-12s %-24s %s\n' % ('shp 字段', '原列名', '说明'))
        for c in df.columns:
            if c in RENAME:
                f.write('%-12s %-24s %s\n' % (RENAME[c], c, NOTE.get(RENAME[c], '')))
        f.write('\n坐标：\n')
        f.write('  m3b_复核点_WGS84.shp   —— 对齐 S2/Esri/Google 影像底图（GEE 原生坐标）\n')
        f.write('  m3b_复核点_GCJ02.shp   —— 对齐 DataV 省界（数据/边界/china_100000_full.json）与高德底图；\n')
        f.write('                            元数据仍写 EPSG:4326（坐标值已偏移，惯例做法），用文件名区分。\n')
    emit('写出 %s' % fpm)
    # 复核：读回
    import fiona
    for fp in (fp1, fp2):
        with fiona.open(fp, encoding='UTF-8') as s:
            names = s.schema['properties'].keys()
            over = [n for n in names if len(n) > 10]
            emit('  读回 %s：%d 要素  CRS=%s  超长字段=%s' % (
                os.path.basename(fp), len(s), s.crs, over or '无'))
    emit('完成')


if __name__ == '__main__':
    main()
