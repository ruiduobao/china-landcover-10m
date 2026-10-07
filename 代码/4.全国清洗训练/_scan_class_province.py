# -*- coding: utf-8 -*-
"""按"类×省"合理性规则扫描 r7 样本库的错位分布（临时诊断脚本）"""
import os, sys
import numpy as np
import pandas as pd

PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
sys.path.insert(0, os.path.join(PROJ, '代码', '5.样本重建'))
sys.path.insert(0, os.path.join(PROJ, '代码', '0.本地流水线'))
from s1_geom import province_of
from lc_conf import CLASSES, class_name

COASTAL = {'辽宁省', '河北省', '天津市', '山东省', '江苏省', '上海市', '浙江省',
           '福建省', '广东省', '广西壮族自治区', '海南省', '台湾省', '香港特别行政区',
           '澳门特别行政区'}
COASTAL_S = COASTAL - {'河北省', '天津市', '山东省', '江苏省', '上海市'}   # 红树林
INLAND_SALINE = {'新疆维吾尔自治区', '青海省', '西藏自治区', '内蒙古自治区', '甘肃省',
                 '宁夏回族自治区', '吉林省', '黑龙江省'}
HIGH_MTN = {'西藏自治区', '青海省', '新疆维吾尔自治区', '四川省', '甘肃省', '云南省',
            '陕西省', '山西省', '内蒙古自治区', '河北省', '吉林省', '黑龙江省',
            '宁夏回族自治区', '辽宁省', '北京市', '重庆市', '河南省', '湖北省'}
LARCH = {'黑龙江省', '吉林省', '辽宁省', '内蒙古自治区', '河北省', '山西省', '陕西省',
         '甘肃省', '新疆维吾尔自治区', '四川省', '云南省', '西藏自治区', '青海省',
         '宁夏回族自治区', '湖北省', '重庆市', '河南省', '北京市', '山东省'}

RULES = {   # 类 -> 允许的省集合（None=全国合理，不检查）
    140: HIGH_MTN,
    184: COASTAL_S,
    185: COASTAL,
    186: COASTAL,
    183: COASTAL | INLAND_SALINE,
    220: {'西藏自治区', '青海省', '新疆维吾尔自治区', '四川省', '甘肃省', '云南省'},
    81: LARCH,
    82: LARCH,
}


def main():
    t = pd.read_parquet(os.path.join(PROJ, '数据/本地处理/样本重建/r7_train.parquet'),
                        columns=['lon', 'lat', 'class_new', 'src'])
    rare = os.path.join(PROJ, '数据/本地处理/全国清洗训练/稀有类补样/r7_rare_samples.parquet')
    r = pd.read_parquet(rare)
    s = pd.concat([t, r[t.columns]], ignore_index=True)
    s['prov'] = province_of(s.lon.to_numpy(), s.lat.to_numpy())
    s['src'] = s.src.astype(str)
    s['is_rare_topup'] = s.src.str.startswith(('fcs10_2023_rare', 'gmw_v3', 'cw_2020'))

    print('类  名称                总点数   违规数  违规占比   违规省份 top5')
    print('-' * 100)
    total_flag = 0
    for c in sorted(RULES):
        g = s[s.class_new == c]
        allow = RULES[c]
        bad = g[~g.prov.isin(allow)]
        if len(g) == 0:
            print(f'{c:<4}{class_name(c):<18}{0:>7}'); continue
        top = bad.prov.value_counts().head(5).to_dict()
        top_s = ' '.join(f'{k.replace("省","").replace("自治区","").replace("维吾尔","").replace("回族","").replace("特别行政区","")}={v}' for k, v in top.items())
        print(f'{c:<4}{class_name(c):<18}{len(g):>7}  {len(bad):>6}   {len(bad)/len(g):>6.1%}   {top_s}')
        total_flag += len(bad)
        if len(bad):
            b = bad[bad.is_rare_topup]
            print(f'     └ 其中本轮稀有类补样带入: {len(b)}  （基线已有: {len(bad)-len(b)}）')
            print(f'     └ 违规点经度范围: {bad.lon.min():.1f}–{bad.lon.max():.1f}, 纬度 {bad.lat.min():.1f}–{bad.lat.max():.1f}')
    print('-' * 100)
    print('规则内类别违规合计:', total_flag)


if __name__ == '__main__':
    main()
