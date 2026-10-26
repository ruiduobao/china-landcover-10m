# -*- coding: utf-8 -*-
"""q1_delivery_slim.py — 交付 2023 GPKG 属性瘦表导出（供下游统计复用）

* 输入：数据/单类样本的单独处理_2026.10.12/<类>/2023/*.gpkg（23 类，与 s1_build_pool_v2.py 的 SRC 完全同文件）
* 输出：reports/quality_review_2026-10-15/samples/delivery2023_slim.parquet
        列 = cl, point_id, lon, lat, tier, src, ct_prov, ct_km, has_tier, has_src  （缺字段以空串计）
* 用法：python q1_delivery_slim.py
* 备注：只读属性表（read_geometry=False），不动交付本体；字段名逐类自适应（pid/point_id/uid/cid）。
"""
import os, sys, time
import pandas as pd
import pyogrio

sys.stdout.reconfigure(encoding='utf-8')
DEL = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12'
OUT = r'F:/lc_work/prod2nd_2023/reports/quality_review_2026-10-15/samples'

# 与 s1_build_pool_v2.py 的 SRC 一致（类目录名 -> 文件, 类码）
SRC = [
    ('草本旱地', '草本旱地/2023/草本旱地_2023_WGS84.gpkg', 1),
    ('灌溉耕地', '灌溉耕地/2023/灌溉耕地_2023_WGS84.gpkg', 3),
    ('草地', '草地/2023/草地_2023_WGS84.gpkg', 11),
    ('地衣苔藓', '地衣苔藓/2023/地衣苔藓_2023_WGS84.gpkg', 12),
    ('稀疏植被', '稀疏植被/2023/稀疏植被_2023_WGS84.gpkg', 13),
    ('裸地', '裸地/2023/裸地_2023_WGS84.gpkg', 22),
    ('湖河滩地', '湖河滩地/2023/湖河滩地_2023_WGS84.gpkg', 16),
    ('盐渍湿地', '盐渍湿地/2023/盐渍湿地_2023_WGS84.gpkg', 17),
    ('红树林', '红树林/扩充_v1/红树林_全量_10718_2023_WGS84.gpkg', 18),
    ('海岸盐沼', '海岸盐沼/2023/海岸盐沼_2023_WGS84.gpkg', 19),
    ('潮滩', '潮滩/2023/潮滩_2023_WGS84.gpkg', 20),
    ('水体', '水体/2023/水体_2023_WGS84.gpkg', 23),
    ('不透水面', '不透水面/2023/不透水面_2023_WGS84.gpkg', 21),
    ('永久冰雪', '永久冰雪/2023/永久冰雪_2023_WGS84.gpkg', 24),
    ('林地_常绿阔叶林', '林地_常绿阔叶林/2023/林地_常绿阔叶林_2023_WGS84.gpkg', 4),
    ('林地_落叶阔叶林', '林地_落叶阔叶林/2023/林地_落叶阔叶林_2023_WGS84.gpkg', 5),
    ('林地_常绿针叶林', '林地_常绿针叶林/2023/林地_常绿针叶林_2023_WGS84.gpkg', 6),
    ('林地_落叶针叶林', '林地_落叶针叶林/2023/林地_落叶针叶林_2023_WGS84.gpkg', 7),
    ('木本沼泽', '木本沼泽/2023/木本沼泽_2023_WGS84.gpkg', 14),
    ('草本沼泽', '草本沼泽/2023/草本沼泽_2023_WGS84.gpkg', 15),
    ('常绿灌木', '常绿灌木/2023/常绿灌木_2023_WGS84.gpkg', 9),
    ('落叶灌木', '落叶灌木/2023/落叶灌木_2023_WGS84.gpkg', 10),
    ('乔灌园地', '乔灌园地/2023/乔灌园地v7_2023_WGS84.gpkg', 2),
]
IDCOLS = ['pid', 'point_id', 'uid', 'cid']


def main():
    os.makedirs(OUT, exist_ok=True)
    parts, meta = [], []
    for name, rel, code in SRC:
        f = os.path.join(DEL, rel)
        info = pyogrio.read_info(f)
        fields = list(info['fields'])
        idc = next((c for c in IDCOLS if c in fields), None)
        cols = [c for c in [idc, 'lon', 'lat', 'tier', 'src', 'ct_prov', 'ct_km'] if c in fields]
        need_geom = ('lon' not in fields) or ('lat' not in fields)
        df = pyogrio.read_dataframe(f, read_geometry=need_geom, columns=cols, use_arrow=True)
        df = df.rename(columns={idc: 'point_id'})
        if need_geom:
            df['lon'] = df.geometry.x.round(6)
            df['lat'] = df.geometry.y.round(6)
            df = df.drop(columns=['geometry'])
        for c in ['lon', 'lat', 'tier', 'src', 'ct_prov', 'ct_km']:
            if c not in df.columns:
                df[c] = None
        df = df[['point_id', 'lon', 'lat', 'tier', 'src', 'ct_prov', 'ct_km']].copy()
        df['point_id'] = df['point_id'].astype(str)
        df['cl'] = code
        df['has_tier'] = 'tier' in fields
        df['has_src'] = 'src' in fields
        parts.append(df[['cl', 'point_id', 'lon', 'lat', 'tier', 'src', 'ct_prov', 'ct_km', 'has_tier', 'has_src']])
        meta.append(dict(cls_dir=name, cl=code, file=rel, n=len(df), idcol=idc,
                         has_tier=('tier' in fields), has_src=('src' in fields),
                         has_ctprov=('ct_prov' in fields),
                         n_ctprov_null=int(df['ct_prov'].isna().sum()) if 'ct_prov' in fields else -1))
        print('%-14s cl=%-2d n=%-7d idcol=%-8s tier=%s src=%s ct_prov=%s' % (
            name, code, len(df), idc, 'tier' in fields, 'src' in fields, 'ct_prov' in fields), flush=True)
        del df
    allp = pd.concat(parts, ignore_index=True)
    allp.to_parquet(os.path.join(OUT, 'delivery2023_slim.parquet'), index=False)
    pd.DataFrame(meta).to_csv(os.path.join(OUT, 'delivery2023_field_meta.csv'), index=False, encoding='utf-8-sig')
    print('总 %d 行 -> delivery2023_slim.parquet' % len(allp))


if __name__ == '__main__':
    main()
