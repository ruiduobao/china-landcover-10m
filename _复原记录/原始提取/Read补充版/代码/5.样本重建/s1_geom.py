# -*- coding: utf-8 -*-
"""
s1_geom.py — r1 流水线唯一几何模块（国界/省界/GUB 城乡边界）
* 全流水线只允许从这里加载边界与做点面判断，消灭旧链 8 份重复实现。
* 全部向量化（shapely 2 contains_xy + STRtree），支持数组批量。
"""
import os, json
import numpy as np
import shapely
from shapely import contains_xy
from shapely.strtree import STRtree

import s0_conf as C

# ---------------- 国界（DataV 省 json → 多边形集合） ----------------
_polys = None
_tree = None

def china_polys():
    global _polys, _tree
    if _polys is None:
        d = json.load(open(C.BOUND_JSON, encoding='utf-8'))
        ps = []
        for f in d['features']:
            sg = shapely.geometry.shape(f['geometry'])
            ps += list(sg.geoms) if sg.geom_type == 'MultiPolygon' else [sg]
        _polys = ps
        _tree = STRtree(ps)
    return _polys

def china_contains(lon, lat):
    """向量化：点数组是否在中国国界内（任一省多边形包含即算）"""
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    polys = china_polys()
    pts = shapely.points(lon, lat)
    keep = np.zeros(len(lon), dtype=bool)
    # bbox 预筛后逐多边形 contains_xy（多边形 ~34，开销可忽略）
    for p in polys:
        bx = (lon >= p.bounds[0]) & (lon <= p.bounds[2]) & \
             (lat >= p.bounds[1]) & (lat <= p.bounds[3])
        if bx.any():
            keep[bx] |= contains_xy(p, lon[bx], lat[bx])
    return keep

# ---------------- GUB 2020 全球城市边界（城乡分割依据） ----------------
_gub = None
_gub_tree = None

def gub_geoms():
    global _gub, _gub_tree
    if _gub is None:
        import geopandas as gpd
        gub = gpd.read_file(C.GUB_SHP)
        _gub = np.array(list(gub.geometry), dtype=object)
        _gub_tree = STRtree(_gub)
    return _gub

def gub_contains(lon, lat):
    """向量化：点是否在任一 GUB 城市多边形内（STRtree 候选 + contains 精判）"""
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    geoms = gub_geoms()
    pts = shapely.points(lon, lat)
    tree = STRtree(geoms) if _gub_tree is None else _gub_tree
    keep = np.zeros(len(lon), dtype=bool)
    pairs = tree.query(pts)          # 2×k: [point_idx, tree_idx]
    if len(pairs):
        pi, gi = pairs[0], pairs[1]
        hit = shapely.contains(geoms[gi], pts[pi])
        keep[pi[hit]] = True
    return keep

# ---------------- 省界归属（DataV，省名 → 多边形） ----------------
_prov_names = None
_prov_geoms = None
_prov_tree = None

def provinces():
    global _prov_names, _prov_geoms, _prov_tree
    if _prov_names is None:
        d = json.load(open(C.BOUND_JSON, encoding='utf-8'))
        names, gs = [], []
        for f in d['features']:
            nm = (f.get('properties') or {}).get('name') or str(f.get('properties'))
            sg = shapely.geometry.shape(f['geometry'])
            gs.append(sg)
            names.append(nm)
        _prov_names = names; _prov_geoms = np.array(gs, dtype=object); _prov_tree = STRtree(gs)
    return _prov_names, _prov_geoms, _prov_tree

def province_of(lon, lat):
    """向量化省归属：返回省名数组（未匹配 → '未匹配'）"""
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    names, geoms, tree = provinces()
    pts = shapely.points(lon, lat)
    out = np.full(len(lon), '未匹配', dtype=object)
    pairs = tree.query(pts)
    if len(pairs):
        pi, gi = pairs[0], pairs[1]
        hit = shapely.contains(geoms[gi], pts[pi])
        for j, i in zip(pi[hit], gi[hit]):
            out[j] = names[i]
    return out

