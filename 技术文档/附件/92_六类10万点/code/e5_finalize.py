# -*- coding: utf-8 -*-
"""e5_finalize.py — 清洗 + 分层 + 配额 + 空间独立性 + 交付集"""
import os,sys,time,json
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
OUT=r'Z:/nb_national_2026/forest6_10w'
AEF=['A%02d'%i for i in range(64)]
ECO={'东北':['黑龙江省','吉林省','辽宁省'],'华北':['北京市','天津市','河北省','山西省','内蒙古自治区'],
 '西北':['陕西省','甘肃省','宁夏回族自治区','新疆维吾尔自治区'],'青藏':['青海省','西藏自治区'],
 '西南':['四川省','重庆市','云南省','贵州省'],'华中':['河南省','湖北省','湖南省','江西省','安徽省'],
 '华东':['江苏省','浙江省','上海市','山东省','福建省','台湾省'],
 '华南':['广东省','广西壮族自治区','海南省','香港特别行政区','澳门特别行政区']}
P2E={p:e for e,ps in ECO.items() for p in ps}
def emit(m): print(time.strftime('[%H:%M:%S] ')+str(m),flush=True)
def main():
    c=pd.read_parquet(os.path.join(OUT,'all_candidates.parquet'))
    a=pd.read_parquet(os.path.join(OUT,'aef_all.parquet'))
    d=c.merge(a,on='uid',how='inner')
    emit('合并 %d'%len(d))
    # --- 清洗 ---
    rej={}
    n0=len(d)
    m=d[AEF].notna().all(axis=1); rej['AEF缺失']=int((~m).sum()); d=d[m]
    m=(d.lon>=73)&(d.lon<=136)&(d.lat>=17)&(d.lat<=54); rej['坐标越界']=int((~m).sum()); d=d[m]
    d['k']=d.lon.round(6).astype(str)+'_'+d.lat.round(6).astype(str)
    dup=d.duplicated('k'); rej['重复坐标']=int(dup.sum()); d=d[~dup].drop(columns='k')
    emit('清洗后 %d（剔除 %s）'%(len(d),rej))
    # --- 空间独立性字段 ---
    d['c01']=(np.floor(d.lon/0.01).astype('int64')*100000+np.floor(d.lat/0.01).astype('int64'))
    d['c05']=(np.floor(d.lon/0.05).astype('int64')*100000+np.floor(d.lat/0.05).astype('int64'))
    for nm,col in (('spatial_cluster_size','c01'),('cluster_size_5km','c05')):
        d[nm]=d.groupby(['cls6',col])[col].transform('size')
    d['is_neighbor']=(d.src=='neighbor').astype('int8')
    # --- 省份补全（用 CTAmap 或缺则空）---
    d['prov']=d.prov.fillna(''); d['eco']=d.eco.fillna('')
    d['eco']=np.where(d.eco=='',d.prov.map(P2E).fillna(''),d.eco)
    # --- 三层分级 ---
    def tier(r):
        if r.quality_tier=='A' and r.spatial_cluster_size==1 and r.cluster_size_5km<=5 and r.is_neighbor==0:
            return 'L-A'
        if r.quality_tier in ('A','B'):
            return 'L-B'
        return 'L-C'
    d['layer']=d.apply(tier,axis=1)
    emit('分层：'+str(d.layer.value_counts().to_dict()))
    # --- 每类上限 100,000（优先 L-A → L-B → L-C）---
    order={'L-A':0,'L-B':1,'L-C':2}
    out=[]
    for cls,g in d.groupby('cls6'):
        g=g.assign(_o=g.layer.map(order)).sort_values('_o')
        # 省内限 15%
        cap=int(100000*0.15); keep=[]
        for pv,gg in g.groupby('prov'):
            if pv=='': keep.append(gg); continue
            keep.append(gg.head(min(len(gg),cap)))
        g=pd.concat(keep).sort_values('_o').head(100000).drop(columns='_o')
        out.append(g)
    d=pd.concat(out,ignore_index=True)
    emit('上限截断后 %d'%len(d))
    print(d.groupby(['cls6','layer']).size().unstack(fill_value=0).to_string())
    d.to_parquet(os.path.join(OUT,'forest6_10w.parquet'),index=False)
    st={'n':int(len(d)),'by_class':{str(k):int(v) for k,v in d.cls6.value_counts().sort_index().items()},
        'by_layer':{k:int(v) for k,v in d.layer.value_counts().items()},
        'class_x_layer':{str(k):{kk:int(vv) for kk,vv in v.items()} for k,v in
                         d.groupby('cls6').layer.value_counts().unstack(fill_value=0).to_dict('index').items()},
        'rejected':rej,'n_before':n0,
        'by_eco':{k:{kk:int(vv) for kk,vv in v.items()} for k,v in
                  d.groupby('eco').cls6.value_counts().unstack(fill_value=0).to_dict('index').items()}}
    json.dump(st,open(os.path.join(OUT,'e5_stats.json'),'w',encoding='utf-8'),ensure_ascii=False,indent=1)
    emit('写 forest6_10w.parquet')
if __name__=='__main__': main()
