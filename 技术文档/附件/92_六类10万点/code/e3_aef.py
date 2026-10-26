# -*- coding: utf-8 -*-
"""e3_aef.py — 多账号并行取 AEF64"""
import os,sys,json,glob,time,subprocess
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
OUT=r'Z:/nb_national_2026/forest6_10w'
ACC=r'F:/地理所/论文/博士论文/中期/实验区域/02_中间数据/gee_accounts'
PART=os.path.join(OUT,'aef_parts'); os.makedirs(PART,exist_ok=True)
CHILD_PATH=os.path.join(OUT,'_child_aef.py')
CHILD='''# -*- coding: utf-8 -*-
import os,sys,json,warnings
import pandas as pd
acct,pid,inp,outp,ACC,PROXY=sys.argv[1:7]
os.environ['HOME']=os.path.join(ACC,acct); os.environ['USERPROFILE']=os.environ['HOME']
os.environ['HTTP_PROXY']=PROXY; os.environ['HTTPS_PROXY']=PROXY; os.environ['ALL_PROXY']=PROXY
os.environ['NO_PROXY']='localhost,127.0.0.1'
import ee
AEF=['A%02d'%i for i in range(64)]
try:
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always'); ee.Initialize(project=pid)
    d=pd.read_parquet(inp)
    emb=ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL').filterDate('2023-01-01','2024-01-01').mosaic()
    out=[]
    for i in range(0,len(d),4000):
        c=d.iloc[i:i+4000]
        pts=ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(x),float(y)]),{'k':int(k)}) for k,x,y in zip(c.uid,c.lon,c.lat)])
        fc=emb.sampleRegions(collection=pts,scale=10,geometries=False,tileScale=4)
        for f in fc.getInfo()['features']:
            pr=f['properties']; row={'uid':pr['k']}
            for b in AEF: row[b]=pr.get(b)
            out.append(row)
    pd.DataFrame(out).to_parquet(outp,index=False)
    print(json.dumps({'ok':True,'n':len(out)}))
except Exception as e:
    print(json.dumps({'ok':False,'err':str(e)[:200]}))
'''
def emit(m): print(time.strftime('[%H:%M:%S] ')+str(m),flush=True)
def main():
    open(CHILD_PATH,'w',encoding='utf-8').write(CHILD)
    P=pd.read_parquet(os.path.join(OUT,'_pending_aef.parquet'))
    P['gx']=(P.lon//2).astype(int); P['gy']=(P.lat//2).astype(int)
    P['grp']=P.gx.astype(str)+'_'+P.gy.astype(str)
    groups=sorted(P.grp.unique())
    emit('待取 %d 点 / %d 个 2° 格'%(len(P),len(groups)))
    H=json.load(open(os.path.join(OUT,'health2.json'),encoding='utf-8'))['healthy']
    emit('可用账号 %d'%len(H))
    PROXY='socks5h://127.0.0.1:7890'
    queue=[]
    for i,g in enumerate(groups):
        outp=os.path.join(PART,'out_%d.parquet'%i)
        if os.path.exists(outp): continue
        inp=os.path.join(PART,'in_%d.parquet'%i)
        if not os.path.exists(inp):
            P[P.grp==g][['uid','lon','lat']].to_parquet(inp,index=False)
        queue.append((i,inp,outp))
    emit('待处理分片 %d'%len(queue))
    if not queue:
        emit('全部分片已完成'); return
    from concurrent.futures import ThreadPoolExecutor, as_completed
    def run(item,slot):
        i,inp,outp=item; h=H[slot%len(H)]
        cmd=[sys.executable,CHILD_PATH,h['acct'],h['proj'],inp,outp,ACC,PROXY]
        try:
            r=subprocess.run(cmd,capture_output=True,text=True,timeout=1200)
            return i,True,(r.stdout or r.stderr).strip()[-150:]
        except Exception as e: return i,False,str(e)[:100]
    done=0; t0=time.time()
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs=[ex.submit(run,it,k) for k,it in enumerate(queue)]
        for f in as_completed(futs):
            i,ok,msg=f.result(); done+=1
            if done%15==0: emit('  %d/%d %.1fmin %s %s'%(done,len(queue),(time.time()-t0)/60,'OK' if ok else 'FAIL',msg[:70]))
    parts=[pd.read_parquet(f) for f in glob.glob(os.path.join(PART,'out_*.parquet'))]
    if parts:
        d=pd.concat(parts,ignore_index=True).drop_duplicates('uid')
        d.to_parquet(os.path.join(OUT,'aef_all.parquet'),index=False)
        emit('合并 %d 点 × %d 列'%(len(d),len(d.columns)))
    emit('DONE %.1f min'%((time.time()-t0)/60))
if __name__=='__main__': main()
