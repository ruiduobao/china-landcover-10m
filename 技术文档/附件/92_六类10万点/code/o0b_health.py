# -*- coding: utf-8 -*-
"""o0b_health.py — 用 acct_proj.json 的真实项目ID做健康检查（每账号试前2个项目，并发）"""
import os,sys,json,subprocess,time
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.stdout.reconfigure(encoding='utf-8')
ACC=r'F:/地理所/论文/博士论文/中期/实验区域/02_中间数据/gee_accounts'
PROJ=r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
sys.path.insert(0,os.path.join(PROJ,'代码','8.全国生产'))
import prod_conf as C
M=json.load(open('acct_proj.json',encoding='utf-8'))
def probe_one(acct,pid):
    code=("import os,warnings,json\n"
      "os.environ['HOME']=r'%s';os.environ['USERPROFILE']=r'%s'\n"
      "os.environ['HTTP_PROXY']=r'%s';os.environ['HTTPS_PROXY']=r'%s';os.environ['ALL_PROXY']=r'%s'\n"
      "os.environ['NO_PROXY']='localhost,127.0.0.1'\n"
      "import ee\n"
      "r={'ok':False}\n"
      "try:\n"
      "    with warnings.catch_warnings(record=True) as ws:\n"
      "        warnings.simplefilter('always'); ee.Initialize(project='%s')\n"
      "    m=' '.join(str(x.message) for x in ws)\n"
      "    ee.Number(1).getInfo(); r['ok']=True\n"
      "    if 'restricted' in m or 'quota' in m: r['ok']=False; r['reason']='restricted'\n"
      "except Exception as e: r['reason']=str(e)[:100]\n"
      "print(json.dumps(r))\n")%(os.path.join(ACC,acct),os.path.join(ACC,acct),C.PROXY_URL,C.PROXY_URL,C.PROXY_URL,pid)
    try:
        r=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=90)
        ln=[l for l in r.stdout.strip().split('\n') if l.startswith('{')]
        return (acct,pid,json.loads(ln[-1]) if ln else {'ok':False,'reason':'noout'})
    except Exception as e: return (acct,pid,{'ok':False,'reason':str(e)[:60]})
def probe(a):
    for pid in M[a][:2]:
        r=probe_one(a,pid)
        if r[2].get('ok'): return a,pid,True,r[2]
    return a,M[a][0],False,r[2]
if __name__=='__main__':
    A=sorted(M.keys()); print('待检账号 %d'%len(A),flush=True)
    res={}; t0=time.time()
    with ThreadPoolExecutor(max_workers=16) as ex:
        for f in as_completed([ex.submit(probe,a) for a in A]):
            a,pid,ok,r=f.result(); res[a]={'ok':ok,'proj':pid,'reason':r.get('reason')}
            if ok: print('  OK %-16s %s'%(a,pid),flush=True)
    ok=[{'acct':a,'proj':res[a]['proj']} for a in res if res[a]['ok']]
    print('\n健康 %d / %d  (%.1f min)'%(len(ok),len(A),(time.time()-t0)/60))
    json.dump({'healthy':ok,'detail':res},open('health2.json','w',encoding='utf-8'),ensure_ascii=False,indent=1)
