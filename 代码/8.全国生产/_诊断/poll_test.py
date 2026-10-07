import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prod_conf as C
C.apply_account_env('zixen8v8'); import ee
ee.Initialize(project=C.ACCOUNTS['zixen8v8']['proj'])
TID='PTPK3RTMLYGO2FLUP4MWWCSE'
for i in range(40):
    time.sleep(45)
    s=ee.data.getTaskStatus(TID)[0]
    st=s.get('state')
    print('[%s] %s' % (time.strftime('%H:%M:%S'), st), flush=True)
    if st in ('COMPLETED','FAILED','CANCELLED'):
        print('err:', s.get('error_message'))
        break
try:
    info=ee.data.getAsset('projects/braided-horizon-508210-a5/assets/lcf_a01/lcs08_X001_2022')
    print('资产:', json.dumps({k:info.get(k) for k in ('type','sizeBytes','updateTime')}, ensure_ascii=False))
except Exception as e:
    print('资产未生成:', str(e)[:90])
