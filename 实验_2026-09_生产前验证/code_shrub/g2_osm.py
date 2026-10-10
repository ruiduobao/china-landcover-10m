# -*- coding: utf-8 -*-
"""g2_osm.py — OSM 灌丛要素抓取（Overpass API；natural=scrub / landuse=scrub / natural=heath）

* 输入：四省 OSM area（按 name:en / ISO3166-2）
* 规则源：OSM 标签语义——natural=scrub（灌木地）、landuse=scrub（灌木农地）、natural=heath（石楠灌丛）
* 门槛：way 取 center；node 取坐标；分省分标签独立请求，失败换镜像重试
* 输出：data/shrub/osm_scrub.json（要素 + 省 + 标签）
* 用法：python g2_osm.py
"""
import json, os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
OUTD = os.path.join(WORK, 'data', 'shrub')
PROV = {'宁夏': 'Ningxia Hui Autonomous Region', '四川': 'Sichuan', '黑龙江': 'Heilongjiang', '福建': 'Fujian'}
MIRRORS = ['https://overpass-api.de/api/interpreter', 'https://overpass.kumi.systems/api/interpreter',
           'https://overpass.osm.ch/api/interpreter']

def emit(m): print(time.strftime('[%m-%d %H:%M:%S] ')+str(m), flush=True)

def query(prov_en):
    return f'''[out:json][timeout:180];
area["name:en"="{prov_en}"][admin_level=4]->.a;
(
  way["natural"="scrub"](area.a);
  node["natural"="scrub"](area.a);
  way["landuse"="scrub"](area.a);
  node["landuse"="scrub"](area.a);
  way["natural"="heath"](area.a);
  node["natural"="heath"](area.a);
);
out center;'''

def main():
    import requests
    os.makedirs(OUTD, exist_ok=True)
    prox = {'http': 'socks5h://127.0.0.1:7890', 'https': 'socks5h://127.0.0.1:7890'}
    out = {}
    for prov, en in PROV.items():
        got = None
        for url in MIRRORS:
            try:
                r = requests.post(url, data={'data': query(en)},
                                  headers={'User-Agent': 'china-lc-shrub-research/1.0 (contact: local)'},
                                  proxies=prox, timeout=300)
                if r.status_code == 200:
                    got = r.json()
                    emit('%s @%s：要素 %d' % (prov, url.split('/')[2], len(got.get('elements', []))))
                    break
                emit('  %s HTTP %d %s' % (url.split('/')[2], r.status_code, r.text[:80]))
            except Exception as e:
                emit('  %s 失败 %s' % (url.split('/')[2], str(e)[:80]))
            time.sleep(5)
        if not got:
            emit('%s 全部镜像失败' % prov); continue
        rows = []
        for e in got.get('elements', []):
            t = e.get('tags', {}) or {}
            c = e.get('center') or {}
            lon = c.get('lon', e.get('lon')); lat = c.get('lat', e.get('lat'))
            if lon is None or lat is None:
                continue
            rows.append(dict(prov=prov, osm_id=e.get('id'), osm_type=e.get('type'),
                             lon=lon, lat=lat,
                             natural=t.get('natural', ''), landuse=t.get('landuse', ''),
                             name=t.get('name', ''), tags=json.dumps(t, ensure_ascii=False)[:300]))
        out[prov] = rows
    json.dump(out, open(os.path.join(OUTD, 'osm_scrub.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    for p, v in out.items():
        emit('%s OSM 要素 %d' % (p, len(v)))

if __name__ == '__main__': main()
