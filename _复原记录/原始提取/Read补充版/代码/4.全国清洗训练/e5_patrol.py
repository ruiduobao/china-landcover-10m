    import pilot_common as PC
    import ee
    PC.set_proxy()
    ee.Initialize(project=PC.ANCHOR_PID[acct])
    FEATS = [f'A{i:02d}' for i in range(64)]
    g = idx_df[idx_df.chunk_id == cid]
    rows = g[['row_id', 'lon', 'lat']].to_dict('records')
    year = int(g.year.iloc[0])
    fp = os.path.join(PARTS, f'chunk_{cid:04d}.parquet')
    for attempt in range(3):
      try:
        ee.Initialize(project=PC.ANCHOR_PID[acct])
        fc = ee.FeatureCollection([
                ee.Feature(ee.Geometry.Point([float(r['lon']), float(r['lat'])]),
                           {'row_id': int(r['row_id'])})
                for r in rows])
            ic = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
                  .filterDate(f'{year}-01-01', f'{year+1}-01-01').filterBounds(fc))
            emb = ic.mosaic().select(FEATS)
            samp = emb.sampleRegions(collection=fc, properties=['row_id'],
                                     scale=10, tileScale=8)
            url = samp.getDownloadURL(filetype='csv', selectors=['row_id'] + FEATS)
            resp = requests.get(url, proxies=PC.PROXY, timeout=2400)
            resp.raise_for_status()
            out = pd.read_csv(io.BytesIO(resp.content))
            out['emb_year'] = year
            out['chunk_id'] = cid
            out.to_parquet(fp, index=False)
            return True
      except Exception as e:
          print(f'  retry {acct} chunk{cid} attempt{attempt+1}: {str(e)[:90]}', flush=True)
          time.sleep(30)
    return False

def workers_alive():
    try:
        r = subprocess.run(['powershell', '-Command',
                            '(Get-Process python -ErrorAction SilentlyContinue | '
                            'Where-Object {$_.StartTime -gt (Get-Date).AddHours(-30)} | '
                            'Measure-Object).Count'],
                           capture_output=True, text=True, timeout=60)
        return int(r.stdout.strip())
    except Exception:
        return -1

def relaunch_workers():
    for acct in ACCOUNTS:
        subprocess.Popen([sys.executable, '-u', WORKER, acct],
                         stdout=open(os.path.join(LOG_DIR, f'w_{acct}.log'), 'a'),
                         stderr=subprocess.STDOUT, cwd='Z:/Mywork/论文/中国土地覆盖数据')
        print(f'  [重启] {acct} 正向 worker', flush=True)
        time.sleep(5)

def main():
    total = total_chunks()
