    col_cache = {}
    ok_n = fail_n = 0
    t0 = time.time()
    for cid in todo:
        fp = os.path.join(OUT_DIR, f'chunkr7_{cid:04d}.parquet')
        g = idx_df[idx_df.chunk_id == cid]
        rows = g[['row_id', 'lon', 'lat']].to_dict('records')
        year = int(g.year.iloc[0])
        done_f = False
        for attempt in range(3):
            try:
                fc = ee.FeatureCollection([
                    ee.Feature(ee.Geometry.Point([float(r['lon']), float(r['lat'])]),
                               {P['prop']: int(r['row_id'])})
                    for r in rows])
                if year not in col_cache:
                    col_cache[year] = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
                                       .filterDate(f'{year}-01-01', f'{year+1}-01-01'))
                ic = col_cache[year].filterBounds(fc)
                emb = ic.mosaic().select(FEATS)
                samp = emb.sampleRegions(collection=fc, properties=[P['prop']],
                                         scale=10, tileScale=P['ts'])
                url = samp.getDownloadURL(filetype='csv', selectors=[P['prop']] + FEATS)
                out = None
                for dtry in range(P['nretry']):     # 下载阶段重试（代理空闲超时防护）
                    try:
                        resp = requests.get(url, proxies=PC.PROXY, timeout=1200)
                        resp.raise_for_status()
                        out = pd.read_csv(io.BytesIO(resp.content))
                        break
                    except Exception as de:
                        print(f'[{acct}] chunk{cid} dl-retry{dtry+1} ERR: '
                              f'{str(de)[:90]}', flush=True)
                        time.sleep(20 + 10 * dtry)
                if out is None:
                    raise RuntimeError('download retries exhausted')
                out = out.rename(columns={P['prop']: 'row_id'})
                out['emb_year'] = year
                out['chunk_id'] = cid
                out.to_parquet(fp, index=False)
                ok_n += 1
                el = time.time() - t0
                print(f'[{acct}] chunk{cid:04d} y{year}: {len(out)} 行 '
                      f'({ok_n}/{len(todo)}, {el/3600:.1f}h)', flush=True)
                done_f = True
                break
            except Exception as e:
                print(f'[{acct}] chunk{cid} attempt{attempt+1} ERR: {str(e)[:110]}', flush=True)
                time.sleep(20 + attempt * 30)
        if not done_f:
