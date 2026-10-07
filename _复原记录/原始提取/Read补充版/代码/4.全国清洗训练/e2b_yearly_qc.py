    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates('row_id')
    E = df[FEATS].to_numpy(np.float32)
    return df.row_id.to_numpy(np.int64), E

def main():
    t0 = time.time()
    idx = pd.read_parquet(os.path.join(BASE, 'chunks_index_r7.parquet'))
    val = pd.read_parquet(os.path.join(WORKR, 'r7_train_validity.parquet'))
    n_idx = len(idx)
    print(f'索引 point-years: {n_idx:,}', flush=True)

    # ---------- 第一遍：逐年有效性/norm/类心余弦 ----------
    per_year = {}
    for y in YEARS:
        r = load_year(y)
        if r is None:
            print(f'y{y}: 无嵌入!', flush=True)
            per_year[y] = None
            continue
        rid, E = r
        ok = np.isfinite(E).all(1) & (np.linalg.norm(E, axis=1) > 1e-6)
        nrm = np.linalg.norm(E, axis=1)
        En = E / (nrm[:, None] + 1e-12)
        meta = idx.set_index('row_id').loc[rid]
        cls = meta.class_new.to_numpy(int)
        cos = np.full(len(E), np.nan, dtype=np.float32)
        for c in np.unique(cls):
            m = (cls == c) & ok
            if m.sum() >= 30:
                mu = E[m].mean(0)
                mu = mu / (np.linalg.norm(mu) + 1e-12)
                cos[m] = (En[m] @ mu).astype(np.float32)
        d = pd.DataFrame({'row_id': rid, 'year': y, 'emb_valid': ok,
                          'norm': nrm.astype(np.float32), 'cos_center': cos})
        d['class_new'] = cls
        per_year[y] = d
        del E, En
        print(f'y{y}: {len(d):,} 行, 有效 {ok.sum():,}', flush=True)

    # ---------- 第二遍：相邻年 delta（按 row_id 对齐） ----------
    dv = {y: pd.Series(np.nan, index=per_year[y].row_id) if per_year[y] is not None else None
          for y in YEARS}
    Emb = {}
    for y in YEARS:
        r = load_year(y)
        Emb[y] = r  # (rid, E) 暂驻，成对用完即删
    for y in YEARS[1:]:
        if per_year[y] is None or per_year[y-1] is None or Emb[y] is None or Emb[y-1] is None:
            continue
        rid_a, E_a = Emb[y - 1]
        rid_b, E_b = Emb[y]
        common, ia, ib = np.intersect1d(rid_a, rid_b, return_indices=True)
        d = np.linalg.norm(E_b[ib] - E_a[ia], axis=1)
        dv[y] = pd.Series(d, index=common)          # 记在较晚年上
        print(f'delta {y-1}→{y}: {len(common):,} 对', flush=True)
    Emb.clear()

    for y in YEARS:
        if per_year[y] is None:
            continue
        per_year[y]['delta_prev'] = per_year[y].row_id.map(dv[y]).to_numpy(np.float32) \
            if dv[y] is not None else np.nan
        nxt = dv[y + 1] if (y + 1) in dv and dv[y + 1] is not None else None
        per_year[y]['delta_next'] = per_year[y].row_id.map(nxt).to_numpy(np.float32) \
            if nxt is not None else np.nan

    # ---------- z-score（逐类×年，median/MAD） ----------
    for col in ['delta_prev', 'delta_next']:
        for y in YEARS:
            d = per_year[y]
            if d is None or col not in d:
                continue
            x = d[col].to_numpy()
            z = np.full(len(d), np.nan, dtype=np.float32)
