# -*- coding: utf-8 -*-
"""
audit_latest.py — 审计每个文件"重建内容"是否为会话记录中的最新版本。

判据：把 Read 工具结果按行号解析——
  * 完整读（起始行=1 且行号连续）才可作为"文件当时真实内容"；
  * 若某次完整读的时间 **晚于** 该文件最后一次 Write/Edit → 该 Read 内容更新
    （说明文件在此之后被脚本就地修改过，如 _path_migrate.py / _append_doc_updates.py）。
输出：_复原记录/latest_audit.json + 控制台摘要。
"""
import sqlite3, json, os, re, collections

DB = r"C:\Users\Administrator\.zcode\cli\db\db.sqlite"
ROOT = r"F:\地理所\论文\中国土地覆盖数据_2017-2024"
PREFIXES = [
    r"Z:\Mywork\论文\中国土地覆盖数据\\",
    r"Z:\Mywork\论文\BaiduSyncdisk\中国土地覆盖数据\\",
    "Z:/Mywork/论文/中国土地覆盖数据/",
    "Z:/Mywork/论文/BaiduSyncdisk/中国土地覆盖数据/",
    r"F:\r7_prod\\", "F:/r7_prod/",
]

def norm(p):
    if not p: return None
    p = p.replace("\\", "/")
    for pre in PREFIXES:
        q = pre.replace("\\", "/")
        if p.startswith(q):
            rel = p[len(q):]
            if q.lower().startswith("f:/r7_prod"):
                return "生产_F盘自包含/" + rel
            return rel
    if "中国土地覆盖数据/" in p:
        return p.split("中国土地覆盖数据/", 1)[-1]
    return None

LINE = re.compile(r"^(\d+)\t(.*)$")
TRUNC = re.compile(r"(more lines|truncat|use offset|offset=|lines? \d+-\d+ of|\[\.\.\.\])", re.I)

def parse_read(text):
    """返回 (content, first, last, n, contiguous, truncated)"""
    lines = text.split("\n")
    nums, body = [], []
    for ln in lines:
        m = LINE.match(ln)
        if m:
            nums.append(int(m.group(1))); body.append(m.group(2))
    if not nums:
        return None
    contiguous = (nums == list(range(nums[0], nums[0] + len(nums))))
    # 截断提示通常出现在非行号行里，或行尾被截的标记
    truncated = bool(TRUNC.search(text.rsplit("\n", 3)[-1])) or bool(TRUNC.search(text[-400:]))
    return "\n".join(body) + "\n", nums[0], nums[-1], len(nums), contiguous, truncated

def main():
    c = sqlite3.connect(DB); cur = c.cursor()
    sess = [r[0] for r in cur.execute(
        "SELECT id FROM session WHERE directory LIKE '%中国土地覆盖%'").fetchall()]
    qm = ",".join("?" * len(sess))
    rows = cur.execute(
        f"SELECT session_id, data, time_created FROM part WHERE session_id IN ({qm}) "
        f"AND data LIKE '%\"type\":\"tool\"%' ORDER BY time_created", sess).fetchall()

    ev = collections.defaultdict(lambda: {"mod": 0, "reads": []})
    for sid, d, t in rows:
        try: j = json.loads(d)
        except Exception: continue
        tool = j.get("tool"); st = j.get("state") or {}
        inp = st.get("input") or {}
        if tool in ("Write", "Edit"):
            if st.get("status") == "error": continue
            rel = norm(inp.get("file_path") or "")
            if rel: ev[rel]["mod"] = max(ev[rel]["mod"], t)
        elif tool == "Read":
            if st.get("status") == "error": continue
            rel = norm(inp.get("file_path") or "")
            out = st.get("output")
            if not rel or not isinstance(out, str) or not out.strip(): continue
            pr = parse_read(out)
            if pr:
                content, first, last, n, cont, trunc = pr
                ev[rel]["reads"].append({"t": t, "content": content,
                                         "first": first, "last": last,
                                         "n": n, "contig": cont, "trunc": trunc})

    report = []
    for rel, v in sorted(ev.items()):
        dst = os.path.join(ROOT, rel.replace("/", os.sep))
        have = None
        if os.path.isfile(dst):
            have = open(dst, encoding="utf-8", errors="replace").read()
        full = [r for r in v["reads"] if r["first"] == 1 and r["contig"]
                and r["n"] >= 3 and not r["trunc"]]
        newer = [r for r in full if r["t"] > v["mod"]]
        newest = None
        for r in sorted(newer, key=lambda r: -r["t"]):
            nc, hc = r["content"], have
            if hc is None:            # 本地缺文件 → 用最新的
                newest = r; break
            if nc.rstrip() == hc.rstrip():
                continue              # 仅空白/换行差异
            if hc.startswith(nc[:max(200, len(nc) - 50)]):
                continue              # 只是旧内容的截断前缀 → 部分读，忽略
            newest = r; break
        status = "OK"
        if have is None:
            status = "MISSING"
        elif newest is not None:
            status = "STALE"
        report.append({
            "rel": rel, "status": status, "mod_time": v["mod"],
            "n_full_reads": len(full), "newest_read_time": newest["t"] if newest else None,
            "have_len": len(have) if have is not None else -1,
            "new_len": len(newest["content"]) if newest else -1,
        })

    outs = os.path.join(ROOT, "_复原记录", "latest_audit.json")
    with open(outs, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)

    stale = [r for r in report if r["status"] == "STALE"]
    missing = [r for r in report if r["status"] == "MISSING"]
    print(f"审计文件 {len(report)}  最新={len(report)-len(stale)-len(missing)}  "
          f"过期(STALE)={len(stale)}  缺失={len(missing)}")
    print("\n== STALE（会话中有更晚的完整 Read）==")
    for r in sorted(stale, key=lambda x: -(x['new_len']-x['have_len'])):
        d = r["new_len"] - r["have_len"]
        print(f"  {d:+9d} B  {r['rel']}")
    if missing:
        print("\n== MISSING（会话提到但本地没有）==")
        for r in missing: print("  ", r["rel"])

    # 把更新版本落盘到 _复原源_LatestRead 供比对
    out_dir = os.path.join(ROOT, "_复原记录", "最新Read版")
    for rel, v in ev.items():
        full = [r for r in v["reads"] if r["first"] == 1 and r["contig"]
                and r["n"] >= 3 and not r["trunc"]]
        newer = [r for r in full if r["t"] > v["mod"]]
        if not newer: continue
        dst0 = os.path.join(ROOT, rel.replace("/", os.sep))
        have0 = open(dst0, encoding="utf-8", errors="replace").read() if os.path.isfile(dst0) else None
        cand = None
        for r in sorted(newer, key=lambda r: -r["t"]):
            nc = r["content"]
            if have0 is not None:
                if nc.rstrip() == have0.rstrip(): continue
                if have0.startswith(nc[:max(200, len(nc) - 50)]): continue
            cand = r; break
        if cand is None: continue
        newest = cand
        fp = os.path.join(out_dir, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        with open(fp, "w", encoding="utf-8", newline="") as f:
            f.write(newest["content"])
    print(f"\n候选最新版已导出: {out_dir}")

if __name__ == "__main__":
    main()
