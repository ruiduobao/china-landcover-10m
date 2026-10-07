# -*- coding: utf-8 -*-
"""从 Read 工具结果中恢复被读取过的项目文件（cat -n 格式去行号）。"""
import sqlite3, json, os, re, collections

DB = r"C:\Users\Administrator\.zcode\cli\db\db.sqlite"
OUT_W = r"F:\地理所\论文\中国土地覆盖数据_2017-2024\_复原源"          # Write 版
OUT_R = r"F:\地理所\论文\中国土地覆盖数据_2017-2024\_复原源_Read补充"   # Read 版
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
    low = p.lower()
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

def strip_linenos(text):
    lines = text.split("\n")
    out = []
    truncated = False
    for ln in lines:
        m = LINE.match(ln)
        if m:
            out.append(m.group(2))
        else:
            s = ln.strip()
            if not s:
                continue
            if s.startswith("(") and ("offset" in s or "truncat" in s.lower() or "lines" in s.lower()):
                truncated = True
                continue
            truncated = True   # 非行号行视为异常/截断
    return "\n".join(out) + "\n", truncated

def main():
    c = sqlite3.connect(DB); cur = c.cursor()
    rows = cur.execute(
        "SELECT session_id, data, time_created FROM part "
        "WHERE data LIKE '%\"type\":\"tool\"%' AND data LIKE '%中国土地覆盖%' "
        "ORDER BY time_created").fetchall()
    best = {}   # rel -> (nlines, content, truncated, t, sid)
    for sid, d, t in rows:
        try: j = json.loads(d)
        except Exception: continue
        if j.get("tool") != "Read": continue
        st = j.get("state") or {}
        if st.get("status") == "error": continue
        inp = st.get("input") or {}
        rel = norm(inp.get("file_path") or "")
        if not rel: continue
        out = st.get("output")
        if not isinstance(out, str) or not out.strip(): continue
        content, trunc = strip_linenos(out)
        n = content.count("\n")
        if n < 3: continue
        if rel not in best or n > best[rel][0]:
            best[rel] = (n, content, trunc, t, sid)

    os.makedirs(OUT_R, exist_ok=True)
    written = set()
    for root, _dirs, files in os.walk(OUT_W):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), OUT_W).replace("\\", "/")
            written.add(rel)

    n_new = n_fill = 0
    report = []
    for rel, (n, content, trunc, t, sid) in sorted(best.items()):
        fp = os.path.join(OUT_R, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        with open(fp, "w", encoding="utf-8", newline="") as f:
            f.write(content)
        in_w = rel in written
        if in_w:
            # 比较大小
            wp = os.path.join(OUT_W, rel.replace("/", os.sep))
            wlen = os.path.getsize(wp) if os.path.exists(wp) else -1
            rlen = len(content.encode("utf-8"))
            report.append((rel, "FILL", n, wlen, rlen, trunc))
            n_fill += 1
        else:
            report.append((rel, "NEW ", n, -1, len(content.encode("utf-8")), trunc))
            n_new += 1

    print(f"Read 恢复文件={len(best)}  仅Read有(新)={n_new}  与Write重复={n_fill}")
    print(f"{'REL':62s} {'态':4s} {'行':>5s} {'Write':>8s} {'Read':>8s} trunc")
    for rel, tag, n, wl, rl, trunc in report:
        print(f"{rel[:62]:62s} {tag} {n:5d} {wl:8d} {rl:8d} {trunc}")

if __name__ == "__main__":
    main()
