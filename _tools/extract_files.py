# -*- coding: utf-8 -*-
"""从 ZCode 会话库重放 Write/Edit/heredoc，重建中国土地覆盖项目源码与文档。"""
import sqlite3, json, os, re, sys, collections

DB = r"C:\Users\Administrator\.zcode\cli\db\db.sqlite"
OUT = r"F:\地理所\论文\中国土地覆盖数据_2017-2024\_复原源"

PREFIXES = [
    r"Z:\Mywork\论文\中国土地覆盖数据\\",
    r"Z:\Mywork\论文\BaiduSyncdisk\中国土地覆盖数据\\",
    "Z:/Mywork/论文/中国土地覆盖数据/",
    "Z:/Mywork/论文/BaiduSyncdisk/中国土地覆盖数据/",
    r"F:\r7_prod\\",
    "F:/r7_prod/",
]

def norm(p):
    if not p:
        return None
    p = p.replace("\\", "/")
    for pre in PREFIXES:
        q = pre.replace("\\", "/")
        if p.startswith(q):
            rel = p[len(q):]
            if q.lower().startswith("f:/r7_prod"):
                return "生产_F盘自包含/" + rel
            return rel
    if p.lower().startswith("z:/mywork/论文/中国土地覆盖数据") or \
       p.lower().startswith("z:/mywork/论文/baidusyncdisk/中国土地覆盖数据"):
        return p.split("中国土地覆盖数据/", 1)[-1]
    return None

def main():
    c = sqlite3.connect(DB)
    cur = c.cursor()
    sess = [r[0] for r in cur.execute(
        "SELECT id FROM session WHERE directory LIKE '%中国土地覆盖%'").fetchall()]
    qm = ",".join("?" * len(sess))
    rows = cur.execute(
        f"SELECT session_id, data, time_created FROM part WHERE session_id IN ({qm}) "
        f"AND data LIKE '%\"type\":\"tool\"%' ORDER BY time_created", sess).fetchall()

    events = collections.defaultdict(list)   # rel -> [(t, op, payload)]
    manifest = []
    n_tool = n_skip = 0
    for sid, d, t in rows:
        try:
            j = json.loads(d)
        except Exception:
            continue
        tool = j.get("tool")
        st = j.get("state") or {}
        inp = st.get("input") or {}
        if st.get("status") == "error":
            continue      # 原会话中执行失败的调用不产生文件变更，跳过
        if tool == "Write":
            rel = norm(inp.get("file_path") or inp.get("path") or "")
            if not rel:
                n_skip += 1; continue
            n_tool += 1
            events[rel].append((t, "W", {"content": inp.get("content", "")}))
            manifest.append((rel, sid, t, "Write"))
        elif tool == "Edit":
            rel = norm(inp.get("file_path") or inp.get("path") or "")
            if not rel or inp.get("old_string") is None:
                n_skip += 1; continue
            n_tool += 1
            events[rel].append((t, "E", {
                "old": inp.get("old_string", ""),
                "new": inp.get("new_string", ""),
                "all": bool(inp.get("replace_all"))}))
            manifest.append((rel, sid, t, "Edit"))
        elif tool == "Bash":
            cmd = inp.get("command", "") or ""
            # heredoc: cat > "path" <<'EOF' ... EOF   (also bare EOF marker)
            for m in re.finditer(
                    r"(?:cat|tee)\s*(?:>\s*)?[\"']?([^\s\"'>|]+)[\"']?\s*<<\s*['\"]?(\w+)['\"]?\r?\n(.*?)\r?\n\2",
                    cmd, re.S):
                path, _mark, body = m.group(1), m.group(2), m.group(3)
                rel = norm(path)
                if not rel:
                    continue
                n_tool += 1
                events[rel].append((t, "W", {"content": body + "\n"}))
                manifest.append((rel, sid, t, "Bash-heredoc"))

    print(f"tool-writes parsed={n_tool} skipped(非本项目)={n_skip} unique_files={len(events)}")

    # 按时间重放
    result = {}
    warn = []
    for rel, evs in events.items():
        evs.sort(key=lambda x: x[0])
        content = None
        for t, op, pl in evs:
            if op == "W":
                content = pl["content"]
            elif op == "E":
                if content is None:
                    warn.append(f"EDIT-NOBASE {rel}")
                    continue
                if pl["old"] == "":
                    content = pl["new"]
                elif pl["old"] in content:
                    content = content.replace(pl["old"], pl["new"], -1 if pl["all"] else 1)
                else:
                    warn.append(f"EDIT-NOMATCH {rel} @{t}")
        if content is not None:
            result[rel] = content

    # 落盘
    if os.path.isdir(OUT):
        import shutil
        shutil.rmtree(OUT, ignore_errors=True)
    n = 0; total = 0
    for rel, content in sorted(result.items()):
        fp = os.path.join(OUT, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        with open(fp, "w", encoding="utf-8", newline="") as f:
            f.write(content)
        n += 1; total += len(content.encode("utf-8"))
    print(f"reconstructed files={n} bytes={total}")

    with open(os.path.join(OUT, "_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"files": sorted(result.keys()),
                   "provenance": [list(m) for m in manifest],
                   "warnings": warn}, f, ensure_ascii=False, indent=1)
    print("warnings:", len(warn))
    for w in warn[:40]:
        print("  ", w)

if __name__ == "__main__":
    main()
