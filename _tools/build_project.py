# -*- coding: utf-8 -*-
"""合并 Write 重放版 + Read 补充版，落成正式项目树。"""
import os, shutil, json, hashlib

ROOT = r"F:\地理所\论文\中国土地覆盖数据_2017-2024"
W = os.path.join(ROOT, "_复原源")
R = os.path.join(ROOT, "_复原源_Read补充")
TARGET = ROOT          # 直接落根，与 Z 盘原布局一致
LOG = os.path.join(ROOT, "_复原记录")
THRESH = 200           # Read 版比 Write 版长这么多字节才采用 Read 版

def read(p):
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

def main():
    os.makedirs(LOG, exist_ok=True)
    files = {}
    for base, tag in ((W, "W"), (R, "R")):
        for root, _d, fs in os.walk(base):
            for f in fs:
                if f.endswith(".pyc"):
                    continue
                fp = os.path.join(root, f)
                rel = os.path.relpath(fp, base).replace("\\", "/")
                if rel.startswith("_") or rel == "_manifest.json":
                    continue
                files.setdefault(rel, {})[tag] = fp

    rows = []
    n = 0
    for rel in sorted(files):
        w, r = files[rel].get("W"), files[rel].get("R")
        src = None; note = ""
        if w and r:
            wl, rl = os.path.getsize(w), os.path.getsize(r)
            if rl > wl + THRESH:
                src, note = r, f"READ+ (W={wl} R={rl})"
            else:
                src, note = w, f"WRITE (W={wl} R={rl})"
        elif w:
            src, note = w, "WRITE only"
        else:
            src, note = r, "READ only"
        dst = os.path.join(TARGET, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        rows.append((rel, note))
        n += 1
    with open(os.path.join(LOG, "merge_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"合并文件数 {n}\n\n")
        for rel, note in rows:
            f.write(f"{note:22s} {rel}\n")
    print("落盘文件数:", n)
    for rel, note in rows:
        if note.startswith("READ+"):
            print("  READ优先:", rel, note)

if __name__ == "__main__":
    main()
