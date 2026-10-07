# -*- coding: utf-8 -*-
"""
probe_candidates_20260912.py — T1 候选账号批量巡检（只读）
* 对 gee_accounts 下 16 个候选账号（非 19 禁动）逐个：
  1) CRM 枚举项目 + 逐仓 listOperations（RUNNING/PENDING/EECU 累计）
  2) AEF 探活：子进程 HOME/USERPROFILE=账号目录，ee.Initialize(project=<首仓>)
     + GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL .size().getInfo()
* 输出: 候选账号盘点_20260912.json + 控制台表格
"""
import os, re, io, json, time, subprocess, sys, threading, queue

BASE = r"F:\地理所\论文\博士论文\中期\实验区域\02_中间数据\gee_accounts"
PROXY = {"http": "socks5h://127.0.0.1:7890", "https": "socks5h://127.0.0.1:7890"}
CANDS = ["1t3j9iq0", "1v6l7i", "1yxth5g", "5tyxedp", "als74akz", "berk95733",
         "fcz0iuwf", "j8tv79n", "newuser", "rbj1et5", "s8xpcl1w", "udn4q4",
         "vidalmvtpronet18", "y30b63ye", "zitwwufh", "zixen8v8"]

import requests
import google.oauth2.credentials
from google.auth.transport.requests import AuthorizedSession
from ee import oauth as ee_oauth


def make_session(acct):
    with open(os.path.join(BASE, acct, ".config", "earthengine", "credentials")) as f:
        c = json.load(f)
    creds = google.oauth2.credentials.Credentials(
        token=None, refresh_token=c["refresh_token"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=ee_oauth.CLIENT_ID, client_secret=ee_oauth.CLIENT_SECRET,
        scopes=c.get("scopes", ["https://www.googleapis.com/auth/earthengine",
                                "https://www.googleapis.com/auth/cloud-platform"]))
    s = AuthorizedSession(creds)
    s.proxies = PROXY
    return s


def req(sess, method, url, pid, body=None, tries=3):
    r = None
    for i in range(tries):
        try:
            r = sess.request(method, url, json=body,
                             headers={"X-Goog-User-Project": pid}, timeout=45)
            if r.status_code < 500:
                return r
        except Exception:
            r = None
            time.sleep(2 * (i + 1))
    return r


def repos_from登记(acct):
    p = os.path.join(BASE, acct, "_任务登记.md")
    if not os.path.isfile(p):
        return []
    out = []
    for m in re.finditer(r"^\|\s*([a-z][a-z0-9-]{10,40})\s*\|", io.open(p, encoding="utf-8").read(), re.M):
        if m.group(1) not in out:
            out.append(m.group(1))
    return out


def list_pids(sess, anchor):
    req(sess, "POST", f"https://serviceusage.googleapis.com/v1/projects/{anchor}/services/cloudresourcemanager.googleapis.com:enable", anchor)
    r = req(sess, "GET", "https://cloudresourcemanager.googleapis.com/v1/projects", anchor)
    if r is None or r.status_code != 200:
        return None
    return [p["projectId"] for p in r.json().get("projects", [])
            if p.get("lifecycleState", "ACTIVE") == "ACTIVE"]


def op_state(op):
    md = op.get("metadata", {})
    return md.get("state") or ("DONE" if op.get("done") else "UNKNOWN")


def op_eecu(op):
    md = op.get("metadata", {})
    for k, v in md.items():
        if "ecu" in k.lower():
            try: return float(v)
            except (TypeError, ValueError): pass
    return 0.0


def aef_probe(acct, pid, q):
    """子进程探 AEF；返回 (ok, detail)"""
    env = dict(os.environ)
    acd = os.path.join(BASE, acct)
    env["HOME"] = acd
    env["USERPROFILE"] = acd
    env["HTTP_PROXY"] = "socks5h://127.0.0.1:7890"
    env["HTTPS_PROXY"] = "socks5h://127.0.0.1:7890"
    code = ("import ee;ee.Initialize(project=%r);"
            "print(ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL').size().getInfo())" % pid)
    try:
        r = subprocess.run([sys.executable, "-c", code], env=env, timeout=120,
                           capture_output=True, text=True)
        ok = (r.returncode == 0 and r.stdout.strip().isdigit())
        q.put((acct, ok, r.stdout.strip()[:40] if ok else (r.stderr.strip()[-160:])))
    except Exception as e:
        q.put((acct, False, str(e)[:160]))


def probe_one(acct, res):
    row = {"acct": acct, "repos_登记": 0, "pids_crm": None, "n_proj": 0,
           "RUNNING": 0, "PENDING": 0, "batchEECU_h": 0.0, "aef": None, "aef_detail": ""}
    try:
        sess = make_session(acct)
    except Exception as e:
        row["aef_detail"] = "凭证加载失败:" + str(e)[:80]
        res.append(row); return
    reps = repos_from登记(acct)
    row["repos_登记"] = len(reps)
    anchor = reps[0] if reps else None
    if not anchor:
        row["aef_detail"] = "无登记仓库"
        res.append(row); return
    pids = list_pids(sess, anchor)
    row["pids_crm"] = bool(pids)
    if pids:
        row["n_proj"] = len(pids)
        for pid in pids:
            r = req(sess, "POST",
                    f"https://earthengine.googleapis.com/v1/projects/{pid}/operations:list",
                    pid, body={"pageSize": 100})
            if r is None or r.status_code != 200:
                continue
            for op in r.json().get("operations", []):
                st = op_state(op)
                if st in ("RUNNING", "PENDING"):
                    row[st] += 1
                if st == "SUCCEEDED":
                    row["batchEECU_h"] += op_eecu(op) / 3600.0
    row["batchEECU_h"] = round(row["batchEECU_h"], 1)
    q2 = queue.Queue()
    t = threading.Thread(target=aef_probe, args=(acct, anchor, q2))
    t.start(); t.join(150)
    if not q2.empty():
        _, ok, det = q2.get_nowait()
        row["aef"] = bool(ok); row["aef_detail"] = det
    else:
        row["aef"] = False; row["aef_detail"] = "AEF 探活超时(150s)"
    res.append(row)


def main():
    res = []
    for acct in CANDS:
        probe_one(acct, res)
        r = res[-1]
        print(f"{acct:18s} 登记{r['repos_登记']:2d}仓 CRM={r['pids_crm']} 项目{r['n_proj']:2d} "
              f"RUN={r['RUNNING']} PEND={r['PENDING']} batchEECU={r['batchEECU_h']:7.1f}h "
              f"AEF={r['aef']} {r['aef_detail'][:50]}", flush=True)
    out = os.path.join(BASE, "候选账号盘点_20260912.json")
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n输出:", out, flush=True)


if __name__ == "__main__":
    main()
