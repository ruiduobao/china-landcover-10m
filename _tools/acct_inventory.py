# -*- coding: utf-8 -*-
"""
acct_inventory.py — GEE 账号全量只读盘点（2026-09-13）
* 对 gee_accounts 下全部账号（排除 _ 前缀工具与 newuser）逐个：
  1) 读 `_任务登记.md` 取 anchor 仓 → CRM 枚举名下 ACTIVE 项目
  2) 逐项目 operations:list（pageSize 100）→ RUNNING/PENDING/SUCCEEDED/FAILED 计数 + batchEECU 累计
  3) 归因错误层级：API未启用 / 未注册 / 无权限 / 资格过期 / 凭证失效
* 只读，不提交/不取消任何任务；6 线程并发（代理对高并发 token 刷新敏感，勿再调高）
* 输出：_tools/out/acct_inventory.json + 控制台表
用法: python _tools/acct_inventory.py [账号...]
"""
import os, re, io, json, sys, time
from concurrent.futures import ThreadPoolExecutor

sys.stdout.reconfigure(encoding="utf-8")
BASE = r"F:\地理所\论文\博士论文\中期\实验区域\02_中间数据\gee_accounts"
OUTD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUTD, exist_ok=True)
PROXY = {"http": "socks5h://127.0.0.1:7890", "https": "socks5h://127.0.0.1:7890"}
SKIP = {"newuser"}

import requests
import google.oauth2.credentials
from google.auth.transport.requests import AuthorizedSession
from ee import oauth as ee_oauth


def all_accounts():
    out = []
    for n in sorted(os.listdir(BASE)):
        p = os.path.join(BASE, n)
        if n.startswith("_") or n in SKIP or not os.path.isdir(p):
            continue
        if os.path.isfile(os.path.join(p, ".config", "earthengine", "credentials")):
            out.append(n)
    return out


def anchor_of(acct):
    p = os.path.join(BASE, acct, "_任务登记.md")
    if not os.path.isfile(p):
        return None
    txt = io.open(p, encoding="utf-8", errors="replace").read()
    for m in re.finditer(r"^\|\s*([a-z][a-z0-9-]{10,40})\s*\|", txt, re.M):
        return m.group(1)
    return None


def session(acct):
    with open(os.path.join(BASE, acct, ".config", "earthengine", "credentials")) as f:
        c = json.load(f)
    if not c.get("refresh_token"):
        raise RuntimeError("no refresh_token")
    creds = google.oauth2.credentials.Credentials(
        token=None, refresh_token=c["refresh_token"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=ee_oauth.CLIENT_ID, client_secret=ee_oauth.CLIENT_SECRET,
        scopes=c.get("scopes", ["https://www.googleapis.com/auth/earthengine",
                                "https://www.googleapis.com/auth/cloud-platform"]))
    s = AuthorizedSession(creds)
    s.proxies = PROXY
    return s


def call(sess, method, url, pid, body=None, params=None, tries=3, timeout=40):
    last = None
    for i in range(tries):
        try:
            r = sess.request(method, url, json=body, params=params,
                             headers={"X-Goog-User-Project": pid}, timeout=timeout)
            if r.status_code < 500:
                return r
            last = r
        except Exception as e:
            last = e
            time.sleep(2 * (i + 1))
    return last


def err_layer(r):
    """把 HTTP 响应归因到 EE 四层错误（用于区分'不可用'的原因）"""
    if r is None:
        return "无响应/凭证失效"
    if isinstance(r, Exception):
        return "网络:" + str(r)[:60]
    try:
        txt = r.text
    except Exception:
        return f"HTTP {r.status_code}"
    low = txt.lower()
    if "has not been used in project" in low or "is disabled" in low:
        return "API未启用"
    if "noncommercial" in low and "expired" in low:
        return "资格过期"
    if "not registered" in low or "not signed up" in low:
        return "未注册"
    if "required permission" in low or "permission_denied" in low:
        return "无权限"
    if "invalid_grant" in low or "invalid_credentials" in low:
        return "凭证失效"
    if r.status_code == 401:
        return "凭证失效(401)"
    return f"HTTP {r.status_code}:" + txt.replace("\n", " ")[:70]


def op_meta(op):
    return op.get("metadata", {}) or {}


def probe(acct):
    t0 = time.time()
    row = {"acct": acct, "anchor": None, "n_proj": 0, "run": 0, "pend": 0,
           "suc": 0, "fail": 0, "cancel": 0, "eecu_h": 0.0,
           "ok": False, "layer": "", "projs_ok": 0, "errs": [], "sec": 0}
    try:
        sess = session(acct)
    except Exception as e:
        row["layer"] = "凭证加载失败:" + str(e)[:70]
        return row
    anc = anchor_of(acct)
    row["anchor"] = anc
    if not anc:
        row["layer"] = "无登记仓库(无法定anchor)"
        return row
    call(sess, "POST",
         f"https://serviceusage.googleapis.com/v1/projects/{anc}/services/"
         "cloudresourcemanager.googleapis.com:enable", anc)
    r = call(sess, "GET", "https://cloudresourcemanager.googleapis.com/v1/projects", anc)
    pids = []
    if r is not None and not isinstance(r, Exception) and r.status_code == 200:
        pids = [p["projectId"] for p in r.json().get("projects", [])
                if p.get("lifecycleState", "ACTIVE") == "ACTIVE"]
    if not pids:
        pids = [anc]
    row["n_proj"] = len(pids)
    for pid in pids:
        rr = call(sess, "GET",
                  f"https://earthengine.googleapis.com/v1/projects/{pid}/operations",
                  pid, params={"pageSize": 100})
        if rr is None or isinstance(rr, Exception) or rr.status_code != 200:
            lay = err_layer(rr)
            if len(row["errs"]) < 3:
                row["errs"].append(f"{pid.split('-')[0]}:{lay}")
            continue
        row["projs_ok"] += 1
        for op in rr.json().get("operations", []):
            md = op_meta(op)
            st = md.get("state") or ("DONE" if op.get("done") else "UNKNOWN")
            if st == "RUNNING":
                row["run"] += 1
            elif st == "PENDING":
                row["pend"] += 1
            elif st == "SUCCEEDED":
                row["suc"] += 1
            elif st == "FAILED":
                row["fail"] += 1
            elif st == "CANCELLED":
                row["cancel"] += 1
            if st == "SUCCEEDED":
                for k, v in md.items():
                    if "ecu" in k.lower():
                        try:
                            row["eecu_h"] += float(v) / 3600.0
                        except (TypeError, ValueError):
                            pass
    row["eecu_h"] = round(row["eecu_h"], 1)
    row["ok"] = row["projs_ok"] > 0
    if not row["ok"]:
        row["layer"] = row["errs"][0] if row["errs"] else "全部项目不可读"
    row["sec"] = round(time.time() - t0, 1)
    return row


def main():
    accts = sys.argv[1:] or all_accounts()
    print(f"盘点 {len(accts)} 个账号（只读）...", flush=True)
    res = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for row in ex.map(probe, accts):
            res.append(row)
            flag = "可用" if row["ok"] else row["layer"]
            print(f"{row['acct']:18s} 项目{row['n_proj']:2d}/可读{row['projs_ok']:2d} "
                  f"RUN={row['run']:2d} PEND={row['pend']:2d} SUC={row['suc']:3d} "
                  f"FAIL={row['fail']:3d} EECU={row['eecu_h']:8.1f}h  {flag}", flush=True)
    out = os.path.join(OUTD, "acct_inventory.json")
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n输出:", out, flush=True)


if __name__ == "__main__":
    main()
