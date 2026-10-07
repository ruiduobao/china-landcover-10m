# -*- coding: utf-8 -*-
"""gen_fleet_new.py — 从各账号 _任务登记.md 生成新编队 FLEET 配置（anchor=登记首仓+确定性指纹）"""
import os, re, io, json

BASE = r"F:\地理所\论文\博士论文\中期\实验区域\02_中间数据\gee_accounts"
OUT = os.path.join(BASE, "fleet_new_13accounts.json")
# 2026-09-12 用户指示：只用 10 个空闲账号；berk95733 已实证；其余 3 个（y30b63ye/zitwwufh/zixen8v8）留作备用
USABLE = ["berk95733", "1t3j9iq0", "1v6l7i", "1yxth5g", "5tyxedp", "als74akz",
          "j8tv79n", "rbj1et5", "s8xpcl1w", "udn4q4"]
PROPS = ['row_id', 'rid', 'ptid', 'sample_id', 'fid_r7', 'pt_idx', 'sid',
         'obs_id', 'key_id', 'rownum', 'sp_id', 'cell_id', 'gidx']


def first_repo(acct):
    p = os.path.join(BASE, acct, "_任务登记.md")
    for m in re.finditer(r"^\|\s*([a-z][a-z0-9-]{10,40})\s*\|",
                         io.open(p, encoding="utf-8").read(), re.M):
        return m.group(1)
    return None


def main():
    fleet = {}
    for i, a in enumerate(USABLE):
        anchor = first_repo(a)
        assert anchor, f"{a} 无登记仓库"
        fleet[a] = [anchor, dict(prop=PROPS[i % len(PROPS)],
                                 ts=[2, 3, 4][i % 3],
                                 rev=bool(i % 2),
                                 sleep=(1 + i % 3, 3 + i % 4),
                                 nretry=3 + i % 2)]
    json.dump(fleet, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("输出:", OUT)
    for a, (pid, fp) in fleet.items():
        print(f"  {a:16s} {pid:28s} prop={fp['prop']:10s} ts={fp['ts']} rev={fp['rev']}")


if __name__ == "__main__":
    main()
