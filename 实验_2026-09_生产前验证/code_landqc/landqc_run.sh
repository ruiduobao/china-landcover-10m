#!/bin/bash
# landqc_run.sh <crop|veg|water|urban> — 按预注册账户表并行启动本组抓取全部分片（幂等，已存在块自动跳过）
cd F:/lc_work/v31_exp/code
BASE=../data/land_qc
mkdir -p $BASE/logs $BASE/feats
case $1 in
  crop)  ACCTS=(ccstqqb ch2fg2t9 chengruiduobao hrbm3pdw ief3nj n08zt4); LIST=$BASE/lists/grp1_crop.parquet; NP=6 ;;
  veg)   ACCTS=(n51j08y nama8lg neg69g nhqz5uj nkhn6s9 s4ezbd sgqv7bf udn4q4); LIST=$BASE/lists/grp2_veg.parquet; NP=8 ;;
  water) ACCTS=(v054r8u vh481j w2qe4hiu); LIST=$BASE/lists/grp3_water.parquet; NP=3 ;;
  urban) ACCTS=(x8diho2n xysongf zsi8emo ughwvm7968.med); LIST=$BASE/lists/grp4_urban.parquet; NP=4 ;;
  *) echo "usage: $0 crop|veg|water|urban"; exit 1 ;;
esac
for i in $(seq 0 $((NP-1))); do
  nohup python v7_land_fetch.py --list $LIST --part $i --nparts $NP --acct ${ACCTS[$i]} --outdir $BASE/feats/$1 > $BASE/logs/$1_p$i.log 2>&1 &
done
sleep 2
echo "started $NP parts for $1"
