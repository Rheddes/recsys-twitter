#!/bin/bash
set -o xtrace

cp ./val_new.tsv val_temp.tsv
for i in {1..4}
do
  tail -n 3246170 ./val_temp.tsv > ./val.$i.tsv # 3246170 = $(wc -l val_new.tsv) / 4 rounded up
  size=$(wc -c < ./val.$i.tsv | bc)
  truncate -s -"$size" ./val_temp.tsv
  echo "iter done"
done
