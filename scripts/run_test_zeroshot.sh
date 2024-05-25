#!/usr/bin/env bash
if [ -f $1 ]; then
  config=$1
else
  echo "need a config file"
  exit
fi

weight=$2

python -m torch.distributed.launch --master_port 1238 --nproc_per_node=1 \
    test_zeroshot.py --config ${config} --weights ${weight} ${@:3}


#bash scripts/run_test_zeroshot.sh  configs/hmdb51/hmdb_zero_shot.yaml exps/k400/ViT-B/32/f8/k400-vitb-32-f8.pt

#用K400(vitl-14)在HMDB51上zero-shot
#bash scripts/run_test_zeroshot.sh  /home/lyx/code/Text4Vis-main/configs/hmdb51/hmdb_zero_shot.yaml /home/lyx/code/Text4Vis-main/exps/k400/ViT-L/14/f8/k400-vitl-14-f8.pt
