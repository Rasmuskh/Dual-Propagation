#!/bin/bash

seedss=("182" "271" "944" "583" "220")

for seeds in "${seedss[@]}"; do
    # BP
    python train.py --model VGG16 -d imagenet -e 200 -b 250 -l 0.015 --wl 5e-3 --we 10 -m 0.9 --wd 5e-4 -c 1000 -a -s $seeds -p 95 -n imagenet_bp_seed-${seeds}
    # DP
    python train.py --model VGG16 -d imagenet -e 200 -b 250 -l 0.015 --wl 5e-3 --we 10 -m 0.9 --wd 5e-4 -c 1000 -s $seeds -p 95 -n imagenet_dp_seed-${seeds}
done