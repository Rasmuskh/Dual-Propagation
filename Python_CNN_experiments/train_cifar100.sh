#!/bin/bash

seedss=("182" "271" "944" "583" "220")

for seeds in "${seedss[@]}"; do
    # BP
    python train.py --model VGG16 -d cifar100 -e 200 -b 50 -l 0.015 --wl 5e-3 --we 10 -m 0.9 --wd 5e-4 -c 100 -a -s $seeds -n cifar100_bp_seed-${seeds}
    # DP
    python train.py --model VGG16 -d cifar100 -e 200 -b 50 -l 0.015 --wl 5e-3 --we 10 -m 0.9 --wd 5e-4 -c 100 -s $seeds -n cifar100_dp_seed-${seeds}
    # KP-DP
    python train.py --model VGG16 -d cifar100 -e 200 -b 50 -l 0.015 --wl 1e-4 --we 30 -m 0.9 --wd 5e-4 --ca --da -c 100 -s $seeds -n cifar100_kp-dp_seed-${seeds}
done