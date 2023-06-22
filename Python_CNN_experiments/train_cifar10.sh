#!/bin/bash

seedss=("182" "271" "944" "583" "220")

for seeds in "${seedss[@]}"; do
    # BP
    python train.py --model VGG16 -d cifar10 -e 130 -b 100 -l 0.025 --wl 5e-3 --we 10 -m 0.9 --wd 5e-4 -c 10 -a -s $seeds -n cifar10_bp_seed-${seeds}
    # DP
    python train.py --model VGG16 -d cifar10 -e 130 -b 100 -l 0.025 --wl 5e-3 --we 10 -m 0.9 --wd 5e-4 -c 10 -s $seeds -n cifar10_dp_seed-${seeds}
    #KP-DP
    python train.py --model VGG16 -d cifar10 -e 130 -b 100 -l 0.025 --wl 1e-4 --we 15 -m 0.9 --wd 5e-4 --ca --da -c 10 -s $seeds -n cifar10_kp-dp_seeds-${seeds}
done