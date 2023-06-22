#!/bin/bash

seedss=("182" "271" "944" "583" "220")

# 

for seeds in "${seedss[@]}"; do
    python train.py --model VGG16 -d cifar10 -e 130 -b 100 -l 0.025 --wl 5e-3 --we 10 -m 0.9 --wd 5e-4 -c 10 -s $seeds -n cifar10_bp_seed-${seeds}
    python train.py --model VGG16 -d cifar10 -e 130 -b 100 -l 0.025 --wl 5e-3 --we 10 -m 0.9 --wd 5e-4 -c 10 -s $seeds -n cifar10_dp_seed-${seeds}
    python train.py --model VGG16 -d cifar10 -e 130 -b 100 -l 0.025 --wl 1e-4 --we 15 -m 0.9 --wd 5e-4 --ca --da -c 10 -s $seeds -n cifar10_kp-dp_seeds-${seeds}
    
    python train.py --model VGG16 -d cifar100 -e 200 -b 50 -l 0.015 --wl 5e-3 --we 10 -m 0.9 --wd 5e-4 -c 100 -s $seeds -n cifar100_bp_seed-${seeds}
    python train.py --model VGG16 -d cifar100 -e 200 -b 50 -l 0.015 --wl 5e-3 --we 10 -m 0.9 --wd 5e-4 -c 100 -s $seeds -n cifar100_dp_seed-${seeds}
    python train.py --model VGG16 -d cifar100 -e 200 -b 50 -l 0.015 --wl 1e-4 --we 30 -m 0.9 --wd 5e-4 --ca --da -c 100 -s $seeds -n cifar100_kp-dp_seed-${seeds}
    
    python train.py --model VGG16 -d imagenet -e 200 -b 250 -l 0.015 --wl 5e-3 --we 10 -m 0.9 --wd 5e-4 -c 1000 -s $seeds -p 95 -n imagenet_bp_seed-${seeds}
    python train.py --model VGG16 -d imagenet -e 200 -b 250 -l 0.015 --wl 5e-3 --we 10 -m 0.9 --wd 5e-4 -c 1000 -s $seeds -p 95 -n imagenet_dp_seed-${seeds}
done