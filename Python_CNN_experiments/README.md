# CNN experiments
The CNN experiments can be rerun by running `./train_cifar10.sh`, `./train_cifar100.sh` and `./train_imagenet32x32.sh` from the command line. By default it will run for five random seeds, so make sure to change that if you just want to do a single run. 
The arguments used correspond to the parameters listed in table 6 in appendix C of the paper.
To see a breakdown of the possible command line arguments run `python train.py --h` in the command line.