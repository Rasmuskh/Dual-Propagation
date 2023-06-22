import argparse

from flax.linen import Conv, Dense, relu

from src import ConvAsym, DenseAsym
from src import VGG_like, VGG11, VGG16
import src.activation_functions as act
from src import get_cifar10, get_cifar100, get_imagenet_32x32

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num-epochs', '-e', default=20, type=int,
                    help='')
parser.add_argument('--batch-size', '-b', default=256, type=int,
                    help='')
parser.add_argument('--learning-rate', '-l', default=0.015, type=float,
                    help='')
parser.add_argument('--warmup-learning-rate', '--wl', default=1e-4, type=float,
                    help='')
parser.add_argument('--warmup-epochs', '--we', default=10, type=int,
                    help='')
parser.add_argument('--momentum', '-m', default=0.9, type=float, const=None, action='store', nargs='?',
                    help='')
parser.add_argument('--momentum_updated', '--mu', default=None, type=float, const=None, action='store', nargs='?',
                    help='')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='')
parser.add_argument('--conv-asym', '--ca', action='store_true',
                    help='This flag gives convolutional layers in the chosen model distinct feedback weights (which are trained with Kolen-Polack learning)')
parser.add_argument('--dense-asym', '--da', action='store_true',
                    help='This flag gives dense (fully connected) layers in the chosen model distinct feedback weights (which are trained with Kolen-Polack learning)')
parser.add_argument('--no-dual-activation', '-a', action='store_true',
                    help='This flag will make the network use back-propagation based relu units rather than the default dual propagation based relu units.')
parser.add_argument('--num-classes', '-c', default=1000, type=int,
                    help='')
parser.add_argument('--seeds', '-s', nargs='+', default=[220], type=int,
                    help='')
parser.add_argument('--percent-train', '-p',  default=95, type=int,
                    help='How much of the training data to use for training (the remainder will be used for validation). Only relevant for Imagenet32x32 (see paper for details).')
parser.add_argument('--experiment-name', '-n', default='test',
                    help='A string denoting the name of the experiment. A directory with this name will be created. If the directory already exists you will get an error (to avoid accidentally overwriting existing expperiments).')
models = dict(VGG_like=VGG_like, VGG11=VGG11, VGG16=VGG16)
parser.add_argument('--model', choices=models.keys(), default='VGG16',
                    help='')
datasets = dict(cifar10=get_cifar10, cifar100=get_cifar100, imagenet=get_imagenet_32x32)
parser.add_argument('--dataset', '-d', choices=datasets.keys(), default='imagenet',
                    help='')

config = parser.parse_args()

config.model = models[config.model](
                training    = True,
                ConvLayer   = ConvAsym if config.conv_asym else Conv,
                DenseLayer  = DenseAsym if config.dense_asym else Dense,
                act         = relu if config.no_dual_activation else act.relu_dualprop,
                num_classes = config.num_classes
            )

if config.dataset == 'imagenet':
    config.train_ds, config.val_ds, config.test_ds = datasets[config.dataset](config.batch_size, percent_train=config.percent_train)
else:
    config.train_ds, config.val_ds, config.test_ds = datasets[config.dataset]()
