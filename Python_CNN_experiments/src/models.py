from flax import linen as nn
from flax.linen import Conv, ConvLocal, Dense                    # The Linen API
import jax.numpy as jnp
from typing import Callable
import sys
sys.path.append("/src/")
from src.custom_layers import ConvAsym, DenseAsym


class VGG_like(nn.Module):
    training: bool
    ConvLayer: Callable
    DenseLayer: Callable
    act: Callable
    num_classes: int

    @nn.compact
    def __call__(self, x):
        assert (self.ConvLayer == Conv) or (self.ConvLayer == ConvLocal) or (self.ConvLayer == ConvAsym)

        # Block 1:
        x = self.ConvLayer(features=128, kernel_size=(3, 3), padding='same', name='c0')(x)
        # x = nn.BatchNorm(use_running_average=not self.training, momentum=0.9, epsilon=1e-5, dtype=jnp.float32)(x)
        x = self.act(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Block 2:
        x = self.ConvLayer(features=128, kernel_size=(3, 3), padding='same', name='c1')(x)
        x = self.act(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Block 3:
        x = self.ConvLayer(features=256, kernel_size=(3, 3), padding='same', name='c2')(x)
        x = self.act(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Block 4:
        x = self.ConvLayer(features=256, kernel_size=(3, 3), padding='same', name='c3')(x)
        x = self.act(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Block 5:
        x = self.ConvLayer(features=512, kernel_size=(3, 3), padding='same', name='c4')(x)
        x = self.act(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Block 6: Fully connected layers
        x = x.reshape((x.shape[0], -1))  # flatten
        x = self.DenseLayer(features=self.num_classes, name='d0')(x)
        return x

class VGG11(nn.Module):
    training: bool
    ConvLayer: Callable
    DenseLayer: Callable
    act: Callable
    num_classes: int

    @nn.compact
    def __call__(self, x):
        assert (self.ConvLayer == Conv) or (self.ConvLayer == ConvLocal) or (self.ConvLayer == ConvAsym)

        # Block 1:
        x = self.ConvLayer(features=64, kernel_size=(3, 3), padding='same', name='c0')(x)
        x = self.act(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Block 2:
        x = self.ConvLayer(features=128, kernel_size=(3, 3), padding='same', name='c1')(x)
        x = self.act(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Block 3:
        x = self.ConvLayer(features=256, kernel_size=(3, 3), padding='same', name='c2')(x)
        x = self.act(x)
        x = self.ConvLayer(features=256, kernel_size=(3, 3), padding='same', name='c3')(x)
        x = self.act(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Block 4:
        x = self.ConvLayer(features=512, kernel_size=(3, 3), padding='same', name='c4')(x)
        x = self.act(x)
        x = self.ConvLayer(features=512, kernel_size=(3, 3), padding='same', name='c5')(x)
        x = self.act(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Block 5:
        x = self.ConvLayer(features=512, kernel_size=(3, 3), padding='same', name='c6')(x)
        x = self.act(x)
        x = self.ConvLayer(features=512, kernel_size=(3, 3), padding='same', name='c7')(x)
        x = self.act(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Block 6: Fully connected layers
        x = x.reshape((x.shape[0], -1))  # flatten
        x = self.DenseLayer(features=4096, name='d0')(x)
        x = self.act(x)
        x = self.DenseLayer(features=4096, name='d1')(x)
        x = self.act(x)
        x = self.DenseLayer(features=self.num_classes, name='d2')(x)
        return x

class VGG16(nn.Module):
    training: bool
    ConvLayer: Callable
    DenseLayer: Callable
    act: Callable
    num_classes: int

    @nn.compact
    def __call__(self, x):
        assert (self.ConvLayer == nn.Conv) or (self.ConvLayer == nn.ConvLocal) or (self.ConvLayer == ConvAsym)

        # Block 1: two 64 channel conv layers
        x = self.ConvLayer(features=64, kernel_size=(3, 3), padding='same', name='c0')(x)
        x = self.act(x)
        x = self.ConvLayer(features=64, kernel_size=(3, 3), padding='same', name='c1')(x)
        x = self.act(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Block 2: two 128 channel conv layers
        x = self.ConvLayer(features=128, kernel_size=(3, 3), padding='same', name='c2')(x)
        x = self.act(x)
        x = self.ConvLayer(features=128, kernel_size=(3, 3), padding='same', name='c3')(x)
        x = self.act(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Block 3: three 256 channel conv layers
        x = self.ConvLayer(features=256, kernel_size=(3, 3), padding='same', name='c4')(x)
        x = self.act(x)
        x = self.ConvLayer(features=256, kernel_size=(3, 3), padding='same', name='c5')(x)
        x = self.act(x)
        x = self.ConvLayer(features=256, kernel_size=(3, 3), padding='same', name='c6')(x)
        x = self.act(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Block 4: three 512 channel conv layers
        x = self.ConvLayer(features=512, kernel_size=(3, 3), padding='same', name='c7')(x)
        x = self.act(x)
        x = self.ConvLayer(features=512, kernel_size=(3, 3), padding='same', name='c8')(x)
        x = self.act(x)
        x = self.ConvLayer(features=512, kernel_size=(3, 3), padding='same', name='c9')(x)
        x = self.act(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Block 5: three 512 channel conv layers
        x = self.ConvLayer(features=512, kernel_size=(3, 3), padding='same', name='c10')(x)
        x = self.act(x)
        x = self.ConvLayer(features=512, kernel_size=(3, 3), padding='same', name='c11')(x)
        x = self.act(x)
        x = self.ConvLayer(features=512, kernel_size=(3, 3), padding='same', name='c12')(x)
        x = self.act(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Block 6: Fully connected layers
        x = x.reshape((x.shape[0], -1))  # flatten
        x = self.DenseLayer(features=4096, name='d0')(x)
        x = self.act(x)
        x = self.DenseLayer(features=4096, name='d1')(x)
        x = self.act(x)
        x = self.DenseLayer(features=self.num_classes, name='d2')(x)
        return x