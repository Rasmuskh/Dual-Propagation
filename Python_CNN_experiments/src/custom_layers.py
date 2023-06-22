# Copyright 2022 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import (Any, Callable, Iterable, List, Optional, Sequence, Tuple,
                    Union)
import jax
from jax.lax import conv_general_dilated, conv_general_dilated_local
import flax.linen as nn
from flax.linen.initializers import lecun_normal
from flax.linen.initializers import zeros
from flax.linen.module import compact
from flax.linen.module import Module
from flax.linen.dtypes import promote_dtype
from jax import eval_shape
from jax import lax
from jax import ShapedArray
import jax.numpy as jnp
import numpy as np
from jax import custom_vjp
from functools import partial
from jax import custom_jvp

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any
PrecisionLike = Union[None, str, lax.Precision, Tuple[str, str],
                      Tuple[lax.Precision, lax.Precision]]

default_kernel_init = lecun_normal()


"""Linear modules."""

#--------------------------------------------------
# Start custom random feedback Dense layer
#--------------------------------------------------

# custom FA matmul
@custom_vjp
def matmul_fa(A, B, C):
  return A@B

def matmul_fa_fwd(A, B, C):
    y = A@B
    return y, (A, B, C)

# f_bwd :: (a_and_z, dy) -> dz
def matmul_fa_bwd(ABC, dy):
    A, B, C = ABC
    # dA = dy@B.T
    dA = dy@C.T
    dB = A.T@dy
    dC = dB
    return (dA, dB, dC)

matmul_fa.defvjp(matmul_fa_fwd, matmul_fa_bwd)


class DenseAsym(nn.Module):

    features: int
    use_bias: bool = True
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros

    @compact
    def __call__(self, inputs: Array) -> Array:
        kernel = self.param('kernel',
                        self.kernel_init,
                        (jnp.shape(inputs)[-1], self.features),
                        self.param_dtype)
        kernel_asym = self.param('kernel_asym',
                    self.kernel_init,
                    (jnp.shape(inputs)[-1], self.features),
                    self.param_dtype)
        # print("::::::::, ", (kernel==kernel_asym).all())
        # kernel = jnp.abs(kernel)
        # kernel_asym = jnp.abs(kernel_asym)

        if self.use_bias:
            bias = self.param('bias', self.bias_init, (self.features,),
                            self.param_dtype)
        else:
            bias = None
        inputs, kernel, kernel_asym, bias = promote_dtype(inputs, kernel, kernel_asym, bias, dtype=self.dtype)
        # y = lax.dot_general(inputs, kernel,
        #                 (((inputs.ndim - 1,), (0,)), ((), ())),
        #                 precision=self.precision)
        # y = matmul_fa(kernel, inputs)
        y = matmul_fa(inputs, kernel, kernel_asym)

        if bias is not None:
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return y

#--------------------------------------------------
# End custom random feedback Dense layer
#--------------------------------------------------

#--------------------------------------------------
# Start custom random feedback Conv(_Local) layer
#--------------------------------------------------





def _conv_dimension_numbers(input_shape):
  """Computes the dimension numbers based on the input shape."""
  ndim = len(input_shape)
  lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
  rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
  out_spec = lhs_spec
  return lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)


PaddingLike = Union[str, int, Sequence[Union[int, Tuple[int, int]]]]
LaxPadding = Union[str, Sequence[Tuple[int, int]]]


def canonicalize_padding(padding: PaddingLike, rank: int) -> LaxPadding:
  """"Canonicalizes conv padding to a jax.lax supported format."""
  if isinstance(padding, str):
    return padding
  if isinstance(padding, int):
    return [(padding, padding)] * rank
  if isinstance(padding, Sequence) and len(padding) == rank:
    new_pad = []
    for p in padding:
      if isinstance(p, int):
        new_pad.append((p, p))
      elif isinstance(p, tuple) and len(p) == 2:
        new_pad.append(p)
      else:
        break
    if len(new_pad) == rank:
      return new_pad
  raise ValueError(
    f'Invalid padding format: {padding}, should be str, int,'
    f' or a sequence of len {rank} where each element is an'
    f' int or pair of ints.')


class _ConvAsym(Module):
  """Convolution Module wrapping `lax.conv_general_dilated[_local]`.

  Attributes:
    features: number of convolution filters.
    kernel_size: shape of the convolutional kernel. For 1D convolution,
      the kernel size can be passed as an integer. For all other cases, it must
      be a sequence of integers.
    strides: an integer or a sequence of `n` integers, representing the
      inter-window strides (default: 1).
    padding: either the string `'SAME'`, the string `'VALID'`, the string
      `'CIRCULAR'` (periodic boundary conditions), or a sequence of `n` `(low,
      high)` integer pairs that give the padding to apply before and after each
      spatial dimension. A single int is interpeted as applying the same padding
      in all dims and passign a single int in a sequence causes the same padding
      to be used on both sides. `'CAUSAL'` padding for a 1D convolution will
      left-pad the convolution axis, resulting in same-sized output.
    input_dilation: an integer or a sequence of `n` integers, giving the
      dilation factor to apply in each spatial dimension of `inputs`
      (default: 1). Convolution with input dilation `d` is equivalent to
      transposed convolution with stride `d`.
    kernel_dilation: an integer or a sequence of `n` integers, giving the
      dilation factor to apply in each spatial dimension of the convolution
      kernel (default: 1). Convolution with kernel dilation
      is also known as 'atrous convolution'.
    feature_group_count: integer, default 1. If specified divides the input
      features into groups.
    use_bias: whether to add a bias to the output (default: True).
    mask: Optional mask for the weights during masked convolution. The mask must
          be the same shape as the convolution weight matrix.
    dtype: the dtype of the computation (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer for the convolutional kernel.
    bias_init: initializer for the bias.
  """
  features: int
  kernel_size: Sequence[int]
  strides: Union[None, int, Sequence[int]] = 1
  padding: PaddingLike = 'SAME'
  input_dilation: Union[None, int, Sequence[int]] = 1
  kernel_dilation: Union[None, int, Sequence[int]] = 1
  feature_group_count: int = 1
  use_bias: bool = True
  mask: Optional[Array] = None
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  precision: PrecisionLike = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros

  @property
  def shared_weights(self) -> bool:
    """Defines whether weights are shared or not between different pixels.

    Returns:
      `True` to use shared weights in convolution (regular convolution).
      `False` to use different weights at different pixels, a.k.a.
      "locally connected layer", "unshared convolution", or "local convolution".

    """
    ...

  @compact
  def __call__(self, inputs: Array) -> Array:
    """Applies a (potentially unshared) convolution to the inputs.

    Args:
      inputs: input data with dimensions (*batch_dims, spatial_dims...,
        features). This is the channels-last convention, i.e. NHWC for a 2d
        convolution and NDHWC for a 3D convolution. Note: this is different from
        the input convention used by `lax.conv_general_dilated`, which puts the
        spatial dimensions last.
        Note: If the input has more than 1 batch dimension, all batch dimensions
        are flattened into a single dimension for the convolution and restored
        before returning.  In some cases directly vmap'ing the layer may yield
        better performance than this default flattening approach.  If the input
        lacks a batch dimension it will be added for the convolution and removed
        n return, an allowance made to enable writing single-example code.

    Returns:
      The convolved data.
    """

    if isinstance(self.kernel_size, int):
      raise TypeError('Expected Conv kernel_size to be a'
                      ' tuple/list of integers (eg.: [3, 3]) but got'
                      f' {self.kernel_size}.')
    else:
      kernel_size = tuple(self.kernel_size)

    def maybe_broadcast(x: Optional[Union[int, Sequence[int]]]) -> (
        Tuple[int, ...]):
      if x is None:
        # backward compatibility with using None as sentinel for
        # broadcast 1
        x = 1
      if isinstance(x, int):
        return (x,) * len(kernel_size)
      return tuple(x)

    # Combine all input batch dimensions into a single leading batch axis.
    num_batch_dimensions = inputs.ndim - (len(kernel_size) + 1)
    if num_batch_dimensions != 1:
      input_batch_shape = inputs.shape[:num_batch_dimensions]
      total_batch_size = int(np.prod(input_batch_shape))
      flat_input_shape = (
          (total_batch_size,) + inputs.shape[num_batch_dimensions:])
      inputs = jnp.reshape(inputs, flat_input_shape)

    # self.strides or (1,) * (inputs.ndim - 2)
    strides = maybe_broadcast(self.strides)
    input_dilation = maybe_broadcast(self.input_dilation)
    kernel_dilation = maybe_broadcast(self.kernel_dilation)

    padding_lax = canonicalize_padding(self.padding, len(kernel_size))
    if padding_lax == 'CIRCULAR':
      kernel_size_dilated = [
          (k - 1) * d + 1 for k, d in zip(kernel_size, kernel_dilation)
      ]
      zero_pad: List[Tuple[int, int]] = [(0, 0)]
      pads = (zero_pad + [((k - 1) // 2, k // 2) for k in kernel_size_dilated] +
              [(0, 0)])
      inputs = jnp.pad(inputs, pads, mode='wrap')
      padding_lax = 'VALID'
    elif padding_lax == 'CAUSAL':
      if len(kernel_size) != 1:
        raise ValueError(
            'Causal padding is only implemented for 1D convolutions.')
      left_pad = kernel_dilation[0] * (kernel_size[0] - 1)
      pads = [(0, 0), (left_pad, 0), (0, 0)]
      inputs = jnp.pad(inputs, pads)
      padding_lax = 'VALID'

    dimension_numbers = _conv_dimension_numbers(inputs.shape)
    in_features = jnp.shape(inputs)[-1]

    if self.shared_weights:
      # One shared convolutional kernel for all pixels in the output.
      assert in_features % self.feature_group_count == 0
      kernel_shape = kernel_size + (
          in_features // self.feature_group_count, self.features)

    else:
      if self.feature_group_count != 1:
        raise NotImplementedError(
            f'`lax.conv_general_dilated_local` does not support '
            f'`feature_group_count != 1`, got `{self.feature_group_count}`.'
        )

      # Need to know the spatial output shape of a standard convolution to
      # create the unshared convolution kernel.
      conv_output_shape = eval_shape(
          lambda lhs, rhs: lax.conv_general_dilated(  # pylint: disable=g-long-lambda
              lhs=lhs,
              rhs=rhs,
              window_strides=strides,
              padding=padding_lax,
              dimension_numbers=dimension_numbers,
              lhs_dilation=input_dilation,
              rhs_dilation=kernel_dilation,
          ),
          inputs,
          ShapedArray(kernel_size + (in_features, self.features), inputs.dtype)
      ).shape

      # One (unshared) convolutional kernel per each pixel in the output.
      kernel_shape = conv_output_shape[1:-1] + (np.prod(kernel_size) *
                                                in_features, self.features)

    if self.mask is not None and self.mask.shape != kernel_shape:
      raise ValueError('Mask needs to have the same shape as weights. '
                       f'Shapes are: {self.mask.shape}, {kernel_shape}')

    kernel = self.param('kernel', self.kernel_init, kernel_shape, self.param_dtype)
    kernel_asym = self.param('kernel_asym', self.kernel_init, kernel_shape, self.param_dtype)
    # kernel = jnp.abs(kernel)
    # kernel_asym = jnp.abs(kernel_asym)

    if self.mask is not None:
      kernel *= self.mask

    if self.use_bias:
      if self.shared_weights:
        # One bias weight per output channel, shared between pixels.
        bias_shape = (self.features,)
      else:
        # One bias weight per output entry, unshared betwen pixels.
        bias_shape = conv_output_shape[1:]

      bias = self.param('bias', self.bias_init, bias_shape, self.param_dtype)
    else:
      bias = None

    inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)
    if self.shared_weights:
      y = my_conv_general_dilated(
          inputs,
          kernel,
          kernel_asym,
          strides,
          padding_lax,
          lhs_dilation=input_dilation,
          rhs_dilation=kernel_dilation,
          dimension_numbers=dimension_numbers,
          feature_group_count=self.feature_group_count,
          precision=self.precision
      )
    else:
      y = lax.conv_general_dilated_local(
          lhs=inputs,
          rhs=kernel,
          window_strides=strides,
          padding=padding_lax,
          filter_shape=kernel_size,
          lhs_dilation=input_dilation,
          rhs_dilation=kernel_dilation,
          dimension_numbers=dimension_numbers,
          precision=self.precision
      )

    if self.use_bias:
      bias = bias.reshape((1,) * (y.ndim - bias.ndim) + bias.shape)
      y += bias

    if num_batch_dimensions != 1:
      output_shape = input_batch_shape + y.shape[1:]
      y = jnp.reshape(y, output_shape)
    return y



class ConvAsym(_ConvAsym):
  """Convolution Module wrapping `lax.conv_general_dilated`."""

  @property
  def shared_weights(self) -> bool:
    return True

class ConvAsymLocal(_ConvAsym):
  """Local convolution Module wrapping `lax.conv_general_dilated_local`."""

  @property
  def shared_weights(self) -> bool:
    return False



@partial(custom_jvp, nondiff_argnums=(3,4,5,6,7,8,9))
def my_conv_general_dilated(inputs, kernel, kernel_asym, strides, padding_lax, lhs_dilation, rhs_dilation, dimension_numbers, feature_group_count, precision):
  return conv_general_dilated(inputs, kernel, strides, padding_lax, lhs_dilation=lhs_dilation, rhs_dilation=rhs_dilation, dimension_numbers=dimension_numbers, feature_group_count=feature_group_count, precision=precision)

@my_conv_general_dilated.defjvp
def my_conv_general_dilated_jvp(strides, padding_lax, lhs_dilation, rhs_dilation, dimension_numbers, feature_group_count, precision, primals, tangents):
  x, kernel, kernel_asym = primals
  x_dot, kernel_dot, kernel_asym_dot = tangents
  # print("Using custom jvp rule")
  primal_out = conv_general_dilated(x, kernel, strides, padding_lax, lhs_dilation=lhs_dilation, rhs_dilation=rhs_dilation, dimension_numbers=dimension_numbers, feature_group_count=feature_group_count, precision=precision)
  T1 = conv_general_dilated(x_dot, kernel_asym, strides, padding_lax, lhs_dilation=lhs_dilation, rhs_dilation=rhs_dilation, dimension_numbers=dimension_numbers, feature_group_count=feature_group_count, precision=precision)
  T2 = conv_general_dilated(x, kernel_dot, strides, padding_lax, lhs_dilation=lhs_dilation, rhs_dilation=rhs_dilation, dimension_numbers=dimension_numbers, feature_group_count=feature_group_count, precision=precision)

  T_KP = conv_general_dilated(x, kernel_asym_dot, strides, padding_lax, lhs_dilation=lhs_dilation, rhs_dilation=rhs_dilation, dimension_numbers=dimension_numbers, feature_group_count=feature_group_count, precision=precision)
 
  tangent_out = T1 + T2 + T_KP
  return primal_out, tangent_out

#--------------------------------------------------
# end custom random feedback Conv(_Local) layer
#--------------------------------------------------      