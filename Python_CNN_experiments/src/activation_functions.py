import jax
import jax.numpy as jnp
from jax import custom_vjp
from typing import Optional
import numpy as np

# Custom dualprop relu function
@custom_vjp
def relu_dualprop(a):
    z = jnp.maximum(0, a)
    return z

# forward pass for use with autodiff. first return value is just f(a). 
# second return value is information needed by the backwards pass
# f_fwd :: a -> (z, (a,z))
def relu_dualprop_fwd(a):
    z = relu_dualprop(a)
    return z, (a, z)

# f_bwd :: (a_and_z, dy) -> dz
def relu_dualprop_bwd(a_and_z, dy):
    beta = 1
    a, _ = a_and_z
    zplus = jnp.maximum(0, a + beta*dy)
    zminus = jnp.maximum(0, a - beta*dy)
    dz = (zplus - zminus)/(2*beta)
    return (dz,)

relu_dualprop.defvjp(relu_dualprop_fwd, relu_dualprop_bwd)