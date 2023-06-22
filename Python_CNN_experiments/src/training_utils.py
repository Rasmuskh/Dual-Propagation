import jax
from jax import vmap # for parallelizing operations accross multiple datapoints (and managing rng across them)
import jax.numpy as jnp # JAX NumPy

from flax.training import train_state # Useful dataclass to keep train state
import dm_pix as pix # for data augmentation

import numpy as np # regular old numpy
import optax # Optimizers
import tensorflow_datasets as tfds # TFDS for loading datasets
import time, datetime # measuring runetime and generating timestamps for experiments
import os # for os.makdirs() function

def cross_entropy_loss(*, logits, labels):
    num_classes = logits.shape[1]
    labels_onehot = jax.nn.one_hot(labels, num_classes=num_classes)
    return optax.softmax_cross_entropy(logits=logits, labels=labels_onehot).mean()
    # return optax.l2_loss(logits, targets=labels_onehot).mean()

def compute_metrics(*, logits, labels):
    loss = cross_entropy_loss(logits=logits, labels=labels)
    accuracy = 100*jnp.mean(jnp.argmax(logits, -1) == labels)

    _, top5_indices = jax.lax.top_k(logits, 5) 
    top5accuracy = 100*(top5_indices == labels[:,None]).sum()/logits.shape[0]

    metrics = {'loss': loss, 'accuracy': accuracy, 'top5accuracy':top5accuracy}
    return metrics

def get_cifar10():
    """Load train and test datasets into memory."""
    ds_builder = tfds.builder('cifar10')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train[:90%]', batch_size=-1))
    val_ds = tfds.as_numpy(ds_builder.as_dataset(split='train[90%:]', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    train_ds['image'] = jnp.float32(train_ds['image']) / 255.
    val_ds['image'] = jnp.float32(val_ds['image']) / 255.
    test_ds['image'] = jnp.float32(test_ds['image']) / 255.
    # some datasets don't have an id key, so if we loaded a 
    # dataset that does have an id key we throw it away
    train_ds.pop('id', None)
    val_ds.pop('id', None)
    test_ds.pop('id', None)

    mean_data = jnp.expand_dims(jnp.array([0.4914, 0.4822, 0.4465]), axis=(0,1))
    std_data = jnp.expand_dims(jnp.array([0.2023, 0.1994, 0.2010]), axis=(0,1))
    train_ds['image'] = (train_ds['image'] - mean_data)/std_data
    val_ds['image'] = (val_ds['image'] - mean_data)/std_data
    test_ds['image'] = (test_ds['image'] - mean_data)/std_data

    return train_ds, val_ds, test_ds

def get_cifar100():
    """Load train and test datasets into memory."""
    ds_builder = tfds.builder('cifar100')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train[:90%]', batch_size=-1))
    val_ds = tfds.as_numpy(ds_builder.as_dataset(split='train[90%:]', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    train_ds['image'] = jnp.float32(train_ds['image']) / 255.
    val_ds['image'] = jnp.float32(val_ds['image']) / 255.
    test_ds['image'] = jnp.float32(test_ds['image']) / 255.
    # some datasets don't have an id key, so if we loaded a 
    # dataset that does have an id key we throw it away
    train_ds.pop('id', None)
    val_ds.pop('id', None)
    test_ds.pop('id', None)
    train_ds.pop('coarse_label', None)
    val_ds.pop('coarse_label', None)
    test_ds.pop('coarse_label', None)

    mean_data = jnp.expand_dims(jnp.array([0.5074,0.4867,0.4411]), axis=(0,1))
    std_data = jnp.expand_dims(jnp.array([0.2011,0.1987,0.2025]), axis=(0,1))
    train_ds['image'] = (train_ds['image'] - mean_data)/std_data
    val_ds['image'] = (val_ds['image'] - mean_data)/std_data
    test_ds['image'] = (test_ds['image'] - mean_data)/std_data

    return train_ds, val_ds, test_ds

def get_imagenet_32x32(batch_size, percent_train=95):
    """Load train and test datasets into memory."""
    ds_builder = tfds.builder('imagenet_resized/32x32')
    ds_builder.download_and_prepare()
    train_ds = ds_builder.as_dataset(split=f'train[:{percent_train}%]', batch_size=batch_size, shuffle_files=True)
    val_ds = tfds.as_numpy(ds_builder.as_dataset(split='train[95%:]', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='validation', batch_size=-1))

    # train_ds['image'] = jnp.float32(train_ds['image']) / 255.
    val_ds['image'] = jnp.float32(val_ds['image']) / 255.
    test_ds['image'] = jnp.float32(test_ds['image']) / 255.
    mean_data = jnp.expand_dims(jnp.array([0.485, 0.456, 0.406]), axis=(0,1))
    std_data = jnp.expand_dims(jnp.array([0.229, 0.224, 0.225]), axis=(0,1))
    val_ds['image'] = (val_ds['image'] - mean_data)/std_data
    test_ds['image'] = (test_ds['image'] - mean_data)/std_data

    return train_ds, val_ds, test_ds

def create_train_state(rng, model, warmup_learning_rate, warmup_epochs, learning_rate, momentum, weight_decay, num_epochs, steps_per_epoch, batch_size):
    """Creates initial `TrainState`."""
    x = jnp.ones([1, 32, 32, 3])
    params = model.init(rng, x)['params']

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=warmup_learning_rate, 
        peak_value=learning_rate, 
        warmup_steps=warmup_epochs*steps_per_epoch, 
        decay_steps = (num_epochs)*steps_per_epoch, 
        end_value=0.0)

    tx = optax.chain( 
        optax.add_decayed_weights(weight_decay=weight_decay, mask=None),
        optax.sgd(schedule, momentum=momentum),
        )

    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def update_train_state(model, state, warmup_epochs, learning_rate, momentum_updated, weight_decay, num_epochs, steps_per_epoch):
    """update `TrainState`."""
    
    schedule = optax.cosine_decay_schedule(learning_rate, (num_epochs - warmup_epochs)*steps_per_epoch, alpha=0.0)

    tx = optax.chain( 
        optax.add_decayed_weights(weight_decay=weight_decay, mask=None),
        optax.sgd(schedule, momentum=momentum_updated),
        )

    return train_state.TrainState.create(apply_fn=model.apply, params=state.params, tx=tx)

def augment_train(image, batch_rng):
    w, h, c = image.shape

    # Random crop
    image = jnp.pad(image, ((4,4), (4,4), (0,0)), 'constant', constant_values=((0,0), (0, 0), (0,0)))
    image = pix.random_crop(batch_rng, image, (w, h, c))
    # Horizonthal flip
    image = pix.random_flip_left_right(batch_rng, image, probability=0.5)

    return image

vmap_augment_train = vmap(augment_train, in_axes=(0, 0))

def augment_train_imagenet_32x32(image, batch_rng):
    w, h, c = image.shape

    image = jnp.float32(image) / 255.
    mean_data = jnp.expand_dims(jnp.array([0.485, 0.456, 0.406]), axis=(0,1))
    std_data = jnp.expand_dims(jnp.array([0.229, 0.224, 0.225]), axis=(0,1))
    image = (image - mean_data)/std_data

    # Random crop
    image = jnp.pad(image, ((4,4), (4,4), (0,0)), 'constant', constant_values=((0,0), (0, 0), (0,0)))
    image = pix.random_crop(batch_rng, image, (w, h, c))
    # Horizonthal flip
    image = pix.random_flip_left_right(batch_rng, image, probability=0.5)

    return image

vmap_augment_train_imagenet_32x32 = vmap(augment_train_imagenet_32x32, in_axes=(0, 0))

@jax.jit
def train_step(state, batch, batch_rng):
    """Train for a single step."""
    batch_rng = jax.random.split(batch_rng, batch['image'].shape[0])
    
    # batch['image'] should have dims batchsize, w, h, ch
    batch['image'] = vmap_augment_train(batch['image'], batch_rng)
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['image'])
        loss = cross_entropy_loss(logits=logits, labels=batch['label'])
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)

    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits=logits, labels=batch['label'])
    return state, metrics

def train_epoch(state, train_ds, batch_size, rng):
    """Train for a single epoch."""
    t0 = time.time()
    train_ds_size = len(train_ds['image'])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, train_ds_size)
    perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))
    batch_metrics = []
    for perm in perms:
        batch = {k: v[perm, ...] for k, v in train_ds.items()}
        rng, batch_rng = jax.random.split(rng)

        state, metrics = train_step(state, batch, batch_rng)
        batch_metrics.append(metrics)

    # compute mean of metrics across each batch in epoch.
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]}
    runtime = time.time() - t0

    return state, epoch_metrics_np['loss'], epoch_metrics_np['accuracy'], runtime

@jax.jit
def train_step_imagenet_32x32(state, batch, batch_rng):
    """Train for a single step."""
    batch_rng = jax.random.split(batch_rng, batch['image'].shape[0])

    
    # batch['image'] should have dims batchsize, w, h, ch
    batch['image'] = vmap_augment_train_imagenet_32x32(batch['image'], batch_rng)
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['image'])
        loss = cross_entropy_loss(logits=logits, labels=batch['label'])
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits=logits, labels=batch['label'])
    return state, metrics

def train_epoch_imagenet32x32(state, train_ds, batch_size, rng):
    """Train for a single epoch."""
    t0 = time.time()
    batch_metrics = []

    shuffle_seed = jax.random.randint(rng,(1,), 1 ,1000000)[0]
    train_ds = train_ds.shuffle(1000, seed=shuffle_seed)
    rng, batch_rng = jax.random.split(rng)
    for batch in train_ds:
        batch = tfds.as_numpy(batch)
        rng, batch_rng = jax.random.split(rng)
        state, metrics = train_step_imagenet_32x32(state, batch, batch_rng)
        batch_metrics.append(metrics)

    # compute mean of metrics across each batch in epoch.
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]}
    runtime = time.time() - t0

    return state, epoch_metrics_np['loss'], epoch_metrics_np['accuracy'], runtime


@jax.jit
def eval_step(state, params, batch):
    logits = state.apply_fn({'params': params}, batch['image'])
    metrics = compute_metrics(logits=logits, labels=batch['label'])
    return metrics

def eval_model(state, params, test_ds, batch_size):
    t0 = time.time()
    test_ds_size = len(test_ds['image'])
    steps = test_ds_size // batch_size
    indices = jnp.arange(0, test_ds_size)
    indices = indices[:steps * batch_size]  # skip incomplete batch
    indices = indices.reshape((steps, batch_size))
    batch_metrics = []

    for idx in indices:
        batch = {k: v[idx, ...] for k, v in test_ds.items()}
        metrics = eval_step(state, params, batch)
        batch_metrics.append(metrics)

    batch_metrics_np = jax.device_get(batch_metrics)
    test_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]}
    summary = jax.tree_util.tree_map(lambda x: x.item(), test_metrics_np)

    runtime = time.time() - t0
    return summary['loss'], summary['accuracy'], summary['top5accuracy'], runtime#, test_grads_np

def get_cosine_sim(v1, v2):
    cosine_sim = jnp.sum(v1*v2)/(jnp.linalg.norm(v1)*jnp.linalg.norm(v2))
    return cosine_sim

