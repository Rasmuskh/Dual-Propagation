import jax
import jax.numpy as jnp # JAX NumPy
from flax.linen import Conv, ConvLocal, Dense # some layers
from flax.training import checkpoints
import numpy as np # regular old numpy
import time, datetime # measuring runtime and generating timestamps for experiments
import os # for os.makdirs() function
import absl  # for logging
from absl import logging # for logging
# Training utils
from src import update_train_state, create_train_state, train_epoch, train_epoch_imagenet32x32, eval_model, get_cosine_sim

# Import configurations
# import config # Use this for the old method
from config.cli_config import config

experiment_dir = "./runs/" + config.experiment_name + "/"
os.makedirs(experiment_dir)

for experiment_index, seed in enumerate(config.seeds):
    # path to save stuff to
    timestamp = datetime.datetime.fromtimestamp(time.time())
    outpath = experiment_dir + timestamp.strftime('%Y_%m_%d_%H_%M_%S')+'/'
    os.makedirs(outpath)
    CKPT_DIR = outpath+'ckpts/'
    os.makedirs(CKPT_DIR)

    logging.use_absl_handler()
    logging.get_absl_handler().use_absl_log_file('absl_logging', outpath) 
    absl.flags.FLAGS.mark_as_parsed() 
    logging.set_verbosity(logging.INFO)

    def loginfo_and_print(msg):
        logging.info(msg)
        print(msg)
    
    loginfo_and_print(f"\n\tStarting experiment {experiment_index+1}/{len(config.seeds)}. Current seed is {seed}")
    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)
    # Define model
    if config.dataset == 'imagenet':
        steps_per_epoch = config.train_ds.cardinality().numpy()
    else:
        steps_per_epoch = len(config.train_ds['image']) // config.batch_size
    state = create_train_state(init_rng, config.model, config.warmup_learning_rate, config.warmup_epochs, config.learning_rate, config.momentum, config.weight_decay, config.num_epochs, steps_per_epoch, config.batch_size)
    del init_rng  # Must not be used anymore.

    # Dict for storing training metrics in
    hist = {'val_loss': np.zeros(config.num_epochs), 'val_accuracy': np.zeros(config.num_epochs), 'val_top5accuracy': np.zeros(config.num_epochs), 'val_time': np.zeros(config.num_epochs),
            'train_loss': np.zeros(config.num_epochs), 'train_accuracy': np.zeros(config.num_epochs), 'train_time': np.zeros(config.num_epochs),
             'test_loss': np.nan, 'test_accuracy': np.nan, 'test_top5accuracy': np.nan, 'test_time': np.nan,
            'weight_cos_sim_l0': np.zeros(config.num_epochs)}#, 'grad_cos_sim_l0': np.zeros(num_epochs)}

    best_accuracy, best_epoch = 0, 0
    epoch = 0

    for epoch in range(1, config.num_epochs+1):
        loginfo_and_print("\n========================== Epoch: %d ==========================" % (epoch))

        if (epoch == (config.warmup_epochs + 1)) and (config.momentum_updated != None):
            loginfo_and_print("Updating optimizer")
            state = update_train_state(config.model, state, config.warmup_epochs, config.learning_rate, config.momentum_updated, config.weight_decay, config.num_epochs, steps_per_epoch)

        # Use a separate PRNG key to permute image data during shuffling
        rng, input_rng = jax.random.split(rng)
        # Run an optimization step over a training batch
        if config.dataset == 'imagenet':
            state, train_loss, train_accuracy, train_time = train_epoch_imagenet32x32(state, config.train_ds, config.batch_size, input_rng)
        else:
            state, train_loss, train_accuracy, train_time = train_epoch(state, config.train_ds, config.batch_size, input_rng)
        loginfo_and_print('train: \tloss: %.4f, \taccuracy: %.2f, \truntime: %.2f' % (train_loss, train_accuracy, train_time))
        hist['train_loss'][epoch-1], hist['train_accuracy'][epoch-1], hist['train_time'][epoch-1] = train_loss, train_accuracy, train_time
        
        # Evaluate on the validation set after each training epoch 
        val_loss, val_accuracy, val_top5_accuracy, val_time = eval_model(state, state.params, config.val_ds, config.batch_size)
        loginfo_and_print('val:  \tloss: %.4f, \taccuracy: %.2f, \ttop5_accuracy: %.2f, \truntime: %.2f' % (val_loss, val_accuracy, val_top5_accuracy, val_time))
        hist['val_loss'][epoch-1], hist['val_accuracy'][epoch-1], hist['val_top5accuracy'][epoch-1], hist['val_time'][epoch-1] = val_loss, val_accuracy, val_top5_accuracy, val_time

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch
            loginfo_and_print("Saving new checkpoint")
            checkpoints.save_checkpoint(ckpt_dir=CKPT_DIR, target=state, step=epoch, keep=config.num_epochs)

        if  'kernel_asym' in state.params['c0'].keys():
            weight_cos_sim_l0 = get_cosine_sim(state.params['c0']['kernel'], state.params['c0']['kernel_asym'])
            loginfo_and_print('    ⤷ Conv0:\tkernel cosine_sim: %.6f' % ( weight_cos_sim_l0))
            hist['weight_cos_sim_l0'][epoch-1] = weight_cos_sim_l0
 
    loginfo_and_print(f"\n====Loading model with best validation accuracy (epoch {best_epoch})====")
    best_state = checkpoints.restore_checkpoint(ckpt_dir=CKPT_DIR, target=state)
    test_loss, test_accuracy, test_top5accuracy, test_time = eval_model(best_state, best_state.params, config.test_ds, config.batch_size)
    hist['test_loss'], hist['test_accuracy'], hist['test_top5accuracy'], hist['test_time'] = test_loss, test_accuracy, test_top5accuracy, test_time
    loginfo_and_print('test:  \tloss: %.4f, \taccuracy: %.2f, \ttop5_accuracy: %.2f, \truntime: %.2f' % (test_loss, test_accuracy, test_top5accuracy, test_time))
    np.save(outpath+"hist.npy", hist)