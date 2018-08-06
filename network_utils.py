### Network utilities
# This file includes all function to handle neural networks
# Specifically, functions to control learning rate, to configure optimizer (Adam, SGD, etc.)
# to initiate network, or get variables to learn


import tensorflow as tf
from tensorflow.contrib import slim as slim

def _configure_learning_rate(init_learning_rate, batch_size, num_samples_per_epoch, global_step,
                             num_epochs_per_decay=1.0, decay_rate=0.95, sync_replicas=False,
                             learning_rate_decay_type = 'exponential', end_learning_rate = 0.0001,
                             replicas_to_aggregate = 1):
  """Configures the learning rate.

  Args:
    num_samples_per_epoch: The number of samples in each epoch of training.
    global_step: The global_step tensor.

  Returns:
    A `Tensor` representing the learning rate.

  Raises:
    ValueError: if
  """
  decay_steps = int(num_samples_per_epoch / batch_size *
                    num_epochs_per_decay)
  if sync_replicas:
    decay_steps /= replicas_to_aggregate

  if learning_rate_decay_type == 'exponential':
    return tf.train.exponential_decay(init_learning_rate,
                                      global_step,
                                      decay_steps,
                                      decay_rate,
                                      staircase=False,
                                      name='exponential_decay_learning_rate')
  elif learning_rate_decay_type == 'fixed':
    return tf.constant(init_learning_rate, name='fixed_learning_rate')
  elif learning_rate_decay_type == 'polynomial':
    return tf.train.polynomial_decay(init_learning_rate,
                                     global_step,
                                     decay_steps,
                                     end_learning_rate,
                                     power=1.0,
                                     cycle=False,
                                     name='polynomial_decay_learning_rate')
  else:
    raise ValueError('learning_rate_decay_type [%s] was not recognized',
                     learning_rate_decay_type)


def _configure_optimizer(learning_rate, optimizer = 'adam',
                        adadelta_rho = 0.95, adagrad_initial_accumulator_value = 0.1,
                        adam_beta1 = 0.9, adam_beta2 = 0.999, opt_epsilon = 1.0,
                        ftrl_learning_rate_power = -0.5, ftrl_initial_accumulator_value = 0.1,
                        ftrl_l1 = 0.0, ftrl_l2 = 0.0, momentum = 0.9,
                        rmsprop_momentum = 0.9, rmsprop_decay = 0.9):
  """Configures the optimizer used for training.

  Args:
    learning_rate: A scalar or `Tensor` learning rate.

  Returns:
    An instance of an optimizer.

  Raises:
    ValueError: if optimizer is not recognized.
  """
  if optimizer == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(
        learning_rate,
        rho=adadelta_rho,
        epsilon=opt_epsilon)
  elif optimizer == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(
        learning_rate,
        initial_accumulator_value=adagrad_initial_accumulator_value)
  elif optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1=adam_beta1,
        beta2=adam_beta2,
        epsilon=opt_epsilon)
  elif optimizer == 'ftrl':
    optimizer = tf.train.FtrlOptimizer(
        learning_rate,
        learning_rate_power=ftrl_learning_rate_power,
        initial_accumulator_value=ftrl_initial_accumulator_value,
        l1_regularization_strength=ftrl_l1,
        l2_regularization_strength=ftrl_l2)
  elif optimizer == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        learning_rate,
        momentum=momentum,
        name='Momentum')
  elif optimizer == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay=rmsprop_decay,
        momentum=rmsprop_momentum,
        epsilon=opt_epsilon)
  elif optimizer == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  else:
    raise ValueError('Optimizer [%s] was not recognized', optimizer)
  return optimizer


def precision_at_k(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def _get_init_fn(checkpoint_path, train_dir=None, checkpoint_exclude_scopes=None,
                 ignore_missing_vars=True):
  """Returns a function run by the chief worker to warm-start the training.

  Note that the init_fn is only run when initializing the model during the very
  first global step.

  Returns:
    An init function run by the supervisor.
  """
  if checkpoint_path is None:
    return None

  # Warn the user if a checkpoint exists in the train_dir. Then we'll be
  # ignoring the checkpoint anyway.
  if tf.train.latest_checkpoint(train_dir):
    tf.logging.info(
        'Ignoring --checkpoint_path because a checkpoint already exists in %s'
        % train_dir)
    return None

  exclusions = []
  if checkpoint_exclude_scopes:
    exclusions = [scope.strip()
                  for scope in checkpoint_exclude_scopes.split(',')]

  # TODO(sguada) variables.filter_variables()
  variables_to_restore = []
  for var in slim.get_model_variables():
    excluded = False
    for exclusion in exclusions:
      if var.op.name.startswith(exclusion):
        excluded = True
        break
    if not excluded:
      variables_to_restore.append(var)

  if tf.gfile.IsDirectory(checkpoint_path):
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
  else:
    checkpoint_path = checkpoint_path

  tf.logging.info('Fine-tuning from %s' % checkpoint_path)

  return slim.assign_from_checkpoint_fn(
      checkpoint_path,
      variables_to_restore,
      ignore_missing_vars=ignore_missing_vars)


def _get_variables_to_train(trainable_scopes=None):
  """Returns a list of variables to train.

  Returns:
    A list of variables to train by the optimizer.
  """
  if trainable_scopes is None: # train all the network
    return tf.trainable_variables()
  else:
    scopes = [scope.strip() for scope in trainable_scopes.split(',')]

  variables_to_train = []
  for scope in scopes:
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    variables_to_train.extend(variables)
  return variables_to_train
