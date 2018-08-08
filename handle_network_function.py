import tensorflow as tf

slim = tf.contrib.slim


def _get_init_fn(checkpoint_path, train_dir, checkpoint_exclude_scopes, ignore_missing_vars='False'):
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
		exclusions = [scope.strip() for scope in checkpoint_exclude_scopes.split(',')]

	print('excluded scopes : ', exclusions)

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

	print('restored scopes : ', variables_to_restore)

	if tf.gfile.IsDirectory(checkpoint_path):
		checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
	else:
		checkpoint_path = checkpoint_path

	# print(checkpoint_path)

	tf.logging.info('Fine-tuning from %s' % checkpoint_path)

	return slim.assign_from_checkpoint_fn(
		checkpoint_path,
		variables_to_restore,
		ignore_missing_vars=ignore_missing_vars)


def _get_variables_to_train(trainable_scopes):
	"""Returns a list of variables to train.
	Returns:
	A list of variables to train by the optimizer.
	"""
	if trainable_scopes is None:
		return tf.trainable_variables()
	else:
		scopes = [scope.strip() for scope in trainable_scopes.split(',')]

	variables_to_train = []
	for scope in scopes:
		variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
		variables_to_train.extend(variables)
	return variables_to_train

