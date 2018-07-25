import tensorflow as tf
import numpy as np
import partial_vgg
import partial_resnet
import network_utils

slim = tf.contrib.slim

class taxonomy_model(object):
	def __init__(self,
				 inputs,
				 true_labels,
	             input_size,
	             batch_size,
	             taxonomy_nums,
	             taxonomy_classes,
	             resnet_version,
	             resnet_pretrained_path=None,
	             resnet_exclude=None,
	             trainable_scopes='',
	             extra_global_feature=None,
	             taxonomy_loss=None,
				 learning_rate=0.001,
				 num_samples = None,
				 beta=0.02,
	             taxonomy_weights=[1.0, 1.0, 1.0, 1.0],
	             all_label=None,
	             all_value=None,
	             batch_weight_range=[0.5, 1.0],
	             is_mode='train',
	             restore_path=''):

		self.input_size = input_size
		self.batch_size = batch_size
		self.taxonomy_nums = taxonomy_nums
		self.taxonomy_classes = taxonomy_classes
		self.resnet_version = resnet_version
		self.extra_global_feature = extra_global_feature
		self.taxonomy_loss = taxonomy_loss
		self.is_mode = is_mode
		self.resnet_pretrained_path = resnet_pretrained_path
		self.resnet_exclude = resnet_exclude
		self.trainable_scopes = trainable_scopes
		self.learning_rate = learning_rate
		self.num_samples = num_samples if num_samples is not None else 500*batch_size
		self.beta = beta
		self.taxonomy_weights = taxonomy_weights
		self.all_label = all_label
		self.all_value = all_value
		self.batch_weight_range = batch_weight_range
		self.restore_path = restore_path

		# self.input = tf.placeholder(tf.float32,
		#                             shape=(self.batch_size,
		#                                    self.input_size[0],
		#                                    self.input_size[1], 3))

		self.resnet_training_flag = tf.placeholder(tf.bool)

		self.vgg19_training_flag = tf.placeholder(tf.bool)

		self.vgg_dropout = tf.placeholder(tf.float32)

		# self.true_labels = tf.placeholder(tf.int64)
		self.input = inputs

		self.true_labels = true_labels

	def resnet_for_global(self):
		if self.resnet_version == 50:
			with slim.arg_scope(partial_resnet.resnet_arg_scope()):
				resnet_partial, resnet_partial_mid = partial_resnet.resnet_v2_partial_50(inputs=self.input,
				                                                                      num_classes=None,
				                                                                      is_training=self.resnet_training_flag,
				                                                                      global_pool=True,
				                                                                      output_stride=None,
				                                                                      spatial_squeeze=True,
				                                                                      reuse=None,
				                                                                      scope='resnet_v2_50',
				                                                                      include_root_block=True)
		elif self.resnet_version == 101:
			with slim.arg_scope(partial_resnet.resnet_arg_scope()):
				resnet_partial, resnet_partial_mid = partial_resnet.resnet_v2_partial_101(inputs=self.input,
				                                                                      num_classes=None,
				                                                                      is_training=self.resnet_training_flag,
				                                                                      global_pool=True,
				                                                                      output_stride=None,
				                                                                      spatial_squeeze=True,
				                                                                      reuse=None,
				                                                                      scope='resnet_v2_101',
				                                                                      include_root_block=True)
		elif self.resnet_version == 152:
			with slim.arg_scope(partial_resnet.resnet_arg_scope()):
				resnet_partial, resnet_partial_mid = partial_resnet.resnet_v2_partial_152(inputs=self.input,
				                                                                      num_classes=None,
				                                                                      is_training=self.resnet_training_flag,
				                                                                      global_pool=True,
				                                                                      output_stride=None,
				                                                                      spatial_squeeze=True,
				                                                                      reuse=None,
				                                                                      scope='resnet_v2_152',
				                                                                      include_root_block=True)

		if self.is_mode == 'train':
			resnet_restore = network_utils._get_init_fn(self.resnet_pretrained_path, '',
			                                                      self.resnet_exclude)
			return resnet_partial_mid, resnet_partial, resnet_restore
		else:
			return resnet_partial_mid, resnet_partial

	def partial_vgg_stage(self, input, num_classes, scope='', num_filters=4096, extra_global_features=None, end_global=None):

		with slim.arg_scope(partial_vgg.vgg_arg_scope()):
			vgg19_partial = partial_vgg.vgg_partial(inputs=input,
			                                        num_classes=None,
			                                        is_training=True,
			                                        dropout_keep_prob=self.vgg_dropout,
			                                        spatial_squeeze=True,
			                                        scope=scope,
			                                        fc_conv_padding='VALID',
			                                        global_pool=False
			                                        )

			with tf.variable_scope(scope, 'vgg_19', [vgg19_partial]) as sc:
				end_points_collection = sc.original_name_scope + '_end_points'
				if extra_global_features is not None:
					resnet_endpoint_for_vgg_conv_1 = slim.conv2d(end_global, 1024, [3, 3], padding='VALID', scope='resnet_to_vgg_conv_1')
					resnet_endpoint_for_vgg_conv_2 = slim.conv2d(resnet_endpoint_for_vgg_conv_1, 512, [3, 3], padding='VALID', scope='resnet_to_vgg_conv_2')
					vgg_out = tf.concat([vgg19_partial, resnet_endpoint_for_vgg_conv_2], axis=3)
				else:
					vgg_out = vgg19_partial

				net = slim.conv2d(vgg_out, num_filters, [3, 3], padding='VALID', scope='fc6')
				net = slim.dropout(net, self.vgg_dropout, is_training=self.vgg19_training_flag,
								 scope='dropout6')
				net = slim.conv2d(net, num_filters, [1, 1], scope='fc7')
				end_points = slim.utils.convert_collection_to_dict(end_points_collection)
				net = slim.dropout(net, self.vgg_dropout, is_training=self.vgg19_training_flag,
				                   scope='dropout7')
				net = slim.conv2d(net, num_classes, [1, 1],
				                  activation_fn=None,
				                  normalizer_fn=None,
				                  scope='fc8')

				# end_points = slim.utils.convert_collection_to_dict(end_points_collection)
				# net = slim.conv2d(vgg_out, num_classes, [3, 3], padding='VALID',
				# 				  activation_fn=None,
				# 				  normalizer_fn=None,
				# 				  scope='fc8')

				## net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
				end_points[sc.name + '/fc8'] = net
				end_points[sc.name + '/fc8/squeezed'] = tf.squeeze(net, [1, 2], name='fc8/squeezed')

				net = slim.conv2d(net, 512, [1, 1], scope='fc9')
				net = tf.tile(net, [1, 7, 7, 1], name='tile')
				end_points[sc.name + '/repeat'] = net

		return end_points

	def build_model(self):

		# Create global_step for handling learning rate
		self.global_step = tf.train.create_global_step()

		if self.is_mode == 'train':
			self.resnet_partial_mid, self.resnet_partial, self.resnet_restore = self.resnet_for_global()
		else:
			self.resnet_partial_mid, self.resnet_partial = self.resnet_for_global()

		self.stage_inputs = []
		self.stage_outputs = []

		self.stage_inputs.append(self.resnet_partial_mid)

		with slim.arg_scope(partial_vgg.vgg_arg_scope()):
			for i in range(0, self.taxonomy_nums):
				self.stage_outputs.append(self.partial_vgg_stage(self.stage_inputs[i],
				                                                 num_classes=self.taxonomy_classes[i],
				                                                 scope=(('stage_%d') % i),
																 num_filters=int(4096), #int(4096/(i*i+1)/10),
				                                                 extra_global_features=self.extra_global_feature,
				                                                 end_global=self.resnet_partial))

				if i < (self.taxonomy_nums - 1):
					with tf.variable_scope((('stage_%d') % i)) as sc:
						next_input_from_global = slim.conv2d(self.resnet_partial_mid, 512, [1, 1], scope='trans_fc')
						next_input_from_stage = self.stage_outputs[i][(('stage_%d') % i) + '/repeat']

						next_input = tf.concat([next_input_from_global, next_input_from_stage], axis=3)

						self.stage_inputs.append(next_input)

		self.net_losses, self.eval_values, self.eval_updates, self.reset_op = [], [], [], []
		if self.is_mode == 'train':

			self.net_losses = self.get_loss()

			self.update_flag = self.set_optimizer()

			self.eval_values, self.eval_updates, self.reset_op = self.get_metrics()
		else:
			self.restore_net = network_utils._get_init_fn(self.restore_path, '', '')

			self.get_loss_for_test()

		return self.stage_inputs, self.stage_outputs, self.net_losses, \
			   self.eval_values, self.eval_updates, self.reset_op

	def make_each_portion_batch(self, batch_label, all_label, all_value, name_order, min_per=0.5, max_per=1.0):
		batch_portion = np.zeros(shape=(np.shape(batch_label)[0], ), dtype=np.float32)

		max_image_num = np.max(all_value)
		min_image_num = np.min(all_value)

		for i in range(0, np.shape(batch_label)[0]):
			for j in range(0, np.shape(all_label)[0]):
				if batch_label[i, name_order] == all_label[j]:
					temp_ratio = np.divide((all_value[j] - min_image_num),
					                       (max_image_num - min_image_num), dtype=np.float32)
					batch_portion[i] = max_per - (max_per - min_per) * temp_ratio
					break

		return batch_portion

	def get_loss_for_test(self):
		with tf.variable_scope('losses'):
			for i in range(0, self.taxonomy_nums):
				if self.taxonomy_loss is not None:
					prior_taxonomy = tf.ones(shape=(tf.shape(self.stage_outputs[i][('stage_%d/fc8/squeezed') % i])[0],),
					                         dtype=tf.float32)
					for j in range(0, i):
						arg_max_ind = tf.squeeze(tf.argmax(self.stage_outputs[j][('stage_%d/fc8/squeezed') % j], axis=1))
						mask_ind = tf.one_hot(arg_max_ind,
						                      tf.shape(self.stage_outputs[j][('stage_%d/fc8/squeezed') % j])[1])
						masked = tf.boolean_mask(self.stage_outputs[j][('stage_%d/fc8/squeezed') % j], mask_ind)
						prior_taxonomy = tf.multiply(prior_taxonomy, masked)
					self.stage_outputs[i][('stage_%d/fc8/squeezed_tax') % i] = tf.multiply(
						tf.tile(tf.reshape(prior_taxonomy,
						                   [tf.shape(self.stage_outputs[i][('stage_%d/fc8/squeezed') % i])[0], 1]),
					                                  [1,
					                                   tf.shape(self.stage_outputs[i][('stage_%d/fc8/squeezed') % i])[1]]),
					                          self.stage_outputs[i][('stage_%d/fc8/squeezed') % i])

	def get_loss(self):

		# This weights for only 'name' class.
		# According to image number of each class, weights will be set
		# The number of 'name' class is : '7'
		# Only activating when image number information published
		if self.all_label is not None and self.all_value is not None:
			loss_weights = tf.py_func(self.make_each_portion_batch, [self.true_labels,
			                                                         self.all_label[7],
			                                                         self.all_value[7],
			                                                         7,
			                                                         self.batch_weight_range[0],
			                                                         self.batch_weight_range[1]],
			                          [tf.float32])
			loss_weights = tf.reshape(loss_weights, shape=[tf.shape(self.true_labels)[0], 1])
		else:
			loss_weights = 1.0

		if self.is_mode == 'train':

			net_losses = {}
			sum_losses = []
			with tf.variable_scope('losses'):
				for i in range(0, self.taxonomy_nums):
					if self.taxonomy_loss is not None:
						prior_taxonomy = tf.ones(shape=(tf.shape(self.stage_outputs[i][('stage_%d/fc8/squeezed') % i])[0], ), dtype=tf.float32)
						for j in range(0, i):
							mask_ind = tf.one_hot(self.true_labels[:, j], tf.shape(self.stage_outputs[j][('stage_%d/fc8/squeezed') % j])[1])
							masked = tf.boolean_mask(self.stage_outputs[j][('stage_%d/fc8/squeezed') % j], mask_ind)
							prior_taxonomy = tf.multiply(prior_taxonomy, masked)
						inout_logit = tf.multiply(tf.tile(tf.reshape(prior_taxonomy, [tf.shape(self.stage_outputs[i][('stage_%d/fc8/squeezed') % i])[0], 1]),
						                                  [1, tf.shape(self.stage_outputs[i][('stage_%d/fc8/squeezed') % i])[1]]),
						                          self.stage_outputs[i][('stage_%d/fc8/squeezed') % i])
					else:
						inout_logit = self.stage_outputs[i][('stage_%d/fc8/squeezed') % i]
					net_losses['stage_%d' % i] = tf.losses.sparse_softmax_cross_entropy(
						labels=self.true_labels[:, i],
						logits=inout_logit,
						weights=loss_weights, scope='loss_%s' % i)
					# sum_losses.append(net_losses['stage_%d' % i])

				for i in range(self.taxonomy_nums-1, -1, -1):
					if i == self.taxonomy_nums-1:
						net_losses['stage_%d_weighted' % i] = self.taxonomy_weights[i] * net_losses['stage_%d' % i]
					else:
						net_losses['stage_%d_weighted' % i] = net_losses['stage_%d_weighted' % (i + 1)] + \
						                                      self.taxonomy_weights[i] * net_losses['stage_%d' % i]
					sum_losses.append(net_losses['stage_%d_weighted' % i])

				net_losses['all'] = tf.divide(tf.add_n(sum_losses), self.taxonomy_nums)

		else:
			net_losses = {}
			with tf.variable_scope('losses'):
				net_losses['all'] = tf.losses.sparse_softmax_cross_entropy(
					labels=tf.squeeze(self.true_labels),
					logits=self.stage_outputs[3][('stage_%d/fc8/squeezed') % 3],
					weights=loss_weights, scope='loss_%s' % 3)

		return net_losses

	def get_metrics(self):

		net_acc1, net_acc_val1 = {}, {}
		net_acc3, net_acc_val3 = {}, {}
		net_prec1_val, net_prec1 = {}, {}
		net_prec3_val, net_prec3 = {}, {}
		net_recall1_val, net_recall1 = {}, {}
		net_recall3_val, net_recall3 = {}, {}

		with tf.variable_scope('metrics') as scope:

			if self.is_mode == 'train':
				consider_layers = range(0, self.taxonomy_nums)
			else:
				consider_layers = [self.taxonomy_nums - 1]

			for i in consider_layers:
				# transform the last labels to calculate metrics (precision, recall)
				one_hot_labels = tf.one_hot(self.true_labels[:, i],
				                            self.taxonomy_classes[i], dtype=tf.int64)

				net_acc_val1['stage_%d' % i], net_acc1['stage_%d' % i] = tf.metrics.accuracy(
					self.true_labels[:, i],
					tf.argmax(self.stage_outputs[i][('stage_%d/fc8/squeezed') % i], 1)
				)
				net_acc_val3['stage_%d' % i], net_acc3['stage_%d' % i] = tf.metrics.mean(tf.nn.in_top_k(
					targets=self.true_labels[:, i],
					predictions=self.stage_outputs[i][('stage_%d/fc8/squeezed') % i], k=3)
				)
				net_prec1_val['stage_%d' % i], net_prec1['stage_%d' % i] = tf.metrics.precision_at_k(
					one_hot_labels,
					self.stage_outputs[i][('stage_%d/fc8/squeezed') % i], k=1
				)
				net_prec3_val['stage_%d' % i], net_prec3['stage_%d' % i] = tf.metrics.precision_at_k(
					one_hot_labels,
					self.stage_outputs[i][('stage_%d/fc8/squeezed') % i], k=3
				)
				net_recall1_val['stage_%d' % i], net_recall1['stage_%d' % i] = tf.metrics.precision_at_k(
					one_hot_labels,
					self.stage_outputs[i][('stage_%d/fc8/squeezed') % i], k=3
				)
				net_recall3_val['stage_%d' % i], net_recall3['stage_%d' % i] = tf.metrics.precision_at_k(
					one_hot_labels,
					self.stage_outputs[i][('stage_%d/fc8/squeezed') % i], k=3
				)

			# Define the metrics:
			eval_values, eval_updates = slim.metrics.aggregate_metric_map({
				'Accuracy_top1': (net_acc_val1, net_acc1),
				'Accuracy_top3': (net_acc_val3, net_acc3),
				'Precision_top1': (net_prec1_val, net_prec1),
				'Precision_top3': (net_prec3_val, net_prec3),
				'Recall_top1': (net_recall1_val, net_recall1),
				'Recall_top3': (net_recall3_val, net_recall3)
			})

			vars = tf.contrib.framework.get_variables(scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
			reset_op = tf.variables_initializer(vars)

			return eval_values, eval_updates, reset_op


	def get_summary_op(self):

		summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

		for name in self.eval_updates.keys():
			for stage in self.eval_updates[name].keys():
				summaries.add(tf.summary.scalar('metrics/%s/%s' % (name, stage), self.eval_updates[name][stage]))

		for name in self.net_losses.keys():
			summaries.add(tf.summary.scalar('losses/%s' % (name), self.net_losses[name]))

		summaries.add(tf.summary.scalar('learning_rate', self.learning_rate))

		summary_op = tf.summary.merge(list(summaries), name='summary_op')
		return summary_op

	def set_optimizer(self):

		self.learning_rate = network_utils._configure_learning_rate(self.learning_rate, self.batch_size,
															   self.num_samples, self.global_step)

		variables_to_train = network_utils._get_variables_to_train(self.trainable_scopes)
		# variables_to_train = handle_network_function._get_variables_to_train(None)
		# get gradients from trainable variables
		grads = tf.gradients(self.net_losses['all'], variables_to_train)
		# Adam Optimizer
		# optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='Adam_Optimizer')
		optimizer = network_utils._configure_optimizer(learning_rate=self.learning_rate, optimizer='adam')
		# Apply Gradients
		apply_op = optimizer.apply_gradients(
			zip(grads, variables_to_train),
			global_step=self.global_step,
			name='train_step')

		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

		update_ops.append(apply_op)
		update_op = tf.group(*update_ops)

		return update_op

	def get_prediction_topk(self, k=3):

		if self.taxonomy_loss is not None:
			_, predictions = tf.nn.top_k(self.stage_outputs[self.taxonomy_nums - 1][
				                             ('stage_%d/fc8/squeezed_tax') % (self.taxonomy_nums - 1)], int(k))
		else:
			_, predictions = tf.nn.top_k(self.stage_outputs[self.taxonomy_nums - 1][
				                             ('stage_%d/fc8/squeezed') % (self.taxonomy_nums - 1)], int(k))

		return predictions










