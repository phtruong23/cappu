import tensorflow as tf
import numpy as np
import vgg
import handle_network_function

import Grasp_csv_Loader_v2
from params import PARAMS
import network_utils

import datetime
import os

slim = tf.contrib.slim

label_order = [5, 3, 2, 4, 0]

def get_loss(network_output, labels):

	net_losses = tf.losses.sparse_softmax_cross_entropy(
		labels=labels,
		logits=network_output,
		scope='resnet_loss')
	return net_losses

def set_optimizer(trainable_scopes, loss, global_step, num_samples):

	learning_rate = network_utils._configure_learning_rate(PARAMS.learning_rate, PARAMS.batch_size,
	                                                       num_samples, global_step)

	variables_to_train = network_utils._get_variables_to_train(trainable_scopes)
	print(variables_to_train)
	# variables_to_train = handle_network_function._get_variables_to_train(None)
	# get gradients from trainable variables
	grads = tf.gradients(loss, variables_to_train)
	# Adam Optimizer
	# optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='Adam_Optimizer')
	optimizer = network_utils._configure_optimizer(learning_rate=PARAMS.learning_rate, optimizer=PARAMS.optimizer)
	# Apply Gradients
	apply_op = optimizer.apply_gradients(
		zip(grads, variables_to_train),
		global_step=global_step,
		name='train_step')

	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

	update_ops.append(apply_op)
	update_op = tf.group(*update_ops)

	return update_op, learning_rate

def get_metrics(label, output):

	net_acc1, net_acc_val1 = {}, {}
	net_acc3, net_acc_val3 = {}, {}
	net_prec1_val, net_prec1 = {}, {}
	net_prec3_val, net_prec3 = {}, {}
	net_recall1_val, net_recall1 = {}, {}
	net_recall3_val, net_recall3 = {}, {}

	with tf.variable_scope('metrics') as scope:

		i = 4
		i_depth = 32

		# transform the last labels to calculate metrics (precision, recall)
		one_hot_labels = tf.one_hot(label[:, i],
		                            i_depth, dtype=tf.int64)

		net_acc_val1['stage_%d' % i], net_acc1['stage_%d' % i] = tf.metrics.accuracy(
			label[:, i],
			tf.argmax(output, 1)
		)
		net_acc_val3['stage_%d' % i], net_acc3['stage_%d' % i] = tf.metrics.mean(tf.nn.in_top_k(
			targets=label[:, i],
			predictions=output, k=3)
		)
		net_prec1_val['stage_%d' % i], net_prec1['stage_%d' % i] = tf.metrics.precision_at_k(
			one_hot_labels,
			output, k=1
		)
		net_prec3_val['stage_%d' % i], net_prec3['stage_%d' % i] = tf.metrics.precision_at_k(
			one_hot_labels,
			output, k=3
		)
		net_recall1_val['stage_%d' % i], net_recall1['stage_%d' % i] = tf.metrics.precision_at_k(
			one_hot_labels,
			output, k=3
		)
		net_recall3_val['stage_%d' % i], net_recall3['stage_%d' % i] = tf.metrics.precision_at_k(
			one_hot_labels,
			output, k=3
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


def get_summary_op(eval_updates, loss, learning_rate):

	summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

	for name in eval_updates.keys():
		for stage in eval_updates[name].keys():
			summaries.add(tf.summary.scalar('metrics/%s/%s' % (name, stage), eval_updates[name][stage]))

	# for name in self.net_losses.keys():
	# 	summaries.add(tf.summary.scalar('losses/%s' % (name), self.net_losses[name]))
	summaries.add(tf.summary.scalar('loss', loss))

	summaries.add(tf.summary.scalar('learning_rate', learning_rate))

	summary_op = tf.summary.merge(list(summaries), name='summary_op')
	return summary_op

def get_summary_op_val(eval_updates, loss):

	summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

	for name in eval_updates.keys():
		for stage in eval_updates[name].keys():
			summaries.add(tf.summary.scalar('metrics_val/%s/%s' % (name, stage), eval_updates[name][stage]))

	# for name in self.net_losses.keys():
	# 	summaries.add(tf.summary.scalar('losses/%s' % (name), self.net_losses[name]))
	summaries.add(tf.summary.scalar('loss_val', loss))

	summary_op = tf.summary.merge(list(summaries), name='summary_op_val')
	return summary_op

# def get_prediction_topk(k=3):
#
# 	_, predictions = tf.nn.top_k(self.stage_outputs[self.taxonomy_nums - 1][
# 		                             ('stage_%d/fc8/squeezed') % (self.taxonomy_nums - 1)], int(k))
#
# 	return predictions


def train():

	grasp_loader = Grasp_csv_Loader_v2.csv_loader(data_path=PARAMS.csv_path,
	                                              csv_filename=PARAMS.csv_filename,
	                                              save_folder=PARAMS.save_folder,
	                                              is_saved=True,
	                                              resize_image_size=PARAMS.image_size,
	                                              train_subject_list=PARAMS.train_list,
	                                              val_subject_list=PARAMS.val_list,
	                                              test_subject_list=PARAMS.test_list,
	                                              divide_by_ratio=True,
	                                              is_divided_saved=True,
	                                              divided_npz_name='divided_dataset.npz',
	                                              label_order=label_order,
	                                              batch_size=PARAMS.batch_size,
	                                              flip_flag=PARAMS.flip_flag,
	                                              trans_range=PARAMS.trans_range,
	                                              rotate_range=PARAMS.rotate_range,
	                                              max_hue_delta=PARAMS.max_hue_delta,
	                                              saturation_range=PARAMS.saturation_range,
	                                              max_bright_delta=PARAMS.max_bright_delta,
	                                              max_contrast_delta=PARAMS.max_contrast_delta,
	                                              is_training=True
	                                              )

	next_element, training_init_op, validation_init_op, test_init_op = \
		grasp_loader.initialization_dataset()

	batch_image, batch_label = next_element

	# with slim.arg_scope(partial_resnet.resnet_arg_scope()):
	# 	resnet_partial, resnet_partial_mid = partial_resnet.resnet_v2_partial_50(inputs=batch_image,
	# 	                                                                         num_classes=32,
	# 	                                                                         is_training=True,
	# 	                                                                         global_pool=True,
	# 	                                                                         output_stride=None,
	# 	                                                                         spatial_squeeze=True,
	# 	                                                                         reuse=None,
	# 	                                                                         scope='resnet_v2_50',
	# 	                                                                         include_root_block=True)

	vgg19_exclude_scopes = 'vgg_19/fc8'

	vgg_dropout = tf.placeholder(tf.float32)

	with slim.arg_scope(vgg.vgg_arg_scope()):
		vgg19, vgg19_end_points = vgg.vgg_19(inputs=batch_image,
		                                     num_classes=32,
		                                     is_training=True,
		                                     dropout_keep_prob=vgg_dropout,
		                                     spatial_squeeze=True,
		                                     scope='vgg_19',
		                                     fc_conv_padding='VALID',
		                                     global_pool=True
		                                     )


	vgg_loss = get_loss(vgg19, batch_label[:, 4])

	vgg19_exclude_scopes = 'vgg_19/fc8'

	vgg_restore = handle_network_function._get_init_fn('./vgg_19_pretrained/vgg_19.ckpt',
	                                                      '', vgg19_exclude_scopes)

	global_step = tf.train.create_global_step()

	update_vgg, learning_rate = set_optimizer(vgg19_exclude_scopes, vgg_loss, global_step, len(grasp_loader.train_info))

	eval_values, eval_updates, reset_op = get_metrics(batch_label, vgg19)

	vgg_summary = get_summary_op(eval_updates, vgg_loss, learning_rate)

	vgg_val_summary = get_summary_op_val(eval_updates, vgg_loss)

	config = tf.ConfigProto()
	# config.gpu_options.per_process_gpu_memory_fraction = 0.5
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	config.gpu_options.visible_device_list = '0'

	saver = tf.train.Saver()

	with tf.Session(config=config) as sess:

		now = datetime.datetime.now()
		folder_log = './' + 'vgg_train_%s_%s_%s_%s_%s' % (now.year, now.month, now.day, now.hour, now.minute)
		# folder_log = '.\\' + 'train_%s_%s_%s_%s_%s' % (now.year, now.month, now.day, now.hour, now.minute)
		print('folder_log: ', folder_log)
		if not os.path.exists(folder_log):
			os.makedirs(folder_log)

		os.system('cp ./*.py %s' % (folder_log))

		# sess.run(tf.global_variables_initializer())

		sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

		vgg_restore(sess)

		summary_writer = tf.summary.FileWriter('%s' % (folder_log), sess.graph)
		total_step_num = 0

		for epoch in range(PARAMS.epochs):

			sess.run(training_init_op)

			while (True):
				try:

					update = sess.run({'batch_image': batch_image, 'batch_label': batch_label,
					                   'vgg19': vgg19,
					                   'vgg_loss': vgg_loss, 'update_vgg': update_vgg,
					                   'eval_update': eval_updates, 'summary': vgg_summary
					                   }, feed_dict={vgg_dropout: 0.5})

					print('vgg loss : ', update['vgg_loss'])
					print('accuracy (top1, top3):', update['eval_update']['Accuracy_top1'], update['eval_update']['Accuracy_top3'])

					summary_writer.add_summary(update['summary'], total_step_num)
					summary_writer.flush()
					total_step_num += 1

				except tf.errors.OutOfRangeError:
					break

			print('Epoch %d done. ' % (epoch + 1))

			checkpoint_file = os.path.join(folder_log, 'vgg.ckpt')
			saver.save(sess, checkpoint_file, global_step=epoch)

			sess.run(reset_op)

			sess.run(validation_init_op)

			while (True):
				try:

					update = sess.run({'batch_image': batch_image, 'batch_label': batch_label,
					                   'vgg19': vgg19,
					                   'vgg_loss': vgg_loss, 'update_vgg': update_vgg,
					                   'eval_update': eval_updates, 'summary': vgg_val_summary
					                   }, feed_dict={vgg_dropout: 1.0})

					print('vgg loss : ', update['vgg_loss'])
					print('accuracy (top1, top3):', update['eval_update']['Accuracy_top1'],
					      update['eval_update']['Accuracy_top3'])

					summary_writer.add_summary(update['summary'], epoch)
					summary_writer.flush()
					total_step_num += 1

				except tf.errors.OutOfRangeError:
					break

			print('Validation %d done. ' % (epoch + 1))

		sess.run(reset_op)


if __name__ == '__main__':
	train()


