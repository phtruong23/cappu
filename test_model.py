import numpy as np
import tensorflow as tf
import os
import csv
import datetime
import taxonomy_model_v2
import argparse
import network_utils

# import Grasp_csv_Loader as GraspLoader
import Grasp_csv_Loader_v2
from params_test import PARAMS

# Original order of batch label is :
# [Grasp, ADL, OppType, PIP, VirtualFingers, Thumb]  : Total 6 classes.
# [32   , 3  , 4      , 4  , 8             , 3    ]
# In this time, don't care about ADL.
# Thumb -> PIP -> OppType -> VirtualFingers -> Grasp
# So, need to change the order of labels
# Memory lacked. So, skip the thumb
label_order = [5, 3, 2, 4, 0]
# label_order = [3, 2, 4, 0]
# label_order = [0]


def test(folder_log, modelname):

	# model_fullpath = os.path.join(folder_log, modelname)
	# if os.path.isfile(model_fullpath + '.meta') is False:
	# 	raise IsADirectoryError('No file at: ' + model_fullpath)

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
	                                              trans_range=PARAMS.trans_range,
	                                              rotate_range=PARAMS.rotate_range,
	                                              max_hue_delta=PARAMS.max_hue_delta,
	                                              saturation_range=PARAMS.saturation_range,
	                                              max_bright_delta=PARAMS.max_bright_delta,
	                                              max_contrast_delta=PARAMS.max_contrast_delta,
	                                              is_training=False
	                                              )
	grasp_loader.do_preprocessing = False # for evaluating the training set

	next_element, training_init_op, validation_init_op, test_init_op = \
		grasp_loader.initialization_dataset()

	batch_image, batch_label = next_element

	# Define Model
	model = taxonomy_model_v2.taxonomy_model(inputs=batch_image,
	                                         true_labels=batch_label,
	                                         input_size=PARAMS.image_size,
	                                         batch_size=PARAMS.batch_size,
	                                         taxonomy_nums=len(grasp_loader.classes_numbers),  #len(grasp_loader.classes_numbers),
	                                         taxonomy_classes=grasp_loader.classes_numbers,
	                                         resnet_version=PARAMS.resnet_version,
	                                         resnet_pretrained_path=PARAMS.resnet_path,
	                                         resnet_exclude=PARAMS.resnet_exclude,
	                                         trainable_scopes=PARAMS.trainable_scopes,
	                                         extra_global_feature=PARAMS.extra_global_feature_flag,
	                                         taxonomy_loss=PARAMS.taxonomy_loss,
	                                         learning_rate=PARAMS.learning_rate,
	                                         num_samples=len(grasp_loader.train_info),
	                                         beta=PARAMS.beta,
	                                         taxonomy_weights=[1.0, 1.0, 1.0, 1.0, 1.0],
	                                         all_label=None,
	                                         all_value=None,
	                                         batch_weight_range=[1.0, 1.0],
	                                         optimizer=PARAMS.optimizer,
	                                         is_mode='train',
	                                         divide_loss=PARAMS.divide_loss,
	                                         vgg_input_from_resnet=PARAMS.vgg_input_from_resnet,
	                                         vgg_fully_connected_num=PARAMS.vgg_fully_connected_num
	                                         )
	all_inputs, end_point, losses, eval_value, eval_update, eval_reset = \
		model.build_model()

	test_predictions = model.get_prediction_topk(k=1)


	config = tf.ConfigProto()
	# config.gpu_options.per_process_gpu_memory_fraction = 0.5
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	config.gpu_options.visible_device_list = PARAMS.gpu_num

	# Create a saver to save and restore all the variables.
	saver = tf.train.Saver()

	net_restore = network_utils._get_init_fn(PARAMS.pretrained_path, '', '', False)

	with tf.Session(config=config) as sess:

		# remember to intiate both global and local variables for training and evaluation
		# sess.run([tf.local_variables_initializer()])
		sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

		# load model from checkpoint
		# saver.restore(sess, PARAMS.pretrained_path)
		net_restore(sess)

		total_step_num = 0
		# predicted_filename = folder_log + '/prediction_%s.csv' % (modelname)
		predicted_filename = folder_log + '/prediction_%s.csv' % (modelname)

		with open(predicted_filename, "a") as output:
			writer = csv.writer(output, lineterminator='\n')
			writer.writerows([['id','predicted']])

		current_time = datetime.datetime.now()  # measure training time for each epoch

		# Test the model with test_dataset
		# initiate the batch extraction using tf.data.Dataset
		sess.run(test_init_op) #training_init_op
		# sess.run(validation_init_op)  # training_init_op
		# sess.run(training_init_op)  # training_init_op

		while (True):
			try:
				update = sess.run({'batch_image': batch_image, 'batch_label': batch_label,
				                   'all_inputs': all_inputs, 'all_outputs': end_point,
				                   'eval_value': eval_value,
				                   'eval_update': eval_update,
				                   'topk': test_predictions
				                   },
				                  feed_dict={
					                  # inputs: images,
					                  # 	true_labels: labels[:,int(-args.layer):],
					                  model.resnet_training_flag: False,
					                  model.vgg19_training_flag: False,
					                  model.vgg_dropout: 1.0})

				print('Accuracy_top1:', update['eval_update']['Accuracy_top1'], 'Accuracy_top3:', update['eval_update']['Accuracy_top3'])

				result_array = [[update['batch_label'][i], update['topk'][i]] for i in range(len(update['batch_label']))]
				with open(predicted_filename, "a") as output:
					writer = csv.writer(output, lineterminator='\n')
					writer.writerows(result_array)

			except tf.errors.OutOfRangeError:
				break

		test_metrics = sess.run(eval_value)
		# test_metrics = update['eval_value']

		print({name: test_metrics[name]['stage_0'] for name in test_metrics.keys()})
		print('Evaluation time: ', datetime.datetime.now() - current_time)

		# convert result in to np array and save to file
		test_metrics_arr = [ [name] + [test_metrics[name][stage] for stage in test_metrics[name].keys()]
						   for name in test_metrics.keys()]
		metric_names = list(test_metrics.keys())
		stages = ['Metrics'] + list(test_metrics[metric_names[0]].keys())
		test_metrics_arr = [stages] + test_metrics_arr

		with open(folder_log + '/test_evaluation.csv', "a") as output:
			writer = csv.writer(output, lineterminator='\n')
			writer.writerows([[modelname]])
			writer.writerows(test_metrics_arr)

		# reset all local variabels so that the streaming metrics reset new calculation
		sess.run(eval_reset)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description='Testing recognition results from trained model.')
	parser.add_argument(
		'-d',
		'--dir',
		type=str,
		dest='directory',
		default='train_2018_8_6_17_16',
		help='trained model directory (default: %(default)s)'
	)
	parser.add_argument(
		'-f',
		'--modelname',
		type=str,
		dest='modelname',
		default='Grasp.ckpt',
		help='checkpoint file name (default: %(default)s)'
	)
	args = parser.parse_args()


	test(args.directory , args.modelname)