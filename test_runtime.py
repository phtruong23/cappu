import numpy as np
import tensorflow as tf
import os
import csv
import datetime
import taxonomy_model

import Grasp_csv_Loader
from params import PARAMS

# Original order of batch label is :
# [Grasp, ADL, OppType, PIP, VirtualFingers, Thumb]  : Total 6 classes.
# [32   , 3  , 4      , 4  , 8             , 3    ]
# In this time, don't care about ADL.
# Thumb -> PIP -> OppType -> VirtualFingers -> Grasp
# So, need to change the order of labels
# Memory lacked. So, skip the thumb
# label_order = [5, 3, 2, 4, 0]
# label_order = [3, 2, 4, 0]
label_order = [0]


def train():

	images = np.random.randn(100,224,224,3)
	labels = np.random.randint(0, 10, 100).reshape((100,1))
	classes = [10]

	batch_image = tf.constant(images, tf.float32)
	batch_label = tf.constant(labels, tf.int64)

	# Define Model
	model = taxonomy_model.taxonomy_model(inputs=batch_image,
	                                      true_labels=batch_label,
	                                      input_size=PARAMS.image_size,
	                                      batch_size=PARAMS.batch_size,
	                                      taxonomy_nums=1,
	                                      taxonomy_classes=classes,
	                                      resnet_version=PARAMS.resnet_version,
	                                      resnet_pretrained_path=PARAMS.resnet_path,
	                                      resnet_exclude=PARAMS.resnet_exclude,
	                                      trainable_scopes=PARAMS.trainable_scopes,
	                                      extra_global_feature=True,
	                                      taxonomy_loss=True,
	                                      learning_rate=PARAMS.learning_rate,
	                                      num_samples=100,
	                                      beta=PARAMS.beta,
	                                      taxonomy_weights=[1.0, 1.0, 1.0, 1.0],
	                                      all_label=None,
	                                      all_value=None,
	                                      batch_weight_range=[1.0, 1.0],
	                                      is_mode='train'
	                                      )
	all_inputs, end_point, losses, eval_value, eval_update, eval_reset = \
		model.build_model()

	train_summary_op = model.get_summary_op()



	config = tf.ConfigProto()
	# config.gpu_options.per_process_gpu_memory_fraction = 0.5
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	config.gpu_options.visible_device_list = '1'

	# Create a saver to save and restore all the variables.
	saver = tf.train.Saver()

	with tf.Session(config=config) as sess:

		now = datetime.datetime.now()

		# remember to intiate both global and local variables for training and evaluation
		sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

		# don't forget to insert these lines
		model.resnet_restore(sess)

		total_step_num = 0
		best_acc = 0.0

		for epoch in range(PARAMS.epochs):

			current_time = datetime.datetime.now()  # measure training time for each epoch

			while (True):
				try:
					step_time = datetime.datetime.now()

					# imgs, lbls = sess.run([batch_image, batch_label])

					# extract batch and training
					update = sess.run(
						{'all_inputs': all_inputs, 'all_outputs': end_point, 'update_op': model.update_flag,
						 'losses': losses, 'eval_update': eval_update,
						 'summary': train_summary_op},
						feed_dict={
							# inputs: images,
							# 	true_labels: labels[:,int(-args.layer):],
							model.resnet_training_flag: True,
							model.vgg19_training_flag: False,
							model.vgg_dropout: 0.5})

					# print(len(update['all_inputs']), len(update['all_outputs']))
					print('losses:', update['losses'], 'accuracy:', update['eval_update']['Accuracy_top1'])
					# print('losses:', update['losses'])

					total_step_num += 1
					print('Step training time: {}'.format(datetime.datetime.now() - step_time))

				except tf.errors.OutOfRangeError:
					break

			print('Epoch %d done. ' % (epoch + 1))
			print('Training time: {}'.format(datetime.datetime.now() - current_time))

			# reset all local variabels so that the streaming metrics reset new calculation
			sess.run(eval_reset)


if __name__ == '__main__':
	train()




