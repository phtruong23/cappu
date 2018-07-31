import numpy as np
import tensorflow as tf
import os
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
	grasp_loader = Grasp_csv_Loader.csv_loader(data_path=PARAMS.csv_path,
											   csv_filename=PARAMS.csv_filename,
											   save_folder=PARAMS.save_folder,
	                                           is_saved=True,
	                                           resize_image_size=PARAMS.image_size,
	                                           train_list=PARAMS.train_list,
	                                           val_list=PARAMS.val_list,
	                                           test_list=PARAMS.test_list,
	                                           label_order=label_order,
	                                           batch_size=PARAMS.batch_size,
	                                           max_hue_delta=PARAMS.max_hue_delta,
	                                           saturation_range=PARAMS.saturation_range,
	                                           max_bright_delta=PARAMS.max_bright_delta,
	                                           max_contrast_delta=PARAMS.max_contrast_delta,
	                                           is_training=True
	                                           )

	next_element, training_init_op, validation_init_op, test_init_op = \
		grasp_loader.initialization_dataset()

	batch_image, batch_label = next_element

	# Define Model
	model = taxonomy_model.taxonomy_model(inputs=batch_image,
	                                      true_labels=batch_label,
	                                      input_size=PARAMS.image_size,
	                                      batch_size=PARAMS.batch_size,
	                                      taxonomy_nums=len(grasp_loader.classes_numbers),
	                                      taxonomy_classes=grasp_loader.classes_numbers,
	                                      resnet_version=PARAMS.resnet_version,
	                                      resnet_pretrained_path=PARAMS.resnet_path,
	                                      resnet_exclude=PARAMS.resnet_exclude,
	                                      trainable_scopes=PARAMS.trainable_scopes,
	                                      extra_global_feature=True,
	                                      taxonomy_loss=True,
	                                      learning_rate=PARAMS.learning_rate,
	                                      num_samples=len(grasp_loader.train_meaningful_jpg_names),
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
	config.gpu_options.visible_device_list = PARAMS.gpu_num

	# Create a saver to save and restore all the variables.
	saver = tf.train.Saver()

	with tf.Session(config=config) as sess:

		now = datetime.datetime.now()
		folder_log = './' + 'train_%s_%s_%s_%s_%s' % (now.year, now.month, now.day, now.hour, now.minute)
		# folder_log = '.\\' + 'train_%s_%s_%s_%s_%s' % (now.year, now.month, now.day, now.hour, now.minute)
		print('folder_log: ', folder_log)
		if not os.path.exists(folder_log):
			os.makedirs(folder_log)

		# For windows
		# os.system('copy .\\*.py %s' % (folder_log))
		# For Linux
		os.system('cp ./*.py %s' % (folder_log))

		# remember to intiate both global and local variables for training and evaluation
		sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

		# don't forget to insert these lines
		model.resnet_restore(sess)

		summary_writer = tf.summary.FileWriter('%s' % (folder_log), sess.graph)
		total_step_num = 0

		for epoch in range(PARAMS.epochs):

			current_time = datetime.datetime.now()  # measure training time for each epoch
			# initiate the batch extraction using tf.data.Dataset
			sess.run(training_init_op)

			while (True):
				try:
					# extract batch and training
					update = sess.run(
						{'all_inputs': all_inputs, 'all_outputs': end_point, 'update_op': model.update_flag,
						 'losses': losses, 'eval_update': eval_update,
						 'summary': train_summary_op},
						feed_dict={
							# inputs: images,
							# 	true_labels: labels[:,int(-args.layer):],
							model.resnet_training_flag: False,
							model.vgg19_training_flag: True,
							model.vgg_dropout: 0.5})
					# print(len(update['all_inputs']), len(update['all_outputs']))
					# print('losses:', update['losses'], 'accuracy:', update['eval_update']['Accuracy'])
					print('losses:', update['losses'])

					summary_writer.add_summary(update['summary'], total_step_num)
					summary_writer.flush()
					total_step_num += 1

				# imgs, lbls = sess.run([images, labels])

				except tf.errors.OutOfRangeError:
					break

			print('Epoch %d done. ' % (epoch + 1))
			print('Training time: {}'.format(datetime.datetime.now() - current_time))

			# reset all local variabels so that the streaming metrics reset new calculation
			sess.run(eval_reset)

			checkpoint_file = os.path.join(folder_log, 'Grasp.ckpt')
			saver.save(sess, checkpoint_file, global_step=epoch)


if __name__ == '__main__':
	train()




