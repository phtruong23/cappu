import csv
# import Grasp_csv_Loader
import Grasp_csv_Loader_v2
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

import taxonomy_model
import network_utils

import os
import datetime

from params import PARAMS


label_order = [5, 3, 2, 4, 0]

# grasp_loader = Grasp_csv_Loader.csv_loader(data_path=csv_path,
# 										   csv_filename=csv_filename,
# 										   save_folder=save_folder,
#                                            label_order=label_order,
# 										   batch_size=500)
#
# # print(len(grasp_loader.all_annotations))
#
# # print(grasp_loader.train_meaningful_jpg_names[89433])
# # temp_image, temp_label = grasp_loader._read_per_image_train(89433)
#
# print(len(grasp_loader.train_meaningful_jpg_names),
# 	  len(grasp_loader.val_meaningful_jpg_names),
# 	  len(grasp_loader.test_meaningful_jpg_names))

grasp_loader = Grasp_csv_Loader_v2.csv_loader(data_path=PARAMS.csv_path,
												csv_filename=PARAMS.csv_filename,
												save_folder=PARAMS.save_folder,
												label_order=label_order,
												divide_by_ratio=True,
												is_divided_saved=True,
												divided_npz_name='divided_dataset.npz',
												batch_size= 10,
												trans_range=PARAMS.trans_range,
												rotate_range=PARAMS.rotate_range,
												max_hue_delta=PARAMS.max_hue_delta,
												saturation_range=PARAMS.saturation_range,
												max_bright_delta=PARAMS.max_bright_delta,
												max_contrast_delta=PARAMS.max_contrast_delta,
												)

total_list = grasp_loader.get_annotation_sorted_by_label('Grasp', Grasp_csv_Loader_v2.grasp_names)

print(len(total_list), len(total_list[0]))

# train_names, train_labels, val_names, val_labels, test_names, test_labels = \
# 	grasp_loader.get_jpg_filenames_labels_from_sorted_annotations(total_list, [0.7, 0.1, 0.2])
train_info, val_info, test_info = \
	grasp_loader.get_jpg_filenames_labels_from_sorted_annotations(total_list, [0.7, 0.1, 0.2])

print(len(train_info), len(val_info), len(test_info))

# grasp_loader.read_frames_and_save_from_mp4(None, 6, 'subject_7_gopro_seg_1.mp4')

# random = np.random.permutation(len(grasp_loader.train_meaningful_jpg_names))
# for i in range(0, len(grasp_loader.train_meaningful_jpg_names)):
# 	print(grasp_loader.train_meaningful_jpg_names[random[i]])
# 	temp_img, temp_label = grasp_loader._read_per_image_train(random[i])
# 	print(np.shape(temp_img), np.shape(temp_label))
#
# random = np.random.permutation(len(grasp_loader.val_meaningful_jpg_names))
# for i in range(0, len(grasp_loader.val_meaningful_jpg_names)):
# 	print(grasp_loader.val_meaningful_jpg_names[random[i]])
# 	temp_img, temp_label = grasp_loader._read_per_image_val(random[i])
# 	print(np.shape(temp_img), np.shape(temp_label))
#
# random = np.random.permutation(len(grasp_loader.test_meaningful_jpg_names))
# for i in range(0, len(grasp_loader.test_meaningful_jpg_names)):
# 	print(grasp_loader.test_meaningful_jpg_names[random[i]])
# 	temp_img, temp_label = grasp_loader._read_per_image_test(random[i])
# 	print(np.shape(temp_img), np.shape(temp_label))

## testing function of loading all test images
# for i in range(0, len(grasp_loader.test_info)):
# 	print(grasp_loader.test_info[i]['Filename'])
# 	temp_img, temp_label = grasp_loader._read_per_image_test(i)
# 	print(np.shape(temp_img), np.shape(temp_label))

# print(train_info[10]['Grasp'])
# print(train_info[10]['Filename'])
# temp_img, temp_label = grasp_loader._read_per_image_train(1000)

next_element, training_init_op, validation_init_op, test_init_op = \
	grasp_loader.initialization_dataset()

batch_image, batch_label = next_element

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.gpu_options.visible_device_list = '1'

with tf.Session(config=config) as sess:
	# sess.run(training_init_op)
	#
	# while True:
	#
	# 	try:
	# 		update = sess.run({'batch_image': batch_image, 'batch_label': batch_label})
	# 		print('training... : ', np.shape(update['batch_image']), np.shape(update['batch_label']))
	#
	# 		# plot input image in the screen
	# 		fig = plt.figure(figsize=(20, 8))
	# 		for i in range(10):
	# 			fig.add_subplot(2,5,i+1)
	# 			plt.imshow(update['batch_image'][i])
	# 		plt.show()
	# 		plt.pause(1)
	# 		plt.close()
	#
	# 	except tf.errors.OutOfRangeError:
	# 		break

	# sess.run(validation_init_op)
	#
	# while True:
	#
	# 	try:
	# 		update = sess.run({'batch_image': batch_image, 'batch_label': batch_label})
	# 		print('validating... : ', np.shape(update['batch_image']), np.shape(update['batch_label']))
	#
	# 		# plot input image in the screen
	# 		fig = plt.figure(figsize=(20, 8))
	# 		for i in range(10):
	# 			fig.add_subplot(2, 5, i + 1)
	# 			plt.imshow(update['batch_image'][i])
	# 		plt.show()
	# 		plt.pause(1)
	# 		plt.close()
	#
	# 	except tf.errors.OutOfRangeError:
	# 		break

	sess.run(test_init_op)

	while True:

		try:
			update = sess.run({'batch_image': batch_image, 'batch_label': batch_label})
			print('testing... : ', np.shape(update['batch_image']), np.shape(update['batch_label']))

			# plot input image in the screen
			fig = plt.figure(figsize=(20, 8))
			for i in range(10):
				fig.add_subplot(2, 5, i + 1)
				plt.imshow(update['batch_image'][i])
			plt.axis("off")
			plt.show()
			plt.pause(1)
			plt.close()

		except tf.errors.OutOfRangeError:
			break
