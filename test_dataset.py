import csv
import Grasp_csv_Loader
import numpy as np
import tensorflow as tf

csv_path = '../Gopro'
# csv_path = '..\\grasp_dataset\\Xsens'

csv_filename = 'SDATA1700291_annotated_data.csv'

save_folder = 'save_frames'

grasp_loader = Grasp_csv_Loader.csv_loader(data_path=csv_path,
										   csv_filename=csv_filename,
										   save_folder=save_folder)

# print(len(grasp_loader.all_annotations))

# print(grasp_loader.train_meaningful_jpg_names[89433])
# temp_image, temp_label = grasp_loader._read_per_image_train(89433)

print(len(grasp_loader.train_meaningful_jpg_names),
	  len(grasp_loader.val_meaningful_jpg_names),
	  len(grasp_loader.test_meaningful_jpg_names))
# print(np.shape(temp_image), np.shape(temp_label))
# print(temp_label)

# random = np.random.permutation(499757)
# for i in range(0, 499757):
# 	print(random[i])
# 	grasp_loader._read_per_image_train(random[i])


next_element, training_init_op, validation_init_op, test_init_op = \
	grasp_loader.initialization_dataset()

batch_image, batch_label = next_element

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.gpu_options.visible_device_list = '0'

with tf.Session(config=config) as sess:
	sess.run(training_init_op)

	while True:

		try:
			update = sess.run({'batch_image': batch_image, 'batch_label': batch_label})
			print('training... : ', np.shape(update['batch_image']), np.shape(update['batch_label']))
		except tf.errors.OutOfRangeError:
			break

	sess.run(validation_init_op)

	while True:

		try:
			update = sess.run({'batch_image': batch_image, 'batch_label': batch_label})
			print('validating... : ', np.shape(update['batch_image']), np.shape(update['batch_label']))
		except tf.errors.OutOfRangeError:
			break

	sess.run(test_init_op)

	while True:

		try:
			update = sess.run({'batch_image': batch_image, 'batch_label': batch_label})
			print('testing... : ', np.shape(update['batch_image']), np.shape(update['batch_label']))
		except tf.errors.OutOfRangeError:
			break








