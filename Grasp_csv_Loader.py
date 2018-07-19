import csv
import cv2
import imageio
import numpy as np
import os
import fnmatch
import tensorflow as tf

# Number of classes : 13
csv_subject_folder_names = ['1. Subject 1',
							'2. Subject 2',
							'3. Subject 3',
							'4. Subject 4',
							'5. Subject 5',
							'6. Subject 6',
							'7. Subject 7',
							'8. Subject 8',
							'9. Subject 9',
							'10. Subject 10',
							'11. Subject 11',
							'12. Subject 12',
							'13. Subject 13']

# Number of classes : 13
csv_subject_label_names = ['subject 1',
						   'subject 2',
						   'subject 3',
						   'subject 4',
						   'subject 5',
						   'subject 6',
						   'subject 7',
						   'subject 8',
						   'subject 9',
						   'subject 10',
						   'subject 11',
						   'subject 12',
						   'subject 13']

# Number of classes : 32
grasp_names = ['sphere 3 finger',
			   'parallel extension',
			   'large diameter',
			   'power sphere',
			   'prismatic 4 finger',
			   'lateral tripod',
			   'tripod',
			   'push',
			   'prismatic 2 finger',
			   'tip pinch',
			   'lateral',
			   'small diameter',
			   'extension type',
			   'adducted thumb',
			   'stick',
			   'fixed hook',
			   'palmar pinch',
			   'inferior pincer',
			   'precision sphere',
			   'quadpod',
			   'prismatic 3 finger',
			   'index finger extension',
			   'sphere 4 finger',
			   'power disk',
			   'writing tripod',
			   'medium wrap',
			   'adduction grip',
			   'ventral',
			   'precision disk',
			   'lift',
			   'light tool',
			   'palmar'
			   ]

# Number of classes : 3
adl_names = ['cooking',
			 'laundry',
			 'housekeeping'
			 ]

# Number of classes : 4
opptype_names = ['Pad',
				 'Palm',
				 'Side',
				 'null',
				 ]

# Number of classes : 4
pip_names = ['Power',
			 'Precision',
			 'Intermediate',
			 'null'
			 ]

# Number of classes : 8
virtual_fingers_names = ['2',
						 '3',
						 '2-3',
						 '2-4',
						 '2-5',
						 '3-4',
						 '3-5',
						 'null']

# Number of classes : 3
thumb_names = ['Abd',
			   'Add',
			   'null'
			   ]

class csv_loader(object):
	def __init__(self,
				 data_path,
				 csv_filename,
				 save_folder,
				 is_saved=True,
				 resize_image_size=[224, 224],
				 train_list=[0, 1, 2, 3, 4, 5, 6, 7, 8],
				 val_list=[9],
				 test_list=[10, 11, 12],
	             label_order=None,
				 batch_size=10,
				 max_hue_delta=0.15,
				 saturation_range=[0.5, 2.0],
				 max_bright_delta=0.25,
				 max_contrast_delta=[0, 0.3],
				 is_training=True
				 ):
		self.csv_subject_folder_names = csv_subject_folder_names
		self.csv_subject_label_names = csv_subject_label_names

		# Names what we will use
		self.grasp_names = grasp_names
		self.adl_names = adl_names
		self.opptype_names = opptype_names
		self.pip_names = pip_names
		self.virtual_fingers_names = virtual_fingers_names
		self.thumb_names = thumb_names
		self.classes_numbers = np.array([len(self.grasp_names),
		                                 len(self.adl_names),
		                                 len(self.opptype_names),
		                                 len(self.pip_names),
		                                 len(self.virtual_fingers_names),
		                                 len(self.thumb_names)])

		self.label_order = label_order

		if self.label_order is not None:
			self.classes_numbers = self.classes_numbers[self.label_order]

		self.data_path = data_path
		self.csv_filename = csv_filename
		self.save_folder = save_folder
		self.resize_image_size = resize_image_size
		self.train_list = train_list
		self.val_list = val_list
		self.test_list = test_list

		self.batch_size = batch_size
		self.max_hue_delta = max_hue_delta
		self.saturation_range = saturation_range
		self.max_bright_delta = max_bright_delta
		self.max_contrast_delta = max_contrast_delta
		self.is_training = is_training

		self.all_annotations = self.get_all_annotations()

		if is_saved is True:
			# It takes some time to get all jpg names. (too many)
			# self.all_jpg_names = self.get_all_jpg_filenames()

			# This makes names from annotations (fast)
			# self.all_meaningful_jpg_names = self.get_all_meaningful_jpg_filenames_from_annotations()

			# This makes thress names from annotations
			# It makes three parts : train, validation, test
			# It is also fast enough
			self.train_meaningful_jpg_names, self.val_meaningful_jpg_names, self.test_meaningful_jpg_names = \
				self.get_divided_meaningful_jpg_filenames_from_annotations(self.train_list,
																		   self.val_list,
																		   self.test_list)



	def get_all_annotations(self):
		with open('%s/%s' % (self.data_path, self.csv_filename), newline='') as csvfile:
			reader = csv.DictReader(csvfile)

			# In the csv file for grasp, many dict exist
			# ID means that just id of each grasp
			# Subject means that that 1-13
			# Grasp means that how to grasp the object
			# ADL means that the environment of the movie
			# Video means that the name of video
			# StartFrame means that the start frame of this grasp start on the vidoe ( not depth)
			# EndFrame means that the end frame of this grasp end on the vidoe ( not depth)
			# OppType is the grasping type (plam, side, pad, null)
			# PIP means that the grasping type (power, precision, intermeidate)
			# VirtualFingers means that the used fingers when grasp the object (not thumb, 2, 2-3, 2-4, 2-5, etc)
			# Thumb means that the thumb shape (add, abd)
			# Duration means that the number of frames about each grasp
			# depthStart means that the start frame of this grasp start on the vidoe (depth)
			# depthEnd means that the end frame of this grasp end on the vidoe (depth)
			# SequenceNum is unknown

			all_annotations = [{'ID': row['ID'],
								'Subject': row['Subject'],
								'Grasp': row['Grasp'],
								'ADL': row['ADL'],
								'Video': row['Video'],
								'StartFrame': row['StartFrame'],
								'EndFrame': row['EndFrame'],
								'OppType': row['OppType'],
								'PIP': row['PIP'],
								'VirtualFingers': row['VirtualFingers'],
								'Thumb': row['Thumb'],
								'Duration': row['Duration'],
								'depthStart': row['depthStart'],
								'depthEnd': row['depthEnd'],
								'SequenceNum': row['SequenceNum']
								} for row in reader]

			return all_annotations

	def read_frames_and_save_from_mp4(self, save_path, subject_num, mp4_name):
		reader = imageio.get_reader('%s/%s/%s' % (self.data_path,
												  self.csv_subject_folder_names[subject_num],
												  mp4_name))
		for i, im in enumerate(reader):
			print('processing... : %d, %s, %d'% (subject_num, mp4_name, i + 1))
			save_filename = '%s.%s.%d.jpg' % (self.csv_subject_label_names[subject_num],
											  mp4_name, i + 1)
			imageio.imwrite('%s/%s' % (save_path, save_filename), im)

	def total_save_from_mp4(self):
		save_path = '%s/%s' % (self.data_path, self.save_folder)
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		for i, sub in enumerate(self.csv_subject_folder_names):
			movie_filenames = os.listdir('%s/%s' % (self.data_path, sub))
			print(movie_filenames)
			for movie in movie_filenames:
				self.read_frames_and_save_from_mp4(save_path, i, movie)

	# This function doesn't care about the extention of file
	def find_label_from_filename(self, filename):
		split_name = filename.split('.')

		labels = [{'Grasp': row['Grasp'],
				   'ADL': row['ADL'],
				   'OppType': row['OppType'],
				   'PIP': row['PIP'],
				   'VirtualFingers': row['VirtualFingers'],
				   'Thumb': row['Thumb']}
				  for row in self.all_annotations
				  if (split_name[0] == row['Subject']) &
				  (split_name[1] == row['Video'].split('.')[0]) &
				  (int(split_name[3]) >= int(row['StartFrame'])) &
				  (int(split_name[3]) <= int(row['EndFrame']))
				  ]

		return labels

	def read_jpg_files_according_subject_and_frames(self, each_row):
		# Assume that the all images has (1280 x 720 - w x h)
		sequence_frames = np.zeros(shape=(len(range(each_row['StartFrame'], each_row['EndFrame'])),
										  720, 1280, 3),
								   dtype=np.float32)
		movie_name = each_row['Video'].split('.')[0]
		for i in range(int(each_row['StartFrame']), int(each_row['EndFrame'])):
			each_filename = ('%s/%s/%s.%s.mp4.%d.jpg' % (self.data_path,
														 self.save_folder,
														 each_row['Subject'],
														 movie_name,
														 i))
			sequence_frames[i, :, :, :] = cv2.imread(each_filename)

		return sequence_frames

	def get_all_jpg_filenames(self):
		saved_path = ('%s/%s' % (self.data_path, self.save_folder))
		listOfFiles = os.listdir(saved_path)
		jpg_pattern = "*.jpg"  # It consists the color image

		jpg_names = [entry for entry in listOfFiles if fnmatch.fnmatch(entry, jpg_pattern)]

		jpg_names = sorted(jpg_names)

		return jpg_names

	# This function make names from annotations artificially
	# For takes only meaningful filenames
	# Meaningful means that these names has annotations
	def get_all_meaningful_jpg_filenames_from_annotations(self):
		meaningful_names = [('%s.%s.mp4.%d.jpg' % (row['Subject'],
												   row['Video'].split('.')[0],
												   i))
							for row in self.all_annotations
							for i in range(int(row['StartFrame']), int(row['EndFrame']))
							]
		return meaningful_names

	# This function is to divide dataset with three parts : training, validation, testing
	# It makes according to the subject
	def get_divided_meaningful_jpg_filenames_from_annotations(self, train_sub_list, val_sub_list, test_sub_list):
		train_meaningful_names = [('%s.%s.mp4.%d.jpg' % (row['Subject'],
														 row['Video'].split('.')[0],
														 i))
								  for row in self.all_annotations
								  for sub_num in train_sub_list
								  for i in range(int(row['StartFrame']), int(row['EndFrame']))
								  if self.csv_subject_label_names[sub_num] == row['Subject']
								  ]

		val_meaningful_names = [('%s.%s.mp4.%d.jpg' % (row['Subject'],
														 row['Video'].split('.')[0],
														 i))
								  for row in self.all_annotations
								  for sub_num in val_sub_list
								  for i in range(int(row['StartFrame']), int(row['EndFrame']))
								  if self.csv_subject_label_names[sub_num] == row['Subject']
								  ]

		test_meaningful_names = [('%s.%s.mp4.%d.jpg' % (row['Subject'],
														 row['Video'].split('.')[0],
														 i))
								  for row in self.all_annotations
								  for sub_num in test_sub_list
								  for i in range(int(row['StartFrame']), int(row['EndFrame']))
								  if self.csv_subject_label_names[sub_num] == row['Subject']
								  ]

		return train_meaningful_names, val_meaningful_names, test_meaningful_names

	# This function get id of our annotations
	# id order decided by each name lists defined above
	# The order of ids like this : [grasp_id, adl_id, opptype_id, pip_id, virtual_fingers_id, thumb_id]
	# It can be permutated by users
	def get_label_indexes(self, each_row):
		grasp_id = [i for i, name in enumerate(self.grasp_names) if each_row['Grasp'] == name]
		adl_id = [i for i, name in enumerate(self.adl_names) if each_row['ADL'] == name]
		opptype_id = [i for i, name in enumerate(self.opptype_names) if each_row['OppType'] == name]
		pip_id = [i for i, name in enumerate(self.pip_names) if each_row['PIP'] == name]
		virtual_fingers_id = [i for i, name in enumerate(self.virtual_fingers_names) if each_row['VirtualFingers'] == name]
		thumb_id = [i for i, name in enumerate(self.thumb_names) if each_row['Thumb'] == name]

		# Change order if you want it
		return [grasp_id[0], adl_id[0], opptype_id[0], pip_id[0], virtual_fingers_id[0], thumb_id[0]]


	# This function will be working on the dataset map function
	def _read_per_image_train(self, num):
		temp_num = np.int(num)

		current_image_name = self.train_meaningful_jpg_names[temp_num]

		filename = ('%s/%s/%s' % (self.data_path,
								  self.save_folder,
								  current_image_name))

		label_raw = self.find_label_from_filename(current_image_name)
		label = np.int64(self.get_label_indexes(label_raw[0]))

		if self.label_order is not None:
			label = label[self.label_order]

		img = imageio.imread(filename)
		img = cv2.resize(img, tuple(self.resize_image_size))
		img = np.float32(img) / 255.0

		return img, label

	# This function will be working on the dataset map function
	def _read_per_image_val(self, num):
		temp_num = np.int(num)

		current_image_name = self.val_meaningful_jpg_names[temp_num]

		filename = ('%s/%s/%s' % (self.data_path,
								  self.save_folder,
								  current_image_name))

		label_raw = self.find_label_from_filename(current_image_name)
		label = np.int64(self.get_label_indexes(label_raw[0]))

		if self.label_order is not None:
			label = label[self.label_order]

		img = np.float32(cv2.resize(cv2.imread(filename), tuple(self.resize_image_size))) / 255.0

		return img, label

	# This function will be working on the dataset map function
	def _read_per_image_test(self, num):
		temp_num = np.int(num)

		current_image_name = self.test_meaningful_jpg_names[temp_num]

		filename = ('%s/%s/%s' % (self.data_path,
								  self.save_folder,
								  current_image_name))

		label_raw = self.find_label_from_filename(current_image_name)
		label = np.int64(self.get_label_indexes(label_raw[0]))

		if self.label_order is not None:
			label = label[self.label_order]

		img = np.float32(cv2.resize(cv2.imread(filename), tuple(self.resize_image_size))) / 255.0

		return img, label



	def _adjust_tf_image_function(self, image, label):

		if self.max_hue_delta is not None:
			# random hue
			image = tf.image.random_hue(image, max_delta=self.max_hue_delta)

		if self.saturation_range is not None:
			# random saturation
			image = tf.image.random_saturation(image,
											   lower=self.saturation_range[0],
											   upper=self.saturation_range[1])

		if self.max_bright_delta is not None:
			# random brightness
			image = tf.image.random_brightness(image, max_delta=self.max_bright_delta)

		if self.max_contrast_delta is not None:
			image = tf.image.random_contrast(image,
											 lower=self.max_contrast_delta[0],
											 upper=self.max_contrast_delta[1])

		return image, label

	def initialization_dataset(self):

		with tf.name_scope('train_dataset'):
			train_size = len(self.train_meaningful_jpg_names)
			train_order = tf.random_shuffle(tf.linspace(0.0, (float(train_size) - 1.0), train_size))

			train_set = tf.data.Dataset.from_tensor_slices(train_order)
			train_set = train_set.map(lambda num: tuple(
				tf.py_func(self._read_per_image_train, [num], [tf.float32, tf.int64])))
			train_set = train_set.map(self._adjust_tf_image_function)

			train_set = train_set.batch(self.batch_size)

		with tf.name_scope('validation_dataset'):
			val_size = len(self.val_meaningful_jpg_names)
			## Validation set don't need shuffle.
			val_order = tf.linspace(tf.cast(val_size, tf.float32), (float(val_size) - 1.0), val_size)

			val_set = tf.data.Dataset.from_tensor_slices(val_order)
			val_set = val_set.map(lambda num: tuple(
				tf.py_func(self._read_per_image_val, [num, 'val'], [tf.float32, tf.int64])))

			val_set = val_set.batch(self.batch_size)

		with tf.name_scope('test_dataset'):
			test_size = len(self.test_meaningful_jpg_names)
			## Test set don't need shuffle.
			test_order = tf.linspace(tf.cast(test_size, tf.float32), (float(test_size) - 1.0), test_size)

			test_set = tf.data.Dataset.from_tensor_slices(test_order)
			test_set = test_set.map(lambda num: tuple(
				tf.py_func(self._read_per_image_test, [num, 'test'], [tf.float32, tf.int64])))

			test_set = test_set.batch(self.batch_size)

		with tf.name_scope('dataset_initializer'):
			# iterator = tf.data.Iterator.from_structure(train_set.output_types, train_set.output_shapes)
			iterator = tf.data.Iterator.from_structure(train_set.output_types,
			                                           (tf.TensorShape([self.batch_size,
			                                                            self.resize_image_size[0],
			                                                            self.resize_image_size[1],
			                                                            3]),
			                                            tf.TensorShape([self.batch_size,
			                                                            len(self.classes_numbers)])
			                                            )
			                                           )
			next_element = iterator.get_next()

			training_init_op = iterator.make_initializer(train_set, name='train_set_initializer')
			validation_init_op = iterator.make_initializer(val_set, name='val_set_initializer')
			test_init_op = iterator.make_initializer(test_set, name='test_set_initializer')

		return next_element, training_init_op, validation_init_op, test_init_op




