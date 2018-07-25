class PARAMS(object):
	# define all paths here
	csv_path = '../Gopro'
	csv_filename = 'SDATA1700291_annotated_data.csv'
	save_folder = 'save_frames'

	train_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
	val_list = [9]
	test_list = [10, 11, 12]

	max_hue_delta = 0.15
	saturation_range = [0.5, 2.0]
	max_bright_delta = 0.25
	max_contrast_delta = [0, 0.3]

	resnet_version = 152
	resnet_path = ('./resnet_v2_%d_pretrained/resnet_v2_%d.ckpt') % (resnet_version, resnet_version)
	resnet_exclude = ''
	trainable_scopes = 'stage_0,stage_1,stage_2,stage_3,stage_4,stage_5'
	ignore_missing_vars = True

	# set test_eval to True to run the test set
	evaluate = True
	test_eval = False
	save_preds = True

	# define all required parameters here
	image_size = [224, 224]
	batch_size = 30
	epochs = 30
	learning_rate = 0.001
	learning_rate_decay_factor = 0.94
	beta = 0.0002
	epoch_decay = 4
	weight_decay = 1e-4
	print_freq = 10

	thread = 8
	gpu_num = '0'