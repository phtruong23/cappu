import numpy as np
import tensorflow as tf
import cv2

class VideoLoader(object):
    """
        This class loads all frames of all video batch into an array
    """
    def __init__(self, filenames, labels, batch_size, input_size=None, crop_height=None, shuffle=True, is_mode='train'):

        self.filenames = filenames
        self.labels = labels
        self.input_size = input_size if input_size is not None else [320, 480]
        self.batch_size = batch_size
        self.num_samples = len(self.filenames)

        self.is_shuffle = shuffle
        self.is_mode = is_mode
        if self.is_mode=='train':
            self.do_preprocess = True
        else:
            self.do_preprocess = False

        # augmentation params
        self.brightness = 0.4
        self.contrast1 = 0.2
        self.contrast2 = 0.6
        self.saturation1 = 0.2
        self.saturation2 = 0.6
        self.hue = 0.25
        self.central_ratio = 0.7
        self.shift1 = -0.2
        self.shift2 = 0.2

    def read_data(self, index):

        video_files = self.filenames[index]
        labels = self.labels[index]

        video_data = []
        seq_len = []

        for video_file in video_files:

            cap = cv2.VideoCapture(video_file)
            frame_len = 0
            while (cap.isOpened()):
                ret, frame = cap.read()
                frame_len += 1

                video_data.append(frame)

            seq_len.append(frame_len)

            cap.release()

        return np.array(video_data), labels, seq_len

    def _preprocess_image(self, image, labels):

        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, self.brightness)
        image = tf.image.random_contrast(image, self.contrast1, self.contrast2)
        image = tf.image.random_saturation(image, self.saturation1, self.saturation2)
        image = tf.image.random_hue(image, self.hue)
        image = tf.image.central_crop(image, self.central_ratio)
        image = tf.image.resize_images(image, self.input_size)

        return tf.clip_by_value(image, 0.0, 1.0), labels

    def initialization(self):

        with tf.name_scope('load_dataset'):

            dataset = tf.linspace(0.0, (self.num_samples - 1), self.num_samples)
            dataset = tf.data.Dataset.from_tensor_slices(dataset)
            dataset = dataset.map(lambda data_num: tuple(tf.py_func(self.read_data,
                                                                    [data_num],
                                                                    [tf.float32, tf.int64, tf.int64])),
                                  num_parallel_calls=8)

            if self.do_preprocess:
                dataset = dataset.map(self._preprocess_image, num_parallel_calls=8)

            dataset = dataset.batch(self.batch_size)

        return dataset
