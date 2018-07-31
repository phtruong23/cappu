import numpy as np
import tensorflow as tf
import cv2
import datetime
import VideoLoader


def main():

    files = '/media/a307/EXPERIMENT/Grasping/Dataset/Gopro/1. Subject 1/subject_1_gopro_seg_1.mp4'
    train_dataset = VideoLoader.VideoLoader(files, 1, 1)

    train_generator = train_dataset.initialization()

    with tf.name_scope('data_generator'):
        iterator = tf.data.Iterator.from_structure(train_generator.output_types, 
                                                   (tf.TensorShape([train_dataset.batch_size,
                                                                    train_dataset.img_size[0],
                                                                    train_dataset.img_size[1],
                                                                    3]),
                                                    tf.TensorShape([train_dataset.batch_size,
                                                                    train_dataset.labels.shape[1]])))

        images, labels, seq_len = iterator.get_next()

        train_init_op = iterator.make_initializer(train_generator)

    # configure GPU to run training
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.gpu_options.visible_device_list = '0'

    with tf.Session(config=config) as sess:

        sess.run([tf.global_variables_initializer()])
        sess.run(train_init_op)

        now = datetime.datetime.now()
        for i in range(10):

            while True:
                try:
                    sess.run({'image': images, 'label': labels})
                except tf.errors.OutOfRangeError:
                    break

        print('Training time: {}'.format(datetime.datetime.now() - current_time))


if __name__ == '__main__':
    main()