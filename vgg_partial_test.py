import tensorflow as tf
import numpy as np
import partial_vgg
import handle_network_function

slim = tf.contrib.slim

# fake_input = np.ones(shape=(10, 28, 28, 256), dtype=np.float32)

fake_input = tf.random_normal([10, 28, 28, 256], mean=-1, stddev=4)

with slim.arg_scope(partial_vgg.vgg_arg_scope()):
    vgg19_partial_1 = partial_vgg.vgg_partial(inputs=fake_input,
                                                      num_classes=1000,
                                                      is_training=False,
                                                      dropout_keep_prob=1.0,
                                                      spatial_squeeze=True,
                                                      scope='vgg_19_1',
                                                      fc_conv_padding='VALID',
                                                      global_pool=False
                                                      )
with slim.arg_scope(partial_vgg.vgg_arg_scope()):
    vgg19_partial_2 = partial_vgg.vgg_partial(inputs=fake_input,
                                                      num_classes=1000,
                                                      is_training=False,
                                                      dropout_keep_prob=1.0,
                                                      spatial_squeeze=True,
                                                      scope='vgg_19_2',
                                                      fc_conv_padding='VALID',
                                                      global_pool=False
                                                      )

vgg19_exclude_scopes = 'vgg_19/conv1,vgg_19/conv2,vgg_19/conv3,vgg_19/fc6,vgg_19/fc7,vgg_19/fc8'


vgg19_restore_1 = handle_network_function._get_init_fn('./vgg_19_pretrained/vgg_19.ckpt',
                                                     '', vgg19_exclude_scopes)

vgg19_restore_2 = handle_network_function._get_init_fn('./vgg_19_pretrained/vgg_19.ckpt',
                                                     '', vgg19_exclude_scopes)


config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.gpu_options.visible_device_list = '0'

with tf.Session(config=config) as sess:

    sess.run(tf.global_variables_initializer())

    vgg19_restore_1(sess)
    vgg19_restore_2(sess)

    # temp_test = sess.run({'vgg19_end_points': vgg19_end_points['vgg_19/fc8']})
    # temp_test = sess.run({'vgg19_end_points': vgg19_end_points['global_pool']})

    temp_test = sess.run({'vgg19_partial_1': vgg19_partial_1, 'vgg19_partial_2': vgg19_partial_2})

    # print(np.shape(temp_test['vgg19_end_points']))
    # print(temp_test['vgg19_end_points'])
    print(np.shape(vgg19_partial_1))
    print(np.shape(vgg19_partial_2))


