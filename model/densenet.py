"""
make the densent tensorflow model
This model output map size = input map size /(2^dense_block_number)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.framework import arg_scope

def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride, padding='SAME')
        return network

def Global_Average_Pooling(x, stride=1):
    width = np.shape(x)[1]
    height = np.shape(x)[2]
    pool_size = [width, height]
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride) # The stride value does not matter

    """
    It is global average pooling without tflearn

    return global_avg_pool(x, name='Global_avg_pooling')
    # But maybe you need to install h5py and curses or not
    """

def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))

def Drop_out(x, rate, training) :
    return tf.layers.dropout(inputs=x, rate=rate, training=training)

def Relu(x):
    return tf.nn.relu(x)

def Average_pooling(x, pool_size=[2,2], stride=2, padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Max_Pooling(x, pool_size=[3,3], stride=2, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Concatenation(layers) :
    return tf.concat(layers, axis=3)

def Linear(x, output_dim, name) :
    return tf.layers.dense(inputs=x, units=output_dim, name=name)

def create_densenet_info(nb_blocks_layers = [6, 12, 48, 32], filters = 24,
                        output_dim = 1000):
    if not isinstance(type(nb_blocks_layers), list):
        tf.logging.fatal('DenseNet: nb_blocks_layers does not list type.')
    if not isinstance(type(filters), int):
        tf.logging.fatal('DenseNet: filters does not int type.')
    if not isinstance(type(output_dim), int):
        tf.logging.fatal('DenseNet: filters does not int type.')

    return { 'nb_blocks_layers': nb_blocks_layers,
             'filters': filters,
             'output_dim': output_dim}

class DenseNet():
    def __init__(self, x, training, dropout_rate, densenet_info):
        self.training = training
        if not isinstance(type(self.training), tf.placeholder(tf.bool)):
            tf.logging.fatal('DenseNet: training_flag does not bool type.')
        self.dropout_rate = dropout_rate
        if not isinstance(type(self.dropout_rate), float):
            tf.logging.fatal('DenseNet: dropout_rate does not bool type.')
        self.densenet_info = densenet_info
        self.nb_blocks_layers = densenet_info['nb_blocks_layers']
        self.filters = densenet_info['filters']
        self.output_dim = densenet_info['output_dim']
        self.model = self.Dense_net(x)

    def bottleneck_layer(self, x, scope):
        # print(x)
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.training,
                                            scope=scope+'_batch1')
            x = Relu(x)
            x = conv_layer(x, filter=4 * self.filters, kernel=[1,1],
                                            layer_name=scope+'_conv1')
            x = Drop_out(x, rate=dropout_rate, training=self.training)

            x = Batch_Normalization(x, training=self.training,
                                        scope=scope+'_batch2')
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[3,3],
                                        layer_name=scope+'_conv2')
            x = Drop_out(x, rate=dropout_rate, training=self.training)

            # print(x)

            return x

    def transition_layer(self, x, scope):
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.training,
                                        scope=scope+'_batch1')
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[1,1],
                                        layer_name=scope+'_conv1')
            x = Drop_out(x, rate=dropout_rate, training=self.training)
            x = Average_pooling(x, pool_size=[2,2], stride=2)

            return x

    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            layers_concat = list()
            layers_concat.append(input_x)

            x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_'
                                                                    + str(0))

            layers_concat.append(x)

            for i in range(nb_layers - 1):
                x = Concatenation(layers_concat)
                x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_'
                                                                + str(i + 1))
                layers_concat.append(x)

            x = Concatenation(layers_concat)

            return x

    def Dense_net(self, input_x):
        x = conv_layer(input_x, filter=2 * self.filters, kernel=[7,7], stride=2,
                                                            layer_name='conv0')
        # x = Max_Pooling(x, pool_size=[3,3], stride=2)

        for i, nb_layers in enumerate(self.nb_blocks_layers, 1):
            if not isinstance(type(nb_layers), int):
                tf.logging.fatal('DenseNet: nb_layers does not int type.')
            # 6 -> 12 -> 48
            x = self.dense_block(input_x=x, nb_layers= nb_layers, layer_name='dense_'
                                                                    +str(i))
            x = self.transition_layer(x, scope='trans_'+str(i))

        # 100 Layer
        x = Batch_Normalization(x, training=self.training, scope='linear_batch')
        x = Relu(x)
        x = Global_Average_Pooling(x)
        x = flatten(x)
        x = Linear(x, self.output_dim, 'merge_linear')

        # x = tf.reshape(x, [-1, 10])
        return x
