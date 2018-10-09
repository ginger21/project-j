'''
creat inception using tensorflow
'''
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python.ops import array_ops
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib

def inception_g(inputs,
                num_classes = 5,
                dropout_keep_prob = 0.8,
                prediction_fn = layers_lib.softmax,
                scope='InceptionG'
                ):
    end_points = {}

    # conv2d
    with arg_scope(
        [layers.conv2d, layers_lib.max_pool2d, layers_lib.avg_pool2d],
        stride=1,
        padding='VALID'):
        # 48 x 48 x 1 
        end_point = 'Conv2d_1a_3x3'
        net = layers.conv2d(inputs, 32, [3, 3], stride=2, scope=end_point)
        end_points[end_point] = net
        # 23 x 23 x 32
        end_point = 'Conv2d_2a_3x3'
        net = layers.conv2d(net, 64, [3, 3], scope=end_point)
        end_points[end_point] = net

        # 21 x 21 x 64
        end_point = 'MaxPool_3a_3x3'
        net = layers_lib.max_pool2d(net, [3, 3], stride=2, scope=end_point)
        end_points[end_point] = net

        # 10 x 10 x 64
        end_point = 'Conv2d_3b_1x1'
        net = layers.conv2d(net, 80, [1, 1], scope=end_point)
        end_points[end_point] = net

        # 8 x 8 x 80
        end_point = 'Conv2d_4a_3x3'
        net = layers.conv2d(net, 128, [3, 3], scope=end_point)
        end_points[end_point] = net
    
    # Inception blocks
    with arg_scope(
        [layers.conv2d, layers_lib.max_pool2d, layers_lib.avg_pool2d],
        stride=1,
        padding='SAME'):
        # 8 x 8 x 144
        end_point = 'Mixed_5a'
        branch_0 = layers.conv2d(
            net, 64, [1, 1], scope='Conv2d_0a_1x1')
        branch_1 = layers.conv2d(
            net, 48, [1, 1], scope='Conv2d_1a_1x1')
        branch_1 = layers.conv2d(
            branch_1, 64, [5, 5], scope='Conv2d_1b_5x5')
        branch_2 = layers.conv2d(
            net, 64, [1, 1], scope='Conv2d_2a_1x1')
        branch_2 = layers.conv2d(
            branch_2, 96, [3, 3], scope='Conv2d_2b_3x3')
        branch_2 = layers.conv2d(
            branch_2, 96, [3, 3], scope='Conv2d_2c_3x3')
        branch_3 = layers_lib.avg_pool2d(net, [3, 3], scope='AvgPool_3a_3x3')
        branch_3 = layers.conv2d(
            branch_3, 64, [1, 1], scope='Conv2d_3c_1x1')
        net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
        end_points[end_point] = net
    
    # aver pooling and softmax
    net = layers_lib.avg_pool2d(
            net,
            [8,8],
            padding='VALID',
            scope='AvgPool_0a_7x7')
    net = layers_lib.dropout(
            net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
    end_points['PreLogits'] = net
    logits = layers.conv2d(
            net,
            num_classes,    # num classes
            [1, 1],
            activation_fn=None,
            normalizer_fn=None,
            scope='Conv2d_1c_1x1')
    logits = array_ops.squeeze(logits, [1, 2], name='SpatialSqueeze')

    end_points['Logits'] = logits
    end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
    return logits, end_points