import numpy as np
import tensorflow as tf


def loss_tower(heatmaps_true, heatmaps_pred, name='mse'):

    if name == 'mse':
        # batch_size = heatmaps_true.get_shape()[0]
        loss = tf.losses.mean_squared_error(labels=heatmaps_true, predictions=heatmaps_pred, scope=name)
        tf.add_to_collection('losses', loss)
    elif name == 'cross_entropy':
        heatmaps_true = tf.cast(heatmaps_true, tf.int64)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=heatmaps_true,
                                                              logits=heatmaps_pred,
                                                              name=name)
        loss_mean = tf.reduce_mean(loss, name='cross_entropy_mean')
        tf.add_to_collection('losses', loss_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')