import tensorflow as tf
import numpy as np


def _variable_initial(name, shape, initializer=tf.truncated_normal_initializer, stddev=5e-2):
    init = initializer(stddev=stddev, dtype=tf.float32)
    weights = tf.get_variable(name=name, shape=shape, initializer=init, dtype=tf.float32, trainable=True)
    return weights


def conv3x3(input_tensor, out_planes, blk_name, stride=1, padding=1, bias=False):
    # sess = tf.Session()
    # size = sess.run(tf.shape(input_tensor))
    # in_planes = size[3]
    kernel = _variable_initial(blk_name,[3, 3,input_tensor.get_shape()[3], out_planes])
    stride = [stride, stride, stride, stride]
    if padding == 1:
        conv = tf.nn.conv2d(input_tensor, kernel, strides=stride, padding='SAME')
    else:
        conv = tf.nn.conv2d(input_tensor, kernel, strides=stride, padding='VALID')

    if bias:
        biases = tf.get_variable(name=blk_name + "b1", shape=[out_planes], initializer=tf.zeros_initializer())
        return tf.nn.bias_add(conv, biases)
    else:
        return conv


class ConvBlock:
    def __init__(self, out_planes, name_base):
        self._out_planes = out_planes
        self._name_base = name_base

    def inference(self, input_tensor):
        '''
        :param input_size: tuple (x, y, c)
        :return: tensor
        '''
        with tf.variable_scope(self._name_base) as scope:
            out1 = tf.layers.batch_normalization(input_tensor)
            out1 = tf.nn.relu(out1)
            out1 = conv3x3(out1, int(self._out_planes / 2), 'weight0')

            out2 = tf.layers.batch_normalization(out1)
            out2 = tf.nn.relu(out2)
            out2 = conv3x3(out2, int(self._out_planes / 4), 'weight1')

            out3 = tf.layers.batch_normalization(out2)
            out3 = tf.nn.relu(out3)
            out3 = conv3x3(out3, int(self._out_planes / 4), 'weight2')

            out3 = tf.concat([out1, out2, out3], axis=3)

            if input_tensor.get_shape()[3] != self._out_planes:
                bn = tf.layers.batch_normalization(input_tensor)
                conv_downsample = tf.nn.relu(
                    tf.nn.conv2d(bn, _variable_initial('downsample', [1, 1, bn.get_shape()[3], self._out_planes]),
                                 strides=[1, 1, 1, 1], padding='VALID')
                )
                out3 = tf.add(out3, conv_downsample)
            return out3


class BottleNeck:
    def __init__(self, planes, name_base, stride=1, downsample=None):
        self._planes = planes
        self._stride = stride
        self._downsample = downsample
        self._name_base = name_base

    def inference(self, input_tensor):
        with tf.variable_scope(self._name_base) as scope:
            stride = [self._stride, self._stride, self._stride, self._stride]
            out1 = tf.nn.conv2d(input_tensor,
                                _variable_initial('0', [1, 1, input_tensor.get_shape()[3], self._planes]),
                                strides=stride, padding='SAME')
            out1 = tf.layers.batch_normalization(out1)
            out1 = tf.nn.relu(out1)

            out2 = tf.nn.conv2d(out1,
                                _variable_initial('1',[3, 3, self._planes, self._planes]),
                                strides=stride,
                                padding='SAME')
            out2 = tf.layers.batch_normalization(out2)
            out2 = tf.nn.relu(out2)

            out3 = tf.nn.conv2d(out2,
                                _variable_initial('2',[1, 1, self._planes, 4 * self._planes]),
                                strides=stride,
                                padding='SAME')
            out3 = tf.layers.batch_normalization(out3)

            if self._downsample is not None:
                conv_downsample = tf.layers.batch_normalization(input_tensor)
                conv_downsample = tf.nn.relu(
                    tf.nn.conv2d(conv_downsample,
                                 _variable_initial('downsample',[1, 1, conv_downsample.get_shape()[3], 4 * self._planes]),
                                 strides=stride,
                                 padding='SAME')
                )
                out3 = tf.add(out3, conv_downsample)
                out3 = tf.nn.relu(out3)

            return out3


class HourGlass:
    def __init__(self, num_modules, depth, num_features, name_base):
        self.num_modules = num_modules
        self.depth = depth
        self.num_features = num_features
        self.name_base = name_base

    def _inference(self, level, input_tensor):
        conv_block1 = ConvBlock(256, self.name_base + str(level) + '0')
        up1 = conv_block1.inference(input_tensor)

        low1 = tf.nn.avg_pool(input_tensor, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        conv_block1 = ConvBlock(256, self.name_base + str(level) + '1')
        low1 = conv_block1.inference(low1)

        if level > 1:
            low2 = self._inference(level - 1, low1)
        else:
            conv_block3 = ConvBlock(256, self.name_base + str(level) + '2')
            low2 = conv_block3.inference(low1)

        conv_block3 = ConvBlock(256, self.name_base + str(level) + '3')
        low3 = conv_block3.inference(low2)
        upsampling2D_layer = tf.keras.layers.UpSampling2D((2, 2))
        up2 = upsampling2D_layer.call(low3)

        return tf.add(up1, up2)

    def inference(self, input_tensor):
        return self._inference(self.depth, input_tensor)


class FAN:
    def __init__(self, num_modules=1, facial2d_landmarks=68):
        self.num_modules = num_modules
        self.facial2d_landmarks = facial2d_landmarks

    def inference(self, input_tensor):
        padded_input_tensor = tf.pad(input_tensor, tf.constant([[0, 0], [3, 3],[3, 3],[0, 0]]))
        out = tf.nn.relu(
            tf.layers.batch_normalization(
                tf.nn.conv2d(padded_input_tensor,
                             filter=_variable_initial('fan_1', shape=[7, 7, input_tensor.get_shape()[3], 64]),
                             strides=[1, 2, 2, 1],
                             padding='VALID'))
        )
        cb1 = ConvBlock(128, 'fan_2')
        out = cb1.inference(out)
        out = tf.nn.avg_pool(out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        cb2 = ConvBlock(128, 'fan_3')
        out = cb2.inference(out)
        cb3 = ConvBlock(256, 'fan_4')
        out = cb3.inference(out)

        outputs = []
        for i in range(self.num_modules):
            hg1 = HourGlass(1, 4, 256, 'fan_modules_hg' + str(i))
            out1 = hg1.inference(out)
            cb3 = ConvBlock(256, 'fan_modules_cb' + str(i))
            out1 = cb3.inference(out1)

            ll = tf.nn.relu(
                tf.layers.batch_normalization(
                    tf.nn.conv2d(out1,
                                 filter=_variable_initial('fan_cv1'+str(i), [1, 1, 256, 256]),
                                 strides=[1, 1, 1, 1],
                                 padding='SAME')
                )
            )

            tmp_out = tf.nn.conv2d(ll,
                                   filter=_variable_initial('fan_cv2'+str(i), [1, 1, 256, self.facial2d_landmarks]),
                                   strides=[1, 1, 1, 1],
                                   padding='SAME')
            outputs.append(tmp_out)

            if i < self.num_modules - 1:
                out2 = tf.nn.conv2d(out,
                                    filter=_variable_initial('fan_cv3'+str(i), [1, 1, 256,  256]),
                                    strides=[1, 1, 1, 1],
                                    padding='SAME')
                tmp_out = tf.nn.conv2d(tmp_out,
                                       filter=_variable_initial('fan_cv4'+str(i), [1, 1, self.facial2d_landmarks,  256]),
                                       strides=[1, 1, 1, 1],
                                       padding='SAME')
                tmp_add = tf.add(out2, tmp_out)
                out = tf.add(out, tmp_add)
        # new_shape = np.ones(5, dtype=int)
        # new_shape[1:] = outputs.get_shape()
        # outputs = tf.reshape(outputs, new_shape.tolist())
        return outputs
