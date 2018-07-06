import tensorflow as tf
import dlib

from face_alignment import loss
from utils import muct_dataset
from face_alignment import model


def train_opt(loss_op, lr=0.05):
    optimizer = tf.train.RMSPropOptimizer(lr)
    return optimizer.minimize(loss_op)


if __name__ == '__main__':
    detector = dlib.get_frontal_face_detector()


    image_root = '../tests/assets/test_dataset'
    training_csv = 'training.csv'
    input_resolution = 256
    output_resolution = 64
    fan = model.FAN(num_modules=1, facial2d_landmarks=76)

    x_holder = tf.placeholder(dtype=tf.float32, shape=[None, input_resolution, input_resolution, 3])
    y_holder = tf.placeholder(dtype=tf.float32, shape=[None, output_resolution, output_resolution, 76])

    pred_opt = fan.inference(x_holder)
    loss_opt = loss.loss_tower(y_holder, pred_opt[-1])
    train_op = train_opt(loss_opt)
    batch = 8
    epoches = 40
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(epoches):
            batch_generator = muct_dataset.get_training_batch(image_root, training_csv, detector,
                                                              batch_size=batch)
            count = 0
            while True:
                try:
                    # print('Start train {} batch of {}th epoch'.format(count, i))
                    res = next(batch_generator)
                    inputs, outputs = res[0], res[1]
                    sess.run(train_op, feed_dict={x_holder: inputs, y_holder: outputs})
                    pred = sess.run(pred_opt, feed_dict={x_holder: inputs})
                    pred = pred[-1]
                    loss = sess.run(loss_opt, feed_dict={x_holder: inputs, y_holder: outputs})
                    print('Finish train {} batch of {}th epoch; current loss is {}'.format(count, i, loss))
                    count += 1
                except StopIteration:
                    print('End of dataset')
                    break


