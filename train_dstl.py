import datetime
import os
import simplejson
import sys
import time

import tensorflow as tf

from dstl_unet.unet import build
from tf_unet import util
from utils import gen_train_batch, get_test_image


data_dir = '/home/onera'

if __name__ == '__main__':
    hypes = './hypes/hypes.json'
    with open(hypes, 'r') as f:
        H = simplejson.load(f)
        # H['loss_function'] = 'dice'
        im_width = H['im_width']
        im_height = H['im_height']
        num_class = H['num_class']
        num_channel = H['num_channel']
        queue_size = H['queue_size']
        save_iter = H['save_iter']
        print_iter = H['print_iter']
        class_type = H['class_type']
        train_iter = H['train_iter']
        lr = H['lr']
        lr_decay_iter = H['lr_decay_iter']
        log_dir = H['log_dir']
        batch_size = H['batch_size']

    data_provider = gen_train_batch(data_dir)

    now = datetime.datetime.now()
    now_path = str(now.month) + '-' + str(now.day) + '_' + \
               str(now.hour) + '-' + str(now.minute) + '_' + H['loss_function']

    print('checkpoint name :{}'.format(now_path))

    ckpt_path = os.path.join(log_dir, now_path, 'ckpt', 'ckpt')
    hypes_path = os.path.join(log_dir, now_path, 'hypes')
    summary_path = os.path.join(log_dir, now_path, 'summary')

    for path in [ckpt_path, hypes_path, summary_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    x_in, y_in, loss, accuracy, train_op, summary_op, learning_rate, global_step = \
        build(H)

    print('Training parameters: {}\n'.format(H))

    with open(os.path.join(hypes_path, 'hypes.json'), 'w') as f:
        simplejson.dump(H, f)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    coord = tf.train.Coordinator()
    threads = {}
    saver = tf.train.Saver(max_to_keep=train_iter / save_iter + 1)

    with tf.Session(config=config).as_default() as sess:
        summary_writer = tf.summary.FileWriter(logdir=summary_path, flush_secs=10)
        summary_writer.add_graph(sess.graph)

        sess.run(tf.global_variables_initializer())

        start = time.time()
        for step in xrange(train_iter):
            batch_x, batch_y = next(data_provider)
            batch_y = util.crop_to_shape(batch_y, (56, 104, 104, 1))

            if step and step % lr_decay_iter == 0:
                lr *= 0.1

            if step % print_iter == 0 or step == (train_iter - 1):
                dt = (time.time() - start) / batch_size / print_iter
                start = time.time()

                _, train_loss, train_accuracy, validate_loss, \
                validate_accuracy, summaries = \
                    sess.run([train_op, loss['train'],
                                accuracy['train'], loss['validate'],
                                accuracy['validate'], summary_op],
                             feed_dict={learning_rate: lr,
                                x_in['train']: batch_x,
                                y_in['train']: batch_y})
                summary_writer.add_summary(
                    summaries, global_step=global_step.eval())
                print('Global step ({0}): LR: {1:0.5f}; '.format(global_step.eval(), lr))
                print('Train loss {0:.2f}; '.format(train_loss))
                print('Train accuracy {}%; '.format(int(100 * train_accuracy)))
                print('Validate loss {0:.2f}; '.format(validate_loss))
                print('Validate accuracy {}%; '.format(int(100 * validate_accuracy)))
            else:
                sess.run([train_op, loss['train']], feed_dict={learning_rate: lr,
                    x_in['train']: batch_x,
                    y_in['train']: batch_y})

            if step % save_iter == 0 or step == (train_iter - 1):
                saver.save(sess, ckpt_path, global_step=global_step.eval())
