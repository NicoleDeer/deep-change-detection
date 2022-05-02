import cv2
import numpy as np
import tensorflow as tf

from tf_unet import unet
from utils import get_test_image

def predict(model_path):
    net = unet.Unet(channels=6, n_class=2, 
            layers=3, features_root=64,
            cost_kwargs=dict(regularizer=0.001),
        )

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(model_path)
        assert ckpt and ckpt.model_checkpoint_path
        net.restore(sess, ckpt.model_checkpoint_path)

        batch_x, batch_y = get_test_image('/home/onera', 'hongkong')
        print batch_y[..., 0]
        print batch_y[..., 1]
        prediction = sess.run(net.predicter, feed_dict={net.x: batch_x,
            net.y: batch_y, net.keep_prob: 1.})
        print prediction[:, :, :, 0]
        print prediction[:, :, :, 1]
        print np.sum(prediction[:, :, :, 0])
        print np.sum(np.argmax(prediction, 3))
        prediction = np.argmax(np.squeeze(prediction, axis=0), 2)*255.0
        prediction = np.repeat(prediction[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
        #prediction = prediction[0, :, :, 1]*255.0
        #prediction = np.repeat(prediction[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
        print(prediction.shape)
        cv2.imshow('image', prediction)
        cv2.waitKey(0)


if __name__ == '__main__':
    predict(model_path='models')
