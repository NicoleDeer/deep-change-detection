import cv2
import numpy as np
import tensorflow as tf

from metrics import visualize_change
from tf_unet import unet, util
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

        batch_x, batch_y, batch_img = get_test_image('/home/onera', 'hongkong')
        #print batch_y[..., 0]
        #print batch_y[..., 1]

        prediction = sess.run(net.predicter, feed_dict={net.x: batch_x,
            net.y: batch_y, net.keep_prob: 1.})
        batch_img = util.crop_to_shape(batch_img, prediction.shape)
        batch_y = util.crop_to_shape(batch_y, prediction.shape).astype(np.bool)
        #print prediction[:, :, :, 0]
        #print prediction[:, :, :, 1]
        #print np.sum(prediction[:, :, :, 0])
        #print np.sum(np.argmax(prediction, 3))
        
        #prediction = np.argmax(np.squeeze(prediction, axis=0), 2)*255.0
        #prediction = np.repeat(prediction[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
        pred = np.argmax(np.squeeze(prediction, axis=0), 2).astype(np.bool)
        
        #prediction = prediction[0, :, :, 1]*255.0
        #prediction = np.repeat(prediction[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
        #print(prediction.shape)
        
        cv2.imshow('image', visualize_change(batch_img[0, ...], pred, batch_y[0, ..., 1]))
        cv2.waitKey(0)


if __name__ == '__main__':
    predict(model_path='models')
