import cv2
import logging
import numpy as np
import os
import pandas as pd
import random
from random import randint
import sys
import tifffile


nr_epoches = 1000
batch_size = 64
patch_size = 144


def imread(img_path):
    return cv2.imread(img_path)


def resize(im, shape):
    return cv2.resize(im, shape, interpolation=cv2.INTER_CUBIC)


def rand_crop(img, mask, width, height):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    assert img.shape[0] == mask.shape[0]
    assert img.shape[1] == mask.shape[1]
    x = randint(0, img.shape[1] - width)
    y = randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    mask = mask[y:y+height, x:x+width]
    return img, mask


class ImageData():

    def __init__(self, data_dir, img_name):
        #self.stat = image_stat(self.image_id)
        
        #image_feature.append(image_data.train_feature[: x_crop, : y_crop, :])
        #image_label.append(image_data.label[: x_crop, : y_crop, :])
        
        # Image pairs to detect changes.
        # RGB images.
        b3_imgs = [imread(os.path.join(data_dir, 'images', img_name, 'pair',
            'img{}.png'.format(i))) for i in [1, 2]]
        self.feature = np.concatenate(b3_imgs, axis=2)/255.0

        # Label is a change map.
        self.label = imread(os.path.join(data_dir, 'labels', img_name, 'cm',
            'cm.png'))[:, :, 0][:, :, np.newaxis]/255.0

        self.img_size = self.label.shape

        self.nr_total_patches = self.img_size[0]*self.img_size[1]/float(
            144*144)
        self.nr_patches = None

        # To add more channel in future.
        self.b4_imgs = None

    
    def get_data(self):
        return self.feature, self.label


def get_all_data(data_dir, train):
    imgs = []

    try:
        with open(os.path.join(data_dir, 'images', 'train.txt' if train else
                'test.txt')) as f:
            img_names = f.readline().strip('\n').split(',')
    except:
        logging.fatal('Data information not found.')

    logging.info('Loading data.')
    total_nr_patcehs = 0
    for i, img_name in enumerate(img_names):
        img_data = ImageData(data_dir, img_name)
        imgs.append(img_data)
        total_nr_patcehs += img_data.nr_total_patches

    return imgs, total_nr_patcehs


def shuffle_two_lists(a, b):
    z = zip(a, b)
    random.shuffle(z)
    a[:], b[:] = zip(*z)


def gen_train_batch(data_dir, train=True):
    imgs, total_nr_patcehs = get_all_data(data_dir, train)
    i_batch = 0

    while i_batch < nr_epoches:
        features = []
        labels = []
        for img_data in imgs:
            nr_patches = int(img_data.nr_total_patches/total_nr_patcehs*batch_size)
            for _ in range(nr_patches):
                img, label = rand_crop(img_data.feature, img_data.label,
                    patch_size, patch_size)
                features.append(process_img(img))
                labels.append(process_labels(label))
        shuffle_two_lists(features, labels)
        yield np.stack(features, axis=0), np.stack(labels, axis=0)


def get_test_image(data_dir, img_name):
    img_data = ImageData(data_dir, img_name)
    return process_img(img_data.feature)[np.newaxis, ...], \
        process_labels(img_data.label)[np.newaxis, ...]


def process_img(img):
    # normalization
    #data = np.clip(data, v_min, v_max)
    img -= np.amin(img)
    amax = np.amax(img)
    if amax != 0:
        img /= amax
    return img


def process_labels(label):
    assert label.shape[2] == 1
    label = label.astype(np.bool)
    return np.concatenate([~label, label], axis=2).astype(np.float)
