import cv2
import numpy as np


'''Pred and label are masks: 2d np array with type bool.'''

# True positive.
def TP(pred, label):
    return np.sum(pred & label)


# Precision = TP / (TP + FP).
def precision(pred, label):
    return TP(pred, label) / float(np.sum(pred))


# Recall = TP / (TP + FN).
def recall(pred, label):
    return TP(pred, label) / float(np.sum(label))


# F1 = 2 * (precision*recall) / (precision + recall).
def f1(pred, label):
    p = precision(pred, label)
    r = recall(pred, label)
    return 2 * p * r / (p+r)


def visualize_change(img, pred, label):
    img = np.copy(img)
    # TP = white.
    TP = pred & label
    img[TP] = 255
    # FP = blue.
    patch = img[(pred == 1) & (label == 0)] = np.array([255, 0, 0])
    # FN = red.
    patch = img[(pred == 0) & (label == 1)] = np.array([0, 0, 255])

    print 'Prevision', precision(pred, label)
    print 'Recall', recall(pred, label)
    print 'F1', f1(pred, label)
    return img
