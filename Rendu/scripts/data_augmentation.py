from constants import *
import math
import numpy as np


def _get_crop(train, label, i, j):
    if train is not None:
        train_cropped = train[i:i + crop_size, j:j + crop_size]
    else:
        train_cropped = None
    if label is not None:
        label_cropped = label[i:i + crop_size, j:j + crop_size]
    else:
        label_cropped = None
    return train_cropped, label_cropped


def get_all_crops(train, label):
    """
    Returns all the crops of size (crop_size, crop_size) in the images
    :param train:
    :param label:
    :return:
    """
    rows = train.shape[0]
    cols = train.shape[1]
    nb_crops = math.ceil(rows / crop_size) * math.ceil(cols / crop_size)
    crops_x = np.zeros((nb_crops, crop_size, crop_size, 1))
    crops_y = np.zeros((nb_crops, crop_size, crop_size, 2))
    k = 0
    for i in range(0, rows - crop_size, crop_size):
        for j in range(0, cols - crop_size, crop_size):
            crops_x[k], crops_y[k] = _get_crop(train, label, i, j)
            k += 1
        # Last columns, not added yet.
        j = cols - crop_size
        crops_x[k], crops_y[k] = _get_crop(train, label, i, j)
        k += 1

    # Last rows, not added yet.
    i = rows - crop_size
    for j in range(0, cols - crop_size, crop_size):
        crops_x[k], crops_y[k] = _get_crop(train, label, i, j)
        k += 1
    # Last columns of last rows, not added yet.
    j = cols - crop_size
    crops_x[k], crops_y[k] = _get_crop(train, label, i, j)

    return crops_x, crops_y


def get_flips_images(train, label):
    """
    Returns the images after three mirror operations + the original image.
    """
    flips_train = np.zeros((4, train.shape[0], train.shape[1], train.shape[2]))
    flips_train[0] = train
    flips_train[1] = np.flip(train, axis=1)
    flips_train[2] = np.flip(train, axis=0)
    flips_train[3] = np.flip(flips_train[0], axis=1)
    flips_label = np.zeros((4, label.shape[0], label.shape[1], label.shape[2]))
    flips_label[0] = label
    flips_label[1] = np.flip(label, axis=1)
    flips_label[2] = np.flip(label, axis=0)
    flips_label[3] = np.flip(flips_label[0], axis=1)

    return flips_train, flips_label
