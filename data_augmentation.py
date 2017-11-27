import math
import numpy as np


def get_random_crops(x, y, nb_crops=4, crop_size=224):
    crops_x = np.zeros((nb_crops, crop_size, crop_size, 1))
    crops_y = np.zeros((nb_crops, crop_size, crop_size, 1))
    for i in range(0, nb_crops):
        row = np.random.randint(0, x.shape[0] - crop_size)
        col = np.random.randint(0, x.shape[1] - crop_size)
        crops_x[i] = x[row:row + crop_size, col:col + crop_size]
        crops_y[i] = y[row:row + crop_size, col:col + crop_size]
    return crops_x, crops_y


def get_all_crops(x, y, crop_size=224):
    rows = x.shape[0]
    cols = x.shape[1]
    nb_crops = math.ceil(rows / crop_size) * math.ceil(cols / crop_size)
    crops_x = np.zeros((nb_crops, crop_size, crop_size, 1))
    crops_y = np.zeros((nb_crops, crop_size, crop_size, 1))
    k = 0
    for i in range(0, rows - crop_size, crop_size):
        for j in range(0, cols - crop_size, crop_size):
            crops_x[k] = x[i:i + crop_size, j:j + crop_size]
            crops_y[k] = y[i:i + crop_size, j:j + crop_size]
            k += 1
        # Last columns, not added yet.
        j = cols - crop_size
        crops_x[k] = x[i:i + crop_size, j:j + crop_size]
        crops_y[k] = y[i:i + crop_size, j:j + crop_size]
        k += 1

    # Last rows, not added yet.
    i = rows - crop_size
    for j in range(0, cols - crop_size, crop_size):
        crops_x[k] = x[i:i + crop_size, j:j + crop_size]
        crops_y[k] = y[i:i + crop_size, j:j + crop_size]
        k += 1
    # Last columns of last rows, not added yet.
    j = cols - crop_size
    crops_x[k] = x[i:i + crop_size, j:j + crop_size]
    crops_y[k] = y[i:i + crop_size, j:j + crop_size]

    return crops_x, crops_y


def get_mirrored_images(actin, axon, dendrite):
    """
    Returns 3 tuples of images, each mirrored differently (only horizontal, only vertical, and both).
    """
    horizontal = np.flip(actin, axis=1), np.flip(axon, axis=1), np.flip(dendrite, axis=1)
    vertical = np.flip(actin, axis=0), np.flip(axon, axis=0), np.flip(dendrite, axis=0)
    both = np.flip(horizontal[0], axis=0), np.flip(horizontal[1], axis=0), np.flip(horizontal[2], axis=0)
    return horizontal, vertical, both
