import math
import numpy as np

# TODO: for now, only dendrites used as label. But later axons will be used too, so we'll be able to remove all the "If _ is not None: ..."

# def get_random_crops(actin, axon, dendrite, nb_crops=4, crop_size=224):
#     crops_actin = np.zeros((nb_crops, crop_size, crop_size, 1))
#     crops_axon = np.zeros((nb_crops, crop_size, crop_size, 1))
#     crops_dendrite = np.zeros((nb_crops, crop_size, crop_size, 1))
#     rows = actin.shape[0]
#     cols = actin.shape[1]
#     for i in range(0, nb_crops):
#         row = np.random.randint(0, rows - crop_size)
#         col = np.random.randint(0, cols - crop_size)
#         crops_actin[i] = actin[row:row + crop_size, col:col + crop_size]
#         crops_axon[i] = axon[row:row + crop_size, col:col + crop_size]
#         crops_dendrite[i] = dendrite[row:row + crop_size, col:col + crop_size]
#     return crops_actin, crops_axon, crops_dendrite


def _get_crop(actin, axon, dendrite, crops_actin, crops_axon, crops_dendrite, k, i, j, crop_size):
    if actin is not None:
        crops_actin[k] = actin[i:i + crop_size, j:j + crop_size]
    if axon is not None:
        crops_axon[k] = axon[i:i + crop_size, j:j + crop_size]
    if dendrite is not None:
        crops_dendrite[k] = dendrite[i:i + crop_size, j:j + crop_size]
    return crops_actin, crops_axon, crops_dendrite


def get_all_crops(actin, axon, dendrite, crop_size=224):
    rows = actin.shape[0]
    cols = actin.shape[1]
    nb_crops = math.ceil(rows / crop_size) * math.ceil(cols / crop_size)
    crops_actin = np.zeros((nb_crops, crop_size, crop_size, 1))
    crops_axon = np.zeros((nb_crops, crop_size, crop_size, 1))
    crops_dendrite = np.zeros((nb_crops, crop_size, crop_size, 1))
    k = 0
    for i in range(0, rows - crop_size, crop_size):
        for j in range(0, cols - crop_size, crop_size):
            crops_actin, crops_axon, crops_dendrite = _get_crop(actin, axon, dendrite, crops_actin, crops_axon,
                                                                crops_dendrite, k, i, j, crop_size)
            k += 1
        # Last columns, not added yet.
        j = cols - crop_size
        crops_actin, crops_axon, crops_dendrite = _get_crop(actin, axon, dendrite, crops_actin, crops_axon,
                                                            crops_dendrite, k, i, j, crop_size)
        k += 1

    # Last rows, not added yet.
    i = rows - crop_size
    for j in range(0, cols - crop_size, crop_size):
        crops_actin, crops_axon, crops_dendrite = _get_crop(actin, axon, dendrite, crops_actin, crops_axon,
                                                            crops_dendrite, k, i, j, crop_size)
        k += 1
    # Last columns of last rows, not added yet.
    j = cols - crop_size
    crops_actin, crops_axon, crops_dendrite = _get_crop(actin, axon, dendrite, crops_actin, crops_axon,
                                                        crops_dendrite, k, i, j, crop_size)

    return crops_actin, crops_axon, crops_dendrite


def get_mirrored_images(actin, axon, dendrite):
    """
    Returns 3 tuples of images, each containing 3 flips of the original image.
    """
    if actin is not None:
        actins = np.flip(actin, axis=1), np.flip(actin, axis=0), np.flip(np.flip(actin, axis=1), axis=0)
    else:
        actins = None
    if axon is not None:
        axons = np.flip(axon, axis=1), np.flip(axon, axis=0), np.flip(np.flip(axon, axis=1), axis=0)
    else:
        axons = None
    if dendrite is not None:
        dendrites = np.flip(dendrite, axis=1), np.flip(dendrite, axis=0), np.flip(np.flip(dendrite, axis=1), axis=0)
    else:
        dendrites = None

    return actins, axons, dendrites
