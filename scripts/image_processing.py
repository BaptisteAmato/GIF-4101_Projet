import cv2
import numpy as np
import skimage.color as color
from skimage.external import tifffile
import matplotlib.pyplot as plt


def split_tif_image(image):
    return image[0], image[1], image[2]


def image_enhancement(image, kernel_erode=2):
    kernel = np.ones((kernel_erode, kernel_erode), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    kernel = np.ones((7, 7), np.float32) / 49
    image = cv2.filter2D(image, -1, kernel)
    return image


def get_colored_images(actin, axon, dendrite, thresh=10):
    """
    Returns the three colored images, plus the merged image (three channels superposed)
    """
    actin = actin.astype(np.uint8)
    actin = cv2.equalizeHist(actin)
    actin = image_enhancement(actin, kernel_erode=5)
    axon = axon.astype(np.uint8)
    axon = image_enhancement(axon)
    axon = cv2.equalizeHist(axon)
    dendrite = dendrite.astype(np.uint8)
    dendrite = image_enhancement(dendrite)
    dendrite = cv2.equalizeHist(dendrite)

    _, actin = cv2.threshold(actin, thresh, 255, cv2.THRESH_TOZERO)
    _, axon = cv2.threshold(axon, thresh, 255, cv2.THRESH_TOZERO)
    _, dendrite = cv2.threshold(dendrite, thresh, 255, cv2.THRESH_TOZERO)
    #
    actin = color.gray2rgb(actin)
    axon = color.gray2rgb(axon)
    dendrite = color.gray2rgb(dendrite)

    # Green
    actin[:, :, 0] = 0
    actin[:, :, 2] = 0

    # Red
    axon[:, :, 1] = 0
    axon[:, :, 2] = 0

    # Blue
    dendrite[:, :, 0] = 0
    dendrite[:, :, 1] = 0

    merged = merge_images(actin, axon, dendrite)

    # merged = image_enhancement(merged, kernel_erode=5)
    # actin = image_enhancement(actin, kernel_erode=5)

    return merged, actin, axon, dendrite


def merge_images(actin, axon, dendrite, lim=0.05):
    # merged = np.copy(actin)
    # index_non_zeros_axon = np.where(axon != [0, 0, 0])
    # merged[index_non_zeros_axon] = axon[index_non_zeros_axon]
    # index_non_zeros_dendrite = np.where(dendrite != [0, 0, 0])
    # merged[index_non_zeros_dendrite] = dendrite[index_non_zeros_dendrite]

    # merged = np.zeros(actin.shape)
    # merged[:, :, 0] = axon[:, :, 0]
    # merged[:, :, 2] = dendrite[:, :, 2]
    # zero_values = merged[:, :, 0] == 0
    # zero_values = np.logical_and(zero_values, merged[:, :, 2] == 0)
    # merged[zero_values] = actin[zero_values]

    merged = np.zeros(actin.shape)
    merged[:, :, 0] = axon[:, :, 0]
    merged[:, :, 2] = dendrite[:, :, 2]
    zero_values = merged[:, :, 0] < lim
    zero_values = np.logical_and(zero_values, merged[:, :, 2] < lim)
    merged[zero_values] = actin[zero_values]
    return merged


def get_contour_map(actin, axon, dendrite, binary_masks, thresh=0.1):
    """
    Retrieve only the colored channel from each image, scale value between 0 and 1, and apply binary threshold if
    binary_masks=True
    :param actin:
    :param axon:
    :param dendrite:
    :param thresh:
    :param binary_masks:
    :return:
    """
    if actin is not None:
        actin = actin[:, :, 1] / 255

    if axon is not None:
        # axon = axon.astype(np.uint8)
        axon = axon[:, :, 0] / 255
        if binary_masks:
            # Make axons mask binary.
            _, axon = cv2.threshold(axon, thresh, 1, cv2.THRESH_BINARY)

    if dendrite is not None:
        dendrite = dendrite[:, :, 2] / 255
        if binary_masks:
            # Make dendrites mask binary.
            _, dendrite = cv2.threshold(dendrite, thresh, 1, cv2.THRESH_BINARY)

    return actin, axon, dendrite
