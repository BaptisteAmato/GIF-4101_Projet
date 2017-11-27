import cv2
import numpy as np
import skimage.color as color


def split_tif_image(image):
    return image[0], image[1], image[2]


# def resize_image(image, width=1024, height=1024):
#     return transform.resize(image, (width, height))


def image_enhancement(image, kernel_erode=2):
    kernel = np.ones((kernel_erode, kernel_erode), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    kernel = np.ones((7, 7), np.float32) / 49
    image = cv2.filter2D(image, -1, kernel)
    return image


def get_colored_images(actin, axon, dendrite, thresh=5):
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

    _, actin = cv2.threshold(actin, thresh, 255, cv2.THRESH_BINARY)
    _, axon = cv2.threshold(axon, thresh, 255, cv2.THRESH_BINARY)
    _, dendrite = cv2.threshold(dendrite, thresh, 255, cv2.THRESH_BINARY)

    actin = color.gray2rgb(actin)
    axon = color.gray2rgb(axon)
    dendrite = color.gray2rgb(dendrite)

    # Green
    index_non_zeros = np.where(actin == [255, 255, 255])
    actin[index_non_zeros[0], index_non_zeros[1], 0] = 0
    actin[index_non_zeros[0], index_non_zeros[1], 2] = 0

    # Red
    index_non_zeros_axon = np.where(axon == [255, 255, 255])
    axon[index_non_zeros_axon[0], index_non_zeros_axon[1], 1] = 0
    axon[index_non_zeros_axon[0], index_non_zeros_axon[1], 2] = 0

    # Blue
    index_non_zeros_dendrite = np.where(dendrite == [255, 255, 255])
    dendrite[index_non_zeros_dendrite[0], index_non_zeros_dendrite[1], 0] = 0
    dendrite[index_non_zeros_dendrite[0], index_non_zeros_dendrite[1], 1] = 0

    _, actin = cv2.threshold(actin, 120, 255, cv2.THRESH_BINARY)
    _, axon = cv2.threshold(axon, thresh, 255, cv2.THRESH_BINARY)
    _, dendrite = cv2.threshold(dendrite, thresh, 255, cv2.THRESH_BINARY)

    merged = np.copy(actin)

    merged[index_non_zeros_axon] = axon[index_non_zeros_axon]
    merged[index_non_zeros_dendrite] = dendrite[index_non_zeros_dendrite]

    merged = image_enhancement(merged, kernel_erode=5)
    actin = image_enhancement(actin, kernel_erode=5)

    return merged, actin, axon, dendrite


def get_contour_map(actin, axon, dendrite):
    if actin is not None:
        colored_indexes = np.where(actin == 255)
        actin = np.zeros((actin.shape[0], actin.shape[1], 1))
        actin[colored_indexes[0], colored_indexes[1]] = 1

    if axon is not None:
        colored_indexes = np.where(axon == 255)
        axon = np.zeros((axon.shape[0], axon.shape[1], 1))
        axon[colored_indexes[0], colored_indexes[1]] = 2

    if dendrite is not None:
        colored_indexes = np.where(dendrite == 255)
        dendrite = np.zeros((dendrite.shape[0], dendrite.shape[1], 1))
        dendrite[colored_indexes[0], colored_indexes[1]] = 3

    return actin, axon, dendrite


def get_image_from_contour_map(actin, axon, dendrite):
    if actin is not None:
        indexes = np.where(actin == 1)
        actin = np.zeros((actin.shape[0], actin.shape[1], 3), dtype=np.uint8)
        actin[indexes[0], indexes[1], 1] = 255

    if axon is not None:
        indexes = np.where(axon == 2)
        axon = np.zeros((axon.shape[0], axon.shape[1], 3), dtype=np.uint8)
        axon[indexes[0], indexes[1], 0] = 255

    if dendrite is not None:
        indexes = np.where(dendrite == 3)
        dendrite = np.zeros((dendrite.shape[0], dendrite.shape[1], 3), dtype=np.uint8)
        dendrite[indexes[0], indexes[1], 2] = 255

    return actin, axon, dendrite


# Apparently, converting to a map and reconverting to an image removes some noise.
def remove_noise(image):
    map = get_contour_map(image)
    return get_image_from_contour_map(map)
