import os
import numpy as np
import cv2
import skimage.external.tifffile as tifffile
import matplotlib.pyplot as plt
import skimage.color as color
import math
from config import *


folder_images_saving = main_folder_path + '/dataset'
folder_images_saving_train_x = folder_images_saving + '/train_x'
folder_images_saving_train_y = folder_images_saving + '/train_y'
# Image example: 2017-11-14 EXP211 Stim KN93/05_KCl_SMI31-STAR580_MAP2-STAR488_PhSTAR635_1.msr_STED640_Conf561_Conf488_merged.tif


def split_tif_image(image):
        return image[0], image[1], image[2]


# def resize_image(image, width=1024, height=1024):
#     return transform.resize(image, (width, height))


# (314, 281)
def get_smallest_image_dimension():
    N = get_number_original_files()
    min_rows = np.inf
    min_cols = np.inf
    i = 0
    for file_path in get_files_path_generator():
        print(str(i+1) + "/" + str(N))
        image = tifffile.imread(file_path)
        # Remove number of channels from shape
        shape = image.shape[1:]
        if min_rows > shape[0]:
            min_rows = shape[0]
        if min_cols > shape[1]:
            min_cols = shape[1]
        i += 1
    return min_rows, min_cols


def get_random_crops(x, y, nb_crops=4, crop_size=224):
    crops_x = np.zeros((nb_crops, crop_size, crop_size, 1))
    crops_y = np.zeros((nb_crops, crop_size, crop_size, 1))
    for i in range(0, nb_crops):
        row = np.random.randint(0, x.shape[0] - crop_size)
        col = np.random.randint(0, x.shape[1] - crop_size)
        crops_x[i] = x[row:row+crop_size, col:col+crop_size]
        crops_y[i] = y[row:row+crop_size, col:col+crop_size]
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


def get_contour_map(colored_image):
    res = np.zeros((colored_image.shape[0], colored_image.shape[1], 1))
    colored_indexes = np.where(colored_image == 255)
    res[colored_indexes[0], colored_indexes[1]] = 1
    return res


def get_image_from_contour_map(map, color='g'):
    if color == 'r':
        channel = 0
    elif color == 'g':
        channel = 1
    else:
        channel = 2
    ones = np.where(map == 1)
    res = np.zeros((map.shape[0], map.shape[1], 3), dtype=np.uint8)
    res[ones[0], ones[1], channel] = 255
    return res


# Apparently, converting to a map and reconverting to an image removes some noise.
def remove_noise(image):
    map = get_contour_map(image)
    return get_image_from_contour_map(map)


def get_files_path_generator():
    for subdir, dirs, files in os.walk(main_folder_path):
        for file in files:
            filename, file_extension = os.path.splitext(file)
            if file_extension == ".tif":
                yield os.path.join(subdir, file)


# 1041
def get_number_original_files():
    i = 0
    for subdir, dirs, files in os.walk(main_folder_path):
        for file in files:
            filename, file_extension = os.path.splitext(file)
            if file_extension == ".tif":
                i += 1
    return i


def save_train_test_images(n=10, square_size=224):
    generator = get_files_path_generator()
    for i in range(0, n):
        print(i)
        file_path = next(generator)
        image = tifffile.imread(file_path)
        actin, axon, dendrite = split_tif_image(image)
        _, actin_colored, axon_colored, dendrite_colored = get_colored_images(actin, axon, dendrite)

        if not os.path.exists(folder_images_saving):
            os.makedirs(folder_images_saving)
        if not os.path.exists(folder_images_saving_train_x):
            os.makedirs(folder_images_saving_train_x)
        if not os.path.exists(folder_images_saving_train_y):
            os.makedirs(folder_images_saving_train_y)

        # np.save(folder_images_saving_train_x + "/" + str(i), get_contour_map(actin_colored[:square_size, :square_size]))
        np.save(folder_images_saving_train_x + "/" + str(i), get_contour_map(actin_colored))
        # np.save(folder_images_saving_train_y + "/" + str(i), get_contour_map(dendrite_colored[:square_size, :square_size]))
        np.save(folder_images_saving_train_y + "/" + str(i), get_contour_map(dendrite_colored))


def display_tif_image(file_path, with_colored_images=True, with_merged_image=True):
    image = tifffile.imread(file_path)
    actin, axon, dendrite = split_tif_image(image)
    print("Image imported: " + file_path)
    # If with_colored_images == False, subplots on one line only.
    plt.subplot(1 + with_colored_images, 3, 1)
    plt.title("Actin")
    plt.imshow(actin)
    plt.subplot(1 + with_colored_images, 3, 2)
    plt.title("Axon")
    plt.imshow(axon)
    plt.subplot(1 + with_colored_images, 3, 3)
    plt.title("Dendrite")
    plt.imshow(dendrite)

    merged, actin, axon, dendrite = get_colored_images(actin, axon, dendrite)
    if with_colored_images:
        plt.subplot(2, 3, 4)
        plt.title("Actin colored")
        plt.imshow(actin)
        plt.subplot(2, 3, 5)
        plt.title("Axon colored")
        plt.imshow(axon)
        plt.subplot(2, 3, 6)
        plt.title("Dendrite colored")
        plt.imshow(dendrite)
    # Show full screen.
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    if with_merged_image:
        plt.figure()
        plt.imshow(merged)
    plt.show()


def display_images_one_by_one():
    for file_path in get_files_path_generator():
        display_tif_image(file_path, True, False)


def print_images_size():
    for file_path in get_files_path_generator():
        image = tifffile.imread(file_path)
        print(image.shape)
        input()


def load_dataset_random_crops(nb_examples=100, nb_crops=4, input_shape=224):
    train_set_x_orig = np.zeros((nb_examples * nb_crops, input_shape, input_shape, 1))
    train_set_y_orig = np.zeros((nb_examples * nb_crops, input_shape, input_shape, 1))
    j = 0
    for i in range(0, nb_examples):
        if i % 100 == 0:
            print(i)
        x = np.load(folder_images_saving_train_x + "/" + str(i) + ".npy")
        y = np.load(folder_images_saving_train_y + "/" + str(i) + ".npy")
        crops_x, crops_y = get_random_crops(x, y)
        for k in range(0, nb_crops):
            train_set_x_orig[i + k] = crops_x[k]
            train_set_y_orig[i + k] = crops_y[k]

    return train_set_x_orig, train_set_y_orig, train_set_x_orig, train_set_y_orig


def load_dataset(nb_examples=100, crop_size=224):
    train_set_x_orig = []
    train_set_y_orig = []
    for i in range(0, nb_examples):
        if i % 100 == 0:
            print(i)
        x = np.load(folder_images_saving_train_x + "/" + str(i) + ".npy")
        y = np.load(folder_images_saving_train_y + "/" + str(i) + ".npy")
        crops_x, crops_y = get_all_crops(x, y, crop_size)
        length = crops_x.shape[0]
        for k in range(0, length):
            train_set_x_orig.append(crops_x[k])
            train_set_y_orig.append(crops_y[k])

    train_set_x_orig = np.array(train_set_x_orig)
    train_set_y_orig = np.array(train_set_y_orig)

    return train_set_x_orig, train_set_y_orig, train_set_x_orig, train_set_y_orig

if __name__ == '__main__':
    # display_images_one_by_one()
    save_train_test_images(9)

    # get_smallest_image_dimension()
    # print_images_size()

    # image = tifffile.imread(
    #     '/media/maewanto/B498-74ED/Data_projet_apprentissage/2017-11-14 EXP211 Stim KN93/05_KCl_SMI31-STAR580_MAP2-STAR488_PhSTAR635_1.msr_STED640_Conf561_Conf488_merged.tif')
    # actin, axon, dendrite = split_tif_image(image)
    # merged, actin, axon, dendrite = get_colored_images(actin, axon, dendrite)

    # square_size = 224
    # plt.imshow(actin)
    # plt.show()
    # plt.imshow(actin[:square_size, :square_size])
    # plt.show()
    # actin = actin[:square_size, :square_size]
    # print(np.where(actin == 255))
    # exit()
    # actin = actin[:square_size, :square_size]
    # exit()

    # image = tifffile.imread(
    #     '/media/maewanto/B498-74ED/Data_projet_apprentissage/2017-11-14 EXP211 Stim KN93/05_KCl_SMI31-STAR580_MAP2-STAR488_PhSTAR635_1.msr_STED640_Conf561_Conf488_merged.tif')
    # actin, axon, dendrite = split_tif_image(image)
    # merged, actin_colored, axon_colored, dendrite_colored = get_colored_images(actin, axon, dendrite)

