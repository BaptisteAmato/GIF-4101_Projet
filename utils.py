import os

import skimage.external.tifffile as tifffile

from config import *
from data_augmentation import *
from image_processing import *

folder_images_saving = main_folder_path + '/dataset'
folder_images_saving_train_x = folder_images_saving + '/train_x'
folder_images_saving_train_y = folder_images_saving + '/train_y'

# Image example: 2017-11-14 EXP211 Stim KN93/05_KCl_SMI31-STAR580_MAP2-STAR488_PhSTAR635_1.msr_STED640_Conf561_Conf488_merged.tif


def get_files_path_generator():
    for subdir, dirs, files in os.walk(main_folder_path):
        for file in files:
            filename, file_extension = os.path.splitext(file)
            if file_extension == ".tif":
                yield os.path.join(subdir, file)


def save_train_test_images(n=10):
    generator = get_files_path_generator()
    for i in range(0, n):
        print(i)
        file_path = next(generator)
        image = tifffile.imread(file_path)
        actin, axon, dendrite = split_tif_image(image)
        _, actin_colored, axon_colored, dendrite_colored = get_colored_images(actin, axon, dendrite)
        actin_contour, axon_contour, dendrite_contour = get_contour_map(actin_colored, axon_colored, dendrite_colored)

        if not os.path.exists(folder_images_saving):
            os.makedirs(folder_images_saving)
        if not os.path.exists(folder_images_saving_train_x):
            os.makedirs(folder_images_saving_train_x)
        if not os.path.exists(folder_images_saving_train_y):
            os.makedirs(folder_images_saving_train_y)

        np.save(folder_images_saving_train_x + "/" + str(i), actin_contour)
        # np.save(folder_images_saving_train_y + "/" + str(i), axon_contour)
        np.save(folder_images_saving_train_y + "/" + str(i), dendrite_contour)


# def load_dataset_random_crops(nb_examples=100, nb_crops=4, input_shape=224):
#     train_set_x_orig = np.zeros((nb_examples * nb_crops, input_shape, input_shape, 1))
#     train_set_y_orig = np.zeros((nb_examples * nb_crops, input_shape, input_shape, 1))
#     j = 0
#     for i in range(0, nb_examples):
#         if i % 100 == 0:
#             print(i)
#         x = np.load(folder_images_saving_train_x + "/" + str(i) + ".npy")
#         y = np.load(folder_images_saving_train_y + "/" + str(i) + ".npy")
#         crops_x, crops_y = get_random_crops(x, y)
#         for k in range(0, nb_crops):
#             train_set_x_orig[i + k] = crops_x[k]
#             train_set_y_orig[i + k] = crops_y[k]
#
#     return train_set_x_orig, train_set_y_orig, train_set_x_orig, train_set_y_orig


# For now, y is just the dendrite. Axons not taken into account during tests.
def load_dataset(nb_examples=100, crop_size=224, min_ones_ratio=0.2):
    """
    :param nb_examples:
    :param crop_size:
    :param min_ones_ratio: ratio of "1" in the entire matrix. Allows not to save empty matrices.
    :return:
    """
    min_ones = crop_size * min_ones_ratio
    train_set_x_orig = []
    train_set_y_orig = []
    for i in range(0, nb_examples):
        if i % 100 == 0:
            print(i)
        x = np.load(folder_images_saving_train_x + "/" + str(i) + ".npy")
        y = np.load(folder_images_saving_train_y + "/" + str(i) + ".npy")
        crops_x, _, crops_y = get_all_crops(x, None, y, crop_size)
        length = crops_x.shape[0]
        for j in range(0, length):
            # We do not want to keep black crops, so we make sure there is some data in it.
            if np.sum(crops_x[j]) > min_ones and np.sum(crops_y[j]) > min_ones:
                actins, _, dendrites = get_mirrored_images(crops_x[j], None, crops_y[j])
                for k in range(0, 3):
                    train_set_x_orig.append(actins[k])
                    train_set_y_orig.append(dendrites[k])

    train_set_x_orig = np.array(train_set_x_orig)
    train_set_y_orig = np.array(train_set_y_orig)

    return train_set_x_orig, train_set_y_orig, train_set_x_orig, train_set_y_orig


if __name__ == '__main__':
    # display_images_one_by_one()
    save_train_test_images(1000)
    # get_smallest_image_dimension()
    # print_images_size()
    # load_dataset()

    # image = tifffile.imread(
    #     '/media/maewanto/B498-74ED/Data_projet_apprentissage/2017-11-14 EXP211 Stim KN93/05_KCl_SMI31-STAR580_MAP2-STAR488_PhSTAR635_1.msr_STED640_Conf561_Conf488_merged.tif')
    # actin, axon, dendrite = split_tif_image(image)
    # merged, actin, axon, dendrite = get_colored_images(actin, axon, dendrite)
    # actin_contour, _, _ = get_contour_map(actin, None, None)
    # actin, _, _ = get_image_from_contour_map(actin_contour, None, None)
    # plt.imshow(actin)
    # plt.show()

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
