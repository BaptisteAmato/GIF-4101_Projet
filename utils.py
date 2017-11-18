import os
import numpy as np
import cv2
import skimage.external.tifffile as tifffile
import matplotlib.pyplot as plt


main_folder_path = '/media/maewanto/B498-74ED/Data_projet_apprentissage'
# Image example: 2017-11-14 EXP211 Stim KN93/05_KCl_SMI31-STAR580_MAP2-STAR488_PhSTAR635_1.msr_STED640_Conf561_Conf488_merged.tif


def split_tif_image(image):
    return image[0], image[1], image[2]


def get_files_path(main_folder_path):
    for subdir, dirs, files in os.walk(main_folder_path):
        for file in files:
            filename, file_extension = os.path.splitext(file)
            if file_extension == ".tif":
                yield os.path.join(subdir, file)


def display_tif_image(file_path):
    image = tifffile.imread(file_path)
    actin, axon, dendrite = split_tif_image(image)
    print("Image imported: " + file_path)
    plt.subplot(1, 3, 1)
    plt.title("Actin")
    plt.imshow(actin)
    plt.subplot(1, 3, 2)
    plt.title("Axon")
    plt.imshow(axon)
    plt.subplot(1, 3, 3)
    plt.title("Dendrite")
    plt.imshow(dendrite)
    # Show full screen.
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()


def display_images_one_by_one():
    for file_path in get_files_path(main_folder_path):
        display_tif_image(file_path)


display_images_one_by_one()

# image = tifffile.imread('/media/maewanto/B498-74ED/Data_projet_apprentissage/2017-11-14 EXP211 Stim KN93/05_KCl_SMI31-STAR580_MAP2-STAR488_PhSTAR635_1.msr_STED640_Conf561_Conf488_merged.tif')
# actin, axon, dendrite = split_tif_image(image)
# plt.imshow(axon, cmap=plt.cm.gray)
# plt.show()






