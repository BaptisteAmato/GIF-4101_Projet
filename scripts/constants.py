import os

from config import *


def get_folder_images_saving(binary_masks=True):
    """
    Returns the folder in which the processed images should be saved (as .npy)
    :param binary_masks:
    :return:
    """
    if binary_masks:
        return main_folder_path + '/dataset_binary'
    else:
        return main_folder_path + '/dataset_non_binary'


def get_folder_images_saving_train_x(binary_masks=True):
    """
    Returns the folder in which the processed axon images should be saved (as .npy)
    :param binary_masks:
    :return:
    """
    return get_folder_images_saving(binary_masks) + '/train_x'


def get_folder_images_saving_train_y(binary_masks=True):
    """
    Returns the folder in which the processed actin and dendrites images should be saved (as .npy)
    :param binary_masks:
    :return:
    """
    return get_folder_images_saving(binary_masks) + '/train_y'


def get_model_weights_path(model_name, channel, binary_masks):
    if binary_masks:
        subfolder = 'binary'
    else:
        subfolder = 'non_binary'

    # Create folders if not exist.
    if not os.path.exists(folder_models_weights):
        os.makedirs(folder_models_weights)
    if not os.path.exists(folder_models_weights + '/' + channel):
        os.makedirs(folder_models_weights + '/' + channel)
    if not os.path.exists(folder_models_weights + '/' + channel + '/' + subfolder):
        os.makedirs(folder_models_weights + '/' + channel + '/' + subfolder)

    return folder_models_weights + '/' + channel + '/' + subfolder + "/" + model_name + model_weights_suffix


def get_model_path(model_name):
    # Create folder if not exist.
    if not os.path.exists(folder_models):
        os.makedirs(folder_models)

    return folder_models + "/" + model_name + ".json"


def get_dataset_h5py_path(binary_masks):
    if binary_masks:
        path = main_folder_path + "/dataset_binary.hdf5"
    else:
        path = main_folder_path + "/dataset_non_binary.hdf5"

    return path


folder_models = main_folder_path + '/models'
folder_models_weights = main_folder_path + '/models_weights'
model_weights_suffix = '_weights.hdf5'
crop_size = 224

# Image example: 2017-11-14 EXP211 Stim KN93/05_KCl_SMI31-STAR580_MAP2-STAR488_PhSTAR635_1.msr_STED640_Conf561_Conf488_merged.tif

