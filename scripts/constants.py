import os

from config import *


def get_folder_images_saving():
    """
    Returns the folder in which the processed images should be saved (as .npy)
    :return:
    """
    return main_folder_path + '/dataset'


def get_folder_images_saving_train_x():
    """
    Returns the folder in which the processed axon images should be saved (as .npy)
    :return:
    """
    return get_folder_images_saving() + '/train_x'


def get_folder_images_saving_train_y():
    """
    Returns the folder in which the processed actin and dendrites images should be saved (as .npy)
    :return:
    """
    return get_folder_images_saving() + '/train_y'


def get_model_weights_path(model_name, channel):
    # Create folders if not exist.
    if not os.path.exists(folder_models_weights):
        os.makedirs(folder_models_weights)
    if not os.path.exists(folder_models_weights + '/' + channel):
        os.makedirs(folder_models_weights + '/' + channel)

    return folder_models_weights + '/' + channel + "/" + model_name + model_weights_suffix


def get_model_path(model_name):
    # Create folder if not exist.
    if not os.path.exists(folder_models):
        os.makedirs(folder_models)

    return folder_models + "/" + model_name + ".json"


def get_model_evaluation_path(model_name):
    # Create folder if not exist.
    if not os.path.exists(folder_models):
        os.makedirs(folder_models)

    return folder_models + "/" + model_name + ".txt"


def get_dataset_h5py_path(channel):
    return main_folder_path + "/dataset_" + channel + ".hdf5"


def get_test_data_folder_after_training(model_name, channel):
    # Create folders if not exist.
    if not os.path.exists(folder_models_weights):
        os.makedirs(folder_models_weights)
    if not os.path.exists(folder_models_weights + '/' + channel):
        os.makedirs(folder_models_weights + '/' + channel)

    path = folder_models_weights + '/' + channel + "/" + model_name
    if not os.path.exists(path):
        os.makedirs(path)

    return path


original_data = main_folder_path + '/original_data'
folder_models = main_folder_path + '/models'
folder_models_weights = main_folder_path + '/models_weights'
model_weights_suffix = '_weights.hdf5'
crop_size = 224
