import os
import h5py

from keras.models import model_from_json
from sklearn.model_selection import train_test_split

from data_augmentation import *
from image_processing import *


def get_files_path_generator():
    for subdir, dirs, files in os.walk(main_folder_path):
        for file in files:
            filename, file_extension = os.path.splitext(file)
            if file_extension == ".tif":
                yield os.path.join(subdir, file)


def get_train_label_images(tif_image):
    actin_original, axon_original, dendrite_original = split_tif_image(tif_image)
    _, actin, axon, dendrite = get_colored_images(actin_original, axon_original, dendrite_original)
    actin, axon, dendrite = get_contour_map(actin, axon, dendrite)
    train = np.expand_dims(actin, axis=2)
    test = np.zeros((train.shape[0], train.shape[1], 2))
    test[:, :, 0] = axon
    test[:, :, 1] = dendrite
    return train, test


def get_images_from_train_label(train, label):
    if train is not None:
        rows = train.shape[0]
        cols = train.shape[1]
        actin = np.zeros((rows, cols, 3))
        actin[:, :, 1] = np.squeeze(train)
    else:
        actin = np.array([])

    if label is not None:
        rows = label.shape[0]
        cols = label.shape[1]
        axon_or_dendrite = np.zeros((rows, cols, 3))
        axon_or_dendrite[:, :, 0] = np.squeeze(label[:, :, 0])
    else:
        axon_or_dendrite = np.array([])

    return actin, axon_or_dendrite


def save_train_label_images(n=10):
    # Create folders if not exist.
    if not os.path.exists(folder_images_saving):
        os.makedirs(folder_images_saving)
    if not os.path.exists(folder_images_saving_train_x):
        os.makedirs(folder_images_saving_train_x)
    if not os.path.exists(folder_images_saving_train_y):
        os.makedirs(folder_images_saving_train_y)

    generator = get_files_path_generator()
    for i in range(0, n):
        print(i)
        file_path = next(generator)
        tif_image = tifffile.imread(file_path)
        train, label = get_train_label_images(tif_image)
        np.save(folder_images_saving_train_x + "/" + str(i), train)
        np.save(folder_images_saving_train_y + "/" + str(i), label)


def get_model_weights_path(model_name):
    # Create folder if not exist.
    if not os.path.exists(folder_models_weights):
        os.makedirs(folder_models_weights)
    return folder_models_weights + "/" + model_name + model_weights_suffix


def get_model_path(model_name):
    # Create folder if not exist.
    if not os.path.exists(folder_models):
        os.makedirs(folder_models)
    return folder_models + "/" + model_name + ".json"


def get_dataset_h5py_path():
    return main_folder_path + "/dataset.hdf5"


def save_dataset(nb_images=100, min_ones_ratio=0.2, max_ones_ratio=0.8):
    """
    :param nb_images:
    :param min_ones_ratio: ratio of "1" in the entire matrix. Helps not saving empty matrices.
    :return: X_train, X_test, y_train, y_test
    """
    print("AUGMENTING THE DATA")
    min_ones = crop_size * min_ones_ratio
    max_ones = crop_size * max_ones_ratio
    train_set_x_orig = []
    train_set_y_axon_orig = []
    train_set_y_dendrite_orig = []
    for i in range(0, nb_images):
        if i % 10 == 0:
            print(i)
        x = np.load(folder_images_saving_train_x + "/" + str(i) + ".npy")
        y = np.load(folder_images_saving_train_y + "/" + str(i) + ".npy")
        crops_x, crops_y = get_all_crops(x, y)
        length = crops_x.shape[0]
        for j in range(0, length):
            # We do not want to keep too many black crops, so we make sure there is some data in both train and label
            # matrices before taking the flips.
            if np.sum(crops_x[j]) > min_ones and np.sum(crops_y[j, :, :, 0]) > min_ones and np.sum(
                    crops_y[j, :, :, 1]) > min_ones and np.sum(crops_x[j]) < max_ones and np.sum(
                    crops_y[j, :, :, 0]) < max_ones and np.sum(crops_y[j, :, :, 1]) < max_ones:
                flips_x, flips_y = get_flips_images(crops_x[j], crops_y[j])
                for k in range(0, 3):
                    train_set_x_orig.append(flips_x[k])
                    train_set_y_axon_orig.append(flips_y[k, :, :, 0])
                    train_set_y_dendrite_orig.append(flips_y[k, :, :, 1])
            else:
                train_set_x_orig.append(crops_x[j])
                train_set_y_axon_orig.append(crops_y[j, :, :, 0])
                train_set_y_dendrite_orig.append(crops_y[j, :, :, 1])

    # Save the created datasets to an hdf5 file.
    print("SAVING THE HDF5 FILE")
    with h5py.File(get_dataset_h5py_path(), 'w') as f:
        length = len(train_set_x_orig)
        print("Length: " + str(length))
        dataset = f.create_dataset("X", (length, crop_size, crop_size, 1))
        dataset[...] = np.array(train_set_x_orig)
        dataset = f.create_dataset("y_axon", (length, crop_size, crop_size, 1))
        dataset[...] = np.expand_dims(np.array(train_set_y_axon_orig), axis=3)
        dataset = f.create_dataset("y_dendrite", (length, crop_size, crop_size, 1))
        dataset[...] = np.expand_dims(np.array(train_set_y_dendrite_orig), axis=3)
    print("DONE")


def load_dataset(return_all=True, nb_examples=100, train_ratio=0.7, channel='axons'):
    """
    :param return_all:
    :param nb_examples:
    :param train_ratio:
    :param channel:
    :return: X_train, X_test, y_train, y_test
    """
    with h5py.File(get_dataset_h5py_path(), 'r') as f:
        X = f['X'].value
        if channel == 'axons':
            y = f['y_axon'].value
        else:
            y = f['y_dendrite'].value
        if return_all:
            return train_test_split(X, y, train_size=train_ratio)
        else:
            return train_test_split(X[:nb_examples], y[:nb_examples], train_size=train_ratio)


def load_model(model_name):
    # Load the model.
    path = get_model_path(model_name)
    print("Loading " + path)
    with open(path) as f:
        my_model = model_from_json(f.read())
    # Load the weights.
    path = get_model_weights_path(model_name)
    print("Loading " + path)
    my_model.load_weights(path)
    # Compile the model
    my_model.compile(optimizer="adam", loss='mean_squared_error', metrics=["accuracy"])
    return my_model


if __name__ == '__main__':
    # display_images_one_by_one()
    save_train_label_images(1040)
    # get_smallest_image_dimension()
    # print_images_size()
    # load_dataset(1)
    # i = 0
    # test_image(i)

