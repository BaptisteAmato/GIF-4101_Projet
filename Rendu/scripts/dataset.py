import h5py
from sklearn.model_selection import train_test_split

from utils import *


def get_files_path_generator():
    """
    Generator of original tif files' path
    :return:
    """
    for subdir, dirs, files in os.walk(original_data):
        for file in files:
            filename, file_extension = os.path.splitext(file)
            if file_extension == ".tif":
                yield os.path.join(subdir, file)


def get_train_label_images(tif_image):
    """
    Process the original image to be then saved as .npy file
    :param tif_image:
    :return:
    """
    actin_original, axon_original, dendrite_original = split_tif_image(tif_image)
    _, actin, axon, dendrite = get_colored_images(actin_original, axon_original, dendrite_original)
    actin, axon, dendrite = get_contour_map(actin, axon, dendrite)
    train = np.expand_dims(actin, axis=2)
    test = np.zeros((train.shape[0], train.shape[1], 2))
    test[:, :, 0] = axon
    test[:, :, 1] = dendrite
    return train, test


def save_train_label_images(number_of_images=10):
    """
    Saves the images after processing, as .npy files
    :param number_of_images:
    :return:
    """
    # Create folders if not exist.
    if not os.path.exists(get_folder_images_saving()):
        os.makedirs(get_folder_images_saving())
    if not os.path.exists(get_folder_images_saving_train_x()):
        os.makedirs(get_folder_images_saving_train_x())
    if not os.path.exists(get_folder_images_saving_train_y()):
        os.makedirs(get_folder_images_saving_train_y())

    generator = get_files_path_generator()
    for i in range(0, number_of_images):
        print(i)
        file_path = next(generator)
        tif_image = tifffile.imread(file_path)
        train, label = get_train_label_images(tif_image)
        np.save(get_folder_images_saving_train_x() + "/" + str(i), train)
        np.save(get_folder_images_saving_train_y() + "/" + str(i), label)


def save_dataset(nb_images, channel, min_ones_ratio=0.2):
    """
    Saves the images after data augmentation, in an .hdf5 file
    :param nb_images:
    :param min_ones_ratio: ratio of non-zero values in the entire matrix.
    :return: X_train, X_test, y_train, y_test
    """
    if channel == 'axons':
        index_y = 0
    elif channel == 'dendrites':
        index_y = 1
    else:
        raise ValueError('channel attribute must either be "axons" or "dendrites"')

    print("AUGMENTING THE DATA")
    min_ones = crop_size * crop_size * min_ones_ratio
    train_set_x_orig = []
    train_set_y_orig = []
    for i in range(0, nb_images):
        if i % 10 == 0:
            print(i)
        x = np.load(get_folder_images_saving_train_x() + "/" + str(i) + ".npy")
        y = np.load(get_folder_images_saving_train_y() + "/" + str(i) + ".npy")
        crops_x, crops_y = get_all_crops(x, y)
        length = crops_x.shape[0]
        for j in range(0, length):
            # The discriminant crops will be the ones where there is some difference between the actin and the
            # axons/dendrites. Only these ones will be flipped.
            crop_x_j = crops_x[j]
            crop_y_j = np.expand_dims(crops_y[j][:, :, index_y], 3)

            positive_channel = np.where(crop_y_j > 0)
            actin_minus_channel = crop_x_j.copy()
            actin_minus_channel[positive_channel] = 0
            if np.sum(actin_minus_channel > 0) > min_ones:
                flips_x, flips_y = get_flips_images(crop_x_j, crop_y_j)
                for k in range(0, 4):
                    train_set_x_orig.append(flips_x[k])
                    train_set_y_orig.append(flips_y[k])
            else:
                train_set_x_orig.append(crop_x_j)
                train_set_y_orig.append(crop_y_j)

    # Save the created data sets to an hdf5 file.
    print("SAVING THE HDF5 FILE")
    with h5py.File(get_dataset_h5py_path(channel), 'w') as f:
        length = len(train_set_x_orig)
        print("Length: " + str(length))
        dataset = f.create_dataset("X", (length, crop_size, crop_size, 1))
        dataset[...] = np.array(train_set_x_orig)
        dataset = f.create_dataset("y", (length, crop_size, crop_size, 1))
        dataset[...] = np.array(train_set_y_orig)
    print("DONE")


def load_dataset(nb_examples, channel, train_test_splitting=True, train_ratio=0.7):
    """
    Returns the train and test datasets
    :param nb_examples:
    :param channel:
    :param train_test_splitting:
    :param train_ratio:
    :return: X_train, X_test, y_train, y_test
    """
    with h5py.File(get_dataset_h5py_path(channel), 'r') as f:
        X = f['X'].value
        y = f['y'].value
        if train_test_splitting:
            if nb_examples is None:
                return train_test_split(X, y, train_size=train_ratio)
            else:
                return train_test_split(X[:nb_examples], y[:nb_examples], train_size=train_ratio)
        else:
            empty_array = np.array([])
            if nb_examples is None:
                return X, empty_array, y, empty_array
            else:
                return X[:nb_examples], empty_array, y[:nb_examples], empty_array
