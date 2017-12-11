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


def get_train_label_images(tif_image, binary_masks):
    """
    Process the original image to be then saved as .npy file
    :param tif_image:
    :param binary_masks:
    :return:
    """
    actin_original, axon_original, dendrite_original = split_tif_image(tif_image)
    _, actin, axon, dendrite = get_colored_images(actin_original, axon_original, dendrite_original)
    actin, axon, dendrite = get_contour_map(actin, axon, dendrite, binary_masks)
    train = np.expand_dims(actin, axis=2)
    test = np.zeros((train.shape[0], train.shape[1], 2))
    test[:, :, 0] = axon
    test[:, :, 1] = dendrite
    return train, test


def save_train_label_images(number_of_images=10, binary_masks=True):
    """
    Saves the images after processing, as .npy files
    :param number_of_images:
    :param binary_masks:
    :return:
    """
    # Create folders if not exist.
    if not os.path.exists(get_folder_images_saving(binary_masks)):
        os.makedirs(get_folder_images_saving(binary_masks))
    if not os.path.exists(get_folder_images_saving_train_x(binary_masks)):
        os.makedirs(get_folder_images_saving_train_x(binary_masks))
    if not os.path.exists(get_folder_images_saving_train_y(binary_masks)):
        os.makedirs(get_folder_images_saving_train_y(binary_masks))

    generator = get_files_path_generator()
    for i in range(0, number_of_images):
        print(i)
        file_path = next(generator)
        tif_image = tifffile.imread(file_path)
        train, label = get_train_label_images(tif_image, binary_masks)
        np.save(get_folder_images_saving_train_x(binary_masks) + "/" + str(i), train)
        np.save(get_folder_images_saving_train_y(binary_masks) + "/" + str(i), label)


def save_dataset(nb_images, binary_masks, min_ones_ratio=0.2, max_ones_ratio=0.8):
    """
    Saves the images after data augmentation, in an .hdf5 file
    :param nb_images:
    :param binary_masks:
    :param min_ones_ratio: ratio of "1" in the entire matrix. Helps not saving empty matrices if binary_masks=False
    :param max_ones_ratio:
    :return: X_train, X_test, y_train, y_test
    """
    print("AUGMENTING THE DATA")
    min_ones = crop_size * crop_size * min_ones_ratio
    max_ones = crop_size * crop_size * max_ones_ratio
    train_set_x_axon_orig = []
    train_set_y_axon_orig = []
    train_set_x_dendrite_orig = []
    train_set_y_dendrite_orig = []
    for i in range(0, nb_images):
        if i % 10 == 0:
            print(i)
        x = np.load(get_folder_images_saving_train_x(binary_masks) + "/" + str(i) + ".npy")
        y = np.load(get_folder_images_saving_train_y(binary_masks) + "/" + str(i) + ".npy")
        crops_x, crops_y = get_all_crops(x, y)
        length = crops_x.shape[0]
        for j in range(0, length):
            # The discriminant crops will be the ones where there is some difference between the actin and the
            # axons/dendrites. Only these ones will be flipped.
            crop_x_j = crops_x[j]
            crop_y_j = crops_y[j]

            positive_axon = np.where(crop_y_j[:, :, 0] > 0)
            actin_minus_axon = crop_x_j.copy()
            actin_minus_axon[positive_axon] = 0
            if np.sum(actin_minus_axon > 0) > min_ones:
                flips_x, flips_y = get_flips_images(crops_x[j], crops_y[j])
                for k in range(0, 4):
                    train_set_x_axon_orig.append(flips_x[k])
                    train_set_y_axon_orig.append(flips_y[k, :, :, 0])
            else:
                train_set_x_axon_orig.append(crops_x[j])
                train_set_y_axon_orig.append(crops_y[j, :, :, 0])

            positive_dendrite = np.where(crop_y_j[:, :, 1] > 0)
            actin_minus_dendrite = crop_x_j.copy()
            actin_minus_dendrite[positive_dendrite] = 0
            if np.sum(actin_minus_dendrite > 0) > min_ones:
                flips_x, flips_y = get_flips_images(crops_x[j], crops_y[j])
                for k in range(0, 4):
                    train_set_x_dendrite_orig.append(flips_x[k])
                    train_set_y_dendrite_orig.append(flips_y[k, :, :, 1])
            else:
                train_set_x_dendrite_orig.append(crops_x[j])
                train_set_y_dendrite_orig.append(crops_y[j, :, :, 1])

    # Save the created data sets to an hdf5 file.
    print("SAVING THE HDF5 FILE")
    with h5py.File(get_dataset_h5py_path(binary_masks), 'w') as f:
        length_axons = len(train_set_x_axon_orig)
        print("Length axons: " + str(length_axons))
        dataset = f.create_dataset("X_axon", (length_axons, crop_size, crop_size, 1))
        dataset[...] = np.array(train_set_x_axon_orig)
        dataset = f.create_dataset("y_axon", (length_axons, crop_size, crop_size, 1))
        dataset[...] = np.expand_dims(np.array(train_set_y_axon_orig), axis=3)

        length_dendrites = len(train_set_x_dendrite_orig)
        print("Length dendrites: " + str(length_dendrites))
        dataset = f.create_dataset("X_dendrite", (length_dendrites, crop_size, crop_size, 1))
        dataset[...] = np.array(train_set_x_dendrite_orig)
        dataset = f.create_dataset("y_dendrite", (length_dendrites, crop_size, crop_size, 1))
        dataset[...] = np.expand_dims(np.array(train_set_y_dendrite_orig), axis=3)
    print("DONE")


def load_dataset(nb_examples, binary_masks, channel, train_test_splitting=True, train_ratio=0.7):
    """
    Returns the train and test datasets
    :param nb_examples:
    :param binary_masks:
    :param channel:
    :param train_test_splitting:
    :param train_ratio:
    :return: X_train, X_test, y_train, y_test
    """
    with h5py.File(get_dataset_h5py_path(binary_masks), 'r') as f:
        if channel == 'axons':
            X = f['X_axon'].value
            y = f['y_axon'].value
        elif channel == 'dendrites':
            X = f['X_dendrite'].value
            y = f['y_dendrite'].value
        else:
            raise ValueError('channel attribute must either be "axons" or "dendrites"')
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
