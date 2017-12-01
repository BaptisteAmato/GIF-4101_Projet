import os

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
    merged = None
    if train is not None:
        rows = train.shape[0]
        cols = train.shape[1]
        actin = np.zeros((rows, cols, 3))
        actin[:, :, 1] = np.squeeze(train) * 255
    else:
        actin = np.array([])

    if label is not None:
        axon, dendrite = get_axon_dendrite_from_label(label)
    else:
        axon = np.array([])
        dendrite = np.array([])

    if train is not None and label is not None:
        merged = merge_images(actin, axon, dendrite)

    return merged, actin, axon, dendrite


def get_axon_dendrite_from_label(label):
    rows = label.shape[0]
    cols = label.shape[1]
    axon = np.zeros((rows, cols, 3))
    axon[:, :, 0] = np.squeeze(label[:, :, 0]) * 255
    dendrite = np.zeros((rows, cols, 3))
    dendrite[:, :, 2] = np.squeeze(label[:, :, 1]) * 255
    return axon, dendrite


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


# def load_dataset(nb_examples=100, train_ratio=0.7, min_ones_ratio=0.2):
#     """
#     :param nb_examples:
#     :param train_ratio:
#     :param min_ones_ratio: ratio of "1" in the entire matrix. Allows not to save empty matrices.
#     :return: X_train, X_test, y_train, y_test
#     """
#     min_ones = crop_size * min_ones_ratio
#     train_set_x_orig = []
#     train_set_y_orig = []
#     for i in range(0, nb_examples):
#         if i % 10 == 0:
#             print(i)
#         x = np.load(folder_images_saving_train_x + "/" + str(i) + ".npy")
#         y = np.load(folder_images_saving_train_y + "/" + str(i) + ".npy")
#         crops_x, crops_y = get_all_crops(x, y)
#         length = crops_x.shape[0]
#         for j in range(0, length):
#             # We do not want to keep black crops, so we make sure there is some data in both train and label matrices.
#             if np.sum(crops_x[j]) > min_ones and np.sum(crops_y[j, :, :, 0]) > min_ones and np.sum(crops_y[j, :, :, 1]) > min_ones:
#                 flips_x, flips_y = get_flips_images(crops_x[j], crops_y[j])
#                 for k in range(0, 3):
#                     train_set_x_orig.append(flips_x[k])
#                     train_set_y_orig.append(flips_y[k])
#     print("Crops: " + str(len(train_set_x_orig)))
#     print("Splitting train/test")
#
#     return train_test_split(np.array(train_set_x_orig), np.array(train_set_y_orig), train_size=train_ratio)


def load_example(i, X_mat, y_mat, min_ones):
    x = np.load(folder_images_saving_train_x + "/" + str(i) + ".npy")
    y = np.load(folder_images_saving_train_y + "/" + str(i) + ".npy")
    crops_x, crops_y = get_all_crops(x, y)
    length = crops_x.shape[0]
    for j in range(0, length):
        # We do not want to keep black crops, so we make sure there is some data in both train and label matrices.
        if np.sum(crops_x[j]) > min_ones and np.sum(crops_y[j, :, :, 0]) > min_ones and np.sum(
                crops_y[j, :, :, 1]) > min_ones:
            flips_x, flips_y = get_flips_images(crops_x[j], crops_y[j])
            for k in range(0, 3):
                X_mat.append(flips_x[k])
                y_mat.append(flips_y[k])
    return X_mat, y_mat


def load_dataset(nb_examples=100, train_ratio=0.7, min_ones_ratio=0.2):
    """
    :param nb_examples:
    :param train_ratio:
    :param min_ones_ratio: ratio of "1" in the entire matrix. Allows not to save empty matrices.
    :return: X_train, X_test, y_train, y_test
    """
    min_ones = crop_size * min_ones_ratio
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    indexes = np.arange(0, nb_examples, 1)
    np.random.shuffle(indexes)
    last_train_index = int(len(indexes) * train_ratio)
    k = 0
    for i in indexes[:last_train_index]:
        if k % 10 == 0:
            print(k)
        X_train, y_train = load_example(i, X_train, y_train, min_ones)
        k += 1
    for i in indexes[last_train_index:]:
        if k % 10 == 0:
            print(k)
        X_test, y_test = load_example(i, X_test, y_test, min_ones)
        k += 1

    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


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

