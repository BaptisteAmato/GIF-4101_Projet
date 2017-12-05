from keras.models import model_from_json

from data_augmentation import *
from image_processing import *


def get_images_from_train_label(train, label):
    """
    Returns the images from train and label data (images that can then be plotted)
    :param train:
    :param label:
    :return:
    """
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

