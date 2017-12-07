from keras.models import model_from_json

from data_augmentation import *
from image_processing import *


def get_images_from_train_label(train, label, channel):
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
        # Axon is saved in channel 0, dendrite is saved in channel 1.
        # Axon's color is red (non-zero channel: 0), dendrite is blue (non-zero channel: 2)
        if channel == 'axons':
            axon_or_dendrite_index = 0
            color_index = 0
        elif channel == 'dendrites':
            axon_or_dendrite_index = 1
            color_index = 2
        else:
            raise ValueError('channel attribute must either be "axons" or "dendrites"')
        axon_or_dendrite = np.zeros((rows, cols, 3))
        axon_or_dendrite[:, :, color_index] = np.squeeze(label[:, :, axon_or_dendrite_index])
    else:
        axon_or_dendrite = np.array([])

    return actin, axon_or_dendrite


def load_model(model_name, channel, binary_masks):
    """

    :param model_name:
    :param channel:
    :param binary:
    :return:
    """
    # Load the model.
    path = get_model_path(model_name)
    print("Loading " + path)
    with open(path) as f:
        my_model = model_from_json(f.read())
    # Load the weights.
    path = get_model_weights_path(model_name, channel, binary_masks)
    print("Loading " + path)
    my_model.load_weights(path)
    # Compile the model
    my_model.compile(optimizer="adam", loss='mean_squared_error', metrics=["accuracy"])
    return my_model

