import sys

import keras.backend as K
from tensorflow import boolean_mask, logical_and, logical_not, equal, not_equal
from keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.python.framework.errors_impl import ResourceExhaustedError

from utils import *


def own_loss_function(y_true, y_pred):
    # The predicted values that are colored while the truth is black should be penalized.
    lambd = 10
    mask = equal(y_true, 0)
    mask2 = not_equal(y_pred, 0)
    mask_to_penalize = logical_and(mask, mask2)
    mask_rest = logical_not(mask_to_penalize)

    y_pred_penalized = boolean_mask(y_pred, mask_to_penalize)
    y_true_penalized = boolean_mask(y_true, mask_to_penalize)
    y_pred_not_penalized = boolean_mask(y_pred, mask_rest)
    y_true_not_penalized = boolean_mask(y_true, mask_rest)

    return K.mean(K.square(lambd * (y_pred_penalized - y_true_penalized)), axis=-1) + \
           K.mean(K.square(y_pred_not_penalized - y_true_not_penalized), axis=-1)


def _predict(my_model, crops_x, batch_size):
    batch_size = int(batch_size)
    # In case of a ResourceExhaustedError, the batch_size in divided by 2 and the fit_model() is retried.
    try:
        return my_model.predict(crops_x, batch_size=batch_size)
    except ResourceExhaustedError:
        print("######## ResourceExhaustedError ###########")
        batch_size = int(batch_size) / 2
        print("######## PREDICTING THE CROPS with batch_size = " + str(batch_size) + " ###########")
    return _predict(my_model, crops_x, batch_size)


def test_image(index, model_name, thresh_results=False, threshold=0.1, batch_size=32):
    """
    :param index: of the image to test the algorithm.
    :return: the predicted axon and dendrite images.
    """
    print("########## LOADING THE IMAGE ##############")
    x = np.load(folder_images_saving_train_x + "/" + str(index) + ".npy")
    y = np.load(folder_images_saving_train_y + "/" + str(index) + ".npy")
    rows = x.shape[0]
    cols = x.shape[1]
    print(rows)
    print(cols)
    print("########## CROPPING THE IMAGE ##############")
    crops_x, _ = get_all_crops(x, None)
    print(crops_x.shape)
    my_model = load_model(model_name)
    print("########## PREDICTING THE CROPS ##############")
    predicted_crops = _predict(my_model, crops_x, batch_size)
    print("########## RECONSTITUTING THE IMAGE ##############")
    predicted_label = np.zeros((rows, cols, 2))
    k = 0
    i = 0
    while i < rows - crop_size:
        j = 0
        while j < cols - crop_size:
            predicted_label[i:i+crop_size, j:j+crop_size] = predicted_crops[k]
            k += 1
            j += crop_size
        # Add the end of the last column's crop.
        predicted_label[i:i + crop_size, cols-crop_size:cols] = predicted_crops[k]
        k += 1
        i += crop_size
    j = 0
    while j < cols - crop_size:
        predicted_label[rows-crop_size:rows, j:j + crop_size] = predicted_crops[k]
        k += 1
        j += crop_size
    # Add the end of the last line and column's crop.
    predicted_label[rows-crop_size:rows, cols-crop_size:cols] = predicted_crops[k]
    k += 1

    merged, actin, axon, dendrite = get_images_from_train_label(x, y)
    plt.title("Truth")
    plt.subplot(131)
    plt.imshow(merged)
    plt.subplot(132)
    plt.imshow(axon)
    plt.subplot(133)
    plt.imshow(dendrite)

    prediction, _, predicted_axon, predicted_dendrite = get_images_from_train_label(x, predicted_label)

    if thresh_results:
        _, prediction = cv2.threshold(prediction, threshold, 255, cv2.THRESH_TOZERO)
        _, predicted_axon = cv2.threshold(predicted_axon, threshold, 255, cv2.THRESH_TOZERO)
        _, predicted_dendrite = cv2.threshold(predicted_dendrite, threshold, 255, cv2.THRESH_TOZERO)

    plt.figure()
    plt.title("Prediction")
    plt.subplot(131)
    plt.imshow(prediction)
    plt.subplot(132)
    plt.imshow(predicted_axon)
    plt.subplot(133)
    plt.imshow(predicted_dendrite)
    plt.show()
    return actin, predicted_axon, predicted_dendrite


def _fit_model(my_model, X_train, y_train, validation_split, epochs, batch_size, callbacks):
    batch_size = int(batch_size)
    # In case of a ResourceExhaustedError, the batch_size in divided by 2 and the fit_model() is retried.
    try:
        return my_model.fit(x=X_train, y=y_train, validation_split=validation_split, epochs=epochs, batch_size=batch_size,
                     callbacks=callbacks)
    except ResourceExhaustedError:
        print("######## ResourceExhaustedError ###########")
        batch_size = int(batch_size) / 2
        print("######## RUNNING THE MODEL with batch_size = " + str(batch_size) + " ###########")
    return _fit_model(my_model, X_train, y_train, validation_split, epochs, batch_size, callbacks)


def train_model(model_name="model_yang", return_all=True, nb_examples=2, epochs=1, batch_size=2, validation_split=0.3,
                use_saved_weights=False, evaluate=True, show_example=False):
    # Load dataset.
    print("######## LOADING THE MODEL ###########")
    X_train, X_test, y_train, y_test = load_dataset(return_all=return_all, nb_examples=nb_examples)

    nb_train_examples = X_train.shape[0]
    nb_test_examples = X_test.shape[0]
    print("number of training examples = " + str(nb_train_examples))
    print("number of test examples = " + str(nb_test_examples))
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(y_test.shape))

    # Get the right model.
    model_module = __import__("models")
    get_model = getattr(model_module, model_name)

    my_model = get_model(X_train.shape[1:])
    # my_model = get_model((crop_size, crop_size, 1))
    if use_saved_weights:
        my_model.load_weights(get_model_weights_path(model_name))
    # my_model.compile(optimizer="adam", loss='mean_squared_error', metrics=["accuracy"])
    my_model.compile(optimizer="adam", loss=own_loss_function, metrics=["accuracy"])
    # Best weights are saved after each epoch.
    checkpointer = ModelCheckpoint(filepath=get_model_weights_path(model_name), verbose=1, save_best_only=True)
    # Write output to a file after each epoch.
    csv_logger = CSVLogger(main_folder_path + '/keras_log.csv', append=True, separator=';')

    # Run the model.
    print("######## RUNNING THE MODEL ###########")
    _fit_model(my_model, X_train, y_train, validation_split, epochs, batch_size, [checkpointer, csv_logger])

    # Load the best weights.
    print("######## LOADING THE BEST WEIGHTS ###########")
    my_model.load_weights(get_model_weights_path(model_name))

    # Save the model to json.
    model_json = my_model.to_json()
    with open(get_model_path(model_name), "w") as json_file:
        json_file.write(model_json)

    if evaluate:
        # Evaluate the model.
        print("######## EVALUATING THE MODEL ###########")
        preds = my_model.evaluate(x=X_test, y=y_test)
        print()
        print("Loss = " + str(preds[0]))
        print("Test Accuracy = " + str(preds[1]))

    if show_example:
        # Test on image.
        print("######## TESTING THE MODEL ON AN IMAGE ###########")
        index = np.random.randint(nb_test_examples)
        test_image(index, model_name)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python main.py <number_of_images_to_load (integer)> <epochs (integer)> <batch_size (integer)>")
        exit()
    nb_images = sys.argv[1]
    nb_epochs = sys.argv[2]
    batch_size = sys.argv[1]
    try:
        nb_images = int(nb_images)
        nb_epochs = int(nb_epochs)
        batch_size = int(batch_size)
    except ValueError:
        print("Usage: python main.py <number_of_images_to_load (integer)> <epochs (integer)> <batch_size (integer)>")
        exit()

    train_model("model_yang", nb_images, nb_epochs, batch_size)