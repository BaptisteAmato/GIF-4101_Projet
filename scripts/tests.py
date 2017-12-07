from keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.python.framework.errors_impl import ResourceExhaustedError

from dataset import *
from image_processing import *
from models import own_loss_function


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


def test_image(index, model_name, channel, binary_masks, thresh_results=False, threshold=0.1, batch_size=32, apply_actin_mask=True):
    """
    Compute prediction of an actin image
    :param index:
    :param model_name:
    :param thresh_results:
    :param threshold:
    :param batch_size:
    :param binary_masks:
    :return:
    """
    print("########## LOADING THE IMAGE ##############")
    x = np.load(get_folder_images_saving_train_x(binary_masks) + "/" + str(index) + ".npy")
    y = np.load(get_folder_images_saving_train_y(binary_masks) + "/" + str(index) + ".npy")
    rows = x.shape[0]
    cols = x.shape[1]
    print(rows)
    print(cols)
    print("########## CROPPING THE IMAGE ##############")
    crops_x, _ = get_all_crops(x, None)
    print(crops_x.shape)
    my_model = load_model(model_name, channel, binary_masks)
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

    actin, axon_or_dendrite = get_images_from_train_label(x, y, channel)
    plt.title("Truth")
    plt.subplot(121)
    plt.imshow(actin)
    plt.subplot(122)
    plt.imshow(axon_or_dendrite)

    actin, predicted_axon_or_dendrite = get_images_from_train_label(x, predicted_label, channel)

    if thresh_results:
        _, predicted_axon_or_dendrite = cv2.threshold(predicted_axon_or_dendrite, threshold, 255, cv2.THRESH_TOZERO)

    if apply_actin_mask:
        predicted_axon_or_dendrite = keep_only_actin_mask_on_prediction(actin, predicted_axon_or_dendrite)

    plt.figure()
    plt.title("Prediction")
    plt.subplot(131)
    plt.imshow(actin)
    plt.subplot(132)
    plt.imshow(predicted_axon_or_dendrite)
    plt.show()
    return actin, predicted_axon_or_dendrite


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


def train_model(model_name="model_yang", nb_examples=2, epochs=1, batch_size=2, validation_split=0.3,
                use_saved_weights=False, evaluate=True, show_example=False, channel='axons', binary_masks=True):
    # Load dataset.
    print("######## LOADING THE MODEL ###########")
    X_train, X_test, y_train, y_test = load_dataset(nb_examples, binary_masks, channel)

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
        my_model.load_weights(get_model_weights_path(model_name, channel, binary_masks))

    if binary_masks:
        loss = 'binary_crossentropy'
        metrics = ['binary_accuracy']
    else:
        # loss = 'mean_squared_error'
        loss = own_loss_function
        metrics = ['accuracy']
    my_model.compile(optimizer="adam", loss=loss, metrics=metrics)

    # Save the model to json.
    model_json = my_model.to_json()
    with open(get_model_path(model_name), "w") as json_file:
        json_file.write(model_json)

    # Best weights are saved after each epoch.
    checkpointer = ModelCheckpoint(filepath=get_model_weights_path(model_name, channel, binary_masks), verbose=1, save_best_only=True)
    # Write output to a file after each epoch.
    csv_logger = CSVLogger(main_folder_path + '/keras_log.csv', append=True, separator=';')

    # Run the model.
    print("######## RUNNING THE MODEL ###########")
    _fit_model(my_model, X_train, y_train, validation_split, epochs, batch_size, [checkpointer, csv_logger])

    # Load the best weights.
    print("######## LOADING THE BEST WEIGHTS ###########")
    my_model.load_weights(get_model_weights_path(model_name, channel, binary_masks))

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

