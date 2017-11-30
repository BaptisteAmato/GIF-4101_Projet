import sys

from keras.callbacks import ModelCheckpoint

from utils import *


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


def test_image(index, model_name):
    """
    :param index: of the image to test the algorithm.
    :return: the predicted axon and dendrite images.
    """
    x = np.load(folder_images_saving_train_x + "/" + str(index) + ".npy")
    y = np.load(folder_images_saving_train_y + "/" + str(index) + ".npy")
    rows = x.shape[0]
    cols = x.shape[1]
    print(rows)
    print(cols)
    crops_x, _ = get_all_crops(x, None)
    print(crops_x.shape)
    my_model = load_model(model_name)
    predicted_crops = my_model.predict(crops_x, batch_size=1)
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

    merged, _, axon, dendrite = get_images_from_train_label(x, y)
    plt.title("Truth")
    plt.subplot(131)
    plt.imshow(merged)
    plt.subplot(132)
    plt.imshow(axon)
    plt.subplot(133)
    plt.imshow(dendrite)
    prediction, _, predicted_axon, predicted_dendrite = get_images_from_train_label(x, predicted_label)
    plt.figure()
    plt.title("Prediction")
    plt.subplot(131)
    plt.imshow(prediction)
    plt.subplot(132)
    plt.imshow(predicted_axon)
    plt.subplot(133)
    plt.imshow(predicted_dendrite)
    plt.show()
    return predicted_axon, predicted_dendrite


def train_model(model_name="model_yang", nb_images=2, epochs=1, batch_size=1, validation_split=0.3, evaluate=False, show_example=False):
    # Load dataset.
    print("######## LOADING THE MODEL ###########")
    X_train, X_test, y_train, y_test = load_dataset(nb_images)

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
    my_model.compile(optimizer="adam", loss='mean_squared_error', metrics=["accuracy"])
    # Best weights are saved after each epoch.
    checkpointer = ModelCheckpoint(filepath=get_model_weights_path(model_name), verbose=1, save_best_only=True)

    # Run the model.
    print("######## RUNNING THE MODEL ###########")
    hist = my_model.fit(x=X_train, y=y_train, validation_split=validation_split, epochs=epochs, batch_size=batch_size, callbacks=[checkpointer])

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