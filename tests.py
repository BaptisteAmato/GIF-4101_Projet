from keras.callbacks import ModelCheckpoint

from model import *
from utils import *

K.set_image_data_format('channels_last')


if __name__ == '__main__':

    # Load dataset.
    nb_images = 100
    X_train, X_test, y_train, y_test = load_dataset(nb_images)

    nb_examples = X_train.shape[0]
    print("number of training examples = " + str(nb_examples))
    print("number of test examples = " + str(X_test.shape[0]))
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(y_test.shape))

    # Load model. Best weights are saved after each epoch.
    myModel = MyModel(X_train.shape[1:])
    myModel.compile(optimizer="adam", loss='mean_squared_error', metrics=["accuracy"])
    checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True)

    # Run the model.
    hist = myModel.fit(x=X_train, y=y_train, validation_split=0.3, epochs=5, batch_size=4, callbacks=[checkpointer])
    # Load the best weights.
    myModel.load_weights('weights.hdf5')
    # Save the model to json.
    model_json = myModel.to_json()
    with open("myModel.json", "w") as json_file:
        json_file.write(model_json)

    # Evaluate the model.
    preds = myModel.evaluate(x=X_test, y=y_test)
    print()
    print("Loss = " + str(preds[0]))
    print("Test Accuracy = " + str(preds[1]))

    # Test on image.
    index = np.random.randint(nb_examples)
    test = X_test[index]
    label = y_test[index]
    true_actin, true_axon, true_dendrite = get_images_from_train_label(test, label)

    data = np.zeros((1, test.shape[0], test.shape[1], test.shape[2]))
    data[0] = test
    prediction = myModel.predict(data)
    prediction[prediction <= 0.5] = 0
    prediction[prediction > 0.5] = 1
    print(prediction[0].shape)
    predicted_axon, predicted_dendrite = get_axon_dendrite_from_label(prediction[0])

    # Input
    plt.title("Actin")
    plt.imshow(true_actin)
    # Prediction
    plt.figure()
    plt.title("Predicted axon")
    plt.imshow(predicted_axon)
    plt.figure()
    plt.title("Predicted dendrite")
    plt.imshow(predicted_dendrite)
    # Truth
    plt.figure()
    plt.title("True axon")
    plt.imshow(true_axon)
    plt.figure()
    plt.title("True dendrite")
    plt.imshow(true_dendrite)
