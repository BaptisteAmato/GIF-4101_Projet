from keras.callbacks import ModelCheckpoint

from model import *
from utils import *


if __name__ == '__main__':

    # Load dataset.
    nb_images = 5
    X_train, X_test, y_train, y_test = load_dataset(nb_images)

    nb_train_examples = X_train.shape[0]
    nb_test_examples = X_test.shape[0]
    print("number of training examples = " + str(nb_train_examples))
    print("number of test examples = " + str(nb_test_examples))
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(y_test.shape))

    # Load model. Best weights are saved after each epoch.
    my_model = MyModel(X_train.shape[1:])
    my_model.compile(optimizer="adam", loss='mean_squared_error', metrics=["accuracy"])
    checkpointer = ModelCheckpoint(filepath=output_folder_path+"/weights.hdf5", verbose=1, save_best_only=True)

    # Run the model.
    hist = my_model.fit(x=X_train, y=y_train, validation_split=0.3, epochs=5, batch_size=4, callbacks=[checkpointer])
    # Save the model to json.
    model_json = my_model.to_json()
    with open(output_folder_path+"/myModel.json", "w") as json_file:
        json_file.write(model_json)

    # Load the best weights.
    my_model.load_weights(output_folder_path + '/weights.hdf5')

    # Evaluate the model.
    preds = my_model.evaluate(x=X_test, y=y_test)
    print()
    print("Loss = " + str(preds[0]))
    print("Test Accuracy = " + str(preds[1]))

    # Test on image.
    index = np.random.randint(nb_test_examples)
    test_image(folder_images_saving_train_x + "/" + str(i) + ".npy")
