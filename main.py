import sys

from keras.callbacks import ModelCheckpoint

from utils import *
from model import *


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
    X_train, X_test, y_train, y_test = load_dataset(nb_images)

    nb_train_examples = X_train.shape[0]
    nb_test_examples = X_test.shape[0]
    print("number of training examples = " + str(nb_train_examples))
    print("number of test examples = " + str(nb_test_examples))
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(y_test.shape))

    myModel = MyModel(X_train.shape[1:])
    myModel.compile(optimizer="adam", loss='mean_squared_error', metrics=["accuracy"])
    checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True)
    hist = myModel.fit(x=X_train, y=y_train, validation_split=0.3, epochs=nb_epochs, batch_size=batch_size, callbacks=[checkpointer])
    myModel.load_weights('weights.hdf5')

    print("Testing on " + str(nb_test_examples) + " images...")
    evaluation = myModel.evaluate(x=X_test, y=y_test)
    print("Loss = " + str(evaluation[0]))
    print("Test Accuracy = " + str(evaluation[1]))

