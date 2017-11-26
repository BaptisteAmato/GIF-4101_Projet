from keras.layers import Activation, ZeroPadding2D, BatchNormalization, Conv2D, UpSampling2D
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, ZeroPadding2D, BatchNormalization, Conv2D, UpSampling2D
from keras.layers import MaxPooling2D, Dropout
from keras.models import Sequential

K.set_image_data_format('channels_last')

from utils import *


def MyModel(input_shape):
    model = Sequential([
        ############# ENCODER ###############
        # 224x224x1
        ZeroPadding2D((1, 1), input_shape=input_shape),
        Conv2D(64, (3, 3), strides=(1, 1), name='conv0'),
        BatchNormalization(axis=3, name='bn0'),
        Activation('relu'),

        # 224x224x64
        ZeroPadding2D((1, 1)),
        Conv2D(64, (3, 3), strides=(1, 1), name='conv1'),
        BatchNormalization(axis=3, name='bn1'),
        Activation('relu'),
        # 224x224x64
        MaxPooling2D((2, 2), strides=(2, 2), name='max_pool1'),

        # 112x112x64
        ZeroPadding2D((1, 1)),
        Conv2D(128, (3, 3), strides=(1, 1), name='conv2'),
        BatchNormalization(axis=3, name='bn2'),
        Activation('relu'),

        # 112x112x128
        ZeroPadding2D((1, 1)),
        Conv2D(128, (3, 3), strides=(1, 1), name='conv3'),
        BatchNormalization(axis=3, name='bn3'),
        Activation('relu'),
        MaxPooling2D((2, 2), strides=(2, 2), name='max_pool3'),

        # 56x56x128
        ZeroPadding2D((1, 1)),
        Conv2D(256, (3, 3), strides=(1, 1), name='conv4'),
        BatchNormalization(axis=3, name='bn4'),
        Activation('relu'),

        # 56x56x256
        ZeroPadding2D((1, 1)),
        Conv2D(256, (3, 3), strides=(1, 1), name='conv5'),
        BatchNormalization(axis=3, name='bn5'),
        Activation('relu'),

        # 56x56x256
        ZeroPadding2D((1, 1)),
        Conv2D(256, (3, 3), strides=(1, 1), name='conv6'),
        BatchNormalization(axis=3, name='bn6'),
        Activation('relu'),
        # 56x56x256
        MaxPooling2D((2, 2), strides=(2, 2), name='max_pool6'),

        # 28x28x256
        ZeroPadding2D((1, 1)),
        Conv2D(512, (3, 3), strides=(1, 1), name='conv7'),
        BatchNormalization(axis=3, name='bn7'),
        Activation('relu'),

        # 28x28x512
        ZeroPadding2D((1, 1)),
        Conv2D(512, (3, 3), strides=(1, 1), name='conv8'),
        BatchNormalization(axis=3, name='bn8'),
        Activation('relu'),

        # 28x28x512
        ZeroPadding2D((1, 1)),
        Conv2D(512, (3, 3), strides=(1, 1), name='conv9'),
        BatchNormalization(axis=3, name='bn9'),
        Activation('relu'),
        # 28x28x512
        MaxPooling2D((2, 2), strides=(2, 2), name='max_pool9'),

        # 14x14x512
        ZeroPadding2D((1, 1)),
        Conv2D(512, (3, 3), strides=(1, 1), name='conv10'),
        BatchNormalization(axis=3, name='bn10'),
        Activation('relu'),

        # 14x14x512
        ZeroPadding2D((1, 1)),
        Conv2D(512, (3, 3), strides=(1, 1), name='conv11'),
        BatchNormalization(axis=3, name='bn11'),
        Activation('relu'),

        # 14x14x512
        ZeroPadding2D((1, 1)),
        Conv2D(512, (3, 3), strides=(1, 1), name='conv12'),
        BatchNormalization(axis=3, name='bn12'),
        Activation('relu'),
        # 14x14x512
        MaxPooling2D((2, 2), strides=(2, 2), name='max_pool12'),

        ############# DECODER ###############
        # 7x7x512
        Conv2D(512, (1, 1), strides=(1, 1), name='conv13'),
        BatchNormalization(axis=3, name='bn13'),
        Activation('relu'),
        Dropout(0.5),

        # 7x7x512
        UpSampling2D(size=(2, 2), name='upsampling14'),
        ZeroPadding2D((2, 2)),
        Conv2D(512, (5, 5), strides=(1, 1), name='conv14'),
        BatchNormalization(axis=3, name='bn14'),
        Activation('relu'),
        Dropout(0.5),

        UpSampling2D(size=(2, 2), name='upsampling15'),
        ZeroPadding2D((2, 2)),
        Conv2D(256, (5, 5), strides=(1, 1), name='conv15'),
        BatchNormalization(axis=3, name='bn15'),
        Activation('relu'),
        Dropout(0.5),

        UpSampling2D(size=(2, 2), name='upsampling16'),
        ZeroPadding2D((2, 2)),
        Conv2D(128, (5, 5), strides=(1, 1), name='conv16'),
        BatchNormalization(axis=3, name='bn16'),
        Activation('relu'),
        Dropout(0.5),

        UpSampling2D(size=(2, 2), name='upsampling17'),
        ZeroPadding2D((2, 2)),
        Conv2D(64, (5, 5), strides=(1, 1), name='conv17'),
        BatchNormalization(axis=3, name='bn17'),
        Activation('relu'),
        Dropout(0.5),

        UpSampling2D(size=(2, 2), name='upsampling18'),
        ZeroPadding2D((2, 2)),
        Conv2D(32, (5, 5), strides=(1, 1), name='conv18'),
        BatchNormalization(axis=3, name='bn18'),
        Activation('relu'),
        Dropout(0.5),

        ZeroPadding2D((2, 2)),
        Conv2D(1, (5, 5), strides=(1, 1), name='conv19'),
        BatchNormalization(axis=3, name='bn19'),
        Activation('sigmoid')],
        name='MyModel')

    # 224x224x1

    return model


# Load dataset.
nb_examples = 2
crop_size = 224
X_train, Y_train, X_test, Y_test = load_dataset(nb_examples, crop_size)
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

# Load model. Best weights are saved after each epoch.
myModel = MyModel(X_train.shape[1:])
myModel.compile(optimizer="adam", loss='binary_crossentropy', metrics=["accuracy"])
checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True)
# Run the model.
hist = myModel.fit(x=X_train, y=Y_train, validation_split=0.3, epochs=10, batch_size=50, callbacks=[checkpointer])
# Load the best weights (maybe over-fitting after some epochs).
myModel.load_weights('weights.hdf5')

# Evaluation of the model.
preds = myModel.evaluate(x=X_test, y=Y_test)
print()
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))

# Test on image: actin + dendrite (no axon for now).
index = 0
x = np.load(folder_images_saving_train_x + '/' + str(index) + '.npy')
y = np.load(folder_images_saving_train_x + '/' + str(index) + '.npy')
crops_x, crops_y = get_all_crops(x, y, crop_size)
crop_index = 0
x = crops_x[crop_index]
y = crops_y[crop_index]
img_x, _, img_y = get_image_from_contour_map(x, None, y)
print()
test = np.zeros((1, x.shape[0], x.shape[1], x.shape[2]))
test[0] = x

prediction = myModel.predict(test)
prediction[prediction <= 0.5] = 0
prediction[prediction > 0.5] = 1

# Input
plt.imshow(img_x)
# Prediction
plt.figure()
plt.imshow(get_image_from_contour_map(prediction[0], 'r'))
# Truth
plt.figure()
plt.imshow(img_y)
