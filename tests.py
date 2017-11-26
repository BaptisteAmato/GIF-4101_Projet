from keras.layers import Activation, ZeroPadding2D, BatchNormalization, Conv2D, UpSampling2D
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, ZeroPadding2D, BatchNormalization, Conv2D, UpSampling2D
from keras.layers import MaxPooling2D, Dropout
from keras.models import Sequential
from utils import *
K.set_image_data_format('channels_last')


# def MyModel(input_shape):
#     X_input = Input(input_shape)
#
#     ############# ENCODER ###############
#     # 224x224x1
#     #     X = ZeroPadding2D((1, 1))(X_input)
#     X = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv0')(X_input)
#     #     X = BatchNormalization(axis = 3, name = 'bn0')(X)
#     #     X = Activation('relu')(X)
#
#
#     # 224x224x64
#     #     X = ZeroPadding2D((1, 1))(X)
#     X = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv1')(X)
#     #     X = BatchNormalization(axis = 3, name = 'bn1')(X)
#     #     X = Activation('relu')(X)
#     # 224x224x64
#     X = MaxPooling2D((2, 2), strides=(2, 2), name='max_pool1')(X)
#
#     # 112x112x64
#     #     X = ZeroPadding2D((1, 1))(X)
#     X = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv2')(X)
#     #     X = BatchNormalization(axis = 3, name = 'bn2')(X)
#     #     X = Activation('relu')(X)
#
#     # 112x112x128
#     #     X = ZeroPadding2D((1, 1))(X)
#     X = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv3')(X)
#     #     X = BatchNormalization(axis = 3, name = 'bn3')(X)
#     #     X = Activation('relu')(X)
#     # 112x112x128
#     X = MaxPooling2D((2, 2), strides=(2, 2), name='max_pool3')(X)
#
#     # 56x56x128
#     #     X = ZeroPadding2D((1, 1))(X)
#     X = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv4')(X)
#     #     X = BatchNormalization(axis = 3, name = 'bn4')(X)
#     #     X = Activation('relu')(X)
#
#     # 56x56x256
#     #     X = ZeroPadding2D((1, 1))(X)
#     X = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv5')(X)
#     #     X = BatchNormalization(axis = 3, name = 'bn5')(X)
#     #     X = Activation('relu')(X)
#
#     # 56x56x256
#     #     X = ZeroPadding2D((1, 1))(X)
#     X = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv6')(X)
#     #     X = BatchNormalization(axis = 3, name = 'bn6')(X)
#     #     X = Activation('relu')(X)
#     # 56x56x256
#     X = MaxPooling2D((2, 2), strides=(2, 2), name='max_pool6')(X)
#
#     # 28x28x256
#     #     X = ZeroPadding2D((1, 1))(X)
#     X = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv7')(X)
#     #     X = BatchNormalization(axis = 3, name = 'bn7')(X)
#     #     X = Activation('relu')(X)
#
#     # 28x28x512
#     #     X = ZeroPadding2D((1, 1))(X)
#     X = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv8')(X)
#     #     X = BatchNormalization(axis = 3, name = 'bn8')(X)
#     #     X = Activation('relu')(X)
#
#     # 28x28x512
#     #     X = ZeroPadding2D((1, 1))(X)
#     X = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv9')(X)
#     #     X = BatchNormalization(axis = 3, name = 'bn9')(X)
#     #     X = Activation('relu')(X)
#     # 28x28x512
#     X = MaxPooling2D((2, 2), strides=(2, 2), name='max_pool9')(X)
#
#     # 14x14x512
#     #     X = ZeroPadding2D((1, 1))(X)
#     X = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv10')(X)
#     #     X = BatchNormalization(axis = 3, name = 'bn10')(X)
#     #     X = Activation('relu')(X)
#
#     # 14x14x512
#     #     X = ZeroPadding2D((1, 1))(X)
#     X = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv11')(X)
#     #     X = BatchNormalization(axis = 3, name = 'bn11')(X)
#     #     X = Activation('relu')(X)
#
#     # 14x14x512
#     #     X = ZeroPadding2D((1, 1))(X)
#     X = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv12')(X)
#     #     X = BatchNormalization(axis = 3, name = 'bn12')(X)
#     #     X = Activation('relu')(X)
#     # 14x14x512
#     X = MaxPooling2D((2, 2), strides=(2, 2), name='max_pool12')(X)
#
#     # 7x7x512
#     print("--")
#     print(X.shape)
#
#     ############# DECODER ###############
#     # 7x7x512
#     X = Conv2D(512, (1, 1), strides=(1, 1), activation='relu', name='conv13')(X)
#     #     X = BatchNormalization(axis = 3, name = 'bn13')(X)
#     #     X = Activation('relu')(X)
#
#     # 7x7x512
#     X = UpSampling2D(size=(2, 2), name='upsampling14')(X)
#     #     X = ZeroPadding2D((2, 2))(X)
#     X = Conv2D(512, (5, 5), strides=(1, 1), padding='same', activation='relu', name='conv14')(X)
#     #     X = BatchNormalization(axis = 3, name = 'bn14')(X)
#     #     X = Activation('relu')(X)
#
#     X = UpSampling2D(size=(2, 2), name='upsampling15')(X)
#     #     X = ZeroPadding2D((2, 2))(X)
#     X = Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', name='conv15')(X)
#     #     X = BatchNormalization(axis = 3, name = 'bn15')(X)
#     #     X = Activation('relu')(X)
#
#     X = UpSampling2D(size=(2, 2), name='upsampling16')(X)
#     #     X = ZeroPadding2D((2, 2))(X)
#     X = Conv2D(128, (5, 5), strides=(1, 1), padding='same', activation='relu', name='conv16')(X)
#     #     X = BatchNormalization(axis = 3, name = 'bn16')(X)
#     #     X = Activation('relu')(X)
#
#     X = UpSampling2D(size=(2, 2), name='upsampling17')(X)
#     #     X = ZeroPadding2D((2, 2))(X)
#     X = Conv2D(64, (5, 5), strides=(1, 1), padding='same', activation='relu', name='conv17')(X)
#     #     X = BatchNormalization(axis = 3, name = 'bn17')(X)
#     #     X = Activation('relu')(X)
#
#     X = UpSampling2D(size=(2, 2), name='upsampling18')(X)
#     #     X = ZeroPadding2D((2, 2))(X)
#     X = Conv2D(32, (5, 5), strides=(1, 1), padding='same', activation='relu', name='conv18')(X)
#     #     X = BatchNormalization(axis = 3, name = 'bn18')(X)
#     #     X = Activation('relu')(X)
#
#     #     X = ZeroPadding2D((2, 2))(X)
#     X = Conv2D(1, (5, 5), strides=(1, 1), padding='same', activation='sigmoid', name='conv19')(X)
#     #     X = BatchNormalization(axis = 3, name = 'bn19')(X)
#     #     X = Activation('sigmoid')(X)
#
#     # 224x224x1
#     print("--")
#     print(X.shape)
#
#     model = Model(inputs=X_input, outputs=X, name='MyModel')
#
#     return model


# TODO: use Conv2DTranspose
def MyModel(input_shape, dropout=0.3):
    model = Sequential([
        ############# ENCODER ###############
        # 224x224x1
        Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv0', input_shape=input_shape),
        # 224x224x64
        Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv1'),
        # 224x224x64
        MaxPooling2D((2, 2), strides=(2, 2), name='max_pool1'),
        # 112x112x64
        Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv2'),
        # 112x112x128
        Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv3'),
        # 112x112x128
        MaxPooling2D((2, 2), strides=(2, 2), name='max_pool3'),
        # 56x56x128
        Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv4'),
        # 56x56x256
        Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv5'),
        # 56x56x256
        Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv6'),
        # 56x56x256
        MaxPooling2D((2, 2), strides=(2, 2), name='max_pool6'),
        # 28x28x256
        Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv7'),
        # 28x28x512
        Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv8'),
        # 28x28x512
        Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv9'),
        # 28x28x512
        MaxPooling2D((2, 2), strides=(2, 2), name='max_pool9'),
        # 14x14x512
        Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv10'),
        # 14x14x512
        Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv11'),
        # 14x14x512
        Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv12'),
        # 14x14x512
        MaxPooling2D((2, 2), strides=(2, 2), name='max_pool12'),

        ############# DECODER ###############
        # 7x7x512
        Conv2D(512, (1, 1), strides=(1, 1), activation='relu', name='conv13'),
        Dropout(dropout),
        # 7x7x512
        UpSampling2D(size=(2, 2), name='upsampling14'),
        # 14x14x512
        Conv2D(512, (5, 5), strides=(1, 1), padding='same', activation='relu', name='conv14'),
        Dropout(dropout),
        # 14x14x512
        UpSampling2D(size=(2, 2), name='upsampling15'),
        # 28x28x512
        Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', name='conv15'),
        Dropout(dropout),
        # 28x28x256
        UpSampling2D(size=(2, 2), name='upsampling16'),
        # 56x56x256
        Conv2D(128, (5, 5), strides=(1, 1), padding='same', activation='relu', name='conv16'),
        Dropout(dropout),
        # 56x56x128
        UpSampling2D(size=(2, 2), name='upsampling17'),
        # 112x112x128
        Conv2D(64, (5, 5), strides=(1, 1), padding='same', activation='relu', name='conv17'),
        Dropout(dropout),
        # 112x112x64
        UpSampling2D(size=(2, 2), name='upsampling18'),
        # 224x224x64
        Conv2D(32, (5, 5), strides=(1, 1), padding='same', activation='relu', name='conv18'),
        Dropout(dropout),
        # 224x224x32
        Conv2D(1, (5, 5), strides=(1, 1), padding='same', activation='sigmoid', name='conv19'),
        # 224x224x1
        ],
        name='MyModel')

    return model


if __name__ == '__main__':

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
    #
    # # Run the model.
    # hist = myModel.fit(x=X_train, y=Y_train, validation_split=0.3, epochs=10, batch_size=50, callbacks=[checkpointer])
    # # Load the best weights (maybe over-fitting after some epochs).
    # myModel.load_weights('weights.hdf5')
    #
    # # Evaluation of the model.
    # preds = myModel.evaluate(x=X_test, y=Y_test)
    # print()
    # print("Loss = " + str(preds[0]))
    # print("Test Accuracy = " + str(preds[1]))
    #
    # # Test on image: actin + dendrite (no axon for now).
    # index = 0
    # x = np.load(folder_images_saving_train_x + '/' + str(index) + '.npy')
    # y = np.load(folder_images_saving_train_x + '/' + str(index) + '.npy')
    # crops_x, crops_y = get_all_crops(x, y, crop_size)
    # crop_index = 0
    # x = crops_x[crop_index]
    # y = crops_y[crop_index]
    # img_x, _, img_y = get_image_from_contour_map(x, None, y)
    # print()
    # test = np.zeros((1, x.shape[0], x.shape[1], x.shape[2]))
    # test[0] = x
    #
    # prediction = myModel.predict(test)
    # prediction[prediction <= 0.5] = 0
    # prediction[prediction > 0.5] = 1
    #
    # # Input
    # plt.imshow(img_x)
    # # Prediction
    # plt.figure()
    # plt.imshow(get_image_from_contour_map(prediction[0], 'r'))
    # # Truth
    # plt.figure()
    # plt.imshow(img_y)
