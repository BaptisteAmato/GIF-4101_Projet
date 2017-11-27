import keras.backend as K
from keras.layers import Activation, ZeroPadding2D, BatchNormalization, Conv2D, UpSampling2D
from keras.layers import MaxPooling2D, Dropout
from keras.models import Sequential

K.set_image_data_format('channels_last')


def MyModel(input_shape):
    model = Sequential([
        ############# ENCODER ###############
        # 224x224x1
        Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv0'),
        # BatchNormalization(axis=3, name='bn0'),

        # 224x224x64
        Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv1'),
        # BatchNormalization(axis=3, name='bn1'),
        # 224x224x64
        MaxPooling2D((2, 2), strides=(2, 2), name='max_pool1'),

        # 112x112x64
        Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv2'),
        # BatchNormalization(axis=3, name='bn2'),

        # 112x112x128
        Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv3'),
        # BatchNormalization(axis=3, name='bn3'),
        MaxPooling2D((2, 2), strides=(2, 2), name='max_pool3'),

        # 56x56x128
        Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv4'),
        # BatchNormalization(axis=3, name='bn4'),

        # 56x56x256
        Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv5'),
        # BatchNormalization(axis=3, name='bn5'),

        # 56x56x256
        Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv6'),
        # BatchNormalization(axis=3, name='bn6'),
        # 56x56x256
        MaxPooling2D((2, 2), strides=(2, 2), name='max_pool6'),

        # 28x28x256
        Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv7'),
        # BatchNormalization(axis=3, name='bn7'),

        # 28x28x512
        Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv8'),
        # BatchNormalization(axis=3, name='bn8'),

        # 28x28x512
        Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv9'),
        # BatchNormalization(axis=3, name='bn9'),
        # 28x28x512
        MaxPooling2D((2, 2), strides=(2, 2), name='max_pool9'),

        # 14x14x512
        Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv10'),
        # BatchNormalization(axis=3, name='bn10'),

        # 14x14x512
        Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv11'),
        # BatchNormalization(axis=3, name='bn11'),

        # 14x14x512
        Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv12'),
        # BatchNormalization(axis=3, name='bn12'),
        # 14x14x512
        MaxPooling2D((2, 2), strides=(2, 2), name='max_pool12'),

        ############# DECODER ###############
        # 7x7x512
        Conv2D(512, (1, 1), strides=(1, 1), activation='relu', name='conv13'),
        # BatchNormalization(axis=3, name='bn13'),
        Dropout(0.3),

        # 7x7x512
        UpSampling2D(size=(2, 2), name='upsampling14'),
        Conv2D(512, (5, 5), strides=(1, 1), padding='same', activation='relu', name='conv14'),
        # BatchNormalization(axis=3, name='bn14'),
        Dropout(0.3),

        UpSampling2D(size=(2, 2), name='upsampling15'),
        Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', name='conv15'),
        # BatchNormalization(axis=3, name='bn15'),
        Dropout(0.3),

        UpSampling2D(size=(2, 2), name='upsampling16'),
        Conv2D(128, (5, 5), strides=(1, 1), padding='same', activation='relu', name='conv16'),
        # BatchNormalization(axis=3, name='bn16'),
        Dropout(0.3),

        UpSampling2D(size=(2, 2), name='upsampling17'),
        Conv2D(64, (5, 5), strides=(1, 1), padding='same', activation='relu', name='conv17'),
        # BatchNormalization(axis=3, name='bn17'),
        Dropout(0.3),

        UpSampling2D(size=(2, 2), name='upsampling18'),
        Conv2D(32, (5, 5), strides=(1, 1), padding='same', activation='relu', name='conv18'),
        # BatchNormalization(axis=3, name='bn18'),
        Dropout(0.3),

        Conv2D(1, (5, 5), strides=(1, 1), padding='same', activation='sigmoid', name='conv19'),
        # BatchNormalization(axis=3, name='bn19'),
    ],

        name='MyModel')

    # 224x224x1

    return model
