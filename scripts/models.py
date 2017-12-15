from keras.layers import BatchNormalization, Conv2D, UpSampling2D, Activation, Conv2DTranspose
from keras.layers import MaxPooling2D, Dropout
from keras.models import Sequential
import keras.backend as K
from tensorflow import boolean_mask, logical_and, logical_not, less, greater


def model_yang(input_shape):
    model = Sequential([
        ############# ENCODER ###############
        # 224x224x1
        Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv0', input_shape=input_shape),
        BatchNormalization(axis=3, name='bn0'),
        Activation('relu'),

        # 224x224x64
        Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv1'),
        BatchNormalization(axis=3, name='bn1'),
        Activation('relu'),
        # 224x224x64
        MaxPooling2D((2, 2), strides=(2, 2), name='max_pool1'),

        # 112x112x64
        Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv2'),
        BatchNormalization(axis=3, name='bn2'),
        Activation('relu'),

        # 112x112x128
        Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv3'),
        BatchNormalization(axis=3, name='bn3'),
        Activation('relu'),
        MaxPooling2D((2, 2), strides=(2, 2), name='max_pool3'),

        # 56x56x128
        Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv4'),
        BatchNormalization(axis=3, name='bn4'),
        Activation('relu'),

        # 56x56x256
        Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv5'),
        BatchNormalization(axis=3, name='bn5'),
        Activation('relu'),

        # 56x56x256
        Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv6'),
        BatchNormalization(axis=3, name='bn6'),
        Activation('relu'),
        # 56x56x256
        MaxPooling2D((2, 2), strides=(2, 2), name='max_pool6'),

        # 28x28x256
        Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv7'),
        BatchNormalization(axis=3, name='bn7'),
        Activation('relu'),

        # 28x28x512
        Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv8'),
        BatchNormalization(axis=3, name='bn8'),
        Activation('relu'),

        # 28x28x512
        Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv9'),
        BatchNormalization(axis=3, name='bn9'),
        Activation('relu'),
        # 28x28x512
        MaxPooling2D((2, 2), strides=(2, 2), name='max_pool9'),

        # 14x14x512
        Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv10'),
        BatchNormalization(axis=3, name='bn10'),
        Activation('relu'),

        # 14x14x512
        Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv11'),
        BatchNormalization(axis=3, name='bn11'),
        Activation('relu'),

        # 14x14x512
        Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv12'),
        BatchNormalization(axis=3, name='bn12'),
        Activation('relu'),
        # 14x14x512
        MaxPooling2D((2, 2), strides=(2, 2), name='max_pool12'),

        # 7x7x512
        Conv2D(4096, (3, 3), strides=(1, 1), padding='same', name='conv12_bis'),
        BatchNormalization(axis=3, name='bn12_bis'),
        Activation('relu'),

        ############# DECODER ###############
        # 7x7x4096
        Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv13'),
        BatchNormalization(axis=3, name='bn13'),
        Activation('relu'),
        Dropout(0.5),

        # 7x7x512
        UpSampling2D(size=(2, 2), name='upsampling14'),
        Conv2D(512, (5, 5), strides=(1, 1), padding='same', name='conv14'),
        BatchNormalization(axis=3, name='bn14'),
        Activation('relu'),
        Dropout(0.5),

        # 14x14x512
        UpSampling2D(size=(2, 2), name='upsampling15'),
        Conv2D(256, (5, 5), strides=(1, 1), padding='same', name='conv15'),
        BatchNormalization(axis=3, name='bn15'),
        Activation('relu'),
        Dropout(0.5),

        # 28x28x256
        UpSampling2D(size=(2, 2), name='upsampling16'),
        Conv2D(128, (5, 5), strides=(1, 1), padding='same', name='conv16'),
        BatchNormalization(axis=3, name='bn16'),
        Activation('relu'),
        Dropout(0.5),

        # 56x56x128
        UpSampling2D(size=(2, 2), name='upsampling17'),
        Conv2D(64, (5, 5), strides=(1, 1), padding='same', name='conv17'),
        BatchNormalization(axis=3, name='bn17'),
        Activation('relu'),
        Dropout(0.5),

        # 112x112x64
        UpSampling2D(size=(2, 2), name='upsampling18'),
        Conv2D(32, (5, 5), strides=(1, 1), padding='same', name='conv18'),
        BatchNormalization(axis=3, name='bn18'),
        Activation('relu'),
        Dropout(0.5),

        # 224x224x32
        Conv2D(1, (5, 5), strides=(1, 1), padding='same', name='conv19'),
        BatchNormalization(axis=3, name='bn19'),
        Activation('sigmoid'),
    ],

        name='model_yang')

    # 224x224x1

    return model