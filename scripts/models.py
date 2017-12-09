from keras.layers import BatchNormalization, Conv2D, UpSampling2D, Activation, Conv2DTranspose
from keras.layers import MaxPooling2D, Dropout
from keras.models import Sequential
import keras.backend as K
from tensorflow import boolean_mask, logical_and, logical_not, less, greater


# def own_loss_function(y_true, y_pred):
#     # The predicted values that are colored while the truth is black should be penalized.
#     lambd = 10
#     limit = 0.2
#     mask = less(y_true, limit)
#     mask2 = greater(y_pred, limit)
#     mask_to_penalize = logical_and(mask, mask2)
#     mask_rest = logical_not(mask_to_penalize)
#
#     y_pred_penalized = boolean_mask(y_pred, mask_to_penalize)
#     y_true_penalized = boolean_mask(y_true, mask_to_penalize)
#     y_pred_not_penalized = boolean_mask(y_pred, mask_rest)
#     y_true_not_penalized = boolean_mask(y_true, mask_rest)
#
#     return K.mean(K.square(lambd * (y_pred_penalized - y_true_penalized)), axis=-1) + \
#            K.mean(K.square(y_pred_not_penalized - y_true_not_penalized), axis=-1)


def model_yang(input_shape):
    model = Sequential([
        ############# ENCODER ###############
        # 448x448x1
        Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv0', input_shape=input_shape),
        BatchNormalization(axis=3, name='bn0'),
        Activation('relu'),

        # 448x448x64
        Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv1'),
        BatchNormalization(axis=3, name='bn1'),
        Activation('relu'),
        # 448x448x64
        MaxPooling2D((2, 2), strides=(2, 2), name='max_pool1'),

        # 224x224x64
        Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv2'),
        BatchNormalization(axis=3, name='bn2'),
        Activation('relu'),

        # 224x224x128
        Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv3'),
        BatchNormalization(axis=3, name='bn3'),
        Activation('relu'),
        MaxPooling2D((2, 2), strides=(2, 2), name='max_pool3'),

        # 112x112x128
        Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv4'),
        BatchNormalization(axis=3, name='bn4'),
        Activation('relu'),

        # 112x112x256
        Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv5'),
        BatchNormalization(axis=3, name='bn5'),
        Activation('relu'),

        # 112x112x256
        Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv6'),
        BatchNormalization(axis=3, name='bn6'),
        Activation('relu'),
        # 112x112x256
        MaxPooling2D((2, 2), strides=(2, 2), name='max_pool6'),

        # 56x56x256
        Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv7'),
        BatchNormalization(axis=3, name='bn7'),
        Activation('relu'),

        # 56x56x512
        Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv8'),
        BatchNormalization(axis=3, name='bn8'),
        Activation('relu'),

        # 56x56x512
        Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv9'),
        BatchNormalization(axis=3, name='bn9'),
        Activation('relu'),
        # 56x56x512
        MaxPooling2D((2, 2), strides=(2, 2), name='max_pool9'),

        # 28x28x512
        Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv10'),
        BatchNormalization(axis=3, name='bn10'),
        Activation('relu'),

        # 28x28x512
        Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv11'),
        BatchNormalization(axis=3, name='bn11'),
        Activation('relu'),

        # 28x28x512
        Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv12'),
        BatchNormalization(axis=3, name='bn12'),
        Activation('relu'),
        # 28x28x512
        MaxPooling2D((2, 2), strides=(2, 2), name='max_pool12'),
        # 14x14x512

        ############# DECODER ###############
        # 14x14x512
        Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv13'),
        BatchNormalization(axis=3, name='bn13'),
        Activation('relu'),
        Dropout(0.3),

        # 14x14x512
        UpSampling2D(size=(2, 2), name='upsampling14'),
        Conv2D(512, (5, 5), strides=(1, 1), padding='same', name='conv14'),
        BatchNormalization(axis=3, name='bn14'),
        Activation('relu'),
        Dropout(0.3),

        # 28x28x512
        UpSampling2D(size=(2, 2), name='upsampling15'),
        Conv2D(256, (5, 5), strides=(1, 1), padding='same', name='conv15'),
        BatchNormalization(axis=3, name='bn15'),
        Activation('relu'),
        Dropout(0.3),

        # 56x56x256
        UpSampling2D(size=(2, 2), name='upsampling16'),
        Conv2D(128, (5, 5), strides=(1, 1), padding='same', name='conv16'),
        BatchNormalization(axis=3, name='bn16'),
        Activation('relu'),
        Dropout(0.3),

        # 112x112x128
        UpSampling2D(size=(2, 2), name='upsampling17'),
        Conv2D(64, (5, 5), strides=(1, 1), padding='same', name='conv17'),
        BatchNormalization(axis=3, name='bn17'),
        Activation('relu'),
        Dropout(0.3),

        # 224x224x64
        UpSampling2D(size=(2, 2), name='upsampling18'),
        Conv2D(32, (5, 5), strides=(1, 1), padding='same', name='conv18'),
        BatchNormalization(axis=3, name='bn18'),
        Activation('relu'),
        Dropout(0.3),

        # 448x448x32
        Conv2D(1, (5, 5), strides=(1, 1), padding='same', name='conv19'),
        BatchNormalization(axis=3, name='bn19'),
        Activation('sigmoid'),
    ],

        name='model_yang')

    # 448x448x1

    return model

