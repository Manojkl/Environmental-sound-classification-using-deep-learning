import librosa
import numpy as np

import os
import logging
from functools import partial

import tensorflow as tf
from tensorflow.keras.models import Model
import tensorflow.keras.layers as tfkl
import tensorflow.keras.backend as K
from keras.layers import GlobalAveragePooling2D, Dense, Flatten

from sklearn.preprocessing import LabelBinarizer
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer, Activation
from keras.models import Sequential
from keras import optimizers
from postprocess import Postprocess
import params
import requests
import pandas as pd
import pickle
#Import wandb libraries
import wandb
wandb.init(project="vgg_training_04_1&2_frozen_layer")
from wandb.keras import WandbCallback

def telegram_bot_sendtext(bot_message):
    
    bot_token = '1153335989:AAE4v1w9FD_vCUaG2qcq-WmuPwh_MBYWWho'
    bot_chatID = '675791133'
    send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message

    response = requests.get(send_text)

    return response.json()


def VGGish(pump=None,
           input_shape=None,
           include_top=False,
           pooling='avg',
           weights='audioset',
           name='vggish',
           compress=False):
    '''A Keras implementation of the VGGish architecture.

    Arguments:
        pump (pumpp.Pump): The model pump object.

        input_shape (tuple): the model input shape. If ``include_top``,
            ``input_shape`` will be set to ``(params.NUM_FRAMES, params.NUM_BANDS, 1)``,
            otherwise it will be ``(None, None, 1)`` to accomodate variable sized
            inputs.

        include_top (bool): whether to include the fully connected layers. Default is False.

        pooling (str): what type of global pooling should be applied if no top? Default is 'avg'

        weights (str, None): the weights to use (see WEIGHTS_PATHS). Currently, there is
            only 'audioset'. Can also be a path to a keras weights file.

        name (str): the name for the model.

    Returns:
        A Keras model instance.
    '''

    with tf.name_scope(name):
        if input_shape:
            pass

        elif include_top:
            input_shape = params.NUM_FRAMES, params.NUM_BANDS, 1

        elif pump:
            # print(type(pump))
            inputs = pump.layers('tf.keras')[params.PUMP_INPUT]

        else:
            input_shape = None, None, 1

        # use input_shape to make input
        if input_shape:
            inputs = tfkl.Input(shape=input_shape, name='input_1')

        # setup layer params
        conv = partial(
            tfkl.Conv2D,
            kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')

        maxpool = partial(
            tfkl.MaxPooling2D, pool_size=(2, 2), strides=(2, 2), padding='same')

        # Block 1
        x = conv(64, name='conv1')(inputs)
        x = maxpool(name='pool1')(x)

        # Block 2
        x = conv(128, name='conv2')(x)
        x = maxpool(name='pool2')(x)

        # Block 3
        x = conv(256, name='conv3/conv3_1')(x)
        x = conv(256, name='conv3/conv3_2')(x)
        x = maxpool(name='pool3')(x)

        # Block 4
        x = conv(512, name='conv4/conv4_1')(x)
        x = conv(512, name='conv4/conv4_2')(x)
        x = maxpool(name='pool4')(x)

        if include_top:
            dense = partial(tfkl.Dense, activation='relu')

            # FC block
            x = tfkl.Flatten(name='flatten_')(x)
            x = dense(4096, name='fc1/fc1_1')(x)
            x = dense(4096, name='fc1/fc1_2')(x)
            x = dense(params.EMBEDDING_SIZE, name='fc2')(x)

            if compress:
                x = Postprocess()(x)
        else:
            globalpool = (
                tfkl.GlobalAveragePooling2D() if pooling == 'avg' else
                tfkl.GlobalMaxPooling2D() if pooling == 'max' else None)

            if globalpool:
                x = globalpool(x)

        # Create model
        model = Model(inputs, x, name='model')
        # print(model.summary())
        model.load_weights('/home/mkolpe2s/rand/CNN/VGGish/vggish_keras/vggish_audioset_weights_without_fc2.h5')
        # load_vggish_weights(model, weights, strict=bool(weights))
        # print("Okay")
    return model

def batch_generator(ids, batch_size, x_data=None, y_data=None):
    batch=[]
    while True:
            np.random.shuffle(ids) 
            for i in ids:
                batch.append(i)
                if len(batch)==batch_size:
                    yield load_data(batch, x_data, y_data)
                    batch=[]

def load_data(ids, x_data, y_data):
    x_array = np.zeros((len(ids), 96,64,1))
    y_array = []
    labels = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling", "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]
    encoder = LabelBinarizer()
    transfomed_label = encoder.fit_transform(labels)

    for i,j in enumerate(ids):
        x_array[i,:,:,:] = x_data[j,:,:,:]
        class_value = y_data[j]
        one_hot = labels.index(class_value)
        y_array.append(transfomed_label[one_hot])
    assert len(x_array)==len(y_array)
    # print("length:",len(y_array))
    # print(y_array)
    return x_array, np.array(y_array)

X_train = np.load("/scratch/mkolpe2s/Train_log_mel_value.npy")
Y_train = np.load("/scratch/mkolpe2s/Train_class_value.npy")
X_validate = np.load("/scratch/mkolpe2s/Test_log_mel_value.npy")
Y_validate = np.load("/scratch/mkolpe2s/Test_class_value.npy")

labels = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling", "enginge_idling", "gun_shot", "jackhammer", "siren", "street_music"]
ids_train = np.arange(len(X_train)) 
ids_test = np.arange(len(X_validate))


vgg = VGGish()

vgg.trainable = False
for layer in vgg.layers:
    layer.trainable = False

pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in vgg.layers]
# pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable']) 

vgg.trainable = True

set_trainable = False
for layer in vgg.layers:
    if layer.name in ['conv4/conv4_1', 'conv4/conv4_2', 'conv3/conv3_2', 'conv3/conv3_1']:
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
        
layers = [(layer, layer.name, layer.trainable) for layer in vgg.layers]
# pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])

input_shape = (96, 64, 1)
model = Sequential()
model.add(vgg)
model.add(Dense(512,input_dim=input_shape))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(optimizers.RMSprop(lr=0.0005, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])
# model.compile(loss="categorical_crossentropy",
#               optimizer="adam",
#               metrics=['accuracy'])
model.summary()



X_all = np.concatenate((X_train,X_validate))
Y_all = np.concatenate((Y_train, Y_validate))

indices = np.arange(X_all.shape[0])
np.random.shuffle(indices)

X_all = X_all[indices]
Y_all = Y_all[indices]

X_all_train = X_all[0:579440,:,:,:]
Y_all_train = Y_all[0:579440]
X_all_validate = X_all[579441:589440,:,:,:]
Y_all_validate = Y_all[579441:589440]
X_all_test = X_all[589441:,:,:,:]
Y_all_test = Y_all[589441:]

ids_all_train = np.arange(len(X_all_train))
ids_all_validate = np.arange(len(X_all_validate))
ids_all_test = np.arange(len(X_all_test))

train_generator = batch_generator(ids_all_train, 64, x_data=X_all_train, y_data=Y_all_train)
valid_generator = batch_generator(ids_all_validate, 64, x_data=X_all_validate, y_data=Y_all_validate)

STEP_SIZE_TRAIN = len(X_all)//64
STEP_SIZE_VALID = len(X_all)//64


model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    callbacks=[WandbCallback(labels=labels)],
                    epochs=60)
# you can pass the full path to the Keras model API
model.save(os.path.join("/home/mkolpe2s/rand/CNN/VGGish/vggish_keras/", "vgg_training_05.h5"))