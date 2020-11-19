"""VGGish model for Keras. A VGG-like model for audio classification

# Reference

- [CNN Architectures for Large-Scale Audio Classification](ICASSP 2017)

"""

import os
import logging
from functools import partial

import tensorflow as tf
from tensorflow.keras.models import Model
import tensorflow.keras.layers as tfkl
import tensorflow.keras.backend as K
from keras.layers import GlobalAveragePooling2D

from postprocess import Postprocess
import params
import requests
import pandas as pd

log = logging.getLogger(__name__)

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
    # include_top=True
    msg = "Status of include top "+str(include_top)
    telegram_bot_sendtext(msg)

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
            msg = "Entered include top"
            telegram_bot_sendtext(msg)
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
                msg = "Entered global pool"
                telegram_bot_sendtext(msg)
                x = globalpool(x)
                msg = str(x.shape)
                telegram_bot_sendtext(msg)

        # Create model
        model = Model(inputs, x, name='model')
        # print(model.summary())
        model.load_weights('/home/mkolpe2s/rand/CNN/VGGish/vggish_keras/vggish_audioset_weights_without_fc2.h5')
        # load_vggish_weights(model, weights, strict=bool(weights))
        # print("Okay")
    return model


def load_vggish_weights(model, weights, strict=False):
    # lookup weights location
    if weights in params.WEIGHTS_PATHS:
        w_name, weights = weights, params.WEIGHTS_PATHS[weights]
        print("entered")
        if not os.path.isfile(weights):
            log.warning(f'"{weights}" weights have not been downloaded. Downloading now...')
            # from .download_helpers import download_weights
            # download_weights.download(w_name)

    print(weights)
    # load weights
    if weights:
        model.load_weights(weights, by_name=True)
    elif strict:
        raise RuntimeError('No weights could be found for weights={}'.format(weights))
    return model




# sound_model = VGGish(include_top=False)

# x = sound_model.get_layer(name="conv4/conv4_2").output
# output_layer = GlobalAveragePooling2D()(x)
# sound_extractor = Model(inputs=sound_model.input, outputs=output_layer)

# print ("loading training data...")
# # training_file = '/mount/hudi/moe/soundnet/train.txt'
# msg = "Started training data embeddings"
# telegram_bot_sendtext(msg)
# training_file = '/home/mkolpe2s/rand/data/Train_data/Train/'
# df = pd.read_csv('/home/mkolpe2s/rand/data/Train_data/train.csv')
# training_data, training_label = loading_data(training_file,df,sound_extractor)
# msg = "Finished training data embeddings"
# telegram_bot_sendtext(msg)