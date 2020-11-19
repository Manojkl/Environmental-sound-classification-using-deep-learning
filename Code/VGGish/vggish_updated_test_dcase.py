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

from postprocess import Postprocess
import params
import requests
import pandas as pd
import pickle

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

def get_embeddings_of_all(files, data_type):
    # train_path = files
    # train_file_names = os.listdir(train_path)
    # train_file_names.sort(key=lambda x: int(x.partition('.')[0]))
    # samples = np.zeros((_,96,64,1))
    labels = []
    total_ommitt = 0
    seg_num = 30
    testing = np.zeros((73021*seg_num,96,64,1))
    # testing = np.zeros((2851*3,96,64,1))
    df_test = pd.read_csv("/scratch/mkolpe2s/DCASE2018-task5-dev_metadata/meta.csv")
    sound_extractor = VGGish()
    # k = 0
    seg_len = 0.96
    track = 0

    for subdir, dirs, files in os.walk(files):
        for value, file in enumerate(files):
            filename,name = os.path.join(subdir, file),file.split('/')[-1].split('.')[0]
            y, sr = librosa.load(filename)
            k = 0
            length = int(sr*seg_len)
            if len(y) > length:
                range_high = len(y) - length
                random_start = np.random.randint(range_high, size=seg_num)
            for j in range(seg_num):
                data = y[random_start[j]:random_start[j]+length]
                mel = librosa.feature.melspectrogram(y=data,sr=sr, hop_length = 221, n_fft = 552, n_mels = 64)
                mel = librosa.power_to_db(mel)
                class_value = "audio/"+name+".wav"
                class_value = df_test.loc[df_test['filename'] == class_value, 'class'].iloc[0]
                labels.append(class_value)
                mel =np.transpose(mel)
                mel = np.expand_dims(mel, axis=2)
                testing[track,:,:,:] = mel
                track+=1

            if track%500 == 0:
                msg = "computed "+str(track)+" test files."+"Class value:"+str(class_value)
                telegram_bot_sendtext(msg)

    print("track: ",track)
    labels = np.array(labels)
    # testing = np.array(testing)
    print("labels",labels.shape)
    print("testing", testing.shape)
    # embeddings = sound_extractor.predict(testing)
    # print(embeddings.shape)
    # for i in embeddings:
    #     print(i)
    #     print(max(i), min(i))                
    # # msg = "Totally ommitted "+str(data_type)+": "+str(total_ommitt)+" files"
    # # telegram_bot_sendtext(msg)
    # embedded_value = np.array(embeddings)
    
    # with open(str("/home/mkolpe2s/rand/CNN/VGGish/vggish_keras/")+str(data_type)+"_embedded_value.txt", "wb") as fp:   #Pickling
    #     pickle.dump(embeddings, fp)
    # np.save(str("/home/mkolpe2s/rand/CNN/VGGish/vggish_keras/")+str(data_type)+"_embedded_value.npy",embedded_value)
    np.save(str("/scratch/mkolpe2s/")+"Dcase_dev_30_class_value.npy",labels)
    np.save(str("/scratch/mkolpe2s/")+"Dcase_dev_30_log_mel_value.npy",testing)

    return None

# get_embeddings_of_all('/home/mkolpe2s/rand/data/Train_data/Train/',data_type ='Train')
get_embeddings_of_all('/scratch/mkolpe2s/Data/',data_type =None)

# get_embeddings_of_all('/home/mkolpe2s/rand/data/Train_data/Train/',data_type ='Train')
# get_embeddings_of_all('/home/mkolpe2s/rand/data/Test_data/Test/',data_type ='Test')