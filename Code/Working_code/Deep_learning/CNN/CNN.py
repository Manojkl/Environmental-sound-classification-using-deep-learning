import requests
# from memory_profiler import memory_usage
# os - operating system allow us to This module provides a portable way of using operating system dependent functionality
# It helps to have greater control over the interaction with file system
import os 
import pandas as pd
# In Python, the glob module is used to retrieve files/pathnames matching a specified pattern. The pattern rules of glob follow standard Unix path expansion rules. 
# Example:
# # Using '?' pattern 
# print('\nNamed with wildcard ?:') 
# for name in glob.glob('/home/geeks/Desktop/gfg/data?.txt'): 
#     print(name) 
from glob import glob
import numpy as np

from sklearn.preprocessing import scale
from keras import layers
from keras import models
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
import keras.backend as K
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib import figure
from keras.preprocessing.image import ImageDataGenerator
# garbage collector
import gc
from path import Path
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
from sklearn.metrics import plot_confusion_matrix

generate_spectrogram = True
plt.switch_backend('agg')

def telegram_bot_sendtext(bot_message):
    
    bot_token = '1153335989:AAE4v1w9FD_vCUaG2qcq-WmuPwh_MBYWWho'
    bot_chatID = '675791133'
    send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message

    response = requests.get(send_text)

    return response.json()

def append_ext(fn):
    return fn+".jpg"

traindf = pd.read_csv("/home/mkolpe2s/rand/data/Train_data/train.csv", dtype=str)
testdf = pd.read_csv("/home/mkolpe2s/rand/data/Test_data/test.csv", dtype=str)
traindf["ID"] = traindf["ID"].apply(append_ext)
testdf["ID"] = testdf["ID"].apply(append_ext)

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.25)

train_generator = datagen.flow_from_dataframe(
    dataframe=traindf,
    directory="/home/mkolpe2s/rand/mfcc_spectrogram/train/",
    x_col="ID",
    y_col="Class",
    subset="training",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(64,64))

valid_generator=datagen.flow_from_dataframe(
    dataframe=traindf,
    directory="/home/mkolpe2s/rand/mfcc_spectrogram/train/",
    x_col="ID",
    y_col="Class",
    subset="validation",
    batch_size=2,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(64,64))

def cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',input_shape=(64,64,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizers.RMSprop(lr=0.0005, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])
    model.summary()

    return model

model = cnn_model()

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
#STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=150) 

model.evaluate_generator(generator=valid_generator, steps=STEP_SIZE_VALID)

test_datagen=ImageDataGenerator(rescale=1./255.)

test_generator=test_datagen.flow_from_dataframe(
    dataframe=testdf,
    directory="/home/mkolpe2s/rand/mfcc_spectrogram/test/",
    x_col="ID",
    y_col=None,
    batch_size=32,
    seed=42,
    shuffle=False,
    class_mode=None,
    target_size=(64,64))

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

test_generator.reset()
pred=model.predict_generator(test_generator, steps=STEP_SIZE_TEST, verbose=1)
predicted_class_indices=np.argmax(pred,axis=1)

#Fetch labels from train gen for testing
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
labels=["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling", "enginge_idling", "gun_shot", "jackhammer", "siren","street_music"]
fig, ax = plt.subplots(figsize=(17, 17))
plot_confusion_matrix(model,test_generator, predicted_class_indices, ax=ax)  # doctest: +SKIP
# confusion_matrix(validation_generator.classes, y_pred)
plt.savefig('/home/mkolpe2s/rand/mfcc_spectrogram/foo.pdf')
plt.show() 
print(predictions[0:6])

