##### import libraries #####
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Dropout, Reshape, Flatten, Permute
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Conv1D, Conv2D, ZeroPadding2D
from keras.layers.pooling import AveragePooling2D, MaxPooling2D, MaxPooling1D, GlobalAveragePooling2D
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam, SGD
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import merge, Input, Bidirectional, GRU
from keras import layers
# from keras_exp.multigpu import get_available_gpus
# from keras_exp.multigpu import make_parallel

# from natsort import natsorted

# for generator
import threading
import time
import numpy as np
import h5py
import glob
import os

if not os.path.exists('weight_crnn/'):       
    os.makedirs('weight_crnn/')

num_class = 4
n_feature = 128
timestep = 43
jump = timestep
lrate = 0.001
patience = 30

# Main start!
train_path = 'h5/'


n_frame = 430
rnn_bin=n_frame/timestep

n_batch = 32 # number of input files
fs = 22050
n_epoch=50

# sorting natural order
(loc, _, fnames) = os.walk(train_path).next()
# fnames = natsorted(fnames)
fnames = np.array(fnames)

print "Train set loading!"
X_train=np.zeros((len(fnames),rnn_bin,timestep,n_feature))
y_train=np.zeros((len(fnames),rnn_bin,timestep,num_class))

for i, file_train in enumerate(fnames) :
    with h5py.File(train_path+file_train,'r') as f:
        X=np.array(f['x']).astype(np.float)
        y=np.array(f['y'])
    X_train[i]=X[:n_frame].reshape((rnn_bin,timestep,n_feature))
    y_train[i]=y[:n_frame].reshape((rnn_bin,timestep,num_class))

    if i % 1000 == 0:
        print str(i)+'th done'

X_train=X_train.reshape((len(fnames),rnn_bin,timestep,n_feature,1))
X_train=X_train.reshape((len(fnames)*rnn_bin,timestep,n_feature,1))
y_train=y_train.reshape((len(fnames)*rnn_bin,timestep,num_class))



# build model...
# print 'build model...'
inputs = layers.Input(shape=(timestep, n_feature,1))

conv = TimeDistributed(Conv1D(32, 4, padding='same'))(inputs)
conv = BatchNormalization()(conv)
conv = TimeDistributed(Activation('relu'))(conv)
conv = TimeDistributed(MaxPooling1D(pool_size=4, padding='same'))(conv)
conv = TimeDistributed(Conv1D(64, 4, padding='same'))(conv)
conv = BatchNormalization()(conv)
conv = TimeDistributed(Activation('relu'))(conv)
conv = TimeDistributed(MaxPooling1D(pool_size=4, padding='same'))(conv)
conv = TimeDistributed(Conv1D(128, 4, padding='same'))(conv)
conv = BatchNormalization()(conv)
conv = TimeDistributed(Activation('relu'))(conv)
conv = TimeDistributed(MaxPooling1D(pool_size=4, padding='same'))(conv)
conv = TimeDistributed(Conv1D(128, 2, padding='same'))(conv)
conv = BatchNormalization()(conv)
conv = TimeDistributed(Activation('relu'))(conv)
conv = TimeDistributed(MaxPooling1D(pool_size=2, padding='same'))(conv)
# conv = Dropout(0.3)(conv)

conv = Reshape((timestep,128))(conv)

lstm1 = Bidirectional(GRU(128, return_sequences=True))(conv)
# lstm1 = Dropout(0.3)(lstm1)

lstm2 = Bidirectional(GRU(128, return_sequences=True))(lstm1)
# lstm2 = Dropout(0.3)(lstm2)

fc = TimeDistributed(Dense(256, activation='linear'))(lstm2)
fc = BatchNormalization()(fc)
fc = Activation('relu')(fc)

outputs = TimeDistributed(Dense(num_class, activation='sigmoid'))(fc)

model = Model(inputs=[inputs], outputs=[outputs])
adam = Adam(lr=lrate)

model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

check=ModelCheckpoint('weight_crnn/weights.{epoch:02d}-{loss:.4f}.h5', verbose=1, save_best_only=True, monitor='loss')

print 'fit model...'
hist=model.fit(X_train, y_train, batch_size=n_batch, epochs=n_epoch, callbacks=[check], verbose=1)