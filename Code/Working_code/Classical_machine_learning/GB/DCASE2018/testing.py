import os
os.environ[ 'NUMBA_CACHE_DIR' ] = '/tmp/'

import numpy as np
import pandas as pd
import os
import librosa
import librosa.display
import soundfile as sf # librosa fails when reading files on Kaggle.

import pickle
# now you can save it to a file
import matplotlib.pyplot as plt
# import matplotli

from sklearn.metrics import confusion_matrix
import seaborn as sns

import scikitplot as skplt
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, confusion_matrix
from sklearn import naive_bayes
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier as gbc
from xgboost import XGBClassifier
import requests
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, scale
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.svm import SVC
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import plot_confusion_matrix
plt.switch_backend('agg')


x_test = np.load("/home/mkolpe2s/rand/Classic_ML/Proper_method/Data/DCASE2018/x_test.npy",allow_pickle=True)
y_test = np.load("/home/mkolpe2s/rand/Classic_ML/Proper_method/Data/DCASE2018/y_test.npy",allow_pickle=True)
scaler = StandardScaler()

x_test_scaled = scaler.fit_transform(x_test)


with open('/home/mkolpe2s/rand/Classic_ML/Proper_method/GB/DCASE2018/GBC_DCASE2018_optimized_parameter.pkl', 'rb') as f:
    model = pickle.load(f)

prediction = model.predict(x_test_scaled)

["absence", "cooking", "dishwashing", "eating", "other", "social_activity", "vacuum_cleaner", "watching_tv", "working"]

target_names = ["AB", "CO", "DI", "EA", "OT", "SA", "VC", "WT", "WO"]

cm = confusion_matrix(y_test, prediction)
# Normalise
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10,10))
sns.set(font_scale=1.2)
sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names)
plt.ylabel('Actual',fontsize=15)
plt.xlabel('Predicted',fontsize=15)
plt.show(block=False)

# skplt.metrics.plot_confusion_matrix(y_test, prediction, figsize=(20,20))
plt.savefig('/home/mkolpe2s/rand/Classic_ML/Proper_method/GB/DCASE2018/GBC_DCASE2018_optimized_parameter_new_sns.pdf')

skplt.metrics.plot_confusion_matrix(y_test, prediction, figsize=(20,20))
plt.savefig('/home/mkolpe2s/rand/Classic_ML/Proper_method/GB/DCASE2018/GBC_DCASE2018_optimized_parameter_new_check.pdf')