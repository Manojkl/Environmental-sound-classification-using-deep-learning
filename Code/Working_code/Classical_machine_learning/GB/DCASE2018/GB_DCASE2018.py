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

def telegram_bot_sendtext(bot_message):
    
    bot_token = '1153335989:AAE4v1w9FD_vCUaG2qcq-WmuPwh_MBYWWho'
    bot_chatID = '675791133'
    send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message

    response = requests.get(send_text)

    return response.json()


x_train = np.load("/home/mkolpe2s/rand/Classic_ML/Proper_method/Data/DCASE2018/x_train.npy",allow_pickle=True)
x_test = np.load("/home/mkolpe2s/rand/Classic_ML/Proper_method/Data/DCASE2018/x_test.npy",allow_pickle=True)
y_train = np.load("/home/mkolpe2s/rand/Classic_ML/Proper_method/Data/DCASE2018/y_train.npy",allow_pickle=True)
y_test = np.load("/home/mkolpe2s/rand/Classic_ML/Proper_method/Data/DCASE2018/y_test.npy",allow_pickle=True)

x_train_new = np.load("/home/mkolpe2s/rand/Classic_ML/Proper_method/Data/DCASE2018/x_train_new.npy",allow_pickle=True)
x_validate = np.load("/home/mkolpe2s/rand/Classic_ML/Proper_method/Data/DCASE2018/x_validate.npy",allow_pickle=True)
y_train_new = np.load("/home/mkolpe2s/rand/Classic_ML/Proper_method/Data/DCASE2018/y_train_new.npy",allow_pickle=True)
y_validate = np.load("/home/mkolpe2s/rand/Classic_ML/Proper_method/Data/DCASE2018/y_validate.npy",allow_pickle=True)

#KNN Default parameter.............................................................................
print("------------------------------------------------")
print("                                                ")
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train_new)
x_validate_scaled = scaler.transform(x_validate)

# Gradient boosting   ------------------------------------------------------------------
print(" Gradient Boosting ... ")
print("------------------------------------------------")
print("                                                ")

# Training...........................................
print("Training...........................")

classifier = gbc()
gbc_model = classifier.fit(x_train_scaled, y_train_new)

with open('/home/mkolpe2s/rand/Classic_ML/Proper_method/GB/DCASE2018/gbc_DCASE2018_default_parameter.pkl', 'wb') as f:
    pickle.dump(gbc_model, f)

print("Train score:", gbc_model.score(x_train_scaled,y_train_new))

#Validation..............................
print("Validation...........................")
print("Validation score:", gbc_model.score(x_validate_scaled, y_validate))

#Testing.................................
print("Testing...........................")
scaler = StandardScaler()

x_test_scaled = scaler.fit_transform(x_test)

print("Test score:", gbc_model.score(x_test_scaled, y_test))

# Classification report...................
print("Classification report GBC default parameter.....................")

prediction = gbc_model.predict(x_test_scaled)

report = classification_report(prediction, y_test)
report_dic = classification_report(prediction, y_test, output_dict=True)
df = pd.DataFrame(report_dic).transpose()
df.to_csv('/home/mkolpe2s/rand/Classic_ML/Proper_method/GB/DCASE2018/classification_report_GBC_default_parameter_DCASE2018.csv')

fig, ax = plt.subplots(figsize=(15, 15))
plot_confusion_matrix(gbc_model, x_test_scaled, y_test, ax=ax)  # doctest: +SKIP
plt.savefig('/home/mkolpe2s/rand/Classic_ML/Proper_method/GB/DCASE2018/GBC_DCASE2018_default_parameter.pdf')
print(report)

#GBC optimized parameter..............................................................
print("------------------------------------------------")
print("                                                ")
scaler = StandardScaler()

x_train_scaled_all = scaler.fit_transform(x_train)

#creating Scoring parameter: 
# scoring = {'accuracy': make_scorer(accuracy_score,average = 'weighted'),
#            'precision': make_scorer(precision_score, average = 'weighted'),'recall':make_scorer(recall_score,average = 'weighted')}

# A sample parameter

parameters = {
    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    "min_samples_split": range(200,1001,200),
    "min_samples_leaf": range(30,71,10),
    "max_depth":range(5,16,2)
    }

#passing the scoring function in the GridSearchCV
model = GridSearchCV(gbc(), param_grid = parameters,cv=8, n_jobs=-1)
model.fit(x_train_scaled_all, y_train)

with open('/home/mkolpe2s/rand/Classic_ML/Proper_method/GB/DCASE2018/GBC_DCASE2018_optimized_parameter.pkl', 'wb') as f:
    pickle.dump(model, f)

#Best score and parameter...................................

print("GBC best score=",model.best_score_)
print("GBC best estimator=",model.best_estimator_)

#Testing.................................
print("Testing...........................")
scaler = StandardScaler()

x_test_scaled = scaler.fit_transform(x_test)

print("Test score:", model.score(x_test_scaled, y_test))

# Classification report...................
print("Classification report GBC optimized parameter.....................")

prediction = model.predict(x_test_scaled)

report = classification_report(prediction, y_test)
report_dic = classification_report(prediction, y_test, output_dict=True)
df = pd.DataFrame(report_dic).transpose()
df.to_csv('/home/mkolpe2s/rand/Classic_ML/Proper_method/GB/DCASE2018/classification_report_GBC_optimized_parameter_DCASE2018.csv')

# fig, ax = plt.subplots(figsize=(15, 15))
# plot_confusion_matrix(model, x_test_scaled, y_test, ax=ax)  # doctest: +SKIP

# test_labels = y_test

skplt.metrics.plot_confusion_matrix(y_test, prediction, figsize=(20,20))
plt.savefig('/home/mkolpe2s/rand/Classic_ML/Proper_method/GB/DCASE2018/GBC_DCASE2018_optimized_parameter_new.pdf')
print(report)

