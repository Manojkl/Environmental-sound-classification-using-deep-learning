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


x_train = np.load("/home/mkolpe2s/rand/Classic_ML/Proper_method/Data/US8K/x_train.npy",allow_pickle=True)
x_test = np.load("/home/mkolpe2s/rand/Classic_ML/Proper_method/Data/US8K/x_test.npy",allow_pickle=True)
y_train = np.load("/home/mkolpe2s/rand/Classic_ML/Proper_method/Data/US8K/y_train.npy",allow_pickle=True)
y_test = np.load("/home/mkolpe2s/rand/Classic_ML/Proper_method/Data/US8K/y_test.npy",allow_pickle=True)

x_train_new = np.load("/home/mkolpe2s/rand/Classic_ML/Proper_method/Data/US8K/x_train_new.npy",allow_pickle=True)
x_validate = np.load("/home/mkolpe2s/rand/Classic_ML/Proper_method/Data/US8K/x_validate.npy",allow_pickle=True)
y_train_new = np.load("/home/mkolpe2s/rand/Classic_ML/Proper_method/Data/US8K/y_train_new.npy",allow_pickle=True)
y_validate = np.load("/home/mkolpe2s/rand/Classic_ML/Proper_method/Data/US8K/y_validate.npy",allow_pickle=True)

#RF Default parameter.............................................................................
print("------------------------------------------------")
print("                                                ")
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train_new)
x_validate_scaled = scaler.transform(x_validate)

# algorithm 2 ------------------------------------------------------------------
print(" Random Forest ... ")
print("------------------------------------------------")
print("                                                ")

# Training...........................................
print("Training...........................")

classifier = RandomForestClassifier()
rf_model = classifier.fit(x_train_scaled, y_train_new)

with open('/home/mkolpe2s/rand/Classic_ML/Proper_method/RF/US8K/random_forest_US8K_default_parameter.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("Train score:", rf_model.score(x_train_scaled,y_train_new))

#Validation..............................
print("Validation...........................")
print("Validation score:", rf_model.score(x_validate_scaled, y_validate))

#Testing.................................
print("Testing...........................")
scaler = StandardScaler()

x_test_scaled = scaler.fit_transform(x_test)

print("Test score:", rf_model.score(x_test_scaled, y_test))

# Classification report...................
print("Classification report RF default parameter.....................")

prediction = rf_model.predict(x_test_scaled)

report = classification_report(prediction, y_test)
report_dic = classification_report(prediction, y_test, output_dict=True)
df = pd.DataFrame(report_dic).transpose()
df.to_csv('/home/mkolpe2s/rand/Classic_ML/Proper_method/RF/US8K/classification_report_RF_default_parameter_US8K.csv')

fig, ax = plt.subplots(figsize=(15, 15))
plot_confusion_matrix(rf_model, x_test_scaled, y_test, ax=ax)  # doctest: +SKIP
plt.savefig('/home/mkolpe2s/rand/Classic_ML/Proper_method/RF/US8K/RF_US8K_default_parameter.pdf')
print(report)

#Optimizing RF....................................................................
scaler = StandardScaler()

x_train_scaled_all = scaler.fit_transform(x_train)

print("Optimizing...........................................")
param_grid = {
    'max_depth': [5, 8, 15, 25, 30],
    'min_samples_leaf': [1, 2, 5, 10],
    'min_samples_split': [2, 5, 10, 15, 100],
    'n_estimators': [100, 300, 500, 800, 1200]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
model = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 8, n_jobs = -1, verbose = 1)

model.fit(x_train_scaled_all, y_train)

with open('/home/mkolpe2s/rand/Classic_ML/Proper_method/RF/US8K/RF_US8K_optimized_parameter.pkl', 'wb') as f:
    pickle.dump(model, f)

#Best score and parameter...................................

print("RF best score=",model.best_score_)
print("RF best estimator=",model.best_estimator_)

#Testing.................................
print("Testing...........................")
scaler = StandardScaler()

x_test_scaled = scaler.fit_transform(x_test)

print("Test score:", model.score(x_test_scaled, y_test))

# Classification report...................
print("Classification report RF=andom forest optimized parameter.....................")

prediction = model.predict(x_test_scaled)

report = classification_report(prediction, y_test)
report_dic = classification_report(prediction, y_test, output_dict=True)
df = pd.DataFrame(report_dic).transpose()
df.to_csv('/home/mkolpe2s/rand/Classic_ML/Proper_method/RF/US8K/classification_report_RF_optimized_parameter_US8K.csv')

fig, ax = plt.subplots(figsize=(15, 15))
plot_confusion_matrix(model, x_test_scaled, y_test, ax=ax)  # doctest: +SKIP
plt.savefig('/home/mkolpe2s/rand/Classic_ML/Proper_method/RF/US8K/RF_US8K_optimized_parameter.pdf')
print(report)


telegram_bot_sendtext(str("RF US8K completed"))