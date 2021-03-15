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
    
    bot_token = ''
    bot_chatID = ''
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

#KNN Default parameter.............................................................................
print("------------------------------------------------")
print("                                                ")
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train_new)
x_validate_scaled = scaler.transform(x_validate)

# Training...........................................
print("Training...........................")
knn = KNeighborsClassifier()
knn.fit(x_train_scaled, y_train_new)

with open('/home/mkolpe2s/rand/Classic_ML/Proper_method/KNN/US8K/KNN_US8K_default_parameter.pkl', 'wb') as f:
    pickle.dump(knn, f)
print("Train score:", knn.score(x_train_scaled,y_train_new))

#Validation..............................
print("Validation...........................")
print("Validation score:", knn.score(x_validate_scaled, y_validate))

#Testing.................................
print("Testing...........................")
scaler = StandardScaler()

x_test_scaled = scaler.fit_transform(x_test)

print("Test score:", knn.score(x_test_scaled, y_test))

# Classification report...................
print("Classification report KNN default parameter.....................")

prediction = knn.predict(x_test_scaled)

report = classification_report(prediction, y_test)
report_dic = classification_report(prediction, y_test, output_dict=True)
df = pd.DataFrame(report_dic).transpose()
df.to_csv('/home/mkolpe2s/rand/Classic_ML/Proper_method/KNN/US8K/classification_report_KNN_default_parameter_US8K.csv')

fig, ax = plt.subplots(figsize=(15, 15))
plot_confusion_matrix(knn, x_test_scaled, y_test, ax=ax)  # doctest: +SKIP
plt.savefig('/home/mkolpe2s/rand/Classic_ML/Proper_method/KNN/US8K/KNN_US8K_default_parameter.pdf')
print(report)

#KNN optimized parameter..............................................................

print("------------------------------------------------")
print("                                                ")
scaler = StandardScaler()

x_train_scaled_all = scaler.fit_transform(x_train)

# Optimizing...............................................
print("Optimizing...........................................")
grid_params = {
    'n_neighbors': [1, 2, 3, 4, 5,6,7,8,9,10],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
kfold = KFold(n_splits=10, random_state=23, shuffle=True)
model = GridSearchCV(KNeighborsClassifier(), grid_params, verbose=1,  cv=8, n_jobs=-1)
model.fit(x_train_scaled_all, y_train)

with open('/home/mkolpe2s/rand/Classic_ML/Proper_method/KNN/US8K/KNN_US8K_optimized_parameter.pkl', 'wb') as f:
    pickle.dump(model, f)

#Best score and parameter...................................

print("KNN best score=",model.best_score_)
print("KNN best estimator=",model.best_estimator_)

#Testing.................................
print("Testing...........................")
scaler = StandardScaler()

x_test_scaled = scaler.fit_transform(x_test)

print("Test score:", model.score(x_test_scaled, y_test))


# Classification report...................
print("Classification report KNN optimized parameter.....................")

prediction = model.predict(x_test_scaled)

report = classification_report(prediction, y_test)
report_dic = classification_report(prediction, y_test, output_dict=True)
df = pd.DataFrame(report_dic).transpose()
df.to_csv('/home/mkolpe2s/rand/Classic_ML/Proper_method/KNN/US8K/classification_report_KNN_optimized_parameter_US8K.csv')

fig, ax = plt.subplots(figsize=(15, 15))
plot_confusion_matrix(knn, x_test_scaled, y_test, ax=ax)  # doctest: +SKIP
plt.savefig('/home/mkolpe2s/rand/Classic_ML/Proper_method/KNN/US8K/KNN_US8K_optimized_parameter.pdf')
print(report)

telegram_bot_sendtext(str("completed KNN for US8K"))