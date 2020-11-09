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

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train_new)
x_validate_scaled = scaler.transform(x_validate)


#SVM default parameter.............................................

print("SVM default parameter..................................................")
# clf = SVC(kernel='linear') 

clf = SVC()
clf.fit(x_train_scaled, y_train_new)

with open('/home/mkolpe2s/rand/Classic_ML/Proper_method/SVM/US8K/US8K_SVM_default_parameter.pkl', 'wb') as f:
    pickle.dump(clf, f)

print("Train score for default:", clf.score(x_train_scaled,y_train_new))

#Validation..............................
print("Validation...........................")
print("Validation score:", clf.score(x_validate_scaled, y_validate))


#Testing.................................
print("Testing...........................")
scaler = StandardScaler()

x_test_scaled = scaler.fit_transform(x_test)

print("Test score:", clf.score(x_test_scaled, y_test))

spectrogram = "Cmmputed default svm of with SVM in cluster ."
telegram_bot_sendtext(spectrogram)

# Classification report...................
print("Classification report SVM default parameter.....................")

prediction = clf.predict(x_test_scaled)

report = classification_report(prediction, y_test)
report_dic = classification_report(prediction, y_test, output_dict=True)
df = pd.DataFrame(report_dic).transpose()
df.to_csv('/home/mkolpe2s/rand/Classic_ML/Proper_method/SVM/US8K/classification_report_SVM_default_parameter_US8K.csv')

fig, ax = plt.subplots(figsize=(15, 15))
plot_confusion_matrix(clf, x_test_scaled, y_test, ax=ax)  # doctest: +SKIP
plt.savefig('/home/mkolpe2s/rand/Classic_ML/Proper_method/SVM/US8K/SVM_US8K_default_parameter.pdf')
print(report)

#SVM optimized.................................................................
print("SVM optimized parameter..................................................")

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 0.01, 0.1, 1],
                     'C': [0.001, 0.01, 0.1, 1, 10]},
                    {'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1, 10]}]

scores = ['precision', 'recall']

scaler = StandardScaler()

x_train_scaled_all = scaler.fit_transform(x_train)

for score in scores:

    print("# Tuning hyper-parameters for %s" % score)
    # print()

    clf = GridSearchCV(
        SVC(), tuned_parameters, scoring='%s_macro' % score
    )
    clf.fit(x_train_scaled_all, y_train)
    
    with open('/home/mkolpe2s/rand/Classic_ML/Proper_method/SVM/US8K/US8K_SVM_optimized_parameter_'+str(score)+'.pkl', 'wb') as f:
        pickle.dump(clf, f)

    print("Train score for optimized:", clf.score(x_train_scaled_all,y_train))

    #Testing.................................
    print("Testing...........................")
    scaler = StandardScaler()

    x_test_scaled = scaler.fit_transform(x_test)

    print("Test score:", clf.score(x_test_scaled, y_test))

    spectrogram = "Cmmputed optimized "+str(score)+" of with SVM in cluster ."
    telegram_bot_sendtext(spectrogram)

    #Best score and parameter...................................

    print("SVM best score=",clf.best_score_)
    print("SVM best estimator=",clf.best_estimator_)

    # Classification report...................
    print("Classification report SVM default parameter.....................")

    prediction = clf.predict(x_test_scaled)

    report = classification_report(prediction, y_test)
    report_dic = classification_report(prediction, y_test, output_dict=True)
    df = pd.DataFrame(report_dic).transpose()
    df.to_csv('/home/mkolpe2s/rand/Classic_ML/Proper_method/SVM/US8K/classification_report_SVM_optimized_parameter_US8K_'+str(score)+'.csv')

    fig, ax = plt.subplots(figsize=(15, 15))
    plot_confusion_matrix(clf, x_test_scaled, y_test, ax=ax)  # doctest: +SKIP
    plt.savefig('/home/mkolpe2s/rand/Classic_ML/Proper_method/SVM/US8K/SVM_US8K_optimized_parameter.pdf')
    print(report)

telegram_bot_sendtext(str("completed SVM for US8K"))