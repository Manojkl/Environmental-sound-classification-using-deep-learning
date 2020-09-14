from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.DataFrame({
    'beta':np.random.beta(5,1,1000)*60, #beta
    'exponential':np.random.exponential(10,1000), # exponential
    'normal_p':np.random.normal(10,2,1000) #normal
})

print(df.head())


mm_scalar = preprocessing.MinMaxScaler()
X_train_minmax = mm_scalar.fit_transform(df)
sns.kdeplot
print(X_train_minmax)



