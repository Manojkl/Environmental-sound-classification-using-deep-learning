from sklearn.model_selection import train_test_split
import numpy as np

X, y = np.arange(10).reshape((5,2)), range(5)

print(X)
print(list(y))

X_train, y_train, X_test, y_test = train_test_split(X, y, test_size = 0.33)

print(X_train)
print(y_train)
print(X_test)
print(y_test)