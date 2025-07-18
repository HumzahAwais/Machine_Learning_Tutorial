### K-Nearest Neighbors Classification ###

# Imports
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Data
data = load_breast_cancer()

# Viewing data
print(data.feature_names)
print(data.target_names)

# Splitting data into testing data and training data
x_train, x_test, y_train, y_test = train_test_split(np.array(data.data), np.array(data.target), test_size=0.2)

# Don't use the same value for n as the number of classes
# If you do then you can have a tie which isn't good :(
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(x_train, y_train)

# See how well the model predicts
print(clf.score(x_test, y_test))