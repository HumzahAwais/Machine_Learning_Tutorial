### Support Vector Machines ###

# Import
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Data
data = load_breast_cancer()
X = data.data
Y = data.target

# Splitting data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Support Vector Machine Classifier
# > Kernel function is linear: fastest and low resources, adds another dimension
# > C=3 means soft margin of 3 so allow 3 errors
clf = SVC(kernel='linear', C=3)

# Train model
clf.fit(x_train, y_train)

# K Nearest Neighbor Classifier
clf2 = KNeighborsClassifier(n_neighbors=3)

# Also train this model
clf2.fit(x_train, y_train)

# See how well the model predicts
print(f'SVC: {clf.score(x_test, y_test)}')
print(f'KNN: {clf2.score(x_test, y_test)}')