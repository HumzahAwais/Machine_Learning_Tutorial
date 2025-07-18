### Decision Trees and Random Forest Classification ###

# Import
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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
clf.fit(x_train, y_train)

# K Nearest Neighbor Classifier
clf2 = KNeighborsClassifier(n_neighbors=3)
clf2.fit(x_train, y_train)

# Decision Tree Classifier
clf3 = DecisionTreeClassifier()
clf3.fit(x_train, y_train)

# Random Forest Classifier
# > Basically Multiple Decision Trees since the order of features has an effect
clf4 = RandomForestClassifier()
clf4.fit(x_train, y_train)

# See how well the model predicts
print(f'SVC: {clf.score(x_test, y_test)}')
print(f'KNN: {clf2.score(x_test, y_test)}')
print(f'DTC: {clf3.score(x_test, y_test)}')
print(f'RFC: {clf4.score(x_test, y_test)}')