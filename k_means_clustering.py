### K-Means Clustering ###

# Imports
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits

# Data
digits = load_digits()
data = scale(digits.data)

# K-Means Clustering
# > we have 10 digits (0 to 9) so we would look for 10 clusters
# > init function is random just to start at a random place
# > n_init is just how many times do we call the init function
model = KMeans(n_cluster=10, init='random', n_init=10)
model.fit(data)

# Can add in pixels and predict which cluster it belongs in
# Won't tell the number though
# model.predict([...])