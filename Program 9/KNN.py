# Write a program to implement the k-Nearest Neighbour algorithm to classify the iris data set. Print both correct and wrong predictions.

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn import metrics

# Load the dataset.
iris = load_iris()
print("Iris Target Names:\n", iris.target_names)

# Split the dataset.
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)

# Implement KNN algorithm with K = 1.
kn = KNeighborsClassifier(n_neighbors=1)
kn.fit(X_train, y_train)

# Get the predictions.
y_pred = kn.predict(X_test)

print("\nFirst few predictions:\nActual\t\t\tPredicted")
target_names = iris.target_names
for i in range(10):
    print(target_names[y_test[i]], "\t\t", target_names[y_pred[i]])

# Print the accuracy of the model.
print("\nAccuracy:", metrics.accuracy_score(y_test, y_pred))
