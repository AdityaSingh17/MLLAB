# Assuming a set of documents that need to be classified, use the naïve Bayesian Classifier model to perform this task. Built-in Java classes/API can be used to write the program.
# Calculate the accuracy, precision, and recall for your data set.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

msg = pd.read_csv("naivetrext1.csv", names=["message", "label"])
print("The dimensions of the dataset", msg.shape)
msg["labelnum"] = msg.label.map({"pos": 1, "neg": 0})
X = msg.message
y = msg.labelnum

# Splitting the dataset into train and test data.
xtrain, xtest, ytrain, ytest = train_test_split(X, y)
print(xtest.shape)
print(xtrain.shape)
print(ytest.shape)
print(ytrain.shape)
print("train data")
print(xtrain)

# Output of the count vectoriser is a sparse matrix.
count_vect = CountVectorizer()
xtrain_dtm = count_vect.fit_transform(xtrain)
xtest_dtm = count_vect.transform(xtest)
print(count_vect.get_feature_names())
df = pd.DataFrame(xtrain_dtm.toarray(), columns=count_vect.get_feature_names())
print(df)  # Tabular representation.
print(xtrain_dtm)  # Sparse matrix representation.

# Training Naive Bayes (NB) classifier on training data.
clf = MultinomialNB().fit(xtrain_dtm, ytrain)
predicted = clf.predict(xtest_dtm)

# Printing the accuracy metrics.
print("Accuracy metrics")
print("Accuracy of the classifer is", metrics.accuracy_score(ytest, predicted))
print("Confusion matrix")
print(metrics.confusion_matrix(ytest, predicted))
print("Recall and Precison ")
print(metrics.recall_score(ytest, predicted))
print(metrics.precision_score(ytest, predicted))

"""docs_new = ['I like this place', 'My boss is not my saviour']
X_new_counts = count_vect.transform(docs_new)
predictednew = clf.predict(X_new_counts)
for doc, category in zip(docs_new, predictednew):
print('%s-&gt;%s' % (doc, msg.labelnum[category]))"""
