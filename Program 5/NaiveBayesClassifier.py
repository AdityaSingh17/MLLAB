# Write a program to implement the naïve Bayesian classifier for a sample training data set stored as a .CSV file. Compute the accuracy of the classifier, considering few test data sets.

import numpy as np
import pandas as pd

mush = pd.read_csv("mushroom.csv")
mush.replace("?", np.nan, inplace=True)
print("Initially", len(mush.columns), "columns. After dropping NA:", len(mush.dropna(axis=1).columns), "columns")

# Drop wherever you have ?, the values are not known.
mush.dropna(axis=1, inplace=True)

# The first column in dataset is "class" which is target variable.
target = "class"
features = mush.columns[mush.columns != target]
classes = mush[target].unique()
test = mush.sample(frac=0.3)
mush = mush.drop(test.index)
probs = {}
probcl = {}

for x in classes:
    mushcl = mush[mush[target] == x][features]
    clsp = {}
    tot = len(mushcl)
    for col in mushcl.columns:
        colp = {}
        for val, cnt in mushcl[col].value_counts().iteritems():
            pr = cnt / tot
            colp[val] = pr
            clsp[col] = colp

    probs[x] = clsp
    probcl[x] = len(mushcl) / len(mush)


def probabs(x):
    # X - pandas Series with index as feature
    if not isinstance(x, pd.Series):
        raise IOError("Arg must of type Series")
    probab = {}

    for cl in classes:
        pr = probcl[cl]
        for col, val in x.iteritems():
            try:
                pr *= probs[cl][col][val]
            except KeyError:
                pr = 0
        probab[cl] = pr
    return probab


def classify(x):
    probab = probabs(x)
    mx = 0
    mxcl = ""
    for cl, pr in probab.items():
        if pr > mx:
            mx = pr
            mxcl = cl
    return mxcl


# Train data
b = []
for i in mush.index:
    # print(classify(mush.loc[i,features]),mush.loc[i,target])
    b.append(classify(mush.loc[i, features]) == mush.loc[i, target])
print("Train dataset:")
print(sum(b), "correct out of", len(mush))
print("Train Accuracy:", sum(b) / len(mush))

# Test data
b = []
for i in test.index:
    # print(classify(mush.loc[i,features]),mush.loc[i,target])
    b.append(classify(test.loc[i, features]) == test.loc[i, target])
print("Test dataset:")
print(sum(b), "correct out of", len(test))
print("Test Accuracy:", sum(b) / len(test))
