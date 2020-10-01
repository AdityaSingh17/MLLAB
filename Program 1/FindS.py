# Implement and demonstratethe FIND-S algorithm for finding the most specific hypothesis based on a given set of training data samples.

import random
import csv


def read_data(filename):
    with open(filename, "r") as csvfile:
        datareader = csv.reader(csvfile, delimiter=",")
        traindata = []
        for row in datareader:
            traindata.append(row)
        return traindata


def findS():  # Function for finding maximally specific set.
    dataarr = read_data("ENJOYSPORT.csv")
    h = ["0", "0", "0", "0", "0", "0"]
    rows = len(dataarr)
    columns = 7
    for x in range(1, rows):
        t = dataarr[x]
        print(t)
        if t[columns - 1] == "1":
            for y in range(0, columns - 1):
                if h[y] == t[y]:
                    pass
                elif h[y] != t[y] and h[y] == "0":
                    h[y] = t[y]
                elif h[y] != t[y] and h[y] != "0":
                    h[y] = "?"
            print(h)

    print("\n Maximally Specific set:")
    print("<", end=" ")
    for i in range(0, len(h)):
        print(h[i], ",", end=" ")
    print(">\n")


findS()
