import csv
import os
import sys
from time import time

import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import preprocessing


"""
Digit Recognition with SVM (poly, degree = 2)

Accuracy on Kaggle: 0.97871

For complete dataset:
Reading (Preprocessing) time ~ 25.5 s
Training running time ~ 146.5 s
Predicting running time ~ 161.5 s
(Processor:1.7 GHz Intel Core i7, 
Memory: 8 GB)

"""

#os.chdir("YOUR_DIRECTORY_HERE")

try:
    train_f = open("train.csv")
    test_f = open("test.csv")
    train_csv = csv.reader(train_f)
    test_csv = csv.reader(test_f)
    
    
    print "----------------------------------------------\n"
    print "Digit Recognition with SVM, poly, degree = 2 \n"
    print "----------------------------------------------\n"
    
    t0 = time()
    
    print "Reading train data...\n"
    features_train = []
    labels_train = []
    train_csv.next()
    
    for row in train_csv:
        features_train.append([float(a) for a in row[1:]])
        labels_train.append(int(row[0]))
        
        
    print "Reading test data...\n"
    
    features_test = []
    test_csv.next()
    
    for row in test_csv:
        features_test.append([float(a) for a in row])
    
    """
    print "Preprocessing data...\n"
    features_train_scaled = preprocessing.scale(features_train)
    features_test_scaled = preprocessing.scale(features_test)
    """
    
    print "reading data time:", round(time()-t0, 3), "s\n"
    
    
    t0 = time()
    clf = svm.SVC(kernel = "poly", C = 100., degree = 2)
    clf.fit(features_train, labels_train)
    print "training time:", round(time()-t0, 3), "s\n"
    
    t0 = time()
    pred = clf.predict(features_test)
    print "predicting time:", round(time()-t0, 3), "s\n"

    #print "accuracy: ", accuracy_score(pred, labels_train[10000:10500]), "\n"


    print "Writing to file pred_svm.csv..."
    pred_f = open("pred_svm.csv", "w")
    pred_f.write("ImageId,Label\n")
    i = 1

    for item in pred:
        pred_f.write("%s,%s\n" % (i, item))
        i += 1

    print "----------------------------------------------\n"
    pred_f.close()

    train_f.close()
    test_f.close()

except IOError:
    print "\nError: File does not appear to exist."


