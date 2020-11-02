#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 13:19:27 2020

@author: erwanrahis
"""


import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


X_TRAIN = np.load('data/mnist1_features_train.npy', allow_pickle=True)
Y_TRAIN = np.load('data/mnist1_labels_train.npy', allow_pickle=True)
X_TEST = np.load('data/mnist1_features_test.npy', allow_pickle=True)
Y_TEST = np.load('data/mnist1_labels_test.npy', allow_pickle=True)


# GridSearchCV with kNN : a simple baseline
KNN = KNeighborsClassifier() # defining classifier
PARAMETERS = {'n_neighbors': [1, 2, 3, 4, 5]} # defining parameter space
CLF = GridSearchCV(KNN, PARAMETERS, cv=3)
CLF.fit(X_TRAIN, Y_TRAIN)

print('Returned hyperparameter: {}'.format(CLF.best_params_))
print('Best classification accuracy in train is: {}'.format(CLF.best_score_))
print('Classification accuracy on test is: {}'.format(CLF.score(X_TEST, Y_TEST)))
