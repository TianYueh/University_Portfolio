#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample code of HW4, Problem 4
"""

import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy import linalg

myfile = open('hw4_problem4_data.pickle', 'rb')
mydict = pickle.load(myfile)

X_train = mydict['X_train']
X_test = mydict['X_test']
Y_train = mydict['Y_train']
Y_test = mydict['Y_test']

predictive_mean = np.empty(X_test.shape[0])
predictive_std = np.empty(X_test.shape[0])

sigma = 0.1
sigma_f = 1.0
ls = 3

#-------- Your code (~10 lines) ---------
def kernel(xi, xj):
  return sigma_f*sigma_f * np.exp(-np.linalg.norm(xi - xj, axis = -1)**2 / (2*ls*ls))

K = [[kernel(X_train[i], X_train[j]) for j in range(len(X_train))] for i in range(len(X_train))]
K = np.array(K)

for i, xk in enumerate(X_test):
  # enumerate wa meiju to yobu mono da, ikai de futatsu no mono wo syouri suru koto ga dekiru, deetabeesu no join mitai na kanji.
  # HW no ue no kousiki wo tsukatte, mean to var wo hyoukenn suru.
  mean = np.dot(kernel(xk, X_train), np.linalg.solve(K + sigma*sigma*np.identity(len(X_train)), Y_train))
  var = kernel(xk, xk) - np.dot(kernel(xk, X_train), np.linalg.solve(K + sigma*sigma*np.identity(len(X_train)), kernel(X_train, xk)))
  
  predictive_mean[i] = mean
  predictive_std[i] = np.sqrt(var)


#---------- End of your code -----------

# Optional: Visualize the training data, testing data, and predictive distributions
fig = plt.figure()
plt.plot(X_train, Y_train, linestyle='', color='b', markersize=5, marker='+',label="Training data")
plt.plot(X_test, Y_test, linestyle='', color='orange', markersize=2, marker='^',label="Testing data")
plt.plot(X_test, predictive_mean, linestyle=':', color='green')
plt.fill_between(X_test.flatten(), predictive_mean - predictive_std, predictive_mean + predictive_std, color='green', alpha=0.13)
plt.fill_between(X_test.flatten(), predictive_mean - 2*predictive_std, predictive_mean + 2*predictive_std, color='green', alpha=0.07)
plt.fill_between(X_test.flatten(), predictive_mean - 3*predictive_std, predictive_mean + 3*predictive_std, color='green', alpha=0.04)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
