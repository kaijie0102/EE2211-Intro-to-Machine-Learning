# -*- coding: utf-8 -*-
"""EE2211tut7_9.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1XF-mLSYtENKjjtiSOBpZGYnuP-nMGdnN
"""

# Tutorial 9 Q3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Juan Helen Zhou
# modified by: jing yang

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt


def our_own_tree(y):
    #the argument given to this function y, is y_train, the ouput for training data. 
    #y MUST be arranged such that the x (NOT y) goes in monotone increasing (or decreasing) order.
    
    # split data at first level
    # L stands for left, R stands for right
    
    # for each feature X_k
    #     resort y according to X_k
    #     find_best_split(y), to get yL_k,yR_k AND MSE_k
    # pick feature X_i get gives smallest MSE_k.
    
    yL, yR = find_best_split(y)

    # split data at second level
    yLL, yLR = find_best_split(yL)
    yRL, yRR = find_best_split(yR)

    # compute prediction for each value of y
    yLL_pred = np.mean(yLL)*np.ones(len(yLL))
    yLR_pred = np.mean(yLR)*np.ones(len(yLR))
    yRL_pred = np.mean(yRL)*np.ones(len(yRL))
    yRR_pred = np.mean(yRR)*np.ones(len(yRR))
    y_pred = np.concatenate([yLL_pred, yLR_pred, yRL_pred, yRR_pred])

    return y_pred

def find_best_split(y):

    # index represents last element in the "below threshold" node
    sq_err_vec = np.zeros(len(y)-1)
    for index in range(0, len(y)-1):

        # split the data at the given "index"
        data_below_threshold = y[:index+1]
        data_above_threshold = y[index+1:]

        # Compute estimate of the y_pred for each node.
        mean_below_threshold = np.mean(data_below_threshold)
        mean_above_threshold = np.mean(data_above_threshold)

        # Compute total square error
        # Note that MSE = total square error divided by number of data points
        below_sq_err =  np.sum(np.square(data_below_threshold - mean_below_threshold))
        above_sq_err = np.sum(np.square(data_above_threshold - mean_above_threshold))
        sq_err_vec[index] = below_sq_err + above_sq_err # save the overall MSE if we choose to split at this index

    best_index = np.argmin(sq_err_vec) #choose the index that minimises the overall MSE at that depth.
    yL = y[:best_index+1]
    yR = y[best_index+1:]
    return yL, yR

# data extraction
housing = fetch_california_housing()
print(housing.target_names)
print(housing.feature_names)
X = housing.data[:,0]
y = housing.target
print(y)
sort_index = X.argsort() #resorting X 
X = X[sort_index] #resorting X
y = y[sort_index] #resorting Y

print(X.reshape(-1,1))
# scikit decision tree regressor
scikit_tree = DecisionTreeRegressor(criterion='mse', max_depth=2)
scikit_tree.fit(X.reshape(-1,1), y) # reshape necessary because tree expects 2D array
scikit_tree_predict = scikit_tree.predict(X.reshape(-1,1))

# Our own tree regressor
tree_predict = our_own_tree(y)

# Plot
# notice in the plot that x is sorted alr.
plt.figure(0, figsize=[9,4.5])
plt.rcParams.update({'font.size': 16})
plt.scatter(X, y, c='steelblue', s=20)
plt.plot(X, scikit_tree_predict, color='black', lw=2, label='scikit-learn')
plt.plot(X, tree_predict, color='red', linestyle='--', lw=2, label='our own tree')
plt.xlabel('MedInc')
plt.ylabel('MedHouseVal')
plt.legend(loc='upper right',ncol=3, fontsize=15)
plt.savefig('FigTut9_Q3.png')