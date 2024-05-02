import pandas as pd
import numpy as np
from numpy.linalg import inv
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank
from numpy.linalg import det

"""
Fill in everything here in one line, row by row. 
Eg, if 3x2: [ [1,1], [1,-1], [1,0] ] = [1,1,1,-1,1,0]
Remember to augment with 1s to add bias to X only
"""
x_array = [1,4, 1,7, 1,10, 1,2, 1,3, 1,9]
x_row = 6
x_col = 2

y_array = [-1,-1,-1,1,1,1]
y_row = 6
y_col = 1

### CHANGE if making prediction (comment out if not in use) ###
# x_to_predict_array = [1, 63, 9]
# x_to_predict_row = 1
# y_to_predict_col = 3

"""
Comment out what you don't need
=========================
"""
### Do not touch ###
x = np.array(x_array).reshape(x_row, x_col)
y = np.array(y_array).reshape(y_row, y_col)
print("x is:\n", x)
print("y is:\n", y)
print()

### even-determined m=d,square (comment out if not in use) ###
# w = inv(x) @ y

### over-determined m>d, tall (comment out if not in use) ###
w = inv(x.T @ x) @ x.T @ y

### under-determined m<d, wide (comment out if not in use) ###
# w = x.T @ inv(x @ x.T) @ y

### Do not touch (comment out if not in use) ###
print("w is: ")
print(w)
print()


### Do not touch (comment out if not in use) ###
# x_to_predict = np.array(x_to_predict_array).reshape(x_to_predict_row, y_to_predict_col)
# y_predicted = x_to_predict @ w
# print("y predicted is:\n", y_predicted)


### Mean Square Regression (comment out if not in use) ###
y_calculated = x @ w
MSE = mean_squared_error(y_calculated, y)
print("MSE\n", MSE)

### Linear Regression --> Sign Function ###
y_class_predicted = np.sign(y_calculated)
print(y_class_predicted)


