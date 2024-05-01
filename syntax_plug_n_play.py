"""
Take code from syntax.py and paste here
"""

import pandas as pd
import numpy as np
from numpy.linalg import inv
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank
from numpy.linalg import det

x = np.array([1,1,1,0]).reshape(2,2) 
y = np.array([1,2,3]).reshape(3,1)
print("x is:\n", x)
print("y is:\n", y)
print()



"""
HAVE YOU RESHAPED??????????
"""

# Rank & Det of matrix #
# print("Rank of x:\n", matrix_rank(x)) # Rank = No. of Rows ==> Full Rank ==> Invertible
# print("Determinant of x:\n", det(x)) # only valid for square matrices. Determinant != 0 ==> Invertible
# print()

# inverse & transpose of x #
# print("Tranpose of x:\n", x.T)
# print("Inverse of x_square:\n", inv(x)) # Invertible if matrix is square and have full rank
# print()

# Even-determined System: m = d # 
# w_even = inv(x) @ y2
# print("Even-determined w:\n", w_even)

# Over-determined System: m > d #
# print("Is left inverse avail?:\n", det(x.T @ x)) # only valid for square matrices. Determinant != 0 ==> Invertible
# w_over = inv(x.T @ x) @ x.T @ y # if equation is X @ w = Y
# w_over = inv(x @ x.T) @ x @ y # if equation is X.T @ w = Y 
# print("Over-determined w:\n", w_over)

# Under-determined System: m < d #
# print("Is right inverse avail?:\n", det(x @ x.T)) # only valid for square matrices. Determinant != 0 ==> Invertible
# w_under = x.T @ inv(x @ x.T) @ y # if equation is X @ w = Y
# # w_under = x @ inv(x.T @ x) @ y # if equation is X.T @ w = Y 
# print("Under-determined w:\n", w_under)
# print()

# Using Mean Squared Error #
# w = w_even
# y_calculated = x @ w
# MSE = mean_squared_error(y_calculated, y)
