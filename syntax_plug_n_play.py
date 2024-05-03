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

# x1 = [4,7,10, 2,3,9]
# x = np.ones([len(x1), 2]) # shape = len(x1), 2
# x[:, 1] = x1
x = np.array([5,0,5,10,17,10,20,0,20]).reshape(3,3) 
# y = np.array([-1,-1,-1, 1,1,1]).reshape(6,1)
print("x is:\n", x)
# print("y is:\n", y)
print()
# y = np.array([5,13,15,25,6,16,2,19])
# MSE = mean_squared_error(13.5, y)
# print("MSe:", MSE)


print("Rank of x:\n", matrix_rank(x)) # Rank = No. of Rows ==> Full Rank ==> Invertible
# print("Determinant of x:\n", det(x_square)) # only valid for square matrices. Determinant != 0 ==> Invertible
# print()


"""
HAVE YOU RESHAPED??????????
"""


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
