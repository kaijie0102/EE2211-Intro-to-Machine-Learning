"""
Do not edit this doc
"""

# libraries #
import pandas as pd
import numpy as np
from numpy.linalg import inv
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank
from numpy.linalg import det

# Writing matrix #
x = np.array([1, 3, 1, 4, 1, 10, 1, 6, 1, 7]).reshape(5,2) # 5 rows of 2
y = np.array([0, 5, 1.5, 4, -3, 8, -4, 10, 1, 6]).reshape(5,2) # 5 rows of 2
print("x is:\n", x)
print("y is:\n", y)
print()
x_square = np.array([1,1,1,-2]).reshape(2,2) # 2 rows of 2
y2 = np.array([1,2,]).reshape(2,1) # 2 rows of 1
"""
x2 = np.array([1,2,3,4,5,6,7,8,9]).reshape(1,9) # 1 row of 9
x3 = np.array([1,2,3,4,5,6,7,8,9]).reshape(9,1) # 9 rows of 1
print(x2)
print(x3)
"""

# Show shape and size #
print("Shape of x:\n", x.shape)
print()

# Rank & Det of matrix #
print("Rank of x:\n", matrix_rank(x)) # Rank = No. of Rows ==> Full Rank ==> Invertible
print("Determinant of x:\n", det(x_square)) # only valid for square matrices. Determinant != 0 ==> Invertible
print()

# inverse & transpose of x #
print("Tranpose of x:\n", x.T)
print("Inverse of x_square:\n", inv(x_square)) # Invertible if matrix is square and have full rank
print()

# Even-determined System: m = d # 
w_even = inv(x_square) @ y2
print("Even-determined w:\n", w_even)

# Over-determined System: m > d #
print("Is left inverse avail?:\n", det(x.T @ x)) # only valid for square matrices. Determinant != 0 ==> Invertible
w_over = inv(x.T @ x) @ x.T @ y # if equation is X @ w = Y
# w_over = inv(x @ x.T) @ x @ y # if equation is X.T @ w = Y 
print("Over-determined w:\n", w_over)

# Under-determined System: m < d #
print("Is right inverse avail?:\n", det(x @ x.T)) # only valid for square matrices. Determinant != 0 ==> Invertible
w_under = x.T @ inv(x @ x.T) @ y # if equation is X @ w = Y
# w_under = x @ inv(x.T @ x) @ y # if equation is X.T @ w = Y 
print("Under-determined w:\n", w_under)
print()

# Using Mean Squared Error #
w = w_even
y_calculated = x @ w
MSE = mean_squared_error(y_calculated, y)

# Plotting Graph #
plt.plot(x[:,1], y[:,0], 'o', label = 'y1')
plt.plot(x[:,1], y[:,1], 'x', label = 'y2')
plt.plot(x[:,1], y_calculated[:,0])
plt.plot(x[:,1], y_calculated[:,1])
plt.xlabel("X")
plt.ylabel("Y")
plt.show()