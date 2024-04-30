import numpy as np
from numpy.linalg import inv
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures


"""
README
1) Change all variables in Section A appropriately
2) If normal regression is involved: proceed to Section B,
    if ridge regression is involved: proceed to Section C

"""


""" Section A: Variables """
# ================================================
# for training
x = np.array([1,0,1, 1,-1,1]).reshape(2,3)
y = np.array([0,1]).reshape(2,1)

# for testing
x_test = np.array([]).reshape
# ================================================

# Generate polynomial feature, P from X
order = 3
poly = PolynomialFeatures(order)
P = poly.fit_transform(x) # transforming x into p
print("Matrix P: \n", P)

""" Section B: Non ridge Regression Choose one, comment out the other """
# ================================================
# Underdetermined System m < d. Comment out line below if it is system is overdetermined
w_dual = P.T @ inv(P @ P.T) @ y
print(w_dual)

# Overdetermined System m > d. Comment out line below if it is system is underdetermined
# w_primal = inv(P.T @ P) @ P.T @ y
# print(w_primal)
# ================================================

""" Section C: Ridge Regression Choose one, comment out the other """
# ================================================
# Underdetermined System m < d. Comment out code below if it is system is overdetermined
reg_L = 0.0001*np.identity(P.shape[0]) # size of P.T @ P = m x m (rows)
w_dual_ridge = P.T @ inv(P @ P.T + reg_L) @ y
print(w_dual_ridge)

# Overdetermined System m > d. Comment out code below if it is system is underdetermined
reg_L = 0.0001*np.identity(P.shape[1]) # size of P.T @ P = d x d (columns)
w_primal_ridge = inv(P.T @ P + reg_L) @ P.T @ y
print(w_primal_ridge)
# ================================================


