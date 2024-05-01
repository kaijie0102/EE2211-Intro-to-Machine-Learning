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

""" Section B: Non ridge Regression. If using this section, comment out Section C """
# ================================================
if (P.shape[1] > P.shape[0]):
    # Underdetermined System m < d
    w_dual = P.T @ inv(P @ P.T) @ y
    print(w_dual)
else: 
    # Overdetermined System m > d
    w_primal = inv(P.T @ P) @ P.T @ y
    print(w_primal)
# ================================================

""" Section C: Ridge Regression. If using this section, comment out Section B """
# ================================================
# Underdetermined System m < d
if (P.shape[1] > P.shape[0]):
    reg_L = 0.0001*np.identity(P.shape[0]) # size of P.T @ P = m x m (rows)
    w_dual_ridge = P.T @ inv(P @ P.T + reg_L) @ y
    print(w_dual_ridge)
else:
    # Overdetermined System m > d
    reg_L = 0.0001*np.identity(P.shape[1]) # size of P.T @ P = d x d (columns)
    w_primal_ridge = inv(P.T @ P + reg_L) @ P.T @ y
    print(w_primal_ridge)
# ================================================


