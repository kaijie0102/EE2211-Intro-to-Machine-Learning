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
x = np.array([4,7,10,2,3,9]).reshape(6,1)
""" 
there is no need for 1 to be in first column of X, 
fit_transform function will do it
"""
y = np.array([-1,-1,-1,1,1,1]).reshape(6,1)

# for testing
x_test = np.array([6]).reshape(1,1)

# ================================================

# Generate polynomial feature, P from X
order = 4
poly = PolynomialFeatures(order)
P = poly.fit_transform(x) # transforming x into p
P_test = poly.fit_transform(x_test)
print("Matrix P: \n", P)

""" Section B: Non ridge Regression. If using this section, comment out Section C """
# ================================================
if (P.shape[1] > P.shape[0]):
    # Underdetermined System m < d
    w_dual = P.T @ inv(P @ P.T) @ y
    print("w_dual: ", w_dual)
    w = w_dual
else: 
    # Overdetermined System m > d
    w_primal = inv(P.T @ P) @ P.T @ y
    print("w_primal: ", w_primal)
    w = w_primal
# ================================================

""" Section C: Ridge Regression. If using this section, comment out Section B """
# ================================================
# Underdetermined System m < d
# if (P.shape[1] > P.shape[0]):
#     reg_L = 0.0001*np.identity(P.shape[0]) # size of P.T @ P = m x m (rows)
#     w_dual_ridge = P.T @ inv(P @ P.T + reg_L) @ y
#     print(w_dual_ridge)
# else:
#     # Overdetermined System m > d
#     reg_L = 0.0001*np.identity(P.shape[1]) # size of P.T @ P = d x d (columns)
#     w_primal_ridge = inv(P.T @ P + reg_L) @ P.T @ y
#     print(w_primal_ridge)
# ================================================

# best training error count
y_calculated = P_test @ w
print("y_calculated: ", y_calculated)

y_class_predicted = np.sign(y_calculated)
print(y_class_predicted)


