import numpy as np
from numpy.linalg import inv
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures

#  Binary Classification (Lecture 6 Demo 1: 0:26:22)
"""
X = np.array([1,-9, 1,-7, 1,-5, 1,1, 1,5, 1,9]).reshape(6,2)
y = np.array([-1, -1, -1, 1, 1, 1]).reshape(6,1)

# Linear Regression for classification
w = inv(X.T @ X) @ X.T @ y
print("Estimated W")
print(w)

X_test = np.array([1,-2]).reshape(1,2)
y_predict = X_test @ w

print("Predicted y")
print(y_predict)

# Applying signum function to make it 1 or -1 (aka one of the 2 classes)
y_class_predicted = np.sign(y_predict)

print("Predicted Y Class")
print(y_class_predicted)
"""

# Multi-Category Classification (Lecture 6 Demo 2: 0:39:54)
"""
X = np.array([1,1,1, 1,-1,1, 1,1,3, 1,1,0]).reshape(4,3)
y_class = np.array([1,2,1,3]).reshape(4,1)
y_one_hot = np.array([1,0,0, 0,1,0, 1,0,0, 0,0,1]).reshape(4,3)

# NOT recommended: Manually assigning the one-hot assignments.
print("One-hot encoding manual")
print(y_class)
print(y_one_hot)

# Recommended: Using libs
print("One-hot encoding function")
onehot_encoder = OneHotEncoder(sparse=False) # sparse=False format improves readability
print(onehot_encoder)

y_train_onehot = onehot_encoder.fit_transform(y_class)
print(y_train_onehot).
 VBN
# Linear Classification
W = inv(X.T @ X) @ X.T @ y_train_onehot
print("Estimated W")
print(W)

X_test = np.array([1,6,8, 1,0,-1]).reshape(2,3)
y_test = X_test @ W
print("Test") 
print(y_test)

# Assigning a class to the output based on the highest value in the axis. Eg [a,b,c] if a is the highest (positive) value, the class is [1,0,0]
# For each row(output vectors), get the column(axis) with the highest value
yt_class = [[1 if col == max(row) else 0 for col in row] for row in y_test ]
print("Class label test")
print(yt_class)
"""

# Polynomial Regression (Lecture 6 Demo 3: 1:28:13)

X = np.array([0,0, 1,1, 1,0, 0,1]).reshape(4,2) # there is no need for 1 to be in first column of X, fit_transform function will do it
y = np.array([-1, -1, 1, 1]).reshape(4,1)

# Generate polynomial features. 
order = 2
poly = PolynomialFeatures(order)
print(poly)
# P is a non linear function of X but P is linear with the model f(x)
P = poly.fit_transform(X) # there is no need for 1 to be in first column of X
print("matrix P")
print(P)


print("Under-determined system, do right inverse to find w(solution)")
w_dual = P.T @ inv(P @ P.T) @ y
print("w: Under-determined --> Lesser equations than features --> Unique Constrained Solution (No Ridge) ")
print(w_dual)

# testing and prediction
print("Prediction")
Xnew = np.array([0.1,0.1, 0.9,0.9, 0.1, 0.9, 0.9,0.1]).reshape(4,2)
Pnew = poly.fit_transform(Xnew) # transform the test input data 
Ynew = Pnew @ w_dual
print("Ynew: \n", Ynew)
print("Class: \n", np.sign(Ynew))
