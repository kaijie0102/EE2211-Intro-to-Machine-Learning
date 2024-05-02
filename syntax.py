"""
Do not edit this doc
This is a syntax bank, for you to plug and play
"""

# libraries #
import pandas as pd
import numpy as np
from numpy.linalg import inv
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank
from numpy.linalg import det
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures

### Section A: Basics ###
# Section A1: Initialisation #
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

# Section A2: Adding Bias programatically #
Y = [1,2,3]
X = np.ones([len(y), 2])
X[:, 1] = Y
print("X: \n",X)

# Section A3: Show shape and size #
print("Shape of x:\n", x.shape)
print()

# Section A4: Rank & Det of matrix #
print("Rank of x:\n", matrix_rank(x)) # Rank = No. of Rows ==> Full Rank ==> Invertible
print("Determinant of x:\n", det(x_square)) # only valid for square matrices. Determinant != 0 ==> Invertible
print()

# Section A5: inverse & transpose of x #
print("Tranpose of x:\n", x.T)
print("Inverse of x_square:\n", inv(x_square)) # Invertible if matrix is square and have full rank
print()

### Section B: Types of systems ###
# Section B1: Even-determined System: m = d # 
w_even = inv(x_square) @ y2
print("Even-determined w:\n", w_even)

# Section B2: Over-determined System: m > d #
print("Is left inverse avail?:\n", det(x.T @ x)) # only valid for square matrices. Determinant != 0 ==> Invertible
w_over = inv(x.T @ x) @ x.T @ y # if equation is X @ w = Y
# w_over = inv(x @ x.T) @ x @ y # if equation is X.T @ w = Y 
print("Over-determined w:\n", w_over)

# Section B3: Under-determined System: m < d #
print("Is right inverse avail?:\n", det(x @ x.T)) # only valid for square matrices. Determinant != 0 ==> Invertible
w_under = x.T @ inv(x @ x.T) @ y # if equation is X @ w = Y
# w_under = x @ inv(x.T @ x) @ y # if equation is X.T @ w = Y 
print("Under-determined w:\n", w_under)
print()

### Section C: Evaluation metrics: Error Rate ###
# Section C1: Using Mean Squared Error #
w = w_even
y_calculated = x @ w
MSE = mean_squared_error(y_calculated, y)

# Section C2: Using Mean Absolute Error #
# Ytrain = P @ w # same as Mean Square Error
y_calculated = x @ w
MeanAbsoluteError = mean_absolute_error(y_calculated, y)
print()
print("Mean Absolute Error:")
print(MeanAbsoluteError)

### Section D: Classification ###
# Section D1: Binary classification (Signum Function) #
y_class_predicted = np.sign(y) # Applying signum function to make it 1 or -1 (aka one of the 2 classes)

# Section D2: Manual One Hot Encoding #
Y = list()
for i in y:
    letter = [0, 0, 0]
    letter[i] = 1
    Y.append(letter)
Y = np.array(Y)

# Section D3: Multi-Category Classification (One Hot Encoding using lib) #
onehot_encoder = OneHotEncoder(sparse=False) # sparse=False format improves readability
y_train_onehot = onehot_encoder.fit_transform(y) # one hot encode y.
# Linear Classification
W = inv(x.T @ x) @ x.T @ y_train_onehot
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

### Section E: Polynomial Regression ###
# Section E1: Generate polynomial feature (Get P from X) #
order = 3
poly = PolynomialFeatures(order)
P = poly.fit_transform(x) # transforming x into p

# Section E2: Ridge Regression #
# Underdetermined System m < d.
reg_L = 0.0001*np.identity(P.shape[0]) # size of P.T @ P = m x m (rows)
w_dual_ridge = P.T @ inv(P @ P.T + reg_L) @ y
print(w_dual_ridge)

# Overdetermined System m > d. 
reg_L = 0.0001*np.identity(P.shape[1]) # size of P.T @ P = d x d (columns)
w_primal_ridge = inv(P.T @ P + reg_L) @ P.T @ y
print(w_primal_ridge)

### Section F: Gradient Descent ###
# Main function is y=x**2
# Gradient is 2x

# Initialization & Parameters
x = -1 #
eta = 0.4 #just nice
print('------ eta = '+str(eta)+' -------')

# Section F1: Convergence Criteria: Maximum k=5 #
for i in range(0,5):
    x= x-2*eta*x
    print(x)

# Section F2: Convergence Criteria: Stop if Percentage change in x < 1% #
count = 0
while count != 10: # max iteration = 10 
    count += 1
    x_new = x - eta * (2*x) # gradient = 2x
    print("Old X: {x}, New X: {x_new}".format(x=x, x_new=x_new))
    if ((x == 0) or ( abs((x_new - x)/x) < 0.01 )):
        print("Optimal X: ", x_new)
        break
    else: 
        x = x_new


### Section G: Graph Plotting ###
# Section G1: Plotting Graph #
plt.plot(x[:,1], y[:,0], 'o', label = 'y1')
plt.plot(x[:,1], y[:,1], 'x', label = 'y2')
plt.plot(x[:,1], y_calculated[:,0])
plt.plot(x[:,1], y_calculated[:,1])
plt.xlabel("X")
plt.ylabel("Y")
plt.show()