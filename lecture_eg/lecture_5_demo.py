#  Linear Regression (Lecture 5 Demo 1: 1:25:30)
"""
import pandas as pd
import numpy as np
from numpy.linalg import inv
from sklearn.metrics import mean_squared_error

x = np.array([ [1,-9], [1,-7], [1,-5], [1,1], [1,5], [1,9] ]) # add one infront to add bias
y = np.array([ [-6],[-6],[-4],[-1],[1],[4] ])

w = inv(x.T @ x) @ x.T @ y
print("w")
print(w)

x_test = np.array([[1,-1]]) # predict results given x using the model weights
y_test = x_test @ w

print("y test results")
print(y_test)

y_calculated = x @ w
MSE = mean_squared_error(y_calculated, y)

print("MSE")
print(MSE)
# """

#  Linear Regression of Matrix(Lecture 5 Demo 2: 1:54:30)
"""
import pandas as pd
import numpy as np
from numpy.linalg import inv
from sklearn.metrics import mean_squared_error

# x = np.array([ [1,1,1], [1,-1,1], [1,1,3], [1,1,0] ])
# y = np.array([ [1,0], [0,1], [2,-1], [-1,3] ])

# x = np.array([ [], [], [], [], [], [] ])
# y = np.array([ [], [], [], [], [], [] ])

x = np.array([ [1,2], [0,6], [1,0], [0,5], [1,7] ])
y = np.array([ [1], [2], [3], [4], [5] ])

w = inv(x.T @ x) @ x.T @ y
print("weights")
print(w)

# prediction
x_test = np.array([[1,6,8], [1,0,-1]])
y_test = x_test @ w

print("Result")
print(y_test)

# difference
y_calculated = x @ w
MSE = mean_squared_error(y_calculated, y)
print("MSE")
print(MSE)
"""

#  Linear Regression of Matrix(Lecture 5 Demo 3: 1:56:35)
# """
import pandas as pd
import numpy as np
from numpy.linalg import inv
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

x = np.array([ [1,1,2], [1,0,6], [1,1,0], [1,0,5], [1,1,7] ])
y = np.array([ [1,1], [1,2], [1,3], [1,4], [1,5] ])


# x = np.array([ [1,3], [1,4], [1,10], [1,6], [1,7] ])
# y = np.array([ [0,5], [1.5,4], [-3,8], [-4,10], [1,6] ])
w = inv(x.T @ x) @ x.T @ y

print("w")
print(w)

# prediction
x_test = np.array([ [1,1,3] ])
y_test = x_test @ w
print("y test")
print(y_test)

y_calculated = x @ w
print("y calculated")
print(y_calculated)
plt.plot(x[:,1], y[:,0], 'o', label = 'y1')
plt.plot(x[:,1], y[:,1], 'x', label = 'y2')
plt.plot(x[:,1], y_calculated[:,0])
plt.plot(x[:,1], y_calculated[:,1])
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
# MSE
MSE = mean_squared_error(y_calculated, y)
print("MSE")
print(MSE)
# """

