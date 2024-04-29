# Rank & Transpose (Lecture 4 Demo 1: 25:00)
"""
import numpy as np
from numpy.linalg import matrix_rank

x = np.array([ [1,4,3], [0,4,2], [1,8,5] ]) # 3 rows of 3

print(matrix_rank(x)) # calculate rank of x
print(x)
print(x.T) # transpose of x
"""

# Product and Inverse (Lecture 4 Demo 2: 45:30)
"""
import numpy as np
from numpy.linalg import inv

x = np.array([ [1,4], [0,4], [3,-2] ])
y = np.array([3,0.5,4]) # 3 by 1 but NOT the recommended way to write column vector

print("x shape(dim): " , x.shape) # show you row x column (3,2)
print("y:" , y)
print("y shape: " , y.shape) # show you row x column (3,) == (3,1)

y1 = np.array([ [3],[0.5],[4] ]) # 3 by 1 ***recommended way to write column vector***
print("y1: ")
print(y1)
print("y1 shape:" , y1.shape)

z = x.T @ y # x transpose times Y
print("Vector-matrix product")
print(z)

z1 = x.T @ y1 
print("Vector-matrix product")
print(z1)

x2 = np.array([ [1,4,3], [0,4,2] ])
Q = x@x2 # 3 by 3
print("Q")
print(Q)

print("matrix inverse")
p = np.array([ [1,4,3], [0,4,2,], [3,-2,9] ])
print(inv(p))
"""

# Even determined system: m=3 (Lecture 4 Demo 3: 1:11:00)
"""
import numpy as np
from numpy.linalg import inv
from numpy.linalg import matrix_rank
import matplotlib.pyplot as plt


x = np.array([ [1,1], [1,-2] ])
y = np.array([ [4], [1] ])
w = inv(x) @ y
print("w")
print(w)

print("more eg")
x1 = np.array([ [1,4,2], [0,4,3], [3,4,9] ])
y1 = np.array([ [39],[40],[50]])
w1 = inv(x1) @ y1
print(w1)

# not full rank == no inverse available --> no solution available
print("rank")
x2 = np.array([ [1,4,2], [0,4,3], [1,8,5] ])
y2 = np.array([ [1], [0], [1] ])
print("y2")
print(y2)
print(matrix_rank(x2))
"""

# Over determined system: m=3 (Lecture 4 Demo 4: 1:20:00)

import numpy as np
from numpy.linalg import inv

# No exact solution in general: can only use approximated solution. X.T @ X @ w = X.T @ Y --> w' = solution = X.T @ Y
# First check if X.T @ X = I (invertible) --> find approximate solution

x = np.array([ [1,1], [1,-1], [1,0] ])
y = np.array([ [1], [0], [2] ])

w = inv(x.T @ x ) @ x.T @ y
print("w")
print(w)

