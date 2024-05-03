"""
Gradient Descent Algo
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Main function is y=x**2
# Gradient is 2x
# Initialization & Parameters
w=3
eta = 0.1 #just nice


# Gradient
for i in range(3):
    w = w-0.1* 3*(np.sin(np.exp**w)**2) * np.cos(np.exp**w) * (np.exp**w)
    # w = w-0.1* 3*(np.sin(np.power(np.exp,w))**2) * np.cos(np.exp**w) * (np.exp**w)
    # np.power(np.exp,w)
    print("w: ",w)



# print('------ eta = '+str(eta)+' -------')
# for i in range(0,5):
#     x= x-2*eta*x
#     print(x)

# x = -1
# eta = 0.01 #too small
# print('------ eta = '+str(eta)+' -------')
# for i in range(0,5):
#     x= x-2*eta*x
#     print(x)

# x = -1
# eta = 0.9 #too big
# print('------ eta = '+str(eta)+' -------')
# for i in range(0,5):
#     x= x-2*eta*x
#     print(x)
