import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# main
# Input (house size and rooms) and output (price)
x = np.array([0.2, 0.7, 1.8, 2.2, 3.7, 4.1, 4.5, 5.1, 6.3, 7.4])
y = np.array([2.1, 1.5, 5.8, 6.1, 9.1, 9.5, 9.8, 12.7, 13.8, 15.9])
mean_y = np.mean(y)

#sorting base on size of house
sort_index = x.argsort()
x = x[sort_index]
y = y[sort_index]
print(x)

squared_diff_root = (y - mean_y) ** 2
mse_root = np.mean(squared_diff_root)
print("MSE at root: ", mse_root)

# split x array using theshold
threshold = 3
x_partition_1 = x[x <= threshold]
y_partition_1 = y[x <= threshold]
mean_y1 = np.mean(y_partition_1)

x_partition_2 = x[x > threshold]
y_partition_2 = y[x > threshold]
mean_y2 = np.mean(y_partition_2)

squared_diff_1 = (y_partition_1 - mean_y1) ** 2
squared_diff_2 = (y_partition_2 - mean_y2) ** 2
mse_1 = np.mean(squared_diff_1)
mse_2 = np.mean(squared_diff_2)
print("MSE below threshold at depth 1: ", mse_1)
print("MSE above threshold at depth 1: ", mse_2)

# Overall mse at depth 1
overall_mse = len(y_partition_1)/len(y) * mse_1 + len(y_partition_2)/len(y) * mse_2
print("Overall MSE at depth 1: ", overall_mse)
