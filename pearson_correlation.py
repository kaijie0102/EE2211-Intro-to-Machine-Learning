import numpy as np

def pearson_correlation(x, y):
    # Calculate means
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    # Calculate covariance
    covariance = np.sum((x - mean_x) * (y - mean_y))
    
    # Calculate standard deviations
    std_dev_x = np.sqrt(np.sum((x - mean_x)**2))
    std_dev_y = np.sqrt(np.sum((y - mean_y)**2))
    
    # Calculate Pearson correlation coefficient
    pearson_coefficient = covariance / (std_dev_x * std_dev_y)
    
    return pearson_coefficient

# Example usage:
# x = [1, 2, 3, 4, 5]
# y = [2, 3, 4, 5, 6]

# Inputs from tutorial 7 question 1: Feature 3 and Target Y
x = [-0.9852, 1.3766, -1.3244, -0.6316, -0.8320]
y = [0.2758, 1.4392, -0.4611, 0.6154, 1.0006]

print("Pearson correlation coefficient:", pearson_correlation(x, y))