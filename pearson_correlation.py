import numpy as np

def pearson_correlation(x1,x2,x3,x4, y):
    # Calculate means
    mean_x1 = np.mean(x1)
    mean_x2 = np.mean(x2)
    mean_x3 = np.mean(x3)
    mean_x4 = np.mean(x4)
    mean_y = np.mean(y)
    
    # Calculate covariance
    covariance1 = np.sum((x1 - mean_x1) * (y - mean_y))
    covariance2 = np.sum((x2 - mean_x2) * (y - mean_y))
    covariance3 = np.sum((x3 - mean_x3) * (y - mean_y))
    covariance4 = np.sum((x4 - mean_x4) * (y - mean_y))
    
    # Calculate standard deviations
    std_dev_x1 = np.sqrt(np.sum((x1 - mean_x1)**2))
    std_dev_x2 = np.sqrt(np.sum((x2 - mean_x2)**2))
    std_dev_x3 = np.sqrt(np.sum((x3 - mean_x3)**2))
    std_dev_x4 = np.sqrt(np.sum((x4 - mean_x4)**2))
    std_dev_y = np.sqrt(np.sum((y - mean_y)**2))
    
    # Calculate Pearson correlation coefficient
    pearson_coefficient1 = covariance1 / (std_dev_x1 * std_dev_y)
    pearson_coefficient2 = covariance2 / (std_dev_x2 * std_dev_y)
    pearson_coefficient3 = covariance3 / (std_dev_x3 * std_dev_y)
    pearson_coefficient4 = covariance4 / (std_dev_x4 * std_dev_y)
    print("r1:", pearson_coefficient1)
    print("r2:", pearson_coefficient2)
    print("r3:", pearson_coefficient3)
    print("r4:", pearson_coefficient4)
    # return pearson_coefficient

# Inputs from tutorial 7 question 1: Feature 3 and Target Y
x1 = [-0.709,1.7255,0.9539,-0.7581,-1.035,-1.049]
x2 = [2.8719,1.5014, 1.8365, -0.5467, 1.8274, 0.3501]
x3  = [-1.8349, 0.4055, 1.0118, 0.5171, 0.7279, 1.2654]
x4  = [2.6354, 2.7448, 1.4616, 0.7258, -1.6893, -1.7512]
y = [0.8206, 1.0639, 0.6895, -0.0252, 0.995, 0.6608]

# print("Pearson correlation coefficient:", pearson_correlation(x1,x2,x3,x4, y))
pearson_correlation(x1,x2,x3,x4, y)