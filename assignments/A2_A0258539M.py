import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from numpy.linalg import inv



def get_polynomial_data(X, order):
    # Based on the order of polynomial matrix required, this function will return P
    poly = PolynomialFeatures(order)
    return poly.fit_transform(X) #includes offset, returns P

def apply_left_inverse(X, y, L = 0):
        # when m > d for P, overdetermined system, left inverse
        # primal form for poly regression
        (m,d) = X.shape
        w = inv(X.T @ X + L * np.eye(d)) @ X.T @ y
        return w
        
def apply_right_inverse(X, y, L = 0):
    # m < d  for P, underdetermined system, right inverse
    # dual form for poly regression
    (m,d) = X.shape
    w = X.T @ inv(X @ X.T + L * np.eye(m)) @ y
    return w

def decode_ohe(Y): 
    # finding the maximum value in each array row
    m,d = Y.shape
    pred_class = -1 * np.ones(m)
    best = np.amax(Y,axis = 1)
    for i in range(m):
        for j in range(d):
            if best[i] == Y[i,j]:
                pred_class[i] = j
    
    return pred_class

def diff_count(x1, x2):
    count = 0
    for i in range(len(x1)):
        if x1[i] != x2[i]:
            count += 1
    return count


# Please replace "MatricNumber" with your actual matric number here and in the filename
def A2_A0258539M(N):
    """
    Input type
    :N type: int

    Return type
    :X_train type: numpy.ndarray of size (number_of_training_samples, 4)
    :y_train type: numpy.ndarray of size (number_of_training_samples,)
    :X_test type: numpy.ndarray of size (number_of_test_samples, 4)
    :y_test type: numpy.ndarray of size (number_of_test_samples,)
    :Ytr type: numpy.ndarray of size (number_of_training_samples, 3)
    :Yts type: numpy.ndarray of size (number_of_test_samples, 3)
    :Ptrain_list type: List[numpy.ndarray]
    :Ptest_list type: List[numpy.ndarray]
    :w_list type: List[numpy.ndarray]
    :error_train_array type: numpy.ndarray
    :error_test_array type: numpy.ndarray
    """
    # your code goes here
    
    ##################### PART A #########################
    # Load Iris dataset
    iris = load_iris()
    X = iris.data  # Features
    y = iris.target  # Type of Iris (0, 1 and 2)

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.7, random_state = N)
    y_train_reshaped = y_train.reshape(-1, 1) #transpose
    y_test_reshaped = y_test.reshape(-1, 1)
    
    
    
    ####################### PART B #######################
    # Finding target output using OHE of training and test sets
    
    ## Training Data
    onehot_encoder = OneHotEncoder(sparse_output = False)
    Ytr = onehot_encoder.fit_transform(y_train_reshaped)
    Yts = onehot_encoder.fit_transform(y_test_reshaped)
    
    
    ####################### PART C #######################
    Ptrain_list = []
    # List of training polynomial matrices from order 1 to 8
    for i in range(8):
        Ptrain_list.append(get_polynomial_data(X_train, order = i + 1))
            
            
    
    Ptest_list = []
    # List of testing polynomial matrices from order 1 to 8
    for i in range(8):
        Ptest_list.append(get_polynomial_data(X_test, order = i + 1))

        
        
    w_list = []
    # List of estimated regression coefficient from order 1 to 8
    for P in Ptrain_list:
        m = len(P) # no_of_rows
        d = len(P[0]) # no_of_cols
        
        if (m > d): # apply primal form (left inverse)
            w = apply_left_inverse(P, Ytr, 0.0001)
            
        else: # apply dual form (right inverse)
            w = apply_right_inverse (P, Ytr, 0.0001)
        w_list.append(w)
            

            
    error_train_array = []
    # List of error count for coefficient from order 1 to 8
    for i in range(8):
        actual_y_poly = y_train
        est_y_poly = decode_ohe(Ptrain_list[i] @ w_list[i]).reshape(-1, 1) #transpose
        error_train_array.append(diff_count(actual_y_poly, est_y_poly))
    error_train_array = np.array(error_train_array)
    
    
    
    error_test_array = []
    # List of error count for coefficient from order 1 to 8
    for i in range(8):
        actual_y_poly = y_test
        est_y_poly = decode_ohe(Ptest_list[i] @ w_list[i]).reshape(-1, 1) #transpose
        error_test_array.append(diff_count(actual_y_poly, est_y_poly))
    error_test_array = np.array(error_test_array)

    
    # return in this order
    return X_train, y_train, X_test, y_test, Ytr, Yts, Ptrain_list, Ptest_list, w_list, error_train_array, error_test_array

print(A2_A0258539M(42))