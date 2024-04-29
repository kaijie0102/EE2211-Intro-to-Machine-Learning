import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from numpy.linalg import inv
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder


# Please replace "MatricNumber" with your actual matric number here and in the filename
def A2_A0254460J(N):
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

    RIDGE_REGRESSION_CONSTANT = 0.0001

    """
    Part A: Import data
    """
    # Load the Iris dataset
    iris = load_iris() # classic dataset for ML classification
    X, y = iris.data, iris.target # X: features of iris flowers  y: species label

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=N)

    # if N = 42(random): (((45, 4), (45,)), ((105, 4), (105,)))
    # print((X_train.shape, y_train.shape), (X_test.shape, y_test.shape))

    """
    Part B: Encoding 
    """
    # Initialize the OneHotEncoder
    one_hot_encoder = OneHotEncoder(sparse=False)

    # Fit and transform training sets - encoder will learn encoding from training data
    Ytr = one_hot_encoder.fit_transform(y_train.reshape(-1, 1)) # encode the transpose

    # transform testing set: Not using fit_transform to ensure that encoder only learns from train data
    Yts = one_hot_encoder.transform(y_test.reshape(-1, 1))
    

    """
    Part C: Perform polynomial regression for orders 1 to 8 to see which models fit best
    """
    Ptrain_list = []
    Ptest_list = []
    w_list = []
    error_train_array = []
    error_test_array = []

    for order in range(8):
        polynomial = PolynomialFeatures(order + 1) # polynomial
        Ptrain_list.append(polynomial.fit_transform(X_train))
        Ptest_list.append(polynomial.transform(X_test))

    # List of estimated regression coefficient from order 1 to 8
    for P in Ptrain_list:
        m = len(P) # no_of_rows
        d = len(P[0]) # no_of_cols
        
        # L2 regularisation: add penalty to loss function: to reduce overfitting
        if (m > d): # apply primal form (left inverse)
            w = inv(P.T @ P + RIDGE_REGRESSION_CONSTANT * np.eye(d)) @ P.T @ Ytr
        else: # apply dual form (right inverse)
            w = P.T @ inv(P @ P.T + RIDGE_REGRESSION_CONSTANT * np.eye(m)) @ Ytr
        w_list.append(w)

    for i in range(8):
        actual_y_poly = y_train
        # inverse_transform is to decode
        est_y_poly = one_hot_encoder.inverse_transform(Ptrain_list[i] @ w_list[i]).reshape(-1, 1) #transpose
        error_train_array.append(np.sum(actual_y_poly != est_y_poly))  # Counting the number of elements that don't match

        actual_y_poly = y_test
        est_y_poly = one_hot_encoder.inverse_transform(Ptest_list[i] @ w_list[i]).reshape(-1, 1) #transpose
        error_test_array.append(np.sum(actual_y_poly != est_y_poly))

        
    error_train_array = np.array(error_train_array)
    error_test_array = np.array(error_test_array)




    # return in this order
    return X_train, y_train, X_test, y_test, Ytr, Yts, Ptrain_list, Ptest_list, w_list, error_train_array, error_test_array

A2_A0254460J(42)