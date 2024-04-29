import numpy as np


# Please replace "StudentMatriculationNumber" with your actual matric number here and in the filename
def A3_A0254460J(learning_rate, num_iters):
    """
    Input type
    :learning_rate type: float
    :num_iters type: int

    Return type
    :a_out type: numpy array of length num_iters
    :f1_out type: numpy array of length num_iters
    :b_out type: numpy array of length num_iters
    :f2_out type: numpy array of length num_iters
    :c_out type: numpy array of length num_iters
    :d_out type: numpy array of length num_iters
    :f3_out type: numpy array of length num_iters
    """
    # your code goes here

    # initialisation
    a_out, f1_out, b_out, f2_out, c_out, d_out, f3_out = np.zeros(num_iters), np.zeros(num_iters), np.zeros(num_iters), np.zeros(num_iters), np.zeros(num_iters), np.zeros(num_iters), np.zeros(num_iters)
    a, b, c, d = 2.5, 0.6, 2, 3

    for i in range(num_iters):

        # (a) Gradient of f1 = 4a^3
        a = a - learning_rate * (4 * a**3)
        a_out[i] = a
        f1_out[i] = a**4

        # (b) Gradient of f2 = 2sin(b)cos(b)
        b = b - learning_rate * (2 * np.sin(b) * np.cos(b))
        b_out[i] = b
        f2_out[i] = (np.sin(b))**2
        
        # (c) Gradient of f3: wrt c is 5c^4 | wrt d is d^2*cos(d) + 2dsin(d)
        c = c - learning_rate * (5 * c**4)
        c_out[i] = c
        d = d - learning_rate * (d**2 * np.cos(d) + 2 * d * np.sin(d))
        d_out[i] = d
        f3_out[i] = c**5 + d**2 * np.sin(d)


    # return in this order
    return a_out, f1_out, b_out, f2_out, c_out, d_out, f3_out 

