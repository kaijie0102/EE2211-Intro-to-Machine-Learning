import numpy as np

# Please replace "StudentMatriculationNumber" with your actual matric number here and in the filename
def A3_A0254482B(learning_rate, num_iters):
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

    # variables
    a_out, f1_out, b_out, f2_out, c_out, d_out, f3_out = np.zeros(num_iters), np.zeros(num_iters), np.zeros(num_iters), np.zeros(num_iters), np.zeros(num_iters), np.zeros(num_iters), np.zeros(num_iters)
    a, b, c, d = 1.5, 0.3, 1, 2

    for i in range(num_iters):

        # (a) Gradient is 5a^4
        a = a - learning_rate * (5 * a**4)
        a_out[i] = a
        f1_out[i] = a**5

        # (b) Gradient is 2sin(x)cos(x)
        b = b - learning_rate * (2 * np.sin(b) * np.cos(b))
        b_out[i] = b
        f2_out[i] = np.sin(b)**2
        
        # (c) Gradient wrt c is 3c^2. Gradient wrt d is (x^2) cosx + 2x sinx
        c = c - learning_rate * (3 * c**2)
        c_out[i] = c
        d = d - learning_rate * (d**2 * np.cos(d) + 2 * d * np.sin(d))
        d_out[i] = d
        f3_out[i] = c**3 + d**2 * np.sin(d)

    # return in this order
    return a_out, f1_out, b_out, f2_out, c_out, d_out, f3_out 


print(A3_A0254482B(0.01, 10))