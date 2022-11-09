"""
Author : Suraj Giri
HW7: Problem 2
"""

"""
Problem:
Confirn that sum(from i = 0 to n-1) of (delta_W_i)^2 converges to 
a constant as n -> infinity
W_t is a standard Brownian Motion
Find the constant.
"""

"""
Answer:
The value of the constant converges to the value of T.
"""

import numpy as np
import matplotlib.pyplot as plt

# defining the function for Standard Brownian Motion
def Standard_Brownian_Motion(N, T, m):
    # defining the time step
    dt = T/N
    S_0 = 0
    ds = np.random.normal(scale=np.sqrt(dt), size=(N-1)) # generating the random numbers
    dw = np.insert(ds, [0], [S_0]) # inserting the initial value
    dw = np.cumsum(dw) # cumulating the random numbers
    return dw

# defining the function for the sum of squares of delta_W
def sum_of_squares(w):
    # calculating the sum of squares of delta_W
    sum = 0
    for i in range(N-1):
        sum += (w[i+1] - w[i])**2
    return sum

# main function
if __name__ == "__main__":
    T = 20
    N = 1000
    m = 1
    
    # generating the brownian motion
    w = Standard_Brownian_Motion(N, T, m)
    # calculating the sum of squares of delta_W
    sum = sum_of_squares(w)
    print("When the value of T is " + str(T) + ", The value of the constant is: " + str(sum))
