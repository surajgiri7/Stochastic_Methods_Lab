"""
Author: Suraj Giri
Program: hw08_p3.py
"""

"""
W_t -> Standard Brownian Motion
Startin point: W_0 = 0
let Brownian Motion crosses the value -1 or 1 
i.e, the time at which interval (-1,1) is left is hitting time
Expectation of hitting time = ?
Variation of hitting time = ?
"""

import matplotlib.pyplot as plt
import numpy as np


# define the function for Standard Brownian Motion
def Standard_Brownian_Motion(T, N):
    # defining the time step
    dt = T/N
    S_0 = 0
    ds = np.random.normal(scale=np.sqrt(dt), size=(N-1)) # generating the random numbers
    dw = np.insert(ds, [0], [S_0]) # inserting the initial value
    dw = np.cumsum(dw) # cumulating the random numbers
    return dw


# main function
if __name__ == "__main__":
    # define the parameters
    T = 10
    N = 1000
    n = 1000

    # empty hitting time array
    HT = np.zeros(n)

    # loop for generating the Brownian Motion Array
    for i in range(n):
        BM = Standard_Brownian_Motion(T, N)
        # loop for generating the Hitting Time Array for each Brownian Motion
        for j in range(N):
            if BM[j] < -1 or BM[j] > 1:
                HT[i] = j * (T/N)
                break       # break the loop if the condition is satisfied 
        
    # removing zeros from the array HT
    HT = HT[HT != 0.]
    # print(HT)
    print("==" * 20)

    # print the expectation of hitting time
    print("The Expectation of Hitting Time is: ", np.mean(HT))
    # print the variation of Hitting Time
    print("The Variance of Hitting Time is: ", np.var(HT))
    print("==" * 20)