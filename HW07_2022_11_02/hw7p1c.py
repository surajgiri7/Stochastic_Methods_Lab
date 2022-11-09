"""
Author: Suraj Giri
HW7: Problem 1c
"""

"""
To find: weawk order of Convergence
weak order of convergence is: |E[S_N - S(T)]|  <= C * (delta_t)^p
"""

import os
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from hw7p1a import eulerMaruyama, exactBrownianSolution, Standard_Brownian_Motion
from scipy.optimize import curve_fit
    
# main function
if __name__ == "__main__":
    mu = 1.5
    sigma = 0.8
    S_0 = 1
    T = 1
    N = 1000

    dt = (1/2)**np.arange(5,16,1)
    E_abs_error = []
    W_abs_error = []
    for d in dt: # looping over the time step
        N = int(T/d)
        abs_error = np.zeros(N)
        weak_sum = 0
        for i in range(1000): # looping over the number of simulations
            # generating the brownian motion
            w = Standard_Brownian_Motion(N, T, 1)
            # calling the function for Euler Maruyama
            S_N, t_N = eulerMaruyama(w,mu, sigma, S_0, T, N)
            # calling the function for exact solution using Geometric Brownian Motion
            S_t, t_t = exactBrownianSolution(w,mu, sigma, S_0, T, N)

            # calculating the absolute error
            abs_error += np.abs(S_N - S_t)

            # calculating the weak error
            weak_sum += np.mean(S_N) - np.mean(S_t) 

        # expectation of the absolute strong error over the number of simulations
        E_abs_error.append(np.mean(abs_error))

        # expectation of the sum of the exact solution
        W_abs_error.append(abs(weak_sum))



    # function to fit the data
    def f(x,c,m): 
        return c*x**m

    a = curve_fit(f, dt, W_abs_error)[0]
    print(a)

    b = curve_fit(f, dt, E_abs_error)[0]
    print(b)

    weak_order_of_convergence = a[1]
    strong_order_of_convergence = b[1]

    print("Weak Order of Convergence", weak_order_of_convergence)
    print("Strong Order of Convergence", strong_order_of_convergence)

    # plotting the absolute errors
    plt.plot(dt,E_abs_error, label = 'Strong Order of Convergence')
    plt.plot(dt,W_abs_error, label = 'weak order of convergence')
    plt.loglog()
    plt.xlabel('dt')
    plt.ylabel("Order of Convergence")
    plt.title("Weak and Strong Order of Convergence")
    plt.grid()
    plt.plot([], [], ' ', label="Weak Order of Convergence: " + str(weak_order_of_convergence))
    plt.plot([], [], ' ', label="Strong Order of Convergence: " + str(strong_order_of_convergence))
    plt.legend()

    # sub folder to save the plots
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig('plots/hw7p1c.pdf')
    plt.show()