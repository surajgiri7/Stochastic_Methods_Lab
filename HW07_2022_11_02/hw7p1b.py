"""
Author: Suraj Giri
HW7: Problem 1b
"""

"""
To find: strong order of Convergence 
strong order of convergence is: E[|S_N - S(T)|]  <= C * (delta_t)^p 
S_t denotes true geometric Brownian motion
S_N denotes the Euler Maruyama approximation at final time T
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
    for d in dt: # looping over the time step
        N = int(T/d)
        abs_error = np.zeros(N)
        for i in range(1000): # looping over the number of simulations
            # generating the brownian motion
            w = Standard_Brownian_Motion(N, T, 1)
            # calling the function for Euler Maruyama
            S_N, t_N = eulerMaruyama(w,mu, sigma, S_0, T, N)
            # calling the function for exact solution using Geometric Brownian Motion
            S_t, t_t = exactBrownianSolution(w,mu, sigma, S_0, T, N)

            # calculating the absolute error
            abs_error += np.abs(S_N - S_t)
        # expectation of the absolute error over the number of simulations
        E_abs_error.append(np.mean(abs_error))



    def f(x,c,m):
        return c*x**m

    a = curve_fit(f, dt, E_abs_error)[0]
    print(a)

    strong_order_of_convergence = a[1]
    print("The strong order of convergence is: ", strong_order_of_convergence)

    # plotting the absolute error
    plt.figure()
    plt.plot(dt, E_abs_error, label="Strong Order of Convergence")
    plt.loglog()
    plt.xlabel('dt')
    plt.ylabel('Order of Convergence')
    plt.title('Strong Order of Convergence')
    plt.grid()
    plt.plot([], [], ' ', label="Strong Order of Convergence: " + str(strong_order_of_convergence))
    plt.legend()

    # sub folder to save the plots
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig('plots/hw7p1b.pdf')
    plt.show()