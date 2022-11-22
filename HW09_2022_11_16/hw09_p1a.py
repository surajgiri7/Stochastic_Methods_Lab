"""
Author: Suraj Giri
Program: hw09_p1a.py
"""

"""
Problem:
Use Ornstein-Uhlenbeck stochastic process
    dX_t = my * (1 - c * ln X_t) * X_t * dt + sigma * X_t * dW_t
mu, sigma > 0 and 0 ≤ c ≤ 1

Generate an ensemble of paths of the process (1) on the interval [0,1], and plot the
empirical mean and standard deviation together with 10 sample paths. Produce at
least two plots with reasonable parameters such that the influence of the parameter
c as compared to geometric Brownian motion becomes visible. Briefly describe what
happens when c is near zero or c is near 1.
"""

"""
Theorectical answer:
When c is near 0, Exponential Ornstein Uhlenbeck process is the same with GBM.
When c is near 1, Exponential Ornstein Uhlenbeck process has smaller range of s, 
but the similar drift and variance behavior.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
    

# defining the function for Ornstein-Uhlenbeck stochastic process
def Ornstein_Uhlenbeck(M, N, T, mu, sigma, c):
    # M = 1000 at least
    dt = T/N
    X_0 = 1
    dW = np.random.normal(scale=np.sqrt(dt), size=(M, N-1))
    dX = mu * (1 - c * np.log(X_0)) * X_0 * dt + sigma * X_0 * dW
    X = np.insert(dX, 0, X_0, axis=1)
    X = np.cumsum(X, axis=1)
    return X

# defining the function for mean and standard deviation
def mean_std_deviate(array):
    mean = np.mean(array, axis=0)
    std_deviate = np.std(array, axis=0)
    return mean, std_deviate

# main function
if __name__ == "__main__":
    # mu, sigma > 0
    mu = 0.7
    sigma = 0.4

    c_list = [0.1, 0.9] # 0 ≤ c ≤ 1
    N = 500 # time steps
    M = 10 # number of sample paths
    T = 1

    t = np.linspace(0, T, N) # time steps

    # plotting both graphs in the same plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Ensemble of paths of Ornstein-Uhlenbeck stochastic process")
    for c in range(len(c_list)):
        # generating the ensemble of paths of the process (1) on the interval [0,1]
        X = Ornstein_Uhlenbeck(M, N, T, mu, sigma, c_list[c])
        # mean and the standard deviation for the ensemble of paths
        mean, std_deviate = mean_std_deviate(X)

        # plotting the empirical mean and standard deviation together with 10 sample paths
        ax[c].plot(t, X.T, color='red', alpha=0.5 )
        ax[c].plot(t, mean, color='blue', label='mean')
        ax[c].plot(t, mean+std_deviate, color='teal', label='standard deviation')
        ax[c].plot(t, mean-std_deviate, color='teal')
        ax[c].set_title("For c = {}".format(c_list[c]))
        ax[c].set_xlabel("Time")
        ax[c].set_ylabel("Value of Ornstein-Uhlenbeck stochastic process")
        ax[c].plot(np.NaN, np.NaN, '-', color='red', label='Sample Paths')
        ax[c].grid()
        ax[c].legend()

    if not os.path.exists("./plots"):
        os.makedirs("plots")
    plt.savefig("plots/hw09_p1a.pdf")
    plt.show()