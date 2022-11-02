"""
Author: Suraj Giri
hw6 problem 2 b
"""

"""
to plot: mean and standard deviation which underlie the binomial tree model
N = 600
r = mu
annualized volatility sigma
No of sample paths = 6
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pandas import array

# defining the function for the Binomial Tree Model

def Binomial_Tree_Model(M, N, r_p, sigma, S, T, seed):
    # using random seed
    np.random.seed(seed)
    # defining the time step
    dt = T/N
    # defining the up and down factors
    u = np.exp(sigma * np.sqrt(dt))
    d = 1/u
    # defining the probability of up and down
    p = (np.exp(r_p * dt) - d)/(u - d)

    dw = np.random.choice([u,d], size = (M, N-1), p = [p, 1-p]) # generating the random numbers
    dw = np.cumprod(dw, axis = 1) # calculating the cumulative product
    dw = np.insert(dw, 0, 1, axis = 1) # inserting the initial value
    return dw

# defining the function for drawing the subplots
def sub_plots(array, mean, std_deviate, steps, mu, sigma):
    plt.plot(steps, mean, label = 'empirical mean', color = 'orange')
    plt.plot(steps, mean - std_deviate, label = 'empirical standard deviation', color = 'green')
    plt.plot(steps, mean + std_deviate,color='green')

    for i in range(6):
        plt.plot(steps, array[i, :], c= 'b', label = 'Sample Binomial Path' if i==0 else "")

    plt.xlabel('time')
    plt.ylabel('value')

    plt.title("Ensemble of Binomial Paths with $\mu = {}, \sigma = {}$".format(mu, sigma))
    # plt.legend()
    return None

# main function
if __name__ == "__main__":
    mu = 0.7
    sigma = 0.4
    N = 600
    M = 1000
    T = 1
    S = 1
    r_p = mu

    steps = np.linspace(0, 1, N)
    seeds = range(0, N)

    # array = np.asarray([Binomial_Tree_Model(M, N, r_p, sigma, S, T, seed) for seed in seeds])
    array = Binomial_Tree_Model(M, N, r_p, sigma, S, T, seeds)
    mean = np.mean(array, axis = 0)
    std_deviate = np.std(array, axis = 0)
    sub_plots(array, mean, std_deviate, steps, mu, sigma)

    # sub folder to save the plots
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.legend()
    plt.savefig('plots/hw6p2b.pdf')
    plt.show()