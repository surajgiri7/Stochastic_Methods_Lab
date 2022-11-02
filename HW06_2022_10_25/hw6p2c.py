"""
Author: Suraj Giri
hw6 problem 2 c
"""

"""
Plot both 2a and 2b in the same figure
see the difference between the two models
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import hw6p2a as p2a
import hw6p2b as p2b

# main
if __name__ == "__main__":
    # initializing the parameters
    N = 500
    mu = 0.7
    sigma = 0.4
    M = 1000
    S = 1
    T = 1
    r_p = mu
    seeds = range(0, N)
    steps = np.linspace(0, 1, N)

    # calling the function for geometric brownian motion
    GBM = p2a.Geometric_Brownian_Motion(M, N, mu, sigma, seeds)
    mean_GBM = np.mean(GBM, axis = 0)
    std_deviate_GBM = np.std(GBM, axis = 0)

    # calling the function for binomial tree
    BTM = p2b.Binomial_Tree_Model(M, N, r_p, sigma, S, T, seeds)
    mean_BTM = np.mean(BTM, axis = 0)
    std_deviate_BTM = np.std(BTM, axis = 0)

    # plotting the subplots
    p2a.sub_plots(GBM, mean_GBM, std_deviate_GBM, steps, mu, sigma)
    p2b.sub_plots(BTM, mean_BTM, std_deviate_BTM, steps, mu, sigma)
    
    # saving the figure
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig('plots/hw6p2c.pdf')
    plt.show()

"""
Description
From fig in problem 2 c,  Geometric Brownian Motion 
and Binomial Tree produce similar results. 
Their means and standard deviations overlap.
Look at details, GBM looks a bit smoother.
"""