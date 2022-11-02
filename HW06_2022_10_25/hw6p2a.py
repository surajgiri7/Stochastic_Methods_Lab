"""
Author: Suraj Giri
hw6 problem 2 a
"""

"""
Compute the ensemble of geometric Brownian paths 
M = 1000 at least
interval [0, T] = [0, 1]
N = 500 time steps
mu = 0.7
sigma = 0.4
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pandas import array

# defining the function for geometric brownian motion
def Geometric_Brownian_Motion(M, N, mu, sigma, seed):
    """
    Parameters
    ----------
    N : int
        number of time steps.
    mu : float
        drift parameter.
    sigma : float
        volatility parameter.
    seed : int
        seed value.

    Returns
    -------
    array : array
        array of geometric brownian motion paths.
    """
    np.random.seed(seed)
    dt = 1/N
    S_0 = 1
    ds = ((mu - 0.5 * sigma**2) * dt + sigma * np.random.normal(scale=np.sqrt(dt), size=(M, N-1)))
    dw = np.insert(np.exp(np.cumsum(ds, axis=1)), 0, S_0, axis=1)
    return dw

    # np.random.seed(seed)
    # dt = 1/N
    # S_0 = 1
    # ds = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.randn(N))
    # path = S_0 * np.cumprod(ds)
    # GBM = np.insert(path, 0, S_0)
    # return GBM
    
# defining the function to plot the empirical mean and standard deviation with 6 sample paths
def sub_plots(array, mean, std_deviate, steps, mu, sigma):
    """
    Parameters
    ----------
    array : array
        array of geometric brownian motion paths.
    mean : array
        array of mean of geometric brownian motion paths.
    std_deviate : array
        array of standard deviation of geometric brownian motion paths.
    steps : int
        number of time steps.
    mu : float
        drift parameter.
    sigma : float
        volatility parameter.

    Returns
    -------
    None.

    """
    plt.plot(steps, mean, label = 'empirical mean', color = 'red')
    plt.plot(steps, mean - std_deviate, label = 'empirical standard deviation', color = 'purple')
    plt.plot(steps, mean + std_deviate,color='purple')

    for i in range(6):
        plt.plot(steps, array[i, :], c= 'c', label = 'Sample Brownian Path' if i==0 else "")    

    plt.xlabel('time')
    plt.ylabel('value')

    plt.title("Ensemble of Geometric Brownian Paths with $\mu = {}, \sigma = {}$".format(mu, sigma))
    plt.legend()
    return None


# main function
if __name__ == "__main__":
    mu = 0.7
    sigma = 0.4
    N = 500
    M = 1000

    steps = np.linspace(0, 1, N)
    seeds = range(0, N)
    # array = np.asarray([Geometric_Brownian_Motion(N, mu, sigma, seed) for seed in seeds])
    array = Geometric_Brownian_Motion(M, N, mu, sigma, seeds)
    mean = np.mean(array, axis = 0)
    std_deviate = np.std(array, axis = 0)
    sub_plots(array, mean, std_deviate, steps, mu, sigma)

    # sub folder to save the plots
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.legend()
    plt.savefig('plots/hw6p2a.pdf')
    plt.show()