"""
Author: Suraj Giri
Program: hw11.py
"""

"""
Suppose a time series data set which beahaves like a Geometric Brownian Motion (GBM) 
with oparameters mu and sigma.

log returns:
r_i = ln(S_(i+1)) - ln(S_i) for i = 0, 1, ..., N-1
"""

"""
Problem a:
Generate GBM path on time interval [0,1] with fixed mu = 0.3 and sigma = 0.5
number of sub intervals is N = 2^k

Estimate sigma_hat and mu_hat on a coarsed dat set which only needs 2^i th data point

Plot sigma_hat and mu_hat vs the log of the number of sample points (semilogx)
Do estimated values converge to the true value of the model?
"""

import numpy as np
import matplotlib.pyplot as plt

# defining the function for geometric brownian motion on the interval [0,1]
def Geometric_Brownian_Motion(mu, sigma, N):
    dt = 1/N
    S_0 = 1
    ds = ((mu - 0.5 * sigma**2) * dt + sigma * np.random.normal(scale=np.sqrt(dt), size=N))
    dw = np.insert(np.exp(np.cumsum(ds)), 0, S_0)
    return dw

# defining the function to calculate the log returns
def log_returns(GBM,i):
    # i lai 0 deki k samma vectorize garne function maa vector pass garne ho

    dt = 1/2**i
    N = len(GBM)
    array = GBM[::int(N/2**i)]
    log_returns = np.diff(np.log(array))
    sigma_hat = np.std(log_returns)/np.sqrt(dt)
    mu_hat = np.mean(log_returns)/(dt) + 0.5 * sigma_hat**2
    return sigma_hat, mu_hat

# main function
if __name__ == "__main__":
    # sampling the points from the geometric brownian motion
    N = 2**10
    mu = 0.3
    sigma = 0.5
    a = []
    GBM = Geometric_Brownian_Motion(mu, sigma, N)
    for i in range(1,11):
            a.append(log_returns(GBM,i))

    plt.plot(2**np.arange(1,11),a)
    plt.xscale('log')
    plt.xlabel('Number of sample points')
    plt.ylabel('sigma_hat and mu_hat')
    plt.title('sigma_hat and mu_hat vs the log of the number of sample points')
    plt.legend(['sigma_hat','mu_hat'])
    plt.show()



