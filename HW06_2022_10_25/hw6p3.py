"""
Author: Suraj Giri
hw6 problem 3
"""

"""
Use geometric Brownian Motion
mu = 0.3
sigma = 0.7
Monte-Carlo valuation of European call option
Strike Price K = 0.8
Maturity T = 1
risk free rate of interest r = mu
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import stats 

# defining the function for payoff
def Payoff(S, K):
    payoff = np.maximum(0, S - K)
    return payoff

# defining the function for geometric brownian motion
def Geometric_Brownian_Motion(N, mu, sigma, m):
    dt = 1/N
    S_0 = 1
    ds = ((mu - 0.5 * sigma**2) * dt + sigma * np.random.normal(scale=np.sqrt(dt), size=(m, N-1)))
    dw = np.insert(np.exp(np.cumsum(ds, axis=1)), [0], [S_0], axis=1)
    return dw

# defining the function for Black-Scholes formula for European CALL option
def Black_Scholes(S, K, T, sigma, rp):
    x = (np.log(S/K) + (rp + sigma**2/2) * T) / (sigma * np.sqrt(T))
    Black_Scholes = S * norm.cdf(x) - K * np.exp(-rp * T) * norm.cdf(x - sigma * np.sqrt(T))
    return Black_Scholes

# main function
if __name__ == "__main__":
    # defining the parameters
    M = 2**np.arange(1,11,1)
    N = 500
    mu = 0.3
    sigma = 0.7
    K = 0.8
    T = 1
    rp = mu
    S = 1

    BS = Black_Scholes(S, K, T, sigma, rp)

    # Absolute Error between price of European call option as Monte-Carlo valuation of Geometric Brownian Motion and price from Black-Scholes formula
    error = np.zeros(len(M))
    for i in range(len(M)):
        m = M[i]
        GBM = Geometric_Brownian_Motion(N, mu, sigma, m)
        payoff = Payoff(GBM[:, -1], K)
        price = np.mean(payoff) * np.exp(-rp * T)
        error[i] = np.abs(price - BS)

    # convergence rate
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(M), np.log(error))
    print('Convergence rate: ', slope)

    # plotting the graph
    plt.figure()
    plt.loglog()
    plt.plot(M, error, label = 'Absolute Error')
    plt.xlabel('Number of Paths')
    plt.ylabel('Absolute Error')
    plt.title('Absolute Error between price of European call option as Monte-Carlo valuation of \n' + 
        'Geometric Brownian Motion and price from Black-Scholes formula', fontsize=10)

    # plotting the slope of convergence rate
    plt.plot(M, np.exp(intercept) * M**slope, label = 'Convergence Rate')

    # printing the slope inside the legend in the graph
    plt.plot([], [], ' ', label="Slope of Convergence Rate = " + str(slope))

    plt.legend()

    # saving the figure
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig('plots/hw6p3.pdf')
    plt.show()

    """
    As the number of values of paths increases, the absolute error decreases,
    i.e, the convergence is bound to happen.
    """

