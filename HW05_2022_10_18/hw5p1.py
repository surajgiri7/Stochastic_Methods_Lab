"""
Author: Suraj Giri
Date: 25/10/2022
Homework 5, Problem 1
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from scipy.stats import norm

# defining the function for binomial tree
def binomial_tree(payoff, n, rp, sigma, S, K, T, opt):
    # Calculating the increase rate and decrease rate
    u = np.exp(sigma * np.sqrt(T / n))
    d = 1 / u

    # Discount the Payoffs by Backwards Induction
    dt = T / n
    p = (np.exp(rp * dt) - d) / (u - d)

    # Calculating a Stock Price Lattice for the Underlying Asset Price
    x, y = np.arange(0, n+1, 1), np.arange(n, -1, -1)
    Stock_Price_Lattice = S * (u ** x) * (d ** y)

    # Calculating the Payoff at the Expiry
    poff = payoff(S=Stock_Price_Lattice, K=K, opt=opt)

    # Backward induction with a single for loop
    for i in np.arange(n, 0, -1):
        v_u = poff[1:i+1] # ïncreasing the values in the y axis for each index
        v_d = poff[0:i]   # keeping the values of the x-axis constant for each index
        poff = np.exp(-rp * dt) * (p * v_u + (1-p) * v_d)
        
    out = poff[0]
    return out

# defining the function for payoff
def payoff(S, K, opt):
    if opt == 'C':
        return np.maximum(0, S-K)
    elif opt == 'P':
        return np.maximum(0, K-S)

# defining x for Black Scholes:
def x(S, K, T, sigma, rp):
    return (np.log(S/K) + (rp + sigma**2/2) * T) / (sigma * np.sqrt(T))

# defining the functiuon for Black Scholes
def black_scholes(S, K, T, sigma, rp, opt):
    if opt == 'C':
        return S * norm.cdf(x(S, K, T, sigma, rp)) - K * np.exp(-rp * T) * norm.cdf(x(S, K, T, sigma, rp) - sigma * np.sqrt(T))
    elif opt == 'P':
        return K * np.exp(-rp * T) * norm.cdf(-x(S, K, T, sigma, rp) + sigma * np.sqrt(T)) - S * norm.cdf(-x(S, K, T, sigma, rp))

# Main
if __name__ == "__main__":
    # Parameters
    n = 1000
    rp = 0.03
    sigma = 0.5
    S = 1
    K = 1.2
    T = 1
    opt = 'C'
    n = np.arange(1, n+1)

    # Calculating the price of the option
    binomTree = [binomial_tree(payoff, i, rp, sigma, S, K, T, opt) for i in n]
    blackSchole = black_scholes(S, K, T, sigma, rp, opt)
    allBlackSchole = np.empty(n.size)
    allBlackSchole.fill(blackSchole)

    relError = np.absolute(allBlackSchole - binomTree)


    plt.loglog(n, relError)
    plt.grid(linestyle='--')
    plt.savefig("hw5p1.pdf")
    plt.show()

"""
Comment:
No We do not obtain a straight line. 
"""