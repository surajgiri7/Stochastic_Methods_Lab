# -*- coding: utf-8 -*-

"""
@author: surajgiri
HW4 Problem 2
"""

import numpy as np

def binomial_tree(payoff, n, rp, sigma, S, K, T):
    # Calculating the increase rate and decrease rate
    u = np.exp(sigma * np.sqrt(T / n))
    d = 1 / u

    # Discount the Payoffs by Backwards Induction
    dt = T / n
    p = (np.exp(rp * dt) - d) / (u - d)

    # Calculating a Stock Price Lattice for the Underlying Asset Price
    x, y = np.arange(0, n+1, 1), np.arange(n, -1, -1)
    Stock_Price_Lattice = S * u ** x * d ** y

    # Calculating the Payoff at the Expiry
    poff = payoff(Stock_Price_Lattice, K=K)

    # Backward induction with a single for loop
    for i in np.arange(n, 0, -1):
        v_u = poff[1:i+1] # ïncreasing the values in the y axis for each index
        v_d = poff[0:i]   # keeping the values of the x-axis constant for each index
        poff = np.exp(-rp * dt) * (p * v_u + (1-p) * v_d)
        
    out = poff[0]
    return out


def payoff(S, K):
    payoff = np.maximum(0, S - K)
    return payoff


if __name__ == '__main__':
    K = 0.7
    rp  = 0.02
    sigma = 0.5
    T = 1
    S = 1
    n = 1000
    european_call = binomial_tree(payoff=payoff, n=n, rp=rp, sigma=sigma, S=S, K=K, T=T)
    print("The Price of European call option =  ", european_call)