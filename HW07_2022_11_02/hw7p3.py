"""
Author: Suraj Giri
HW7: Problem 3
"""

"""
Problem:
use Black Scholes formula and plot the Call Price C against
1. Stock Price S
2. interest rate r
3. volatility sigma
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# defining the function for Black Scholes formula
def Black_Scholes_Formula(S_0, K, T, r, sigma):
    d_1 = (np.log(S_0/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
    d_2 = d_1 - sigma*np.sqrt(T)
    C = S_0*norm.cdf(d_1) - K*np.exp(-r*T)*norm.cdf(d_2)
    return C

# main function
if __name__ == "__main__":
    # defining the parameters
    S_0 = 100
    K = np.arange(50, 150, 1)
    T = 1
    r = np.arange(0, 1, 0.01)
    sigma = np.arange(0, 1, 0.01)
    stock = np.arange(0,1, 0.01)
    # calculating the call price
    # C = Black_Scholes_Formula(S_0, K, T, r, sigma)
    # plotting the graph into 3 subplots
    fig, ax = plt.subplots(3, 1)
    # changing the size of the figure
    fig.set_size_inches(8, 8)

    # plotting the graph against Stock Price S
    ax[0].plot(stock, Black_Scholes_Formula(stock, K, T, 0.5, 0.5))
    ax[0].set_title("Call Price against Stock Price")
    ax[0].set_xlabel("Stock Price")
    ax[0].set_ylabel("Call Price")
    ax[0].grid()

    # plotting the graph against interest rate r
    ax[1].plot(r, Black_Scholes_Formula(S_0, 70, T, r, 0.5))
    ax[1].set_title("Call Price against Interest Rate")
    ax[1].set_xlabel("Interest Rate")
    ax[1].set_ylabel("Call Price")
    ax[1].grid()

    # plotting the graph against volatility sigma
    ax[2].plot(sigma, Black_Scholes_Formula(S_0, 70, T, 0.5, sigma))
    ax[2].set_title("Call Price against Volatility")
    ax[2].set_xlabel("Volatility")
    ax[2].set_ylabel("Call Price")
    ax[2].grid()
    
    plt.tight_layout()

    # saving the figure
    if not os.path.exists('plots'):
        os.makedirs('plots')
    # plt.savefig('plots/hw7p3.pdf')
    plt.show()