"""
Author: Suraj Giri
Program: hw08_p2.py
"""

"""
Look up stock option quotes for European or American call options on the stock of a
major corporation (make sure you choose a non-dividend paying stock). 
Plot the implied volatility (i.e., the parameter σ(sigma) given the market value of the option) vs. the strike price,
while the time to maturity is fixed. (The applicable interest rate is the spot rate for zero
coupon bonds of the same maturity.) Here it would be easiest to use the Black-Scholes
formula for the option pricing. Make sure to mark the current stock price and some
historical volatility (which you have to look up) in the plot, and to label the plot nicely.

Ideas:
Could use Brentq
T must be fixed

"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as si
import pandas as pd
from scipy.stats import norm
import random
import yfinance as yf # for getting the stock data from Yahoo Finance

# define the function for Black-Scholes formula copied from hw6
def Black_Scholes(S, K, T, sigma, rp):
    x = (np.log(S/K) + (rp + sigma**2/2) * T) / (sigma * np.sqrt(T))
    Black_Scholes = S * norm.cdf(x) - K * np.exp(-rp * T) * norm.cdf(x - sigma * np.sqrt(T))
    return Black_Scholes
        
# define the function for implied volatility
def Implied_Volatility(S, K, T, rp, market_price):
    sigma = 0.5
    for i in range(100):
        price = Black_Scholes(S, K, T, sigma, rp)
        vega = S * norm.cdf((np.log(S/K) + (rp + sigma**2/2) * T) / (sigma * np.sqrt(T))) * np.sqrt(T)
        sigma = sigma - ((price - market_price)/vega)
    return sigma

# main function
if __name__ == "__main__":
    T = 1 # time to maturity
    r = 0.01 # risk free interest rate
    # get the stock data from Yahoo Finance
    stock = yf.Ticker("AAPL") # Getting the stock data for Apple

    # Random expiration date
    # exp = stock.options[random.randint(0, len(stock.options)-1)] 

    """
    setting random expiration date sometimes used Sundays for the Expiration date
    which provided some weird graohs so I decided to use a specific non-Sunday date
    """
    # setting the expiration date to a 2023-02-17 [You can change the date here!]
    exp = "2023-02-17"

    # get the option chain for the expiration date
    opt = stock.option_chain(exp)
    # get the call data
    call = opt.calls
    # get the put data
    put = opt.puts

    # get the current stock price
    S = stock.history(period="1d")["Close"][0]
    # get the strike price
    K = call["strike"]
    # get the market price
    market_price = call["lastPrice"]
    # get the implied volatility
    sigma = np.zeros(len(K))
    for i in range(len(K)):
        sigma[i] = Implied_Volatility(S, K[i], T, r, market_price[i])

    # print("==" * 20)
    # print(sigma)
    # print("==" * 20)

    # plot the implied volatility for each strike price
    plt.plot(K, sigma, label="Implied Volatility", color="red", marker="o")
    plt.xlabel("Strike Price")
    plt.ylabel("Implied Volatility")
    plt.title("Implied Volatility vs. Strike Price")
    plt.plot(" ", " ", label="Date Used: " + str(exp))
    plt.legend()

    if not os.path.exists("plots"):
        os.makedirs("plots")
    # plt.savefig("plots/hw08_p2.pdf")
    plt.show()