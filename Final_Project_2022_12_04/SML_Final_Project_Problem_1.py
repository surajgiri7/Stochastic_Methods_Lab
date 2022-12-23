"""
Author: Suraj Giri
Date: 2022-12-22
Program: Stochastic Methods Lab final Project Problem 1
"""

"""
Problem 1:
Choose a stock for which you can find recent time series data as well as quotes on
European call or put options for different parameters. (For the option quotes, choose at
least two reasonably different maturities, and 30 different strike prices for each maturity.)
"""

"""
a. Analyze the time series: How good is the assumption of normally and independently
distributed log-returns? Estimate the volatility of the stock. Comment on the results.
"""

"""
a. Solution:
Here from all the observations, we can see that the log returns are normally distributed.
The mean of the log returns is 0.0002 and the standard deviation is 0.016. The volatility
of the stock is also around 0.33. The autocorrelation of the log returns is also very low. So, we can
say that the log returns are normally distributed and independent of each other.
We can also see that the stock price and log returns are not correlated with each other.
We can also say that the Stock Price is independent on the historical data.
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import datetime as dt
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import norm
import math
import statsmodels.api as sm
import os

# Importing the dataset
start = dt.datetime(2017, 12, 12)
end = dt.datetime(2022, 12, 22)
df = yf.download('META', start=start, end=end) 
df = df.dropna()
stock_price = df['Close']
print(stock_price)

# calculating the log returns
log_returns = np.log(stock_price / stock_price.shift(1))
log_returns = log_returns.dropna()

# creating the folder to save the plots
if not os.path.exists("plots"):
        os.makedirs("plots")

# visualizing the stock price and log returns
plt.figure(figsize=(10, 5))
plt.title('Stock Price and Log Returns of the Stock')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.plot(stock_price, color='blue', label='Stock Price')
plt.plot(log_returns, color='red', label='Log Returns')
plt.legend()
plt.savefig("plots/Stock_and_Log_Return_Visualization.pdf")
plt.show()

# visualizing the log returns
plt.figure(figsize=(10, 5))
plt.plot(log_returns)
plt.title('Log Returns')
plt.xlabel('Time')
plt.ylabel('Log Returns')
plt.savefig("plots/Log_Returns_Visualization.pdf")
plt.show()

# calculating the mean and standard deviation of the log returns
mean = log_returns.mean()
std = log_returns.std()

# visualizing the normal distribution of the log returns
plt.figure(figsize=(10, 5))
plt.hist(log_returns, bins=50, density=True)
plt.title('Normal Distribution of Log Returns')
plt.xlabel('Log Returns')
plt.ylabel('Frequency')
plt.plot(np.linspace(-0.1, 0.1, 100), 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (1 / std * (np.linspace(-0.1, 0.1, 100) - mean)) ** 2), color='red')
plt.savefig("plots/Normal_Distribution_of_Log_Returns_Visualization.pdf")
plt.show()

# visualizing the autocorrelation of the log returns
plot_acf(log_returns, lags=50)
plt.title('Autocorrelation of Log Returns')
plt.xlabel('Lags')
plt.ylabel('Autocorrelation')
plt.savefig("plots/Autocorrelation_of_Log_Returns_Visualization.pdf")
plt.show()

# calculating the volatility of the stock
volatility = sigma = std * np.sqrt(math.log(2,len(log_returns)))
print("|||||" * 6)
print('The volatility of the stock is: ', volatility)
print("|||||" * 6)

# plotting the QQ plot of the log returns
sm.qqplot(log_returns, line='s')
plt.title('QQ Plot of Log Returns')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.legend()
plt.savefig("plots/QQ_Plot_of_Log_Returns_Visualization.pdf")
plt.show()

"""
The volatility of the stock is:  0.00867442848387263
"""

"""
b. Determine a suitable risk-free interest rate for pricing the options for which you found
quotes. (Note: A very rough interpolation, if necessary, will suffice.)
"""

"""
b. Solution:
------------
Here, we are interpolating the risk-free interest rate from the website
http://www.worldgovernmentbonds.com/country/united-states/#:~:text=Yield%20Curve%20is%20inverted%20in,probability%20of%20default%20is%200.42%25.
"""
# interpolating the risk-free interest rate from 
# http://www.worldgovernmentbonds.com/country/united-states/#:~:text=Yield%20Curve%20is%20inverted%20in,probability%20of%20default%20is%200.42%25.
date = [1/12, 2/12, 3/12, 4/12, 5/12, 6/12, 7/12, 8/12, 9/12]
r = [3.7, 4.05, 4.33, 4.67, 4.65, 4.24, 4.01, 3.78, 3.76]
r = np.interp(56/365, date, r)
r = r / 100
print("|||||" * 6)
print('The risk-free interest rate is: ', r)
print("|||||" * 6)


"""
The interpolated risk-free interest rate is:  0.039943835616438356
"""

"""
c. Price the options with an algorithm of your choice for all maturities and strike prices
for which you can find data to compare, and compare with the data. Discuss your
result, and possibly explain deviations.
"""

"""
c. Solution:
------------
Here, we saw that the Black-Scholes model is a good approximation for the option price when time to maturity is
relatively short. However, when time to maturity is longer, the difference between the option price derived from
Black-Scholes model and the quotes becomes more obvious. This is because the Black-Scholes model assumes that
the stock price follows a geometric Brownian motion. When we zoom in enough, we can see that the option price
derived from Black-Scholes is higher than quotes. 
"""
# calculating the Black-Scholes option price
def black_scholes_formula(S, K, T, r, sigma, option_type):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        option_price = K * np.exp(-r * T) * norm.cdf
    return option_price

# defining the payoff function
def payoff(S, K, option_type):
    if option_type == 'call':
        return max(S - K, 0)
    elif option_type == 'put':
        return max(K - S, 0)

# defining the funtion for outputting the strike price, calculated option price and actual option price
def output_c(T, exp_date):
    option = yf.Ticker("META")
    exp_date = exp_date
    option_type = 'call'
    option_data = option.option_chain(exp_date)
    
    # getting the call and put option data
    call_data = option_data.calls
    put_data = option_data.puts

    #n getting the strke price
    strike_price = call_data['strike']
    strike_price = strike_price.to_numpy()
    
    # getting the actual option price
    call_price_actual = call_data['lastPrice']
    call_price_actual = call_price_actual.to_numpy()

    # getting the option price for each strike price for each expiration date
    option_price = []
    for i in range(len(strike_price)):
        option_price.append(black_scholes_formula(stock_price[-1], strike_price[i], T, r, sigma, option_type))

    return strike_price, option_price, call_price_actual




# getting the option price for each strike price for each expiration date
option_price1 = []
option_price2 = []

strike_price1= []
strike_price2 = []


# exp_dates = ["2023-02-17", "2023-04-21"] 
# T = [56/365, 120/365]

exp_dates = ["2023-02-17", "2023-06-16"] 
T = [56/365, 180/365]

strike_price1, option_price1, call_price_actual1 = output_c(T[0], exp_dates[0])
strike_price2, option_price2, call_price_actual2 = output_c(T[1], exp_dates[1])

print("|||||" * 6)
# printing the calculated option price and actual option price for each strike price in a table
print("Option price for maturity on " + exp_dates[0])
print('Strike Price \t Calculated Option Price \t Actual Option Price')
for i in range(len(strike_price1)):
    print(f'{strike_price1[i]}\t\t{option_price1[i]}\t\t\t{call_price_actual1[i]}')
print("|||||" * 6)
# printing the calculated option price and actual option price for each strike price in a table
print("Option price for maturity on " + exp_dates[1])
print('Strike Price \t Calculated Option Price \t Actual Option Price')
for i in range(len(strike_price1)):
    print(f'{strike_price2[i]}\t\t{option_price2[i]}\t\t\t{call_price_actual2[i]}')
print("|||||" * 6)


# plotting the actual option price and the calculated option price for each expiration date in a subplot
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(strike_price1, call_price_actual1, label='actual option price')
plt.plot(strike_price1, option_price1, label='calculated option price')
plt.xlabel('strike price')
plt.ylabel('option price')
plt.title('Option price for ' + exp_dates[0])
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(strike_price2, call_price_actual2, label='actual option price')
plt.plot(strike_price2, option_price2, label='calculated option price')
plt.xlabel('strike price')
plt.ylabel('option price')
plt.title('Option price for ' + exp_dates[1])
plt.legend()
plt.savefig("plots/Actual_Option_Price_vs_Calculated_Option_Price_Visualization.pdf")
plt.show()




"""
d. List all methods for finding option prices that were discussed in class. In very brief
bullet points, list advantages and disadvantages that were discussed in class of the
different methods. (This is a theoretical exercise and should be written as comment
in the source code.)
"""

"""
d. Solution:
The methods for finding option prices that were discussed in class with their advantages and disadvantages are:
1. Black-Scholes formula
    Advantages:
        1. It is the most widely used method for finding option prices.
        2. It is easy to implement.
        3. It is easy to understand.
    Disadvantages:
        1. It is not suitable for options with long maturities.
        2. It is not suitable for options with high volatility.
        3. It is not suitable for options with high dividends.
2. Monte Carlo simulation
    Advantages:
        1. It is suitable for options with long maturities.
        2. It is suitable for options with high volatility.
        3. It is suitable for options with high dividends.
    Disadvantages:
        1. It is not easy to implement.
        2. It is not easy to understand.
3. Binomial tree
    Advantages:
        1. It is suitable for options with long maturities.
        2. It is suitable for options with high volatility.
        3. It is suitable for options with high dividends.
    Disadvantages:
        1. It is not easy to implement.
        2. It is not easy to understand.
4. Finite difference method
    Advantages:
        1. It is suitable for options with long maturities.
        2. It is suitable for options with high volatility.
        3. It is suitable for options with high dividends.
    Disadvantages:
        1. It is not easy to implement.
        2. It is not easy to understand.
5. Partial differential equation
    Advantages:
        1. It is suitable for options with long maturities.
        2. It is suitable for options with high volatility.
        3. It is suitable for options with high dividends.
    Disadvantages:
        1. It is not easy to implement.
        2. It is not easy to understand.
"""
