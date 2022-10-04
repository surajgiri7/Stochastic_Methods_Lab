#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: surajgiri
HW3 Problem 4
"""
import numpy as np
import matplotlib.pyplot as plt

def EUPayoff(s, x, type, position):
    zero = np.zeros(len(s))
    if type == "call" and position == "long":
        option = np.maximum(s-x, zero)
        return option
    elif type == "call" and position == "short":
        option = np.minimum(x-s, zero)
        return option
    elif type == "put" and position == "long":
        option = np.maximum(x-s, zero)
        return option
    elif type == "put" and position == "short":
        option = np.minimum(s-x, zero)
        return option
    else:
        print("Invalid Option Input")

if __name__ == '__main__':
    s = np.arange(30, 110)
    payoff_butterfly = EUPayoff(s=s, x=50, type='call', position='long') +\
                       EUPayoff(s=s, x=70, type='call', position='short') * 2 +\
                       EUPayoff(s=s, x=90, type='call', position='long') 
    plt.plot(s, payoff_butterfly)
    plt.xlabel('Stock Price [$]')
    plt.ylabel('Payoff [$]')
    plt.title('Butterfly Spread')
    plt.grid()
    plt.show()