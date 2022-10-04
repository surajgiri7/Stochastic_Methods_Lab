#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: surajgiri
HW3 Problem 3
"""

""""
Payoff Function for the 
European Call Option
European Put Option
Strike Price 100 $
Plot Graph
"""

import numpy as np
import pandas as pd
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

if __name__ == "__main__":
    s = np.arange(0, 150, 0.1)
    x = 100
    types = ["call","put"]
    positions = ["long", "short"]

    i = 0
    for type in types:
        for position in positions:
            i = i + 1
            plt.subplots()
            plt.plot(s, EUPayoff(s, x, type, position))
            plt.ylabel('Payoff')
            plt.xlabel('Price')
            plt.grid()
            plt.title(position + ' ' + type + ' option')
    plt.legend()
    plt.show()