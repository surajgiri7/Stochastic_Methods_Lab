#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: surajgiri
HW2 Problem 5
"""
import numpy as np
import matplotlib.pyplot as plt

def volatility(r,c, m, n, F):
    # using the formula of volatility form the book
    C = F * c/m
    y = r/ m
    numerator = (C/y)*n - (C/y**2)*((1+y)**(n+1) - (1+y)) - n*F 
    denominator = (C/y)*((1+y)**(n+1) - (1.0+y)) + F*(1+y)
    out =  (-1) *numerator/denominator
    return out

if __name__ == "__main__":
    C = [0.02, 0.06, 0.12]
    r = 0.06
    m = 2
    F = 1000

    n_years = np.arange(0, 101, 1)
    V = []
    # max_val=0

    for c in C:
        P = [volatility(r, c, m, n, F) for n in n_years]
        plt.plot(n_years, P, label='c = {:.0f}%'.format(c * 100))
        # max_val = max(max_val, max(P))

    # plt.ylim(0, max_val)
    # plt.xlim(min(n_years),max(n_years))
    plt.xlabel("Time to Maturity")
    plt.ylabel("Price Volatility")
    plt.title("Price Volatility vs. Time to Maturity plot")
    plt.legend()
    plt.savefig("hw2p5.pdf")
    plt.show()
