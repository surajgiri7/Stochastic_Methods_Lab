#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: surajgiri
HW2 Problem 6
"""
import numpy as np
import matplotlib.pyplot as plt


def PV(r, c, m, n, F):
    C = F * c / m
    i = np.arange(1, m * n +1)
    out = np.sum(C/(1+r/m)**i) + (F/(1+r/m)**(m*n))
    return out

def FV(r, c, m, n, F):
    FV =  PV(r, c, m, n, F)*(1+r/m)**(m*n)
    return FV

if __name__ == "__main__":
    m = 2
    r = [0.06, 0.08, 0.1]
    n = np.arange(0, 16, 1)
    F = 1
    c = 0.08

    max_val = 0

    for rate in r:
        P = [FV(rate, c, m, year, F) for year in n]
        plt.plot(n, P, label='c $= {:.0f}\%$'.format(rate * 100))
        max_val = max(max_val, max(P))

    plt.xlabel("Years to maturity")
    plt.ylabel("Bond Value")
    plt.title("Years vs. Bond Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig('hw2p6.pdf')
    plt.show()