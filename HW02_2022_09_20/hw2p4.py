#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: surajgiri
HW2 Problem 4
"""

import numpy as np
import matplotlib.pyplot as plt

def PV(r, c, m, n, d, F):
    C = F * c / m
    i = np.arange(1, m * n +1)
    out = np.sum(C/(1+r/m)**i) + (F/(1+r/m)**(m*n))
    return out

if __name__ == "__main__":
    c = 0.08
    n = 10
    F = 1000
    m = 1

    r = np.linspace(0, 1, 100)
    P = [PV(i,c,1,n,0,F) for i in r]

    plt.plot(r, P)
    plt.xlabel("Yield")
    plt.ylabel("Price")
    plt.title("Price vs. Yield")
    plt.savefig('hw2p4.pdf')
    plt.legend()
    plt.show()
