#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: surajgiri
HW2 Problem 3
"""
import numpy as np
import matplotlib.pyplot as plt

def PV(r, c, m, n, F):
    C = F * c / m
    i = np.arange(1, m * n +1)
    out = np.sum(C/(1+r/m)**i) + (F/(1+r/m)**(m*n))
    return out

if __name__ == "__main__":
    F = 1000
    C = [0.02, 0.06, 0.12]
    r = 0.06

    years = np.arange(0, 11)

    P = []
    for c in C:
        P = [PV(r,c,2,year,F) for year in years]
        plt.plot(years, P, label='c $= {:.0f}\%$'.format(c * 100))
    
    plt.xlim(max(years),0)
    plt.xlabel("Time to Maturity (Years)")
    plt.ylabel("Price")
    plt.title("Price vs. Time to Maturity for Level Coupon Bonds")
    plt.tight_layout()
    plt.legend()
    plt.savefig('hw2p3.pdf')
    plt.show()
