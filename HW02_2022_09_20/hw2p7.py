#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: surajgiri
HW2 Problem 7
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import  optimize


def PV(r, c, m, n, F):
    C = F * c / m
    i = np.arange(1, m * n +1)
    out = np.sum(C/(1+r/m)**i) + (F/(1+r/m)**(m*n))
    return out

def FV(r,c,m,n,F,H):
    FV = ((1+r)**(H*m))*PV(r,c,m,n,F)
    return FV

if __name__ == "__main__":
    n = 30
    c = 0.1
    m = 1
    F = 1
    H = 10

    min = optimize.brent(FV, args=(c, m, n, F,H))
    ymin = FV(min,c,m,n,F,H)

    r = np.linspace(0,0.3,1000)
    F = [FV(x,c,m,n,F,H) for x in r]
    minimum = (min, ymin)

    plt.figure(figsize=(8,6))
    plt.plot(r,F)
    plt.scatter(min, ymin, marker='o',color="Red")
    plt.xlabel("yield")
    plt.ylabel("future value")
    plt.title("future value vs yield of a bond")
    print("Minimum = ", minimum)
   
    plt.plot()
    plt.plot([], [], ' ', label="Minimum" + str(minimum))
    plt.legend()
    
    plt.savefig('hw2p7.pdf')

    plt.show()