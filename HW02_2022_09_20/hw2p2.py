#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: surajgiri

HW2 Problem 2
"""

import numpy as np
from scipy.optimize import brentq

def PV(x, c, m, n, d, F):
    C = F * c / m
    i = np.arange(1, m * n +1)
    out = np.sum(C/(1+x/m)**i) + (F/(1+x/m)**(m*n)) - d * F
    return out

if __name__ == "__main__":
    c = 0.1
    d = 0.75
    n = 10
    m = 2
    F = 1

    iRR = brentq(PV, 0, 1, args=(c, m, n, d, F))
    out = iRR * 100
    print("Yield to maturity: ", out, "%")
