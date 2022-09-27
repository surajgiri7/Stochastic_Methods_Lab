#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: surajgiri

HW2 Problem 1
"""

import numpy as np
import timeit
from scipy import optimize


#polynomial f
def f(x):
    x = x**np.arange(1,len(C)+1,1)
    return np.dot(C, x) - P

#Derivative of f
def df(x):
    a = np.arange(1,len(C)+1,1)
    x = x**np.arange(0,len(C),1)
    return np.dot(C*a,x)

# Bisection Method 
def Bisection(f, x1, x2, error):
    if (f(x1)*f(x2) < 0 and (x2-x1) < error):
        return (x1+x2)/2
    elif (f(x1)*f(x2) < 0):
        x3 = (x1+x2)/2
        if (f(x1)*f(x3) < 0):
            return Bisection(f,x1,x3,error)
        else:
            return Bisection(f,x3,x2,error)
    
# Secant Method 
def Secant(f, x1, x2, error):
    while(f(x2-error)*f(x2+error) > 0):
        x3 = x2
        x2 = x2 - f(x2)*(x2-x1)/(f(x2)-f(x1))
        x1 = x3
    return x2        

# Newton Method 
def Newton(f, f1, x, error):
    while(f(x-error)*f(x+error) > 0):
        x = x - f(x)/df(x)
    return x

# Brentq Function 
def Brentq(f,x1,x2):
    return optimize.brentq(f, x1, x2)


if __name__ == "__main__":
    N = 300
    C = 120.0 * np.arange(10,N+10)
    P = 15000

    x1 = 0.1
    x2 = 0.9
    error = 0.00001

    root_bisect = Bisection(f, x1, x2, error)
    root_secant = Secant(f, x1, x2, error)
    root_newton = Newton(f, df, x2, error)
    root_brentq = Brentq(f, x1, x2)

    irr_bisect = (1/root_bisect-1) * 100
    irr_secant = (1/root_secant-1) * 100
    irr_newton = (1/root_newton-1) * 100
    irr_brentq = (1/root_brentq-1) * 100

    t_bisect = timeit.Timer('Bisection(f, x1, x2, error)', 'from __main__ import Bisection ,f,x1,x2,error')
    t_secant = timeit.Timer('Secant(f, x1, x2, error)', 'from __main__ import Secant ,f,x1,x2,error')
    t_newton = timeit.Timer('Newton(f, df, x2, error)', 'from __main__ import Newton ,f, df, x1, x2, error')
    t_brentq = timeit.Timer('Brentq(f, x1, x2)', 'from __main__ import Brentq,f,x1,x2,error')

    print("Values of IRR for different Methods: ")
    print("IRR with Bisection Method = " + str(irr_bisect) + " %")
    print("IRR with Secant Method = " + str(irr_secant) + " %")
    print("IRR with Newton Method = " + str(irr_newton) + " %")
    print("IRR with Brentq Function = " + str(irr_brentq) + " %")

    print("Times Taken by each method: ")
    print("Run time for Bisection = ", t_bisect.timeit(number=1000), "s")
    print("Run time for Secant = ", t_secant.timeit(number=1000), "s")
    print("Run time for Newton = ", t_newton.timeit(number=1000), "s")
    print("Run time for Brentq = ", t_brentq.timeit(number=1000), "s")


