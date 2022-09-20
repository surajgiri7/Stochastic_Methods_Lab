# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 15:14:00 2022

@author: surajgiri
Problem 1
"""

import numpy as np
import timeit

"""
C => Array of Current Values
r => Yearly rate of interest
PV => Present Value

"""

# an explicit Python loop summing up each summand
def Sum_up_each_summand(C, r):
    PV = 0
    x = 1.0/(1+r)
    for i, C_i in enumerate(C):
        PV += C_i * x ** (i+1)
    return PV

# an explicit Python loop using Horner’s scheme 
def Horners_scheme_loop(C, r):
    PV = 0
    x = 1.0/(1+r)
    n = C.shape[0]
    reverse_array = C[::-1]
    for i in range(n):
        PV = (PV*x) + reverse_array[i]
    return PV*x
    
# the polyval function
def Polyval_function(C, r):
    PV = 0
    x = 1.0/(1+r)
    reverse_array = C[::-1]
    reverse_array = np.append(reverse_array,0)
    PV = np.polyval(reverse_array, x)
    return PV

# dot product of vectors
def Dot_product_vectors(C, r):
    rate = np.arange(1, len(C)+1)
    x = 1/(1+r)
    A = x**rate
    PV = np.dot(C,A)

    return PV

if __name__ == "__main__":
    
    C = 120.0 * np.arange(500,1200)
    r = 0.01
    
    start = timeit.default_timer()
    PV = Sum_up_each_summand(C, r)
    end = timeit.default_timer()
    print("Present Value using sum of each summand: ", PV)
    print("Time using sum of each summand: ", end-start)
    print("\n")
    
    start = timeit.default_timer()
    PV = Horners_scheme_loop(C, r)
    end = timeit.default_timer()
    print("Present Value using horners scheme loop: ", PV)
    print("Time using horners scheme loop: ", end-start)
    print("\n")
    
    start = timeit.default_timer()
    PV = Polyval_function(C, r)
    end = timeit.default_timer()
    print("Present Value using polyval: ", PV)
    print("Time using polyval: ", end-start)
    print("\n")

    start = timeit.default_timer()
    PV = Dot_product_vectors(C, r)
    end = timeit.default_timer()
    print("Present Value using Dot Product of Vectors?: ", PV)
    print("Time using Dot Product of Vectors?: ", end-start)
    print("\n")
