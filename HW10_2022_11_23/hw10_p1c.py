"""
Author: Suraj Giri
problem 1c
"""

"""
Tridiagonal matrix solver
"""

import numpy as np
from scipy.linalg import solve_banded
import timeit

# defining the tridiagonal matrix
def tridiag_solver(a, b):
    n = len(b)
    c = np.zeros(n)
    d = np.zeros(n)
    c[0] = a[0,1]/a[0,0] #
    d[0] = b[0]/a[0,0]
    for i in range(1,n-1):
        c[i] = a[i,i+1]/(a[i,i]-a[i,i-1]*c[i-1])
        d[i] = (b[i]-a[i,i-1]*d[i-1])/(a[i,i]-a[i,i-1]*c[i-1])
    d[n-1] = (b[n-1]-a[n-1,n-2]*d[n-2])/(a[n-1,n-1]-a[n-1,n-2]*c[n-2])
    x = np.zeros(n)
    x[n-1] = d[n-1]
    for i in range(n-2,-1,-1):
        x[i] = d[i]-c[i]*x[i+1]
    return x

# main function
if __name__ == '__main__':
    a = np.array([[2,-1,0],[-1,2,-1],[0,-1,2]])
    b = np.array([1,2,3])
    print('a: \n',a)
    print('b: \n', b)

    start_func = timeit.default_timer()
    x_func = tridiag_solver(a, b)
    end_func = timeit.default_timer()

    start_scipy = timeit.default_timer()
    x_banded = solve_banded((1,1), a, b)
    end_scipy = timeit.default_timer()
    
    print('Solution using defined function: \n ', x_func)
    print('Solution using scipy: \n ', x_banded)
    print('Time taken by defined function: ', end_func-start_func)
    print('Time taken by scipy: ', end_scipy-start_scipy)

