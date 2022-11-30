"""
Author: Suraj Giri
problem 1b
"""

"""
Tridiagonal matrix solver
"""

import numpy as np

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
    x = tridiag_solver(a, b)
    print('a: \n',a)
    print('b: \n', b)
    print('Solution: \n ', x)
