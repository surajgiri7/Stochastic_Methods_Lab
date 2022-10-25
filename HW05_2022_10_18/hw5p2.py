"""
Author: Suraj Giri
Date: 25/10/2022
Homework 5, Problem 2
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import special

# Define the function stirling approx
def stirling(n):
    return np.sqrt(2*np.pi*n)*(n/np.exp(1))**n

# Define the function relative error
def rel_error(n):
    return (factorial(n) - stirling(n))/stirling(n)

# Define the function factorial
def factorial(n):
    return special.gamma(n+1)


# Main
if __name__ == "__main__":
    N = 100
    N_array = np.arange(1, N+1)
    rel_error_array = np.zeros(N)

    stirling_array = stirling(N_array)
    factorial_array = factorial(N_array)
    rel_error_array = rel_error(N_array)

    slope = (np.log10(rel_error_array[-1]) - np.log10(rel_error_array[0]))/(np.log10(N_array[-1]) - np.log10(N_array[0]))
    print("Slope of the line: {}".format(slope))

    plt.figure(1)
    plt.title('Value of "n" vs "Relative Error"')
    plt.xlabel('n')
    plt.ylabel('Relative Error')
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(N_array, rel_error_array, 'b-')
    plt.plot([], [], ' ', label="slope = {}".format(slope))
    plt.legend()
    plt.grid(linestyle='--')
    plt.savefig('hw5p2.pdf')
    plt.show()

"""
    Comment:
    Yes, we can obtain a straight line whose slope of the log10 of rel error
    is -1.0027686782494103.
"""