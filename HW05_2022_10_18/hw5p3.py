"""
Author: Suraj Giri
Date: 25/10/2022
Homework 5, Problem 3
"""

from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from scipy.stats import binom

# Define the function to convert x coordinate to binomial distribution
def binomial_x(n, p):
    j = np.arange(0, n+1)
    n = np.ones(n+1)*n
    q = 1-p
    x_j = (j - n * p) / np.sqrt(n * p * q)
    return x_j

# Define the function to convert y coordinate to binomial distribution
def binomial_y(n, p):
    j = np.arange(0, n+1)
    n = np.ones(n+1)*n
    q = 1-p
    y_j = np.sqrt(n * p * q) * binom.pmf(j, n, p) # needs a look one more time
    return y_j

# Define the function for CDF of Standard Gaussian Distribution
def gaussian_cdf(x):
    return 1/np.sqrt(2 * np.pi) * np.exp(-x**2/2)

# Define the function to draw the sub plots in a single plot
def draw_subplots(n, p):
    x_j = binomial_x(n, p)
    y_j = binomial_y(n, p)
    plt.plot(x_j, y_j, label='Binomial')
    plt.plot(x_j, gaussian_cdf(x_j), label='Gaussinan')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('n = {}, p = {}'.format(n, p))
    plt.legend()
    plt.grid()

# Main
if __name__ == "__main__":
    plt.rc('figure', figsize=(12, 8))

    fig = plt.figure()
    fig.suptitle('Binomial Distribution vs. Gaussian Distribution')

    fig.add_subplot(2, 2, 1)
    draw_subplots(n=10, p=0.5)

    fig.add_subplot(2, 2, 2)
    draw_subplots(n=100, p=0.5)

    fig.add_subplot(2, 2, 3)
    draw_subplots(n=10, p=0.2)

    fig.add_subplot(2, 2, 4)
    draw_subplots(n=100, p=0.2)

    fig.tight_layout()
    plt.grid(linestyle='-')
    plt.savefig('hw5p3.pdf')
    plt.show()

"""
Comment:
The approximation generally improves as n increases and is better when p is not near to 0 or 1.
"""