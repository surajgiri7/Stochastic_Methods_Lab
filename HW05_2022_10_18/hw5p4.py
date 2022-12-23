"""
Author: Suraj Giri
Date: 25/10/2022
Homework 5, Problem 4
"""

from random import sample
import numpy as np
import matplotlib.pyplot as plt

# Define the function to rescale the sample
def rescale(sample):
    return (sample - np.mean(sample))/np.std(sample)

# Main
if __name__ == "__main__":
    N = 1000
    p = 0.5

    binomial_sample = np.random.binomial(n=N, p=p, size=N)
    normal_sample = np.random.normal(loc=0, scale=p, size=N)

    rescaled_binomial_sample = rescale(binomial_sample)
    rescaled_normal_sample = rescale(normal_sample)

    sorted_binomial_sample = np.sort(rescaled_binomial_sample)
    sorted_normal_sample = np.sort(rescaled_normal_sample)

    plt.figure(1)
    plt.title('Rescaled Binomial Sample vs. Rescaled Normal Sample')
    plt.xlabel('Sorted Scaled Normal Distribution')
    plt.ylabel('Sorted Scaled Binomial Distribution')
    plt.plot(sorted_normal_sample, sorted_normal_sample, label='Diagonal')
    plt.scatter(sorted_normal_sample, sorted_binomial_sample, s=5, c='g', label='Q-Q Plot')
    plt.legend()
    plt.grid(linestyle='--')
    # plt.savefig('hw5p4.pdf')
    plt.show()

"""
Comment:
Normal Distribution and Binomial Distribution are approximately equal.
We see that the binomial distribution with n trials and probability p of success
gets closer and closer to a normal distribution, which is the statement of Central Limit Theorem.
"""