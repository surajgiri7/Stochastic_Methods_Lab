"""
Author: Suraj Giri
Program: hw09_p2.py
"""

"""
Problem:
Rewrite previous binomial tree function binomial_tree(payoff,n,r,sigma,S,K,T) 
such that it stores the option value at each node of tree. 

Then visualize the tree using imshow for some reasonable parameters. Here, you
have to think about an appropriate color map, how to mask the missing values 
(hint: use Numpy masked arrays), and how to best map the computed values to pixel coordinates.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# defining the function for binomial tree such that it stores the option value at each node of tree
def binomial_tree(payoff, n, r, sigma, S, K, T):
    dt = T/n
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    p = (np.exp(r*dt) - d)/(u - d)
    q = 1 - p
    # initialize the binomial tree
    tree = np.zeros((n+1, n+1))
    # initialize the option value at the last node
    x, y = np.arange(0, n+1, 1), np.arange(n, -1, -1)
    Stock_Price_Lattice = S * (u ** x) * (d ** y)
    tree[n] = payoff(Stock_Price_Lattice, K)
    # calculate the option value at each node
    for i in range(n-1, -1, -1):
        for j in range(i+1):
            tree[i, j] = np.exp(-r*dt)*(p*tree[i+1, j] + q*tree[i+1, j+1])
    return tree


# defining the function for payoff
def payoff(S, K):
    return np.maximum(S-K, 0)

    
# main function
if __name__ == "__main__":
    # parameters
    n = 10 # time steps
    r = 0.05
    sigma = 0.2
    S = 100
    K = 100
    T = 1

    # binomial tree
    tree = binomial_tree(payoff, n, r, sigma, S, K, T)
    # print("||||| Tree |||||")
    # print(tree)
    # print("||||| Tree Shape |||||")
    # print(tree.shape)
    
    # visualize the tree using imshow
    plt.imshow(np.fliplr(tree), cmap=cm.jet, interpolation='nearest')
    plt.colorbar()

    # showing the values of option value at each node of tree with imshow for each stock price
    #for i in range(n+1):
    #    for j in range(i+1):
    #        plt.text(j, i, round(tree[i, j], 2), ha="center", va="center", color="w")

    plt.xlabel('Stock Price')
    plt.ylabel('Option Value')
    plt.title('Binomial Tree with Option Value at Each Node')
    plt.legend()
    plt.show()