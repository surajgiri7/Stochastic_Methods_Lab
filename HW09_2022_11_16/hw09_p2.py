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
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# defining the function for binomial tree such that it stores the option value at each node of tree
def binomial_tree(payoff, n, r, sigma, S, K, T):
    dt = T/n
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    p = (np.exp(r*dt) - d)/(u - d)
    # initialize the option value at the last node
    x, y = np.arange(0, n+1, 1), np.arange(n, -1, -1)
    Stock_Price_Lattice = S * (u ** x) * (d ** y)

    # Calculating the Payoff at the Expiry
    poff = payoff(Stock_Price_Lattice, K=K)
    # initialize the option value at each node of tree
    tree = [] 
    # Backward induction with a single for loop
    for i in np.arange(n, 0, -1):
        v_u = poff[1:i+1] # ïncreasing the values in the y axis for each index
        v_d = poff[0:i]   # keeping the values of the x-axis constant for each index
        poff = np.exp(-r * dt) * (p * v_u + (1-p) * v_d)
        tree.append(poff)
    return tree


# defining the function for payoff
def payoff(S, K):
    return np.maximum(S-K, 0)

    
# main function
if __name__ == "__main__":
    # parameters
    n = 50 # time steps
    r = 0.02
    sigma = 0.5
    S = 1
    K = 0.7    
    T = 1

    # binomial tree
    tree = binomial_tree(payoff, n, r, sigma, S, K, T)
    tree.reverse()
    print("||||| Tree |||||")
    print(tree)
    
    # taking the length of the the longest array in the tree
    max_len = max([len(i) for i in tree])
    # padding the array with negative value equally from both sides if the array is less than the max_len by even number
    # and from the left side if the array is less than the max_len by odd number
    tree = [np.pad(i, (int((max_len - len(i))/2), int((max_len - len(i))/2 + (max_len - len(i))%2)), 'constant', constant_values=(-1, -1)) for i in tree]
    # print("||||| Tree after padding |||||")
    # print(tree)

    # masking the negative values
    tree = np.ma.masked_where(np.array(tree) < 0, np.array(tree))
    # print("||||| Tree after masking |||||")
    # print(tree)

    # plotting the tree using imshow after masking the negative values
    plt.imshow(tree, cmap=cm.coolwarm, interpolation='nearest', vmin=0, vmax=1) # vmin and vmax are used to set the range of the colorbar
    plt.colorbar()
    plt.title(f"Binomial Tree for n = {n} ".format(n))
    plt.xlabel("Stock Price")
    plt.ylabel("Time")

    if not os.path.exists("plots"):
        os.makedirs("plots")
    # plt.savefig("plots/hw09_p2.pdf")
    plt.show()
