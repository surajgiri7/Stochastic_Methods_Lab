"""
Author: Suraj Giri
HW7: Problem 1a
"""

"""
Use Euler Maruyama to solve equation 1 in HW7
equation 1 is: d_S_t = mu * S_t * dt + sigma * S_t * dW_t
mu = 1.5
sigma = 0.8
S_0 = 1 up
Final time T = 1

To do: Compare the result in a plot pathwise against the exact solution in equation 2
equation 2 is: S_t = S_0 * exp(*mu - sigma^2/2) * t + sigma * W_t)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# defining the function for Standard Brownian Motion
def Standard_Brownian_Motion(N, T, m):
    # defining the time step
    dt = T/N
    S_0 = 0
    ds = np.random.normal(scale=np.sqrt(dt), size=(N-1)) # generating the random numbers
    dw = np.insert(ds, [0], [S_0]) # inserting the initial value
    dw = np.cumsum(dw) # cumulating the random numbers
    return dw

# defining the function for Euler Maruyama
def eulerMaruyama(w,mu, sigma, S_0, T, N):
    dt = T/N
    t = np.linspace(0, T, N)
    S = np.zeros(N)
    S[0] = S_0
    for i in range(1, t.size):
        # W_i = np.random.normal(0, np.sqrt(dt)) # 
        S[i] = (S[i-1]) + (mu * S[i-1] * dt) + (sigma * S[i-1] * (w[i] - w[i-1]))
    return S, t

# defining the function for exact solution using equation 2 i.e Geometric Brownian Motion
def exactBrownianSolution(w,mu, sigma, S_0, T, N):
    t = np.linspace(0, T, N)
    S = S_0 * np.exp((mu - (sigma**2)/2) * t + sigma * w)
    return S, t


# main function
if __name__ == "__main__":
    mu = 1.5
    sigma = 0.8
    S_0 = 1
    T = 1
    N = 1000

    # generating the brownian motion
    w = Standard_Brownian_Motion(N, T, 1)   
    # calling the function for Euler Maruyama
    S, t = eulerMaruyama(w, mu, sigma, S_0, T, N)
    # calling the function for exact solution using Brownian motion
    S_exact, t_exact = exactBrownianSolution(w, mu, sigma, S_0, T, N)

    # plotting the graph
    plt.plot(t, S, label = "Euler Maruyama")
    plt.plot(t_exact, S_exact, label = "Exact Solution")
    plt.xlabel('t')
    plt.ylabel('S(t)')
    plt.title('Euler Maruyama vs Exact Brownian Solution')
    plt.grid()
    plt.legend()

    # sub folder to save the plots
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig('plots/hw7p1a.pdf')
    plt.show()