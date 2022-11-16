"""
Author: Suraj Giri
Program: hw08_p1b.py
"""

"""
Problem
=======
X = X_t Itô process
dX = f(X,t)dt + g(X,t)dW
F(X,t) -> twice continuously differentiable
mu = 0.5
sigma = 3
F(X,t) = (1+t^2) cos(X) => True Solution

Todo: Plot the comparision between the True Solution and the Euler-Maruyama method!
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# define the function for Standard Brownian Motion
def Standard_Brownian_Motion(T, N):
    # defining the time step
    dt = T/N
    S_0 = 0
    ds = np.random.normal(scale=np.sqrt(dt), size=(N-1)) # generating the random numbers
    dw = np.insert(ds, [0], [S_0]) # inserting the initial value
    dw = np.cumsum(dw) # cumulating the random numbers
    return dw

# define the fnction for Geometric Brownian Motion
def Geometric_Brownian_Motion(w,mu, sigma, S_0, T, N):
    t = np.linspace(0, T, N)
    S = S_0 * np.exp((mu - (sigma**2)/2) * t + sigma * w)
    return S

# define the function for Euler-Maruyama method
def EulerMaruyama(w,mu, sigma, S_0, T, N):
    dt = T/N
    t = np.linspace(0, T, N)
    S = np.zeros(N)
    S[0] = np.cos(1)
    for i in range(1, t.size):
        # W_i = np.random.normal(0, np.sqrt(dt)) # 
        # S[i] = (S[i-1]) + (mu * S[i-1] * dt) + (sigma * S[i-1] * (w[i] - w[i-1]))
        S[i] = (S[i-1] + (((2*S[i-1]/(1+t[i-1])) - mu*np.arccos(S[i-1]/(1+t[i-1])**2)*(np.sqrt((1+t[i-1])**4 - S[i-1]**2)) - (sigma**2/2)*S[i-1]*(np.arccos(S[i-1]/(1+t[i-1])**2))**2) *dt - sigma*np.arccos(S[i-1]/(1+t[i-1])**2)*(np.sqrt((1+t[i-1])**4 - S[i-1]**2)*(w[i] - w[i-1])) ) )

        # S[i] = (S[i-1] + (((2*S[i-1]/(1+t[i-1])) - mu*np.arccos(S[i-1]/(1+t[i-1])**2)(np.sqrt((1+t[i-1])**4 )))))

    return S

# define the function for True solution
def TrueSolution(t, s):
    out = (1 + t)**2 * np.cos(s)
    return out

# define the main function
if __name__ == "__main__":
    # define the parameters
    mu = 0.5
    sigma = 3
    S_0 = np.cos(1)
    T = 1
    N = 1000
    t = np.linspace(0, T, N)

    # generate the standard brownian motion
    w = Standard_Brownian_Motion(T, N)

    # generate the geometric brownian motion
    S = Geometric_Brownian_Motion(w, mu, sigma, S_0, T, N)

    # generate the euler-maruyama method
    S1 = EulerMaruyama(w, mu, sigma, S_0, T, N)

    # generate the true solution
    s = TrueSolution(t, S)

    # plot the graph
    plt.plot(t, S1, label = "Euler-Maruyama Method")
    plt.plot(t, s, label = "$F(X,t) = (1+t^2) cos(X)$")
    plt.legend()
    # saving the plot to a folder
    if not os.path.exists('plots'):
        os.makedirs('plots')
    # plt.savefig('plots/hw08_p1b.pdf')
    plt.show()