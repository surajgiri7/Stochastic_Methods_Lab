"""
Author: Suraj Giri
Problem: 3a
"""

"""
Take option price from the Black Scholes formula
a. the explicit form of finite difference approximation of the Black Scholes
is given in question. Write a code which uses the explicit finite difference scheme to 
price a European call option. Make sure to include a discussion of the boundary conditions.

Formula for the explicit finite difference scheme is given by
((V_m_n+1 - V_n_m) / delta_t) + ((sigma**2 /2) * (V_n-1_m - 2*V_n_m + V_n+1_m) / delta_X**2) + (r - sigma**2 / 2) * ((V_n+1_m - V_n-1_m) / (2 * delta_x)) - r * V_n_m) = 0

b. Show that the explicit code becomes unstable unless the time step ∆t is much smaller
than delta_X.

c. Modify your code to use the implicit finite difference scheme from given in the question.
Show that it is stable even when the time step delta_t is large.

d. Demonstrate the order of convergence of the implicit finite difference method with
the same number of meshpoints in the t and in the X direction.

"""

"""
Plotting the explicit error vs delta_t for different delta_X using explicit finite difference scheme
for option pricing using Black Scholes formula
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.linalg import solve_banded
from pylab import *

# function for the Black Scholes formula
def Black_Scholes(S, K, T, sigma, rp):
    x = (np.log(S/K) + (rp + sigma**2/2) * T) / (sigma * np.sqrt(T))
    Black_Scholes = S * norm.cdf(x) - K * np.exp(-rp * T) * norm.cdf(x - sigma * np.sqrt(T))
    return Black_Scholes

# function for the explicit finite difference scheme
# def explicit_finite_diff(S, K, T, sigma, rp, delta_X, delta_t):
def explicit_finite_diff(S, K, T, M, N, sigma, r):
    delta_X = np.log(S) / N
    delta_t = T/M
    S = np.linspace(0, S, N)
    t = np.linspace(0, T, M)

    # boundary
    V = np.zeros((M, N))
    V[-1, :] = np.maximum(S - K, 0)
    V[:, -1] = S - K * np.exp(-r * (T - t))

    # explicit finite difference scheme
    for j in range(M-1, 0, -1):
        V[j-1, 1:N-1] = V[j, 1:N-1] + delta_t * (sigma**2 / 2 * (V[j, 0:N-2] - 2 * V[j, 1:N-1] + V[j, 2:N]) / delta_X**2 + (r - sigma**2 / 2) * (V[j, 2:N] - V[j, 0:N-2]) / (2 * delta_X) - r * V[j, 1:N-1])
   # for m in range(M-1,0,-1):
    #     V[m-1][1:N-1] = V[m][1:N-1]  + delta_t*(sigma**2/2*(V[m][0:N-2] - 2*V[m][1:N-1] + V[m][2:N])/delta_X**2 + (r - sigma**2/2)*(V[m][2:N] - V[m][0:N-2])/(2*delta_X) - r*V[m][1:N-1])


            
    return t, S, np.fliplr(V)

# function for the implicit finite difference scheme
def implicit_finite_diff(S, K, T, M, N, sigma, r):
    delta_X = np.log(S) / N
    delta_t = T/M
    S = np.linspace(0, S, N)
    t = np.linspace(0, T, M)

    # boundary
    V = np.zeros((M, N))
    V[-1, :] = np.maximum(S - K, 0)
    V[:, -1] = S - K * np.exp(-r * (T - t))

    # banded matrix components and diagonal
    # calculating upper diagonal, diagonal and lower diagonal terms as banded matrix
    a = np.zeros(N-2)
    b = np.zeros(N-2)
    c = np.zeros(N-2)
    d = np.zeros(N-2)
    for j in range(M-1, 0, -1):
        a = -delta_t * (sigma**2 / 2 * (1 / delta_X**2) + (r - sigma**2 / 2) / (2 * delta_X))
        b = 1 + delta_t * (sigma**2 / delta_X**2 + r)
        c = -delta_t * (sigma**2 / 2 * (1 / delta_X**2) - (r - sigma**2 / 2) / (2 * delta_X))
        d = V[j, 1:N-1]
        # boundary conditions
        d[0] = d[0] - a[0] * V[j, 0]
        d[N-3] = d[N-3] - c[N-3] * V[j, N-1]
        # solving the tridiagonal system
        V[j-1, 1:N-1] = solve_banded((1, 1), np.array([a, b, c]), d)

    return t, S, np.fliplr(V)

   
    # for j in range(1, M-1):
    #     for i in range(1, N-1):
    #         V[i,j] = V[i, j+1] + delta_t * (((sigma**2/2) * (V[i-1,j] - 2*V[i,j] + V[i+1,j]) / delta_X**2) + (r - sigma**2/2) * (V[i+1,j] - V[i-1,j]) / (2 * delta_X) + r * V[i,j])
    
# main function
if __name__ == "__main__":
    r = 0.05
    sigma = 0.5
    S = 200
    K = 100
    T = 1
    M = 100
    N = 100
    S_0 = 100

    # # explicit finite difference scheme
    # V_explicit = explicit_finite_diff(S, K, T, M, N, sigma, r)
    # print("The option price using explicit finite difference scheme is: ", V_explicit)

    # # implicit finite difference scheme
    # V_implicit = implicit_finite_diff(S, K, T, M, N, sigma, r)
    # print("The option price using implicit finite difference scheme is: ", V_implicit)

    # plotting the explicit error vs delta_t for different delta_X
    fig = figure()
    t,s,V = explicit_finite_diff(S, K, T, M, N, sigma, r)
    ax = fig.add_subplot(projection='3d')
    T,S = meshgrid(t, s)
    ax.plot_surface(T,S,V.T)
    ax.set_xlabel("Time")
    ax.set_ylabel("Stock Price")
    ax.set_zlabel("Option Price")
    ax.set_title('Explicit method')
    show()

    fig = figure()
    t,s,V = implicit_finite_diff(S, K, T, M, N, sigma, r)
    ax = fig.add_subplot(projection='3d')
    T,S = meshgrid(t, s)
    ax.plot_surface(T,S,V)
    ax.set_xlabel("Time")
    ax.set_ylabel("Stock Price")
    ax.set_zlabel("Option Price")
    ax.set_title('Implicit method')
    show()


    MN = arange(1000,11000,1000)
    Black_Scholes_Price = Black_Scholes(S_0, K, T, sigma, r)
    price = []
    for n in MN:
        index = int(floor(log(S_0)/log(S)*n))
        t,s,V = implicit_finite_diff(r,sigma,K,n,n,S_0,S,T)
        price.append(V[index,0])

    fig = figure()
    ax = fig.add_subplot(projection='3d')
    s = linspace(0,S,N)
    t = linspace(0,T,M)
    T,S = meshgrid(t, s)
    ax.plot_surface(T,S,Black_Scholes(r,sigma,S,K,T))
    ax.set_xlabel("Time")
    ax.set_ylabel("Stock Price")
    ax.set_zlabel("Option Price")
    ax.set_title('Black Scholes')
    show()
    
    # Paramater estimation using linear regression in loglog plot
    # def slope(x,a,k):
    #     return a*x**k

    # params = curve_fit(slope, MN, price)[0]

    figure(figsize=(8,6))
    plot(MN,abs(price-Black_Scholes_Price))
    plot(MN,19.70*(MN)**(-3.658925544e-03),label= r'slope = {:.2}'.format(-3.658925544e-03),color='blue')
    loglog()
    legend()
    show()
