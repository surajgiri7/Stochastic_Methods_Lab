"""
Author: Suraj Giri
hw6 problem 1
"""

"""
Ensemble atleast 1000 of std Brownian paths W(t)
interval [0, T] = [0, 1]
N = 600 time steps
plot mean and standard deviation of the ensemble as a function of time
plot 10 sample paths
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# defining the function for standard brownian motion
def Standard_Brownian_Motion(N, T, m):
    # defining the time step
    dt = T/N
    S_0 = 0
    ds = np.random.normal(scale=np.sqrt(dt), size=(m, N-1)) # generating the random numbers
    dw = np.insert(ds, [0], [S_0], axis=1) # inserting the initial value
    dw = np.cumsum(dw, axis=1) # cumulating the random numbers
    return dw
    



# main function
if __name__ == "__main__":
    N = 600
    m = 1000
    T = 1
    array = Standard_Brownian_Motion(N, T, m)
    mean_brownian = np.mean(array, axis=0)
    std_deviate_brownian = np.std(array, axis=0)

    # plotting  10 sample paths, mean and std_deviate_brownian in the same plot
    steps = np.linspace(0, 1, N)
    plt.figure(figsize=(10, 6))

    # plotting 10 sample paths
    plt.plot(steps, array[0:10].T, color='blue', alpha=0.5)

    # plotting mean and std deviation
    plt.plot(steps, mean_brownian, color='red', label='mean_brownian')
    plt.plot(steps, mean_brownian + std_deviate_brownian, color='green', label='std_deviate_brownian')
    plt.plot(steps, mean_brownian - std_deviate_brownian, color='green', label='std_deviate')
    plt.xlabel('time')
    plt.ylabel('value')
    plt.title('Ensemble of standard Brownian paths')
    plt.legend()
    
    # sub folder to save the plots
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig('plots/hw6p1.pdf')
    plt.show()
