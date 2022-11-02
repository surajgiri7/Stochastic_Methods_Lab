"""
Author: Suraj Giri
hw6 problem 4
"""
"""
Approximate Ito integral and Statonovich integral
W(t) = standard Brownian motion
t_i = i * dt
dt = T/N
X(t) = W(t)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# defining the function for standard brownian motion
def Standard_Brownian_Motion(N, T):
    # the time step
    dt = T/N
    S_0 = 0
    ds = np.random.normal(0, 1 ,size=(N-1)) * np.sqrt(dt) # generating the random numbers
    dw = np.insert(ds, [0], [S_0]) # inserting the initial value
    dw = np.cumsum(dw) # cumulating the random numbers
    return dw

# defining the function for Itô integral using std Brownian motion
def Ito_integral(B,N, T):
    # getting the standard Brownian motion and using it to calculate the Ito integral
    brownian_model = B
    ito_step_0 = brownian_model[:-1] # getting the first step
    ito_step_1 = brownian_model[1:]    # getting the second step
    Ito = ito_step_0 * (ito_step_1 - ito_step_0) # calculating the Ito integral
    Ito = np.cumsum(Ito) # cumulating the Ito integral
    return Ito

# defining the function for Stratonovich integral using std Brownian motion
def Stratonovich_integral(B, N, T):
    # getting the standard Brownian motion and using it to calculate the Stratonovich integral
    brownian_model = B
    stratonovich_step_0 = brownian_model[0:-1] # getting the first step
    stratonovich_step_1 = brownian_model[1:] # getting the second step
    Stratonovich = (stratonovich_step_0 + stratonovich_step_1) / 2 * (stratonovich_step_1 - stratonovich_step_0) # calculating the Stratonovich integral
    Stratonovich = np.cumsum(Stratonovich) # cumulating the Stratonovich integral
    return Stratonovich

# defining the function to plot the realization of Brownian motion, corresponding Ito integral and Stratonovich integral
def plot_models(brownian_model, ito_model, stratonovich_model, N, T):
    # plotting the realization of Brownian motion, corresponding Ito integral and Stratonovich integral
    plt.figure(figsize=(10, 6))
    plt.plot(brownian_model, color='blue', label='Brownian motion')
    plt.plot(ito_model, color='red', label='Ito integral')
    plt.plot(stratonovich_model, color='green', label='Stratonovich integral')
    plt.xlabel('time')
    plt.ylabel('value')
    plt.title('Realization of Brownian motion, corresponding Ito integral and Stratonovich integral')
    plt.plot(stratonovich_model - ito_model, color='orange', label='Stratonovich integral - Ito integral')
    plt.legend()
    return None
    
# main function
if __name__ == "__main__":
    N = 10000
    T = 1
    S_0 = 0
    
    # getting the standard Brownian motion and using it to calculate the Ito integral and Stratonovich integral
    brownian = Standard_Brownian_Motion(N, T)
    ito = Ito_integral(brownian, N, T)
    statonovich = Stratonovich_integral(brownian, N, T)

    # plotting all the paths
    plot_models(brownian, ito, statonovich, N, T)
     
    # sub folder to save the plots
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig('plots/hw6p4.pdf')
    plt.show()