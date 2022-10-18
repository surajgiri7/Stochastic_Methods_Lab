import numpy as np
import matplotlib.pyplot as plt

def binomial_tree_EU(payoff, n, rp, sigma, S, K, T, opt):
    # Calculating the increase rate and decrease rate
    u = np.exp(sigma * np.sqrt(T / n))
    d = 1 / u

    # Discount the Payoffs by Backwards Induction
    dt = T / n
    p = (np.exp(rp * dt) - d) / (u - d)

    # Calculating a Stock Price Lattice for the Underlying Asset Price
    x, y = np.arange(0, n+1, 1), np.arange(n, -1, -1)
    Stock_Price_Lattice = S * u ** x * d ** y

    # Calculating the Payoff at the Expiry
    payoff_vec = np.vectorize(payoff)
    poff = payoff_vec(S=Stock_Price_Lattice, K=K, opt=opt)

    # Backward induction with a single for loop
    for i in np.arange(n, 0, -1):
        v_u = poff[1:i+1] # ïncreasing the values in the y axis for each index
        v_d = poff[0:i]   # keeping the values of the x-axis constant for each index
        poff = np.exp(-rp * dt) * (p * v_u + (1-p) * v_d)
        
    out = poff[0]
    return out

def binomial_tree_American(payoff, n, rp, sigma, S, K, T, opt):
    u = np.exp(sigma * np.sqrt(T/n))
    d = 1 / u

    R = np.exp(rp*(T/n))

    p = (R-d)/(u-d)

    # Calculating a Stock Price Lattice for the Underlying Asset Price
    x, y = np.arange(0, n+1, 1), np.arange(n, -1, -1)
    Stock_Price_Lattice = S * u ** x * d ** y

    # Calculating the Payoff at the Expiry
    payoff_vec = np.vectorize(payoff)
    poff = payoff_vec(Stock_Price_Lattice, K=K, opt=opt)

    for j in np.arange(n-1, -1, -1):

        a = np.arange(j, -1, -1)
        b = np.arange(0, j+1,1)

        v_u = poff[0:j+1]
        v_d = poff[1:j+2]

        Stock_Price_Lattice_2 = S * u ** a * d ** b
        m1 = p * v_u + (1-p)* v_d

        poff = np.maximum(m1/R, K - Stock_Price_Lattice_2)

    out = poff[0]
    return out


# def payoff(S, K):
#     payoff = np.maximum(0, S - K)
#     return payoff

def payoff(S, K, opt):
    if opt == 'C':
        return np.maximum(0, S-K)
    elif opt == 'P':
        return np.maximum(0, K-S)

if __name__ == "__main__":
    K = np.linspace(90,101, 100)
    S = 95 
    rp = 0.08
    T = 1/12
    n = 1000
    sigma = 0.25
    
    put_america = [binomial_tree_American(payoff,n,rp,sigma,S,stock,T,'P') for stock in K]
    put_europe = [binomial_tree_EU(payoff,n,rp,sigma,S,stock,T,'P') for stock in K]
    
    plt.plot(K, put_america, label='American Put Option')
    plt.plot(K, put_europe, label='European Put Option', linestyle='dashed')
    
    plt.xlabel('Stock Price ($)')
    plt.ylabel('Put Option')
    plt.legend()
    
    if (put_america > put_europe):
        answerTxt = "American put option $>$ European put option \n(Identical Parameters)"
    else:
        answerTxt = r"American put option $\leq$ European put option\n(Identical Parameters)"
    plt.text(98.5,1.5,answerTxt,verticalalignment='center', horizontalalignment='center')
    
    parameterTxt = "\n".join((
        r'$\sigma=%.3f$' % (sigma, ),
        r'$K=%.3f$' % (S, ),
        r'$T=%.3f$' % (T, ),
        r'$n=%d$' % (n, ),
        r'$r_p=%.4f$' % (rp, ),
        ))
    
    boxProp = dict(facecolor='none', edgecolor='gray')
    
    plt.text(91,7,parameterTxt,verticalalignment='center', horizontalalignment='center', bbox=boxProp)
    
    plt.show()
    
