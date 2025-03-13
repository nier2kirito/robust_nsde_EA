import numpy as np
from scipy.stats import norm
import torch

def bs_call_price(S, K, T, r, sigma):
    if T <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put_price(S, K, T, r, sigma):
    
    if T <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def generate_option_prices(S, r, sigma, maturities, strikes, option_type='call'):
    prices = np.zeros((len(maturities), len(strikes)))
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            if option_type == 'call':
                prices[i, j] = bs_call_price(S, K, T, r, sigma)
            elif option_type == 'put':
                prices[i, j] = bs_put_price(S, K, T, r, sigma)
            else:
                raise ValueError("option_type must be 'call' or 'put'")
    return prices

def save_option_prices_to_file(filename, prices):
    prices_tensor = torch.tensor(prices, dtype=torch.float32)
    torch.save(prices_tensor, filename)
    print(f"Option prices saved to '{filename}'")
