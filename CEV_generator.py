import numpy as np
import torch

def simulate_cev(S0, r, sigma, beta, T, n_steps, MC_samples):
    """
    Simulates paths for a CEV process defined by:
      dS_t = r * S_t dt + sigma * S_t^beta dW_t,
    using an Euler scheme.
    
    Parameters:
      S0: initial asset price.
      r: risk-free rate.
      sigma: volatility parameter.
      beta: elasticity parameter (beta = 1 gives GBM).
      T: time to maturity.
      n_steps: number of time steps.
      MC_samples: number of Monte Carlo paths.
    
    Returns:
      S_paths: a NumPy array of shape (MC_samples, n_steps+1) with simulated paths.
    """
    dt = T / n_steps
    S_paths = np.zeros((MC_samples, n_steps+1))
    S_paths[:, 0] = S0
    for i in range(1, n_steps+1):
        dW = np.random.randn(MC_samples) * np.sqrt(dt)
        S_prev = S_paths[:, i-1]
        # Euler scheme update
        S_paths[:, i] = S_prev + r * S_prev * dt + sigma * (S_prev ** beta) * dW
    return S_paths

def cev_call_price_mc(S0, K, T, r, sigma, beta, n_steps, MC_samples):
    """
    Estimates the European call option price for the CEV process using Monte Carlo simulation.
    
    Parameters:
      S0: initial asset price.
      K: strike price.
      T: time to maturity.
      r: risk-free rate.
      sigma: volatility parameter.
      beta: elasticity parameter.
      n_steps: number of time steps in simulation.
      MC_samples: number of Monte Carlo samples.
    
    Returns:
      price: the Monte Carlo estimate of the call price.
    """
    paths = simulate_cev(S0, r, sigma, beta, T, n_steps, MC_samples)
    payoffs = np.maximum(paths[:, -1] - K, 0)
    price = np.exp(-r * T) * np.mean(payoffs)
    return price

def generate_cev_option_prices(S, r, sigma, beta, maturities, strikes, n_steps=96, MC_samples=100000, option_type='call'):
    prices = np.zeros((len(maturities), len(strikes)))
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            if option_type == 'call':
                prices[i, j] = cev_call_price_mc(S, K, T, r, sigma, beta, n_steps, MC_samples)
            elif option_type == 'put':
                paths = simulate_cev(S, r, sigma, beta, T, n_steps, MC_samples)
                payoffs = np.maximum(K - paths[:, -1], 0)
                prices[i, j] = np.exp(-r * T) * np.mean(payoffs)
            else:
                raise ValueError("option_type must be 'call' or 'put'")
    return prices

def save_cev_option_prices_to_file(filename, prices):
    prices_tensor = torch.tensor(prices, dtype=torch.float32)
    torch.save(prices_tensor, filename)
    print(f"CEV option prices saved to '{filename}'")
