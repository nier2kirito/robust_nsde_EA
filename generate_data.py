import torch
import numpy as np

# Black-Scholes model
def black_scholes_price(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    N_d1 = 0.5 * (1 + torch.erf(torch.tensor(d1) / np.sqrt(2)))
    N_d2 = 0.5 * (1 + torch.erf(torch.tensor(d2) / np.sqrt(2)))
    return S0 * N_d1 - K * torch.exp(-r * T) * N_d2

# SABR model simulation (Euler scheme)
def sabr_model(S0, K, T, r, alpha, beta, rho, nu, n_steps=100):
    dt = T / n_steps
    S, V = S0, alpha
    for _ in range(n_steps):
        dW1 = np.random.normal(0, np.sqrt(dt))
        dW2 = rho * dW1 + np.sqrt(1 - rho**2) * np.random.normal(0, np.sqrt(dt))
        V = V * np.exp(-0.5 * nu**2 * dt + nu * dW2)
        S = S + r * S * dt + V * S**beta * dW1
    return max(S - K, 0) * np.exp(-r * T)

# Rough Bergomi model
def rough_bergomi_model(S0, K, T, r, eta, H, xi, n_steps=100):
    dt = T / n_steps
    S = S0
    V = xi * np.exp(eta * np.random.normal(0, np.sqrt(dt)) * (T**H))
    for _ in range(n_steps):
        dW = np.random.normal(0, np.sqrt(dt))
        V = V * np.exp(-0.5 * eta**2 * dt + eta * dW)
        S = S + r * S * dt + np.sqrt(V) * S * dW
    return max(S - K, 0) * np.exp(-r * T)

# Variance Gamma model
def variance_gamma_model(S0, K, T, r, sigma, theta, nu, n_steps=100):
    dt = T / n_steps
    S = S0
    for _ in range(n_steps):
        dG = np.random.gamma(shape=dt/nu, scale=nu)
        dW = np.random.normal(0, np.sqrt(dG))
        S = S * np.exp((r - 0.5 * sigma**2) * dt + theta * dG + sigma * dW)
    return max(S - K, 0) * np.exp(-r * T)

# Generate independent train and test datasets
def generate_independent_option_prices(model, S0=1.0, T_values=[0.1, 0.5, 1.0], K_values=np.linspace(0.8, 1.2, 21), r=0.02, n_train_samples=1000, n_test_samples=200):
    def generate_prices(n_samples):
        call_prices = []
        for _ in range(n_samples):
            T = np.random.choice(T_values)
            K = np.random.choice(K_values)
            if model == "black_scholes":
                price = black_scholes_price(S0, K, T, r, sigma=0.2)
            elif model == "sabr":
                price = sabr_model(S0, K, T, r, alpha=0.2, beta=1, rho=-0.5, nu=0.3)
            elif model == "rough_bergomi":
                price = rough_bergomi_model(S0, K, T, r, eta=1.0, H=0.1, xi=0.04)
            elif model == "variance_gamma":
                price = variance_gamma_model(S0, K, T, r, sigma=0.2, theta=-0.1, nu=0.2)
            else:
                raise ValueError("Unsupported model type")
            call_prices.append(price)
        return call_prices

    train_data = generate_prices(n_train_samples)
    test_data = generate_prices(n_test_samples)
    return train_data, test_data

# User selects the model
selected_model = "sabr"  # Change to "black_scholes", "sabr", "rough_bergomi", or "variance_gamma" as needed
train_data, test_data = generate_independent_option_prices(selected_model)

torch.save(train_data, f"train_option_prices_{selected_model}.pt")
torch.save(test_data, f"test_option_prices_{selected_model}.pt")
