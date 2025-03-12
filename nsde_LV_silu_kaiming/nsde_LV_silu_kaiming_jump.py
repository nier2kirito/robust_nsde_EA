import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from random import randrange
import copy
import argparse
from scipy.stats import norm
import matplotlib.pyplot as plt
from geomloss import SamplesLoss
import sys 
import os 
# Add parent directory to Python path to import networks
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networks import *  # Ensure your custom networks (e.g., Net_timegrid) are defined appropriately.

# --------------------------
# Jump Diffusion Simulation
# --------------------------
def simulate_jump_diffusion_path(S0, T, r, sigma, jump_lambda, jump_mean, jump_std, n_steps=1000):
    """
    Simulate an asset path using a Merton jump diffusion model.
    
    Parameters:
        S0 (float): initial asset price.
        T (float): time horizon.
        r (float): risk-free rate.
        sigma (float): volatility.
        jump_lambda (float): jump intensity (expected number of jumps per unit time).
        jump_mean (float): mean of jump (in log scale).
        jump_std (float): standard deviation of jump (in log scale).
        n_steps (int): number of time steps.
        
    Returns:
        prices (np.array): simulated asset price path.
    """
    dt = T / n_steps
    prices = np.zeros(n_steps + 1)
    prices[0] = S0
    for t in range(1, n_steps + 1):
        # Standard Brownian motion increment
        dW = np.random.normal(0, np.sqrt(dt))
        # Check if a jump occurs (using dt * lambda as probability)
        jump = 1.0
        if np.random.uniform(0, 1) < jump_lambda * dt:
            # Jump factor is lognormally distributed: exp(jump_mean + jump_std * N(0,1))
            jump = np.exp(np.random.normal(jump_mean, jump_std))
        # Update asset price using Euler's method
        drift = (r - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * dW
        prices[t] = prices[t-1] * np.exp(drift + diffusion) * jump
    return prices

def jump_diffusion_price(S0, K, T, r, sigma, jump_lambda, jump_mean, jump_std, n_samples=100, n_steps=1000):
    """
    Compute a European call option price using Monte Carlo simulation under jump diffusion.
    """
    option_prices = []
    for _ in range(n_samples):
        prices = simulate_jump_diffusion_path(S0, T, r, sigma, jump_lambda, jump_mean, jump_std, n_steps)
        option_prices.append(np.exp(-r * T) * max(prices[-1] - K, 0))
    return np.mean(option_prices)

def generate_jump_diffusion_test_data(S0, strikes, maturities, r, sigma, jump_lambda, jump_mean, jump_std, n_samples=100):
    """
    Generate a matrix of option prices (rows: maturities, columns: strikes)
    using the jump diffusion model.
    """
    test_data = torch.zeros(len(maturities), len(strikes))
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            test_data[i, j] = jump_diffusion_price(S0, K, T, r, sigma, jump_lambda, jump_mean, jump_std, n_samples)
    return test_data.float()

def generate_independent_test_data(S0, strikes, maturities, r, sigma=0.2, jump_lambda=0.1, jump_mean=0.0, jump_std=0.1, n_samples=100, n_steps=1000):
    """
    Generate independent test data for comprehensive evaluation.
    """
    return generate_jump_diffusion_test_data(S0, strikes, maturities, r, sigma, jump_lambda, jump_mean, jump_std, n_samples)

# --------------------------
# Model Tester Class
# --------------------------
class ModelTester:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.results = {}
    
    def generate_test_scenarios(self, n_scenarios=1000, n_steps=96):
        """Generate random test scenarios."""
        return torch.randn(n_scenarios, n_steps, device=self.device)
    
    def test_vanilla_pricing(self, S0, test_data, z_test, T):
        """Evaluate vanilla option pricing."""
        with torch.no_grad():
            pred, var, _, _, _, _ = self.model(S0, z_test, z_test.shape[0], T)
            pricing_error = torch.mean((pred - test_data)**2)
            relative_error = torch.mean(torch.abs(pred - test_data) / test_data)
        return {
            'predictions': pred.cpu().numpy(),
            'true_values': test_data.cpu().numpy(),
            'mse': pricing_error.item(),
            'relative_error': relative_error.item()
        }
    
    def test_exotic_pricing(self, S0, z_test, T):
        """Evaluate exotic option pricing."""
        with torch.no_grad():
            _, _, exotic_price, mean_price, price_var, _ = self.model(S0, z_test, z_test.shape[0], T)
        return {
            'mean_price': mean_price.item(),
            'variance': price_var.item(),
            'prices': exotic_price.cpu().numpy()
        }
    
    def run_comprehensive_test(self, S0, strikes, maturities, r):
        """Run comprehensive tests using independent test data."""
        test_data = generate_independent_test_data(S0, strikes, maturities, r)
        test_data = test_data.to(self.device)
        z_test = self.generate_test_scenarios()
        vanilla_results = self.test_vanilla_pricing(S0, test_data, z_test, max(maturities))
        exotic_results = self.test_exotic_pricing(S0, z_test, max(maturities))
        return {
            'vanilla_results': vanilla_results,
            'exotic_results': exotic_results,
            'test_data': test_data.cpu().numpy()
        }
    
# Modified Architecture for Neural SDE
class Net_LV_Improved(nn.Module):
    """
    Calibration of LV model: dS_t = S_t*r*dt + L(t,S_t,theta)dW_t to vanilla prices at different maturities
    Improved architecture with residual connections and deeper layers
    """
    def __init__(self, dim, timegrid, strikes_call,  n_layers, vNetWidth, device, rate, maturities, n_maturities, activation="relu"):
        super(Net_LV_Improved, self).__init__()

        self.dim = dim
        self.timegrid = timegrid
        self.device = device
        self.strikes_call = strikes_call
        self.maturities = maturities
        self.rate = rate
        self.activation = activation
        # Residual connections in the diffusion model
        self.diffusion = Net_timegrid(dim=dim+1, nOut=1, n_layers=n_layers, vNetWidth=vNetWidth, n_maturities=n_maturities, activation_output="softplus")

        # Control Variates
        self.control_variate_vanilla = Net_timegrid(dim=dim+1, nOut=len(strikes_call)*n_maturities, n_layers=4, vNetWidth=30, n_maturities=n_maturities)
        self.control_variate_exotics = Net_timegrid(dim=dim*len(self.timegrid)+1, nOut=1, n_layers=4, vNetWidth=20, n_maturities=n_maturities)

    def forward(self, S0, z, MC_samples, ind_T, period_length=30):
        """
        Forward pass for evaluation with added residuals
        """

        ones = torch.ones(MC_samples, 1, device=self.device)
        path = torch.zeros(MC_samples, len(self.timegrid), device=self.device)
        S_old = ones * S0
        path[:,0] = S_old.squeeze(1)

        cv_vanilla = torch.zeros(S_old.shape[0], len(self.strikes_call)*len(self.maturities), device=self.device)
        price_vanilla_cv = torch.zeros(len(self.maturities), len(self.strikes_call), device=self.device)
        var_price_vanilla_cv = torch.zeros_like(price_vanilla_cv)

        cv_exotics = torch.zeros(S_old.shape[0], 1, device=self.device)

        exotic_option_price = torch.zeros_like(S_old)

        running_max = S_old

        # Euler's scheme for diffusion
        for i in range(1, ind_T+1):
            idx = (i-1)//period_length  # assume maturities are evenly distributed
            t = torch.ones_like(S_old) * self.timegrid[i-1]
            h = self.timegrid[i] - self.timegrid[i-1]
            dW = (torch.sqrt(h) * z[:, i-1]).reshape(MC_samples, 1)

            current_time = ones * self.timegrid[i-1]
            diffusion = self.diffusion.forward_idx(idx, torch.cat([t, S_old], 1))
            S_new = S_old + self.rate * S_old * h / (1 + self.rate * S_old.detach() * torch.sqrt(h)) + S_old * diffusion * dW / (1 + S_old.detach() * diffusion.detach() * torch.sqrt(h))

            # Residual connections
            cv_vanilla += torch.exp(-self.rate * self.timegrid[i-1]) * S_old.detach() * diffusion.detach() * self.control_variate_vanilla.forward_idx(idx, torch.cat([t, S_old.detach()], 1)) * dW.repeat(1, len(self.strikes_call)*len(self.maturities))
            cv_exotics += torch.exp(-self.rate * self.timegrid[i-1]) * S_old.detach() * diffusion.detach() * self.control_variate_exotics.forward_idx(idx, torch.cat([t, path], 1)) * dW

            S_old = S_new
            path[:, i] = S_old.detach().squeeze(1)

            running_max = torch.max(running_max, S_old)
            if i in self.maturities:
                ind_maturity = self.maturities.index(i)
                for idx, strike in enumerate(self.strikes_call):
                    cv = cv_vanilla.view(-1, len(self.maturities), len(self.strikes_call))
                    price_vanilla = torch.exp(-self.rate * self.timegrid[i]) * torch.clamp(S_old - strike, 0).squeeze(1) - cv[:, ind_maturity, idx]
                    price_vanilla_cv[ind_maturity, idx] = price_vanilla.mean()
                    var_price_vanilla_cv[ind_maturity, idx] = price_vanilla.var()

        exotic_option_price = running_max - S_old
        error = torch.exp(-self.rate * self.timegrid[ind_T]) * exotic_option_price.detach() - torch.mean(torch.exp(-self.rate * self.timegrid[ind_T]) * exotic_option_price.detach()) - cv_exotics.detach()
        exotic_option_price = torch.exp(-self.rate * self.timegrid[ind_T]) * exotic_option_price - cv_exotics

        return price_vanilla_cv, var_price_vanilla_cv, exotic_option_price, exotic_option_price.mean(), exotic_option_price.var(), error

#censé etre mieux pour relu et silu 
def init_weights_lu(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')  
        if m.bias is not None:
            m.bias.data.zero_()
#censé etre mieux pour tanh
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=1.5)



def train_nsde(model, z_test, config,loss_calibration='MSE',init='xavier'): #METTRE loss_calibration='Wass' pour loss de Wasserstein (juste la calibration pas le hedging) et init='kaiming' pour init a kiaming, sinon parametre par defaut sont comme avant les changements 
        
    
    if loss_calibration=='Wass':
        loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.05)
    else:
        loss_fn = nn.MSELoss() 
    
    n_maturities = len(maturities)
    model = model.to(device)
    if init=='kaiming':
        model.apply(init_weights_lu)
    else:
        model.apply(init_weights)
    params_SDE = list(model.diffusion.parameters())#+list(model.driftV.parameters()) + list(model.diffusionV.parameters()) + [model.rho, model.v0]
    

    n_epochs = config["n_epochs"]
    T = config["maturities"][-1]
    # we take the target data that we are interested in
    target_mat_T = torch.tensor(config["target_data"][:len(config["maturities"]),:len(config["strikes_call"])], device=device).float()
    
    optimizer_SDE = torch.optim.Adam(params_SDE,lr=0.001)
    optimizer_CV = torch.optim.Adam(list(model.control_variate_vanilla.parameters()) + list(model.control_variate_exotics.parameters()),lr=0.001)
    scheduler_SDE = torch.optim.lr_scheduler.MultiStepLR(optimizer_SDE, milestones=[500,800], gamma=0.2)
    
    loss_val_best = 10
    itercount=0
    
    errors_hedge_2=torch.zeros(n_epochs)
    errors_hedge_inf=torch.zeros(n_epochs)
    for epoch in range(n_epochs):
        
        # We alternate Neural SDE optimisation and Hedging strategy optimisation
        requires_grad_CV = (epoch+1) % 2 == 0
        requires_grad_SDE = not requires_grad_CV
        if requires_grad_CV:
            model.control_variate_vanilla.unfreeze() 
            model.control_variate_exotics.unfreeze() 
            model.diffusion.freeze()

        else:
            model.diffusion.unfreeze()
            model.control_variate_vanilla.freeze()
            model.control_variate_exotics.freeze()
        
        
        
        batch_size = config["batch_size"]
        
        # we go through an epoch, i.e. 20*batch size paths
        for i in range(0,20*batch_size, batch_size):
            batch_z = torch.randn(batch_size, config["n_steps"], device=device) # just me being paranoid to be sure that we have independent samples in the batch. Sampling from an antithetic dataset does not make sense to me 

            optimizer_SDE.zero_grad()
            optimizer_CV.zero_grad()
            
            init_time = time.time()
            pred, var, _, exotic_option_price, exotic_option_var, _ = model(S0, batch_z, batch_size,T, period_length=16)
            time_forward = time.time() - init_time

            itercount += 1
            if requires_grad_CV:
                loss = var.sum() + exotic_option_var
                init_time = time.time()
                loss.backward()
                time_backward = time.time() - init_time
                nn.utils.clip_grad_norm_(list(model.control_variate_vanilla.parameters()) + list(model.control_variate_exotics.parameters()), 3)
                optimizer_CV.step()
            else:
                Wass = loss_fn(pred, target_mat_T)
                loss = Wass
                init_time = time.time()
                loss.backward()
                nn.utils.clip_grad_norm_(params_SDE, 5)
                time_backward = time.time() - init_time
                optimizer_SDE.step()
        
        scheduler_SDE.step() 
        
        #evaluate and print RMSE validation error at the start of each epoch
        with torch.no_grad():
            pred, _, exotic_option_price, exotic_price_mean, exotic_price_var, error = model(S0, z_test, z_test.shape[0], T, period_length=16)
        
        
        # Exotic option price hedging strategy error
        error_hedge = error
        error_hedge_2 = torch.mean(error_hedge**2)
        errors_hedge_2[epoch] = error_hedge_2
        error_hedge_inf = torch.max(torch.abs(error_hedge))
        errors_hedge_inf[epoch] = error_hedge_inf
        with open("error_hedge.txt","a") as f:
            f.write("{},{:.4f},{:.4f},{:.4f}\n".format(epoch,error_hedge_2, error_hedge_inf,exotic_price_var.item()))
        if (epoch+1)%100 == 0:
            torch.save(error_hedge, "error_hedge.pth.tar")
        
        # Evaluation Error of calibration to vanilla option prices
        if loss_calibration=='Wass':
            loss_val= loss_fn(pred, target_mat_T)
        else:
            loss_val= torch.sqrt(loss_fn(pred,target_mat_T))

        with open("log_eval.txt","a") as f:
            f.write('{},{:.4e}\n'.format(epoch, loss_val.item()))
        
        # save checkpooint
        if loss_val < loss_val_best:
             model_best = model
             loss_val_best=loss_val
             type_bound = "no"#"lower" if args.lower_bound else "upper"
             filename = "Neural_SDE_exp{}_{}bound_maturity{}_AugmentedLagrangian.pth.tar".format(args.experiment,type_bound,T)
             checkpoint = {"state_dict":model.state_dict(),
                     "exotic_price_mean": exotic_price_mean,
                     "exotic_price_var":exotic_price_var,
                     "T":T,
                     "pred":pred,
                     "target_mat_T": target_mat_T}

             torch.save(checkpoint, filename)

        if loss_val.item() < 2e-5:
            break

    n=range(1,n_epochs+1)
    print('\n Loss for calibration of NSDE coefficient:', loss_calibration, '\n Activation fonction: ' ,model.activation)
    plt.figure(figsize=(3, 5))  
    plt.plot(n, errors_hedge_2.cpu().numpy(), label="Error Hedge 2")
    plt.plot(n, errors_hedge_inf.cpu().numpy(), label="Error Hedge Inf")
    plt.xlabel("Nombre d'epochs")
    plt.ylabel("Erreur")
    plt.legend()
    plt.grid(True)
    plt.savefig("plot2.png", dpi=300)
    plt.show()

    return model_best




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--vNetWidth', type=int, default=50)
    parser.add_argument('--experiment', type=int, default=0)

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = f'cuda:{args.device}'
        torch.cuda.set_device(args.device)
    else:
        device = "cpu"

    # Setup training parameters
    strikes_call = np.arange(0.8, 1.21, 0.02)
    n_steps = 96
    timegrid = torch.linspace(0, 1, n_steps+1).to(device)
    maturities = range(16, 65, 16)
    n_maturities = len(maturities)

    # Parameters for data generation
    S0 = 1.0
    rate = 0.025  # risk-free rate
    sigma = 0.2   # Black-Scholes volatility

    # Heston model parameters
    kappa = 2.0   # mean reversion speed
    theta = 0.04  # long-term variance
    sigma = 0.3   # volatility of variance
    rho = -0.7    # correlation
    nu0 = 0.04    # initial variance

    # Generate training data using Heston model
    training_data = generate_jump_diffusion_test_data(
        S0=S0, 
        strikes=strikes_call, 
        maturities=maturities, 
        r=rate,
        sigma=sigma,
        jump_lambda=0.1,
        jump_mean=0.0,
        jump_std=0.1
    )
    training_data = training_data.to(device)  # Move to correct device

    # Initialize model
    model = Net_LV_Improved(
        dim=1, 
        timegrid=timegrid, 
        strikes_call=strikes_call, 
        n_layers=args.n_layers, 
        vNetWidth=args.vNetWidth, 
        device=device, 
        n_maturities=n_maturities, 
        maturities=maturities, 
        activation='silu',
        rate=rate
    )
    model.to(device)
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)  # Xavier initialization for weight
            if m.bias is not None:
                nn.init.zeros_(m.bias)  # Initialize bias to zero
        elif isinstance(m, nn.Conv2d):  # If using convolutional layers
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model.apply(init_weights)

    # Monte Carlo test data
    MC_samples_test = 200000
    z_test = torch.randn(MC_samples_test, n_steps, device=device)
    z_test = torch.cat([z_test, -z_test], 0)

    # Configuration for training
    CONFIG = {
        "batch_size": 2000,
        "n_epochs": 50,
        "maturities": maturities,
        "n_maturities": n_maturities,
        "strikes_call": strikes_call,
        "timegrid": timegrid,
        "n_steps": n_steps,
        "target_data": training_data  # Use our generated Black-Scholes data
    }

    # Train the model
    model = train_nsde(model, z_test, CONFIG,loss_calibration='MSE',init='kaiming') #loss_calibration='Wass' sinon par defaut MSE #init='kaiming' sinon par defaut xavier
    # Test the model
    tester = ModelTester(model, device)

    # Generate test data using Heston (with slightly different parameters to test generalization)
    test_data = generate_jump_diffusion_test_data(
        S0=S0, 
        strikes=strikes_call, 
        maturities=maturities, 
        r=rate,
        sigma=sigma,
        jump_lambda=0.1,
        jump_mean=0.0,
        jump_std=0.1
    )
    test_data = test_data.to(device)

    # Modified test function to use Black-Scholes test data
    def run_jump_test(model, S0, test_data, z_test, strikes, maturities, rate):
        vanilla_results = tester.test_vanilla_pricing(S0, test_data, z_test, max(maturities))
        exotic_results = tester.test_exotic_pricing(S0, z_test, max(maturities))
        
        return {
            'vanilla_results': vanilla_results,
            'exotic_results': exotic_results,
            'test_data': test_data.cpu().numpy()
        }

    # Run tests
    test_results = run_jump_test(
        model,
        S0=S0,
        test_data=test_data,
        z_test=z_test,
        strikes=strikes_call,
        maturities=maturities,
        rate=rate
    )

    # Save test results
    def save_test_results(results, filename="test_results_jump_nn.txt"):
        with open(filename, "w") as f:
            f.write("Vanilla Option Testing Results:\n")
            f.write(f"MSE: {results['vanilla_results']['mse']}\n")
            f.write(f"Relative Error: {results['vanilla_results']['relative_error']}\n")
            
            f.write("\nExotic Option Testing Results:\n")
            f.write(f"Mean Price: {results['exotic_results']['mean_price']}\n")
            f.write(f"Price Variance: {results['exotic_results']['variance']}\n")

    save_test_results(test_results)

    # Plot results
    def plot_results(results):
        import matplotlib.pyplot as plt
        
        # Plot vanilla option predictions vs true values
        plt.figure(figsize=(10, 6))
        plt.scatter(results['vanilla_results']['true_values'], 
                   results['vanilla_results']['predictions'])
        plt.plot([0, 1], [0, 1], 'r--')  # Perfect prediction line
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('Model Predictions vs True Values')
        plt.savefig('prediction_comparison_jump_nn.png')
        plt.close()

    plot_results(test_results)