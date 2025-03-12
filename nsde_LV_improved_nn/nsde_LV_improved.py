import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from random import randrange
import copy
import argparse

import sys 
import os 
# Add parent directory to Python path to import networks
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networks2 import *  # Ensure your custom networks (e.g., Net_timegrid) are defined appropriately.


# Modified Architecture for Neural SDE
class Net_LV_Improved(nn.Module):
    """
    Calibration of LV model: dS_t = S_t*r*dt + L(t,S_t,theta)dW_t to vanilla prices at different maturities
    Improved architecture with residual connections and deeper layers
    """
    def __init__(self, dim, timegrid, strikes_call,  n_layers, vNetWidth, device, rate, maturities, n_maturities):
        super(Net_LV_Improved, self).__init__()

        self.dim = dim
        self.timegrid = timegrid
        self.device = device
        self.strikes_call = strikes_call
        self.maturities = maturities
        self.rate = rate

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


def train_nsde_improved(model, z_test, config):
    """
    Training with the updated architecture.
    """

    loss_fn = nn.MSELoss()
    n_maturities = len(maturities)
    model = model.to(device)
    model.apply(init_weights)

    n_epochs = config["n_epochs"]
    T = config["maturities"][-1]

    target_mat_T = torch.tensor(config["target_data"][:len(config["maturities"]), :len(config["strikes_call"])], device=device).float()

    optimizer_SDE = torch.optim.Adam(model.diffusion.parameters(), lr=0.001)
    optimizer_CV = torch.optim.Adam(list(model.control_variate_vanilla.parameters()) + list(model.control_variate_exotics.parameters()), lr=0.001)
    scheduler_SDE = torch.optim.lr_scheduler.MultiStepLR(optimizer_SDE, milestones=[500, 800], gamma=0.2)

    loss_val_best = 10
    itercount = 0

    for epoch in range(n_epochs):
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

        print(f'Epoch {epoch}/{n_epochs}')

        batch_size = config["batch_size"]

        for i in range(0, 20 * batch_size, batch_size):
            batch_z = torch.randn(batch_size, config["n_steps"], device=device)

            optimizer_SDE.zero_grad()
            optimizer_CV.zero_grad()

            init_time = time.time()
            pred, var, _, exotic_option_price, exotic_option_var, _ = model(S0, batch_z, batch_size, T, period_length=16)
            time_forward = time.time() - init_time

            itercount += 1
            if requires_grad_CV:
                loss = var.sum() + exotic_option_var
                loss.backward()
                nn.utils.clip_grad_norm_(list(model.control_variate_vanilla.parameters()) + list(model.control_variate_exotics.parameters()), 3)
                optimizer_CV.step()
            else:
                MSE = loss_fn(pred, target_mat_T)
                loss = MSE
                loss.backward()
                nn.utils.clip_grad_norm_(model.diffusion.parameters(), 5)
                optimizer_SDE.step()

            time_backward = time.time() - init_time
            print(f'Iteration {itercount}, Loss={loss.item():.4f}, Time Forward={time_forward:.4f}, Time Backward={time_backward:.4f}')

        scheduler_SDE.step()

        with torch.no_grad():
            pred, _, exotic_option_price, exotic_price_mean, exotic_price_var, error = model(S0, z_test, z_test.shape[0], T, period_length=16)
            print("Validation Prediction:", pred)
            print("Target Data:", target_mat_T)

        error_hedge = error
        error_hedge_2 = torch.mean(error_hedge ** 2)
        error_hedge_inf = torch.max(torch.abs(error_hedge))

        with open("error_hedge_improved.txt", "a") as f:
            f.write(f"{epoch},{error_hedge_2},{error_hedge_inf},{exotic_price_var.item()}\n")

        if (epoch + 1) % 100 == 0:
            torch.save(error_hedge, "error_hedge.pth.tar")

        MSE = loss_fn(pred, target_mat_T)
        loss_val = torch.sqrt(MSE)
        print(f"Epoch {epoch}, Loss {loss_val.item():.4f}")

        with open("log_train.txt", "a") as f:
            f.write(f"Epoch {epoch}, Loss {loss_val.item():.4f}\n")

        if loss_val < loss_val_best:
            model_best = model
            loss_val_best = loss_val
            print('New Best Model Found:', loss_val_best)
            filename = f"Neural_SDE_Improved_{epoch}_AugmentedLagrangian.pth.tar"
            checkpoint = {
                "state_dict": model.state_dict(),
                "exotic_price_mean": exotic_price_mean,
                "exotic_price_var": exotic_price_var,
                "T": T,
                "pred": pred,
                "target_mat_T": target_mat_T
            }
            torch.save(checkpoint, filename)

        if loss_val.item() < 2e-5:
            break
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

    # Load market prices and set training target
    data = torch.load("Call_prices_59.pt", weights_only=False)

    # Setup training parameters
    strikes_call = np.arange(0.8, 1.21, 0.02)
    n_steps = 96
    timegrid = torch.linspace(0, 1, n_steps+1).to(device)
    maturities = range(16, 65, 16)
    n_maturities = len(maturities)

    S0 = 1
    rate = 0.025  # risk-free rate
    model = Net_LV_Improved(dim=1, timegrid=timegrid, strikes_call=strikes_call, n_layers=args.n_layers, vNetWidth=args.vNetWidth, device=device, n_maturities=n_maturities, maturities=maturities, rate=rate)
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

    CONFIG = {
        "batch_size": 4000,
        "n_epochs": 100,
        "maturities": maturities,
        "n_maturities": n_maturities,
        "strikes_call": strikes_call,
        "timegrid": timegrid,
        "n_steps": n_steps,
        "target_data": data
    }

    model = train_nsde_improved(model, z_test, CONFIG)
