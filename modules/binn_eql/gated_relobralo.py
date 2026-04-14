import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedReLoBRaLo:
    def __init__(self, num_losses=3, temperature=0.1, alpha=0.99, buffer_size=1000, device='cuda'):
        self.num_losses = num_losses
        self.T = temperature
        self.alpha = alpha
        self.buffer_size = buffer_size
        self.device = device
        
        # ALL state now lives permanently on the GPU
        self.L0 = None                 
        self.active_state = torch.zeros(num_losses, dtype=torch.bool, device=self.device)       
        self.lambdas = torch.ones(num_losses, device=self.device) 
        
        # Pre-allocated GPU Ring Buffer for history (Replaces the slow Python list)
        self.L_hist = torch.zeros((buffer_size, num_losses), device=self.device)
        self.hist_ptr = 0
        self.hist_filled = 0 # Tracks how many valid entries exist

    def compute_weights(self, current_losses, active_mask):
        current_losses = current_losses.detach()
        active_bool = active_mask > 0
        num_active = active_bool.sum().float()

        if num_active <= 1:
            return active_mask.to(self.device)
            
        if self.L0 is None:
            self.L0 = current_losses.clone()
            self.active_state.copy_(active_bool)
            
            # Store in ring buffer
            self.L_hist[self.hist_ptr] = current_losses
            self.hist_ptr = (self.hist_ptr + 1) % self.buffer_size
            self.hist_filled = 1
            
            return active_mask.to(self.device)
            
        # "The Awakening" Logic 
        just_woke_up = active_bool & (~self.active_state)
        if just_woke_up.any():
            self.L0[just_woke_up] = current_losses[just_woke_up].clone()
            
        self.active_state.copy_(active_bool)

        # Random Lookback from valid history using pure PyTorch
        idx = torch.randint(0, self.hist_filled, (1,), device=self.device)[0]
        L_lookback = self.L_hist[idx]
        
        rho_0 = current_losses / (self.L0 + 1e-8)
        rho_hist = current_losses / (L_lookback + 1e-8)
        
        logits_0 = rho_0 / self.T
        logits_hist = rho_hist / self.T
        
        logits_0[~active_bool] = -float('inf')
        logits_hist[~active_bool] = -float('inf')
        
        new_lambdas_0 = num_active * F.softmax(logits_0, dim=0)
        new_lambdas_hist = num_active * F.softmax(logits_hist, dim=0)
        
        lambda_hat = 0.5 * new_lambdas_0 + 0.5 * new_lambdas_hist
        
        # Pure GPU Exponential Moving Average (No more .cpu() syncs!)
        self.lambdas[active_bool] = self.alpha * self.lambdas[active_bool] + (1 - self.alpha) * lambda_hat[active_bool]
        self.lambdas[~active_bool] = 0.0 
        
        # Fast Ring Buffer Update
        self.L_hist[self.hist_ptr] = current_losses
        self.hist_ptr = (self.hist_ptr + 1) % self.buffer_size
        if self.hist_filled < self.buffer_size:
            self.hist_filled += 1
            
        return self.lambdas