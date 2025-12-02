# import math, torch, torch.nn as nn

# class HardConcreteGate(nn.Module):
#     """
#     Stochastic hard-concrete gate per feature. Returns a gate in [0,1].
#     Uses parameters log_alpha (one per gate). Expected L0 penalty approximated by sigmoid(log_alpha).
#     Simpler & robust variant: expected open prob = sigmoid(log_alpha).
#     """
#     def __init__(self, n_gates, droprate_init=0.5, temp=2./3., low=-0.1, high=1.1):
#         super().__init__()
#         init_log_alpha = math.log(1. - droprate_init) - math.log(droprate_init)
#         self.log_alpha = nn.Parameter(torch.ones(n_gates) * init_log_alpha)  # shape [n_gates]
#         self.temp = float(temp)
#         self.low = float(low)
#         self.high = float(high)

#     def _sample_u(self, shape, device):
#         # avoid 0/1 exact
#         return torch.rand(shape, device=device).clamp(1e-6, 1-1e-6)

#     def _concrete_sample(self, training):
#         if training:
#             u = self._sample_u(self.log_alpha.shape, self.log_alpha.device)
#             s = torch.sigmoid((torch.log(u) - torch.log(1. - u) + self.log_alpha) / self.temp)
#             # stretch to [low, high]
#             s_bar = s * (self.high - self.low) + self.low
#             z = s_bar.clamp(0.0, 1.0)
#             return z  # continuous in [0,1]
#         else:
#             # deterministic at eval: use open probability (soft) or hard threshold
#             p = torch.sigmoid(self.log_alpha)
#             # option A: return p (soft)
#             # return p
#             # option B: hard threshold:
#             return (p >= 0.5).float()

#     def forward(self, training=True):
#         # Return gate of shape [1, n_gates] (broadcastable to fc.weight)
#         z = self._concrete_sample(training)
#         return z.view(1, -1)

#     def expected_l0(self):
#         # Simple surrogate: expected gate-open prob = sigmoid(log_alpha)
#         # multiply by lambda in loss
#         return torch.sigmoid(self.log_alpha).sum()
#         # factor = - (self.gamma) / (self.zeta)
#         # term = self.log_alpha - self.beta * torch.log(torch.tensor(factor, device=self.log_alpha.device, dtype=self.log_alpha.dtype))
#         # return torch.sigmoid(term).sum()

import torch
import torch.nn as nn
import torch.nn.functional as F

class HardConcreteGate(nn.Module):
    """
    Hard-Concrete gate (Louizos et al., 2018) with:
      - sampling in forward() during training
      - deterministic gate in eval()
      - optional hard (binary) straight-through discretization
      - expected-L0 analytic estimate for the L0 penalty
    
    Usage:
        gate = HardConcreteGate(num_gates)
        z = gate(sample=True, hard=False)   # training: stochastic continuous in (0,1)
        z = gate(sample=False)              # eval: deterministic continuous (0,1)
        z = gate(sample=True, hard=True)    # training: binary 0/1 with STE
    """
    def __init__(self, num_gates, 
                 init_log_alpha=2.0,   # start with gates mostly "open" (positive -> sigmoid>0.88)
                 beta=2/3,
                 gamma=-0.1,
                 zeta=1.1,
                 eps=1e-6,
                 device=None,
                 dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        # learnable log-alpha parameter controls gate probability
        self.log_alpha = nn.Parameter(torch.full((num_gates,), float(init_log_alpha), **factory_kwargs))
        # hard-concrete hyperparams (paper defaults used commonly)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.zeta = float(zeta)
        self.eps = float(eps)
    
    def _sample_uniform(self, shape, device):
        """Draw U ~ Uniform(eps, 1-eps)"""
        return torch.empty(shape, device=device).uniform_(self.eps, 1.0 - self.eps)
    
    def _concrete_sample(self, u):
        """
        Concrete (stretched sigmoid) sample from logistic noise reparameterization.
        u: uniform sample of same shape as log_alpha.
        returns continuous gate values in (0,1) after stretching+clamping.
        """
        # logistic transform of u
        logit_u = torch.log(u) - torch.log1p(-u)  # log(u/(1-u))
        # pre-sigmoid
        pre_sigmoid = (logit_u + self.log_alpha) / self.beta
        s = torch.sigmoid(pre_sigmoid)
        s_stretched = s * (self.zeta - self.gamma) + self.gamma
        z = torch.clamp(s_stretched, min=0.0, max=1.0)
        return z
    
    def forward(self, sample=True, hard=False):
        """
        Forward pass.
        Args:
            sample (bool): if True (training mode), sample stochastic gates;
                           if False (eval/deterministic), use deterministic gate.
            hard (bool): if True, create hard binary gates (0 or 1) with straight-through
                         estimator (useful if you want discrete gates during forward while
                         keeping gradients through continuous relaxation).
        Returns:
            z (Tensor): shape (num_gates,) with values in [0,1] (or {0,1} if hard=True).
        """
        device = self.log_alpha.device
        if self.training and sample:
            # stochastic continuous sample
            u = self._sample_uniform(self.log_alpha.shape, device)
            z = self._concrete_sample(u)
        else:
            # deterministic continuous relaxation (use the mode/mean proxy)
            # We use sigmoid(log_alpha) stretched and clamped as deterministic proxy.
            s = torch.sigmoid(self.log_alpha)
            s_stretched = s * (self.zeta - self.gamma) + self.gamma
            z = torch.clamp(s_stretched, min=0.0, max=1.0)
        
        if hard:
            # Straight-through estimator: forward passes binary but backward flows through z (continuous)
            z_hard = (z >= 0.5).to(z.dtype)
            # STE trick: replace forward value with z_hard but let gradients flow to z
            z = z_hard.detach() - z.detach() + z
        return z
    
    def get_gates(self):
        """
        Deterministic gate values (use for logging / extracting current gate strengths).
        Equivalent to forward(sample=False).
        """
        return self.forward(sample=False, hard=False).detach()
    
    def get_binary_mask(self, threshold=0.5):
        """
        Hard mask: returns binary mask (0/1) from deterministic gate values using threshold.
        Use when you want permanent pruning decisions.
        """
        g = self.get_gates()
        return (g >= threshold).to(dtype=g.dtype)
    
    def expected_l0(self):
        """
        Returns a vector of probabilities P(z > 0) for each gate.
        """
        factor = - (self.gamma) / (self.zeta)
        
        # Defensive check for broken config
        if factor <= 0:
            return torch.sigmoid(self.log_alpha * 0.0 + 1e6)
            
        term = self.log_alpha - self.beta * torch.log(torch.tensor(factor, device=self.log_alpha.device, dtype=self.log_alpha.dtype))
        
        # CHANGE: Return the vector, do NOT sum here.
        return torch.sigmoid(term)