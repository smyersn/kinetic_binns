import torch

class BoundScale(torch.nn.Module):
    def __init__(self, min_val, max_val):
        super().__init__()
        self.min = min_val
        self.range = max_val - min_val

    def forward(self, x):
        # x assumed already in [0,1] from sigmoid
        return self.min + self.range * x

class Hypernet(torch.nn.Module):
    def __init__(self, param_min, param_max, hill=False):
        super().__init__()
        # No gating for Hill parameters (Hill = True)
        self.hill = hill
        
        # basis network: two FC layers to expand dummy input
        self.basis = torch.nn.Sequential(
            torch.nn.Linear(1, 50), torch.nn.ReLU(),
            torch.nn.Linear(50, 50), torch.nn.ReLU()
        )
        # heads for μ, logvar, p
        self.mu_net     = torch.nn.Sequential(
            torch.nn.Linear(50, 20), torch.nn.ReLU(),
            torch.nn.Linear(20, 1), torch.nn.Sigmoid(),
            BoundScale(param_min, param_max)
        )
        self.logvar_net = torch.nn.Sequential(
            torch.nn.Linear(50, 20), torch.nn.ReLU(),
            torch.nn.Linear(20, 1)
        )
        self.p_net      = torch.nn.Sequential(
            torch.nn.Linear(50, 20), torch.nn.ReLU(),
            torch.nn.Linear(20, 1), torch.nn.Sigmoid()
        )
        
    def forward(self, device, k=0, inference=False):
        # Create dummy input
        dummy = torch.tensor(1).float().to(device)
        
        # Get parameters
        z = self.basis(dummy.unsqueeze(0))
        mu = self.mu_net(z)
        logvar = self.logvar_net(z)
        
        # Reparameterization
        sigma = (logvar * 0.5).exp()
        epsilon = torch.empty(()).normal_()
        W_sampled = mu + sigma * epsilon
        
        # Gating
        if self.hill:
            p = torch.tensor(0)
            W = W_sampled
        else:
            if inference:
                p  = self.p_net(z)
                gate = (p >= 0.5).float()  # hard 0/1
                W = mu * gate
            else:
                p  = self.p_net(z)
                threshold = torch.empty(()).uniform_()
                gate = torch.sigmoid(k * (p - threshold))
                W = W_sampled * gate
        
        return W, p, mu, sigma