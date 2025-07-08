import torch

class Hypernet(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        # basis network: two FC layers to expand dummy input
        self.basis = torch.nn.Sequential(
            torch.nn.Linear(1, 50), torch.nn.ReLU(),
            torch.nn.Linear(50, 50), torch.nn.ReLU()
        )
        # heads for μ, logvar, p
        self.mu_net     = torch.nn.Sequential(
            torch.nn.Linear(50, 20), torch.nn.ReLU(),
            torch.nn.Linear(20, 1)
        )
        self.logvar_net = torch.nn.Sequential(
            torch.nn.Linear(50, 20), torch.nn.ReLU(),
            torch.nn.Linear(20, 1)
        )
        self.p_net      = torch.nn.Sequential(
            torch.nn.Linear(50, 20), torch.nn.ReLU(),
            torch.nn.Linear(20, 1), torch.nn.Sigmoid()
        )
        self.shape = shape

    def forward(self, k=0, dummy=1.0):
        # Get parameters
        z = self.basis(dummy.unsqueeze(0))  # shape (1,50)
        mu = self.mu_net(z).view(self.shape)
        logvar = self.logvar_net(z).view(self.shape)
        p  = self.p_net(z).view(self.shape)
        
        # Reparameterization
        sigma = (logvar * 0.5).exp()
        epsilon = torch.empty(()).normal_()
        W_sampled = mu + sigma * epsilon
        
        # Gating
        threshold = torch.empty(()).uniform_()
        gate = torch.sigmoid(k * (p - threshold))
        W = W_sampled * gate
        
        return W
    
    def forward_inference(self, dummy=1.0):
        # Get parameters
        z = self.basis(dummy.unsqueeze(0))  # shape (1,50)
        mu = self.mu_net(z).view(self.shape)
        p  = self.p_net(z).view(self.shape)

        # No reparameterization
        W_unsampled = mu

        # Gating
        gate = (p >= 0.5).float()  # hard 0/1
        W = W_unsampled * gate

        # 4) linear output
        return W
