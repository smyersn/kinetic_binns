import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.binn_eql.hard_concrete_gate import HardConcreteGate
       
class HillFunction(nn.Module):
    def __init__(self, param_bounds, increasing=True):
        super(HillFunction, self).__init__()
        self.param_bounds = param_bounds
        self.increasing = increasing
        
        # n is strictly bounded [1, 4] via Sigmoid
        self.raw_n = nn.Parameter(torch.empty(1).uniform_(-4, 4))
        
        # K is Unbounded positive (Softplus). 
        # The Soft Wall Loss will constrain its physical value.
        self.raw_K = nn.Parameter(torch.empty(1).uniform_(-2, 2))

    def forward(self, x):
        n = torch.sigmoid(self.raw_n) * 3 + 1
        
        # Softplus ensures K > 0 but allows it to grow large (e.g. 1000)
        # This is necessary because it takes normalized inputs (u/S)
        # K_net ~ K_phys * S^n
        K = F.softplus(self.raw_K)
        
        x_n = x.pow(n)
        
        if self.increasing:
            return x_n / (1 + K * x_n)
        else:
            # Epsilon for stability
            return (1 / (K + 1e-8)) - (x_n / (1 + K * x_n))

class PolynomialFeatures(nn.Module):
    def __init__(self, species, duplicates):
        super(PolynomialFeatures, self).__init__()
        self.species = species
        self.duplicates = duplicates

    def forward(self, x):
        features = [x]                  # linear terms
        features.append(x**2)           # squared terms
        
        cross_terms = []
        for i in range(self.species):
            for j in range(i+1, self.species):
                cross_terms.append((x[:, i:i+1] * x[:, j:j+1]))
        if cross_terms:
            features.append(torch.cat(cross_terms, dim=1))
            
        return torch.cat(features * self.duplicates, dim=1)

class HillFeatures(nn.Module):
    def __init__(self, species, param_bounds):
        super(HillFeatures, self).__init__()
        self.num_proteins = species
        
        self.hill_inc_raw = nn.ModuleList([HillFunction(param_bounds, increasing=True) for _ in range(species)])
        self.hill_dec_raw = nn.ModuleList([HillFunction(param_bounds, increasing=False) for _ in range(species)])
        
        self.hill_inc_cross = nn.ModuleDict()
        self.hill_dec_cross = nn.ModuleDict()
        for i in range(species):
            for j in range(species):
                if i != j:
                    self.hill_inc_cross[f"{i}_{j}"] = HillFunction(param_bounds, increasing=True)
                    self.hill_dec_cross[f"{i}_{j}"] = HillFunction(param_bounds, increasing=False)
    
    def forward(self, x):
        inc_features = []
        dec_features = []
        
        for i in range(self.num_proteins):
            xi = x[:, i:i+1]
            inc_features.append(self.hill_inc_raw[i](xi))
            dec_features.append(self.hill_dec_raw[i](xi))
        
        for i in range(self.num_proteins):
            for j in range(self.num_proteins):
                if i != j:
                    xi = x[:, i:i+1]
                    xj = x[:, j:j+1]
                    inc_features.append(self.hill_inc_cross[f"{i}_{j}"](xi) * xj)
                    dec_features.append(self.hill_dec_cross[f"{i}_{j}"](xi) * xj)
                    
        return torch.cat(inc_features + dec_features, dim=1)

class DuplicateHillFeatures(nn.Module):
    def __init__(self, species, param_bounds, duplicates):
        super(DuplicateHillFeatures, self).__init__()
        self.hill_modules = nn.ModuleList([
            HillFeatures(species, param_bounds) for _ in range(duplicates)
        ])

    def forward(self, x):
        features = [module(x) for module in self.hill_modules]
        return torch.cat(features, dim=1)
    
# EQL Layer that combines polynomial and hill features.
class EQLLayer(nn.Module):
    def __init__(self, species, duplicates, param_bounds, max_scale):
        super(EQLLayer, self).__init__()
        self.species = species
        self.duplicates = duplicates
        self.param_bounds = param_bounds
        
        # Max scale [s_u, s_v] used for physical conversion in Loss
        self.register_buffer('max_scale', max_scale)

        self.poly = PolynomialFeatures(species, duplicates)
        self.hill = DuplicateHillFeatures(species, param_bounds, duplicates)

        # Feature Counts
        self.num_poly_features = duplicates * (species + species + (species * (species - 1)) // 2)
        self.num_hill_features = duplicates * 2 * (species + species * (species - 1))
        self.total_features = self.num_poly_features + self.num_hill_features
                
        self.fc = nn.Linear(self.total_features, 1, bias=False)
        self.l0_gate = HardConcreteGate(self.total_features)
        
        # Initialize small. 
        # Since we removed constraints, weights can grow as needed.
        nn.init.uniform_(self.fc.weight, a=-0.1, b=0.1)
        
    def forward(self, x):
        # 1. Standard Forward Pass (FAST)
        # No scaling math, no tanh. Just feature generation and dot product.
        poly_feats = self.poly(x)
        hill_feats = self.hill(x)
        features = torch.cat([poly_feats, hill_feats], dim=1)
        
        w = self.fc.weight
        z = self.l0_gate()

        out = (features * (w * z)).sum(dim=1, keepdim=True)
        return out

    def get_physical_parameters(self):
        """
        Calculates Physical Weights and Physical K values.
        Called by the Loss Function to enforce Soft Walls.
        """
        # 1. Generate Scales (S^n)
        scales = self._generate_scales()
        
        # 2. Physical Weights = w_net / S^n
        w_net = self.fc.weight.view(-1)
        w_phys = w_net / (scales + 1e-8)
        
        # 3. Physical K = K_net / S^n
        # We need to gather K_net and its corresponding scale
        k_phys_list = []
        
        s = self.max_scale[0]
        
        for hm in self.hill.hill_modules:
            # Helper to unscale K
            def unscale_k(module, scale_base):
                # K_net is Softplus
                k_net = F.softplus(module.raw_K)
                n = torch.sigmoid(module.raw_n) * 3 + 1
                # K_phys = K_net / S^n
                return k_net / (scale_base ** n)

            # Inc Raw
            for i in range(self.species):
                k_phys_list.append(unscale_k(hm.hill_inc_raw[i], s[i]))
            # Inc Cross
            for i in range(self.species):
                for j in range(self.species):
                    if i!=j: 
                        k_phys_list.append(unscale_k(hm.hill_inc_cross[f"{i}_{j}"], s[i]))
            # Dec Raw
            for i in range(self.species):
                k_phys_list.append(unscale_k(hm.hill_dec_raw[i], s[i]))
            # Dec Cross
            for i in range(self.species):
                for j in range(self.species):
                    if i!=j: 
                        k_phys_list.append(unscale_k(hm.hill_dec_cross[f"{i}_{j}"], s[i]))

        k_phys_tensor = torch.cat([k.view(1) for k in k_phys_list])
        
        return w_phys, k_phys_tensor

    def _generate_scales(self):
        """Internal helper to generate S^n for all terms"""
        s = self.max_scale[0]
        scales_list = []

        # Poly Scales
        poly_block = []
        for i in range(self.species): poly_block.append(s[i])
        for i in range(self.species): poly_block.append(s[i]**2)
        for i in range(self.species):
            for j in range(i+1, self.species): poly_block.append(s[i]*s[j])
        
        scales_list.extend(poly_block * self.duplicates)

        # Hill Scales
        for hm in self.hill.hill_modules:
            # Helper to get scale
            def get_scale(module, base_scale, mult_scale=1.0):
                n = torch.sigmoid(module.raw_n.view(())) * 3 + 1
                return (base_scale ** n) * mult_scale

            # Inc Raw
            for i in range(self.species): scales_list.append(get_scale(hm.hill_inc_raw[i], s[i]))
            # Inc Cross
            for i in range(self.species):
                for j in range(self.species):
                    if i!=j: scales_list.append(get_scale(hm.hill_inc_cross[f"{i}_{j}"], s[i], s[j]))
            # Dec Raw
            for i in range(self.species): scales_list.append(get_scale(hm.hill_dec_raw[i], s[i]))
            # Dec Cross
            for i in range(self.species):
                for j in range(self.species):
                    if i!=j: scales_list.append(get_scale(hm.hill_dec_cross[f"{i}_{j}"], s[i], s[j]))

        return torch.stack(scales_list).view(1, -1)