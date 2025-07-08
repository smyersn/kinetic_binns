# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from modules.binn_eql_hypernet.hypernet import Hypernet
       
# class HillFunction(nn.Module):
#     def __init__(self, param_bounds, increasing=True):
#         super(HillFunction, self).__init__()
        
#         self.increasing = increasing

#         self.pn = Hypernet(0, 5, hill=True)
#         self.pK = Hypernet(0, param_bounds, hill=True)

#     def forward(self, x, k):
#         # Get raw parameter values
#         n, _ = self.pn(x.device, k)
#         K, _ = self.pK(x.device, k)

#         x_n = x.pow(n)
        
#         if self.increasing:
#             hill_feat = x_n / (1 + K * x_n)          
#         else:
#             hill_feat = (1 / K) - (x_n / (1 + K * x_n))
 
#         return hill_feat

# # Generate polynomial features for an arbitrary number of proteins.
# class PolynomialFeatures(nn.Module):
#     def __init__(self, species):
#         super(PolynomialFeatures, self).__init__()
#         self.species = species

#     def forward(self, x):
#         # x: [batch, N]
#         features = [x]          # linear terms
#         features.append(x**2)     # squared terms
#         # Cross terms: only include for i < j to avoid redundancy.
#         cross_terms = []
#         for i in range(self.species):
#             for j in range(i+1, self.species):
#                 cross_terms.append((x[:, i:i+1] * x[:, j:j+1]))
#         if cross_terms:
#             features.append(torch.cat(cross_terms, dim=1))
            
#         # Concatenate polynomial features
#         poly_feats = torch.cat(features, dim=1)

#         return poly_feats
    
# class HillFeatures(nn.Module):
#     def __init__(self, species, param_bounds):
#         super(HillFeatures, self).__init__()
#         self.species = species
        
#         # Raw Hill functions: one per protein.
#         self.hill_inc_raw = nn.ModuleList([HillFunction(param_bounds, increasing=True) for _ in range(species)])
#         self.hill_dec_raw = nn.ModuleList([HillFunction(param_bounds, increasing=False) for _ in range(species)])
        
#         # Cross-term Hill functions: one for each pair (i,j) where i != j.
#         self.hill_inc_cross = nn.ModuleDict()
#         self.hill_dec_cross = nn.ModuleDict()
#         for i in range(species):
#             for j in range(species):
#                 if i != j:
#                     self.hill_inc_cross[f"{i}_{j}"] = HillFunction(param_bounds, increasing=True)
#                     self.hill_dec_cross[f"{i}_{j}"] = HillFunction(param_bounds, increasing=False)
    
#     def forward(self, x, k):
#         # x has shape [batch, species]
#         inc_features = []
#         dec_features = []
        
#         # Raw features: hill(x_i)
#         for i in range(self.species):
#             xi = x[:, i:i+1]
#             inc_features.append(self.hill_inc_raw[i](xi, k))
#             dec_features.append(self.hill_dec_raw[i](xi, k))
        
#         # Cross features: hill(x_i) * x_j for i != j, using separate parameters.
#         for i in range(self.species):
#             for j in range(self.species):
#                 if i != j:
#                     xi = x[:, i:i+1]
#                     xj = x[:, j:j+1]
#                     inc_features.append(self.hill_inc_cross[f"{i}_{j}"](xi, k) * xj)
#                     dec_features.append(self.hill_dec_cross[f"{i}_{j}"](xi, k) * xj)
        
#         # Concatenate increasing and decreasing hill features
#         hill_feats = torch.cat(inc_features + dec_features, dim=1)
        
#         return hill_feats

# # EQL Layer that combines polynomial and hill features.
# class EQLLayer(nn.Module):
#     def __init__(self, species, param_bounds):
#         super(EQLLayer, self).__init__()
#         self.poly = PolynomialFeatures(species)
#         self.hill = HillFeatures(species, param_bounds)

#         # Compute feature sizes
#         self.n_poly = (species +                                # linear terms
#                        species +                                # squared terms
#                        (species * (species - 1)) // 2           # cross terms
#                         )
        
#         self.n_hill = 2 * (species +                         # raw Hill terms
#                            species * (species - 1)           # cross Hill terms  
#                            )

#         self.n_features = self.n_poly + self.n_hill
                
#         # One Hypernet per scalar weight
#         self.hypernets = nn.ModuleList([
#             Hypernet(-param_bounds, param_bounds) for _ in range(self.n_features)])

#     def forward(self, x, k=0, inference=False):
#         poly_feats = self.poly(x)
#         hill_feats = self.hill(x, k)
#         features = torch.cat([poly_feats, hill_feats], dim=1)
        
#         # Sample each weight from its hypernet
#         weights = []
#         ps = []

#         for hypernet in self.hypernets:
#             if inference:
#                 mu, p, w = hypernet.get_params(x.device)
#                 weights.append(torch.tensor(w).unsqueeze(0))
#                 ps.append(torch.tensor(p).unsqueeze(0))
#             else:
#                 w, p = hypernet(x.device, k)  # scalar tensor [1]
#                 weights.append(w)
#                 ps.append(p)

#         # Stack and reshape to (1, total_features)
#         weight_tensor = torch.cat(weights, dim=0).view(1, -1)  # [1, total_features]
#         p_tensor = torch.cat(ps, dim=0)

#         # Linear combination with sampled weights
#         out = F.linear(features, weight_tensor, bias=None)  # [B, 1]

#         return out, p_tensor   

import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.binn_eql_hypernet.hypernet import Hypernet
       
class HillFunction(nn.Module):
    def __init__(self, param_bounds, increasing=True):
        super(HillFunction, self).__init__()
        
        self.pn = Hypernet(-10, 10, hill=True)
        self.pK = Hypernet(-10, 10, hill=True)

        self.param_bounds = param_bounds
        self.increasing = increasing

    def forward(self, x, k):
        # Get raw parameter values (can be negative)
        n_raw = self.pn(x.device)[0]
        K_raw = self.pK(x.device)[0]
        
        # Scales values from 0-5 and 0-param_bounds, respectively
        n = torch.sigmoid(n_raw) * 5
        K = torch.sigmoid(K_raw) * self.param_bounds

        x_n = x.pow(n)
        
        if self.increasing:
            hill_feat = x_n / (1 + K * x_n)          
        else:
            hill_feat = (1 / K) - (x_n / (1 + K * x_n))
            
        if torch.isnan(hill_feat).any(): 
            print('Hill nan:')
            print(f'nraw, kraw: {n_raw, K_raw}')
            print(f'n, k: {n, K}')
 
        return hill_feat

# Generate polynomial features for an arbitrary number of proteins.
class PolynomialFeatures(nn.Module):
    def __init__(self, species):
        super(PolynomialFeatures, self).__init__()
        self.species = species

    def forward(self, x):
        # x: [batch, N]
        features = [x]          # linear terms
        features.append(x**2)     # squared terms
        # Cross terms: only include for i < j to avoid redundancy.
        cross_terms = []
        for i in range(self.species):
            for j in range(i+1, self.species):
                cross_terms.append((x[:, i:i+1] * x[:, j:j+1]))
        if cross_terms:
            features.append(torch.cat(cross_terms, dim=1))
            
        # Concatenate polynomial features
        poly_feats = torch.cat(features, dim=1)
        
        if torch.isnan(poly_feats).any(): 
            nan_columns = torch.isnan(poly_feats).any(dim=0)  # Boolean mask of shape [num_features]
            nan_indices = nan_columns.nonzero(as_tuple=True)[0]  # Indices of columns with NaNs
            print('Poly nan:')
            print("Poly nan columns:", nan_indices.tolist())

        return poly_feats
    
class HillFeatures(nn.Module):
    def __init__(self, species, param_bounds):
        super(HillFeatures, self).__init__()
        self.species = species
        
        # Raw Hill functions: one per protein.
        self.hill_inc_raw = nn.ModuleList([HillFunction(param_bounds, increasing=True) for _ in range(species)])
        self.hill_dec_raw = nn.ModuleList([HillFunction(param_bounds, increasing=False) for _ in range(species)])
        
        # Cross-term Hill functions: one for each pair (i,j) where i != j.
        self.hill_inc_cross = nn.ModuleDict()
        self.hill_dec_cross = nn.ModuleDict()
        for i in range(species):
            for j in range(species):
                if i != j:
                    self.hill_inc_cross[f"{i}_{j}"] = HillFunction(param_bounds, increasing=True)
                    self.hill_dec_cross[f"{i}_{j}"] = HillFunction(param_bounds, increasing=False)
    
    def forward(self, x, k):
        # x has shape [batch, species]
        inc_features = []
        dec_features = []
        
        # Raw features: hill(x_i)
        for i in range(self.species):
            xi = x[:, i:i+1]
            inc_features.append(self.hill_inc_raw[i](xi, k))
            dec_features.append(self.hill_dec_raw[i](xi, k))
        
        # Cross features: hill(x_i) * x_j for i != j, using separate parameters.
        for i in range(self.species):
            for j in range(self.species):
                if i != j:
                    xi = x[:, i:i+1]
                    xj = x[:, j:j+1]
                    inc_features.append(self.hill_inc_cross[f"{i}_{j}"](xi, k) * xj)
                    dec_features.append(self.hill_dec_cross[f"{i}_{j}"](xi, k) * xj)
        
        # Concatenate increasing and decreasing hill features
        hill_feats = torch.cat(inc_features + dec_features, dim=1)
        
        if torch.isnan(hill_feats).any(): 
            nan_columns = torch.isnan(hill_feats).any(dim=0)  # Boolean mask of shape [num_features]
            nan_indices = nan_columns.nonzero(as_tuple=True)[0]  # Indices of columns with NaNs
            print('Hill nan:')
            print("Hill nan columns:", nan_indices.tolist())

        return hill_feats

# EQL Layer that combines polynomial and hill features.
class EQLLayer(nn.Module):
    def __init__(self, species, param_bounds):
        super(EQLLayer, self).__init__()
        self.poly = PolynomialFeatures(species)
        self.hill = HillFeatures(species, param_bounds)

        # Compute feature sizes
        self.n_poly = (species +                                # linear terms
                       species +                                # squared terms
                       (species * (species - 1)) // 2           # cross terms
                        )
        
        self.n_hill = 2 * (species +                         # raw Hill terms
                           species * (species - 1)           # cross Hill terms  
                           )

        self.n_features = self.n_poly + self.n_hill
                
        # One Hypernet per scalar weight
        self.hypernets = nn.ModuleList([
            Hypernet(-param_bounds, param_bounds) for _ in range(self.n_features)])

    def forward(self, x, k=0, inference=False):
        poly_feats = self.poly(x)
        hill_feats = self.hill(x, k)
        features = torch.cat([poly_feats, hill_feats], dim=1)
        
        # Sample each weight from its hypernet
        weights = []
        ps = []

        for hypernet in self.hypernets:
            if inference:
                w, p, mu, sigma = hypernet(x.device, k, inference=True)
                weights.append(w)
                ps.append(p)
            else:
                w, p, mu, sigma = hypernet(x.device, k, inference=False)
                weights.append(w)
                ps.append(p)

        # Stack and reshape to (1, total_features)
        weight_tensor = torch.cat(weights, dim=0).view(1, -1)  # [1, total_features]
        p_tensor = torch.cat(ps, dim=0)

        # Linear combination with sampled weights
        out = F.linear(features, weight_tensor, bias=None)  # [B, 1]

        return out, p_tensor   
