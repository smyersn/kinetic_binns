import torch
import torch.nn as nn
import torch.nn.functional as F
       
class HillFunction(nn.Module):
    def __init__(self, param_bounds, increasing=True):
        super(HillFunction, self).__init__()
        
        self.param_bounds = param_bounds
        
        # Initialize raw_n and raw_K so n between 0 and 5 and K between 0 and param_bounds
        # self.raw_n = nn.Parameter(torch.randn(1))
        # self.raw_K = nn.Parameter(torch.randn(1))
        self.raw_n = nn.Parameter(torch.empty(1).uniform_(-4, 4))
        self.raw_K = nn.Parameter(torch.empty(1).uniform_(-4, 4))

        self.increasing = increasing

    def forward(self, x):
        # Use sigmoid to ensure n and K are positive and scale to maxes
        n = torch.sigmoid(self.raw_n) * 5
        K = torch.sigmoid(self.raw_K) * self.param_bounds
        x_n = x.pow(n)
        
        if self.increasing:
            hill_feat = x_n / (1 + K * x_n)          
        else:
            hill_feat = (1 / K) - (x_n / (1 + K * x_n))
 
        return hill_feat

# Generate polynomial features for an arbitrary number of proteins.
class PolynomialFeatures(nn.Module):
    def __init__(self, species, duplicates):
        super(PolynomialFeatures, self).__init__()
        self.species = species
        self.duplicates = duplicates

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
        poly_feats = torch.cat(features * self.duplicates, dim=1)

        # if torch.isnan(poly_feats).any():
        #     print('poly nans')

        return poly_feats
    
class HillFeatures(nn.Module):
    def __init__(self, species, param_bounds):
        super(HillFeatures, self).__init__()
        self.num_proteins = species
        
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
    
    def forward(self, x):
        # x has shape [batch, num_proteins]
        inc_features = []
        dec_features = []
        
        # Raw features: hill(x_i)
        for i in range(self.num_proteins):
            xi = x[:, i:i+1]
            inc_features.append(self.hill_inc_raw[i](xi))
            dec_features.append(self.hill_dec_raw[i](xi))
        
        # Cross features: hill(x_i) * x_j for i != j, using separate parameters.
        for i in range(self.num_proteins):
            for j in range(self.num_proteins):
                if i != j:
                    xi = x[:, i:i+1]
                    xj = x[:, j:j+1]
                    inc_features.append(self.hill_inc_cross[f"{i}_{j}"](xi) * xj)
                    dec_features.append(self.hill_dec_cross[f"{i}_{j}"](xi) * xj)
        
        # Concatenate increasing and decreasing hill features
        hill_feats = torch.cat(inc_features + dec_features, dim=1)
        
        # if torch.isnan(hill_feats).any():
        #     print('hill nans')

        return hill_feats

# Duplicates Hill features with unique K and n values
class DuplicateHillFeatures(nn.Module):
    def __init__(self, species, param_bounds, duplicates):
        super(DuplicateHillFeatures, self).__init__()
        self.hill_modules = nn.ModuleList([HillFeatures(species, param_bounds) for _ in range(duplicates)])

    def forward(self, x):
        # Compute features from each independently initialized HillFeatures module
        features = [module(x) for module in self.hill_modules]
        return torch.cat(features, dim=1)

# EQL Layer that combines polynomial and hill features.
class EQLLayer(nn.Module):
    def __init__(self, species, duplicates, param_bounds):
        super(EQLLayer, self).__init__()
        self.poly = PolynomialFeatures(species, duplicates)
        self.hill = DuplicateHillFeatures(species, param_bounds, duplicates)

        # Compute feature sizes
        self.num_poly_features = duplicates * (
            species +                               # linear terms
            species +                               # squared terms
            (species * (species - 1)) // 2          # cross terms
        )
        
        self.num_hill_features = duplicates * 2 * ( # incresaing and decreasing  
            species +                               # raw Hill terms
            species * (species - 1)                 # cross Hill terms  
        )

        self.total_features = self.num_poly_features + self.num_hill_features
                
        self.fc = nn.Linear(self.total_features, 1, bias=False)
        # nn.init.uniform_(self.fc.weight, a=-param_bounds, b=param_bounds)
        nn.init.uniform_(self.fc.weight, a=-1, b=1)

    def forward(self, x):
        poly_feats = self.poly(x)
        hill_feats = self.hill(x)
        features = torch.cat([poly_feats, hill_feats], dim=1)
        return self.fc(features)       