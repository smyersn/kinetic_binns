import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.binn_eql.hard_concrete_gate import HardConcreteGate
       
class HillFunction(nn.Module):
    def __init__(self, param_bounds, scale, increasing=True):
        super(HillFunction, self).__init__()
        self.param_bounds = param_bounds
        self.increasing = increasing
        # Register scale as a buffer so it saves with the model but isn't a parameter
        self.register_buffer('scale', torch.tensor(scale))
        
        self.raw_n = nn.Parameter(torch.empty(1).uniform_(-4, 4))
        self.raw_K = nn.Parameter(torch.empty(1).uniform_(-2, 2))
        
    def forward(self, x):
        n = torch.sigmoid(self.raw_n) * 3 + 1
        
        # 1. Constrain Physical K to [0, param_bounds]
        # This GUARANTEES K_phys never exceeds 10
        K_phys = torch.sigmoid(self.raw_K) * self.param_bounds
        
        # 2. Convert to Network K to match normalized input x
        # Math: 1 + K_phys * u^n  ===  1 + K_net * (u/S)^n
        # Therefore: K_net = K_phys * S^n
        K_net = K_phys * (self.scale ** n)
        
        x_n = x.pow(n)
        
        if self.increasing:
            return x_n / (1 + K_net * x_n)
        else:
            return (1 / (K_net + 1e-8)) - (x_n / (1 + K_net * x_n))
        
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
    def __init__(self, species, param_bounds, max_scale):
        super(HillFeatures, self).__init__()
        self.num_proteins = species
        
        # Extract scalar values for s_u, s_v
        # Assuming max_scale is a tensor [s_u, s_v]
        scales = max_scale.cpu().numpy().flatten()
        
        self.hill_inc_raw = nn.ModuleList()
        self.hill_dec_raw = nn.ModuleList()
        
        # Initialize Raw Hills (Scale matches species i)
        for i in range(species):
            s = scales[i]
            self.hill_inc_raw.append(HillFunction(param_bounds, s, increasing=True))
            self.hill_dec_raw.append(HillFunction(param_bounds, s, increasing=False))
        
        self.hill_inc_cross = nn.ModuleDict()
        self.hill_dec_cross = nn.ModuleDict()
        
        # Initialize Cross Hills (Hill(u) * v)
        # The Hill term acts on 'u', so it uses scale s_u
        for i in range(species):
            for j in range(species):
                if i != j:
                    s = scales[i] # Scale of the input to the Hill function
                    self.hill_inc_cross[f"{i}_{j}"] = HillFunction(param_bounds, s, increasing=True)
                    self.hill_dec_cross[f"{i}_{j}"] = HillFunction(param_bounds, s, increasing=False)
                        
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
    def __init__(self, species, param_bounds, duplicates, max_scale):
        super(DuplicateHillFeatures, self).__init__()
        self.hill_modules = nn.ModuleList([
            HillFeatures(species, param_bounds, max_scale) for _ in range(duplicates)
        ])
        
    def forward(self, x):
        # Compute features from each independently initialized HillFeatures module
        features = [module(x) for module in self.hill_modules]
        return torch.cat(features, dim=1)

# EQL Layer that combines polynomial and hill features.
class EQLLayer(nn.Module):
    def __init__(self, species, duplicates, param_bounds, max_scale):
        super(EQLLayer, self).__init__()
        self.species = species
        self.duplicates = duplicates
        self.param_bounds = param_bounds
        self.register_buffer('max_scale', max_scale)
        self.poly = PolynomialFeatures(species, duplicates)
        self.hill = DuplicateHillFeatures(species, param_bounds, duplicates, max_scale)

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
        self.l0_gate = HardConcreteGate(self.total_features)
        # nn.init.uniform_(self.fc.weight, a=-param_bounds, b=param_bounds)
        nn.init.uniform_(self.fc.weight, a=-0.1, b=0.1)
        
    def get_features(self, x):
        poly_feats = self.poly(x)
        hill_feats = self.hill(x)
        features = torch.cat([poly_feats, hill_feats], dim=1)
        
        return features

    def generate_term_scales(self):
        """
        Calculates S^n for ALL terms (Poly + Hill) dynamically based on specific species.
        Corrects shape mismatch between Poly scalars and Hill 1D tensors.
        """
        s = self.max_scale[0] # [s_u, s_v] (Shape: [2])
        scales_list = []

        # --- 1. POLYNOMIAL SCALES (Scalars) ---
        poly_block = []
        
        # A. Linear
        for i in range(self.species):
            poly_block.append(s[i])
            
        # B. Squared
        for i in range(self.species):
            poly_block.append(s[i] ** 2)
            
        # C. Cross
        for i in range(self.species):
            for j in range(i+1, self.species):
                poly_block.append(s[i] * s[j])
        
        # Extend master list
        for _ in range(self.duplicates):
            scales_list.extend(poly_block)

        # --- 2. HILL SCALES (Force to Scalar) ---
        for hm in self.hill.hill_modules:
            # --- INCREASING BLOCK ---
            
            # A. Raw Inc
            for i in range(self.species):
                # FIX: .view(()) converts [1] -> [] (scalar)
                raw_n_scalar = hm.hill_inc_raw[i].raw_n.view(()) 
                n = torch.sigmoid(raw_n_scalar) * 3 + 1
                scales_list.append(s[i] ** n)
                
            # B. Cross Inc
            for i in range(self.species):
                for j in range(self.species):
                    if i != j:
                        key = f"{i}_{j}"
                        # FIX: .view(())
                        raw_n_scalar = hm.hill_inc_cross[key].raw_n.view(())
                        n = torch.sigmoid(raw_n_scalar) * 3 + 1
                        scales_list.append((s[i] ** n) * s[j])

            # --- DECREASING BLOCK ---
            
            # C. Raw Dec
            for i in range(self.species):
                # FIX: .view(())
                raw_n_scalar = hm.hill_dec_raw[i].raw_n.view(())
                n = torch.sigmoid(raw_n_scalar) * 3 + 1
                scales_list.append(s[i] ** n)
                
            # D. Cross Dec
            for i in range(self.species):
                for j in range(self.species):
                    if i != j:
                        key = f"{i}_{j}"
                        # FIX: .view(())
                        raw_n_scalar = hm.hill_dec_cross[key].raw_n.view(())
                        n = torch.sigmoid(raw_n_scalar) * 3 + 1
                        scales_list.append((s[i] ** n) * s[j])

        # Stack into a single tensor matching feature dimension (1, Total_Feats)
        # Now all inputs are size [], so stack works.
        return torch.stack(scales_list).view(1, -1)
    
    def forward(self, x):
        # 1. Generate Dimensionless Features
        features = torch.cat([self.poly(x), self.hill(x)], dim=1)
        
        # 2. Get Physical Weight (Bounded [-bounds, +bounds])
        w_phys = torch.tanh(self.fc.weight) * self.param_bounds
        
        # 3. Calculate Inverse Scales (S^n)
        scales = self.generate_term_scales()
        
        # 4. Effective Weight = w_phys * Scale
        w_effective = w_phys * scales
        
        z = self.l0_gate()
        
        # 5. Output
        out = (features * (w_effective * z)).sum(dim=1, keepdim=True)
        return out