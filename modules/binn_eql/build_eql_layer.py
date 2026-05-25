import itertools
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
        # We initialize it randomly here, but EQLLayer will overwrite it 
        # immediately with _smart_initialize_K to prevent explosion.
        self.raw_K = nn.Parameter(torch.empty(1).uniform_(-2, 2))

    def forward(self, x):
        n = torch.sigmoid(self.raw_n) * 3 + 1
        
        # Softplus ensures K > 0 but allows it to grow large (e.g. 10^5)
        # to match the inverse scaling of S^n
        K = F.softplus(self.raw_K)
        
        x_n = x.pow(n)
        
        if self.increasing:
            return x_n / (1 + K * x_n)
        else:
            # Epsilon for stability
            return (1 / (K + 1e-8)) - (x_n / (1 + K * x_n))
        
class PolynomialFeatures(nn.Module):
    def __init__(self, species, duplicates, degree):
        super(PolynomialFeatures, self).__init__()
        self.species = species
        self.duplicates = duplicates
        self.degree = degree
        self.powers = self._generate_powers()

    def _generate_powers(self):
        """Generates all combinations of powers where 1 <= sum <= degree"""
        powers = []
        for p in itertools.product(range(self.degree + 1), repeat=self.species):
            if 1 <= sum(p) <= self.degree:
                powers.append(p)
        
        # Sort by total degree first, then descending order of first species power
        # This keeps the output clean: u, v, u^2, uv, v^2, u^3...
        powers.sort(key=lambda x: (sum(x), tuple(-power for power in x)))
        return powers

    def forward(self, x):
        features = []
        for p in self.powers:
            # Start with a column of 1s
            term = torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype)
            for i, power in enumerate(p):
                if power > 0:
                    term = term * (x[:, i:i+1] ** power)
            features.append(term)
            
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
    def __init__(self, species, duplicates, param_bounds, max_scale, degree):
        super(EQLLayer, self).__init__()
        self.species = species
        self.duplicates = duplicates
        self.param_bounds = param_bounds
        self.degree = degree
        self.register_buffer('max_scale', max_scale) # [1, 2] tensor of max values

        self.poly = PolynomialFeatures(species, duplicates, degree)
        self.hill = DuplicateHillFeatures(species, param_bounds, duplicates)
        
        # --- Feature Counts ---
        self.num_poly_features = duplicates * len(self.poly.powers)
        
        # Hill: IncRaw(N) + IncCross(N*(N-1)) + DecRaw(N) + DecCross(N*(N-1))
        n_hill_single = (species + species * (species - 1)) * 2 
        self.num_hill_features = duplicates * n_hill_single
        
        self.total_features = self.num_poly_features + self.num_hill_features
                
        self.fc = nn.Linear(self.total_features, 1, bias=False)
        self.l0_gate = HardConcreteGate(self.total_features)
        
        # --- 1. FLATTEN HILL MODULES (For Speed & Indexing) ---
        self.all_hill_funcs = []
        for hm in self.hill.hill_modules:
            # Order must match generation: IncRaw -> IncCross -> DecRaw -> DecCross
            self.all_hill_funcs.extend(hm.hill_inc_raw)
            for i in range(species):
                for j in range(species):
                    if i!=j: self.all_hill_funcs.append(hm.hill_inc_cross[f"{i}_{j}"])
            
            self.all_hill_funcs.extend(hm.hill_dec_raw)
            for i in range(species):
                for j in range(species):
                    if i!=j: self.all_hill_funcs.append(hm.hill_dec_cross[f"{i}_{j}"])

        # --- 2. SMART K INITIALIZATION ---
        # Initialize K_phys in safe range [0, 10] BEFORE weights
        # (This ensures 'n' and 'K' are set before we calculate scales for weights)
        self._smart_initialize_K()

        # --- 3. WEIGHT INITIALIZATION ---
        # Initialize w_phys in safe range [-1, 1]
        nn.init.uniform_(self.fc.weight, a=-1, b=1)

    def forward(self, x, training=True):
        # 1. Generate Features
        poly_feats = self.poly(x)
        hill_feats = self.hill(x)
        features = torch.cat([poly_feats, hill_feats], dim=1)
        
        # 2. Get Weights and Gate
        w = self.fc.weight
        z = self.l0_gate() 
        
        out = (features * (w * z)).sum(dim=1, keepdim=True)
        return out

    def get_physical_parameters(self, epsilon=0.2):
        """
        Calculates Physical Weights, K, and a Dynamic K-Ceiling.
        epsilon: The minimum fraction of u_max where the half-max (Kd) can occur.
        """
        w_phys = self.fc.weight.view(-1) 
        
        k_phys_list = []
        k_ceiling_list = []
        
        s = self.max_scale[0] # This IS u_max for each species
        N = self.species
        ptr = 0
        
        def get_k_and_ceiling(hf, u_max):
            n = torch.sigmoid(hf.raw_n) * 3 + 1
            k_phys = F.softplus(hf.raw_K)
            
            # Dynamic Ceiling: K_max = 1 / (epsilon * u_max)^n
            k_d_min = epsilon * (u_max + 1e-6) # 1e-6 prevents div/0
            k_ceiling = 1.0 / (k_d_min ** n)
            
            return k_phys.view(1), k_ceiling.view(1)

        for _ in range(self.duplicates):
            for i in range(N): # Inc Raw
                k, ceil = get_k_and_ceiling(self.all_hill_funcs[ptr], s[i]); ptr+=1
                k_phys_list.append(k); k_ceiling_list.append(ceil)
            for i in range(N): # Inc Cross
                for j in range(N):
                    if i!=j:
                        k, ceil = get_k_and_ceiling(self.all_hill_funcs[ptr], s[i]); ptr+=1
                        k_phys_list.append(k); k_ceiling_list.append(ceil)
            for i in range(N): # Dec Raw
                k, ceil = get_k_and_ceiling(self.all_hill_funcs[ptr], s[i]); ptr+=1
                k_phys_list.append(k); k_ceiling_list.append(ceil)
            for i in range(N): # Dec Cross
                for j in range(N):
                    if i!=j:
                        k, ceil = get_k_and_ceiling(self.all_hill_funcs[ptr], s[i]); ptr+=1
                        k_phys_list.append(k); k_ceiling_list.append(ceil)

        k_phys = torch.cat(k_phys_list)
        k_ceilings = torch.cat(k_ceiling_list)
        return w_phys, k_phys, k_ceilings
    
    def _smart_initialize_K(self):
        """
        Initializes raw_K so that the physical K is strictly below the dynamic ceiling.
        """
        s = self.max_scale[0]
        N = self.species
        ptr = 0

        def set_k(hf, scale):
            with torch.no_grad():
                # 1. Get the currently initialized 'n'
                n = torch.sigmoid(hf.raw_n) * 3 + 1
                
                # 2. Pick a random multiplier (alpha) strictly > 1.0
                # By forcing Kd to be 1x to 3.0x of the max physical concentration, 
                # we guarantee the curve starts out acting like a healthy polynomial.
                alpha = torch.empty(1).uniform_(1, 3)
                kd_initial = alpha * (scale + 1e-6) # 1e-6 prevents div/0
                
                # 3. Convert Kd back to the physical K our network learns
                # K_phys = 1 / (Kd^n)
                k_phys_target = 1.0 / (kd_initial ** n)
                
                # 4. Safe Inverse Softplus 
                # (Prevents PyTorch overflow crashes if 'scale' is very tiny)
                if k_phys_target.item() > 20.0:
                    val = k_phys_target.item()
                else:
                    val = torch.log(torch.exp(k_phys_target) - 1 + 1e-9).item()
                
                # 5. Overwrite the raw parameter
                hf.raw_K.data.fill_(val)

        # Apply to all Hill functions
        for _ in range(self.duplicates):
            for i in range(N): # Inc Raw
                set_k(self.all_hill_funcs[ptr], s[i]); ptr+=1
            for i in range(N): # Inc Cross
                for j in range(N):
                    if i!=j: set_k(self.all_hill_funcs[ptr], s[i]); ptr+=1
            for i in range(N): # Dec Raw
                set_k(self.all_hill_funcs[ptr], s[i]); ptr+=1
            for i in range(N): # Dec Cross
                for j in range(N):
                    if i!=j: set_k(self.all_hill_funcs[ptr], s[i]); ptr+=1
                                                                                        
    def get_features(self, x):
        poly_feats = self.poly(x)
        hill_feats = self.hill(x)
        features = torch.cat([poly_feats, hill_feats], dim=1)
        
        return features