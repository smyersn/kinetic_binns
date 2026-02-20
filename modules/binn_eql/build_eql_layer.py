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
        self.register_buffer('max_scale', max_scale) # [1, 2] tensor of max values

        self.poly = PolynomialFeatures(species, duplicates)
        self.hill = DuplicateHillFeatures(species, param_bounds, duplicates)

        # --- Feature Counts ---
        self.num_poly_features = duplicates * (species + species + (species * (species - 1)) // 2)
        
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

        # 3. Apply Scaling (CRITICAL FIX)
        # We multiply by scales here so 'w' can stay small (physical magnitude)
        # while the output matches the large physical derivatives.
        scales = self._generate_scales_fast().view(1, -1)
        
        out = (features * (w * z * scales)).sum(dim=1, keepdim=True)
        return out
    
    def get_physical_parameters(self):
        """
        Calculates Physical Weights and K for the Loss Function.
        """
        # 1. Weights: w is ALREADY physical because we scaled in forward()
        w_phys = self.fc.weight.view(-1) 
        
        # K parameters still need unscaling because HillFunction internal math hasn't changed
        k_phys_list = []        
        s = self.max_scale[0]
        N = self.species
        ptr = 0
        
        def unscale_k(hf, base_scale):
            n = torch.sigmoid(hf.raw_n) * 3 + 1
            k_net = F.softplus(hf.raw_K)
            return (k_net / (base_scale ** n)).view(1)

        for _ in range(self.duplicates):
            for i in range(N): # Inc Raw
                k_phys_list.append(unscale_k(self.all_hill_funcs[ptr], s[i])); ptr+=1
            for i in range(N): # Inc Cross
                for j in range(N):
                    if i!=j: k_phys_list.append(unscale_k(self.all_hill_funcs[ptr], s[i])); ptr+=1
            for i in range(N): # Dec Raw
                k_phys_list.append(unscale_k(self.all_hill_funcs[ptr], s[i])); ptr+=1
            for i in range(N): # Dec Cross
                for j in range(N):
                    if i!=j: k_phys_list.append(unscale_k(self.all_hill_funcs[ptr], s[i])); ptr+=1

        k_phys = torch.cat(k_phys_list)
        return w_phys, k_phys

    def _generate_scales_fast(self):
        """Generates S^n vector for all terms (Poly + Hill)."""
        s = self.max_scale[0]
        scales_list = []

        # --- Poly Scales ---
        poly_block = []
        for i in range(self.species): poly_block.append(s[i])
        for i in range(self.species): poly_block.append(s[i]**2)
        for i in range(self.species):
            for j in range(i+1, self.species): poly_block.append(s[i]*s[j])
        
        scales_list.extend(poly_block * self.duplicates)

        # --- Hill Scales ---
        ptr = 0
        N = self.species
        
        def get_s(hf, scale_val):
            n = torch.sigmoid(hf.raw_n.view(())) * 3 + 1
            return (scale_val ** n)

        for _ in range(self.duplicates):
            for i in range(N): # Inc Raw
                scales_list.append(get_s(self.all_hill_funcs[ptr], s[i])); ptr+=1
            for i in range(N): # Inc Cross
                for j in range(N):
                    if i!=j: scales_list.append(get_s(self.all_hill_funcs[ptr], s[i]) * s[j]); ptr+=1
            for i in range(N): # Dec Raw
                scales_list.append(get_s(self.all_hill_funcs[ptr], s[i])); ptr+=1
            for i in range(N): # Dec Cross
                for j in range(N):
                    if i!=j: scales_list.append(get_s(self.all_hill_funcs[ptr], s[i]) * s[j]); ptr+=1

        return torch.stack(scales_list).view(1, -1)

    def _smart_initialize_K(self):
        """Sets raw_K so K_phys starts in safe range [0, param_bounds]."""
        s = self.max_scale[0]
        N = self.species
        ptr = 0

        def set_k(hf, scale):
            with torch.no_grad():
                # target_phys is a 1D tensor [val] (can be between 0 and 1)
                target_phys = torch.empty(1).uniform_(0.0, 1.0)
                
                # n is a 1D tensor [val]
                n = torch.sigmoid(hf.raw_n) * 3 + 1
                
                # target_net is a 1D tensor [val]
                target_net = target_phys * (scale ** n)
                
                # Inverse Softplus: log(exp(y) - 1)
                # FIX: Use .item() to convert 1D tensor to Python float for fill_()
                if target_net.item() > 20.0:
                    hf.raw_K.data.fill_(target_net.item())
                else:
                    # Calculate value then convert to item
                    val = torch.log(torch.exp(target_net) - 1 + 1e-9)
                    hf.raw_K.data.fill_(val.item())

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
        scales = self._generate_scales_fast().view(1, -1)
        
        return features * scales