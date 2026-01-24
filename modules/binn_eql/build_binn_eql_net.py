import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils

from modules.binn.build_mlp import build_mlp
from modules.utils.gradient import gradient
from modules.binn_eql.build_eql_layer import EQLLayer

# ---------------------------------------------------------
# 1. SUB-NETWORKS
# ---------------------------------------------------------
class D_PARAMS(nn.Module):
    def __init__(self, input_features=2, param_bounds=10):
        super().__init__()
        self.param_bounds = param_bounds
        # Initialize closer to 0 for stability
        self.raw_D = nn.Parameter(torch.empty(input_features).uniform_(-4, 0))
        
    def forward(self):     
        return torch.sigmoid(self.raw_D) * self.param_bounds

class uv_MLP(nn.Module):
    def __init__(self, input_features, layers=[256, 256, 256, 2]):
    # def __init__(self, input_features, layers=[512, 512, 512, 512, 2], fourier_scale=10):
        super().__init__()
               
        # # MLP
        # self.mlp = build_mlp(
        #     input_features=input_features, 
        #     layers=layers,
        #     activation=nn.Tanh(),
        #     linear_output=False,
        #     output_activation=nn.Softplus()) # Softplus ensures u,v > 0

        # 1. Pass GELU directly (build_mlp accepts an activation arg)
        self.mlp = build_mlp(
            input_features=input_features, 
            layers=layers,
            activation=nn.GELU(),          # <--- Change 1: GELU
            linear_output=False,
            output_activation=nn.Softplus())

        # 2. Apply Weight Norm "Post-Hoc"
        # We iterate through the network we just built and wrap every Linear layer
        for module in self.mlp.MLP:
            if isinstance(module, nn.Linear):
                utils.parametrizations.weight_norm(module)  # <--- Change 2: Weight Norm
                
    def forward(self, inputs):
        # inputs are [-1, 1]
        return self.mlp(inputs) # Outputs [0, 1] roughly (scaled space)

class F_EQL(nn.Module):
    def __init__(self, species, duplicates, param_bounds, max_scale):
        super(F_EQL, self).__init__()
        self.eql_layer = EQLLayer(species, duplicates, param_bounds, max_scale)

    def forward(self, x):
        return self.eql_layer(x)
        
# ---------------------------------------------------------
# 2. THE GOLD STANDARD BINN
# ---------------------------------------------------------
class BINN(nn.Module):
    def __init__(self, dimensions, species, train_data, duplicates=1,
                 diff_coeffs=None, uv_layers=None, degree=2, param_bounds=10):
        
        super().__init__()
        self.dimensions = dimensions        
        self.species = species
        self.train_data = train_data
        self.duplicates = duplicates
        self.diff_coeffs = diff_coeffs
        self.param_bounds = param_bounds

        # ---------------------------------------------------------
        # A. REGISTER BOUNDS & SCALES (Buffers)
        # ---------------------------------------------------------
        # Spatial/Temporal Bounds
        x_min = torch.min(train_data[:, :dimensions])
        x_max = torch.max(train_data[:, :dimensions])
        t_min = torch.min(train_data[:, dimensions])
        t_max = torch.max(train_data[:, dimensions])
        
        # Register for Input Normalization [-1, 1]
        # Shape [1, dims+1] for broadcasting
        lb_tensor = torch.cat([torch.full((dimensions,), x_min), torch.tensor([t_min])])
        ub_tensor = torch.cat([torch.full((dimensions,), x_max), torch.tensor([t_max])])
        self.register_buffer('lb', lb_tensor.view(1, -1)) 
        self.register_buffer('ub', ub_tensor.view(1, -1))
        
        # Ranges for Chain Rule (Derivative Scaling)
        self.register_buffer('x_range', x_max - x_min)
        self.register_buffer('t_range', t_max - t_min)
                
        # Concentration Scales (Physical -> Dimensionless)
        ## We use 99th percentile to be robust against outliers
        s_u_max = torch.quantile(train_data[:, -2].abs(), 0.99)
        s_v_max = torch.quantile(train_data[:, -1].abs(), 0.99)
        # Shape [1, species]
        self.register_buffer('max_scale', torch.tensor([s_u_max, s_v_max]).view(1, -1))
        
        # GLS Mean Scale
        s_u_mean = train_data[:, -2].abs().mean()
        s_v_mean = train_data[:, -1].abs().mean()
        self.register_buffer('mean_scale', torch.tensor([s_u_mean, s_v_mean]).view(1, -1))

        # ---------------------------------------------------------
        # B. INITIALIZE SUB-NETWORKS
        # ---------------------------------------------------------
        # Diffusion Fitter
        if not self.diff_coeffs:
            self.diffusion_fitter = D_PARAMS(self.species, param_bounds)
        else:
            self.diffusion_fitter = None
                
        # Surface Fitter (Dimensionless)
        # Input: dimensions + time (normalized)
        if uv_layers:
            self.surface_fitter = uv_MLP(input_features=dimensions+1, layers=uv_layers)
        else:
            self.surface_fitter = uv_MLP(input_features=dimensions+1)
        
        # Reaction (Dimensionless Input -> Dimensionless Rate)
        self.reaction = F_EQL(species, duplicates, self.param_bounds, self.max_scale)
        
        # Sampling config
        self.num_samples = 10000
        self.name = 'Dumlp_Dvmlp_Fmlp'

    def normalize(self, inputs):
        """ Maps Physical [lb, ub] -> Dimensionless [-1, 1] """
        return 2.0 * (inputs - self.lb) / (self.ub - self.lb) - 1.0

    def forward(self, inputs):
        """ Returns PREDICTED u (Scaled [0,1]) from Physical Inputs """       
        # 1. Normalize Inputs
        inputs_hat = self.normalize(inputs)
        
        # 2. Predict Surface (Dimensionless)
        return self.surface_fitter(inputs_hat)

    # -----------------------
    # Loss Functions
    # -----------------------
    def gls_loss(self, pred, true):
        residual = ((pred - true) / self.mean_scale)**2
        return torch.mean(residual)
    
    def pde_loss(self, inputs, outputs, epoch):
        # unpack outputs
        u = outputs.clone()
        u_scaled = u / self.max_scale

        # create arrays to store partial derivatives
        points = len(inputs)
        uxx_array = torch.zeros((self.species, points, self.dimensions)).to(inputs.device)
        ut_array = torch.zeros((points, self.species)).to(inputs.device)

        # partial derivative computations
        for i in range(self.species):
            d1 = gradient(u[:, i], inputs, order=1)
            ut = d1[:, -1]
            ut_array[:, i] = ut

            for j in range(self.dimensions):
                d2 = gradient(d1[:, j], inputs, order=1)
                uxx = d2[:, j]
                uxx_array[i, :, j] = uxx
                                        
        # reaction
        F = self.reaction(u_scaled)
        
        # diffusion
        if self.diff_coeffs:
            Du, Dv = torch.tensor(self.diff_coeffs[0]), torch.tensor(self.diff_coeffs[1])
            
        else:
            D = self.diffusion_fitter()
            Du, Dv = D[0], D[1]
                    
        lap_u = Du * torch.sum(uxx_array[0, :, :], dim=1, keepdim=True)
        lap_v = Dv * torch.sum(uxx_array[1, :, :], dim=1, keepdim=True)
                    
        # Reaction-diffusion equation       
        LHS_u = ut_array[:, 0][:,None]
        RHS_u = lap_u + F
        LHS_v = ut_array[:, 1][:,None]
        RHS_v = lap_v - F
        pde_loss = (LHS_u - RHS_u)**2 + (LHS_v - RHS_v)**2
        
        # if epoch % 1000 == 0:
        #     print(f'PDE LOSS START:')
        #     print(f'Du, Dv: {Du, Dv}')
        #     print(f'xtuv: {torch.concat([inputs, u], dim=1)[:20]}')
        #     print(f'LHSU RHSU: {torch.concat([LHS_u, RHS_u], dim=1)[:20]}')
        #     print(f'RHSU, LAPU, F: {torch.concat([RHS_u, lap_u, F], dim=1)[:20]}')
        #     print(f'LHSV RHSV: {torch.concat([LHS_v, RHS_v], dim=1)[:20]}')
        #     print(f'RHSV, LAPV, -F: {torch.concat([RHS_v, lap_v, -F], dim=1)[:20]}')
        #     print(f'loss: {pde_loss[:20]}')
        #     print(f'pde loss: {torch.mean(pde_loss)}\n')

        return torch.mean(pde_loss)
                    
    def reg_loss(self, lux_tax, epoch):
        # 1. Get probabilities
        gate_probs = self.reaction.eql_layer.l0_gate.expected_l0()
        
        # 2. Slice
        num_poly = self.reaction.eql_layer.num_poly_features
        poly_probs = gate_probs[:num_poly]
        hill_probs = gate_probs[num_poly:]
        
        # 3. Luxury Tax
        l0_poly = poly_probs.sum()
        l0_hill = hill_probs.sum() * lux_tax
        total_l0 = l0_poly + l0_hill
                
        # 5. Total weighted L0 norm
        total_l0 = l0_poly + l0_hill

        # total_l0 = self.reaction.eql_layer.l0_gate.expected_l0().sum()
        
        # Penalize cheating Hill functions (K = 0)
        def small_K_hinge_penalty(K_vals, K_thresh=1e-3, weight=1e3):
            # K_vals: torch tensor of K for all hill funcs (on device)
            # penalize only when K < K_thresh
            diff = torch.clamp(K_thresh - K_vals, min=0.0)
            return weight * torch.mean(diff * diff)   # MSE hinge
        
        # gather K_vals (example, adapt to your model)
        rawK_list = []
        for hm in self.reaction.eql_layer.hill.hill_modules:
            for hf in hm.hill_inc_raw + list(hm.hill_inc_cross.values()) + hm.hill_dec_raw + list(hm.hill_dec_cross.values()):
                # K_val = torch.exp(hf.raw_logK)
                K_val = torch.sigmoid(hf.raw_K) * self.param_bounds
                rawK_list.append(K_val.view(-1))

        K_vals = torch.cat(rawK_list)
        K_pen = small_K_hinge_penalty(K_vals)
                              
        return total_l0 + K_pen

    def loss(self, pred, true, epoch, gls_weight, pde_weight, l0_weight, lux_tax):       
        # GLS Loss
        self.gls_loss_val = gls_weight * self.gls_loss(pred, true)
        
        # PDE Sampling
        x = torch.empty(self.num_samples, self.dimensions, device=pred.device).uniform_(self.lb[0,0], self.ub[0,0])
        t = torch.empty(self.num_samples, 1, device=pred.device).uniform_(self.lb[0,-1], self.ub[0,-1])
        inputs_rand = torch.cat([x, t], dim=1).requires_grad_()
        inputs_rand_norm = self.normalize(inputs_rand)
        outputs_rand = self.surface_fitter(inputs_rand_norm)
        
        # PDE Loss
        self.pde_loss_val = pde_weight * self.pde_loss(inputs_rand, outputs_rand,
                                                       epoch)
              
        # Reg Loss            
        l0_loss = self.reg_loss(lux_tax, epoch)
        self.reg_loss_val = l0_weight * l0_loss  
              
        # if epoch % 1000 == 0:
        #     print(f'L0 norm, weight, loss: {l0_loss, l0_weight_eff, self.reg_loss_val}')
        
        return (self.gls_loss_val + self.pde_loss_val + self.reg_loss_val), self.gls_loss_val, self.pde_loss_val, self.reg_loss_val

    # -----------------------
    # Feature generation and equation formatting
    # -----------------------
    def generate_terms(self):
        poly_terms = []
        hill_terms = []
        
        # Calculate linear poly terms
        for i in range(self.species):
            poly_terms.append((i,))
        
        # Calculate squared poly terms
        for i in range(self.species):
            poly_terms.append((i, i))
        
        # Calculate cross poly terms  
        for i in range(self.species):
            for j in range(i+1, self.species):
                poly_terms.append((i, j))

        # Calculate raw Hill terms
        for i in range(self.species):
            hill_terms.append((i,))

        # Calculate cross Hill terms
        for i in range(self.species):
            for j in range(self.species):
                if i != j:
                    hill_terms.append((i, j))
                    
        return poly_terms, hill_terms
    
    def extract_params(self, full=True):
        """
        Extracts PHYSICAL parameters.
        Adapted for Inverse Scaling with Physical Inputs:
        - Weights are already physical (no division needed).
        - K is already physical (no division needed).
        """
        eql = self.reaction.eql_layer
        
        # 1. Get Physical Weights directly
        # The forward pass applied constraints and scaling, so raw_w here represents 
        # the physical rate constant bounded by [-param_bounds, param_bounds].
        raw_w_t = (torch.tanh(eql.fc.weight[0]) * eql.param_bounds).detach()

        # 2. Get Gates
        try:
            gates_t = eql.l0_gate.get_gates().detach()
        except:
            # Fallback for hard concrete gate (stretched sigmoid)
            log_alpha = eql.l0_gate.log_alpha.detach()
            gates_t = torch.sigmoid(log_alpha).clamp(0, 1)

        # 3. Effective Physical Weights
        effective_t = (raw_w_t * gates_t).detach()
        
        # Convert to numpy
        raw_w = raw_w_t.cpu().numpy().reshape(-1)
        gates = gates_t.cpu().numpy().reshape(-1)
        effective = effective_t.cpu().numpy().reshape(-1)

        # Gather structure
        num_poly = int(eql.num_poly_features)
        num_hill = int(eql.num_hill_features)
        poly_terms, hill_terms = self.generate_terms()
        n_hill_single = len(hill_terms)
        dup = int(self.duplicates)
        
        # --- EXTRACT HILL PARAMS (n, K) ---
        # Since we feed PHYSICAL inputs (x * scale) into the HillFunction,
        # the learned K is the Physical K. No unscaling is required.
        
        raw_ns_inc_list, raw_Ks_inc_list = [], []
        raw_ns_dec_list, raw_Ks_dec_list = [], []

        for hill_module in eql.hill.hill_modules:
            # Helper to extract Sigmoid n and Sigmoid K
            def get_phys_NK(raw_n, raw_K):
                n_val = torch.sigmoid(raw_n) * 3 + 1
                # Standard Sigmoid K (Physical)
                K_val = torch.sigmoid(raw_K) * eql.param_bounds
                return n_val.item(), K_val.item()

            # Iterate in the exact order of feature generation:
            # 1. Inc Raw
            for hf in hill_module.hill_inc_raw:
                n, k = get_phys_NK(hf.raw_n, hf.raw_K)
                raw_ns_inc_list.append(n); raw_Ks_inc_list.append(k)
            # 2. Inc Cross
            for key in getattr(hill_module, 'hill_inc_cross', {}):
                hf = hill_module.hill_inc_cross[key]
                n, k = get_phys_NK(hf.raw_n, hf.raw_K)
                raw_ns_inc_list.append(n); raw_Ks_inc_list.append(k)
            
            # 3. Dec Raw
            for hf in hill_module.hill_dec_raw:
                n, k = get_phys_NK(hf.raw_n, hf.raw_K)
                raw_ns_dec_list.append(n); raw_Ks_dec_list.append(k)
            # 4. Dec Cross
            for key in getattr(hill_module, 'hill_dec_cross', {}):
                hf = hill_module.hill_dec_cross[key]
                n, k = get_phys_NK(hf.raw_n, hf.raw_K)
                raw_ns_dec_list.append(n); raw_Ks_dec_list.append(k)

        ns_inc = np.array(raw_ns_inc_list)
        Ks_inc = np.array(raw_Ks_inc_list)
        ns_dec = np.array(raw_ns_dec_list)
        Ks_dec = np.array(raw_Ks_dec_list) # Fixed typo (was Ks_inc)

        # --- POLYNOMIAL COEFFICIENTS ---
        # No unscaling needed. The weight is physical.
        poly_coeffs_unscaled = effective[:num_poly] if num_poly > 0 else np.array([])
        
        # --- HILL COEFFICIENTS ---
        # Slice the effective array into increasing/decreasing blocks for each duplicate
        hill_block = effective[num_poly : num_poly + num_hill] if num_hill > 0 else np.array([])
        
        hill_inc_unscaled_list = []
        hill_dec_unscaled_list = []
        ptr = 0
        
        for d in range(dup):
            # Increasing Block
            inc_slice = hill_block[ptr : ptr + n_hill_single]
            hill_inc_unscaled_list.extend(inc_slice)
            ptr += n_hill_single
            
            # Decreasing Block
            dec_slice = hill_block[ptr : ptr + n_hill_single]
            hill_dec_unscaled_list.extend(dec_slice)
            ptr += n_hill_single

        hill_inc_unscaled = np.array(hill_inc_unscaled_list)
        hill_dec_unscaled = np.array(hill_dec_unscaled_list)

        # --- K VALUES ---
        # No unscaling needed. We used physical inputs.
        Ks_inc_unscaled = Ks_inc
        Ks_dec_unscaled = Ks_dec

        # --- RECONSTRUCT RAW_W_UNSCALED ---
        # Just for consistency with other parts of your code that expect this key
        # We reconstruct it to match the order: Poly -> (Inc -> Dec) * Duplicates
        raw_w_unscaled = raw_w 
        effective_unscaled = effective

        if not full:
             return {'raw_w_unscaled': raw_w_unscaled, 'effective_unscaled': effective_unscaled}

        return {
            'raw_w': raw_w, 
            'raw_w_unscaled': raw_w_unscaled, 
            'gates': gates,
            'effective': effective, 
            'effective_unscaled': effective_unscaled,
            'num_poly': num_poly, 'num_hill': num_hill,
            'ns_inc': ns_inc, 'Ks_inc': Ks_inc, 
            'ns_dec': ns_dec, 'Ks_dec': Ks_dec,
            'poly_terms': poly_terms, 'hill_terms': hill_terms,
            'poly_coeffs_unscaled': poly_coeffs_unscaled,
            'hill_inc_unscaled': hill_inc_unscaled, 
            'hill_dec_unscaled': hill_dec_unscaled,
            'Ks_inc_unscaled': Ks_inc_unscaled,
            'Ks_dec_unscaled': Ks_dec_unscaled
        }
                        
    @torch.no_grad()
    def fine_tune_eql(self, threshold=0.01, epsilon=0.05):
        """
        Fine-tunes the discovered EQL equation.
        Sequence: Zeroing -> Poly Merging -> Hill Merging/Averaging -> Poly Simplification.
        """
        eql = self.reaction.eql_layer
        device = eql.fc.weight.device
        
        # 1. EVALUATION DATA (Standard Normalized Range)
        u_norm = self.train_data[:, -2:-1].to(device) / self.max_scale[0, 0]
        v_norm = self.train_data[:, -1:].to(device) / self.max_scale[0, 1]
        uv_norm = torch.cat([u_norm, v_norm], dim=1)
        
        # 2. ALIGNED FEATURES: Sequential Duplicate-Major [Poly, Inc, Dec]
        features = eql.get_features(uv_norm) 
        
        # --- TASK 1: ZEROING (Pruning Noise) ---
        params = self.extract_params(full=True)
        eff_unscaled = torch.tensor(params['effective_unscaled'], device=device)
        
        # Identify terms that contribute less than the threshold to the physical rate
        small_mask = torch.abs(eff_unscaled) < threshold
        eql.fc.weight.data[0, small_mask] = 0.0
        eql.l0_gate.log_alpha.data[small_mask] = -10.0 # Lock gate shut

        # Refresh params after pruning
        params = self.extract_params(full=True)
        num_poly = eql.num_poly_features
        num_hill = eql.num_hill_features
        n_poly_single = num_poly // self.duplicates
        n_hill_single = len(params['hill_terms'])

        # --- TASK 2: COMBINE DUPLICATE POLYNOMIALS ---
        for i in range(n_poly_single):
            indices = [i + j * n_poly_single for j in range(self.duplicates)]
            primary = indices[0]
            for other in indices[1:]:
                if torch.abs(eql.fc.weight.data[0, other]) < 1e-8: continue
                eql.fc.weight.data[0, primary] += eql.fc.weight.data[0, other]
                eql.fc.weight.data[0, other] = 0.0
                # Transfer gate importance
                eql.l0_gate.log_alpha.data[primary] = torch.max(
                    eql.l0_gate.log_alpha.data[primary], 
                    eql.l0_gate.log_alpha.data[other]
                )
                eql.l0_gate.log_alpha.data[other] = -10.0

        # --- TASK 3A: MERGE DUPLICATE HILLS (With Parameter Averaging) ---
        for i in range(num_hill):
            h_idx = num_poly + i
            if torch.abs(eql.fc.weight.data[0, h_idx]) < 1e-8: continue
            
            f_hill = features[:, h_idx]
            
            for next_h_idx in range(h_idx + 1, num_poly + num_hill):
                if torch.abs(eql.fc.weight.data[0, next_h_idx]) < 1e-8: continue
                
                f_other = features[:, next_h_idx]
                # Shape-only check (Normalized)
                diff = torch.mean(torch.abs(f_hill/f_hill.max() - f_other/f_other.max()))
                
                if diff < epsilon:
                    print(f"Merging Duplicate Hills: {h_idx} and {next_h_idx} (diff: {diff:.4f})")
                    
                    # Update Hill weights and average internal n/K parameters
                    eql.fc.weight.data[0, h_idx] += eql.fc.weight.data[0, next_h_idx]
                    eql.fc.weight.data[0, next_h_idx] = 0.0
                    
                    self._average_hill_params(h_idx - num_poly, next_h_idx - num_poly)
                    
                    # Consolidate Gate
                    eql.l0_gate.log_alpha.data[h_idx] = torch.max(
                        eql.l0_gate.log_alpha.data[h_idx], 
                        eql.l0_gate.log_alpha.data[next_h_idx]
                    )
                    eql.l0_gate.log_alpha.data[next_h_idx] = -10.0

        # --- TASK 3B: SIMPLIFY TO POLYNOMIALS (With Least-Squares Optimization) ---
        # Refresh params again to get updated n/K values for projection math
        params = self.extract_params(full=True)
        
        for i in range(num_hill):
            h_idx = num_poly + i
            if torch.abs(eql.fc.weight.data[0, h_idx]) < 1e-8: continue
            
            f_hill = features[:, h_idx]
            
            for p_idx in range(num_poly):
                f_poly = features[:, p_idx]
                
                # 1. Gatekeeper: Shape Similarity (Normalized)
                diff = torch.mean(torch.abs(f_hill/f_hill.max() - f_poly/f_poly.max()))
                
                if diff < epsilon:
                    # 2. Optimization: Find optimal weight multiplier m*
                    # m* = dot(f_hill, f_poly) / norm(f_poly)^2
                    dot_product = torch.sum(f_hill * f_poly)
                    poly_norm_sq = torch.sum(f_poly * f_poly)
                    m_star = dot_product / (poly_norm_sq + 1e-12)
                    
                    print(f"Simplifying Hill {h_idx} to Poly {p_idx}")
                    print(f"  Shape Diff: {diff:.4f}, Multiplier: {m_star:.4f}")
                    
                    # 3. Transfer Weight (scaled) and Gate log_alpha
                    eql.fc.weight.data[0, p_idx] += eql.fc.weight.data[0, h_idx] * m_star
                    eql.l0_gate.log_alpha.data[p_idx] = eql.l0_gate.log_alpha.data[h_idx].clone()
                    
                    # 4. Kill Hill
                    eql.fc.weight.data[0, h_idx] = 0.0
                    eql.l0_gate.log_alpha.data[h_idx] = -10.0
                    break

        # Final Sync
        _ = self.extract_params(full=True)
        print("Fine-tuning committed.")

    def _average_hill_params(self, idx1, idx2):
        """
        Helper to average n and logK for two Hill modules.
        Ensures 'Consolidated Hills' maintain correct physical shapes.
        """
        eql = self.reaction.eql_layer
        
        # Construct a flat list of Hill modules following Sequential Ptr logic
        all_hf = []
        for hm in eql.hill.hill_modules:
            all_hf.extend(hm.hill_inc_raw + list(hm.hill_inc_cross.values()))
            all_hf.extend(hm.hill_dec_raw + list(hm.hill_dec_cross.values()))
            
        hf1 = all_hf[idx1]
        hf2 = all_hf[idx2]
        
        with torch.no_grad():
            # Update primary module with average parameters
            hf1.raw_n.data = (hf1.raw_n.data + hf2.raw_n.data) / 2.0
            # hf1.raw_logK.data = (hf1.raw_logK.data + hf2.raw_logK.data) / 2.0
            hf1.raw_logK.data = (hf1.raw_K.data + hf2.raw_K.data) / 2.0
            
            # Prune parameters of merged module
            hf2.raw_n.data.fill_(0.0)
            # hf2.raw_logK.data.fill_(0.0)
            hf2.raw_K.data.fill_(0.0)
        
    def generate_equation(self, eps=1e-12):
        p = self.extract_params(full=True)
        poly_terms = p['poly_terms']
        hill_terms = p['hill_terms']
        dup = int(self.duplicates)
        species = ['u','v']
        terms = []

        # poly coeffs unscaled
        poly_coeffs = np.asarray(p['poly_coeffs_unscaled']) if 'poly_coeffs_unscaled' in p else np.asarray(p.get('raw_w_unscaled', []))[:p['num_poly']]

        for term, coeff in zip(poly_terms * dup, poly_coeffs):
            if abs(coeff) > eps:
                s = f"{float(coeff):.3f}"
                for ind in term:
                    s += f" * {species[ind]}"
                terms.append(s)

        # increasing hills (use unscaled coefficients and Ks and ns)
        inc_b = np.asarray(p['hill_inc_unscaled'])
        Ks_inc = np.asarray(p['Ks_inc_unscaled'])
        ns_inc = np.asarray(p.get('ns_inc_unscaled', p.get('ns_inc', [])))

        for term, coeff, K, n in zip(hill_terms * dup, inc_b, Ks_inc, ns_inc):
            if abs(coeff) <= eps: 
                continue
            coeff_f = float(coeff); K_f = float(K); n_f = float(n)
            if len(term) == 1:
                s = f"{coeff_f:.3f} * {species[term[0]]}^{n_f:.3f} / (1 + {K_f:.3f} * {species[term[0]]}^{n_f:.3f})"
            else:
                s = f"{coeff_f:.3f} * {species[term[1]]} * {species[term[0]]}^{n_f:.3f} / (1 + {K_f:.3f} * {species[term[0]]}^{n_f:.3f})"
            terms.append(s)

        # decreasing hills
        dec_b = np.asarray(p['hill_dec_unscaled'])
        Ks_dec = np.asarray(p['Ks_dec_unscaled'])
        ns_dec = np.asarray(p.get('ns_dec_unscaled', p.get('ns_dec', [])))

        for term, coeff, K, n in zip(hill_terms * dup, dec_b, Ks_dec, ns_dec):
            if abs(coeff) <= eps:
                continue
            coeff_f = float(coeff); K_f = float(K); n_f = float(n)
            if len(term) == 1:
                s = f"{coeff_f:.3f} * (1 / {K_f:.3f} - {species[term[0]]}^{n_f:.3f} / (1 + {K_f:.3f} * {species[term[0]]}^{n_f:.3f}))"
            else:
                s = f"{coeff_f:.3f} * {species[term[1]]} * (1 / {K_f:.3f} - {species[term[0]]}^{n_f:.3f} / (1 + {K_f:.3f} * {species[term[0]]}^{n_f:.3f}))"
            terms.append(s)

        return terms

    def eval_equation_from_params(self, uv_np, dec=10):
        """
        Evaluate analytic equation described by params at points uv_np (N,2).
        domain: 'unscaled' -> evaluate in original u,v using _unscaled arrays (the default pretty equation)
                'scaled'   -> evaluate in scaled domain (u',v') using *_scaled arrays (so matches model input).
        """
        params = self.extract_params()
        s_u, s_v = self.max_scale[0, 0], self.max_scale[0, 1]
        
        u = np.asarray(uv_np)[:,0].astype(float)
        v = np.asarray(uv_np)[:,1].astype(float)
        N = len(u)
        z = np.zeros(N, dtype=float)

        poly_terms = params['poly_terms']
        hill_terms = params['hill_terms']
        dup = int(params.get('duplicates', 1))

        poly_coeffs = np.round(np.asarray(params['poly_coeffs_unscaled']), dec)
        inc_b = np.round(np.asarray(params['hill_inc_unscaled']), dec)
        dec_b = np.round(np.asarray(params['hill_dec_unscaled']), dec)
        Ks_inc = np.round(np.asarray(params['Ks_inc_unscaled']), dec)
        Ks_dec = np.round(np.asarray(params['Ks_dec_unscaled']), dec)
        ns_inc = np.round(np.asarray(params['ns_inc']), dec)
        ns_dec = np.round(np.asarray(params['ns_dec']), dec)

        # polynomials
        for term, coeff in zip(poly_terms * dup, poly_coeffs):
            if abs(coeff) < 1e-12:
                continue
            feat = np.ones(N)
            for ind in term:
                feat = feat * (u if ind == 0 else v)
            z += float(coeff) * feat

        # inc hills
        for term, coeff, K, n in zip(hill_terms * dup, inc_b, Ks_inc, ns_inc):
            if abs(coeff) < 1e-12:
                continue
            if len(term) == 1:
                reg = u if term[0] == 0 else v
                term_val = (reg ** n) / (1.0 + K * (reg ** n))
            else:
                reg = u if term[0] == 0 else v
                mult = u if term[1] == 0 else v
                term_val = mult * ((reg ** n) / (1.0 + K * (reg ** n)))
            z += float(coeff) * term_val
            
        # dec hills
        for term, coeff, K, n in zip(hill_terms * dup, dec_b, Ks_dec, ns_dec):
            if abs(coeff) < 1e-12:
                continue
            if len(term) == 1:
                reg = u if term[0] == 0 else v
                term_val = (1.0 / K) - (reg ** n) / (1.0 + K * (reg ** n))
            else:
                reg = u if term[0] == 0 else v
                mult = u if term[1] == 0 else v
                term_val = mult * ((1.0 / K) - (reg ** n) / (1.0 + K * (reg ** n)))
            z += float(coeff) * term_val
        return z