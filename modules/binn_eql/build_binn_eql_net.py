import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils

from modules.binn_eql.build_mlp import build_mlp
from modules.utils.gradient import gradient
from modules.binn_eql.build_eql_layer import EQLLayer

# ---------------------------------------------------------
# 1. SUB-NETWORKS
# ---------------------------------------------------------
# class D_PARAMS(nn.Module):
#     def __init__(self, input_features=2, min_val=0.01, max_val=10.0):
#         super().__init__()
#         self.min_val = min_val
#         self.max_val = max_val
        
#         # Initialize at 0.0. 
#         # sigmoid(0) = 0.5, which puts the initial D exactly 
#         # halfway between min and max in log space (e.g., D = 0.1)
#         self.raw_D = nn.Parameter(torch.zeros(input_features))
        
#     def forward(self):     
#         s = torch.sigmoid(self.raw_D)
#         return self.min_val * (self.max_val / self.min_val) ** s
    
class D_PARAMS(nn.Module):
    def __init__(self, input_features=2, base_val=0.5, noise_std=1):
        super().__init__()
        base_log = torch.log(torch.tensor(base_val))         
        noise = torch.randn(input_features) * noise_std
        self.raw_D = nn.Parameter(base_log + noise)
        
    def forward(self):     
        return torch.exp(self.raw_D)
    
class uv_MLP(nn.Module):
    def __init__(self, input_features, layers=[256, 256, 256, 2]):
        super().__init__()
        # 1. Pass GELU directly (build_mlp accepts an activation arg)
        self.mlp = build_mlp(
            input_features=input_features, 
            layers=layers,
            activation=nn.GELU(),
            linear_output=False,
            output_activation=nn.Softplus())

        # 2. Apply Weight Norm "Post-Hoc"
        for module in self.mlp.MLP:
            if isinstance(module, nn.Linear):
                utils.parametrizations.weight_norm(module)
                
    def forward(self, inputs):
        return self.mlp(inputs)

class F_EQL(nn.Module):
    def __init__(self, species, duplicates, param_bounds, max_scale, degree):
        super(F_EQL, self).__init__()
        # Pass max_scale down to EQLLayer for physical conversion
        self.eql_layer = EQLLayer(species, duplicates, param_bounds, max_scale, degree)

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
        self.degree = degree

        # ---------------------------------------------------------
        # A. REGISTER BOUNDS & SCALES (Buffers)
        # ---------------------------------------------------------
        # Spatial/Temporal Bounds
        x_min = torch.min(train_data[:, :dimensions])
        x_max = torch.max(train_data[:, :dimensions])
        t_min = torch.min(train_data[:, dimensions])
        t_max = torch.max(train_data[:, dimensions])
        
        # Register for Input Normalization [-1, 1]
        lb_tensor = torch.cat([torch.full((dimensions,), x_min), torch.tensor([t_min])])
        ub_tensor = torch.cat([torch.full((dimensions,), x_max), torch.tensor([t_max])])
        self.register_buffer('lb', lb_tensor.view(1, -1)) 
        self.register_buffer('ub', ub_tensor.view(1, -1))
                        
        # Concentration Scales (Physical -> Dimensionless)
        s_u_max = torch.quantile(train_data[:, -2].abs(), 0.99)
        s_v_max = torch.quantile(train_data[:, -1].abs(), 0.99)
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
            self.diffusion_fitter = D_PARAMS(self.species)
        else:
            self.diffusion_fitter = None
                
        # Surface Fitter (Dimensionless)
        if uv_layers:
            self.surface_fitter = uv_MLP(input_features=dimensions+1, layers=uv_layers)
        else:
            self.surface_fitter = uv_MLP(input_features=dimensions+1)
        
        # Reaction (Input: Normalized -> Output: Unscaled Rate)
        # We pass max_scale so EQLLayer can calculate physical values for the Loss
        self.reaction = F_EQL(species, duplicates, self.param_bounds, self.max_scale, self.degree)
        
        # Sampling config
        self.num_samples = 10000
        self.name = 'Dumlp_Dvmlp_Fmlp'
        
    def normalize(self, inputs):
        """ Maps Physical [lb, ub] -> Dimensionless [-1, 1] """
        return 2.0 * (inputs - self.lb) / (self.ub - self.lb) - 1.0

    def forward(self, inputs):
        """ Returns PREDICTED u (Scaled [0,1]) from Physical Inputs """       
        inputs_hat = self.normalize(inputs)
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
        # u_scaled = u / self.max_scale # Normalize inputs for EQL

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
        # F = self.reaction(u_scaled)
        F = self.reaction(u)
        
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
        
        return torch.mean(pde_loss)
                        
    def reg_loss(self, lux_tax, epoch):
        """
        Soft Wall Regularization:
        1. L0 Sparsity
        2. Physical Bound Penalty (ReLU(|w_phys| - bound))
        """
        # 1. L0 Sparsity
        gate_probs = self.reaction.eql_layer.l0_gate.expected_l0()
        num_poly = self.reaction.eql_layer.num_poly_features
        
        l0_poly = gate_probs[:num_poly].sum()
        l0_hill = gate_probs[num_poly:].sum() * lux_tax
        total_l0 = l0_poly + l0_hill

        return total_l0
    
    def soft_wall_loss(self):
        # We pass epsilon=0.15 to ensure the curve doesn't saturate 
        # before 15% of the physical domain.
        w_phys, k_phys, k_ceilings = self.reaction.eql_layer.get_physical_parameters(epsilon=0.15)
        
        # A. Weight Penalty (Rates)
        w_violation = torch.relu(torch.abs(w_phys) - self.param_bounds)
        w_loss = torch.sum(w_violation) * 100 
        
        # B. Dynamic K Penalty (Affinities)
        k_violation = torch.relu(k_phys - k_ceilings)
        k_loss = torch.sum(k_violation) * 100

        # C. Diffusion L2 Anchor (The Anti-Scaling Fix)
        d_loss = 0.0
        # if not self.diff_coeffs: # Only apply if we are actively learning D
        #     D_phys = self.diffusion_fitter()
        #     # Penalize the square of the physical magnitude. 
        #     # A multiplier of 1.0 or 0.1 provides a gentle but firm downward pressure.
        #     d_loss = torch.sum(D_phys ** 2) * 1.0 

        return w_loss + k_loss + d_loss
        
    def loss(self, pred, true, epoch, lux_tax):       
        # 1. GLS Loss (RAW)
        raw_gls = self.gls_loss(pred, true)
        
        # 2. PDE Sampling
        x = torch.empty(self.num_samples, self.dimensions, device=pred.device).uniform_(self.lb[0,0], self.ub[0,0])
        t = torch.empty(self.num_samples, 1, device=pred.device).uniform_(self.lb[0,-1], self.ub[0,-1])
        inputs_rand = torch.cat([x, t], dim=1).requires_grad_()
        inputs_rand_norm = self.normalize(inputs_rand)
        outputs_rand = self.surface_fitter(inputs_rand_norm)
             
        # 3. PDE Loss (RAW)
        raw_pde = self.pde_loss(inputs_rand, outputs_rand, epoch)
              
        # 4. Reg Loss (RAW L0)          
        raw_l0 = self.reg_loss(lux_tax, epoch)
        
        # 5. Soft Wall (RAW - Always Enforced)
        raw_softwall = self.soft_wall_loss()
                      
        # Return all 4 separated, unweighted tensors
        return raw_gls, raw_pde, raw_l0, raw_softwall
    
    # -----------------------
    # Parameter Extraction (Unscaling)
    # -----------------------
    def generate_terms(self):
        # 1. Dynamically grab the exact powers used by the layer!
        # Returns tuples like (3, 0) for u^3, or (1, 2) for u*v^2
        poly_terms = self.reaction.eql_layer.poly.powers
        
        hill_terms = []
        # Raw Hill
        for i in range(self.species): hill_terms.append((i,))
        # Cross Hill
        for i in range(self.species):
            for j in range(self.species):
                if i != j: hill_terms.append((i, j))
                
        return poly_terms, hill_terms
    
    def extract_params(self, full=True):
        """
        Extracts PHYSICAL parameters from the BINN.
        
        - Network Weights (w) are ALREADY physical (w_net = w_phys).
        - Hill K parameters (K) need unscaling (K_phys = K_net / S^n).
        """
        eql = self.reaction.eql_layer
        
        # 1. Get Network Weights (Now Physical)
        # We clone to avoid modifying the graph
        raw_w_t = eql.fc.weight[0].detach() 
        w_phys_t = raw_w_t  # No division needed!
        
        # 2. Get Gates
        try:
            gates_t = eql.l0_gate.get_gates().detach()
        except:
            log_alpha = eql.l0_gate.log_alpha.detach()
            gates_t = torch.sigmoid(log_alpha).clamp(0.0, 1.0)

        # 3. Calculate Effective Physical Weights
        effective_t = w_phys_t * gates_t

        # 4. Get Scales (S^n) - Needed ONLY for unscaling Hill K's
        # scales_t = eql._generate_scales_fast().view(-1).detach()

        # Convert to numpy for export
        raw_w = raw_w_t.cpu().numpy().reshape(-1)
        raw_w_phys = w_phys_t.cpu().numpy().reshape(-1)
        gates = gates_t.cpu().numpy().reshape(-1)
        effective = effective_t.cpu().numpy().reshape(-1)

        # Gather structure
        num_poly = int(eql.num_poly_features)
        num_hill = int(eql.num_hill_features)
        poly_terms, hill_terms = self.generate_terms()
        n_hill_single = len(hill_terms)
        dup = int(self.duplicates)
        
        s_u, s_v = self.max_scale[0, 0].item(), self.max_scale[0, 1].item()

        # --- EXTRACT HILL PARAMS (K_phys = K_net / S^n) ---
        # We must iterate through the Hill modules to unscale K using the correct S^n
        
        raw_ns_inc_list, raw_Ks_inc_list = [], []
        raw_ns_dec_list, raw_Ks_dec_list = [], []
        
        # Flat list of all hill functions for easy indexing if needed, 
        # but iterating the module list is safer for matching S_u vs S_v
        
        for hill_module in eql.hill.hill_modules:
            # def get_vals(module, base_scale):
            #     n = torch.sigmoid(module.raw_n) * 3 + 1
            #     k_net = F.softplus(module.raw_K)
            #     # Unscale K: K_phys = K_net / S^n
            #     k_phys = k_net / (base_scale ** n)
            #     return n.item(), k_phys.item()
            
            def get_vals(module, base_scale):
                n = torch.sigmoid(module.raw_n) * 3 + 1
                # K is already physical
                k_phys = F.softplus(module.raw_K)
                return n.item(), k_phys.item()

            # Inc Raw
            for i in range(self.species):
                n, k = get_vals(hill_module.hill_inc_raw[i], s_u if i==0 else s_v)
                raw_ns_inc_list.append(n); raw_Ks_inc_list.append(k)
            # Inc Cross
            for i in range(self.species):
                for j in range(self.species):
                    if i != j:
                        key = f"{i}_{j}"
                        n, k = get_vals(hill_module.hill_inc_cross[key], s_u if i==0 else s_v)
                        raw_ns_inc_list.append(n); raw_Ks_inc_list.append(k)
            # Dec Raw
            for i in range(self.species):
                n, k = get_vals(hill_module.hill_dec_raw[i], s_u if i==0 else s_v)
                raw_ns_dec_list.append(n); raw_Ks_dec_list.append(k)
            # Dec Cross
            for i in range(self.species):
                for j in range(self.species):
                    if i != j:
                        key = f"{i}_{j}"
                        n, k = get_vals(hill_module.hill_dec_cross[key], s_u if i==0 else s_v)
                        raw_ns_dec_list.append(n); raw_Ks_dec_list.append(k)

        ns_inc = np.array(raw_ns_inc_list)
        Ks_inc = np.array(raw_Ks_inc_list)
        ns_dec = np.array(raw_ns_dec_list)
        Ks_dec = np.array(raw_Ks_dec_list)

        # --- ORGANIZE ARRAYS ---
        poly_coeffs_unscaled = effective[:num_poly] if num_poly > 0 else np.array([])
        
        hill_block = effective[num_poly : num_poly + num_hill] if num_hill > 0 else np.array([])
        hill_inc_unscaled_list = []
        hill_dec_unscaled_list = []
        ptr = 0
        
        for d in range(dup):
            hill_inc_unscaled_list.extend(hill_block[ptr : ptr + n_hill_single])
            ptr += n_hill_single
            hill_dec_unscaled_list.extend(hill_block[ptr : ptr + n_hill_single])
            ptr += n_hill_single

        hill_inc_unscaled = np.array(hill_inc_unscaled_list)
        hill_dec_unscaled = np.array(hill_dec_unscaled_list)

        # K's are already unscaled by the loop above
        Ks_inc_unscaled = Ks_inc
        Ks_dec_unscaled = Ks_dec

        # --- RECONSTRUCT RAW_W_UNSCALED ---
        raw_w_unscaled = raw_w_phys

        if not full:
            return {'raw_w_unscaled': raw_w_unscaled, 'effective_unscaled': effective}

        return {
            'raw_w': raw_w,                 # Network weights (Physical)
            'raw_w_unscaled': raw_w_unscaled, # Physical weights (Same as raw_w)
            'gates': gates,
            'effective': effective,         # Effective Physical weights
            'effective_unscaled': effective, 
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
        Sequence: Zeroing -> Poly Merging -> Hill Merging -> Poly Simplification.
        
        Improvements:
        1. Uses Synthetic Grid for density-independent shape comparison.
        2. Enforces 'Same-Form' check so Inc/Dec terms aren't mixed.
        3. Uses 'epsilon' for both merging and simplification.
        """
        eql = self.reaction.eql_layer
        device = eql.fc.weight.device
        
        # --- TASK 0: SYNTHETIC GRID GENERATION ---
        # Robust 100x100 mesh to capture all feature behaviors
        steps = 100 
        u_space = torch.linspace(0, 1, steps, device=device)
        v_space = torch.linspace(0, 1, steps, device=device)
        grid_u, grid_v = torch.meshgrid(u_space, v_space, indexing='ij')
        
        # Shape (10000, 2)
        uv_synthetic = torch.stack([grid_u.flatten(), grid_v.flatten()], dim=1)
        
        # Calculate ALL features on this grid
        features = eql.get_features(uv_synthetic) 
        
        # --- TASK 1: ZEROING (Pruning Noise) ---
        params = self.extract_params(full=True)
        eff_unscaled = torch.tensor(params['effective_unscaled'], device=device)
        
        # Identify weak terms
        small_mask = torch.abs(eff_unscaled) < threshold
        eql.fc.weight.data[0, small_mask] = 0.0
        eql.l0_gate.log_alpha.data[small_mask] = -10.0 # Lock gate

        # Refresh params/counts
        params = self.extract_params(full=True)
        num_poly = eql.num_poly_features
        num_hill = eql.num_hill_features
        
        # Determine the "Species" of each Hill term
        # e.g. If you have [Inc, Dec] repeated 5 times, n_hill_single = 2.
        # Term 0 is Inc, Term 1 is Dec, Term 2 is Inc...
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
                
                eql.l0_gate.log_alpha.data[primary] = torch.max(
                    eql.l0_gate.log_alpha.data[primary], 
                    eql.l0_gate.log_alpha.data[other]
                )
                eql.l0_gate.log_alpha.data[other] = -10.0

        # --- TASK 3A: MERGE DUPLICATE HILLS (Same Form Only) ---
        for i in range(num_hill):
            h_idx = num_poly + i
            if torch.abs(eql.fc.weight.data[0, h_idx]) < 1e-8: continue
            
            # 1. Identify Form: 0 for Inc, 1 for Dec (for example)
            form_id_i = (h_idx - num_poly) % n_hill_single
            
            f_hill = features[:, h_idx]
            # Center for Pearson Correlation
            f_hill_c = f_hill - torch.mean(f_hill)
            norm_hill_c = torch.norm(f_hill_c) + 1e-9
            
            for next_h_idx in range(h_idx + 1, num_poly + num_hill):
                if torch.abs(eql.fc.weight.data[0, next_h_idx]) < 1e-8: continue
                
                # 2. Strict Form Check
                form_id_next = (next_h_idx - num_poly) % n_hill_single
                
                # If they are different forms, skip immediately.
                if form_id_i != form_id_next:
                    continue
                
                # 3. Correlation Check
                f_other = features[:, next_h_idx]
                f_other_c = f_other - torch.mean(f_other)
                norm_other_c = torch.norm(f_other_c) + 1e-9
                
                correlation = torch.sum(f_hill_c * f_other_c) / (norm_hill_c * norm_other_c)
                dist = 1.0 - torch.abs(correlation)
                
                if dist < epsilon:
                    print(f"Merging Duplicate Hills: {h_idx} and {next_h_idx} (Dist: {dist:.4f})")
                    
                    # Merge
                    eql.fc.weight.data[0, h_idx] += eql.fc.weight.data[0, next_h_idx]
                    eql.fc.weight.data[0, next_h_idx] = 0.0
                    
                    self._average_hill_params(h_idx - num_poly, next_h_idx - num_poly)
                    
                    eql.l0_gate.log_alpha.data[h_idx] = torch.max(
                        eql.l0_gate.log_alpha.data[h_idx], 
                        eql.l0_gate.log_alpha.data[next_h_idx]
                    )
                    eql.l0_gate.log_alpha.data[next_h_idx] = -10.0

        # --- TASK 3B: SIMPLIFY HILLS TO POLYNOMIALS (1-to-1 Best Fit) ---
        hills_to_remove = []
        
        for i in range(num_hill):
            h_idx = num_poly + i
            current_hill_weight = eql.fc.weight.data[0, h_idx]
            if torch.abs(current_hill_weight) < 1e-8: continue
            
            f_hill = features[:, h_idx]
            norm_hill = torch.norm(f_hill) + 1e-9
            
            best_fit = {"p_idx": -1, "error": float('inf'), "m": 0.0}
            
            for p_idx in range(num_poly):
                f_poly = features[:, p_idx]
                norm_poly_sq = torch.sum(f_poly ** 2) + 1e-9
                
                # Optimal Multiplier
                m = torch.sum(f_hill * f_poly) / norm_poly_sq
                
                # Relative Residual Error
                residual = f_hill - (m * f_poly)
                rel_error = torch.norm(residual) / norm_hill
                
                if rel_error < best_fit["error"]:
                    best_fit["error"] = rel_error.item()
                    best_fit["p_idx"] = p_idx
                    best_fit["m"] = m.item()

            if best_fit["error"] < epsilon:
                p_idx = best_fit["p_idx"]
                m = best_fit["m"]
                
                print(f"Simplifying Hill {h_idx} -> Poly {p_idx}")
                print(f"  Error: {best_fit['error']:.4f}, Multiplier: {m:.4f}")
                
                eql.fc.weight.data[0, p_idx] += current_hill_weight * m
                
                eql.l0_gate.log_alpha.data[p_idx] = torch.max(
                    eql.l0_gate.log_alpha.data[p_idx],
                    eql.l0_gate.log_alpha.data[h_idx]
                )
                
                hills_to_remove.append(h_idx)

        # --- FINAL CLEANUP ---
        if hills_to_remove:
            indices = torch.tensor(hills_to_remove, device=device)
            eql.fc.weight.data[0, indices] = 0.0
            eql.l0_gate.log_alpha.data[indices] = -10.0
            
        _ = self.extract_params(full=True)
        print(f"Fine-tuning committed.")
                
    def _average_hill_params(self, idx1, idx2):
        """
        Helper to average n and raw_K for two Hill modules.
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
            hf1.raw_K.data = (hf1.raw_K.data + hf2.raw_K.data) / 2.0
            
            # Prune parameters of merged module
            hf2.raw_n.data.fill_(0.0)
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
                # 'term' is now a tuple of powers, e.g., (2, 1) for u^2 * v
                for i, power in enumerate(term):
                    if power == 1:
                        s += f" * {species[i]}"
                    elif power > 1:
                        s += f" * {species[i]}^{power}"
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