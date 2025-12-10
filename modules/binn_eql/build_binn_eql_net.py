import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
               
        # MLP
        self.mlp = build_mlp(
            input_features=input_features, 
            layers=layers,
            activation=nn.Tanh(),
            linear_output=False,
            output_activation=nn.Softplus()) # Softplus ensures u,v > 0

    def forward(self, inputs):
        # inputs are [-1, 1]
        return self.mlp(inputs) # Outputs [0, 1] roughly (scaled space)

class F_EQL(nn.Module):
    def __init__(self, species, duplicates, param_bounds):
        super(F_EQL, self).__init__()
        self.eql_layer = EQLLayer(species, duplicates, param_bounds)

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
        # s_u_max = torch.quantile(train_data[:, -2].abs(), 0.99)
        # s_v_max = torch.quantile(train_data[:, -1].abs(), 0.99)
        s_u_max = torch.max(train_data[:, -2].abs())
        s_v_max = torch.max(train_data[:, -1].abs())
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
        self.reaction = F_EQL(species, duplicates, self.param_bounds)
        
        # Sampling config
        self.num_samples = 10000
        self.name = 'Dumlp_Dvmlp_Fmlp'

    # -----------------------
    # Normalization Helpers
    # -----------------------
    def scale_inputs(self, inputs):
        """ Maps Physical [lb, ub] -> Dimensionless [-1, 1] """
        return 2.0 * (inputs - self.lb) / (self.ub - self.lb) - 1.0

    def forward(self, inputs):
        """ Returns PREDICTED u (Scaled [0,1]) from Physical Inputs """       
        # 1. Normalize Inputs
        inputs_hat = self.scale_inputs(inputs)
        
        # 2. Predict Surface (Dimensionless)
        return self.surface_fitter(inputs_hat)

    def get_physical_derivatives(self, inputs_hat, u_hat):
        """ 
        Calculates du/dt and d2u/dx2 in PHYSICAL units 
        using the Chain Rule on the scaled variables.
        """
        # 1. Compute gradients in SCALED space (d_u_hat / d_x_hat)
        grads = torch.autograd.grad(
            u_hat, inputs_hat, 
            grad_outputs=torch.ones_like(u_hat), 
            create_graph=True
        )[0]
        
        dudx_hat = grads[:, :-1] # Spatial dims
        dudt_hat = grads[:, -1:] # Time dim
        
        # 2. Compute Second Derivative (d2_u_hat / d_x_hat2)
        # We assume 1D or 2D space. 
        d2udx2_hat_list = []
        for i in range(self.dimensions):
             g2 = torch.autograd.grad(
                dudx_hat[:, i], inputs_hat,
                grad_outputs=torch.ones_like(dudx_hat[:, i]),
                create_graph=True
             )[0][:, i]
             d2udx2_hat_list.append(g2)
        d2udx2_hat = torch.stack(d2udx2_hat_list, dim=1)

        # 3. Chain Rule Factors
        dt_factor = (2.0 / self.t_range)
        dx_factor = (2.0 / self.x_range)
        
        # 4. Convert to Physical Units
        # u_t_phys = (d_uhat/d_that) * (max_scale) * (d_that/dt)
        u_t_phys = dudt_hat * self.max_scale * dt_factor
        
        # u_xx_phys = (d2_uhat/d_xhat2) * (max_scale) * (d_xhat/dx)^2
        u_xx_phys = d2udx2_hat.unsqueeze(2) * self.max_scale.unsqueeze(1) * (dx_factor ** 2)
        
        # Return shapes: u_t [N, Species], u_xx [N, Dims, Species]
        # Reshape u_xx for compatibility with laplacian sum later
        u_xx_phys = u_xx_phys.permute(0, 2, 1) # [N, Species, Dims]
        
        return u_t_phys, u_xx_phys

    # -----------------------
    # Loss Functions
    # -----------------------
    # def gls_loss(self, pred, true):
    #     # pred is Dimensionless [0, 1], true is physical       
    #     # Convert pred to physical for loss calculation (or scale true down)
    #     # Scaling true down is numerically more stable for GLS denominator
    #     true_scaled = true / self.mean_scale
    #     pred_phys = pred * self.max_scale
    #     pred_scaled = pred_phys / self.mean_scale
        
    #     residual = (pred_scaled - true_scaled)**2
    #     return torch.mean(residual)

    def gls_loss(self, pred, true):
            # 1. Scale True Data (Physical -> Dimensionless)
            true_norm = true / self.max_scale
            
            # 2. Calculate Squared Residuals
            # pred is already [0,1], true_norm is [0,1]
            residuals = (pred - true_norm)**2
            
            # 3. Hard Example Mining (Top 10%)
            # We flatten the batch and spatial dims to find the worst individual points
            res_flat = residuals.view(-1)
            
            # Calculate number of hard examples (e.g., 10% of total pixels)
            num_hard = int(0.10 * res_flat.numel())
            
            # Select the top k largest errors
            # This automatically finds the wave front edges and peak errors
            top_k_loss, _ = torch.topk(res_flat, num_hard)
            
            # 4. Standard MSE (for stability) + Hard Loss (for sharpness)
            # We combine them so the network doesn't completely ignore the easy parts
            total_loss = torch.mean(residuals) + 4.0 * torch.mean(top_k_loss)
            
            return total_loss
    
    def pde_loss(self, inputs, epoch):
        # 1. Prepare Scaled Inputs (Requires Grad for PDE)
        inputs_hat = self.scale_inputs(inputs).requires_grad_(True)
        
        # 2. Re-Run Forward Pass (Tracked Graph)
        u_hat_tracked = self.surface_fitter(inputs_hat)
        
        # 3. Calculate Physical Derivatives
        # u_t: [N, Species], u_xx: [N, Species, Dims]
        u_t_phys, u_xx_phys = self.get_physical_derivatives(inputs_hat, u_hat_tracked)
        
        # 4. Calculate Reaction (Scaled -> Scaled)
        F_hat = self.reaction(u_hat_tracked)
        
        # 5. Scale Reaction to Physical Units (Output Scaling to u_max)
        F_phys = F_hat * self.max_scale[0, 0]

        # 6. Laplacian (Physical)
        lap_u_phys = torch.sum(u_xx_phys, dim=2) # Sum over dims -> [N, Species]

        # 7. Diffusion (Physical)
        if self.diff_coeffs:
             D = torch.tensor(self.diff_coeffs).to(inputs.device)
        else:
             D = self.diffusion_fitter() 
        
        # 8. Residual (u and v specific)
        # u_t - (D*lap + F)        
        # Equation: u_t = Du * lap_u + F
        res_u = u_t_phys[:, 0:1] - (D[0] * lap_u_phys[:, 0:1] + F_phys)
        
        # Equation: v_t = Dv * lap_v - F
        res_v = u_t_phys[:, 1:2] - (D[1] * lap_u_phys[:, 1:2] - F_phys)
        
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

        return torch.mean(res_u**2 + res_v**2)
                    
    def reg_loss(self, epoch):
        # # 1. Get probabilities
        # gate_probs = self.reaction.eql_layer.l0_gate.expected_l0()
        
        # # 2. Slice
        # num_poly = self.reaction.eql_layer.num_poly_features
        # poly_probs = gate_probs[:num_poly]
        # hill_probs = gate_probs[num_poly:]
        
        # # 3. Luxury Tax
        # l0_poly = poly_probs.sum()
        # l0_hill = hill_probs.sum() * self.l05_weight
        # total_l0 = l0_poly + l0_hill
                
        # # 5. Total weighted L0 norm
        # total_l0 = l0_poly + l0_hill

        total_l0 = self.reaction.eql_layer.l0_gate.expected_l0().sum()
        
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
                K_val = torch.exp(hf.raw_logK)
                rawK_list.append(K_val.view(-1))

        K_vals = torch.cat(rawK_list)
        K_pen = small_K_hinge_penalty(K_vals)
                              
        return total_l0 + K_pen

    def loss(self, pred, true, epoch, pde_weight, l0_weight):       
        # GLS Loss
        self.gls_loss_val = self.gls_loss(pred, true)
        
        # PDE Sampling
        x = torch.empty(self.num_samples, self.dimensions, device=pred.device).uniform_(self.lb[0,0], self.ub[0,0])
        t = torch.empty(self.num_samples, 1, device=pred.device).uniform_(self.lb[0,-1], self.ub[0,-1])
        inputs_rand = torch.cat([x, t], dim=1)
        
        # PDE Loss
        self.pde_loss_val = pde_weight * self.pde_loss(inputs_rand, epoch)
              
        # Reg Loss            
        l0_loss = self.reg_loss(epoch)
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
        Return a dict of parameter arrays on CPU (numpy) that are safe to log/plot.
        Handles the conversion from Dimensionless Network Weights -> Physical Constants.
        """
        eql = self.reaction.eql_layer

        # ---------------------------------------------------------
        # 1. GET RAW NETWORK WEIGHTS (SCALED SPACE)
        # ---------------------------------------------------------
        # raw linear weights (1 x M) as tensor on device
        raw_w_t = eql.fc.weight[0].detach()

        # deterministic gate values (stretched-sigmoid proxy)
        try:
            gates_t = eql.l0_gate.get_gates().detach()  # tensor (M,)
        except Exception:
            # fallback: stretched-sigmoid proxy from log_alpha
            log_alpha = eql.l0_gate.log_alpha.detach()
            gamma = float(eql.l0_gate.gamma)
            zeta = float(eql.l0_gate.zeta)
            s = torch.sigmoid(log_alpha)
            s_stretched = s * (zeta - gamma) + gamma
            gates_t = s_stretched.clamp(0.0, 1.0)

        # effective (gated) weights (tensor)
        effective_t = (raw_w_t * gates_t).detach()

        # Convert main arrays to numpy (single batched transfers)
        raw_w = raw_w_t.cpu().numpy().reshape(-1)
        gates = gates_t.cpu().numpy().reshape(-1)
        effective = effective_t.cpu().numpy().reshape(-1)

        # ---------------------------------------------------------
        # 2. GATHER HILL PARAMETERS (n, K)
        # ---------------------------------------------------------
        num_poly = int(eql.num_poly_features)
        num_hill = int(eql.num_hill_features)
        poly_terms, hill_terms = self.generate_terms()  # lists for single duplicate
        n_hill_single = len(hill_terms)
        dup = int(self.duplicates)

        # Collect raw hill parameter tensors into lists (on device) then stack once
        raw_ns_inc_list = []
        raw_Ks_inc_list = []
        raw_ns_dec_list = []
        raw_Ks_dec_list = []

        for hill_module in eql.hill.hill_modules:
            # Increasing Terms
            for hf in hill_module.hill_inc_raw:
                raw_ns_inc_list.append(hf.raw_n.view(-1))
                raw_Ks_inc_list.append(hf.raw_logK.view(-1))
            for key in getattr(hill_module, 'hill_inc_cross', {}):
                hf = hill_module.hill_inc_cross[key]
                raw_ns_inc_list.append(hf.raw_n.view(-1))
                raw_Ks_inc_list.append(hf.raw_logK.view(-1))

            # Decreasing Terms
            for hf in hill_module.hill_dec_raw:
                raw_ns_dec_list.append(hf.raw_n.view(-1))
                raw_Ks_dec_list.append(hf.raw_logK.view(-1))
            for key in getattr(hill_module, 'hill_dec_cross', {}):
                hf = hill_module.hill_dec_cross[key]
                raw_ns_dec_list.append(hf.raw_n.view(-1))
                raw_Ks_dec_list.append(hf.raw_logK.view(-1))

        # Helper to stack
        def _stack_to_numpy(lst):
            if len(lst) == 0:
                return np.array([])
            stacked = torch.cat(lst, dim=0).view(-1)
            return stacked.detach().cpu().numpy()
        
        raw_ns_inc = _stack_to_numpy(raw_ns_inc_list)
        raw_Ks_inc = _stack_to_numpy(raw_Ks_inc_list)
        raw_ns_dec = _stack_to_numpy(raw_ns_dec_list)
        raw_Ks_dec = _stack_to_numpy(raw_Ks_dec_list)

        # Map raw parameters to numbers (Sigmoid/Exp as defined in HillFunction)
        # Note: Must match the math in your HillFunction class exactly!
        ns_inc = (1 / (1 + np.exp(-raw_ns_inc))) * 3 + 1
        ns_dec = (1 / (1 + np.exp(-raw_ns_dec))) * 3 + 1
        Ks_inc = np.exp(raw_Ks_inc)
        Ks_dec = np.exp(raw_Ks_dec)

        # Diffusion (if present)
        D_vals = None
        if hasattr(self, 'diffusion_fitter') and self.diffusion_fitter is not None:
            try:
                with torch.no_grad():
                    D_vals = self.diffusion_fitter().detach().cpu().numpy()
            except Exception:
                D_vals = None

        # ---------------------------------------------------------
        # 3. UNSCALING LOGIC (DIMENSIONLESS -> PHYSICAL)
        # ---------------------------------------------------------
        # 1. Get Input Scales
        s_u, s_v = self.max_scale[0, 0].item(), self.max_scale[0, 1].item()
        
        # 2. Get Output Scale
        # Assuming single output F for species u. 
        # If your EQL outputs a vector [F_u, F_v], you need to select s_u or s_v accordingly.
        # Here we assume standard reaction-diffusion where F is the rate for u.
        s_out = s_u 

        # --- A. POLYNOMIALS ---
        poly_coeffs_scaled = effective[:num_poly] if num_poly > 0 else np.array([])
        poly_coeffs_unscaled = []
        for term_tuple, coeff_scaled in zip(poly_terms * dup, poly_coeffs_scaled):
            # Count powers
            p = sum(1 for ind in term_tuple if ind == 0)
            q = sum(1 for ind in term_tuple if ind == 1)
            
            # Math: W_phys = W_net * S_out / (S_u^p * S_v^q)
            input_scale = (s_u ** p) * (s_v ** q)
            a_orig = (coeff_scaled * s_out) / (input_scale + 1e-9)
            
            poly_coeffs_unscaled.append(a_orig)
        poly_coeffs_unscaled = np.array(poly_coeffs_unscaled)

        # --- B. PREPARE HILL BLOCKS ---
        hill_block = effective[num_poly : num_poly + num_hill] if num_hill > 0 else np.array([])
        if hill_block.size:
            # Try to reshape if strict structure exists
            try:
                hb = hill_block.reshape(dup, 2 * n_hill_single)
            except Exception:
                hb = hill_block.reshape(dup, -1)
            hill_inc_all = hb[:, :n_hill_single].reshape(-1) if n_hill_single > 0 else np.array([])
            hill_dec_all = hb[:, n_hill_single:].reshape(-1) if n_hill_single > 0 else np.array([])
        else:
            hill_inc_all = np.array([])
            hill_dec_all = np.array([])

        # --- C. INCREASING HILL TERMS ---
        # Form: Amp * (u^n) / (1 + K*u^n)
        hill_inc_unscaled = []
        Ks_inc_unscaled = []
        
        for (term_tuple, coeff_scaled, K_scaled, n_val) in zip(hill_terms * dup, hill_inc_all, Ks_inc, ns_inc):
            n_f = float(n_val)
            reg_species = term_tuple[0] # Species inside the Hill function
            
            # Check for Multiplier (e.g. v * Hill(u))
            multiplier_power = 1 if len(term_tuple) > 1 else 0
            mult_species = term_tuple[1] if multiplier_power else None

            s_reg = s_u if reg_species == 0 else s_v
            s_mult = 1.0
            if multiplier_power:
                s_mult = s_u if mult_species == 0 else s_v

            # 1. Unscale Amplitude
            # Amp_phys = Amp_net * S_out / (S_reg^n * S_mult)
            b_orig = (coeff_scaled * s_out) / ((s_reg ** n_f) * (s_mult ** multiplier_power) + 1e-9)
            hill_inc_unscaled.append(float(b_orig))
            
            # 2. Unscale K
            # K_phys = K_net / (S_reg^n)
            # Because net term is 1 + K_net*(u/S)^n = 1 + (K_net/S^n)*u^n
            K_orig = float(K_scaled / (s_reg ** n_f + 1e-9))
            Ks_inc_unscaled.append(K_orig)
            
        hill_inc_unscaled = np.array(hill_inc_unscaled)
        Ks_inc_unscaled = np.array(Ks_inc_unscaled)

        # --- D. DECREASING HILL TERMS ---
        # Form: Amp * [ (1/K) - u^n/(1 + K*u^n) ]
        hill_dec_unscaled = []
        Ks_dec_unscaled = []
        
        for (term_tuple, coeff_scaled, K_scaled, n_val) in zip(hill_terms * dup, hill_dec_all, Ks_dec, ns_dec):
            n_f = float(n_val)
            reg_species = term_tuple[0]
            multiplier_power = 1 if len(term_tuple) > 1 else 0
            mult_species = term_tuple[1] if multiplier_power else None

            s_reg = s_u if reg_species == 0 else s_v
            s_mult = 1.0
            if multiplier_power:
                s_mult = s_u if mult_species == 0 else s_v

            # 1. Unscale Amplitude
            # Same logic as increasing
            b_orig = (coeff_scaled * s_out) / ((s_reg ** n_f) * (s_mult ** multiplier_power) + 1e-9)
            hill_dec_unscaled.append(float(b_orig))
            
            # 2. Unscale K
            K_orig = float(K_scaled / (s_reg ** n_f + 1e-9))
            Ks_dec_unscaled.append(K_orig)
            
        hill_dec_unscaled = np.array(hill_dec_unscaled)
        Ks_dec_unscaled = np.array(Ks_dec_unscaled)

        # --- E. RECONSTRUCT RAW WEIGHTS VECTOR ---
        # This is for visualization/logging consistency
        raw_w_unscaled_list = []
        if poly_coeffs_unscaled.size:
            raw_w_unscaled_list.extend(poly_coeffs_unscaled.tolist())
        if hill_inc_all.size:
            # Must maintain network order: Inc block then Dec block (per duplicate)
            raw_w_unscaled_list.extend(hill_inc_unscaled.tolist())
            raw_w_unscaled_list.extend(hill_dec_unscaled.tolist())

        raw_w_unscaled = np.array(raw_w_unscaled_list) if len(raw_w_unscaled_list) else np.array([])
        effective_unscaled = (raw_w_unscaled * gates)
        
        # -----------------------
        # RETURN
        # -----------------------
        if not full:
            return {
                'raw_w_unscaled': raw_w_unscaled,
                'effective_unscaled': effective_unscaled
            }

        return {
            'raw_w': raw_w,
            'raw_w_unscaled': raw_w_unscaled,
            'gates': gates,
            'effective': effective,
            'effective_unscaled': effective_unscaled,
            'num_poly': num_poly,
            'num_hill': num_hill,
            'ns_inc': ns_inc,
            'Ks_inc': Ks_inc,
            'ns_dec': ns_dec,
            'Ks_dec': Ks_dec,
            'D': D_vals,
            'poly_terms': poly_terms,
            'hill_terms': hill_terms,
            'poly_coeffs_unscaled': poly_coeffs_unscaled,
            'hill_inc_unscaled': hill_inc_unscaled,
            'hill_dec_unscaled': hill_dec_unscaled,
            'Ks_inc_unscaled': Ks_inc_unscaled,
            'Ks_dec_unscaled': Ks_dec_unscaled
        }
        
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