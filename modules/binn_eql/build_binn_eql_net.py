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
                utils.weight_norm(module)  # <--- Change 2: Weight Norm
                
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
        self.reaction = F_EQL(species, duplicates, self.param_bounds)
        
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
        inputs_rand = torch.cat([x, t], dim=1).requires_grad_()
        inputs_rand_norm = self.normalize(inputs_rand)
        outputs_rand = self.surface_fitter(inputs_rand_norm)
        
        # PDE Loss
        self.pde_loss_val = pde_weight * self.pde_loss(inputs_rand, outputs_rand,
                                                       epoch)
              
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

        If full is False returns only:
            {'raw_w_unscaled', 'gates', 'effective'}

        If full is True returns many more keys (see docstring in code).
        """
        eql = self.reaction.eql_layer

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

        # gather feature structure
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
            for hf in hill_module.hill_inc_raw:
                raw_ns_inc_list.append(hf.raw_n.view(-1))
                raw_Ks_inc_list.append(hf.raw_logK.view(-1))
            for key in getattr(hill_module, 'hill_inc_cross', {}):
                hf = hill_module.hill_inc_cross[key]
                raw_ns_inc_list.append(hf.raw_n.view(-1))
                raw_Ks_inc_list.append(hf.raw_logK.view(-1))

            for hf in hill_module.hill_dec_raw:
                raw_ns_dec_list.append(hf.raw_n.view(-1))
                raw_Ks_dec_list.append(hf.raw_logK.view(-1))
            for key in getattr(hill_module, 'hill_dec_cross', {}):
                hf = hill_module.hill_dec_cross[key]
                raw_ns_dec_list.append(hf.raw_n.view(-1))
                raw_Ks_dec_list.append(hf.raw_logK.view(-1))

        # stack (if empty, create empty numpy arrays)
        def _stack_to_numpy(lst):
            if len(lst) == 0:
                return np.array([])
            stacked = torch.cat(lst, dim=0).view(-1)
            return stacked.detach().cpu().numpy()
        
        raw_ns_inc = _stack_to_numpy(raw_ns_inc_list)
        raw_Ks_inc = _stack_to_numpy(raw_Ks_inc_list)
        raw_ns_dec = _stack_to_numpy(raw_ns_dec_list)
        raw_Ks_dec = _stack_to_numpy(raw_Ks_dec_list)

        # map raw to interpretable numeric params (numpy)
        if raw_ns_inc.size:
            # ns_inc = (1.0 / (1.0 + np.exp(-raw_ns_inc))) * 5.0
            ns_inc = (1 / (1 + np.exp(-raw_ns_inc))) * 3 + 1
        else:
            ns_inc = np.array([])
        if raw_ns_dec.size:
            # ns_dec = (1.0 / (1.0 + np.exp(-raw_ns_dec))) * 5.0
            ns_dec = (1 / (1 + np.exp(-raw_ns_dec))) * 3 + 1
        else:
            ns_dec = np.array([])

        if raw_Ks_inc.size:
            # Ks_inc = (1.0 / (1.0 + np.exp(-raw_Ks_inc))) * float(self.param_bounds)
            Ks_inc = np.exp(raw_Ks_inc)
        else:
            Ks_inc = np.array([])
        if raw_Ks_dec.size:
            # Ks_dec = (1.0 / (1.0 + np.exp(-raw_Ks_dec))) * float(self.param_bounds)
            Ks_dec = np.exp(raw_Ks_dec)
        else:
            Ks_dec = np.array([])

        # diffusion (if present) - try to get in one shot
        D_vals = None
        if hasattr(self, 'diffusion_fitter') and self.diffusion_fitter is not None:
            try:
                with torch.no_grad():
                    D_vals = self.diffusion_fitter().detach().cpu().numpy()
            except Exception:
                try:
                    rawD = self.diffusion_fitter.raw.detach().cpu().numpy()
                    s = 1.0 / (1.0 + np.exp(-rawD))
                    D_vals = s * float(self.param_bounds)
                except Exception:
                    D_vals = None

        # ------- Build unscaled coefficients (original units) -------
        # We assume EQL was trained on scaled inputs using self.u_scale_pct / self.v_scale_pct.
        s_u, s_v = self.max_scale[0, 0], self.max_scale[0, 1]
        
        # POLYNOMIALS: first num_poly entries correspond to poly_terms * duplicates ordering
        poly_coeffs_scaled = effective[:num_poly] if num_poly > 0 else np.array([])
        poly_coeffs_unscaled = []
        for term_tuple, coeff_scaled in zip(poly_terms * dup, poly_coeffs_scaled):
            # count powers of u (index 0) and v (index 1)
            p = sum(1 for ind in term_tuple if ind == 0)
            q = sum(1 for ind in term_tuple if ind == 1)
            a_orig = coeff_scaled / ((s_u ** p) * (s_v ** q) + 0.0)
            poly_coeffs_unscaled.append(a_orig.cpu().detach())
        poly_coeffs_unscaled = np.array(poly_coeffs_unscaled)

        # HILLS: extract hill block and split into inc/dec in duplicate-major order
        hill_block = effective[num_poly : num_poly + num_hill] if num_hill > 0 else np.array([])
        if hill_block.size:
            # shape (dup, 2 * n_hill_single)
            try:
                hb = hill_block.reshape(dup, 2 * n_hill_single)
            except Exception:
                # fall back: if shapes don't match, flatten to inc/dec halves conservatively
                hb = hill_block.reshape(dup, -1)
            hill_inc_all = hb[:, :n_hill_single].reshape(-1) if n_hill_single > 0 else np.array([])
            hill_dec_all = hb[:, n_hill_single:].reshape(-1) if n_hill_single > 0 else np.array([])
        else:
            hill_inc_all = np.array([])
            hill_dec_all = np.array([])

        # Unscale hill coefficients using learned n and s_u/s_v
        hill_inc_unscaled = []
        # Ks_inc, ns_inc arrays are in duplicate-major order already (from stacking)
        for (term_tuple, coeff_scaled, K_scaled, n_val) in zip(hill_terms * dup, hill_inc_all, Ks_inc, ns_inc):
            n_f = float(n_val)
            reg_species = term_tuple[0]
            multiplier_power = 1 if len(term_tuple) > 1 else 0
            mult_species = term_tuple[1] if multiplier_power else None

            s_reg = s_u if reg_species == 0 else s_v
            s_mult = 1.0
            if multiplier_power:
                s_mult = s_u if mult_species == 0 else s_v

            b_orig = coeff_scaled / ((s_reg ** n_f) * (s_mult ** multiplier_power) + 0.0)
            # K unscaling will be handled separately in Ks_inc_unscaled
            hill_inc_unscaled.append(float(b_orig))
        hill_inc_unscaled = np.array(hill_inc_unscaled)

        hill_dec_unscaled = []
        for (term_tuple, coeff_scaled, K_scaled, n_val) in zip(hill_terms * dup, hill_dec_all, Ks_dec, ns_dec):
            n_f = float(n_val)
            reg_species = term_tuple[0]
            multiplier_power = 1 if len(term_tuple) > 1 else 0
            mult_species = term_tuple[1] if multiplier_power else None

            s_reg = s_u if reg_species == 0 else s_v
            s_mult = 1.0
            if multiplier_power:
                s_mult = s_u if mult_species == 0 else s_v

            b_orig = coeff_scaled / ((s_reg ** n_f) * (s_mult ** multiplier_power) + 0.0)
            hill_dec_unscaled.append(float(b_orig))
        hill_dec_unscaled = np.array(hill_dec_unscaled)

        # Ks unscaled: K_orig = K_scaled / (s_reg ** n)
        Ks_inc_unscaled = []
        for (term_tuple, K_scaled, n_val) in zip(hill_terms * dup, Ks_inc, ns_inc):
            n_f = float(n_val)
            reg_species = term_tuple[0]
            s_reg = s_u if reg_species == 0 else s_v
            K_orig = float(K_scaled / (s_reg ** n_f + 0.0))
            Ks_inc_unscaled.append(K_orig)
        Ks_inc_unscaled = np.array(Ks_inc_unscaled)

        Ks_dec_unscaled = []
        for (term_tuple, K_scaled, n_val) in zip(hill_terms * dup, Ks_dec, ns_dec):
            n_f = float(n_val)
            reg_species = term_tuple[0]
            s_reg = s_u if reg_species == 0 else s_v
            K_orig = float(K_scaled / (s_reg ** n_f + 0.0))
            Ks_dec_unscaled.append(K_orig)
        Ks_dec_unscaled = np.array(Ks_dec_unscaled)

        # Build raw_w_unscaled: same length as effective. For poly entries use poly_coeffs_unscaled,
        # for hill entries use concatenation of inc then dec arrays
        raw_w_unscaled_list = []
        # poly part
        if poly_coeffs_unscaled.size:
            raw_w_unscaled_list.extend(poly_coeffs_unscaled.tolist())
        # hill part: concatenate inc then dec in the same ordering as effective's hill block
        if hill_inc_all.size:
            # hill_block order was dup rows [inc_block | dec_block], flattened to dup*n_hill_single ordering.
            # hill_inc_unscaled and hill_dec_unscaled are already in exactly that duplicate-major ordering.
            raw_w_unscaled_list.extend(hill_inc_unscaled.tolist())
            raw_w_unscaled_list.extend(hill_dec_unscaled.tolist())

        raw_w_unscaled = np.array(raw_w_unscaled_list) if len(raw_w_unscaled_list) else np.array([])
        effective_unscaled = (raw_w_unscaled * gates)
        
        # Non-full (quick) return: minimal keys requested
        if not full:
            return {
                'raw_w_unscaled': raw_w_unscaled,
                'effective_unscaled': effective_unscaled
            }

        # Full return (everything)
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