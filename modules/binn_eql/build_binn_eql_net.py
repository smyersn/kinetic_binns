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
class D_PARAMS(nn.Module):
    def __init__(self, input_features=2, base_val=0.1, noise_std=0.5):
        super().__init__()
        base_log = torch.log(torch.tensor(base_val))         
        noise = torch.randn(input_features) * noise_std
        self.raw_D = nn.Parameter(base_log + noise)
        
    def forward(self):     
        return torch.exp(self.raw_D)
        # return torch.clamp(torch.exp(self.raw_D), min=1e-2, max=10.0)
    
class FourierFeatureEncoding(nn.Module):
    def __init__(self, in_features, mapping_size, scale=1.0):
        super().__init__()
        # Create a static, non-trainable random Gaussian matrix
        self.B = nn.Parameter(torch.randn(in_features, mapping_size) * scale, requires_grad=True)
        
    def forward(self, x):
        # Project inputs into high frequencies
        x_proj = (2.0 * np.pi * x) @ self.B
        # Return both sine and cosine projections
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

# class uv_MLP(nn.Module):
#     def __init__(self, input_features, mapping_size=64, scale=1.0, layers=[256, 256, 256, 2]):
#         super().__init__()
        
#         self.encoder = FourierFeatureEncoding(in_features=input_features, mapping_size=mapping_size, scale=scale)
        
#         # ADD the raw input features to the encoded feature count
#         encoded_features = (mapping_size * 2) + input_features 
        
#         self.mlp = build_mlp(
#             input_features=encoded_features, 
#             layers=layers,
#             activation=nn.GELU(),
#             linear_output=False,
#             output_activation=nn.Softplus(beta=10.0) # See Point 2 below!
#         )

#         # Apply Weight Norm "Post-Hoc"
#         for module in self.mlp.MLP:
#             if isinstance(module, nn.Linear):
#                 utils.parametrizations.weight_norm(module)

#     def forward(self, inputs):
#         encoded_x = self.encoder(inputs)
#         # Concatenate the raw inputs with the high-frequency features
#         full_features = torch.cat([inputs, encoded_x], dim=-1) 
#         return self.mlp(full_features)
    
class uv_MLP(nn.Module):
    def __init__(self, input_features, mapping_size=256, scale=1.0, layers=[256, 256, 256, 2]):
    # def __init__(self, input_features, mapping_size=128, scale=10.0, layers=[256, 256, 256, 2]):
        super().__init__()
        
        # 1. Initialize the Fourier Encoder
        self.encoder = FourierFeatureEncoding(
            in_features=input_features, 
            mapping_size=mapping_size, 
            scale=scale
        )
        
        # The encoder outputs BOTH sin and cos for each mapping dimension
        encoded_features = mapping_size * 2
        
        # 2. Build the MLP, passing the NEW encoded feature size
        self.mlp = build_mlp(
            input_features=encoded_features, # <-- CRITICAL CHANGE
            layers=layers,
            activation=nn.GELU(),
            linear_output=False,
            output_activation=nn.Softplus()
        )

        # 3. Apply Weight Norm "Post-Hoc"
        for module in self.mlp.MLP:
            if isinstance(module, nn.Linear):
                utils.parametrizations.weight_norm(module)
                
    def forward(self, inputs):
        # Pass raw inputs through the encoder first
        encoded_x = self.encoder(inputs)
        # Pass the high-frequency features into the MLP
        return self.mlp(encoded_x)
     
# class uv_MLP(nn.Module):
#     def __init__(self, input_features, layers=[256, 256, 256, 2]):
#         super().__init__()
#         # 1. Pass GELU directly (build_mlp accepts an activation arg)
#         self.mlp = build_mlp(
#             input_features=input_features, 
#             layers=layers,
#             activation=nn.GELU(),
#             linear_output=False,
#             output_activation=nn.Softplus())

#         # 2. Apply Weight Norm "Post-Hoc"
#         for module in self.mlp.MLP:
#             if isinstance(module, nn.Linear):
#                 utils.parametrizations.weight_norm(module)
                
#     def forward(self, inputs):
#         return self.mlp(inputs)

class F_EQL(nn.Module):
    def __init__(self, species, duplicates, param_bounds, max_scale, degree):
        super(F_EQL, self).__init__()
        # Pass max_scale down to EQLLayer for physical conversion
        self.eql_layer = EQLLayer(species, duplicates, param_bounds, max_scale, degree)

    def forward(self, x):
        return self.eql_layer(x)
            
# ---------------------------------------------------------
# 2. BINN
# ---------------------------------------------------------
class BINN(nn.Module):
    def __init__(self, dimensions, species, train_data, duplicates=1,
                diff_coeffs=None, uv_layers=None, degree=2, param_bounds=10,
                fourier_scale=1.0, fourier_mapping_size=64): 
 
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
            self.surface_fitter = uv_MLP(input_features=dimensions+1, layers=uv_layers,
                                          scale=fourier_scale, mapping_size=fourier_mapping_size)
        else:
            self.surface_fitter = uv_MLP(input_features=dimensions+1,
                                          scale=fourier_scale, mapping_size=fourier_mapping_size)
        
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
        # Cache inputs for GLS loss 
        self.inputs = inputs  
        inputs_hat = self.normalize(inputs)
        return self.surface_fitter(inputs_hat)

    # -----------------------
    # Loss Functions
    # -----------------------
    @torch.no_grad()
    def _pde_scale_forward_pass(self, chunk):
        """Helper: not used directly (kept for clarity of what needs grad)."""
        pass

    def register_pde_scale(self, train_data, quantile_percent=0.01, chunk_size=50_000):
        """
        Locks robust PDE normalization scales (pde_scale_u, pde_scale_v),
        analogous to register_curvature_scale. Computes the top-quantile
        mean of u_t^2 / v_t^2 from the CURRENT surface_fitter, so this must
        be called once the surface has been fit (start of Phase 2), not
        at __init__ time when the surface is still random.
        """
        if getattr(self, 'pde_scales_locked', False):
            return  # already locked, don't recompute mid-training

        # print("\n--- Calculating Robust PDE Scales (Top {}%) ---".format(quantile_percent * 100))
        print("\n--- Calculating Robust PDE Scales (90th percentile) ---")

        was_training = self.surface_fitter.training
        self.surface_fitter.eval()

        all_u_sq = []
        all_v_sq = []
        total_points = len(train_data)

        for chunk_start in range(0, total_points, chunk_size):
            chunk = train_data[chunk_start:chunk_start + chunk_size].clone().requires_grad_(True)
            outputs = self.surface_fitter(self.normalize(chunk[:, :self.dimensions + 1]))

            u_t = gradient(outputs[:, 0], chunk, order=1)[:, self.dimensions]
            v_t = gradient(outputs[:, 1], chunk, order=1)[:, self.dimensions]

            all_u_sq.append((u_t ** 2).detach().cpu())
            all_v_sq.append((v_t ** 2).detach().cpu())

            del chunk, outputs, u_t, v_t

        if was_training:
            self.surface_fitter.train()

        global_u_sq = torch.cat(all_u_sq)
        global_v_sq = torch.cat(all_v_sq)

        robust_max_u = torch.quantile(global_u_sq, 0.90).item()
        robust_max_v = torch.quantile(global_v_sq, 0.90).item()
        
        self.pde_scale_u = robust_max_u + 1e-6
        self.pde_scale_v = robust_max_v + 1e-6
        self.pde_scales_locked = True 
        
        print(f"Locked 90th Percentile Scales -> u: {self.pde_scale_u:.4e}, v: {self.pde_scale_v:.4e}\n")

    def register_mass_scale(self, train_data, t_cutoff=None, chunk_size=50_000):
        """
        Locks a robust normalization scale for mass_loss, analogous to
        register_pde_scale. Computes the 90th-percentile w_t^2 = (u_t+v_t)^2
        from the CURRENT surface_fitter, restricted to t >= mass_t_cutoff (the
        same region mass_loss samples from). Must be called once the surface
        has converged (start of Phase 2), same timing as register_pde_scale.
        """
        if getattr(self, 'mass_scale_locked', False):
            return

        t_cutoff = t_cutoff if t_cutoff is not None else getattr(self, 'mass_t_cutoff', 2.0)
        print(f"\n--- Calculating Robust Mass Scale (90th percentile, t>={t_cutoff}) ---")

        was_training = self.surface_fitter.training
        self.surface_fitter.eval()

        mask = train_data[:, self.dimensions] >= t_cutoff
        subset = train_data[mask]

        all_wt_sq = []
        for chunk_start in range(0, len(subset), chunk_size):
            chunk = subset[chunk_start:chunk_start + chunk_size].clone().requires_grad_(True)
            outputs = self.surface_fitter(self.normalize(chunk[:, :self.dimensions + 1]))

            u_t = gradient(outputs[:, 0], chunk, order=1)[:, self.dimensions]
            v_t = gradient(outputs[:, 1], chunk, order=1)[:, self.dimensions]
            w_t = u_t + v_t

            all_wt_sq.append((w_t ** 2).detach().cpu())
            del chunk, outputs, u_t, v_t, w_t

        if was_training:
            self.surface_fitter.train()

        global_wt_sq = torch.cat(all_wt_sq)
        robust_max_wt = torch.quantile(global_wt_sq, 0.90).item()

        self.mass_scale = robust_max_wt + 1e-8
        self.mass_scale_locked = True

        print(f"Locked 90th Percentile Mass Scale -> {self.mass_scale:.4e}\n")

    def register_l0_scale(self, pde_history, window_frac=0.2, phase_1_end=None, phase_2_end=None):
        """
        Locks a data-derived scale for the L0 weight, analogous to
        register_pde_scale / register_mass_scale. Gate decisions compare
        d(pde)/d(gate_i) against l0_weight, so the effective pruning pressure
        depends on the magnitude of the PDE loss -- a dataset whose PDE loss
        plateaus 7x higher gets ~7x weaker pruning from the same l0_weight.
        Normalizing by the achievable Phase-2 floor (all terms active, no
        regularization yet) makes l0_weight dimensionless and comparable
        across datasets.
        """
        if getattr(self, 'l0_scale_locked', False):
            return

        recent = [p for p in pde_history if p > 0]
        if not recent:
            self.l0_scale = 1.0
        else:
            tail = recent[-max(1, int(window_frac * len(recent))):]
            self.l0_scale = float(np.median(tail))

        self.l0_scale_locked = True
        print(f"\n--- Locked L0 scale (median Phase-2 PDE floor) -> {self.l0_scale:.4e} ---\n")

    def refresh_collocation_cache(self, cache_size=200_000, mass_t_cutoff=None, chunk_size=20_000):
        """
        Precomputes ut_array/uxx_array for a large fixed pool of collocation
        points using the CURRENT surface_fitter, then detaches. Chunked because
        the double-backward derivative graph (create_graph=True, twice) is much
        more memory-hungry per point than a plain forward pass -- building it for
        cache_size points in one shot OOMs even on an H200, let alone a shared one.
        """
        t_cutoff = mass_t_cutoff if mass_t_cutoff is not None else getattr(self, 'mass_t_cutoff', 2.0)

        was_training = self.surface_fitter.training
        self.surface_fitter.eval()

        outputs_list, ut_list, uxx_list, mask_list = [], [], [], []

        for chunk_start in range(0, cache_size, chunk_size):
            n = min(chunk_size, cache_size - chunk_start)
            x = torch.empty(n, self.dimensions, device=self.lb.device).uniform_(
                self.lb[0, 0].item(), self.ub[0, 0].item())
            t = torch.empty(n, 1, device=self.lb.device).uniform_(
                self.lb[0, -1].item(), self.ub[0, -1].item())
            inputs = torch.cat([x, t], dim=1).requires_grad_(True)
            outputs = self.surface_fitter(self.normalize(inputs))
            ut_array, uxx_array = self.compute_field_derivatives(inputs, outputs)

            outputs_list.append(outputs.detach())
            ut_list.append(ut_array.detach())
            uxx_list.append(uxx_array.detach())
            mask_list.append((inputs[:, -1] >= t_cutoff).detach())

            del x, t, inputs, outputs, ut_array, uxx_array
            torch.cuda.empty_cache()

        if was_training:
            self.surface_fitter.train()

        self._collocation_cache = {
            'outputs': torch.cat(outputs_list, dim=0),
            'ut_array': torch.cat(ut_list, dim=0),
            'uxx_array': torch.cat(uxx_list, dim=1),   # species dim is axis 0, points is axis 1
            'mass_mask': torch.cat(mask_list, dim=0),
        }
        print(f"Refreshed collocation cache: {cache_size} points ({chunk_size}/chunk), "
            f"{self._collocation_cache['mass_mask'].sum().item()} pass mass cutoff")
    
    def compute_field_derivatives(self, inputs, outputs):
        points = len(inputs)
        uxx_array = torch.zeros((self.species, points, self.dimensions), device=inputs.device)
        ut_array = torch.zeros((points, self.species), device=inputs.device)
        for i in range(self.species):
            d1 = gradient(outputs[:, i], inputs, order=1)
            ut_array[:, i] = d1[:, -1]
            for j in range(self.dimensions):
                uxx_array[i, :, j] = gradient(d1[:, j], inputs, order=1)[:, j]
        return ut_array, uxx_array
  
    # def gls_loss(self, pred, true):
    #     residual = ((pred - true) / self.mean_scale)**2
    #     return torch.mean(residual)

    def gls_loss(self, pred, true):
        # Calculate raw squared residuals
        residual = ((pred - true) / self.mean_scale)**2

        # Create mask for inputs at t = 0
        ic_mask = self.inputs[:, -1:] == 0
                
        # Weight the residuals before taking the mean
        weights = torch.where(ic_mask, 10.0, 1.0)
        weighted_residual = residual * weights
        
        return torch.mean(weighted_residual)    

    def gls_loss_time_weighted(self, pred, true, time_scale=5.0, max_weight=50.0):
        """
        Generalizes the existing ic_mask weighting (which only boosts
        exact t=0 rows) into a smooth decay over early time. The t=0.5
        transient carries the hardest curvature in the trajectory AND
        is one of the rarest timesteps in the dataset under uniform
        sampling - this compensates on the loss side without touching
        how batches are sampled.

        weight(t) = 1 + (max_weight - 1) * exp(-(t - t_min) / time_scale)
        At t = t_min: weight = max_weight
        As t grows:   weight decays smoothly back to 1
        """
        residual = ((pred - true) / self.mean_scale) ** 2

        t = self.inputs[:, -1:]
        t_min = self.lb[0, -1]
        weights = 1.0 + (max_weight - 1.0) * torch.exp(-(t - t_min) / time_scale)

        weighted_residual = residual * weights
        return torch.mean(weighted_residual)

    def pde_loss_from_derivatives(self, outputs, ut_array, uxx_array, epoch):
        u = outputs
        F = self.reaction(u)

        if self.diff_coeffs:
            Du, Dv = torch.tensor(self.diff_coeffs[0]), torch.tensor(self.diff_coeffs[1])
        else:
            D = self.diffusion_fitter()
            Du, Dv = D[0], D[1]

        lap_u = Du * torch.sum(uxx_array[0, :, :], dim=1, keepdim=True)
        lap_v = Dv * torch.sum(uxx_array[1, :, :], dim=1, keepdim=True)

        LHS_u, RHS_u = ut_array[:, 0][:, None], lap_u + F
        LHS_v, RHS_v = ut_array[:, 1][:, None], lap_v - F

        scale_u = torch.sqrt(torch.tensor(getattr(self, 'pde_scale_u', 1.0), device=outputs.device))
        scale_v = torch.sqrt(torch.tensor(getattr(self, 'pde_scale_v', 1.0), device=outputs.device))

        res_u = (LHS_u - RHS_u) / scale_u
        res_v = (LHS_v - RHS_v) / scale_v
        target_zero = torch.zeros_like(res_u)

        pde_loss_u = torch.nn.functional.smooth_l1_loss(res_u, target_zero, beta=1.0)
        pde_loss_v = torch.nn.functional.smooth_l1_loss(res_v, target_zero, beta=1.0)

        return torch.mean(pde_loss_u + pde_loss_v)
    
    def mass_loss_from_derivatives(self, ut_array, uxx_array, mask):
        if self.diff_coeffs:
            return torch.tensor(0.0, device=ut_array.device)
        D = self.diffusion_fitter()
        Du, Dv = D[0], D[1]

        u_t = ut_array[mask, 0]
        v_t = ut_array[mask, 1]
        w_t = u_t + v_t

        lap_u = torch.sum(uxx_array[0, mask, :], dim=1)
        lap_v = torch.sum(uxx_array[1, mask, :], dim=1)

        scale = torch.sqrt(torch.tensor(getattr(self, 'mass_scale', 1.0), device=ut_array.device))
        residual = (w_t - (Du * lap_u + Dv * lap_v)) / scale

        weights = torch.abs(w_t).detach()
        weights = weights / (weights.mean() + 1e-8)
        return torch.mean(weights * residual ** 2)
    
    def reg_loss(self, epoch):
        """
        Soft Wall Regularization:
        1. L0 Sparsity
        2. Physical Bound Penalty (ReLU(|w_phys| - bound))
        """
        # 1. L0 Sparsity
        gate_probs = self.reaction.eql_layer.l0_gate.expected_l0()

        return gate_probs.sum()
    
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

        return w_loss + k_loss

    def loss(self, pred, true, epoch, phase=None):
        raw_gls = self.gls_loss_time_weighted(pred, true)

        raw_softwall = self.soft_wall_loss()   # always enforced, cheap regardless of phase
        raw_l0 = self.reg_loss(epoch)          # cheap, gate probs only

        if phase == 1:
            # Surface-only phase: reaction/diffusion are frozen (requires_grad=False),
            # so pde_loss/mass_loss contribute zero gradient here regardless -- but
            # without this check we'd still pay for collocation sampling + a full
            # double-backward derivative pass through surface_fitter just to throw
            # the result away via base_weights=0. Skip the computation entirely.
            zero = torch.tensor(0.0, device=pred.device)
            return raw_gls, zero, raw_l0, raw_softwall, zero

        cache = getattr(self, '_collocation_cache', None)
        if cache is not None:
            idx = torch.randint(0, cache['outputs'].shape[0], (self.num_samples,), device=pred.device)
            outputs_b = cache['outputs'][idx]
            ut_b = cache['ut_array'][idx]
            uxx_b = cache['uxx_array'][:, idx, :]      # FIX: points is axis 1 here, not axis 0
            mask_b = cache['mass_mask'][idx]
        
        else:
            # phase > 1 but cache not built yet -- shouldn't normally happen once
            # refresh_collocation_cache() is wired into Phase 2 entry, but fall
            # back to the uncached path rather than crashing.
            x = torch.empty(self.num_samples, self.dimensions, device=pred.device).uniform_(
                self.lb[0, 0], self.ub[0, 0])
            t = torch.empty(self.num_samples, 1, device=pred.device).uniform_(
                self.lb[0, -1], self.ub[0, -1])
            inputs_rand = torch.cat([x, t], dim=1).requires_grad_()
            outputs_b = self.surface_fitter(self.normalize(inputs_rand))
            ut_b, uxx_b = self.compute_field_derivatives(inputs_rand, outputs_b)
            t_cutoff = getattr(self, 'mass_t_cutoff', 2.0)
            mask_b = inputs_rand[:, -1] >= t_cutoff

        raw_pde = self.pde_loss_from_derivatives(outputs_b, ut_b, uxx_b, epoch)
        raw_mass = (self.mass_loss_from_derivatives(ut_b, uxx_b, mask_b)
                    if mask_b.any() else torch.tensor(0.0, device=pred.device))

        return raw_gls, raw_pde, raw_l0, raw_softwall, raw_mass
        
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
    def fine_tune_eql(self, threshold=0.01, epsilon=0.1):
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
        s_u, s_v = self.max_scale[0, 0].item(), self.max_scale[0, 1].item()
        u_space = torch.linspace(0, s_u, steps, device=device)
        v_space = torch.linspace(0, s_v, steps, device=device)
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
            weight_primary = eql.fc.weight.data[0, h_idx]
            if torch.abs(weight_primary) < 1e-8: continue
            
            # 1. Identify Form: 0 for Inc, 1 for Dec (for example)
            form_id_i = (h_idx - num_poly) % n_hill_single
            
            f_hill = features[:, h_idx]
            # Center for Pearson Correlation
            f_hill_c = f_hill - torch.mean(f_hill)
            norm_hill_c = torch.norm(f_hill_c) + 1e-9
            
            for next_h_idx in range(h_idx + 1, num_poly + num_hill):
                weight_duplicate = eql.fc.weight.data[0, next_h_idx]
                if torch.abs(weight_duplicate) < 1e-8: continue
                
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
                    
                    # Merge internal parameters using a weighted average!
                    self._average_hill_params(h_idx - num_poly, next_h_idx - num_poly, 
                                              weight_primary, weight_duplicate)
                    
                    # Consolidate the linear coefficient weights
                    eql.fc.weight.data[0, h_idx] += eql.fc.weight.data[0, next_h_idx]
                    eql.fc.weight.data[0, next_h_idx] = 0.0
                    
                    # Keep the strongest gate open
                    eql.l0_gate.log_alpha.data[h_idx] = torch.max(
                        eql.l0_gate.log_alpha.data[h_idx], 
                        eql.l0_gate.log_alpha.data[next_h_idx]
                    )
                    eql.l0_gate.log_alpha.data[next_h_idx] = -10.0
                    
                    # Update primary weight for any subsequent merges in the loop
                    weight_primary = eql.fc.weight.data[0, h_idx]

        # --- TASK 3B: HEURISTIC SIMPLIFICATION (Strict Rule-Based Mapping) ---
        for i in range(num_hill):
            h_idx = num_poly + i
            hill_weight = eql.fc.weight.data[0, h_idx].item()
            if abs(hill_weight) < 1e-8: continue

            # Grab the specific HillFunction object
            hf = eql.all_hill_funcs[i]
            
            # 1. Extract physical k and n
            n_val = (torch.sigmoid(hf.raw_n) * 3 + 1).item()
            k_val = F.softplus(hf.raw_K).item()
            
            # 2. Calculate heuristics
            n_rounded = round(n_val)
            max_u = s_u
            max_denom = 1.0 + k_val * (max_u ** n_val)
            
            # 3. YOUR LOGIC GATES
            is_flat_denom = max_denom < 2.0
            is_integer_exp = abs(n_val - n_rounded) < epsilon
            
            if is_flat_denom and is_integer_exp:               
                # --- FIND THE CORRESPONDING POLYNOMIAL ---                
                backup_K = hf.raw_K.data.clone()
                backup_n = hf.raw_n.data.clone()
                
                # Force k = 0
                hf.raw_K.data.fill_(-20.0) 
                
                # Force n = n_rounded
                target_sigmoid = max(min((n_rounded - 1.0) / 3.0, 0.999), 0.001) 
                hf.raw_n.data.fill_(torch.logit(torch.tensor(target_sigmoid)).item())
                
                # Generate the perfect polynomial shape
                perfect_shape = eql.get_features(uv_synthetic)[:, h_idx]
                perfect_norm = torch.norm(perfect_shape) + 1e-9
                
                # Search the polynomial basis for the exact match
                best_p_idx = -1
                best_corr = -1.0
                
                for p_idx in range(num_poly):
                    f_poly = features[:, p_idx]
                    poly_norm = torch.norm(f_poly) + 1e-9
                    
                    # Cosine similarity (1.0 = identical shape)
                    corr = torch.sum(perfect_shape * f_poly) / (perfect_norm * poly_norm)
                    
                    if corr > best_corr:
                        best_corr = corr.item()
                        best_p_idx = p_idx
                
                # 4. EXECUTE THE TRANSFER
                # If we found a perfect structural match (correlation > 0.99)
                if best_corr > 0.99:
                    print(f"Moving weight {hill_weight:.4f} directly to Poly {best_p_idx}, max_denom={max_denom:.2f}, n={n_val:.3f}")
                    
                    # Move the exact coefficient
                    eql.fc.weight.data[0, best_p_idx] += hill_weight
                    
                    # Transfer Gate L0 Importance
                    eql.l0_gate.log_alpha.data[best_p_idx] = torch.max(
                        eql.l0_gate.log_alpha.data[best_p_idx],
                        eql.l0_gate.log_alpha.data[h_idx]
                    )
                    
                    # Kill the Hill term
                    eql.fc.weight.data[0, h_idx] = 0.0
                    eql.l0_gate.log_alpha.data[h_idx] = -10.0
                    
                else:
                    # Failsafe: The required polynomial doesn't exist in your basis 
                    # (e.g., Hill became u^3, but poly basis stops at degree 2).
                    print(f"Target polynomial not in basis. Restoring Hill term.")
                    hf.raw_K.data = backup_K
                    hf.raw_n.data = backup_n

        # --- TASK 4: FINAL ZEROING (Pruning Noise) ---
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

        _ = self.extract_params(full=True)
        print(f"Fine-tuning committed.")

    def _average_hill_params(self, idx1, idx2, weight1, weight2):
        """
        Helper to average n and raw_K for two Hill modules using a weighted average
        based on the magnitude of their linear coefficients.
        """
        eql = self.reaction.eql_layer
        
        # Construct a flat list of Hill modules following Sequential Ptr logic
        all_hf = []
        for hm in eql.hill.hill_modules:
            all_hf.extend(hm.hill_inc_raw + list(hm.hill_inc_cross.values()))
            all_hf.extend(hm.hill_dec_raw + list(hm.hill_dec_cross.values()))
            
        hf1 = all_hf[idx1]
        hf2 = all_hf[idx2]
        
        # Calculate the proportional weight of each term
        abs_w1 = torch.abs(weight1)
        abs_w2 = torch.abs(weight2)
        total_w = abs_w1 + abs_w2
        
        # Prevent division by zero (though handled by the 1e-8 check in the main loop)
        if total_w < 1e-8:
            prop1, prop2 = 0.5, 0.5
        else:
            prop1 = abs_w1 / total_w
            prop2 = abs_w2 / total_w
        
        with torch.no_grad():
            # Update primary module with WEIGHTED average parameters
            hf1.raw_n.data = (hf1.raw_n.data * prop1) + (hf2.raw_n.data * prop2)
            hf1.raw_K.data = (hf1.raw_K.data * prop1) + (hf2.raw_K.data * prop2)
            
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