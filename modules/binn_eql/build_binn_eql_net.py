import torch, time
import torch.nn as nn
import torch.nn.functional as F

from modules.binn.build_mlp import build_mlp
from modules.utils.gradient import gradient
from modules.utils.triangle import lltriangle
from modules.utils.numpy_torch_conversion import *
from modules.activations.softplus_relu import softplus_relu
from modules.symbolic_net.custom_norm import custom_norm
from modules.binn_eql.build_eql_layer import EQLLayer


class D_PARAMS(nn.Module):
    
    '''
    Construct MLP surrogate model for the unknown diffusivity function. 
    Includes three hidden layers with 32 sigmoid-activated neurons. Output
    is softplus-activated to keep predicted diffusivities non-negative.
    
    Inputs:
        input_features (int): number of input features
        scale        (float): input scaling factor
    
    Args:
        u (torch tensor): predicted u values with shape (N, 1)
        t (torch tensor): optional time values with shape (N, 1)
        
    Returns:
        D (torch tensor): predicted diffusivities with shape (N, 1)
    '''
    
    def __init__(self, input_features=2, param_bounds=10):
        
        super().__init__()
        self.input_features = input_features
        self.param_bounds = param_bounds

        self.raw_D = nn.Parameter(torch.empty(input_features).uniform_(-4, 4))
        
    def forward(self):     
        # D = self.activation(self.params) * self.param_bounds
        D = torch.sigmoid(self.raw_D) * self.param_bounds
        return D

class uv_MLP(nn.Module):
    
    '''
    Construct MLP surrogate model for the solution of the governing PDE. 
    Includes three hidden layers with 128 sigmoid-activated neurons. Output
    is softplus-activated to keep predicted species concentrations non-negative.
    
    Inputs:
        scale (float): output scaling factor, defaults to carrying capacity
    
    Args:
        inputs (torch tensor): x and t pairs with shape (N, 2)
        
    Returns:
        outputs (torch tensor): predicted u and v values with shape (N, 2)
    '''
    
    def __init__(self, input_features, layers=[128, 128, 128, 2]):
    # def __init__(self, input_features, layers=[256, 256, 256, 256, 2]):
        super().__init__()
        self.mlp = build_mlp(
            input_features=input_features, 
            layers=layers,
            activation=nn.Sigmoid(), 
            linear_output=False,
            output_activation=softplus_relu())
        # super().__init__()
        # self.mlp = build_mlp(
        #     input_features=input_features, 
        #     layers=layers,
        #     activation=nn.Tanh(), 
        #     linear_output=False,
        #     output_activation=nn.Softplus())
    
    def forward(self, inputs):
        outputs = self.mlp(inputs)
        return outputs

class F_EQL(nn.Module):
    def __init__(self, species, duplicates, param_bounds):
        super(F_EQL, self).__init__()
        self.eql_layer = EQLLayer(species, duplicates, param_bounds)
        self.min = -param_bounds
        self.max = param_bounds

    def forward(self, x):
        return self.eql_layer(x)
    
class BINN(nn.Module):
    
    '''
    Constructs a biologically-informed neural network (BINN) composed of
    cell density dependent diffusion and growth MLPs with an optional time 
    delay MLP.
    
    Inputs:
        delay (bool): whether to include time delay MLP
        
    
    '''
    
    def __init__(self, dimensions, species, train_data, duplicates=1,
                 diff_coeffs=None, uv_layers=None, degree=2, gls_weight=1, 
                 pde_weight=1, l05_weight=0.01, param_bounds=10, warm_up=0):
        
        super().__init__()
        self.dimensions = dimensions        
        self.species = species
        self.train_data = train_data
        self.duplicates = duplicates
        self.diff_coeffs = diff_coeffs
        self.degree = degree
        self.param_bounds = param_bounds
        self.warm_up = warm_up
        
        # diffusion fitter
        if not self.diff_coeffs:
            self.diffusion_fitter = D_PARAMS(self.species, param_bounds)
            
            # diffusion extrema
            self.D_min = 0
            self.D_max = self.diffusion_fitter.param_bounds
            
            # loss weight
            self.D_weight = 1e10 / self.D_max
                
        # surface fitter
        if uv_layers:
            self.surface_fitter = uv_MLP(input_features=dimensions+1, layers=uv_layers)
        else:
            self.surface_fitter = uv_MLP(input_features=dimensions+1)
        
        # reaction
        self.reaction = F_EQL(species, duplicates, self.param_bounds)
        
        # reaction extrema
        self.coeff_min = self.reaction.min
        self.coeff_max = self.reaction.max
        
        # input extrema
        self.x_min = float(torch.min(train_data[:, :self.dimensions]).item())
        self.x_max = float(torch.max(train_data[:, :self.dimensions]).item())
        self.t_min = float(torch.min(train_data[:, self.dimensions]).item())
        self.t_max = float(torch.max(train_data[:, self.dimensions]).item())
            
        # loss weights
        # self.gls_weight = 1e0
        # self.pde_weight = 1e0
        self.IC_weight = 1e1
        self.gls_weight = gls_weight
        self.pde_weight = pde_weight
        self.l05_weight = l05_weight
        self.param_weight = 1e10 / self.param_bounds
        
        # proportionality constant
        self.gamma = 0.2

        # number of samples for pde loss
        self.num_samples = 10000
        
        # model name
        self.name = 'Dumlp_Dvmlp_Fmlp'
    
    def forward(self, inputs):
        # cache input batch for pde loss
        self.inputs = inputs
        return self.surface_fitter(self.inputs)
        
    def gls_loss(self, pred, true):
        denom = true.abs() + 1e-6
        residual = ((pred - true) / denom)**2
        return torch.mean(residual)

    def pde_loss(self, inputs, outputs, epoch):
        # unpack outputs
        u = outputs.clone()
        
        # create arrays to store partial derivatives
        points = len(inputs)
        uxx_array = torch.zeros((self.species, points, self.dimensions)).to(self.inputs.device)
        ut_array = torch.zeros((points, self.species)).to(self.inputs.device)

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
        F = self.reaction(outputs)
        
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
        # Sparsity Regularization
        l05_norm = self.reaction.eql_layer.l0_gate.expected_l0()
                       
        return l05_norm

    def loss(self, pred, true, epoch):
        # load cached inputs from forward pass
        inputs = self.inputs
        
        self.gls_loss_val = self.gls_weight*self.gls_loss(pred, true)
       
        # randomly sample from input domain for PDE loss
        x = torch.empty(self.num_samples, self.dimensions, dtype=torch.float32, device=inputs.device).uniform_(0, 1)
        x = x * (self.x_max - self.x_min) + self.x_min
        t = torch.empty(self.num_samples, 1, dtype=torch.float32, device=inputs.device).uniform_(0, 1)
        t = t * (self.t_max - self.t_min) + self.t_min
        inputs_rand = torch.cat([x, t], dim=1).requires_grad_()
        
        # predict surface fitter at sampled locations
        outputs_rand = self.surface_fitter(inputs_rand)
        
        # compute PDE loss at sampled locations
        self.pde_loss_val = self.pde_weight*self.pde_loss(inputs_rand, outputs_rand, epoch)
        
        # Compute effective l05 weight
        if self.warm_up == 0:
            l0_weight_eff = self.l05_weight
        else:     
            if epoch < self.warm_up:
                l0_weight_eff = 0
            elif epoch < self.warm_up*2:
                l0_weight_eff = ((epoch - self.warm_up) / self.warm_up) * self.l05_weight
            else:
                l0_weight_eff = self.l05_weight
        
        # compute loss from regularization
        l0_loss = self.reg_loss(epoch)
        self.reg_loss_val = l0_weight_eff*l0_loss
        
        if epoch % 1000 == 0:
            print(f'L0 norm, weight, loss: {l0_loss, l0_weight_eff, self.reg_loss_val}')
        
        return (self.gls_loss_val + self.pde_loss_val + self.reg_loss_val), self.gls_loss_val, self.pde_loss_val, self.reg_loss_val

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
 
    def unpack_coeffs(self):       
        # Unpack coefficients for polynomial and Hill terms
        coeffs = self.reaction.eql_layer.fc.weight[0]
        
        # Get gates and calculate effective coefficients
        gates = self.reaction.eql_layer.l0_gate.get_binary_mask()
        eff_coeffs = coeffs * gates
        
        poly_coeffs = eff_coeffs[:self.reaction.eql_layer.num_poly_features]
        hill_coeffs = eff_coeffs[self.reaction.eql_layer.num_poly_features:]
        
        # Generate terms
        poly_terms, hill_terms = self.generate_terms()
        n_hill = len(hill_terms)

        # Extract increasing and decreasing Hill coefficients (alternating)
        hill_coeffs_inc = torch.cat([hill_coeffs[i : i + n_hill] for i in range(0, len(hill_coeffs), 2 * n_hill)])
        hill_coeffs_dec = torch.cat([hill_coeffs[i + n_hill : i + 2 * n_hill] for i in range(0, len(hill_coeffs), 2 * n_hill)])
        
        return poly_coeffs, hill_coeffs_inc, hill_coeffs_dec

    def unpack_hill_params(self):
        ns_inc = []
        ns_dec = []
        Ks_inc = []
        Ks_dec = []

        # Access the hill feature module (works if you use DuplicateHillFeatures or just HillFeatures)
        hill_features_list = self.reaction.eql_layer.hill.hill_modules

        # Iterate over all HillFeatures modules
        for hill_features in hill_features_list:
            for hill_func in hill_features.hill_inc_raw:
                ns_inc.append(torch.sigmoid(hill_func.raw_n).item() * 5)
                Ks_inc.append(torch.sigmoid(hill_func.raw_K).item() * self.param_bounds)
                
            # Process increasing cross hill functions
            for key in list(hill_features.hill_inc_cross.keys()):
                hill_func = hill_features.hill_inc_cross[key]
                ns_inc.append(torch.sigmoid(hill_func.raw_n).item() * 5)
                Ks_inc.append(torch.sigmoid(hill_func.raw_K).item() * self.param_bounds)

            # Process decreasing raw hill functions
            for hill_func in hill_features.hill_dec_raw:
                ns_dec.append(torch.sigmoid(hill_func.raw_n).item() * 5)
                Ks_dec.append(torch.sigmoid(hill_func.raw_K).item() * self.param_bounds)
                
            # Process decreasing cross hill functions
            for key in list(hill_features.hill_dec_cross.keys()):
                hill_func = hill_features.hill_dec_cross[key]
                ns_dec.append(torch.sigmoid(hill_func.raw_n).item() * 5)
                Ks_dec.append(torch.sigmoid(hill_func.raw_K).item() * self.param_bounds)
                
        return ns_inc, ns_dec, Ks_inc, Ks_dec
            
    def remove_insignificant_terms(self, uv, thresh):
        # # Define epsilon to avoid division by zero
        # eps=1e-12
        
        # # compute raw features (N x M)
        # poly_feats = self.reaction.eql_layer.poly(uv)        # shape [N, M_poly]
        # hill_feats = self.reaction.eql_layer.hill(uv)        # shape [N, M_hill]
        # feats = torch.cat([poly_feats, hill_feats], dim=1)   # shape [N, M]

        # # weights vector for the single-output fc (assume out_features==1)
        # # use data (not requiring_grad); choose device automatically
        # weights = self.reaction.eql_layer.fc.weight.detach().view(-1)  # shape [M]

        # # per-feature RMS (scale) across the uv sample
        # feat_rms = torch.sqrt((feats.detach() ** 2).mean(dim=0) + eps)  # shape [M]

        # # absolute per-feature contribution (L2-style): |w_i| * feat_rms_i
        # contrib = weights.abs() * feat_rms  # shape [M]

        # # fractional contribution relative to total contribution
        # total = contrib.sum() + eps
        # frac = contrib / total  # shape [M], sums to ~1

        # # build keep/prune mask: keep features whose fraction >= thresh_frac
        # keep_mask = (frac >= thresh)   # boolean mask shape [M]
        
        # # print(f'weights: {weights}')
        # # print(f'feat rms: {feat_rms}')
        # print(f'contrib: {contrib}')
        # print(f'frac: {frac}')
        # print(f'keep mask: {keep_mask}')

        # # zero-out pruned features (use no_grad)
        # with torch.no_grad():
        #     # if fc has shape [1, M], index accordingly
        #     self.reaction.eql_layer.fc.weight[0, ~keep_mask] = 0.0

        
        
        # removes all terms from individual that have minor impact on surface
        poly_feats = self.reaction.eql_layer.poly(uv)
        hill_feats = self.reaction.eql_layer.hill(uv)
        feats = torch.cat([poly_feats, hill_feats], dim=1)

        weights = self.reaction.eql_layer.fc.weight[0]
        weighted_feats = feats * weights  
        
        surface = weighted_feats.sum(dim=1)  
                
        # Calculate RMSE if any feature is removed
        rmse = torch.sqrt((weighted_feats**2).mean(dim=0))          # [M]
        # print(f'rmse: {rmse}')
        
        # Determine which features are insignificant
        surface_range = torch.mean(torch.abs(surface))              # scalar
        # surface_range = torch.max(surface) - torch.min(surface)   # scalar
        # print(f'surface range: {surface_range}')
        
        # Calculate coefficient of variation
        coeffs = rmse / surface_range                               # [M]
        # print(f'coeffs: {coeffs}')
        
        # build boolean mask of insignificant features
        mask = (coeffs < thresh)
                                
        # zero them out insignificant features
        with torch.no_grad():
            # self.reaction.eql_layer.fc.raw_weight[0][mask] = 0
            self.reaction.eql_layer.fc.weight[0][mask] = 0
            
    def fix_cheating_hill_functions(self, uv, thresh):
        # Symbolic net sometimes "cheats" by approximating polynomial terms with
        # increasing Hill functions. This method corrects for this mistake.
        
        # Unpack coefficients
        poly_coeffs, hill_coeffs_inc, hill_coeffs_dec = self.unpack_coeffs()  
        
        # Unpack Hill params
        ns_inc, ns_dec, Ks_inc, Ks_dec = self.unpack_hill_params()
        
        # Generate terms
        poly_terms, hill_terms = self.generate_terms()

        # Check for cheating increasing Hill functions
        # check if denominator is roughly constant
        for i, term in enumerate(hill_terms*self.duplicates):
            # check non-zero increasing Hill functions
            if hill_coeffs_inc[i] != 0:
                # define parameters
                n = ns_inc[i]
                K = Ks_inc[i]
                
                hill_specie = uv[:, term[0]]
                
                # check if difference between hill function and corresponding
                # polynomial function is insignificant
                hill_surface = hill_specie**n / (1 + K * hill_specie**n)
                poly_surface = hill_specie**n
                rmse = torch.sqrt(((poly_surface - hill_surface)**2).mean(dim=0))
                                
                if rmse < thresh:
                    n = int(torch.round(torch.tensor(n)))
                    
                    # find corresponding polynomial term Hill function is approximating
                    # with cheating
                    hill_idx = len(poly_terms) + i
                    
                    if len(term) == 1:
                        poly_term = (term[0],) * n
                    if len(term) > 1:
                        poly_term = (term[-1],) + (term[0],) * n
                                            
                    if poly_term in poly_terms:    
                        poly_idx = poly_terms.index(poly_term)
                        
                        with torch.no_grad(): 
                            # get poly raw weight from weight      
                            # weight = poly_coeffs[poly_idx] + hill_coeffs_inc[i]           
                            # sig = (weight + self.param_bounds) / (2 * self.param_bounds)
                            # poly_raw_weight = torch.log(sig / (1 - sig))

                            # self.reaction.eql_layer.fc.raw_weight[0][poly_idx] = poly_raw_weight
                            # self.reaction.eql_layer.fc.raw_weight[0][hill_idx] = 0
                            self.reaction.eql_layer.fc.weight[0][poly_idx] = poly_coeffs[poly_idx] + hill_coeffs_inc[i]
                            self.reaction.eql_layer.fc.weight[0][hill_idx] = 0
                            
                    else:
                        break
            
            # Check for cheating decreasing Hill functions        
            if hill_coeffs_dec[i] != 0 and len(term) > 1:
                # define parameters
                n = ns_dec[i]
                K = Ks_dec[i]
                
                hill_specie = uv[:, term[0]]
                poly_specie = uv[:, term[-1]]

                # check if difference between hill function and corresponding
                # polynomial function is insignificant
                hill_surface = (hill_coeffs_dec[i] * poly_specie) * ((1 / K) - hill_specie**n / (1 + K * hill_specie**n))
                poly_surface = hill_coeffs_dec[i] * poly_specie * (1 / K)
                rmse = torch.sqrt(((poly_surface - hill_surface)**2).mean(dim=0))
                
                if rmse < thresh:                    
                    # find corresponding polynomial term Hill function is approximating
                    # with cheating
                    poly_term = (term[-1],)
                                            
                    if poly_term in poly_terms:    
                        poly_idx = poly_terms.index(poly_term)
                        
                        with torch.no_grad():                   
                            # Change poly coefficient to cheating Hill coefficient
                            poly_coeffs[poly_idx] = poly_coeffs[poly_idx] + hill_coeffs_dec[i] / K
                            
                            # Make cheating Hill coefficient 0
                            hill_coeffs_dec[i] = 0
                            
                            # Gather updated coefficients, update model params
                            updated_coeffs = torch.cat((poly_coeffs, hill_coeffs_inc, hill_coeffs_dec))
                            
                            self.reaction.eql_layer.fc.weight[0] = updated_coeffs
                            
                    else:
                        break       
                              
    def prune(self, thresh=1):       
        # Get uv values from training data
        uv = self.train_data[:, -2:]

        # Prune
        self.remove_insignificant_terms(uv, thresh)
        self.fix_cheating_hill_functions(uv, thresh)
        
        # keep_mask = self.reaction.eql_layer.fc.weight != 0
        
        # return keep_mask

    def generate_equation(self):
        # Unpack coefficients
        poly_coeffs, hill_coeffs_inc, hill_coeffs_dec = self.unpack_coeffs()      
        
        # Unpack Hill params
        ns_inc, ns_dec, Ks_inc, Ks_dec = self.unpack_hill_params()
                
        terms = []
        species = ['u', 'v']
        
        # Generate terms
        poly_terms, hill_terms = self.generate_terms()

        for term, coeff in zip(poly_terms*self.duplicates, poly_coeffs):
            if coeff != 0:
                string = f'{coeff:.3f}'
                for ind in term:
                    string += f' * {species[ind]}'
                terms.append(string)
        
        for term, coeff, k, n in zip(hill_terms*self.duplicates, hill_coeffs_inc, Ks_inc, ns_inc):
            if coeff != 0:
                string = f'{coeff:.3f}'
                if len(term) == 1:
                    string += f' * {species[term[0]]}^{n:.3f} / (1 + {k:.3f} * {species[term[0]]}^{n:.3f})'
                else:     
                    string += f' * {species[term[1]]} * {species[term[0]]}^{n:.3f} / (1 + {k:.3f} * {species[term[0]]}^{n:.3f})'
                terms.append(string)
            
        for term, coeff, k, n in zip(hill_terms*self.duplicates, hill_coeffs_dec, Ks_dec, ns_dec):
            if coeff != 0:
                string = f'{coeff:.3f}'
                if len(term) == 1:
                    string += f' * (1 / {k:.3f} - {species[term[0]]}^{n:.3f} / (1 + {k:.3f} * {species[term[0]]}^{n:.3f}))'
                else:     
                    string += f' * {species[term[1]]} * (1 / {k:.3f} - {species[term[0]]}^{n:.3f} / (1 + {k:.3f} * {species[term[0]]}^{n:.3f}))'
                terms.append(string)
        
        return terms