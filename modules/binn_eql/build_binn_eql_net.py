import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.binn.build_mlp import build_mlp
from modules.utils.gradient import gradient
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
    
    def __init__(self, input_features=2):
        
        super().__init__()
        self.input_features = input_features
        self.activation = softplus_relu()
        self.min = 0
        self.max = 10
        self.params = nn.Parameter(torch.rand(self.input_features))
        # self.params = nn.Parameter(torch.tensor([0.01, 1]))
        
    def forward(self):     
        D = self.activation(self.params)
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
        
        super().__init__()
        self.mlp = build_mlp(
            input_features=input_features, 
            layers=layers,
            activation=nn.Sigmoid(), 
            linear_output=False,
            output_activation=softplus_relu())
    
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
    
    def __init__(self, dimensions, species, duplicates=1, data=None, 
                 diff_coeffs=None, degree=2, gls_weight=1, pde_weight=1, 
                 l05_weight=0, l1_weight=0, param_bounds=10):
        
        super().__init__()
        self.dimensions = dimensions        
        self.species = species
        self.duplicates = duplicates
        self.diff_coeffs = diff_coeffs
        self.degree = degree
        self.param_bounds = param_bounds

        # diffusion fitter
        if not self.diff_coeffs:
            self.diffusion_fitter = D_PARAMS(input_features=self.species)
            
            # diffusion extrema
            self.D_min = self.diffusion_fitter.min
            self.D_max = self.diffusion_fitter.max
            
            # loss weight
            self.D_weight = 1e10 / self.D_max
                
        # surface fitter
        self.surface_fitter = uv_MLP(input_features=dimensions+1)
        
        # reaction
        self.reaction = F_EQL(species, duplicates, self.param_bounds)
        
        # reaction extrema
        self.coeff_min = self.reaction.min
        self.coeff_max = self.reaction.max
        
        # input extrema
        if data is not None:
            self.x_min = float(torch.min(data[:, :self.dimensions]).item())
            self.x_max = float(torch.max(data[:, :self.dimensions]).item())
            self.t_min = float(torch.min(data[:, self.dimensions]).item())
            self.t_max = float(torch.max(data[:, self.dimensions]).item())
        else:
            self.x_min = 0
            self.x_max = 10
            self.t_min = 0
            self.t_max = 25
            
        # loss weights
        # self.gls_weight = 1e0
        # self.pde_weight = 1e0
        self.IC_weight = 1e1
        self.gls_weight = gls_weight
        self.pde_weight = pde_weight
        self.l05_weight = l05_weight
        self.l1_weight = l1_weight
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
    
    # def gls_loss(self, pred, true):
    #     residual = (pred - true)**2
    #     residual *= pred.abs().clamp(min=1.0)**(-self.gamma)

    #     return torch.mean(residual)
    
    def gls_loss(self, pred, true):
        
        residual = (pred - true)**2
        
        # # add weight to initial condition
        # residual *= torch.where(self.inputs[:, self.dimensions][:, None]==0, 
        #                         self.IC_weight*torch.ones_like(pred), 
        #                         torch.ones_like(pred))
        
        # # proportional GLS weighting
        # residual *= pred.abs().clamp(min=1.0)**(-self.gamma)
        
        return torch.mean(residual)

    
    def pde_loss(self, inputs, outputs):
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
        
        # print(f'Du, Dv: {Du, Dv}')
        # print(f'xtuv: {torch.concat([inputs, u], dim=1)[:20]}')
        # print(f'LHSU RHSU: {torch.concat([LHS_u, RHS_u], dim=1)[:20]}')
        # print(f'RHSU, LAPU, F: {torch.concat([RHS_u, lap_u, F], dim=1)[:20]}')
        # print(f'LHSV RHSV: {torch.concat([LHS_v, RHS_v], dim=1)[:20]}')
        # print(f'RHSV, LAPV, -F: {torch.concat([RHS_v, lap_v, -F], dim=1)[:20]}')
        # print(f'loss: {pde_loss[:20]}')
        # print(f'mean loss: {torch.mean(pde_loss)}')

        return torch.mean(pde_loss)
        
    def reg_loss(self):
        # constraints on learned parameters
        self.coeff_loss = 0
        self.D_loss = 0
        self.sparsity_loss = 0
        
        # Get coefficients for terms
        coeffs = self.reaction.eql_layer.fc.weight

        self.coeff_loss += torch.mean(self.param_weight * torch.relu(self.coeff_min - coeffs)**2)
        self.coeff_loss += torch.mean(self.param_weight * torch.relu(coeffs - self.coeff_max)**2)
                
        if not self.diff_coeffs:
            D = self.diffusion_fitter()

            self.D_loss += torch.mean(self.D_weight * torch.relu(self.D_min - D)**2)
            self.D_loss += torch.mean(self.D_weight * torch.relu(D - self.D_max)**2)
            
        # Sparsity Regularization
        if self.l05_weight:
            l05_norm = custom_norm(coeffs, 0.01)
            self.sparsity_loss += self.l05_weight * l05_norm
            
        if self.l1_weight:
            l1_norm = torch.norm(coeffs, p=1)
            self.sparsity_loss += self.l1_weight * l1_norm        
        
        return torch.mean(self.coeff_loss + self.D_loss + self.sparsity_loss)

    def loss(self, pred, true):
        self.gls_loss_val = 0
        self.pde_loss_val = 0       
        self.reg_loss_val = 0
        
        # load cached inputs from forward pass
        inputs = self.inputs
     
        self.gls_loss_val = self.gls_weight*self.gls_loss(pred, true)

        # randomly sample from input domain for PDE loss
        x = torch.rand(self.num_samples, self.dimensions, requires_grad=True) 
        x = x*(self.x_max - self.x_min) + self.x_min
        t = torch.rand(self.num_samples, 1, requires_grad=True)
        t = t*(self.t_max - self.t_min) + self.t_min
        inputs_rand = torch.cat([x, t], dim=1).float().to(inputs.device)
        # inputs_rand = torch.cat([x, t], dim=1).float()
        
        # predict surface fitter at sampled locations
        outputs_rand = self.surface_fitter(inputs_rand)

        # compute PDE loss at sampled locations
        self.pde_loss_val += self.pde_weight*self.pde_loss(inputs_rand, outputs_rand)
        
        # compute loss from regularization
        self.reg_loss_val += self.reg_loss()
        
        if torch.isnan((self.gls_loss_val + self.pde_loss_val + self.reg_loss_val)):
            print("NaN in loss")

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
        
        poly_coeffs = coeffs[:self.reaction.eql_layer.num_poly_features]
        
        hill_coeffs = coeffs[self.reaction.eql_layer.num_poly_features:]
        
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
    
    def remove_insignificant_terms(self, uv):
        # removes all terms from individual that have minor impact on total
        # surface shape and magnitude        
        poly_feats = self.reaction.eql_layer.poly(uv)
        hill_feats = self.reaction.eql_layer.hill(uv)
        feats = torch.cat([poly_feats, hill_feats], dim=1)

        weights = self.reaction.eql_layer.fc.weight[0]
        weighted_feats = feats * weights  
        
        surface = weighted_feats.sum(dim=1)      
        
        for i in range(len(weights)):
            if weights[i] != 0:
                surface_wo_feat = surface - weighted_feats[:, i]
                rmse = torch.sqrt(torch.mean((surface - surface_wo_feat)**2))
                # surface_range = torch.max(surface) - torch.min(surface)
                surface_range = torch.mean(torch.abs(surface))
                coeff = rmse / surface_range

                if coeff < 1.5:
                    with torch.no_grad():
                        self.reaction.eql_layer.fc.weight[0][i] = 0
            
    def fix_cheating_hill_functions(self, uv):
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
                
                specie = uv[:, term[0]]
                
                # check that denominator is roughly constant
                denom_vals = 1 + K * specie**n
                
                if denom_vals.std() < 0.25:
                    n = int(torch.round(torch.tensor(n)))
                    
                    # find corresponding polynomial term Hill function is approximating
                    # with cheating
                    if len(term) == 1:
                        poly_term = (term[0],) * n
                    if len(term) > 1:
                        poly_term = (term[-1],) + (term[0],) * n
                                            
                    if poly_term in poly_terms:    
                        poly_idx = poly_terms.index(poly_term)
                        
                        with torch.no_grad():                   
                            # Change poly coefficient to cheating Hill coefficient
                            poly_coeffs[poly_idx] = poly_coeffs[poly_idx] + hill_coeffs_inc[i]
                            
                            # Make cheating Hill coefficient 0
                            hill_coeffs_inc[i] = 0
                            
                            # Gather updated coefficients, update model params
                            updated_coeffs = torch.cat((poly_coeffs, hill_coeffs_inc, hill_coeffs_dec))
                            
                            self.reaction.eql_layer.fc.weight[0] = updated_coeffs
                            
                    else:
                        break
            
            # Check for cheating decreasing Hill functions        
            if hill_coeffs_dec[i] != 0 and len(term) > 1:
                # define parameters
                n = ns_dec[i]
                K = Ks_dec[i]
                
                hill_specie = uv[:, term[0]]
                poly_specie = uv[:, term[-1]]
                                
                # check that dec Hill vals roughly equal to poly vals
                hill_vals = (hill_coeffs_dec[i] * poly_specie) * ((1 / K) - hill_specie**n / (1 + K * hill_specie**n))
                poly_vals = hill_coeffs_dec[i] * poly_specie * (1 / K)
                
                if (hill_vals - poly_vals).std() < 0.25:                    
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