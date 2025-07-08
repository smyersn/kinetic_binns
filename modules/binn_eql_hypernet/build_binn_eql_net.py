import torch, time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from modules.binn.build_mlp import build_mlp
from modules.utils.gradient import gradient
from modules.utils.triangle import lltriangle
from modules.utils.numpy_torch_conversion import *
from modules.activations.softplus_relu import softplus_relu
from modules.symbolic_net.custom_norm import custom_norm
from modules.binn_eql_hypernet.build_eql_layer import EQLLayer
from modules.binn_eql_hypernet.hypernet import Hypernet

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
        self.hypernets = nn.ModuleList(
            [Hypernet(0, 10) for _ in range(input_features)])
        
    def forward(self, k, inference=False):     
        Ds = []

        for hypernet in self.hypernets:
            if inference:
                w = hypernet(inference=True)[0]
            else:
                w = hypernet(inference=False)[0]  # scalar tensor [1]
            Ds.append(w)

        # Stack and reshape to (1, total_features)
        D_tensor = torch.cat(Ds, dim=0).view(1, -1)  # [1, total_features]

        return D_tensor       

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
            activation=nn.Tanh(), 
            linear_output=False,
            output_activation=nn.Softplus())
    
    def forward(self, inputs):
        outputs = self.mlp(inputs)
        
        return outputs
    
    def forward(self, inputs):
        x = inputs.clone()
        
        if torch.isnan(x).any():
            cpu = x.cpu().detach().numpy()
            print(f"NaN at input:")
            print("Min:", np.nanmin(cpu).item(), "Max:", np.nanmax(cpu).item(),
                "Mean:", np.nanmean(cpu).item(), "Std:", np.nanstd(cpu).item())

        for i, layer in enumerate(self.mlp.MLP):
            x = layer(x)
            if torch.isnan(x).any():
                cpu = x.cpu().detach().numpy()
                print(f"NaN at layer {i}:")
                print("Min:", np.nanmin(cpu).item(), "Max:", np.nanmax(cpu).item(),
                    "Mean:", np.nanmean(cpu).item(), "Std:", np.nanstd(cpu).item())
                break
        return x


class F_EQL(nn.Module):
    def __init__(self, species, param_bounds):
        super(F_EQL, self).__init__()
        self.eql_layer = EQLLayer(species, param_bounds)
        self.min = -param_bounds
        self.max = param_bounds

    def forward(self, x, k):
        return self.eql_layer(x, k)
    
class BINN(nn.Module):
    
    '''
    Constructs a biologically-informed neural network (BINN) composed of
    cell density dependent diffusion and growth MLPs with an optional time 
    delay MLP.
    
    Inputs:
        delay (bool): whether to include time delay MLP
        
    
    '''
    
    def __init__(self, dimensions, species, train_data, diff_coeffs=None, 
                 degree=2, gls_weight=1, pde_weight=1, param_bounds=10):
        
        super().__init__()
        self.dimensions = dimensions        
        self.species = species
        self.train_data = train_data
        self.diff_coeffs = diff_coeffs
        self.degree = degree
        self.param_bounds = param_bounds
        
        # diffusion fitter
        if not self.diff_coeffs:
            self.diffusion_fitter = D_PARAMS(input_features=self.species)
                            
        # surface fitter
        self.surface_fitter = uv_MLP(input_features=dimensions+1)
        
        # reaction
        self.reaction = F_EQL(species, self.param_bounds)
        
        # reaction extrema
        self.coeff_min = self.reaction.min
        self.coeff_max = self.reaction.max
        
        # input extrema
        self.x_min = float(torch.min(train_data[:, :self.dimensions]).item())
        self.x_max = float(torch.max(train_data[:, :self.dimensions]).item())
        self.t_min = float(torch.min(train_data[:, self.dimensions]).item())
        self.t_max = float(torch.max(train_data[:, self.dimensions]).item())
            
        # loss weights
        self.IC_weight = 1e1
        self.gls_weight = gls_weight
        self.pde_weight = pde_weight
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
                
    def loss(self, pred, true, k, lambda_):       
        # load cached inputs from forward pass
        inputs = self.inputs
        
        ### CALCULATE GLS LOSS ###
        gls_loss = self.gls_weight * torch.mean((pred - true)**2)
       
       
        ### GENERATE VIRTUAL POINTS FOR PDE LOSS ###
        # randomly sample from input domain for PDE loss
        x = torch.empty(self.num_samples, self.dimensions, dtype=torch.float32, device=inputs.device).uniform_(0, 1)
        x = x * (self.x_max - self.x_min) + self.x_min
        t = torch.empty(self.num_samples, 1, dtype=torch.float32, device=inputs.device).uniform_(0, 1)
        t = t * (self.t_max - self.t_min) + self.t_min
        inputs_rand = torch.cat([x, t], dim=1).requires_grad_()
        
        # predict surface fitter at sampled locations
        outputs_rand = self.surface_fitter(inputs_rand)
        
        if torch.isnan(outputs_rand).any():
            print("nans in UV MLP outputs")
            print(outputs_rand[torch.isnan(outputs_rand)])

        ### CALCULATE PDE LOSS ###
        # create arrays to store partial derivatives
        points = len(inputs_rand)
        uxx_array = torch.zeros((self.species, points, self.dimensions)).to(inputs_rand.device)
        ut_array = torch.zeros((points, self.species)).to(inputs_rand.device)

        # partial derivative computations
        for i in range(self.species):
            d1 = gradient(outputs_rand[:, i], inputs_rand, order=1)
            ut = d1[:, -1]
            ut_array[:, i] = ut

            for j in range(self.dimensions):
                d2 = gradient(d1[:, j], inputs_rand, order=1)
                uxx = d2[:, j]
                uxx_array[i, :, j] = uxx
                                        
        # reaction
        F, p_tensor = self.reaction(outputs_rand, k)
        
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
        pde_mse = torch.mean((LHS_u - RHS_u)**2 + (LHS_v - RHS_v)**2)
        pde_loss = self.pde_weight * pde_mse
    

        ### CALCULATE REGULARIZATION LOSS ###
        reg_loss = lambda_ * p_tensor.sum()

        return (gls_loss + pde_loss + reg_loss), gls_loss, pde_loss, reg_loss

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
        n_poly, n_hill = self.reaction.eql_layer.n_poly, self.reaction.eql_layer.n_hill

        # Unpack coefficients for polynomial and Hill terms
        coeffs = []
        for hypernet in self.reaction.eql_layer.hypernets:
                coeffs.append(hypernet(self.train_data.device, inference=True)[0].item())
                        
        poly_coeffs = coeffs[:n_poly]
        hill_coeffs = coeffs[n_poly:]
        
        # First half are increasing, second half are decreasing
        hill_coeffs_inc = hill_coeffs[:int(n_hill / 2)]
        hill_coeffs_dec = hill_coeffs[int(n_hill / 2):]
        
        return poly_coeffs, hill_coeffs_inc, hill_coeffs_dec

    def unpack_hill_params(self):
        ns_inc = []
        ns_dec = []
        Ks_inc = []
        Ks_dec = []

        # Access the hill feature module
        hill_features = self.reaction.eql_layer.hill

        # Iterate over all HillFeatures modules
        for hill_func in hill_features.hill_inc_raw:
            n_raw = hill_func.pn(self.train_data.device, inference=True)[0]
            K_raw = hill_func.pK(self.train_data.device, inference=True)[0]
            ns_inc.append((torch.sigmoid(n_raw) * 5).item())
            Ks_inc.append((torch.sigmoid(K_raw) * self.param_bounds).item())
            
        # Process increasing cross hill functions
        for key in list(hill_features.hill_inc_cross.keys()):
            hill_func = hill_features.hill_inc_cross[key]
            n_raw = hill_func.pn(self.train_data.device, inference=True)[0]
            K_raw = hill_func.pK(self.train_data.device, inference=True)[0]
            ns_inc.append((torch.sigmoid(n_raw) * 5).item())
            Ks_inc.append((torch.sigmoid(K_raw) * self.param_bounds).item())

        # Process decreasing raw hill functions
        for hill_func in hill_features.hill_dec_raw:
            n_raw = hill_func.pn(self.train_data.device, inference=True)[0]
            K_raw = hill_func.pK(self.train_data.device, inference=True)[0]
            ns_dec.append((torch.sigmoid(n_raw) * 5).item())
            Ks_dec.append((torch.sigmoid(K_raw) * self.param_bounds).item())
            
        # Process decreasing cross hill functions
        for key in list(hill_features.hill_dec_cross.keys()):
            hill_func = hill_features.hill_dec_cross[key]
            n_raw = hill_func.pn(self.train_data.device, inference=True)[0]
            K_raw = hill_func.pK(self.train_data.device, inference=True)[0]
            ns_dec.append((torch.sigmoid(n_raw) * 5).item())
            Ks_dec.append((torch.sigmoid(K_raw) * self.param_bounds).item())

        return ns_inc, ns_dec, Ks_inc, Ks_dec
                
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
        for i, term in enumerate(hill_terms):
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

        for term, coeff in zip(poly_terms, poly_coeffs):
            if coeff != 0:
                string = f'{coeff:.3f}'
                for ind in term:
                    string += f' * {species[ind]}'
                terms.append(string)
        
        for term, coeff, k, n in zip(hill_terms, hill_coeffs_inc, Ks_inc, ns_inc):
            if coeff != 0:
                string = f'{coeff:.3f}'
                if len(term) == 1:
                    string += f' * {species[term[0]]}^{n:.3f} / (1 + {k:.3f} * {species[term[0]]}^{n:.3f})'
                else:     
                    string += f' * {species[term[1]]} * {species[term[0]]}^{n:.3f} / (1 + {k:.3f} * {species[term[0]]}^{n:.3f})'
                terms.append(string)
            
        for term, coeff, k, n in zip(hill_terms, hill_coeffs_dec, Ks_dec, ns_dec):
            if coeff != 0:
                string = f'{coeff:.3f}'
                if len(term) == 1:
                    string += f' * (1 / {k:.3f} - {species[term[0]]}^{n:.3f} / (1 + {k:.3f} * {species[term[0]]}^{n:.3f}))'
                else:     
                    string += f' * {species[term[1]]} * (1 / {k:.3f} - {species[term[0]]}^{n:.3f} / (1 + {k:.3f} * {species[term[0]]}^{n:.3f}))'
                terms.append(string)
        
        return terms