import torch, pdb
import torch.nn as nn
import torchist

from modules.binn.build_mlp import build_mlp
from modules.utils.gradient import gradient
from modules.activations.softplus_relu import softplus_relu
from modules.binn_symbolic_net.individual import individual
from modules.symbolic_net.custom_norm import custom_norm
from modules.genetic_algorithm.genetic_algorithm_base.custom_deap_functions import (
    calculate_poly_terms, calculate_hill_terms)

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
        self.max = 5
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


class symbolic_net(nn.Module):
    
    '''
    Construct MLP surrogate model for the unknown reaction function. Includes 
    three hidden layers with 32 sigmoid-activated neurons. Output is linearly 
    activated to allow positive and negative reaction values.
    
    Inputs:
        input_features (int): number of input features
        scale        (float): input scaling factor
    
    Args:
        u, v (torch tensor): predicted u and v values with shape (N, 2)
        t (torch tensor): optional time values with shape (N, 1)
        
    Returns:
        F (torch tensor): predicted reaction values with shape (N, 1)
    '''
    
    def __init__(self, poly_terms, hill_terms, param_bounds):
        super().__init__()   
        self.poly_terms = poly_terms
        self.hill_terms = hill_terms
             
        self.num_params = len(self.poly_terms) + 3 * 2 * len(self.hill_terms) 
                        
        # K and n in Hill functions shouldn't be negative
        self.min = torch.cat((torch.full((len(self.poly_terms),), -param_bounds),
                               torch.full((2 * len(self.hill_terms),), -param_bounds),
                               torch.full((2 * len(self.hill_terms),), 0),
                               torch.full((2 * len(self.hill_terms),), 0)), 
                             dim=0)
        
        # n in Hill function shouldn't exceed 5
        self.max = torch.cat((torch.full((len(self.poly_terms),), param_bounds),
                               torch.full((2 * len(self.hill_terms),), param_bounds),
                               torch.full((2 * len(self.hill_terms),), param_bounds),
                               torch.full((2 * len(self.hill_terms),), 5)), 
                             dim=0)
    
        random_vals = torch.rand(self.num_params)

        self.params = nn.Parameter(self.min + (self.max - self.min) * random_vals)
        self.individual = individual(self.params, self.poly_terms, self.hill_terms)
        
    def forward(self, input):
        self.individual = individual(self.params, self.poly_terms, self.hill_terms)

        output = self.individual.predict_f(input)

        return output
    
class BINN(nn.Module):
    
    '''
    Constructs a biologically-informed neural network (BINN) composed of
    cell density dependent diffusion and growth MLPs with an optional time 
    delay MLP.
    
    Inputs:
        delay (bool): whether to include time delay MLP
        
    
    '''
    
    def __init__(self, dimensions, species, data=None, diff_coeffs=None,
                 degree=2, l05_reg=0.1, param_bounds=10):
        
        super().__init__()
        self.dimensions = dimensions        
        self.species = species
        self.diff_coeffs = diff_coeffs
        self.degree = degree
        self.l05_reg = l05_reg
        
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
        poly_terms = calculate_poly_terms(species, degree)
        hill_terms = calculate_hill_terms(species, degree)
        self.reaction = symbolic_net(poly_terms, hill_terms, param_bounds)
        
        # reaction extrema
        self.param_min = self.reaction.min.to('cuda')
        self.param_max = self.reaction.max.to('cuda')
        
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
        self.IC_weight = 1e1
        self.surface_weight = 1e0
        self.pde_weight = 1e0
        self.param_weight = 1e10 / param_bounds
        
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
        residual = (pred - true)**2
        residual *= pred.abs().clamp(min=1.0)**(-self.gamma)
        
        return torch.mean(residual)
    
    def pde_loss(self, inputs, outputs):
        # unpack outputs
        u = outputs.clone()
        
        # create arrays to store partial derivatives
        rows = len(inputs)
        uxx_array = torch.zeros((self.species, rows, self.dimensions)).to(self.inputs.device)
        ut_array = torch.zeros((rows, self.species)).to(self.inputs.device)

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
        
        # Reaction-diffusion equation       
        LHS_u = ut_array[:, 0][:,None]
        RHS_u = Du * torch.sum(uxx_array[0, :, :], dim=1, keepdim=True) + F
        LHS_v = ut_array[:, 1][:,None]
        RHS_v = Dv * torch.sum(uxx_array[1, :, :], dim=1, keepdim=True) - F
        pde_loss = (LHS_u - RHS_u)**2 + (LHS_v - RHS_v)**2
                
        # constraints on learned parameters
        self.param_loss = 0
        self.D_loss = 0
        self.l05_reg_loss = 0

        self.param_loss += torch.mean(self.param_weight * torch.relu(self.param_min - self.reaction.params)**2)
        self.param_loss += torch.mean(self.param_weight * torch.relu(self.reaction.params - self.param_max)**2)
        
        if not self.diff_coeffs:
            self.D_loss += torch.mean(self.D_weight * torch.relu(self.D_min - D)**2)
            self.D_loss += torch.mean(self.D_weight * torch.relu(D - self.D_max)**2)
            
        # L1 Regularization
        if self.l05_reg != 0:
            l05_norm = custom_norm(self.reaction.params, 0.01)
            self.l05_reg_loss += self.l05_reg * l05_norm

        return torch.mean(pde_loss + self.param_loss + self.D_loss + self.l05_reg_loss)
    
    def loss(self, pred, true):
        self.gls_loss_val = 0
        self.pde_loss_val = 0       
        
        # load cached inputs from forward pass
        inputs = self.inputs
     
        self.gls_loss_val = self.surface_weight*self.gls_loss(pred, true)

        # randomly sample from input domain for PDE loss
        x = torch.rand(self.num_samples, self.dimensions, requires_grad=True) 
        x = x*(self.x_max - self.x_min) + self.x_min
        t = torch.rand(self.num_samples, 1, requires_grad=True)
        t = t*(self.t_max - self.t_min) + self.t_min
        inputs_rand = torch.cat([x, t], dim=1).float().to(inputs.device)

        # predict surface fitter at sampled locations
        outputs_rand = self.surface_fitter(inputs_rand)

        # compute PDE loss at sampled locations
        self.pde_loss_val += self.pde_weight*self.pde_loss(inputs_rand, outputs_rand)
        
        return self.gls_loss_val + self.pde_loss_val, self.gls_loss_val, self.pde_loss_val