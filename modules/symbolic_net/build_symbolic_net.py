import torch, pdb
import torch.nn as nn
import torchist

from modules.binn.build_mlp import build_mlp
from modules.utils.gradient import gradient
from modules.utils.histogram import calc_density
from modules.activations.softplus_relu import softplus_relu
from modules.utils.numpy_torch_conversion import to_torch
from modules.symbolic_net.individual import individual
from modules.symbolic_net.custom_norm import custom_norm
from modules.genetic_algorithm.genetic_algorithm_base.custom_deap_functions import (
    calculate_poly_terms, calculate_hill_terms)


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
    
    def __init__(self, species, degree, param_bounds, device, l1_reg=0, 
                 nonzero_term_reg=0, density_weight=0, hist=None, edges=None):
        super().__init__()
        self.F_min = -100
        self.F_max = 100
        self.F_weight = 1e10 / self.F_max
        
        self.param_weight = 1e10 / param_bounds
        
        self.species = species
        self.degree = degree
        self.l1_reg = l1_reg
        self.nonzero_term_reg = nonzero_term_reg
        
        self.poly_terms = calculate_poly_terms(species, degree)
        self.hill_terms = calculate_hill_terms(species, degree)
        self.num_params = len(self.poly_terms) + 3 * 2 * len(self.hill_terms) 
        
        self.hist = hist
        self.edges = edges
        self.density_weight = density_weight
        
        self.device = device
        
        # K and n in Hill functions shouldn't be negative
        self.param_min = torch.cat((torch.full((len(self.poly_terms),), -param_bounds),
                               torch.full((2 * len(self.hill_terms),), -param_bounds),
                               torch.full((2 * len(self.hill_terms),), 0),
                               torch.full((2 * len(self.hill_terms),), 0)), 
                             dim=0).to(self.device)
        
        # n in Hill function shouldn't exceed 5
        self.param_max = torch.cat((torch.full((len(self.poly_terms),), param_bounds),
                               torch.full((2 * len(self.hill_terms),), param_bounds),
                               torch.full((2 * len(self.hill_terms),), param_bounds),
                               torch.full((2 * len(self.hill_terms),), 5)), 
                             dim=0).to(self.device)
    
        random_vals = torch.rand(self.num_params).to(self.device)

        self.params = nn.Parameter(self.param_min + (self.param_max - self.param_min) * random_vals).to(self.device)
        self.individual = individual(self.params, self.species, self.degree)
        
    def forward(self, input):
        self.individual = individual(self.params, self.species, self.degree)

        output = self.individual.predict_f(input)

        return output
    
    def loss(self, x, pred, true):
        self.MSE_loss = 0
        self.F_loss = 0
        self.param_loss = 0
        self.l1reg_loss = 0
        
        # MSE loss
        # self.MSE_loss += nn.functional.mse_loss(pred, true.view(-1))
        self.squared_error = (pred - true.view(-1))**2
        
        # F loss and param loss
        self.F_loss += self.F_weight * torch.relu(self.F_min - pred)**2
        self.F_loss += self.F_weight * torch.relu(pred - self.F_max)**2
        
        self.param_loss += self.param_weight * torch.relu(self.param_min - self.params)**2
        self.param_loss += self.param_weight * torch.relu(self.params - self.param_max)**2
        
        # L1 Regularization
        if self.l1_reg != 0:
            l1_norm = custom_norm(self.params, 0.01)
            self.l1reg_loss += self.l1_reg * l1_norm
                                
        if self.density_weight != 0:
            density = calc_density(x, self.hist, self.edges)
            self.squared_error *= (density.to(self.device) * self.density_weight)
            
        total_loss = (torch.mean(self.squared_error) + self.l1reg_loss + torch.mean(self.F_loss) + torch.mean(self.param_loss))

        return total_loss