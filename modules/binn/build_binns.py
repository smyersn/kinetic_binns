import torch, pdb
import torch.nn as nn
import torchist

from modules.binn.build_mlp import build_mlp
from modules.utils.gradient import gradient
from modules.activations.softplus_relu import softplus_relu
from modules.utils.numpy_torch_conversion import to_torch


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
    
    
    # def __init__(self, input_features=2):
        
    #     super().__init__()
    #     self.input_features = input_features
    #     self.activation = nn.ReLU()      
    #     self.max = 10   
    #     self.diffusion_coeffs = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(self.input_features)])
        
    # def forward(self, uv):     
    #     D = torch.tensor([self.activation(D) for D in self.diffusion_coeffs])
    #     return D
    
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


class F_MLP(nn.Module):
    
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
    
    def __init__(self, input_features, layers=[32, 32, 32, 1]):
        
        super().__init__()
        self.inputs = 2
        self.min = -200
        self.max = 200
        self.mlp = build_mlp(
            input_features=input_features, 
            layers=layers,
            activation=nn.Sigmoid(), 
            linear_output=True)
    
    def forward(self, uv):       
        F = self.max * self.mlp(uv)
        return F
    
class BINN(nn.Module):
    
    '''
    Constructs a biologically-informed neural network (BINN) composed of
    cell density dependent diffusion and growth MLPs with an optional time 
    delay MLP.
    
    Inputs:
        delay (bool): whether to include time delay MLP
        
    
    '''
    
    def __init__(self, dimensions, species, data=None, uv_arch=None, 
                 f_arch=None, diff_coeffs=[0.01, 1], diff=False):
        
        super().__init__()
        
        self.dimensions = dimensions        
        self.species = species
        self.diff_coeffs = diff_coeffs
        self.diff = diff
                
        # surface fitter
        if uv_arch is None:
            self.surface_fitter = uv_MLP(input_features=dimensions+1)
        else:
            self.surface_fitter = uv_MLP(input_features=dimensions+1, layers=uv_arch)
            
        # diffusion fitter
        if diff is True:
            self.diffusion_fitter = D_PARAMS(input_features=self.species)
            
            # diffusion extrema
            self.D_min = self.diffusion_fitter.min
            self.D_max = self.diffusion_fitter.max
            
            # loss weight
            self.D_weight = 1e10 / self.D_max
        
        # pde functions
        if f_arch is None:
            self.reaction = F_MLP(input_features=species)
        else:
            self.reaction = F_MLP(input_features=species, layers=f_arch)
        
        # reaction extrema
        self.F_min = self.reaction.min
        self.F_max = self.reaction.max
        
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
        self.F_weight = 1e10 / self.F_max
        
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
    
    def gls_loss(self, pred, true, density_weight, hist, edges):
        residual = (pred - true)**2
        residual *= pred.abs().clamp(min=1.0)**(-self.gamma)
        
        if hist is not None and edges is not None:
            reciprocal_density = calc_reciprocal_density(true, hist, edges)
            density_weight = torch.tensor([[density_weight, density_weight]]).to(self.inputs.device)
            reciprocal_density = torch.column_stack((reciprocal_density, reciprocal_density)).to(self.inputs.device)
            residual *= density_weight * reciprocal_density

        return torch.mean(residual)
    
    def pde_loss(self, inputs, outputs, dimensions, species, density_weight, hist, edges):

        # unpack inputs
        x = inputs[:, :dimensions]
        t = inputs[:, dimensions][:,None]
        u = outputs.clone()
        
        # create arrays to store partial derivatives
        rows = len(inputs)
        uxx_array = torch.zeros((species, rows, dimensions)).to(self.inputs.device)
        ut_array = torch.zeros((rows, species)).to(self.inputs.device)

        # partial derivative computations
        for i in range(species):
            d1 = gradient(u[:, i], inputs, order=1)
            ut = d1[:, dimensions]
            ut_array[:, i] = ut

            for j in range(dimensions):
                d2 = gradient(d1[:, j], inputs, order=1)
                uxx = d2[:, j]
                uxx_array[i, :, j] = uxx
                                        
        # reaction
        F = self.reaction(outputs)
        
        # diffusion
        if self.diff == True:
            D = self.diffusion_fitter()
            Du, Dv = D[0], D[1]
            
        else:
            Du, Dv = torch.tensor(self.diff_coeffs[0]), torch.tensor(self.diff_coeffs[1])
        
        # Reaction-diffusion equation       
        LHS_u = ut_array[:, 0][:,None]
        RHS_u = Du * torch.sum(uxx_array[0, :, :], dim=1, keepdim=True) + F
        LHS_v = ut_array[:, 1][:,None]
        RHS_v = Dv * torch.sum(uxx_array[1, :, :], dim=1, keepdim=True) - F
        pde_loss = (LHS_u - RHS_u)**2 + (LHS_v - RHS_v)**2
                
        # # Calculate Euler loss (prevents negative concentrations) 
        # timestep = 0.0001
        
        # u_updated = u[:, 0] + RHS_u.squeeze() * timestep
        # v_updated = u[:, 1] + RHS_v.squeeze() * timestep
        
        self.Euler_loss = 0
        
        # self.Euler_loss += torch.where(u_updated < 0, 10000 * (u_updated)**2, 
        #                                torch.zeros_like(u_updated)) + torch.where(
        #                                    v_updated < 0, 10000 * (v_updated)**2, 
        #                                    torch.zeros_like(v_updated))
                        
        # if hist is not None and edges is not None:
        #     # add weight based on density
        #     reciprocal_density = calc_reciprocal_density(outputs, hist, edges).to(self.inputs.device)
        #     pde_loss *= density_weight * reciprocal_density.unsqueeze(1)

        # constraints on learned parameters
        self.F_loss = 0
        self.D_loss = 0
        
        self.F_loss += self.F_weight*torch.where(
            F < self.F_min, (F-self.F_min)**2, torch.zeros_like(F))
        self.F_loss += self.F_weight*torch.where(
            F > self.F_max, (F-self.F_max)**2, torch.zeros_like(F))
        
        if self.diff == True:
            self.D_loss += self.D_weight*torch.sum(torch.where(
                D < self.D_min, (D-self.D_min)**2, torch.zeros_like(D)))
            self.D_loss += self.D_weight*torch.sum(torch.where(
                D > self.D_max, (D-self.D_max)**2, torch.zeros_like(D)))
                        
        # no derivative constraints included (Du and Dv constant, no obvious relationship for F)

        return torch.mean(pde_loss + self.Euler_loss + self.F_loss + self.D_loss)
    
    def loss(self, pred, true, density_weight=0, hist=None, edges=None):
        self.gls_loss_val = 0
        self.pde_loss_val = 0       
        
        # load cached inputs from forward pass
        inputs = self.inputs
     
        self.gls_loss_val = self.surface_weight*self.gls_loss(pred, true, density_weight, hist, edges)

        # randomly sample from input domain for PDE loss
        x = torch.rand(self.num_samples, self.dimensions, requires_grad=True) 
        x = x*(self.x_max - self.x_min) + self.x_min
        t = torch.rand(self.num_samples, 1, requires_grad=True)
        t = t*(self.t_max - self.t_min) + self.t_min
        inputs_rand = torch.cat([x, t], dim=1).float().to(inputs.device)

        # predict surface fitter at sampled locations
        outputs_rand = self.surface_fitter(inputs_rand)

        # compute PDE loss at sampled locations
        self.pde_loss_val += self.pde_weight*self.pde_loss(inputs_rand, outputs_rand, self.dimensions, self.species, density_weight, hist, edges)
        
        return self.gls_loss_val + self.pde_loss_val, self.gls_loss_val, self.pde_loss_val