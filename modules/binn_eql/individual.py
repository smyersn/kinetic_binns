import sys, os
file_dir = os.path.dirname(os.path.realpath(__file__))
repo_start = f'{file_dir}/../../'
sys.path.append(repo_start)

from modules.utils.imports import *

class individual():   
    def __init__(self, params, poly_terms, hill_terms):
        self.params = params
        self.poly_terms = poly_terms
        self.hill_terms = hill_terms

        self.update_attributes(params)
            
    def update_attributes(self, params):
        # unpacks params and updates individual's attributes
        self.params = params
        
        # Define parameter indices
        self.hill_params_start = len(self.poly_terms)
        self.hill_ks_start = self.hill_params_start + 2 * len(self.hill_terms)
        self.hill_ns_start = self.hill_ks_start + 2 * len(self.hill_terms)
        
        # Unpack individual's parameters
        self.poly_params = self.params[:self.hill_params_start]

        self.hill_params = self.params[self.hill_params_start:self.hill_ks_start]
        self.hill_params_increasing = self.hill_params[:len(self.hill_params) // 2]
        self.hill_params_decreasing = self.hill_params[len(self.hill_params) // 2:]

        self.hill_ks = self.params[self.hill_ks_start:self.hill_ns_start]
        self.hill_ks_increasing = self.hill_ks[:len(self.hill_ks) // 2]
        self.hill_ks_decreasing = self.hill_ks[len(self.hill_ks) // 2:]
    
        self.hill_ns = self.params[self.hill_ns_start:]
        self.hill_ns_increasing = self.hill_ns[:len(self.hill_ns) // 2]
        self.hill_ns_decreasing = self.hill_ns[len(self.hill_ns) // 2:]
        
        # Define coefficients (multiplier for each term in library)
        self.coeffs = torch.cat((self.poly_params, self.hill_params_increasing,
                            self.hill_params_decreasing))
    
    def predict_f(self, uv, terms=False):
        # Predicts values of f based on learned function (for each term 
        # individually or as sum of all terms)
        
        # Contribution to pred from polynomial terms
        poly_pred = torch.zeros(len(uv), len(self.poly_terms)).to(uv.device)
        
        for i, term, param in zip(range(len(self.poly_terms)), 
                                  self.poly_terms, 
                                  self.poly_params):
            
            poly_pred[:, i] = param * torch.prod(uv[:, term], dim=1)
        
        # Contriubtion to pred from increasing Hill function terms
        inc_hill_pred = torch.zeros(len(uv), len(self.hill_terms)).to(uv.device)
        
        for i, term, param, k, n in zip(range(len(self.hill_terms)), 
                                        self.hill_terms, 
                                        self.hill_params_increasing, 
                                        self.hill_ks_increasing, 
                                        self.hill_ns_increasing):
            
            if len(term) == 1:
                inc_hill_pred[:, i] = param * (uv[:, term[0]]**n / (1 + k * uv[:, term[0]]**n))
            else:
                inc_hill_pred[:, i] = param * ((uv[:, term[0]] * uv[:, term[1]]**n) / (1 + k * uv[:, term[1]]**n))

        # Contriubtion to pred from decreasing Hill function terms
        dec_hill_pred = torch.zeros(len(uv), len(self.hill_terms)).to(uv.device)
        
        for i, term, param, k, n in zip(range(len(self.hill_terms)), 
                                        self.hill_terms, 
                                        self.hill_params_decreasing, 
                                        self.hill_ks_decreasing, 
                                        self.hill_ns_decreasing): 
                        
            if len(term) == 1:
                dec_hill_pred[:, i] = param * (1 - (uv[:, term[0]]**n / (1 + k * uv[:, term[0]]**n)))
            else:
                dec_hill_pred[:, i] = param * ((uv[:, term[0]] * (1 - (uv[:, term[1]]**n) / (1 + k * uv[:, term[1]]**n))))

        term_fs = torch.cat((poly_pred, inc_hill_pred, dec_hill_pred), dim=1)

        if terms == True:
            return term_fs
        
        else:
            return torch.sum(term_fs, dim=1)
        
    def fix_insignificant_terms(self, uv):
        # removes all terms from individual that have minor impact on total
        # surface shape and magnitude
        term_surfaces = self.predict_f(uv.clone().detach(), terms=True)

        for i in range(len(self.coeffs)):
            term_surface = term_surfaces[:, i]
            if (term_surface.max() - term_surface.min()) < 2:
                self.coeffs[i] = 0
                
        self.update_attributes(torch.cat((self.coeffs, self.hill_ks, self.hill_ns)))
                
        # set k and n values for insignificant Hill functions to zero
        mask = self.hill_params == 0
        self.hill_ks = self.hill_ks.masked_fill(mask, 0)
        self.hill_ns = self.hill_ns.masked_fill(mask, 0)
        
        self.update_attributes(torch.cat((self.coeffs, self.hill_ks, self.hill_ns)))
    
    def fix_cheating_hill_functions(self, uv):
        # Symbolic net sometimes "cheats," approximating polynomial terms with
        # Hill functions. This method corrects for this mistake.
        
        # check if denominator equals 1
        for i, term in enumerate(self.hill_terms):
            # only check if Hill function appears in final equation
            if self.hill_params_increasing[i] != 0:
                # define parameters
                k = self.hill_ks_increasing[i]
                n = self.hill_ns_increasing[i]
                specie = uv[:, term[-1]]
                
                # check if denominator is roughly 1 for all u and v (sign of cheating)
                denom_vals = 1 + k * specie**n
                if denom_vals.max() - 1 < 0.1 and 1 - denom_vals.max() - 1 < 0.1:
                    n = int(torch.round(n))
                    
                    # find corresponding polynomial term Hill function is approximating
                    # with cheating
                    if len(term) == 1:
                        poly_term = (term[-1],) * n
                    if len(term) > 1:
                        poly_term = (term[0],) + (term[-1],) * n
                    
                    if poly_term in self.poly_terms:    
                        poly_idx = self.poly_terms.index(poly_term)
                    else:
                        break
                    
                    # swap Hill and poly params to correct the cheating
                    self.poly_params[poly_idx] = self.hill_params_increasing[i]
                    self.hill_params_increasing[i] = 0
                    
        coeffs = torch.cat((self.poly_params, 
            self.hill_params_increasing,
            self.hill_params_decreasing))
                        
        self.update_attributes(torch.cat((coeffs, self.hill_ks, self.hill_ns)))  
        
    def abic(self, true_vals, predicted_vals):   
        
        # Calculate the residual sum of squares (RSS)   
        residuals = true_vals - predicted_vals
        
        rss = torch.sum(residuals ** 2)
        
        # Number of data points
        n = len(predicted_vals)

        # Estimate the variance of the residuals
        sigma_squared = rss / n

        # Calculate the log-likelihood
        log_likelihood = -n / 2 * (torch.log(2 * np.pi * sigma_squared) + 1)

        # Number of parameters (k)
        k = len(self.params)

        # Calculate AIC
        aic = 2 * k - 2 * log_likelihood

        # Calculate BIC
        bic = k * np.log(n) - 2 * log_likelihood
        
        return aic, bic
    
    def write_terms(self):
        
        terms = []
        species = ['u', 'v']
    
        for term, param in zip(self.poly_terms, self.poly_params):
            if param != 0:
                string = f'{param:.3f}'
                for ind in term:
                    string += f' * {species[ind]}'
                terms.append(string)
        
        for term, param, k, n in zip(self.hill_terms, self.hill_params_increasing, self.hill_ks_increasing, self.hill_ns_increasing):
            if param != 0:
                string = f'{param:.3f}'
                if len(term) == 1:
                    string += f' * {species[term[0]]}^{n:.3f} / (1 + {k:.3f} * {species[term[0]]}^{n:.3f})'
                else:     
                    string += f' * {species[term[0]]} * {species[term[1]]}^{n:.3f} / (1 + {k:.3f} * {species[term[1]]}^{n:.3f})'
                terms.append(string)
            
        for term, param, k, n in zip(self.hill_terms, self.hill_params_decreasing, self.hill_ks_decreasing, self.hill_ns_decreasing):
            if param != 0:
                string = f'{param:.3f}'
                if len(term) == 1:
                    string += f' * (1 - {species[term[0]]}^{n:.3f} / (1 + {k:.3f} * {species[term[0]]}^{n:.3f}))'
                else:     
                    string += f' * {species[term[0]]} * (1 - {species[term[1]]}^{n:.3f} / (1 + {k:.3f} * {species[term[1]]}^{n:.3f}))'
                terms.append(string)
        
        return terms
