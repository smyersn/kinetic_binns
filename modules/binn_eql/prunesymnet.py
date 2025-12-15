import torch
import torch.nn as nn
import torch.nn.functional as F
import sympy
import numpy as np
from scipy.optimize import minimize
import copy

# --- Configuration & Helpers ---
OPERATORS = {
    '+': lambda x, y: x + y,
    '-': lambda x, y: x - y,
    '*': lambda x, y: x * y,
    '/': lambda x, y: x / (y + 1e-6), # Epsilon for stability
    'sin': lambda x: torch.sin(x),
    'cos': lambda x: torch.cos(x),
    'exp': lambda x: torch.exp(torch.clamp(x, max=20)), # Clamp for stability
    'log': lambda x: torch.log(torch.abs(x) + 1e-6),
    'square': lambda x: x ** 2,
    'sqrt': lambda x: torch.sqrt(torch.abs(x) + 1e-6)
}

def get_complexity(expr_str):
    """Calculates complexity of a string expression."""
    ops = ['+', '-', '*', '/', 'sin', 'cos', 'exp', 'log', 'sqrt', '**']
    complexity = 0
    for op in ops:
        complexity += expr_str.count(op)
    return complexity

# --- The Neural Network Module ---

class SymbolicLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        return self.linear(x)

class SymbolicNetwork(nn.Module):
    def __init__(self, vari_num, const_num, hidden_layer_num, op_list, op_arity_list):
        super().__init__()
        self.op_list = op_list
        self.op_arity = op_arity_list # List of input counts per op (e.g. [2, 2, 1, 1] for +, -, sin, cos)
        self.hidden_num = hidden_layer_num
        self.check_nan = True
        
        # Dimensions
        self.vari_dim = sum(op_arity_list) # Total inputs needed for next layer's ops
        self.op_output_dim = len(op_list)  # Number of operations performed
        
        # Build Layers dynamically
        self.layers = nn.ModuleList()
        
        # Input Layer
        input_dim = vari_num + const_num
        self.layers.append(SymbolicLayer(input_dim, self.vari_dim))
        
        # Hidden Layers
        current_input_dim = input_dim + self.op_output_dim
        for _ in range(hidden_layer_num - 1):
            self.layers.append(SymbolicLayer(current_input_dim, self.vari_dim))
            current_input_dim += self.op_output_dim
            
        # Final Layer
        self.final_layer = nn.Linear(current_input_dim + self.op_output_dim, 1, bias=False)

    def _apply_operations(self, x_in, x_routing):
        """
        x_in: The accumulated features from previous layers.
        x_routing: The output of the linear layer, routing inputs to specific ops.
        """
        results = []
        pointer = 0
        
        for i, op_name in enumerate(self.op_list):
            func = OPERATORS[op_name]
            
            # Determine if unary or binary
            if self.op_arity[i] == 2:
                val = func(x_routing[:, pointer], x_routing[:, pointer + 1])
                pointer += 2
            else:
                val = func(x_routing[:, pointer])
                pointer += 1
            
            # Check for NaNs if enabled (simplified version of original logic)
            if self.check_nan and (torch.isnan(val).any() or torch.isinf(val).any()):
                val = torch.where(torch.isnan(val) | torch.isinf(val), torch.zeros_like(val), val)
                
            results.append(val.unsqueeze(1))
            
        return torch.cat(results, dim=1)

    def forward(self, x):
        # x starts as raw variables
        features = x 
        
        # Process Hidden Layers
        for layer in self.layers:
            # Linear map decides which features go to which operator
            routing = layer(features)
            
            # Apply mathematical operations
            new_features = self._apply_operations(features, routing)
            
            # Concatenate new features to history
            features = torch.cat([features, new_features], dim=1)
            
        # Final aggregation
        out = self.final_layer(features)
        return out, 0, [] # Maintaining return signature of original for compatibility

# --- Equation Decoding (SymPy conversion) ---

class EquationDecoder:
    """Helper class to convert the network weights back into a SymPy expression."""
    
    def __init__(self, network):
        self.net = network
        self.params = list(network.parameters())
        self.sym_ops = self.net.op_list
    
    def get_symbolic_expression(self, vari_num):
        """Main entry point to get the string equation."""
        # Create base variables ['x0', 'x1'...]
        vars_list = [f"x{i}" for i in range(vari_num)]
        # Add 1.0 for bias/constants
        vars_list.append("1.0") 
        
        # Start recursion from the final layer
        final_layer_idx = len(self.params) - 1
        expr = self._recursive_decode(final_layer_idx, 0, vars_list, is_final=True)
        
        # SymPy cleanup
        sym_expr = sympy.sympify(expr)
        return sympy.expand(sympy.simplify(sym_expr))

    def _recursive_decode(self, layer_idx, node_idx, input_vars, is_final=False):
        """
        Recursively traces back inputs.
        layer_idx: current weight matrix index
        node_idx: row index in the weight matrix
        """
        if layer_idx < 0:
            return input_vars[node_idx] if node_idx < len(input_vars) else "0"

        weights = self.params[layer_idx]
        
        # If this is the final aggregation layer
        if is_final:
            terms = []
            for i in range(weights.shape[1]):
                w = weights[0, i].item()
                if abs(w) > 1e-4: # Threshold for "active" connection
                    # Trace back where this input came from
                    # In this architecture, inputs to layer L are [Original_Vars + Outputs_L0 + Outputs_L1...]
                    # We need to map 'i' to specific operation output or original variable
                    term = self._resolve_input_source(layer_idx - 1, i, input_vars)
                    terms.append(f"({w} * {term})")
            return " + ".join(terms) if terms else "0"

        # Logic for Hidden Layers (routing to operators)
        # This part requires mapping specific weight rows to operator inputs.
        # Note: This is a simplified reconstruction. The original code's recursion 
        # specifically handled the unique cumulative input structure of this specific architecture.
        # For brevity, this function assumes a standard connection. 
        
        # In a real implementation of PruneSymNet, you must map the 
        # cumulative index 'i' back to (Layer L, Operator O).
        return "0" # Placeholder for complex recursive mapping

    def _resolve_input_source(self, layer_idx, input_idx, input_vars):
        """
        Determines if an input index corresponds to a raw variable or a previous layer's operator output.
        """
        # Calculate offset logic based on net structure
        # (Omitted for brevity as it depends strictly on vari_dim sizes)
        return input_vars[0] 

# --- Optimization (BFGS) ---

class ConstantOptimizer:
    """Handles the SciPy BFGS optimization to fine-tune constants."""
    
    def __init__(self):
        self.const_placeholders = [f"c{i}" for i in range(1, 21)]

    def _numpy_func(self, constants, X, expr_str):
        # Inject constants
        local_expr = expr_str
        for i, c in enumerate(constants):
            local_expr = local_expr.replace(self.const_placeholders[i], str(c))
        
        # Safe evaluation context
        context = {
            "sin": np.sin, "cos": np.cos, "exp": np.exp, 
            "log": np.abs, "sqrt": np.sqrt
        }
        
        # Inject variables x0, x1...
        for i in range(X.shape[1]):
            context[f"x{i}"] = X[:, i]
            
        try:
            return eval(local_expr, {"__builtins__": None}, context)
        except:
            return np.ones(X.shape[0]) * 1e9 # Return high error on failure

    def optimize(self, X, y, expr_str):
        # Extract constants (replace floats in string with c1, c2...)
        # This is a stub: real implementation needs regex to find floats in expr_str
        # and replace them with self.const_placeholders
        
        def loss(p):
            preds = self._numpy_func(p, X, expr_str)
            return np.mean((preds - y)**2)

        p0 = np.ones(len(self.const_placeholders)) # Initial guess
        res = minimize(loss, p0, method='BFGS', options={'gtol': 1e-6})
        return res.x, res.fun, expr_str # Simplified return

# --- Pruning ---

def prune_weights(model, threshold=0.01):
    """Simple magnitude-based pruning."""
    total = 0
    pruned = 0
    with torch.no_grad():
        for param in model.parameters():
            mask = torch.abs(param) >= threshold
            param.data *= mask.float()
            total += param.numel()
            pruned += (param.numel() - mask.sum().item())
    return total, pruned

# --- Main Training Loop Wrapper ---

def train_symbolic_net(input_data, target_data, vari_num, epochs=1000):
    # Setup
    op_list = ["+", "-", "*", "sin", "cos"]
    arity = [2, 2, 2, 1, 1]
    
    model = SymbolicNetwork(vari_num, const_num=1, hidden_layer_num=2, 
                            op_list=op_list, op_arity_list=arity)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # Train
    for epoch in range(epochs):
        optimizer.zero_grad()
        output, _, _ = model(input_data)
        loss = criterion(output, target_data)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            # Prune small weights
            prune_weights(model, threshold=0.01)
            print(f"Epoch {epoch}: Loss {loss.item()}")
            
    # Extract
    decoder = EquationDecoder(model)
    # eq = decoder.get_symbolic_expression(vari_num) # Requires full recursive logic implementation
    # print("Discovered Equation:", eq)
    
    return model

# --- Example Usage ---
if __name__ == "__main__":
    # Fake data: y = x0 + x1
    X = torch.rand(100, 2)
    y = X[:, 0:1] + X[:, 1:2]
    
    model = train_symbolic_net(X, y, vari_num=2, epochs=200)