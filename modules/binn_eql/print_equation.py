def print_equation(model, protein_names=None, threshold=1e-3):
    # Assume model is an instance of ReactionFunctionEQL with attribute num_proteins
    N = model.eql_layer.poly.num_proteins
    if protein_names is None:
        protein_names = [f"u{i+1}" for i in range(N)]
    weight = model.eql_layer.fc.weight.detach().cpu().numpy().flatten()
    bias = model.eql_layer.fc.bias.item()
    
    expressions = []
    # Polynomial features
    # Linear terms:
    for i in range(N):
        expressions.append(f"{protein_names[i]}")
    # Squared terms:
    for i in range(N):
        expressions.append(f"({protein_names[i]})^2")
    # Cross terms (i<j):
    for i in range(N):
        for j in range(i+1, N):
            expressions.append(f"{protein_names[i]}*{protein_names[j]}")
            
    # Hill features (increasing)
    for i in range(N):
        expressions.append(f"hill_inc({protein_names[i]})")
    for i in range(N):
        for j in range(N):
            expressions.append(f"hill_inc({protein_names[i]})*{protein_names[j]}")
            
    # Hill features (decreasing)
    for i in range(N):
        expressions.append(f"hill_dec({protein_names[i]})")
    for i in range(N):
        for j in range(N):
            expressions.append(f"hill_dec({protein_names[i]})*{protein_names[j]}")
    
    # Combine terms with weights
    terms = []
    for w, expr in zip(weight, expressions):
        if abs(w) > threshold:
            terms.append(f"({w:.4f})*({expr})")
    eq_str = " + ".join(terms)
    print(f"F = {bias:.4f} + {eq_str}")