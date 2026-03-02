import torch

def training_test_split(training_data, device):
    N = training_data.shape[0]
    
    # Create a random permutation of indices
    perm = torch.randperm(N, device=training_data.device)
    
    # Shuffle the data
    training_data_shuffled = training_data[perm]
    
    # Split into training and validation sets
    split = int(0.8 * N)
    train_data = training_data_shuffled[:split].to(device)
    val_data = training_data_shuffled[split:].to(device)

    return train_data, val_data